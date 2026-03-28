"""Write a t raining script from scratch to easily tune things to have a story for the paper and then bring it all back."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"

import json
import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from tqdm import tqdm

from biolearn import BioGNN, BioGnnModel, FastProduce
from biolearn.models.hill import EdgeType
from biolearn.models.nfc import NFC, MoormanNFC
from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.quadrotor import HeightMaintain
from biolearn.specifications.ss_classification import PhiXorFast

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from biolearn.models.base import BioModel, SimulateConfig
from biolearn.utils import sample_hypercube_faces


class SamplingBuffer(eqx.Module):
    buffer: jax.Array
    index: jax.Array
    size: jax.Array


def make_buffer(dimension: int, capacity: int) -> SamplingBuffer:
    return SamplingBuffer(jnp.zeros((capacity, dimension)), jnp.array(0), jnp.array(0))


def buffer_insert(
    buffer: SamplingBuffer, x: jax.Array, mask: jax.Array
) -> SamplingBuffer:
    """Insert into buffer where mask is true to avoid resizing."""
    len = x.shape[0]
    capacity = buffer.buffer.shape[0]
    order = jnp.argsort(~mask)

    sorted_x = x[order]
    n_new = jnp.sum(mask).astype(jnp.int32)

    positions = (buffer.index + jnp.arange(len)) % capacity
    write_mask = jnp.arange(len) < n_new

    new_buffer = buffer.buffer.at[positions].set(
        jnp.where(write_mask[:, None], sorted_x, buffer.buffer[positions])
    )

    new_index = (buffer.index + n_new) % capacity
    new_size = jnp.minimum(buffer.size + n_new, capacity)

    return SamplingBuffer(new_buffer, new_index, new_size)


def buffer_sample(buffer: SamplingBuffer, key: jax.Array, n_samples: int) -> jax.Array:
    safe_size = jnp.maximum(buffer.size, 1)
    indices = jr.randint(key, (n_samples,), 0, buffer.buffer.shape[0]) % safe_size
    return buffer.buffer[indices]


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, state_size: int, hidden_size: int = 32):
        k1, k2, k3 = jr.split(key, 3)
        out_layer = eqx.nn.Linear(hidden_size, state_size, key=k3)
        self.layers = [
            eqx.nn.Linear(state_size, hidden_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            jax.nn.relu,
            out_layer,
        ]

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class BufferModel(eqx.Module):
    """A module wrapper that adds a residual NN to the ODE."""

    nominal_model: BioModel
    residual_model: MLP
    slack: jax.Array

    def __init__(self, key: jax.Array, nominal: BioModel):
        self.nominal_model = nominal
        state_size = math.prod(nominal.shape)
        residual_model = MLP(key, state_size=state_size, hidden_size=64)
        self.residual_model = jax.tree_util.tree_map(
            lambda x: x * 1e-4 if eqx.is_inexact_array(x) else x,
            residual_model,
        )
        self.slack = jnp.array(0.01)

    def _step(self, t, y, args):
        mlp_out = self.residual_model(y.flatten()).reshape(y.shape)
        return self.nominal_model.diffrax_step(t, y, args) + mlp_out * y

    def simulate(
        self,
        x: jax.Array,
        ts: jax.Array,
        x_ts=None,
        config: SimulateConfig = SimulateConfig(),
    ):
        if isinstance(self.nominal_model, NFC):
            interp = NFC._handle_inputs(x, x_ts)
            args = (interp,)
            y0 = jnp.zeros(self.nominal_model.shape)
        else:
            args = None
            y0 = x

        solver = diffrax.Kvaerno5() if config.stiff else diffrax.Tsit5()

        stepsize_controller = diffrax.PIDController(
            pcoeff=0.3 if config.stiff else 0.0,
            icoeff=0.3 if config.stiff else 1.0,
            dcoeff=0,
            rtol=config.rtol,
            atol=config.atol,
            dtmax=config.max_stepsize,
            dtmin=1e-10,
        )
        term = diffrax.ODETerm(self._step)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=config.max_steps,
            throw=config.throw,
            args=args,
        )
        return solution.ys, solution


def residual_l2(model: BufferModel) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(model.residual_model, eqx.is_inexact_array)
    )
    return sum((jnp.sum(w**2) for w in leaves), jnp.zeros(()))


def quadrotor_sampler(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    n_samples: int,
    n_per_face: int = 1,
) -> jax.Array:
    nonzero = [0, 1, 2, 3, 4, 5, 7, 9, 11]
    l = low[jnp.array(nonzero)]
    h = high[jnp.array(nonzero)]
    key, subkey = jr.split(key)
    edge_samples = sample_hypercube_faces(subkey, l, h, n_per_face=n_per_face)

    xs = jnp.concatenate(
        [
            edge_samples,
            jr.uniform(key, (n_samples, len(nonzero)), minval=l, maxval=h),
        ]
    )

    result = jnp.zeros((xs.shape[0], 12))
    result = result.at[:, nonzero].set(xs)

    return result


def xor_sampler(
    key: jax.Array, low: jax.Array, high: jax.Array, n_samples: int, n_face: int = 2
) -> jax.Array:
    key, subkey = jr.split(key)
    edge_points = jr.uniform(subkey, (n_face * 4,))
    step = n_face

    x0 = jnp.stack([jnp.zeros(step) + 0.1, edge_points[:step]], axis=-1)
    x1 = jnp.stack([jnp.ones(step) * 0.9, edge_points[step : 2 * step]], axis=-1)
    x2 = jnp.stack([edge_points[2 * step : 3 * step], 0.1 + jnp.zeros(step)], axis=-1)
    x3 = jnp.stack([edge_points[3 * step :], 0.9 * jnp.ones(step)], axis=-1)

    xs = jnp.concatenate([x0, x1, x2, x3, jr.uniform(key, (n_samples, 2))], axis=0)
    return xs


def hill_sampler(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    n_samples: int,
    n_per_face: int = 1,
) -> jax.Array:
    key, subkey = jr.split(key)
    edge_samples = sample_hypercube_faces(
        subkey,
        low,
        high,
        n_per_face=n_per_face,
    )

    inside_samples = jr.uniform(key, (n_samples, 6)) * (high - low) + (high + low) / 2.0
    xs = jnp.concatenate([edge_samples, inside_samples])

    return xs


def run_one(key: jax.Array, args: Args):
    key, subkey = jr.split(key)
    t_start = time.monotonic()

    kd = jr.key_data(key)
    run_id = f"{args.system}_{kd[0]:x}{kd[1]:x}"
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    model_keys = jr.split(subkey, args.num_initializations)

    match args.system:
        case "quadrotor":
            models = [QuadModel(Quadrotor(k)) for k in model_keys]

            spec = HeightMaintain()
            sim_cfg = SimulateConfig(
                to_ss=False,
                stiff=False,
                throw=True,
                max_steps=int(3e4),
                rtol=1e-3,
                atol=1e-4,
                max_stepsize=0.5,
                progress_bar=False,
            )

            importance_buffers = jax.tree.map(
                lambda *bs: jnp.stack(bs),
                *[make_buffer(12, 1024) for _ in range(args.num_instantiations)],
            )

            ts = jnp.arange(0.0, 5.0, 1.0)

            def ss_to_traj_q(y_trace, _):
                y_traj = y_trace[..., 4:6]  # (B, T, 1)
                return y_traj

            ss_to_traj = ss_to_traj_q
            xs = jnp.linspace(-0.4, 0.4, 4)
            ys = jnp.linspace(-0.02, 0.02, 4)
            xs = jnp.meshgrid(
                *[
                    xs,
                    xs,
                    xs,
                    xs,
                    xs,
                    xs,
                    jnp.zeros(1),
                    ys,
                    jnp.zeros(1),
                    ys,
                    jnp.zeros(1),
                    ys,
                ]
            )
            low = jnp.array(
                [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, -0.02, 0.0, -0.02, 0.0, -0.02]
            )
            high = jnp.array(
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02]
            )

            x_test = jnp.stack([x.flatten() for x in xs], axis=-1)

            t_horizons = [5.0]
            sampler = quadrotor_sampler
        case "nfc":
            models = [
                MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=k) for k in model_keys
            ]

            spec = PhiXorFast()

            sim_cfg = SimulateConfig(
                to_ss=False,
                stiff=True,
                throw=True,
                max_steps=int(3e4),
                rtol=1e-4,
                atol=1e-5,
                max_stepsize=0.5,
                progress_bar=False,
            )

            importance_buffers = jax.tree.map(
                lambda *bs: jnp.stack(bs),
                *[make_buffer(2, 1024) for _ in range(args.num_instantiations)],
            )

            ts = jnp.arange(0.0, 20.0, 1.0)

            def ss_to_traj_xor(y_trace, x):
                y_traj = y_trace[..., -1, 0][..., None]  # (B, T, 1)
                x_traj = jnp.ones_like(y_traj) * x[..., None, :]  # (B, T, 2)

                return jnp.concatenate([x_traj, y_traj], axis=-1)

            low = jnp.array([0.1, 0.1])
            high = jnp.array([0.9, 0.9])
            ss_to_traj = ss_to_traj_xor

            xs = jnp.linspace(0.1, 0.9, 32)
            xs = jnp.meshgrid(*[xs, xs])
            x_test = jnp.stack([x.flatten() for x in xs], axis=-1)

            t_horizons = [20.0]
            sampler = xor_sampler
        case "hill":
            graph = [
                (0, 1, EdgeType.Activation),  # x1 -> x2
                (2, 3, EdgeType.Inhibition),  # x3 -| x4
                (3, 0, EdgeType.Inhibition),  # x4 -| x1
                (3, 1, EdgeType.Activation),  # x4 -> x2
                (4, 1, EdgeType.Activation),  # x5 -> x2
                (4, 2, EdgeType.Activation),  # x5 -> x3
                (4, 5, EdgeType.Activation),  # x5 -> x6
            ]

            models = [BioGnnModel(BioGNN(k, graph, 2.0)) for k in model_keys]

            spec = FastProduce()
            sim_cfg = SimulateConfig(
                to_ss=False,
                stiff=True,
                throw=True,
                max_steps=int(2e4),
                rtol=1e-3,
                atol=1e-4,
                max_stepsize=0.5,
                progress_bar=False,
            )

            importance_buffers = jax.tree.map(
                lambda *bs: jnp.stack(bs),
                *[make_buffer(6, 1024) for _ in range(args.num_instantiations)],
            )

            t_final = 15.0
            ts = jnp.arange(0.0, t_final, 1.0)

            def ss_to_traj_hill(y_trace, x):
                return y_trace

            ss_to_traj = ss_to_traj_hill

            low = jnp.array([0.0, 0.0, 0.01, 0.01, 0.99, 0.99])
            high = jnp.array([0.2, 0.2, 0.04, 0.04, 1.0, 1.0])

            xs = jnp.linspace(low, high, 10)
            xs = jnp.meshgrid(*xs.T)
            x_test = jnp.stack([x.flatten() for x in xs], axis=-1)

            t_horizons = [t_final]
            sampler = hill_sampler

    match args.loss:
        case "relu":
            group_loss = lambda rhos, _: jax.nn.relu(1e-6 - rhos).mean()
        case "softrelu":
            group_loss = lambda rhos, _: (
                jax.nn.relu(-rhos) + 1e-2 * jax.nn.sigmoid(jax.nn.relu(rhos))
            ).mean()
        case "slackrelu":
            group_loss = lambda rhos, epsilon: (
                jax.nn.relu(epsilon - rhos) - 0.05 * epsilon
            ).mean()
        case "leakyrelu":
            group_loss = lambda rhos, _: jax.nn.leaky_relu(-rhos).mean()
        case "exponential":
            group_loss = lambda rhos, _: jnp.exp(-rhos).mean()

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(model, x0, ts, reg=args.regularizer):
        sim = partial(model.simulate, ts=ts, config=sim_cfg)
        y_traces, _ = jax.vmap(sim)(x0)
        safe_traj = ss_to_traj(y_traces, x0)
        rhos = jax.vmap(spec.evaluate)(safe_traj)
        task_loss = group_loss(rhos, model.slack)
        return task_loss + reg * residual_l2(model), rhos

    def nominal_loss(model, x0, ts, reg=args.regularizer):
        sim = partial(model.simulate, ts=ts, config=sim_cfg)
        y_traces, _ = jax.vmap(sim)(x0)
        safe_traj = ss_to_traj(y_traces, x0)
        rhos = jax.vmap(spec.evaluate)(safe_traj)
        task_loss = group_loss(rhos, jnp.array(0.0))
        return task_loss

    # Test the models to find the top-k parameter vectors to optimize.
    losses = jnp.zeros(len(models))
    model_pbar = tqdm(enumerate(models), total=len(models), desc="Selecting models")
    for i, model in model_pbar:
        key, subkey = jr.split(key)
        xs = sampler(key, low, high, args.n_samples, args.boundary_samples)

        loss = nominal_loss(model, xs, ts)
        losses = losses.at[i].set(loss)
        model_pbar.set_postfix(loss=f"{float(loss):.2e}")

    # Get the top k
    np.savez(
        run_dir / "init_selection.npz",
        losses=np.asarray(losses),
    )
    _, inds = jax.lax.top_k(-losses, args.num_instantiations)

    key, subkey = jr.split(key)
    model_keys = jr.split(subkey, args.num_instantiations)

    bms = [BufferModel(model_keys[i], models[i]) for i in inds]
    all_params = [eqx.partition(bm, eqx.is_array)[0] for bm in bms]
    _, static = eqx.partition(bms[0], eqx.is_array)
    stacked_params = jax.tree.map(lambda *s: jnp.stack(s), *all_params)
    models = eqx.combine(stacked_params, static)

    fast_schedule = optax.exponential_decay(
        args.lr, args.decay_interval, 0.6, staircase=True
    )
    slow_schedule = optax.exponential_decay(
        args.lr * 0.05, args.decay_interval, 0.6, staircase=True
    )

    params, static = eqx.partition(models, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: "standard", params)
    labels = eqx.tree_at(lambda p: p.slack, labels, "slow")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "standard": optax.adabelief(fast_schedule),
                "slow": optax.adabelief(slow_schedule),
            },
            labels,
        ),
    )

    assert args.num_epochs % len(t_horizons) == 0, (
        f"Number of epochs ({args.num_epochs}) can't be split into len(t_horizons)={len(t_horizons)} chunks evenly."
    )

    opt_states = jax.tree.map(
        lambda *ps: jnp.stack(ps), *[optimizer.init(p) for p in all_params]
    )

    all_losses = []
    n_points = args.n_samples
    n_boundary = args.boundary_samples
    n_importance = args.num_importance_samples
    n_inside = max(n_points - n_boundary, 16)
    reg_gamma = (args.regularizer / args.final_reg) ** (1 / args.num_epochs)

    ms, _ = eqx.partition(models, eqx.is_array)

    for t_final in tqdm(t_horizons, desc="Horizons"):
        ts = jnp.arange(0.0, t_final, 1.0)
        n_steps = args.num_epochs // len(t_horizons)

        def _scan_step(carry, step_idx):
            key, params, opt_state, importance_buffer = carry
            key, subkey = jr.split(key)
            xs = sampler(subkey, low, high, n_inside, n_boundary)

            if args.importance_sample:
                key, ik = jr.split(key)
                importance_samples = buffer_sample(importance_buffer, ik, n_importance)
                importance_samples = jnp.where(
                    importance_buffer.size > n_importance,
                    importance_samples,
                    xs[:n_importance],
                )
                xs = jnp.concatenate([xs, importance_samples], axis=0)

            model = eqx.combine(params, static)
            (loss, rhos), grads = grad_loss(
                model,
                xs,
                ts,
                reg=args.final_reg * (reg_gamma ** (args.num_epochs - step_idx)),
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            if args.importance_sample:
                failed = rhos < 0
                importance_buffer = buffer_insert(
                    importance_buffer, jax.lax.stop_gradient(xs), failed
                )

            params, _ = eqx.partition(model, eqx.is_array)
            return (key, params, opt_state, importance_buffer), loss

        @eqx.filter_vmap(in_axes=(0, 0, 0, 0))
        def _train_horizon(key, params, opt_state, importance_buffer):
            init = (key, params, opt_state, importance_buffer)
            (_, final_params, final_opt_state, final_buf), losses = jax.lax.scan(
                _scan_step, init, jnp.arange(n_steps)
            )
            return final_params, final_opt_state, final_buf, losses

        key, subkey = jr.split(key)
        keys = jr.split(subkey, args.num_instantiations)
        ms, opt_states, importance_buffers, horizon_losses = _train_horizon(
            keys, ms, opt_states, importance_buffers
        )
        all_losses.append(horizon_losses)

    all_losses = jnp.concatenate(all_losses, axis=1)

    t_train = time.monotonic() - t_start

    # Save training curves
    np.savez(run_dir / "training_curves.npz", losses=all_losses)

    fig, ax = plt.subplots()
    ax.plot(np.arange(all_losses.shape[1]), all_losses.T)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(run_dir / "losses.png")
    plt.close(fig)

    # Test all models, pick the best one
    models = eqx.combine(ms, static)  # pyright: ignore

    all_rhos = []
    for i in range(args.num_instantiations):
        model_params_i = jax.tree.map(lambda x: x[i], ms)  # pyright: ignore
        model_i = eqx.combine(model_params_i, static)
        sim = partial(model_i.nominal_model.simulate, ts=ts, config=sim_cfg)
        y_traces, _ = jax.vmap(sim)(x_test)
        traj = ss_to_traj(y_traces, x_test)
        rhos_i = jax.vmap(spec.evaluate)(traj)
        all_rhos.append(rhos_i)

        weight_filter = list(
            filter(
                eqx.is_array,
                jax.tree_util.tree_leaves(model_i.nominal_model),
            )
        )

        print(f"Raw model parameters for model {i}: {[w for w in weight_filter]}")

    rhos = jnp.stack(all_rhos)

    sats = jnp.sum(rhos > 0, axis=1).astype(int)
    max_sat = jnp.max(sats)
    ties = sats == max_sat

    # of the ties, pick the one with the least
    min_robs = jnp.min(rhos, axis=1)
    min_robs = jnp.where(ties, min_robs, -jnp.inf)  # only consider contenders

    best_idx = int(jnp.argmax(min_robs))

    print(
        f"Model {best_idx} satisfied {max_sat}/{x_test.shape[0]} of initial conditions with least robustness {min_robs[best_idx]:.2e}."
    )

    best_model = jax.tree.map(lambda x: x[best_idx], ms)  # pyright: ignore
    best_model = eqx.combine(best_model, static)

    weight_filter = list(
        filter(
            eqx.is_inexact_array, jax.tree_util.tree_leaves(best_model.nominal_model)
        )
    )

    print(f"Raw model parameters: {[w for w in weight_filter]}")

    # Save holdout evaluation
    np.savez(
        run_dir / "holdout.npz",
        rhos=np.asarray(rhos),
        sats=np.asarray(sats),
        min_robs=np.asarray(min_robs),
        x_test=np.asarray(x_test),
    )

    # Save best model
    eqx.tree_serialise_leaves(run_dir / "best_model.eqx", best_model)

    # Save all trained models
    all_models = eqx.combine(ms, static)
    eqx.tree_serialise_leaves(run_dir / "all_models.eqx", all_models)

    # Save summary
    summary = {
        "best_model_idx": best_idx,
        "best_sat": int(max_sat),
        "total_test_points": int(x_test.shape[0]),
        "best_min_robustness": float(min_robs[best_idx]),
        "final_loss_per_model": all_losses[:, -1].tolist(),
        "train_time_s": t_train,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {run_dir}/ (train time: {t_train:.1f}s)")


@dataclass
class Args:
    num_epochs: int = 1000
    """Number of batches to sample, essentially"""
    lr: float = 1e-2
    """The learning rate"""
    decay_interval: int = 500
    """Learning rate decay interval."""
    runs: int = 3
    """Number of seeds to try out."""
    system: Literal["nfc", "quadrotor", "hill"] = "quadrotor"
    """The system to test."""
    regularizer: float = 1e-5
    """The starting regularization on the l2 norm of the MLP. Interpolated exponentially."""
    final_reg: float = 1e-1
    """The final regularization on the l2 norm of the MLP."""
    loss: Literal["relu", "softrelu", "leakyrelu", "slackrelu", "exponential"] = "relu"
    """The loss function to use to weight different initial conditions."""
    importance_sample: bool = False
    """Whether or not to do pseudo-importance sampling."""
    n_samples: int = 256
    """The batch size, or number of samples to take for the integral estimation."""
    boundary_samples: int = 2
    """The number of samples to take on each domain boundary."""
    num_instantiations: int = 1
    """The number of parameter initializations to optimize in parallel."""
    num_initializations: int = 10
    """The number of parameters to test before starting to optimize."""
    num_importance_samples: int = 32


def main():
    key = jr.key(42)
    args = tyro.cli(Args)
    for k in jr.split(key, args.runs):
        run_one(k, args)


if __name__ == "__main__":
    main()
