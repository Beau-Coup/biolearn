"""Write a t raining script from scratch to easily tune things to have a story for the paper and then bring it all back."""

import math
from dataclasses import dataclass
from functools import partial
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


class SamplingBuffer:
    buffer: jax.Array

    def __init__(self, dimension: int, capacity: int = 512):
        self.buffer = jnp.empty((capacity, dimension))
        self.size = 0
        self.capacity = capacity
        self.add_to = 0

    def insert(self, x: jax.Array):
        """Add samples
        x: (B, n)
        """
        if x.ndim == 1:
            x = x.reshape((1, -1))

        len = x.shape[0]
        if self.add_to + len > self.capacity:
            limit = self.capacity - self.add_to

            self.buffer = self.buffer.at[self.add_to : self.capacity].set(x[:limit])

            len = len - limit
            self.add_to = 0
            x = x[limit:]

        self.buffer = self.buffer.at[self.add_to : self.add_to + len].set(x)
        self.add_to = (self.add_to + len) % self.capacity
        self.size = max(self.add_to, self.capacity)

    def sample(self, key: jax.Array, n_samples: int) -> jax.Array:
        indices = jr.choice(key, self.size, (n_samples,))
        return self.buffer[indices]


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
    n_samples: int,
    n_per_face: int = 1,
) -> jax.Array:
    key, subkey = jr.split(key)
    edge_samples = sample_hypercube_faces(
        subkey, jnp.ones(6) * -0.4, 0.4 * jnp.ones(6), n_per_face=n_per_face
    )

    xs = jnp.concatenate(
        [
            edge_samples,
            jr.uniform(key, (n_samples, 6), minval=-0.4, maxval=0.4),
        ]
    )
    xs = jnp.concatenate([xs, jnp.zeros_like(xs)], axis=1)

    return xs


def xor_sampler(key: jax.Array, n_samples: int, n_face: int = 2) -> jax.Array:
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
    n_samples: int,
    n_per_face: int = 1,
) -> jax.Array:
    low = jnp.array([0.01, 0.01, 0.01, 0.01, 0.99, 0.99])
    high = jnp.array([0.04, 0.04, 0.04, 0.04, 1.0, 1.0])

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

    match args.system:
        case "quadrotor":
            quadrotor = Quadrotor(subkey)
            nominal_model = QuadModel(quadrotor)
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

            importance_buffer = SamplingBuffer(12, capacity=1024)

            ts = jnp.arange(0.0, 5.0, 1.0)

            def ss_to_traj_q(y_trace, _):
                y_traj = y_trace[..., 4:6]  # (B, T, 1)
                return y_traj

            ss_to_traj = ss_to_traj_q
            xs = jnp.linspace(-0.4, 0.4, 6)
            ys = jnp.linspace(-0.02, 0.02, 6)
            xs = jnp.meshgrid(
                *[
                    xs,
                    xs,
                    xs,
                    xs,
                    xs,
                    xs,
                    jnp.zeros(0),
                    ys,
                    jnp.zeros(0),
                    ys,
                    jnp.zeros(0),
                    ys,
                ]
            )
            x_test = jnp.stack([x.flatten() for x in xs], axis=-1)

            t_horizons = [5.0]
            sampler = quadrotor_sampler
        case "nfc":
            nominal_model = MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=subkey)
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

            importance_buffer = SamplingBuffer(2, capacity=1024)

            ts = jnp.arange(0.0, 20.0, 1.0)

            def ss_to_traj_xor(y_trace, x):
                y_traj = y_trace[..., -1, 0][..., None]  # (B, T, 1)
                x_traj = jnp.ones_like(y_traj) * x[..., None, :]  # (B, T, 2)

                return jnp.concatenate([x_traj, y_traj], axis=-1)

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
            key, subkey = jr.split(key, 2)
            nominal_model = BioGNN(subkey, graph, 2.0)
            spec = FastProduce()
            nominal_model = BioGnnModel(nominal_model)

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

            importance_buffer = SamplingBuffer(6, capacity=1024)

            ts = jnp.arange(0.0, 15.0, 1.0)

            def ss_to_traj_hill(y_trace, x):
                return y_trace

            ss_to_traj = ss_to_traj_hill

            xs = jnp.linspace(0.01, 0.04, 10)
            ys = jnp.linspace(0.99, 1.0, 10)
            xs = jnp.meshgrid(*[xs, xs, xs, xs, ys, ys])
            x_test = jnp.stack([x.flatten() for x in xs], axis=-1)

            t_horizons = [15.0]
            sampler = hill_sampler

    key, subkey = jr.split(key)
    model = BufferModel(subkey, nominal_model)

    fast_schedule = optax.exponential_decay(
        args.lr, args.decay_interval, 0.6, staircase=True
    )
    slow_schedule = optax.exponential_decay(
        args.lr * 0.05, args.decay_interval, 0.6, staircase=True
    )

    params, static = eqx.partition(model, eqx.is_array)
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

    assert args.num_epochs % len(t_horizons) == 0, (
        f"Number of epochs ({args.num_epochs}) can't be split into len(t_horizons)={len(t_horizons)} chunks evenly."
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    all_losses = []
    n_points = args.n_samples
    n_boundary = args.boundary_samples
    n_inside = max(n_points - n_boundary, 16)
    reg_gamma = (args.regularizer / args.final_reg) ** (1 / args.num_epochs)

    for t_final in tqdm(t_horizons, desc="Horizons"):
        ts = jnp.arange(0.0, t_final, 1.0)

        pbar = tqdm(range(args.num_epochs // len(t_horizons)), desc=f"t={t_final}")
        for i in pbar:
            # Sample points
            key, subkey = jr.split(key)
            xs = sampler(subkey, n_inside, n_boundary)

            # Append importance samples
            if args.importance_sample and importance_buffer.size > 32:
                key, subkey = jr.split(key)
                samples = importance_buffer.sample(subkey, 32)
                xs = jnp.concatenate([xs, samples], axis=0)

            (loss, rhos), grads = grad_loss(
                model, xs, ts, reg=args.final_reg * (reg_gamma ** (args.num_epochs - i))
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            tqdm.set_description(pbar, f"{loss:.2e}")
            model = eqx.apply_updates(model, updates)

            if args.importance_sample:
                failed = jnp.nonzero(rhos < 0)
                if failed[0].shape[0] > 0:
                    importance_buffer.insert(jax.lax.stop_gradient(xs[failed]))

            all_losses.append(float(loss))

    plt.plot(np.arange(len(all_losses)), all_losses)
    plt.yscale("log")
    plt.show()
    kd = jr.key_data(key)
    plt.savefig(f"figures/losses-{args.system}-{kd[0]}{kd[1]}.png")

    sim = partial(model.nominal_model.simulate, ts=ts, config=sim_cfg)
    y_traces, _ = jax.vmap(sim)(x_test)
    traj = ss_to_traj(y_traces, x_test)
    rhos = jax.vmap(spec.evaluate)(traj)

    sats = jnp.sum(rhos > 0).astype(int)
    print(rhos.shape, sats.shape)

    print(f"Satisfied {sats}/{x_test.shape[0]} of initial conditions")

    weight_filter = list(
        filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(model.nominal_model))
    )

    print(f"Raw model parameters: {[w for w in weight_filter]}")


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


def main():
    key = jr.key(42)
    args = tyro.cli(Args)
    for k in jr.split(key, args.runs):
        run_one(k, args)


if __name__ == "__main__":
    main()
