"""Write a training script from scratch to easily tune things to have a story for the paper and then bring it all back."""

import math
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from biolearn.models import MoormanNFC
from biolearn.models.base import BioModel, SimulateConfig
from biolearn.models.nfc import NFC
from biolearn.specifications import PhiXorFast


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, state_size: int, hidden_size: int = 32):
        k1, k2, k3 = jr.split(key, 3)
        out_layer = eqx.nn.Linear(hidden_size, state_size, key=k3)
        out_layer = eqx.tree_at(lambda l: l.weight, out_layer, out_layer.weight * 0.01)
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

    def __init__(self, key: jax.Array, nominal: BioModel):
        self.nominal_model = nominal
        state_size = math.prod(nominal.shape)
        self.residual_model = MLP(key, state_size=state_size, hidden_size=64)

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
        interp = NFC._handle_inputs(x, x_ts)
        y0 = jnp.zeros(self.nominal_model.shape)

        solver = diffrax.Kvaerno5() if config.stiff else diffrax.Tsit5()
        # solver = diffrax.Tsit5()

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
            args=(interp,),
        )
        return solution.ys, solution


def residual_l2(model: BufferModel) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(model.residual_model, eqx.is_inexact_array)
    )
    return sum((jnp.sum(w**2) for w in leaves), jnp.zeros(()))


def run_one(key: jax.Array, lr: float, n_epochs: int, reg_weight: float = 1e-4):
    key, subkey = jr.split(key)
    nominal_model = MoormanNFC(2, [2, 1], k=0.8, key=subkey)
    key, subkey = jr.split(key)

    model = BufferModel(subkey, nominal_model)

    spec = PhiXorFast()

    sim_cfg = SimulateConfig(
        to_ss=False,
        stiff=True,
        throw=True,
        max_steps=int(2e3),
        rtol=1e-3,
        atol=1e-4,
        max_stepsize=0.5,
        progress_bar=False,
    )

    ts = jnp.arange(0.0, 20.0, 1.0)

    schedule = optax.exponential_decay(lr, 400, 0.5)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(schedule),
    )

    def ss_to_traj(x, y_trace):
        y_traj = y_trace[..., -1, 0][..., None]  # (B, T, 1)
        x_traj = jnp.ones_like(y_traj) * x[..., None, :]  # (B, T, 2)

        return jnp.concatenate([x_traj, y_traj], axis=-1)

    @eqx.filter_value_and_grad
    def grad_loss(model, x0, ts):
        sim = partial(model.simulate, ts=ts, config=sim_cfg)
        y_traces, solutions = jax.vmap(sim)(x0)
        # ok = jax.vmap(lambda s: s.result == diffrax.RESULTS.successful)(solutions)
        # ok_f = ok.astype(jnp.float32)
        # safe_traj = jnp.where(ok[:, None, None], ss_to_traj(x0, y_traces), 0.0)
        safe_traj = ss_to_traj(x0, y_traces)
        rhos = jax.vmap(spec.evaluate)(safe_traj)
        # n = jnp.array(x0.shape[0], dtype=jnp.float32)
        task_loss = jax.nn.relu(-rhos).mean()
        return task_loss + reg_weight * residual_l2(model)

    # TODO: Change the dataset
    xs = jnp.linspace(0, 1.0, 32)
    xs, ys = jnp.meshgrid(*[xs, xs])
    x_test = jnp.stack([xs.flatten(), ys.flatten()], axis=-1)

    t_horizons = [20.0]

    assert n_epochs % len(t_horizons) == 0, (
        f"Number of epochs ({n_epochs}) can't be split into len(t_horizons)={len(t_horizons)} chunks evenly."
    )

    all_losses = []
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    n_points = 256
    n_boundary = n_points // 4
    n_inside = n_points - n_boundary
    for t_final in tqdm(t_horizons, desc="Horizons"):
        ts = jnp.arange(0.0, t_final, 1.0)

        pbar = tqdm(range(n_epochs // len(t_horizons)), desc=f"t={t_final}")
        for _ in pbar:
            # Sample points
            key, subkey = jr.split(key)
            edge_points = jr.uniform(subkey, (n_boundary,))
            step = n_boundary // 4
            x0 = jnp.stack([jnp.zeros(step), edge_points[:step]], axis=-1)
            x1 = jnp.stack([jnp.ones(step), edge_points[step : 2 * step]], axis=-1)
            x2 = jnp.stack([edge_points[2 * step : 3 * step], jnp.zeros(step)], axis=-1)
            x3 = jnp.stack([edge_points[3 * step :], jnp.ones(step)], axis=-1)

            key, subkey = jr.split(key)
            xs = jnp.concatenate([x0, x1, x2, x3, jr.uniform(subkey, (n_inside, 2))])

            # TODO: batch
            loss, grads = grad_loss(model, xs, ts)
            updates, opt_state = optimizer.update(grads, opt_state)
            tqdm.set_description(pbar, f"{loss:.2e}")
            model = eqx.apply_updates(model, updates)
            all_losses.append(float(loss))

    plt.plot(np.arange(len(all_losses)), all_losses)
    plt.yscale("log")
    plt.show()
    kd = jr.key_data(key)
    plt.savefig(f"losses-{kd[0]}{kd[1]}.png")

    sim = partial(model.simulate, ts=ts, config=sim_cfg)
    y_traces, _ = jax.vmap(sim)(x_test)
    traj = ss_to_traj(x_test, y_traces)
    rhos = jax.vmap(spec.evaluate)(traj)

    sats = jnp.sum(rhos > 0).astype(int)
    print(rhos.shape, sats.shape)

    print(f"Satisfied {sats}/{x_test.shape[0]} of initial conditions")


def main():
    n_epochs = 1000
    lr = 1e-2
    n_runs = 3
    reg = 1e-1

    key = jr.key(42)

    for k in jr.split(key, n_runs):
        run_one(k, lr, n_epochs, reg_weight=reg)


if __name__ == "__main__":
    main()
