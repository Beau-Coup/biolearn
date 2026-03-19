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

    def __init__(self, key: jax.Array, state_size: int, hidden_size: int = 128):
        k1, k2, k3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_size, hidden_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, state_size, key=k3),
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
        self.residual_model = MLP(key, state_size=state_size, hidden_size=128)

    def _step(self, t, y, args):
        return self.nominal_model.diffrax_step(t, y, args) + self.residual_model(
            y.flatten()
        ).reshape(y.shape)

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


def run_one(key: jax.Array, lr: float, n_epochs: int):

    key, subkey = jr.split(key)
    nominal_model = MoormanNFC(2, [2, 1], k=0.8, key=subkey)
    key, subkey = jr.split(key)

    model = BufferModel(subkey, nominal_model)

    spec = PhiXorFast()

    sim_cfg = SimulateConfig(
        to_ss=False,
        stiff=True,
        throw=True,
        max_steps=int(1e6),
        rtol=1e-6,
        atol=1e-6,
        max_stepsize=None,
        progress_bar=False,
    )

    # TODO: Update the 20.0 to increase over time
    ts = jnp.arange(0.0, 20.0, 1.0)

    optimizer = optax.adabelief(lr)

    def ss_to_traj(x, y_trace):
        y_traj = y_trace[..., -1, 0][..., None]  # (B, T, 1)
        x_traj = jnp.ones_like(y_traj) * x[..., None, :]  # (B, T, 2)

        return jnp.concatenate([x_traj, y_traj], axis=-1)

    @eqx.filter_value_and_grad
    def grad_loss(model, x0, ts):
        sim = partial(model.simulate, ts=ts, config=sim_cfg)
        y_traces, _ = jax.vmap(sim)(x0)
        traj = ss_to_traj(x0, y_traces)
        rhos = jax.vmap(spec.evaluate)(traj)
        return jax.nn.relu(-rhos).mean()

    # TODO: Change the dataset
    xs = jnp.linspace(0, 1.0, 11)
    xs, ys = jnp.meshgrid(*[xs, xs])
    xs = jnp.stack([xs.flatten(), ys.flatten()], axis=-1)

    t_horizons = [20.0]

    assert n_epochs % len(t_horizons) == 0, (
        f"Number of epochs ({n_epochs}) can't be split into len(t_horizons)={len(t_horizons)} chunks evenly."
    )

    all_losses = []
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    for t_final in tqdm(t_horizons, desc="Horizons"):
        ts = jnp.arange(0.0, t_final, 1.0)

        params, static = eqx.partition(model, eqx.is_array)

        @eqx.filter_jit
        def run_epoch(carry, _):
            params, opt_state, key = carry
            model = eqx.combine(params, static)

            # Sample points
            key, subkey = jr.split(key)
            edge_points = jr.uniform(subkey, (32,))
            x0 = jnp.stack([jnp.zeros(8), edge_points[:8]], axis=-1)
            x1 = jnp.stack([jnp.ones(8), edge_points[8:16]], axis=-1)
            x2 = jnp.stack([edge_points[16:24], jnp.zeros(8)], axis=-1)
            x3 = jnp.stack([edge_points[24:], jnp.ones(8)], axis=-1)

            key, subkey = jr.split(key)
            xs = jnp.concatenate([x0, x1, x2, x3, jr.uniform(subkey, (96, 2))])

            # TODO: batch
            loss, grads = grad_loss(model, xs, ts)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            params, _ = eqx.partition(model, eqx.is_array)
            return (params, opt_state, key), loss

        (params, opt_state, key), losses = jax.lax.scan(
            run_epoch,
            (params, opt_state, key),
            xs=None,
            length=n_epochs // len(t_horizons),
        )

        all_losses.extend(losses)

    plt.plot(np.arange(len(all_losses)), all_losses)
    plt.yscale("log")
    plt.show()

    sim = partial(model.simulate, ts=ts, config=sim_cfg)
    y_traces, _ = jax.vmap(sim)(xs)
    traj = ss_to_traj(xs, y_traces)
    rhos = jax.vmap(spec.evaluate)(traj)

    sats = jnp.sum(rhos > 0).astype(int)
    print(rhos.shape, sats.shape)

    print(f"Satisfied {sats}/{xs.shape[0]} of initial conditions")


def main():
    n_epochs = 100
    lr = 0.05
    n_runs = 3

    key = jr.key(42)

    for k in jr.split(key, n_runs):
        run_one(k, lr, n_epochs)


if __name__ == "__main__":
    main()
