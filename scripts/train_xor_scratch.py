"""Write a training script from scratch to easily tune things to have a story for the paper and then bring it all back."""

from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from biolearn.models import MoormanNFC
from biolearn.models.base import SimulateConfig
from biolearn.specifications import PhiXorFast


def run_one(key: jax.Array, lr: float, n_epochs: int):

    key, subkey = jr.split(key)
    model = MoormanNFC(2, [2, 1], k=0.8, key=subkey)
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
        ts = jnp.arange(0.0, t_final, 0.5)

        @eqx.filter_jit
        def run_epoch(carry, _):
            model, opt_state = carry

            # TODO: batch
            loss, grads = grad_loss(model, xs, ts)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return (model, opt_state), loss

        (model, opt_state), losses = jax.lax.scan(
            run_epoch, (model, opt_state), xs=None, length=n_epochs // len(t_horizons)
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
    n_epochs = 1000
    lr = 0.05
    n_runs = 3

    key = jr.key(42)

    for k in jr.split(key, n_runs):
        run_one(k, lr, n_epochs)


if __name__ == "__main__":
    main()
