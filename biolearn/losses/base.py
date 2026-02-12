"""Base class for loss functions.

A loss function evaluates a trajectory

Author: Alex Beaudin
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..models import BioSyst


def make_loss(
    group_loss: Callable[[jax.Array], jax.Array],
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    **kwargs,
):
    @eqx.filter_jit
    def _loss(system: BioSyst, xs, _ys):
        def _run_single(x0):
            y_trace, _ = system.simulate(
                x=x0,
                ts=ts,
                to_ss=False,
                stiff=True,
                max_steps=int(1e6),
                rtol=1e-6,
                atol=1e-6,
                progress_bar=False,
            )
            x_traj = jnp.array([[x0[0], x0[1]]])
            x_traj = jnp.repeat(x_traj, repeats=y_trace.shape[0], axis=0)

            y_out = y_trace[:, -1][:, None]
            traj = jnp.concatenate([x_traj, y_out], axis=1)

            return specification(traj, **kwargs)

        ros = jax.vmap(_run_single)(xs)

        return group_loss(ros)

    return _loss
