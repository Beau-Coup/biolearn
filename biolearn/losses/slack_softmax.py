"""Slack soft maximum loss function.

A max loss with a slack variables to encourage robustness
The slack lives on a SlackModel wrapper so that it is optimized as a regular
model parameter — no changes to the training loop are needed.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from biolearn.losses import SlackModel


def slack_softmax_loss(
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    C: float = 1e-2,
    temperature: float = 1.0,
    **kwargs,
):
    """Create a slack-ReLU loss function over initial conditions.

    Returns a plain function with signature ``(system, xs, _ys) -> scalar``,
    compatible with the standard training loop.  ``system`` must be a
    :class:`SlackModel` so the loss can read the slack variables.
    """

    @eqx.filter_jit
    def _loss(system: SlackModel, xs, _ys):
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

        loss = jax.nn.logsumexp((system.slack - ros) / temperature) - C * system.slack
        return loss.mean()

    return _loss
