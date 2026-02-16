"""Slack ReLU loss function.

A ReLU-like loss with per-sample slack variables that keep gradients alive.
The slack lives on a SlackModel wrapper so that it is optimized as a regular
model parameter — no changes to the training loop are needed.

Loss per sample: relu(slack_i - robustness_i) - c * slack_i
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from biolearn.models.nfc import BioSyst


class SlackModel(eqx.Module):
    """Wraps a model with per-sample slack variables.

    The slack array becomes a trainable parameter of the combined model,
    so existing training loops optimise it alongside the original parameters.

    Usage::

        wrapped = SlackModel(my_biosyst, num_samples=len(x_train))
        loss_fn = slack_relu_ic_loss(specification=spec, ts=ts)
        # train wrapped as usual, then unwrap:
        trained_model = trained_wrapped.model
    """

    model: BioSyst
    slack: jax.Array

    def __init__(self, model: BioSyst):
        self.model = model
        self.slack = jnp.zeros(()) - 10.0

    def simulate(self, *args, **kwargs):
        return self.model.simulate(*args, **kwargs)


def slack_relu_ic_loss(
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    C: float = 1.0,
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

        loss = jax.nn.relu(system.slack - ros) - C * system.slack
        return loss.mean()

    return _loss
