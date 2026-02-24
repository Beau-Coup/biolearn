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

from .base import make_slack_loss


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
    slack_raw: jax.Array

    def __init__(self, model: BioSyst):
        self.model = model
        self.slack_raw = jnp.asarray(0.1)

    @property
    def slack(self) -> jax.Array:
        return jax.nn.relu(self.slack_raw)

    def simulate(self, *args, **kwargs):
        return self.model.simulate(*args, **kwargs)


def slack_relu_ic_loss(
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    C: float = 1e-2,
    **kwargs,
):
    """Create a slack-ReLU loss function over initial conditions.

    Returns a plain function with signature ``(system, xs, _ys) -> scalar``,
    compatible with the standard training loop.  ``system`` must be a
    :class:`SlackModel` so the loss can read the slack variables.
    """
    return make_slack_loss(
        slack_group_loss=lambda ros, slack: (jax.nn.relu(slack - ros) - C * slack).mean(),
        specification=specification,
        ts=ts,
        **kwargs,
    )
