"""Slack soft maximum loss function.

A max loss with a slack variables to encourage robustness
The slack lives on a SlackModel wrapper so that it is optimized as a regular
model parameter — no changes to the training loop are needed.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from .base import make_slack_loss


def slack_softmax_loss(
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    C: float = 1e-2,
    temperature: float = 1.0,
    **kwargs,
):
    """Create a slack-softmax loss function over initial conditions.

    Returns a plain function with signature ``(system, xs, _ys) -> scalar``,
    compatible with the standard training loop.  ``system`` must be a
    :class:`~biolearn.losses.slack_relu.SlackModel` so the loss can read the slack variables.
    """
    return make_slack_loss(
        slack_group_loss=lambda ros, slack: (
            jax.nn.logsumexp((slack - ros) / temperature) - C * slack
        ).mean(),
        specification=specification,
        ts=ts,
        **kwargs,
    )
