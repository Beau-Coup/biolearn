"""Loss function that rates a specification on a set of initial conditions.

The total robustness is the sigmoidal sum of the robustnesses on each initial condition.
"""

import jax

from .base import make_loss


def sigmoid_combination(robustnesses: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(-robustnesses).mean() - 0.5


def sigmoid_ic_loss(specification, ts, **kwargs):
    if "group_loss" in kwargs:
        raise TypeError(
            "group_loss is fixed for sigmoid_ic_loss; use make_loss directly."
        )
    return make_loss(
        group_loss=sigmoid_combination, specification=specification, ts=ts, **kwargs
    )
