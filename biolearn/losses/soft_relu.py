from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .base import make_loss


def make_softrelu_loss(
    specification: Callable[[jax.Array], jax.Array], ts: jax.Array, C: float = 1e-2
):
    return make_loss(
        lambda r: jnp.mean(jax.nn.relu(-r) + C * jax.nn.sigmoid(jax.nn.relu(r))),
        specification,
        ts,
    )
