from __future__ import annotations

from functools import partial

import jax

from .base import make_loss

make_temporal_xor_ss_loss = partial(
    make_loss, group_loss=lambda r: jax.nn.relu(-r).sum()
)
