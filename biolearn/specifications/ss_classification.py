from __future__ import annotations

import jax
import jax.numpy as jnp


def xor_ss_spec(
    traj: jax.Array,
    *,
    eps1: float = 0.1,
    eps2: float = 0.05,
    t1: int = 5,
) -> jax.Array:
    """Simple steady-state XOR robustness for a trajectory.

    Expects traj shape (T, 3) with columns [x1, x2, y].
    """
    x1 = traj[0, 0]
    x2 = traj[0, 1]
    y_target = jax.nn.relu(jnp.abs(x2 - x1) - eps1)

    idx = min(max(int(t1), 0), traj.shape[0] - 1)
    y_pred = traj[idx, 2]

    return eps2 - jnp.abs(y_pred - y_target)
