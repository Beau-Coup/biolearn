import itertools

import jax
import jax.numpy as jnp
import jax.random as jr


def sample_hypercube_faces(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    n_per_face: int = 4,
    max_k: int | None = None,
) -> jax.Array:
    """Sample points from faces of the hypercube [low, high]^n.

    A k-face fixes (n-k) dimensions to low/high, leaving k dimensions free.
    k=0 gives corners , k>=1 samples uniform points per face.

    Parameters:
    ----------
        key: PRNG key.
        low: (n,) Box lower bound
        high: (n,) Box upper bound
        n_per_face: Number of samples per face for k >= 1.
        max_k: Highest face dimension to include. Defaults to n-1.

    Returns:
    -------
        Array of shape (total_points, n) of sampled boundary points.
    """
    n = low.shape[0]
    if max_k is None:
        max_k = n - 1

    blocks = []

    # k=0: corners — all 2^n combinations of {low, high}
    corners = []
    for bits in itertools.product([0, 1], repeat=n):
        corner = jnp.array([high[i] if b else low[i] for i, b in enumerate(bits)])
        corners.append(corner)

    blocks.append(jnp.stack(corners))

    for k in range(1, max_k + 1):
        # Choose which k dims are free
        for free_dims in itertools.combinations(range(n), k):
            fixed_dims = [d for d in range(n) if d not in free_dims]
            # Each fixed dim can be low or high
            for fixed_vals in itertools.product([0, 1], repeat=n - k):
                key, subkey = jr.split(key)
                # Sample free dims uniformly in [0, 1], then scale
                uniform = jr.uniform(subkey, (n_per_face, k))
                points = jnp.empty((n_per_face, n))
                # Set fixed dims
                for fd, fv in zip(fixed_dims, fixed_vals):
                    val = high[fd] if fv else low[fd]
                    points = points.at[:, fd].set(val)
                # Set free dims
                for j, fd in enumerate(free_dims):
                    points = points.at[:, fd].set(
                        low[fd] + uniform[:, j] * (high[fd] - low[fd])
                    )
                blocks.append(points)

    return jnp.concatenate(blocks, axis=0)


def sample_hypercube_budget(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    n_boundary: int,
    max_k: int | None = None,
) -> jax.Array:
    """Sample boundary points with a fixed total budget.

    Allocates points with exponential decay across face dimensions:
    k=0 (corners) gets ~half the budget, k=1 ~quarter, etc.
    Within each level, samples are spread across faces round-robin
    with 1 sample per face until the level's budget is exhausted.

    Parameters:
    ----------
        key: PRNG key.
        low: (n,) Box lower bound
        high: (n,) Box upper bound
        n_boundary: Total number of boundary points to return.
        max_k: Highest face dimension to include. Defaults to n-1.

    Returns:
    -------
        Array of shape (n_boundary, n).
    """
    n = low.shape[0]
    if max_k is None:
        max_k = n - 1

    # Exponential budget: k=0 gets ~half, k=1 gets ~quarter, ...
    budgets = []
    remaining = n_boundary
    for k in range(max_k + 1):
        if k == max_k:
            budgets.append(remaining)
        else:
            alloc = max(1, remaining // 2)
            budgets.append(alloc)
            remaining -= alloc

    blocks = []

    for k in range(max_k + 1):
        budget_k = budgets[k]
        if budget_k <= 0:
            continue

        if k == 0:
            corners = []
            for bits in itertools.product([0, 1], repeat=n):
                if len(corners) >= budget_k:
                    break
                corners.append(
                    jnp.array([high[i] if b else low[i] for i, b in enumerate(bits)])
                )
            blocks.append(jnp.stack(corners))
        else:
            # Enumerate all faces at this k-level
            face_list = []
            for free_dims in itertools.combinations(range(n), k):
                fixed_dims = [d for d in range(n) if d not in free_dims]
                for fixed_vals in itertools.product([0, 1], repeat=n - k):
                    face_list.append((free_dims, fixed_dims, fixed_vals))

            n_faces = len(face_list)
            n_per_face = max(1, budget_k // n_faces)
            count = 0

            for free_dims, fixed_dims, fixed_vals in face_list:
                if count >= budget_k:
                    break
                key, subkey = jr.split(key)
                take = min(n_per_face, budget_k - count)
                uniform = jr.uniform(subkey, (take, k))
                points = jnp.empty((take, n))
                for fd, fv in zip(fixed_dims, fixed_vals):
                    val = high[fd] if fv else low[fd]
                    points = points.at[:, fd].set(val)
                for j, fd in enumerate(free_dims):
                    points = points.at[:, fd].set(
                        low[fd] + uniform[:, j] * (high[fd] - low[fd])
                    )
                blocks.append(points)
                count += take

    return jnp.concatenate(blocks, axis=0)[:n_boundary]
