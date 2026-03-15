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
