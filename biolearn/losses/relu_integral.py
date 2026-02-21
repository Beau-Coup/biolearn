"""ReLU integral loss function.

Monte-Carlo integral of ReLU(-robustness) over a box domain, using
:func:`~biolearn.losses.base.make_integral_loss` for random sampling.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import BoxDomain, make_integral_loss


def relu_integral_ic_loss(
    domain: BoxDomain,
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    n_points: int = 100,
    *,
    key: jax.Array,
    **kwargs,
):
    """Create a ReLU-integral loss over initial conditions.

    The loss approximates via Monte-Carlo sampling::

        integral_domain relu(-rho(x)) dx

    Random sample points are drawn from *domain* each step using
    :func:`make_integral_loss`.  A unique PRNG key is derived per call
    by folding in a hash of the *xs* batch, so no training-loop changes
    are needed.

    Returns a function with signature ``(system, xs, _ys) -> scalar``.
    """
    _integral_fn = make_integral_loss(
        integrand=lambda r: jax.nn.relu(-r).mean(),
        domain=domain,
        specification=specification,
        ts=ts,
        **kwargs,
    )

    @eqx.filter_jit
    def _loss(system, xs, _ys):
        hash_val = jnp.bitwise_xor.reduce(
            jax.lax.bitcast_convert_type(xs.flatten(), jnp.int32)
        )
        step_key = jax.random.fold_in(key, hash_val.sum())
        return _integral_fn(step_key, system, n_points)

    return _loss
