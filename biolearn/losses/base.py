"""Base class for loss functions.

A loss function evaluates a trajectory

Author: Alex Beaudin
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..models import BioSyst


def _robustnesses(system, xs, ts, specification, **kwargs):
    """Simulate each initial condition and return per-sample robustnesses."""

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

    return jax.vmap(_run_single)(xs)


def make_loss(
    group_loss: Callable[[jax.Array], jax.Array],
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    domain: "BoxDomain | None" = None,
    n_points: int = 128,
    key: "jax.Array | None" = None,
    **kwargs,
):
    """Create a compiled loss function.

    When *domain* is ``None`` (the default), robustnesses are computed on the
    batch *xs* supplied by the training loop.

    When a :class:`BoxDomain` and *key* are provided, Monte-Carlo sampling is
    used instead: *n_points* are drawn uniformly from *domain* each step and
    the batch *xs* is only used to derive a per-step PRNG key.
    """
    if domain is not None:
        assert key is not None, "key is required when domain is set"
        _integral_fn = make_integral_loss(
            weighting_fn=group_loss,
            domain=domain,
            specification=specification,
            ts=ts,
            **kwargs,
        )

        @eqx.filter_jit
        def _loss_integral(system: BioSyst, xs, _ys):
            hash_val = jnp.bitwise_xor.reduce(
                jax.lax.bitcast_convert_type(xs.flatten(), jnp.int32)
            )
            step_key = jax.random.fold_in(key, hash_val.sum())
            return _integral_fn(step_key, system, n_points)

        return _loss_integral

    @eqx.filter_jit
    def _loss(system: BioSyst, xs, _ys):
        ros = _robustnesses(system, xs, ts, specification, **kwargs)
        return group_loss(ros)

    return _loss


def make_slack_loss(
    slack_group_loss: Callable[[jax.Array, jax.Array], jax.Array],
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    **kwargs,
):
    """Like ``make_loss`` but for losses that use a learnable slack variable.

    ``slack_group_loss`` receives ``(robustnesses, slack)`` where *slack* is
    the scalar from :class:`~biolearn.losses.slack_relu.SlackModel`.
    """

    @eqx.filter_jit
    def _loss(system, xs, _ys):
        ros = _robustnesses(system, xs, ts, specification, **kwargs)
        return slack_group_loss(ros, system.slack)

    return _loss


class BoxDomain(eqx.Module):
    low: jax.Array
    high: jax.Array
    volume: jax.Array

    def __init__(self, low, high):
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self.volume = (high - low).prod()


def make_integral_loss(
    weighting_fn: Callable[[jax.Array], jax.Array],
    domain: BoxDomain,
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    **kwargs,
):
    """Monte-Carlo integration loss.
    Computes the integral of the loss over the specified domain.

    :param weighting_gn: the function to apply to the robustnesses
    :param domain: the bounding box over which to integrate
    :param specification: the STL spec to use to determine robustness
    :param ts: the time domain to simulate the system over
    """

    @eqx.filter_jit
    def _estimate_integral(key: jax.Array, system: BioSyst, n_points: int):
        points = jr.uniform(
            key,
            (
                n_points,
                domain.low.shape[0],
            ),
            minval=domain.low,
            maxval=domain.high,
        )
        ros = _robustnesses(system, points, ts, specification, **kwargs)
        return weighting_fn(ros)

    return _estimate_integral
