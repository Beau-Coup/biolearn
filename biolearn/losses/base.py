"""Base class for loss functions.

A loss function evaluates a trajectory

Author: Alex Beaudin
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..models import BioModel
from ..utils import sample_hypercube_faces
from ..models.base import SimulateConfig

_DEFAULT_SIM_CONFIG = SimulateConfig(
    to_ss=False, stiff=True, rtol=1e-6, atol=1e-6, progress_bar=False
)


def _default_traj_fn(x0, y_trace):
    """XOR-style: repeat 2D input + last species output."""
    x_traj = jnp.repeat(jnp.array([[x0[0], x0[1]]]), y_trace.shape[0], axis=0)
    y_out = y_trace[:, -1][:, None]
    return jnp.concatenate([x_traj, y_out], axis=1)


def _robustnesses(system, xs, ts, specification, traj_fn=None, config=None, **kwargs):
    """Simulate each initial condition and return per-sample robustnesses."""
    _make_traj = traj_fn if traj_fn is not None else _default_traj_fn
    if config is None:
        config = _DEFAULT_SIM_CONFIG

    def _run_single(x0):
        y_trace, _ = system.simulate(x=x0, ts=ts, config=config)
        traj = _make_traj(x0, y_trace)
        return specification(traj, **kwargs)

    return jax.vmap(_run_single)(xs)


def make_loss(
    group_loss: Callable[[jax.Array], jax.Array],
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    domain: "BoxDomain | None" = None,
    n_points: int = 128,
    key: "jax.Array | None" = None,
    n_boundary_points: int = 0,
    traj_fn=None,
    config: "SimulateConfig | None" = None,
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
            traj_fn=traj_fn,
            config=config,
            **kwargs,
        )

        @eqx.filter_jit
        def _loss_integral(system: BioModel, xs, _ys):
            hash_val = jnp.bitwise_xor.reduce(
                jax.lax.bitcast_convert_type(xs.flatten(), jnp.int32)
            )
            step_key = jax.random.fold_in(key, hash_val.sum())
            return _integral_fn(step_key, system, n_points, n_boundary_points)

        return _loss_integral

    @eqx.filter_jit
    def _loss(system: BioModel, xs, _ys):
        ros = _robustnesses(
            system, xs, ts, specification, traj_fn=traj_fn, config=config, **kwargs
        )
        return group_loss(ros)

    return _loss


def make_slack_loss(
    slack_group_loss: Callable[[jax.Array, jax.Array], jax.Array],
    specification: Callable[[jax.Array], jax.Array],
    ts: jax.Array,
    traj_fn=None,
    config: "SimulateConfig | None" = None,
    **kwargs,
):
    """Like ``make_loss`` but for losses that use a learnable slack variable.

    ``slack_group_loss`` receives ``(robustnesses, slack)`` where *slack* is
    the scalar from :class:`~biolearn.losses.slack_relu.SlackModel`.
    """

    @eqx.filter_jit
    def _loss(system, xs, _ys):
        ros = _robustnesses(
            system, xs, ts, specification, traj_fn=traj_fn, config=config, **kwargs
        )
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
    traj_fn=None,
    config: "SimulateConfig | None" = None,
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
    def _estimate_integral(
        key: jax.Array, system: BioModel, n_points: int, n_boundary_points: int
    ):
        key, bkey = jr.split(key)
        d = domain.low.shape[0]
        n_faces = d * 2 ** (d - 1)
        n_per_face = max(1, n_boundary_points // n_faces)
        boundary_points = sample_hypercube_faces(
            bkey, domain.low, domain.high, n_per_face=n_per_face, max_k=1
        )
        points = jr.uniform(
            key,
            (
                n_points,
                domain.low.shape[0],
            ),
            minval=domain.low,
            maxval=domain.high,
        )

        all_points = jnp.concatenate([points, boundary_points], axis=0)
        ros = _robustnesses(
            system,
            all_points,
            ts,
            specification,
            traj_fn=traj_fn,
            config=config,
            **kwargs,
        )
        return weighting_fn(ros)

    return _estimate_integral
