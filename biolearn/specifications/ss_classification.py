from __future__ import annotations

import jax
import jax.numpy as jnp
import pystl
from pystl.api import Signal

from .common import BaseSpec, get_semantics


class PhiXorFast(BaseSpec):
    """
    STL specification implementing an XOR classification
    at steady state.

    Error should be less than ep1 at t1,
    and eventually less than eps2.

    Expects traj shape (T, 3) with columns [x1, x2, y].
    """

    def __init__(
        self,
        eps1: float = 0.1,
        eps2: float = 0.05,
        t1: int = 5,
        semantics: str = "classical",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        interval = (0, int(t1)) if t1 is not None else (0, None)

        err_low = pystl.Predicate("err<eps1", fn=lambda sig, t: eps1 - sig[t])
        err_lowest = pystl.Predicate("err<eps2", fn=lambda sig, t: eps2 - sig[t])

        phi1 = pystl.Eventually(pystl.Always(err_low), interval=interval)
        phi2 = pystl.Eventually(pystl.Always(err_lowest))
        phi = phi1 & phi2

        self.sematics = get_semantics(semantics, dgmsr_p, smooth_temperature)
        self.spec = phi

    def evaluate(
        self,
        traj: jax.Array,
    ) -> jax.Array:
        x_diff = traj[:, 1] - traj[:, 0]
        y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
        y_pred = traj[:, 2]
        err = jnp.abs(y_true - y_pred)
        ro = self.spec.evaluate(Signal(err), self.semantics, t=0)
        return jnp.asarray(ro).squeeze()


class PhiXorSS(BaseSpec):
    """
    STL specification implementing an XOR classification
    at steady state.

    Error should eventually be less than eps.

    Expects traj shape (T, 3) with columns [x1, x2, y].
    """

    def __init__(
        self,
        eps: float = 0.1,
        semantics: str = "dgmsr",
        dgmsr_p: int = 3,
        smooth_temperature: float = 1.0,
    ):
        self.semantics = get_semantics(semantics, dgmsr_p, smooth_temperature)
        err_low = pystl.Predicate("err<eps1", fn=lambda sig, t: eps - sig[t])

        self.spec = pystl.Eventually(pystl.Always(err_low))

    def evaluate(self, traj: jax.Array) -> jax.Array:
        x_diff = traj[:, 1] - traj[:, 0]
        y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
        y_pred = traj[:, 2]
        err = jnp.abs(y_true - y_pred)

        ro = self.spec.evaluate(Signal(err), self.semantics, t=0)
        return jnp.asarray(ro).squeeze()
