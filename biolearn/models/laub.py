from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt

from .base import BioModel, SimulateConfig


class LaubLoomis(eqx.Module):
    log_linear: jax.Array
    log_bilinear: jax.Array

    def __init__(self, key: jax.Array):
        kl, kbl = jr.split(key, 2)

        self.log_linear = jr.normal(kl, (7,)) * 1.5
        self.log_bilinear = jr.normal(kbl, (7,)) * 2.0

        self.g = 9.81
        self.h_ref = 1.0

    @property
    def shape(self) -> Tuple[int]:
        return (7,)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Integrate the quadrotor state, 12-D.
        x = [pn, dpn, pe, dpe, h, dh, phi, dphi, theta, dtheta, psi, dpsi]
        """
        linear = jnp.exp(self.log_linear)
        bilinear = jnp.exp(self.log_bilinear)

        linear_x = jnp.array([x[6], x[4], x[6], 1.0, x[0], x[0], x[5]])
        bilinear_x = jnp.array(
            [x[0] * x[1], x[1], x[1] * x[2], x[2] * x[3], x[3] * x[4], x[5], x[6]]
        )

        return linear * linear_x - bilinear * bilinear_x


class LLModel(BioModel):
    model: LaubLoomis

    @property
    def shape(self) -> Tuple[int]:
        return self.model.shape

    def diffrax_step(self, t: jt.ScalarLike, y: jax.Array, args: Tuple) -> jax.Array:
        return self.model(y)

    def simulate(
        self,
        x: jax.Array,
        ts: Optional[jax.Array],
        config: SimulateConfig = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=True,
            max_steps=int(1e6),
            rtol=1e-6,
            atol=1e-6,
            max_stepsize=None,
            progress_bar=False,
        ),
    ):
        if ts is None:
            ts = jnp.arange(0, 5, 1.0)
        return self._simulate(x, ts, config)
