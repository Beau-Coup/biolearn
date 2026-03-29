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

        nominal_linear = jnp.log(jnp.array([1.4, 0.6, 0.6, 0.5, 0.4, 0.1, 0.3]))
        nominal_bilinear = jnp.log(jnp.array([0.9, 0.3, 0.8, 0.5, 0.2, 0.4, 0.1]))

        self.log_linear = nominal_linear + jr.normal(kl, (7,)) * 0.5
        self.log_bilinear = nominal_bilinear + jr.normal(kbl, (7,)) * 0.5

    @property
    def shape(self) -> Tuple[int]:
        return (7,)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Integrate the state"""
        linear = jnp.exp(self.log_linear)
        linear = linear.at[jnp.array([0, 2])].set(jnp.array([1.4, 0.6]))
        bilinear = jnp.exp(self.log_bilinear)
        bilinear = bilinear.at[jnp.array([0, 2])].set(jnp.array([0.9, 0.8]))

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
            ts = jnp.arange(0, 20, 1.0)
        return self._simulate(x, ts, config)
