from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt

from .base import BioModel, SimulateConfig


class Quadrotor(eqx.Module):
    log_body_mass: jax.Array
    log_rotor_mass: jax.Array
    log_length: jax.Array
    log_radius: jax.Array
    kp: jax.Array
    kd: jax.Array

    g: float = eqx.field(static=True)
    h_ref: float = eqx.field(static=True)

    def __init__(self, key: jax.Array):
        km, kl, krm, kr, kp, kd = jr.split(key, 6)

        self.log_body_mass = jr.normal(km) * 1.5
        self.log_rotor_mass = jr.normal(krm) * 0.5

        self.log_length = jr.normal(kl) * 1.0
        self.log_radius = jr.normal(kr) * 1.0

        self.kp = jr.uniform(kp, minval=0.1, maxval=15.0)
        self.kd = jr.uniform(kd, minval=0.1, maxval=5.0)

        self.g = 9.81
        self.h_ref = 1.0

    @property
    def shape(self) -> Tuple[int]:
        return (12,)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Integrate the quadrotor state, 12-D.
        x = [pn, dpn, pe, dpe, h, dh, phi, dphi, theta, dtheta, psi, dpsi]
        """
        bm = jnp.exp(self.log_body_mass)
        rm = jnp.exp(self.log_rotor_mass)
        mass = bm + rm
        body_inertia = 0.4 * bm * (jnp.exp(2 * self.log_radius))
        rotor_inertia = 2.0 * rm * (jnp.exp(2.0 * self.log_length))

        inertia_x = body_inertia + rotor_inertia
        inertia_y = body_inertia + rotor_inertia
        inertia_y = body_inertia + 2.0 * rotor_inertia

        phi = x[6]
        theta = x[8]
        psi = x[10]
        controller_f = mass * self.g - self.kp * (x[4] - self.h_ref) - self.kd * x[5]
        controller_f = (jnp.tanh(controller_f / mass / self.g) + 2) * mass * self.g / 2

        phi_torque = -phi - x[7]
        theta_torque = -theta - x[9]

        dpn = x[1]
        ddpn = (
            controller_f
            / mass
            * (
                -jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi)
                - jnp.sin(phi) * jnp.sin(psi)
            )
        )

        dpe = x[3]
        ddpe = (
            controller_f
            / mass
            * (
                -jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi)
                + jnp.sin(phi) * jnp.cos(psi)
            )
        )

        dh = x[5]
        ddh = controller_f / mass * (jnp.cos(phi) * jnp.cos(theta)) - self.g

        dphi = x[7]
        ddphi = phi_torque / inertia_x
        dtheta = x[9]
        ddtheta = theta_torque / inertia_y
        dpsi = x[11]
        ddpsi = jnp.array(0.0)

        return jnp.zeros_like(x) + jnp.array(
            [dpn, ddpn, dpe, ddpe, dh, ddh, dphi, ddphi, dtheta, ddtheta, dpsi, ddpsi]
        )


class QuadModel(BioModel):
    model: Quadrotor

    @property
    def shape(self) -> Tuple[int]:
        return (12,)

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
