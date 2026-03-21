from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..models import BioGnnModel
from ..models.base import SimulateConfig
from ..specifications import FastProduce
from ..utils import sample_hypercube_faces
from . import Task


class QuadDynamics(eqx.Module):
    mass: jax.Array

    def __init__(self, key: jax.Array):
        self.mass = jnp.ones(1)
        self.g = jnp.array([9.81])
        self.h_ref = jnp.array(1.0)
        self.inertia = jnp.ones(3)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Integrate the quadrotor state, 12-D.
        x = [pn, dpn, pe, dpe, h, dh, phi, dphi, theta, dtheta, psi, dpsi]
        """
        phi = x[6]
        theta = x[8]
        psi = x[10]
        controller_f = self.mass * self.g - 10.0 * (x[4] - self.h_ref) - 3.0 * x[5]
        phi_torque = -phi - x[7]
        theta_torque = -theta - x[9]

        dpn = x[1]
        ddpn = (
            controller_f
            / self.mass
            * (
                -jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi)
                - jnp.sin(phi) * jnp.sin(psi)
            )
        )

        dpe = x[3]
        ddpe = (
            controller_f
            / self.mass
            * (
                -jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi)
                + jnp.sin(phi) * jnp.cos(psi)
            )
        )

        dh = x[5]
        ddh = controller_f / self.mass * (jnp.cos(phi) * jnp.cos(theta)) - self.g

        dphi = x[7]
        ddphi = phi_torque / self.inertia[0]
        dtheta = x[9]
        ddtheta = theta_torque / self.inertia[1]
        dpsi = x[11]
        ddpsi = jnp.array(0.0)

        return jnp.zeros_like(x) + jnp.array(
            [dpn, ddpn, dpe, ddpe, dh, ddh, dphi, ddphi, dtheta, ddtheta, dpsi, ddpsi]
        )


class QuadRotor(Task):
    def __init__(self):
        self.domain_hi = jnp.ones(6)
        self.domain_low = jnp.zeros_like(self.domain_hi)
        self.ts = jnp.arange(0, 25, 1.0)

        # The spec for evaluating satisfaction, not for gradients
        self.spec = FastProduce("classical")

        # The model to use

        self.model = BioGnnModel(gnn)
        self.eval_set = self._generate_eval_points(jr.key(42), 3000)

    def _generate_eval_points(self, key: jax.Array, n_points: int) -> jax.Array:
        # Sample from the domain and measure the satisfaction rate
        eval_points = jr.uniform(
            key, (n_points, 6), minval=self.domain_low, maxval=self.domain_hi
        )

        # Generate samples on the boundary of the domain.
        boundary_eval_points = sample_hypercube_faces(
            key, self.domain_low, self.domain_hi, n_per_face=5
        )

        return jnp.concatenate([eval_points, boundary_eval_points], axis=0)

    def evaluate_robustness(self, config: Optional[SimulateConfig] = None) -> jax.Array:
        """Evaluate the robustness of the model w.r.t. the fixed evaluation points.
        Acts as a test set for the task.
        """

        cfg = (
            config
            if config
            else SimulateConfig(
                to_ss=False,
                stiff=True,
                throw=True,
                max_steps=int(1e6),
                rtol=1e-6,
                atol=1e-6,
                max_stepsize=None,
                progress_bar=False,
            )
        )

        def _eval_single(x0):
            y_trace, _ = self.model.simulate(x0, ts=self.ts, config=cfg)
            return self.spec.evaluate(y_trace)

        run = jax.vmap(_eval_single)

        return run(self.eval_set)
