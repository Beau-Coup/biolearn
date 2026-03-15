from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from ..models import MoormanNFC, SimulateConfig
from ..specifications.ss_classification import PhiXorFast
from ..utils import sample_hypercube_faces
from . import Task


class NfcXor(Task):
    def __init__(self, key: jax.Array, layer_sizes):
        self.domain_hi = jnp.ones(2)
        self.domain_low = jnp.zeros_like(self.domain_hi)
        self.ts = jnp.arange(0, 20, 1.0)

        # The spec for evaluating satisfaction, not for gradients
        self.spec = PhiXorFast(semantics="classical")

        key1, key2 = jr.split(key)
        self.model = MoormanNFC(
            n_inputs=2,
            layer_sizes=layer_sizes,
            gamma=1000,
            beta=1.0,
            k=0.8,
            key=key1,
        )

        self.eval_set = self._generate_eval_points(key2, 512)

    def traj_fn(self, x0, y_trace):
        """XOR: repeat 2D input + last species output."""
        x_traj = jnp.repeat(jnp.array([[x0[0], x0[1]]]), y_trace.shape[0], axis=0)
        y_out = y_trace[:, -1][:, None]
        return jnp.concatenate([x_traj, y_out], axis=1)

    def _generate_eval_points(self, key: jax.Array, n_points: int) -> jax.Array:
        # Sample from the domain and measure the satisfaction rate
        eval_points = jr.uniform(
            key,
            (n_points, self.domain_low.shape[0]),
            minval=self.domain_low,
            maxval=self.domain_hi,
        )

        # Generate samples on the boundary of the domain.
        boundary_eval_points = sample_hypercube_faces(
            key, self.domain_low, self.domain_hi, n_per_face=32
        )

        return jnp.concatenate([eval_points, boundary_eval_points], axis=0)

    def evaluate_robustness(self, config: Optional[SimulateConfig] = None) -> jax.Array:
        """Evaluate the robustness of the model w.r.t. the fixed evaluation points.
        Acts as a test set for the task.
        """
        if not config:
            trajectories = self.model.simulate(self.eval_set)
        else:
            trajectories = self.model.simulate(self.eval_set, config)

        return self.spec.evaluate(trajectories)
