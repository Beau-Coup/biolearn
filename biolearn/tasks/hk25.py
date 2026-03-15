from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from ..models import BioGNN, BioGnnModel, EdgeType
from ..models.base import SimulateConfig
from ..specifications import FastProduce
from ..utils import sample_hypercube_faces
from . import Task


class HK25(Task):
    def __init__(self):
        self.domain_hi = jnp.ones(6)
        self.domain_low = jnp.zeros_like(self.domain_hi)

        # The spec for evaluating satisfaction, not for gradients
        self.spec = FastProduce("classical")

        # The model to use
        gnn = BioGNN(
            [
                (0, 1, EdgeType.Activation),  # x1 -> x2
                (2, 3, EdgeType.Inhibition),  # x3 -| x4
                (3, 0, EdgeType.Inhibition),  # x4 -| x1
                (3, 1, EdgeType.Activation),  # x4 -> x2
                (4, 1, EdgeType.Activation),  # x5 -> x2
                (4, 2, EdgeType.Activation),  # x5 -> x3
                (4, 5, EdgeType.Activation),  # x5 -> x6
            ],
            2.0,
        )

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
        if not config:
            trajectories = self.model.simulate(self.eval_set)
        else:
            trajectories = self.model.simulate(self.eval_set, config)

        return self.spec.evaluate(trajectories)
