import jax
import jax.numpy as jnp

from ..models import BioModel, SimulateConfig


class Task:
    """Abstract base class for defining a benchmark task."""

    domain_low: jax.Array  # Lower coordinate of box domain
    domain_hi: jax.Array  # Upper coordinate of box domain

    model: BioModel  # The model type

    ts: jax.Array  # Simulation time domain

    def traj_fn(self, x0, y_trace):
        """Convert raw simulation output to spec-compatible trajectory."""
        return y_trace

    def evaluate_robustness(
        self, config: SimulateConfig = SimulateConfig()
    ) -> jax.Array:
        """The specification to satisfy."""
        raise NotImplementedError
