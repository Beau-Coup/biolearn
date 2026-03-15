import jax

from ..models import BioModel, SimulateConfig


class Task:
    """Abstract base class for defining a benchmark task."""

    domain_low: jax.Array  # Lower coordinate of box domain
    domain_hi: jax.Array  # Upper coordinate of box domain

    model: BioModel  # The model type

    def evaluate_robustness(
        self, config: SimulateConfig = SimulateConfig()
    ) -> jax.Array:
        """The specification to satisfy."""
        raise NotImplementedError
