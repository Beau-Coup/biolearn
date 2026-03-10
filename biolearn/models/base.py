from abc import abstractmethod

import equinox as eqx


class BioModel(eqx.Module):
    """A base model class to abstract simulation behavior."""

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def _simulate(self):
        raise NotImplementedError
