import equinox as eqx


class Model(eqx.Module):
    """A base model class to abstract simulation behavior."""

    def __init__(self):
        pass

    def _step(self):
        raise NotImplementedError

    def _simulate(self):
        raise NotImplementedError
