import equinox as eqx
import jax
import jax.numpy as jnp


class NonNegativeLinear(eqx.Module):
    """
    Non-negative linear activation function.
    """

    _log_cnt: float = eqx.field(static=True)
    _weight: jax.Array
    _bias: jax.Array

    in_features: int = eqx.field(static=True, converter=int)
    out_features: int = eqx.field(static=True, converter=int)

    def __init__(self, in_features: int, out_features: int, *, key):
        wkey, bkey = jax.random.split(key, 2)

        wshape = (out_features, in_features)
        bshape = (out_features,)

        lim = 1 / jnp.sqrt(in_features)
        _weight = jax.random.uniform(wkey, wshape, minval=0.1, maxval=lim)
        _bias = jax.random.uniform(bkey, bshape, minval=0.001, maxval=0.01)

        self._log_cnt = 1e-7
        self._weight = jnp.log(_weight + self._log_cnt)
        self._bias = jnp.log(_bias + self._log_cnt)

        self.in_features = in_features
        self.out_features = out_features

    @property
    def weight(self):
        return jnp.exp(self._weight) - self._log_cnt

    @property
    def bias(self):
        return jax.nn.relu(jnp.exp(self._bias) - self._log_cnt)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.weight @ x + self.bias
        return x
