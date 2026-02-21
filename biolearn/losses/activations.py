import jax

from .base import make_loss


def _activation_loss_factory(group_loss):
    def ic_loss(specification, ts, *, domain=None, n_points=128, key=None, **kwargs):
        return make_loss(
            group_loss=group_loss,
            specification=specification,
            ts=ts,
            domain=domain,
            n_points=n_points,
            key=key,
            **kwargs,
        )

    return ic_loss


activations = {
    "sigmoid": lambda r: jax.nn.sigmoid(-r).mean(),
    "swish": lambda r: jax.nn.swish(-r).mean(),
    "silu": lambda r: jax.nn.silu(-r).mean(),
    "softmax": lambda r: jax.nn.softmax(-r).mean(),
    "tanh": lambda r: jax.nn.tanh(-r).mean(),
    "logsumexp": lambda r: jax.nn.logsumexp(-r).mean(),
    "leaky_relu": lambda r: jax.nn.leaky_relu(-r).mean(),
    "elu": lambda r: jax.nn.elu(-5.0 * r).mean(),
}

_g = globals()
for _name, _fn in activations.items():
    _g[f"make_{_name}_loss"] = _activation_loss_factory(_fn)
