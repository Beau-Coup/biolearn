import jax

activations = {
    "relu": lambda r: jax.nn.relu(-r).mean(),
    "sigmoid": lambda r: jax.nn.sigmoid(-r).mean(),
    "swish": lambda r: jax.nn.swish(-r).mean(),
    "silu": lambda r: jax.nn.silu(-r).mean(),
    "softmax": lambda r: jax.nn.softmax(-r).sum(),
    "tanh": lambda r: jax.nn.tanh(-r).mean(),
    "logsumexp": lambda r: jax.nn.logsumexp(-r).sum(),
    "leaky_relu": lambda r: jax.nn.leaky_relu(-r).mean(),
    "elu": lambda r: jax.nn.elu(-5.0 * r).mean(),
}
