import equinox as eqx
import jax
import jax.numpy as jnp


class EdgeType(eqx.Enumeration):
    Activation = 0
    Inhibition = 1


class Edge(eqx.Module):
    """Abstract Edge class module for the Hill Model."""

    start: jax.Array
    end: jax.Array

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def contribution(self, x1: jax.Array, x2: jax.Array) -> jax.Array:  # pyright: ignore[reportUnusedParameter]
        """Compute the edge contribution for each of its endpoints.

        The edge is oriented from node x1 to node x2

        Parameters:
        ----------
            x1: (), the value at the origin node
            x2: (), the value at the destination node

        Returns:
        -------
            jax.Array (2,) containing the contribution of the edge to x1 and x2, respectively.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.contribution(x[self.start], x[self.end])


class Activation(Edge):
    hill_coefficient: jax.Array
    k: jax.Array

    def __init__(self, start, end, hill_coefficient):
        super().__init__(start, end)
        self.k = jnp.array(1.0)
        self.hill_coefficient = hill_coefficient

    def contribution(self, x1: jax.Array, x2: jax.Array) -> jax.Array:  # pyright: ignore [reportUnusedParameter]
        f = self.k * (x1**self.hill_coefficient)
        return jnp.array([0.0, f])


class Inhibition(Edge):
    hill_coefficient: jax.Array
    k: jax.Array

    def __init__(self, start, end, hill_coefficient):
        super().__init__(start, end)
        self.k = jnp.array(1.0)
        self.hill_coefficient = hill_coefficient

    def contribution(self, x1: jax.Array, x2: jax.Array) -> jax.Array:  # pyright: ignore [reportUnusedParameter]
        f = self.k * (x1**self.hill_coefficient)
        return jnp.array([0.0, f])


class BioGNN(eqx.Module):
    """Implementation of the bio-inspired graph neural network."""

    def __init__(self, g):
        # Go through the edges and init the message passing function

        pass
