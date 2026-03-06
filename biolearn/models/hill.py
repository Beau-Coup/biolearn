from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp


class EdgeType(eqx.Enumeration):
    Activation = "Activation"
    Inhibition = "Inhibition"


class Edge(eqx.Module):
    """Abstract Edge class module for the Hill Model."""

    start: int
    end: int

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
    hill_coefficient: float

    def __init__(self, start, end, hill_coefficient):
        super().__init__(start, end)
        self.k = jnp.array(1.0)
        self.hill_coefficient = hill_coefficient


class Inhibition(Edge):
    hill_coefficient: float

    def __init__(self, start, end, hill_coefficient):
        super().__init__(start, end)
        self.k = jnp.array(1.0)
        self.hill_coefficient = hill_coefficient


class InhibitActivateAggregator(eqx.Module):
    """A class to pre-aggregate Activation/Inhibition contributions.
    Each node has one of these classes.
    """

    hill_inhibit: jax.Array = eqx.field(static=True)
    hill_activate: jax.Array = eqx.field(static=True)
    inhibit_indices: List[int]
    activate_indices: List[int]

    k_inhibit: jax.Array
    k_activate: jax.Array

    def __init__(self, activations: List[Activation], inhibitions: List[Inhibition]):
        self.k_inhibit = jnp.ones(len(inhibitions))
        self.k_activate = jnp.ones(len(activations))

        self.hill_inhibit = jnp.array([edge.hill_coefficient for edge in inhibitions])
        self.hill_activate = jnp.array([edge.hill_coefficient for edge in activations])
        self.inhibit_indices = [edge.start for edge in inhibitions]
        self.activate_indices = [edge.start for edge in activations]

    def _denominator_fun(self, x: jax.Array) -> jax.Array:
        return 1.0 + jnp.sum(
            self.k_inhibit * (x[jnp.array(self.inhibit_indices)] ** self.hill_inhibit)
        )

    def _numerator_fun(self, x: jax.Array) -> jax.Array:
        if not self.activate_indices:
            return jnp.array(1.0)

        return jnp.sum(
            self.k_activate
            * (x[jnp.array(self.activate_indices)] ** self.hill_activate)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._numerator_fun(x) / self._denominator_fun(x)


class BioGNN(eqx.Module):
    """Implementation of the bio-inspired graph neural network."""

    aggregators: List[eqx.Module]
    agg_indices: List[int]
    n_nodes: int
    decay: jax.Array
    growth: jax.Array
    nu: jax.Array

    def __init__(self, graph, hill_coefficient):
        # Go through the edges and init the message passing function

        n_nodes = max([max(a, b) for a, b, _ in graph]) + 1
        self.n_nodes = n_nodes

        recipient_activations = [[] for _ in range(n_nodes)]
        recipient_inhibitions = [[] for _ in range(n_nodes)]

        for edge in graph:
            from_node, to_node, edge_type = edge

            match edge_type:
                case EdgeType.Activation:
                    edge_func = Activation(from_node, to_node, hill_coefficient)
                    recipient_activations[to_node].append(edge_func)
                case EdgeType.Inhibition:
                    edge_func = Inhibition(from_node, to_node, hill_coefficient)
                    recipient_inhibitions[to_node].append(edge_func)
                case e:
                    raise Exception(
                        f"Edge type {e} not known.\n Available edges are {EdgeType}"
                    )

        # For each node that has either ingoing inhibition or activation edges, make an aggregator
        self.aggregators = []
        self.agg_indices = []
        for node in range(n_nodes):
            if recipient_activations[node] or recipient_inhibitions[node]:
                self.agg_indices.append(node)
                self.aggregators.append(
                    InhibitActivateAggregator(
                        recipient_activations[node], recipient_inhibitions[node]
                    )
                )

        self.decay = jnp.ones(n_nodes)
        self.growth = jnp.ones_like(self.decay)
        self.nu = jnp.ones_like(self.decay)

    def _aggregator_sum(self, x: jax.Array) -> jax.Array:
        out = jnp.zeros_like(x)
        for agg, dest in zip(self.aggregators, self.agg_indices):
            out = out.at[dest].set(agg(x))  # pyright: ignore

        return out

    def __call__(self, x: jax.Array) -> jax.Array:
        dx = jnp.zeros_like(x)

        dx_agg = self._aggregator_sum(x)
        dx = dx + dx_agg

        return self.nu * dx - self.decay * x + self.growth
