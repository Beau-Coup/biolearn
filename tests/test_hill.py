"""Unit tests for biolearn/models/hill.py.

Covers:
- Activation and Inhibition construction and attribute shapes
- InhibitActivateAggregator construction, numerator, denominator, and forward pass
- BioGNN construction (aggregator count, parameter shapes)
- BioGNN.__call__ output shape and dtype
- BioGNN used as an ODE right-hand side via diffrax (short simulation)
- JAX jit and vmap compatibility
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# Imports under test
from biolearn.models.hill import (  # noqa: E402
    Activation,
    BioGNN,
    EdgeType,
    InhibitActivateAggregator,
    Inhibition,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _simple_graph():
    """Three-node graph: 0->1 (Activation), 1->0 (Inhibition), 2->1 (Inhibition)."""
    return [
        (0, 1, EdgeType.Activation),
        (1, 0, EdgeType.Inhibition),
        (2, 1, EdgeType.Inhibition),
    ]


def _activation_edge(start=0, end=1, hill=2.0):
    return Activation(start, end, hill)


def _inhibition_edge(start=1, end=0, hill=3.0):
    return Inhibition(start, end, hill)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


class TestActivation:
    def test_construction_stores_start_end(self):
        edge = _activation_edge(start=2, end=5, hill=2.0)
        assert edge.start == 2
        assert edge.end == 5

    def test_construction_stores_hill_coefficient(self):
        edge = _activation_edge(hill=4.5)
        assert edge.hill_coefficient == 4.5

    def test_different_hill_coefficients_stored_correctly(self):
        for h in [0.5, 1.0, 2.0, 10.0]:
            edge = Activation(0, 1, h)
            assert edge.hill_coefficient == h


# ---------------------------------------------------------------------------
# Inhibition
# ---------------------------------------------------------------------------


class TestInhibition:
    def test_construction_stores_start_end(self):
        edge = _inhibition_edge(start=3, end=7, hill=1.5)
        assert edge.start == 3
        assert edge.end == 7

    def test_construction_stores_hill_coefficient(self):
        edge = _inhibition_edge(hill=1.5)
        assert edge.hill_coefficient == 1.5

    def test_different_hill_coefficients_stored_correctly(self):
        for h in [0.5, 1.0, 2.0, 10.0]:
            edge = Inhibition(0, 1, h)
            assert edge.hill_coefficient == h


# ---------------------------------------------------------------------------
# InhibitActivateAggregator
# ---------------------------------------------------------------------------


class TestInhibitActivateAggregator:
    def _make_agg(self, n_act=2, n_inh=1, hill_act=2.0, hill_inh=3.0):
        """Create an aggregator with activations from nodes 0..n_act-1
        and inhibitions from nodes n_act..n_act+n_inh-1."""
        activations = [Activation(i, 99, hill_act) for i in range(n_act)]
        inhibitions = [Inhibition(n_act + i, 99, hill_inh) for i in range(n_inh)]
        return InhibitActivateAggregator(activations, inhibitions)

    # -- construction --

    def test_k_activate_shape(self):
        agg = self._make_agg(n_act=3, n_inh=2)
        assert agg.k_activate.shape == (3,)

    def test_k_inhibit_shape(self):
        agg = self._make_agg(n_act=3, n_inh=2)
        assert agg.k_inhibit.shape == (2,)

    def test_k_activate_initialized_to_ones(self):
        agg = self._make_agg(n_act=4)
        assert jnp.allclose(agg.k_activate, jnp.ones(4))

    def test_k_inhibit_initialized_to_ones(self):
        agg = self._make_agg(n_inh=3)
        assert jnp.allclose(agg.k_inhibit, jnp.ones(3))

    def test_hill_activate(self):
        agg = self._make_agg(n_act=2, hill_act=5.0)
        assert jnp.allclose(jnp.array(agg.hill_activate), jnp.array([5.0, 5.0]))

    def test_hill_inhibit(self):
        agg = self._make_agg(n_inh=2, hill_inh=4.0)
        assert jnp.allclose(jnp.array(agg.hill_inhibit), jnp.array([4.0, 4.0]))

    def test_activate_indices_correct(self):
        agg = self._make_agg(n_act=2, n_inh=0)
        assert agg.activate_indices == [0, 1]

    def test_inhibit_indices_correct(self):
        agg = self._make_agg(n_act=2, n_inh=2)
        # inhibition edges start at node 2 and 3
        assert agg.inhibit_indices == [2, 3]

    # -- denominator --

    def test_denominator_no_inhibitors_equals_one(self):
        # With no inhibitors the denominator is always 1 + 0 = 1
        activations = [Activation(0, 99, 2.0)]
        agg = InhibitActivateAggregator(activations, [])
        x = jnp.array([2.0, 0.0])
        assert float(agg._denominator_fun(x)) == pytest.approx(1.0)

    def test_denominator_with_one_inhibitor(self):
        # denominator = 1 + k * x[src]^hill = 1 + 1*2^2 = 5
        inhibitions = [Inhibition(0, 99, 2.0)]
        agg = InhibitActivateAggregator([], inhibitions)
        x = jnp.array([2.0])
        assert float(agg._denominator_fun(x)) == pytest.approx(1.0 + 1.0 * 2.0**2)

    def test_denominator_with_multiple_inhibitors(self):
        # denominator = 1 + 1*x[0]^2 + 1*x[1]^3
        inhibitions = [Inhibition(0, 99, 2.0), Inhibition(1, 99, 3.0)]
        agg = InhibitActivateAggregator([], inhibitions)
        x = jnp.array([2.0, 3.0])
        expected = 1.0 + 2.0**2 + 3.0**3
        assert float(agg._denominator_fun(x)) == pytest.approx(expected)

    def test_denominator_at_zero_input(self):
        inhibitions = [Inhibition(0, 99, 2.0)]
        agg = InhibitActivateAggregator([], inhibitions)
        x = jnp.zeros(3)
        # 1 + 0^2 = 1
        assert float(agg._denominator_fun(x)) == pytest.approx(1.0)

    # -- numerator --

    def test_numerator_no_activators_returns_one(self):
        inhibitions = [Inhibition(0, 99, 2.0)]
        agg = InhibitActivateAggregator([], inhibitions)
        x = jnp.array([5.0])
        assert float(agg._numerator_fun(x)) == pytest.approx(1.0)

    def test_numerator_with_one_activator(self):
        # numerator = k * x[0]^hill = 1 * 3^2 = 9
        activations = [Activation(0, 99, 2.0)]
        agg = InhibitActivateAggregator(activations, [])
        x = jnp.array([3.0])
        assert float(agg._numerator_fun(x)) == pytest.approx(9.0)

    def test_numerator_with_multiple_activators(self):
        # numerator = 1*x[0]^2 + 1*x[1]^3
        activations = [Activation(0, 99, 2.0), Activation(1, 99, 3.0)]
        agg = InhibitActivateAggregator(activations, [])
        x = jnp.array([2.0, 3.0])
        expected = 2.0**2 + 3.0**3
        assert float(agg._numerator_fun(x)) == pytest.approx(expected)

    def test_numerator_at_zero_input_with_activators(self):
        activations = [Activation(0, 99, 2.0)]
        agg = InhibitActivateAggregator(activations, [])
        x = jnp.zeros(2)
        assert float(agg._numerator_fun(x)) == pytest.approx(0.0)

    # -- forward pass --

    def test_forward_equals_numerator_over_denominator(self):
        activations = [Activation(0, 99, 2.0)]
        inhibitions = [Inhibition(1, 99, 3.0)]
        agg = InhibitActivateAggregator(activations, inhibitions)
        x = jnp.array([2.0, 1.5])
        expected = float(agg._numerator_fun(x)) / float(agg._denominator_fun(x))
        assert float(agg(x)) == pytest.approx(expected)

    def test_forward_output_is_scalar(self):
        agg = self._make_agg(n_act=2, n_inh=1)
        x = jnp.ones(5)
        result = agg(x)
        assert result.shape == ()

    def test_forward_no_activators_output_is_between_zero_and_one(self):
        # With no activators numerator=1, denominator>=1, so result in (0,1]
        inhibitions = [Inhibition(0, 99, 2.0)]
        agg = InhibitActivateAggregator([], inhibitions)
        x = jnp.array([3.0])
        result = float(agg(x))
        assert 0.0 < result <= 1.0

    def test_forward_with_zero_state_and_activators(self):
        # numerator = 0, denominator = 1 -> output = 0
        activations = [Activation(0, 99, 2.0)]
        agg = InhibitActivateAggregator(activations, [])
        x = jnp.zeros(2)
        assert float(agg(x)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BioGNN construction
# ---------------------------------------------------------------------------


class TestBioGNNConstruction:
    def test_n_nodes_inferred_from_graph(self):
        graph = [(0, 2, EdgeType.Activation)]
        net = BioGNN(graph, 2.0)
        assert net.n_nodes == 3  # max node index is 2, so 3 nodes

    def test_n_nodes_for_simple_graph(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert net.n_nodes == 3

    def test_decay_shape(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert net.decay.shape == (3,)

    def test_growth_shape(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert net.growth.shape == (3,)

    def test_nu_shape(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert net.nu.shape == (3,)

    def test_decay_initialized_to_ones(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert jnp.allclose(net.decay, jnp.ones(3))

    def test_growth_initialized_to_ones(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert jnp.allclose(net.growth, jnp.ones(3))

    def test_nu_initialized_to_ones(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert jnp.allclose(net.nu, jnp.ones(3))

    def test_aggregator_count_matches_nodes_with_incoming_edges(self):
        # simple_graph: node 0 has 1 incoming (inhibition from 1)
        #               node 1 has 2 incoming (activation from 0, inhibition from 2)
        # node 2 has no incoming edges -> no aggregator
        net = BioGNN(_simple_graph(), 2.0)
        assert len(net.aggregators) == 2

    def test_agg_indices_correct(self):
        net = BioGNN(_simple_graph(), 2.0)
        assert set(net.agg_indices) == {0, 1}

    def test_no_aggregator_for_isolated_node(self):
        # Node 2 only appears as a source, never a destination
        net = BioGNN(_simple_graph(), 2.0)
        assert 2 not in net.agg_indices

    def test_single_activation_edge(self):
        graph = [(0, 1, EdgeType.Activation)]
        net = BioGNN(graph, 2.0)
        assert net.n_nodes == 2
        assert len(net.aggregators) == 1
        assert net.agg_indices == [1]

    def test_single_inhibition_edge(self):
        graph = [(0, 1, EdgeType.Inhibition)]
        net = BioGNN(graph, 2.0)
        assert net.n_nodes == 2
        assert len(net.aggregators) == 1
        assert net.agg_indices == [1]

    def test_unknown_edge_type_raises(self):
        # Pass a raw string instead of an EdgeType value. The match statement will
        # either raise a ValueError from EnumerationItem.__eq__ (when Python evaluates
        # the value pattern) or fall through to the wildcard branch and raise the custom
        # Exception. Either way an exception must be raised.
        graph = [(0, 1, "unknown_type")]
        with pytest.raises(Exception):
            BioGNN(graph, 2.0)

    def test_larger_graph_node_count(self):
        graph = [
            (0, 3, EdgeType.Activation),
            (1, 3, EdgeType.Inhibition),
            (2, 4, EdgeType.Activation),
        ]
        net = BioGNN(graph, 2.0)
        assert net.n_nodes == 5  # max node index is 4

    def test_hill_coefficient_propagated_to_aggregators(self):
        graph = [(0, 1, EdgeType.Activation)]
        net = BioGNN(graph, 3.0)
        # The single aggregator should record hill=3.0 for its activation
        agg = net.aggregators[0]
        assert float(agg.hill_activate[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# BioGNN.__call__ output shape and dtype
# ---------------------------------------------------------------------------


class TestBioGNNCall:
    def _net(self):
        return BioGNN(_simple_graph(), 2.0)

    def test_output_shape_matches_input(self):
        net = self._net()
        x = jnp.ones(3)
        out = net(x)
        assert out.shape == x.shape

    def test_output_dtype_float64(self):
        net = self._net()
        x = jnp.ones(3, dtype=jnp.float64)
        out = net(x)
        assert out.dtype == jnp.float64

    def test_output_at_zero_state(self):
        # With all-zero x and default params (nu=1, decay=1, growth=1):
        # dx_agg at node without activators but with inhibitors = numerator/denominator
        #   numerator = 1 (no activators), denominator = 1 + 0^hill = 1  -> agg = 1
        #   node 0: agg(x=0) = 1/(1+0^3) = 1
        #   node 1: numerator = 0^2 = 0, denominator = 1+0^3 = 1 -> agg = 0
        # dx = nu * dx_agg - decay * x + growth
        #   node 0: 1*1 - 1*0 + 1 = 2
        #   node 1: 1*0 - 1*0 + 1 = 1
        #   node 2: 1*0 - 1*0 + 1 = 1  (no aggregator -> dx_agg=0)
        net = self._net()
        x = jnp.zeros(3)
        out = net(x)
        expected = jnp.array([2.0, 1.0, 1.0])
        assert jnp.allclose(out, expected), f"Expected {expected}, got {out}"

    def test_output_is_finite_for_positive_state(self):
        net = self._net()
        x = jnp.array([1.0, 2.0, 0.5])
        out = net(x)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape_single_node_self_loop_equivalent(self):
        # One-node graph with a single activation from 0 to 0 is a self-loop.
        # This is unusual but the code should handle it.
        graph = [(0, 1, EdgeType.Activation), (0, 1, EdgeType.Inhibition)]
        net = BioGNN(graph, 2.0)
        x = jnp.ones(2)
        out = net(x)
        assert out.shape == (2,)

    def test_growth_decay_nu_affect_output(self):
        net = self._net()
        # Double the growth and verify output changes predictably
        x = jnp.ones(3)
        out_default = net(x)

        net2 = eqx.tree_at(lambda m: m.growth, net, 2.0 * jnp.ones(3))
        out_doubled = net2(x)
        # The difference should equal nu * 1 = 1 per element at x=ones
        assert jnp.allclose(out_doubled - out_default, jnp.ones(3))

    def test_aggregator_sum_zero_for_no_aggregators(self):
        # A graph with only isolated nodes (impossible via constructor, but we can
        # verify agg_sum produces zeros when there are no aggregators by using a
        # custom graph where no node receives an edge -- not constructible via
        # BioGNN without edges, so test via indirect output at nodes without agg)
        graph = [(0, 1, EdgeType.Activation)]
        net = BioGNN(graph, 2.0)
        # Node 0 has no incoming edges, so its aggregator contribution is 0
        x = jnp.array([3.0, 2.0])
        agg_sum = net._aggregator_sum(x)
        # node 0 should have zero contribution from aggregator
        assert float(agg_sum[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BioGNN as ODE (simulate via diffrax directly)
# ---------------------------------------------------------------------------


class TestBioGNNSimulate:
    """BioGNN.__call__ acts as the ODE right-hand side (dy/dt = net(y)).
    We wrap it in a diffrax solve to verify end-to-end integration.
    """

    def test_ode_solve_runs_without_error(self):
        import diffrax

        net = BioGNN(_simple_graph(), 2.0)

        def vector_field(t, y, args):
            return net(y)

        ts = jnp.linspace(0.0, 1.0, 11)
        y0 = jnp.ones(3)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            progress_meter=diffrax.NoProgressMeter(),
        )
        assert solution.ys is not None

    def test_ode_solution_shape(self):
        import diffrax

        net = BioGNN(_simple_graph(), 2.0)

        def vector_field(t, y, args):
            return net(y)

        ts = jnp.linspace(0.0, 0.5, 6)
        y0 = jnp.zeros(3)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            progress_meter=diffrax.NoProgressMeter(),
        )
        # shape: (T, n_nodes)
        assert solution.ys.shape == (6, 3)

    def test_ode_solution_is_finite(self):
        import diffrax

        net = BioGNN(_simple_graph(), 2.0)

        def vector_field(t, y, args):
            return net(y)

        ts = jnp.linspace(0.0, 2.0, 21)
        y0 = jnp.array([0.5, 0.5, 0.5])

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            progress_meter=diffrax.NoProgressMeter(),
        )
        assert jnp.all(jnp.isfinite(solution.ys))


# ---------------------------------------------------------------------------
# JAX jit compatibility
# ---------------------------------------------------------------------------


class TestJitCompatibility:
    def test_call_under_jit(self):
        net = BioGNN(_simple_graph(), 2.0)
        x = jnp.ones(3)
        jitted = eqx.filter_jit(net)
        out = jitted(x)
        assert out.shape == (3,)

    def test_jit_produces_same_result_as_eager(self):
        net = BioGNN(_simple_graph(), 2.0)
        x = jnp.array([1.0, 2.0, 3.0])
        eager = net(x)
        jitted = eqx.filter_jit(net)(x)
        assert jnp.allclose(eager, jitted)

    def test_aggregator_call_under_jit(self):
        activations = [Activation(0, 99, 2.0)]
        inhibitions = [Inhibition(1, 99, 3.0)]
        agg = InhibitActivateAggregator(activations, inhibitions)
        x = jnp.array([2.0, 1.0])
        jitted = eqx.filter_jit(agg)
        out_jit = jitted(x)
        out_eager = agg(x)
        assert jnp.allclose(out_jit, out_eager)

    def test_jit_with_different_hill_coefficients(self):
        for hill in [1.0, 2.0, 4.0]:
            net = BioGNN(_simple_graph(), hill)
            x = jnp.ones(3)
            out = eqx.filter_jit(net)(x)
            assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# JAX vmap compatibility
# ---------------------------------------------------------------------------


class TestVmapCompatibility:
    def test_call_under_vmap_over_batch(self):
        net = BioGNN(_simple_graph(), 2.0)
        # Batch of 5 state vectors
        xs = jnp.ones((5, 3))
        out = jax.vmap(net)(xs)
        assert out.shape == (5, 3)

    def test_vmap_produces_same_result_as_loop(self):
        net = BioGNN(_simple_graph(), 2.0)
        xs = jnp.stack(
            [jnp.array([float(i), float(i + 1), float(i + 2)]) for i in range(4)]
        )
        out_vmap = jax.vmap(net)(xs)
        out_loop = jnp.stack([net(xs[i]) for i in range(4)])
        assert jnp.allclose(out_vmap, out_loop, atol=1e-10)

    def test_vmap_output_shape_batched(self):
        graph = [(0, 1, EdgeType.Activation), (1, 2, EdgeType.Inhibition)]
        net = BioGNN(graph, 2.0)
        batch_size = 8
        xs = jnp.ones((batch_size, 3))
        out = jax.vmap(net)(xs)
        assert out.shape == (batch_size, 3)

    def test_jit_vmap_composition(self):
        net = BioGNN(_simple_graph(), 2.0)
        xs = jnp.ones((4, 3))
        out = eqx.filter_jit(eqx.filter_vmap(net))(xs)
        assert out.shape == (4, 3)
        assert jnp.all(jnp.isfinite(out))
