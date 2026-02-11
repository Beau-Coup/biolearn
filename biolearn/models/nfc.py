"""Copies the logic from the paper

STL-based Optimization of Biomolecular Neural Networks for Regression and Control
by Eric Palanques-Tost, Hanna Krasowski, Murat Arcak, Ron Weiss, Calin Belta
"""

from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Type, final

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt

from .activations import NonNegativeLinear


class BioSyst(eqx.Module):
    """
    Base class to create biological systems inheriting the diffeqsolve function.
    """

    @abstractmethod
    def diffrax_step(self, t: jt.ScalarLike, y: jax.Array, args: Tuple) -> jax.Array:
        """
        ODE step for the model.
        Args:
            t: time
            y: state vector
            args: additional arguments
        Returns:
            dydt: the derivative of the state vector y
        """
        raise NotImplementedError()

    def _simulate(
        self,
        y0: jt.Float[jt.Array, "..."],
        ts: Optional[jax.Array],
        to_ss: bool = False,
        args: Tuple[Any, ...] = (),
        stiff: bool = True,
        throw: bool = True,
        max_steps: int = int(1e6),
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_stepsize: Optional[int] = None,
        progress_bar: bool = True,
    ):
        stepsize_controller = diffrax.PIDController(
            pcoeff=0.3 if stiff else 0.0,
            icoeff=0.3 if stiff else 1.0,
            dcoeff=0,
            rtol=rtol,
            atol=atol,
            dtmax=max_stepsize,
            dtmin=1e-10,
        )

        if to_ss:
            saveat = diffrax.SaveAt(t1=True)
            event = diffrax.Event(
                diffrax.steady_state_event(rtol=2 * rtol, atol=2 * atol)
            )
            t0, t1, dt0 = 0.0, jnp.inf, None
        else:
            saveat = diffrax.SaveAt(ts=ts)
            event = None
            assert ts is not None
            t0, t1, dt0 = ts[0], ts[-1], 0.001

        progress_bar = (
            diffrax.TqdmProgressMeter() if progress_bar else diffrax.NoProgressMeter()
        )
        solver = diffrax.Kvaerno5() if stiff else diffrax.Tsit5()

        term = diffrax.ODETerm(self.diffrax_step)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=throw,
            args=args,
            progress_meter=progress_bar,
            event=event,
        )

        y_pred = solution.ys

        return y_pred, solution


class NFCNodeBase(eqx.Module):
    """
    Base class for an NFC node.
    """

    activation: eqx.Module

    n_species: int = eqx.field(converter=int, static=True)
    n_inputs: int = eqx.field(converter=int, static=True)
    is_first: bool = eqx.field(static=True, converter=bool)
    is_last: bool = eqx.field(static=True, converter=bool)

    def __init__(
        self,
        n_species: int,
        n_inputs: int,
        activation: Optional[eqx.Module] = None,
        is_first: bool = True,
        is_last: bool = False,
        *,
        key: jt.Array,
    ):
        """
        Initializes an NFC node.

        Args:
            n_species: number of species in the node
            n_inputs: number of external inputs
            activation: activation function of the node. Defaults to Linear.
            key: Random key to use for initialization
        """
        self.n_species = n_species
        self.n_inputs = n_inputs

        if activation is None:
            _, key = jax.random.split(key)
            in_features = n_inputs
            out_features = self.n_species
            activation = NonNegativeLinear(in_features, out_features, key=key)
            # activation = jax.tree_util.tree_map(jnp.abs, activation)
            # activation = self._init_weights(activation, key=key)
        self.activation = activation
        self.is_first = is_first
        self.is_last = is_last

    @staticmethod
    def _init_weights(activation, *, key: jt.PRNGKeyArray):
        sk1, sk2 = jax.random.split(key, 2)

        new_weight = jax.random.uniform(
            sk1, shape=activation.weight.shape, minval=0.0, maxval=0.1
        )
        new_bias = jax.random.uniform(
            sk2, shape=activation.bias.shape, minval=0.0, maxval=0.1
        )
        new_act = eqx.tree_at(lambda x: x.weight, activation, new_weight)
        new_act = eqx.tree_at(lambda x: x.bias, new_act, new_bias)
        return new_act

    @property
    def size(self) -> int:
        return self.n_species

    @property
    def shape(self) -> Tuple[int,]:
        return (self.n_species,)

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        """
        Calculates the ODE step for the node.
        Args:
            x: external input. Should have shape (n_inputs, )
            z: current state of the node. Should have shape (n_species, )

        Returns:
            dz: the derivative of the state of the node. Should have shape (n_species, )

        """
        return self.ode_step(x, z)

    @final
    def ode_step(self, x: jax.Array, z: jax.Array) -> jax.Array:
        """
        Calculates the ODE step for the node.
        Args:
            x: external input. Should have shape (n_inputs, )
            z: current state of the node. Should have shape (n_species, )

        Returns:
            dz: the derivative of the state of the node. Should have shape (n_species, )

        """
        if type(self).ode_step is not NFCNodeBase.ode_step:
            raise NotImplementedError(
                "The method `ode_step` should not be overwritten."
                "Overwrite the self._ode_step method instead."
            )
        if z.shape[-1] != self.n_species:
            raise ValueError(
                f"Invalid dimensions for state array."
                f"Expected {self.n_species}, got {z.shape}"
            )
        if x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Invalid dimensions for input array."
                f"Expected {self.n_inputs}, got {x.shape}"
            )
        dz = self._ode_step(x, z)

        if dz.shape[-1] != self.n_species:
            raise ValueError(
                f"Invalid dimensions for output array of the ode step dz/dt."
                f" Expected {self.n_species}, got {dz.shape}"
            )
        return dz

    @final
    def find_ss(self, x: jax.Array) -> jt.ScalarLike:
        """
        Finds the steady-state of the output species of the node given the input.
        Args:
            x: external input. Should have shape (n_inputs, )

        Returns:
            z1: the estimated steady-state concentration
                of the output species of the node.
        """
        if type(self).find_ss is not NFCNodeBase.find_ss:
            raise NotImplementedError(
                "The method `find_ss` should not be overwritten."
                "Overwrite the self._find_ss method instead."
            )
        if x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Invalid dimensions for input array."
                f"Expected {self.n_inputs}, got {x.shape}"
            )
        y = self._find_ss(x)

        if y.size != 1:
            raise ValueError(
                f"Invalid dimensions for output array of the steady-state estimation."
                f" Expected 1, got {y.size}"
            )

        return y

    @abstractmethod
    def _ode_step(self, x: jax.Array, z: jax.Array) -> jax.Array:
        raise NotImplementedError()

    def get_weights(self):
        return list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(self)))

    def _find_ss(self, x: jax.Array) -> jt.ScalarLike:
        raise NotImplementedError()


class MoormanPerceptron(NFCNodeBase):
    """
    Implements a biomolecular perceptron as described in Moorman et al. (2019).

    A. Moorman, C. C. Samaniego, C. Maley and R. Weiss,
    "A Dynamical Biomolecular Neural Network,"
    2019 IEEE 58th Conference on Decision and Control (CDC),
    Nice, France, 2019, pp. 1797-1802,
    doi: 10.1109/CDC40024.2019.9030122.
    """

    gamma: float = eqx.field(static=True, converter=float)
    beta: float = eqx.field(static=True, converter=float)
    k: float = eqx.field(static=True, converter=float)

    def __init__(
        self,
        n_inputs: int,
        activation: Optional[eqx.Module] = None,
        gamma: jt.ScalarLike = 1000.0,
        beta: jt.ScalarLike = 1.0,
        k: jt.ScalarLike = 0.5,
        is_first: bool = True,
        is_last: bool = False,
        *,
        key: jt.PRNGKeyArray,
    ):
        super().__init__(
            n_species=2,
            n_inputs=n_inputs,
            activation=activation,
            is_first=is_first,
            is_last=is_last,
            key=key,
        )

        self.gamma = gamma
        self.beta = beta
        self.k = k

    def phi(self, x: jax.Array) -> jax.Array:
        return x / (self.k + x) if not self.is_first else x

    def _ode_step(self, x: jax.Array, z: jax.Array) -> jax.Array:
        seq = self.gamma * jnp.prod(z)
        dz = self.activation(self.phi(x)) - seq - self.beta * z
        return dz

    def _find_ss(self, x: jax.Array) -> jt.ScalarLike:
        u, v = self.activation(self.phi(x))

        b = self.beta / self.gamma + (v - u) / self.beta
        c = -u / self.gamma

        z1 = 1 / 2 * (-b + jnp.sqrt(b**2 - 4 * c))
        return z1


class NFCLayer(eqx.Module):
    """
    Base class for an NFC layer
    """

    nodes: NFCNodeBase

    n_species: int = eqx.field(converter=int, static=True)
    n_inputs: int = eqx.field(converter=int, static=True)
    n_nodes: int = eqx.field(converter=int, static=True)

    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        node_class: Type[NFCNodeBase],
        is_first: bool = True,
        *,
        key: jt.PRNGKeyArray,
        **node_kwargs,
    ):
        """
        Initializes an NFC layer.
        Args:
            n_inputs: number of external inputs.
            n_nodes: number of nodes in the layer.
            node_class: Class of the nodes in the layer.
            key: Random key to use for initialization.
            **node_kwargs: kwarg arguments for the nodes
        """
        subkeys = jax.random.split(key, n_nodes)

        def _ini_node(_k):
            return node_class(
                n_inputs=n_inputs, is_first=is_first, key=_k, **node_kwargs
            )

        self.nodes = eqx.filter_vmap(_ini_node)(subkeys)
        self.n_species = self.nodes.n_species * n_nodes
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes

    def __len__(self) -> int:
        return self.n_nodes

    @property
    def size(self) -> int:
        return self.n_nodes * self.nodes.n_species

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_nodes, self.nodes.n_species

    def __call__(
        self,
        x: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        return self.ode_step(x, z)

    def ode_step(
        self,
        x: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        """
        Deterministic update step of a layer of the NFC.
        Calculates dy/dt of the layer for the given input and state

        :param x: value of the inputs (n_inputs, )
        :param z: Current state of the species. Shape: (n_nodes, node_size)
        :return: dy for the layer. Shape: (n_nodes, node_size)
        """
        if x.shape != (self.n_inputs,):
            raise ValueError(
                f"Invalid dimensions for input array."
                f"Expected {self.n_inputs}, got {x.shape}"
            )
        if z.shape != (self.n_nodes, self.nodes.n_species):
            raise ValueError(
                f"Invalid dimensions for state array."
                f"Expected {(self.n_nodes, self.nodes.n_species)}, got {z.shape}"
            )

        def _call(_node, _x, _z):
            return _node.ode_step(_x, _z)

        return eqx.filter_vmap(_call, in_axes=(eqx.if_array(0), None, eqx.if_array(0)))(
            self.nodes, x, z
        )

    def ss_estimation(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Estimation of the SS of the layer given static inputs

        :param x: value of the static inputs (n_inputs, )
        :return: steady-state of the species (n_nodes, n_node_species)
        """
        if x.shape != (self.n_inputs,):
            raise ValueError(
                f"Invalid dimensions for input array."
                f"Expected {self.n_inputs}, got {x.shape}"
            )

        def _call(_node, _x):
            return _node.find_ss(_x)

        y = eqx.filter_vmap(_call, in_axes=(eqx.if_array(0), None))(self.nodes, x)

        if y.shape != (self.n_nodes,):
            raise ValueError(
                f"Invalid dimensions for output array of the steady-state estimation."
                f"Expected {(self.n_nodes,)}, got {y.shape}"
            )
        return y

    def get_weights(self):
        return list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(self)))


class NFC(BioSyst):
    """
    Base class for an NFC model
    """

    layers: List[NFCLayer]

    n_species: int = eqx.field(converter=int, static=True)
    n_inputs: int = eqx.field(converter=int, static=True)
    n_nodes: int = eqx.field(converter=int, static=True)
    n_layers: int = eqx.field(converter=int, static=True)

    def __init__(
        self,
        n_inputs: int,
        layer_sizes: Sequence[int],
        node_class: Type[NFCNodeBase],
        *,
        key: jt.PRNGKeyArray,
        **node_kwargs,
    ):
        self.n_layers = len(layer_sizes)
        self.n_nodes = sum(layer_sizes)
        self.n_inputs = n_inputs

        subkeys = jax.random.split(key, self.n_layers)

        layers = []
        input_size = n_inputs
        for i, (sk, n_nodes) in enumerate(zip(subkeys, layer_sizes)):
            layer = NFCLayer(
                input_size,
                n_nodes,
                node_class,
                is_first=(i == 0),
                is_last=(i == self.n_layers - 1),
                key=sk,
                **node_kwargs,
            )
            layers.append(layer)
            input_size = layer.n_nodes

        self.layers = layers
        self.n_species = sum(l.n_species for l in self.layers)

    @property
    def node_species(self) -> int:
        return self.layers[0].nodes.n_species

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_nodes, self.node_species

    @property
    def out_nodes(self):
        return self.layers[-1].n_nodes

    def ode_step(self, x: jax.Array, z: jax.Array) -> jax.Array:
        """
        Calculates the ODE step for the NFC model.
        Args:
            x: external input. Should have shape (n_inputs, )
            z: current state of the model. Should have shape (n_nodes, node_size)

        Returns:
            dz: the derivative of the state of the model.
                Should have shape (n_nodes, node_size)
        """
        if x.shape != (self.n_inputs,):
            raise ValueError(
                f"Invalid dimensions for input array."
                f"Expected {self.n_inputs}, got {x.shape}"
            )
        if z.shape != (self.n_nodes, self.node_species):
            raise ValueError(
                f"Invalid dimensions for state array."
                f"Expected {(self.n_nodes, self.node_species)}, got {z.shape}"
            )

        dz_vec = []
        x_in = x
        idx_start = 0
        for layer in self.layers:
            idx_end = idx_start + layer.n_nodes

            z_layer = z[idx_start:idx_end]

            dz = layer(x_in, z_layer)
            dz_vec.append(dz)

            x_in = z_layer[:, 0]
            idx_start = idx_end

        dz = jnp.concat(dz_vec, axis=0)

        return dz

    @staticmethod
    def _handle_inputs(
        x: jax.Array | jax.Array, ts: Optional[jax.Array]
    ) -> diffrax.LinearInterpolation:
        x = jnp.atleast_1d(x)

        if x.ndim == 1:
            x = jnp.array([x, x])
            ts = jnp.array([0.0, 1.0])

        interp = diffrax.LinearInterpolation(ts, x)
        return interp

    @eqx.filter_jit
    def diffrax_step(self, t, y, args):
        x = args[0].evaluate(t)
        out = self.ode_step(x, y)
        return out

    def simulate(
        self,
        x: jax.Array | jax.Array,
        ts: Optional[jax.Array],
        x_ts: Optional[jax.Array] = None,
        to_ss: bool = False,
        stiff: bool = True,
        throw: bool = True,
        max_steps: int = int(1e6),
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_stepsize: Optional[int] = None,
        progress_bar: bool = True,
    ) -> Tuple[jnp.ndarray, diffrax.Solution]:
        """
        Simulates the NFC evolution for a fixed input (using diffrax).
        The "predict" method is a wrapper around this one that only returns the
        output of the NFC.
        Args:
            x: value of the inputs. If the inputs are stable: [n_inputs, ].
                If the inputs are time-dependent: [timesteps, n_inputs]
            ts: time array for the simulation
            x_ts: time steps of the inputs x. If the inputs are stable, None.
            to_ss: Whether to simulate until steady-state or not.
                If True, the simulation will run until the steady-state is reached.
                As a result, ts will be ignored.
            stiff: True to use a stiff solver. Set to True for larger NFCs.
            throw: Raises an error when there's an error on the simulation.
            max_steps: maximum steps of the integration step
            rtol: Relative tolerance of the PID controller that determines the step size
            atol: Absolute tolerance of the PID controller that determines the step size
            max_stepsize: Maximum tolerable step size.
            progress_bar: Whether to show a progress bar.
        Returns:
            Simulated trajectories (time steps, n_nodes, node_species)

        """
        if x.shape[-1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, but got {x.shape[-1]}")

        if x.ndim == 2:
            if x_ts is None:
                raise ValueError(
                    "Input is a time-series, a time array for the input "
                    "should be provided"
                )
            if x.shape[0] != x_ts.size:
                raise ValueError(
                    f"Time-series for the input doesn't match "
                    f"the input length ts_x: {ts.shape}, x: {x.shape}"
                )
        if (ts is None) and (not to_ss):
            raise ValueError(
                "If the simulation is not to steady-state, a time array "
                "for the simulation should be provided."
            )
        if to_ss and (ts is not None):
            raise ValueError(
                "If the simulation is to steady-state, a time array "
                "for the simulation should not be provided."
            )

        y0 = jnp.zeros(self.shape)
        args = (self._handle_inputs(x, x_ts),)

        ys, sol = self._simulate(
            y0=y0,
            ts=ts,
            to_ss=to_ss,
            args=args,
            stiff=stiff,
            throw=throw,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            max_stepsize=max_stepsize,
            progress_bar=progress_bar,
        )
        return ys, sol

    def ss_estimation(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Returns an estimation of the steady-state of the system

        Args:
            x: value of the inputs. If the inputs here
                need to be stable (not time-varying): [n_inputs, ].
        Returns:
            Estimated steady-state (n_nodes, node_species)
        """
        if x.shape[-1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, but got {x.shape[-1]}")

        if x.ndim > 2:
            raise ValueError(
                f"Steady-state estimation only for on static inputs. Got {x.ndim}-D inp"
            )

        y = x
        for layer in self.layers:
            y = layer.ss_estimation(y)

        if y.size != self.layers[-1].shape[0]:
            raise ValueError(
                f"Invalid dimensions for output array of the steady-state estimation."
                f"Expected {self.layers[-1].shape[0]}, got {y.shape}"
            )

        return y

    def copy_weights(self, nfc: "NFC"):
        """
        Copies the weights from another NFC model
        """
        return jax.tree_util.tree_map(lambda _, y: y, self, nfc)

    def to_txt(self):
        """
        Returns a string representation of the ODEs layer
        """
        raise NotImplementedError()

    def to_sbml(self):
        """
        Returns a string representation of the ODEs layer in SBML format
        """
        raise NotImplementedError()

    def get_weights(self):
        return list(filter(eqx.is_inexact_array, jax.tree_util.tree_leaves(self)))


class MoormanNFC(NFC):  # pylint: disable=abstract-method
    """
    Utility wrapper around the NFC class directly implementing a Moorman NFC
    """

    def __init__(
        self,
        n_inputs: int,
        layer_sizes: Sequence[int],
        gamma: jt.ScalarLike = 1000.0,
        beta: jt.ScalarLike = 1.0,
        k: jt.ScalarLike = 0.2,
        activation: Optional[eqx.Module] = None,
        *,
        key: jt.PRNGKeyArray,
    ):
        super().__init__(
            n_inputs=n_inputs,
            layer_sizes=layer_sizes,
            node_class=MoormanPerceptron,
            gamma=gamma,
            beta=beta,
            k=k,
            activation=activation,
            key=key,
        )
