from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt


@dataclass
class SimulateConfig:
    to_ss: bool = True
    stiff: bool = True
    throw: bool = True
    max_steps: int = int(1e6)
    rtol: float = 1e-10
    atol: float = 1e-10
    max_stepsize: Optional[int] = None
    progress_bar: bool = True
    """Args for diffeqsolve"""
    args: Tuple = ()


class BioModel(eqx.Module):
    """
    Base class to create biological systems inheriting the diffeqsolve function.
    """

    @abstractmethod
    def ode_step(self, t: jt.ScalarLike, y: jax.Array, args: Tuple) -> jax.Array:
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
        config: SimulateConfig,
    ):
        stepsize_controller = diffrax.PIDController(
            pcoeff=0.3 if config.stiff else 0.0,
            icoeff=0.3 if config.stiff else 1.0,
            dcoeff=0,
            rtol=config.rtol,
            atol=config.atol,
            dtmax=config.max_stepsize,
            dtmin=1e-10,
        )

        if config.to_ss:
            saveat = diffrax.SaveAt(t1=True)
            event = diffrax.Event(
                diffrax.steady_state_event(rtol=2 * config.rtol, atol=2 * config.atol)
            )
            t0, t1, dt0 = 0.0, jnp.inf, None
        else:
            saveat = diffrax.SaveAt(ts=ts)
            event = None
            assert ts is not None
            t0, t1, dt0 = ts[0], ts[-1], 0.001

        progress_bar = (
            diffrax.TqdmProgressMeter()
            if config.progress_bar
            else diffrax.NoProgressMeter()
        )
        solver = diffrax.Kvaerno5() if config.stiff else diffrax.Tsit5()

        term = diffrax.ODETerm(self.ode_step)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=config.max_steps,
            throw=config.throw,
            args=config.args,
            progress_meter=progress_bar,
            event=event,
        )

        y_pred = solution.ys

        return y_pred, solution

    def simulate(self, *args, config: SimulateConfig = SimulateConfig(), **kwargs):
        """Simulate the biological model."""
        raise NotImplementedError
