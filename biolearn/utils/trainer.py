from typing import Tuple

import jax.numpy as jnp
import jaxtyping as jt


class EarlyStopper:
    """
    Early stopping class for training loops.
    """

    def __init__(self, delta_loss_tol, patience, loss_tol, delta_tol):
        self.delta_loss_tol = delta_loss_tol
        self.max_patience = patience
        self.loss_tol = loss_tol
        self.delta_tol = delta_tol

        self.patience = 0
        self.steps = 0
        self.loss_traj = []
        self.grad_traj = []
        self.previous_loss = jnp.inf

    def evaluate(
        self, loss: jt.ScalarLike, grad_mag: jt.ScalarLike = jnp.inf
    ) -> Tuple[bool, str]:
        """
        Evaluates the current state of the training and decides if the
        training should be stopped early, and why.

        Args:
            loss: current loss.
            grad_mag: current gradient_magnitude
        Returns:
            stop: whether to stop the training.
            msg: message explaining the reason to stop.
        """
        self.loss_traj.append(loss)
        self.grad_traj.append(grad_mag)
        self.steps += 1

        stop = False
        msg = ""

        if jnp.abs(loss - self.previous_loss) < self.delta_loss_tol:
            self.patience += 1
        else:
            self.patience = 0

        if self.patience > self.max_patience:
            msg = (
                f"Early stopping due to no improvement in loss: "
                f"({jnp.abs(loss - self.previous_loss)} < {self.delta_loss_tol})"
            )
            stop = True
        elif grad_mag < self.delta_tol:
            msg = (
                f"Early stopping due to small magnitude of parameter change: "
                f"({grad_mag}<{self.delta_tol})"
            )
            stop = True
        if jnp.isnan(loss):
            msg = "Early stopping due to NaN loss"
            stop = True

        if loss < self.loss_tol:
            msg = f"Early stopping due to low loss: ({loss}<{self.loss_tol})"
            stop = True

        self.previous_loss = loss

        return stop, msg
