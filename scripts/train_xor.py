import dataclasses
import os
import traceback
from typing import Any, Callable, Iterator, List, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tyro
from tqdm import tqdm

import wandb
from biolearn import (
    NFC,
    BioSyst,
    MoormanNFC,
)
from biolearn.losses import *
from biolearn.losses.base import _robustnesses
from biolearn.losses.slack_relu import SlackModel
from biolearn.specifications.ss_classification import phi_xor_fast, phi_xor_ss

SS_CLASS_EXP_RESULTS_PATH = "data/results"

jax.config.update("jax_enable_x64", True)

LossType = Literal[
    "relu",
    "sigmoid",
    "slack_relu",
    "silu",
    "softmax",
    "logsumexp",
    "slack_softmax",
    "leaky_relu",
    "elu",
    "softrelu",
]

SpecType = Literal["phi_xor_ss", "phi_xor_fast"]

ELEMENTWISE_LOSSES: dict[str, Callable] = {
    "sigmoid": lambda r: jax.nn.sigmoid(-r),
    "silu": lambda r: jax.nn.silu(-r),
    "softmax": lambda r: jnp.expm1(-r) / jnp.max(jnp.expm1(-r)),
    "logsumexp": lambda r: jnp.expm1(-r) / jnp.max(jnp.expm1(-r)),
    "leaky_relu": lambda r: jax.nn.leaky_relu(-r),
    "elu": lambda r: jax.nn.elu(-5.0 * r),
    "relu": lambda r: jax.nn.relu(-r),
    "softrelu": lambda r: jax.nn.relu(-r) + 1e-2 * jax.nn.sigmoid(jax.nn.relu(r)),
}


def _get_elementwise_loss(loss_name: str, model=None) -> Callable:
    """Return a per-element loss function ``r -> loss`` for the given loss name.

    For slack-based losses the current slack value is read from *model*.
    """
    slack = getattr(model, "slack", None)
    if loss_name == "slack_relu" and slack is not None:
        return lambda r: jax.nn.relu(slack - r)
    if loss_name == "slack_softmax" and slack is not None:
        return lambda r: jnp.expm1(-r) / jnp.max(jnp.expm1(-r))

    return ELEMENTWISE_LOSSES.get(loss_name, lambda r: jax.nn.relu(-r))


@dataclasses.dataclass
class TrainConfig:
    """Train XOR classification with different loss functions."""

    loss: LossType
    """Loss function to use for training."""
    spec: SpecType = "phi_xor_fast"
    """STL specification to evaluate on trajectories."""
    semantics: Literal["dgmsr", "smooth", "classical", "agm"] = "dgmsr"
    """PySTL robustness semantics used by `phi_xor_ss`."""
    dgmsr_p: int = 3
    """D-GMSR 'p' parameter when `--semantics dgmsr`."""
    smooth_temperature: float = 1.0
    """Temperature used when `--semantics smooth`."""
    seed: int = 0
    """Random seed."""
    epochs: int = 1000
    """Number of training epochs."""
    lr: float = 0.05
    """Learning rate."""
    n_seeds: int = 3
    """Number of random seeds to train with."""
    n_samples: int = 121
    """Number of samples (should be a perfect square)."""
    show: bool = False
    """Show plots after training."""
    save: bool = True
    """Save results to disk."""
    integral: bool = False
    """Use Monte-Carlo integral loss over the input domain."""
    n_points: int = 128
    """Number of MC sample points when --integral is set."""
    wandb_project: str = "biolearn"
    """Weights & Biases project name."""
    log_interval: int = 100
    """Log contour plots to wandb every N epochs."""


def _clip_pytree(
    pytree: jt.PyTree,
    clip_filter_fn: Optional[Callable[[jt.PyTree], jt.Union[Any, Sequence[Any]]]],
    min_val: jt.ScalarLike,
    max_val: jt.ScalarLike,
):
    """
    Clipping function for PyTrees.
    Args:
        pytree: pytree to clip.
        min_val: minimum value to clip to.
        max_val: maximum value to clip to.

    Returns:
        pytree: clipped pytree.
    """
    if clip_filter_fn is None:
        new_nfc = jax.tree_util.tree_map(
            lambda x: jnp.clip(x, min_val, max_val),
            pytree,
            is_leaf=eqx.is_inexact_array,
        )
    else:
        new_nfc = eqx.tree_at(
            clip_filter_fn,
            pytree,
            replace_fn=lambda _x: jnp.clip(_x, min_val, max_val),
            is_leaf=eqx.is_inexact_array,
        )

    return new_nfc


def adam(
    lr: jt.ScalarLike,
    transition_steps: jt.ScalarLike = 10,
    decay_rate: jt.ScalarLike = 0.9,
    transition_begin: jt.ScalarLike = 50,
    clip_margin: Tuple[jt.ScalarLike, jt.ScalarLike] = (1e-3, jnp.inf),
    clip_filter: Optional[Callable[[jt.PyTree], jt.Union[Any, Sequence[Any]]]] = None,
):
    """
    Adam optimizer customized to NFCs. It incorporates:
        * non-negative parameters
        * clipping of parameters with custom filter and margins
        * learning rate scheduler.

    Args:
        lr: initial learning rate
        transition_steps: transition steps before the decay rate is applied
        decay_rate: decay rate of the learning rate
        transition_begin: number of steps before the initial decay is applied
        clip_margin: margins to clip the parameters
        clip_filter: function to select a subset of parameters
            on which to apply clipping
    Returns:
        optax.GradientTransformation: Adam optimizer
            with custom learning rate scheduler
    """

    def non_negative():
        def init_fn(params):  # pylint: disable=unused-argument
            return ()

        def update_fn(updates, state, params=None):
            _new_params = eqx.apply_updates(params, updates)
            _non_neg_params = jax.tree_util.tree_map(
                jax.nn.relu, _new_params, is_leaf=eqx.is_inexact_array
            )

            clipped_updates = jax.tree_util.tree_map(
                lambda clipped, old: clipped - old, _non_neg_params, params
            )
            return clipped_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    def clip_params():  # pylint: disable=unused-variable
        def init_fn(params):  # pylint: disable=unused-argument
            return ()

        def update_fn(updates, state, params=None):
            _new_params = eqx.apply_updates(params, updates)
            _non_neg_params = jax.tree_util.tree_map(
                jax.nn.relu, _new_params, is_leaf=eqx.is_inexact_array
            )
            _clipped_params = _clip_pytree(
                _non_neg_params, clip_filter, clip_margin[0], clip_margin[1]
            )

            clipped_updates = jax.tree_util.tree_map(
                lambda clipped, old: clipped - old, _clipped_params, params
            )
            return clipped_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=transition_begin,
    )

    optimizer = optax.chain(
        optax.scale_by_adam(),
        # optax.clip_by_global_norm(10.0),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
        non_negative(),
        # clip_params(),
    )

    return optimizer


def compute_grad_mag(grads):
    """Compute the magnitude of gradients."""
    squared_grads = jax.tree_util.tree_map(
        lambda g: jnp.sum(g**2), grads, is_leaf=eqx.is_inexact_array
    )
    total_grad = sum(
        jax.tree_util.tree_leaves(squared_grads, is_leaf=eqx.is_inexact_array)
    )
    grad_mag = jnp.sqrt(total_grad)
    return grad_mag


def compute_difference(pytree1: jt.PyTree, pytree2: jt.PyTree) -> jt.PyTree:
    """Computes the difference (in abs) of two identically structured pytrees"""

    pytree1_params, pytree2_params = (
        eqx.filter(pytree1, eqx.is_array),
        eqx.filter(pytree2, eqx.is_array),
    )

    pytree_sub = jax.tree_util.tree_map(
        lambda x, y: jnp.abs(x - y).sum(),
        pytree1_params,
        pytree2_params,
        is_leaf=eqx.is_inexact_array,
    )
    total_sub = sum(jax.tree_util.tree_leaves(pytree_sub, is_leaf=eqx.is_inexact_array))
    return total_sub


# Toy dataloader
def dataloader(
    arrays: jt.PyTree, batch_size: int, *, key: jt.PRNGKeyArray
) -> Iterator[jt.PyTree]:
    """
    Loads data in batches. This is an infinite generator.
    Args:
        arrays: pytree containing the data.
        batch_size: size of the batches.
        key: random key generator for the data loader
    Returns:

    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)

    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        _, key = jax.random.split(key, 2)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def freeze_parameter_mask(nfc: NFC, parameters: List[str]) -> NFC:
    """
    Creates an NFC-type pytree with a mask that is True only for inexact arrays
    that are not on the parameters list.
    Args:
        nfc: nfc to be masked
        parameters: list of parameters to freeze
    Returns:
        NFC-like pytree with the mask
    """
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, nfc)

    for param in parameters:
        for i, layer in enumerate(mask.layers):
            current_val = getattr(layer.nodes, param)
            current_val_false = jax.tree_util.tree_map(lambda _: False, current_val)
            mask = eqx.tree_at(
                lambda _x, i=i, param=param: getattr(_x.layers[i].nodes, param),
                mask,
                replace=current_val_false,
            )

    return mask


def _log_contour_plots(
    model,
    plot_xs,
    plot_grid,
    ts,
    specification,
    specification_kwargs,
    epoch,
    loss_name,
):
    """Compute robustness over the grid and log contour plots to wandb."""
    ros = _robustnesses(model, plot_xs, ts, specification, **specification_kwargs)
    grid_x, grid_y = plot_grid
    ros_grid = jax.nn.hard_tanh(ros.reshape(grid_x.shape))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = 0.1
    n_levels = 30

    # Robustness contour
    cf0 = axes[0].contourf(
        grid_x,
        grid_y,
        ros_grid,
        levels=jnp.linspace(-vmax, vmax, n_levels),
        cmap="RdBu_r",
        extend="both",
    )
    axes[0].contour(
        grid_x,
        grid_y,
        ros_grid,
        levels=[0.0],
        colors=["#c0392b"],
        linewidths=2.5,
        zorder=6,
    )
    fig.colorbar(cf0, ax=axes[0])
    axes[0].set_title(f"Robustness (epoch {epoch})")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    # Loss contour (per-sample loss using the chosen loss function)
    elem_loss = _get_elementwise_loss(loss_name, model)
    loss_grid = elem_loss(ros_grid)
    cf1 = axes[1].contourf(
        grid_x,
        grid_y,
        loss_grid,
        levels=n_levels,
        cmap="RdBu_r",
        extend="both",
    )
    fig.colorbar(cf1, ax=axes[1])
    axes[1].set_title(f"Loss (epoch {epoch})")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({"contour_plots": wandb.Image(fig)}, commit=False)
    plt.close(fig)


def train(
    model: jt.PyTree,
    optimizer: optax.GradientTransformation,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[jt.PyTree, jt.Array, jt.Array], jt.Scalar],
    x_train: jt.Array,
    freeze_mask: Optional[jt.PyTree] = None,
    verbose: bool = True,
    log_interval: int = 0,
    plot_xs: Optional[jt.Array] = None,
    plot_grid: Optional[Tuple[jt.Array, jt.Array]] = None,
    ts: Optional[jt.Array] = None,
    specification: Optional[Callable] = None,
    specification_kwargs: Optional[dict[str, Any]] = None,
    loss_name: str = "",
    *,
    key: jt.PRNGKeyArray,
) -> Tuple[jt.PyTree, List[float]]:
    """
    Training loop for an NFC model.

    Args:
        model: model to train, normally an NFC, but could be any pytree.
        optimizer: optimizer to use for training.
        epochs: number of training epochs.
        batch_size: batch size.
        loss_fn: loss function to use.
        x_train: training data.
        y_train: target training data.
        freeze_mask: function masking trainable parameters or the model.
            All inexact arrays are trained by default
        verbose: whether to print training info.
        log_interval: log contour plots to wandb every N epochs (0 = disabled).
        plot_xs: grid points for contour plotting.
        plot_grid: (grid_x, grid_y) meshgrid for contour plotting.
        ts: time array for simulation (needed for contour plots).
        specification: STL specification function (needed for contour plots).
        loss_name: name of the loss function (for element-wise contour plotting).
        key: random key generator for the data loader

    Returns:
        Tuple of (trained model, loss trajectory).
    """
    if batch_size > x_train.shape[0]:
        raise ValueError("Batch size cannot be larger than the dataset size.")

    freeze_mask = eqx.is_inexact_array if freeze_mask is None else freeze_mask

    def get_loss(_diff_model, _static_model, _x_batch):
        _joint_model = eqx.combine(_diff_model, _static_model)
        # Loss functions from `biolearn.losses` follow the signature
        # (system, xs, _ys). `_ys` is unused for STL losses.
        _ys_dummy = jnp.zeros((_x_batch.shape[0],), dtype=float)
        _loss = loss_fn(_joint_model, _x_batch, _ys_dummy)
        return _loss

    @eqx.filter_jit
    def batch_step(_model, _x_batch, _optim_state):
        _diff_model, _static_model = eqx.partition(
            _model, freeze_mask, is_leaf=eqx.is_inexact_array
        )
        _loss, _grads = eqx.filter_value_and_grad(get_loss)(
            _diff_model, _static_model, _x_batch
        )
        _updates, _new_optim_state = optimizer.update(
            _grads,
            _optim_state,
            _diff_model,
            value=_loss,
            grad=_grads,
            value_fn=lambda _x: get_loss(_x, _static_model, _x_batch),
        )
        _new_model = eqx.apply_updates(_model, _updates)
        _delta = compute_difference(_model, _new_model)
        _grad_mag = compute_grad_mag(_grads)
        return _loss, _delta, _grad_mag, _new_model, _new_optim_state

    data_iter = dataloader((x_train,), batch_size, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_traj = []

    do_plots = (
        log_interval > 0
        and plot_xs is not None
        and plot_grid is not None
        and ts is not None
        and specification is not None
    )
    specification_kwargs = {} if specification_kwargs is None else specification_kwargs

    pbar = tqdm(range(epochs), total=epochs, disable=not verbose)
    for epoch_idx in pbar:
        (x_batch,) = next(data_iter)
        try:
            loss, delta_mag, grad_mag, model, opt_state = batch_step(
                model, x_batch, opt_state
            )
        except Exception:  # pylint: disable=broad-except
            tqdm.write(f"Stopped due to error raised: {epochs} epochs")
            traceback.print_exc()
            break

        loss_val = float(loss)
        loss_traj.append(loss_val)
        if wandb.run is not None:
            wandb.log(
                {
                    "loss": loss_val,
                    "grad_mag": float(grad_mag),
                    "delta_mag": float(delta_mag),
                }
            )
        if loss_val <= 0.0:
            if verbose:
                tqdm.write(
                    f"Early stopping: loss reached 0 at epoch {epoch_idx + 1}."
                )
            break
        if do_plots and (epoch_idx + 1) % log_interval == 0:
            _log_contour_plots(
                model,
                plot_xs,
                plot_grid,
                ts,
                specification,
                specification_kwargs,
                epoch_idx + 1,
                loss_name,
            )
        pbar.set_description(
            f"Loss: {loss:.4f}, Delta Mag: {delta_mag:.4f}, Grad Mag: {grad_mag:.4f}"
        )
    else:
        if verbose:
            tqdm.write(f"Reached the end of the {epochs} epochs")
    return model, loss_traj


class ClassModel(BioSyst):
    """
    Implements a system with an NFC + a fluorescent protein
    """

    nfc: NFC
    k: float = eqx.field(static=True, converter=float)
    delta: float = eqx.field(static=True, converter=float)
    w: float = eqx.field(static=True, converter=float)

    def __init__(self, nfc: NFC, k: float, delta: float, w: float):
        self.nfc = nfc
        self.k = k
        self.delta = delta
        self.w = w

    @eqx.filter_jit
    def diffrax_step(self, t, y, args):
        z, o = y[:-1].reshape(self.nfc.shape), y[-1]
        x_in = args[0].evaluate(t)
        dz = self.nfc.ode_step(x_in, z)
        u = z[-self.nfc.out_nodes :, 0]
        do = self.w * u / (u + self.k) - self.delta * o
        dx = jnp.concat([dz.flatten(), do])
        return dx

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
    ) -> jnp.ndarray:
        args = (self.nfc._handle_inputs(x, x_ts),)
        y0 = jnp.zeros(self.nfc.n_species + 1)

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


def create_dataset(n_samples: int, max_val=1.0):
    """
    Create a 2D grid dataset for training.
    The dataset is created by generating a grid of points in the range [0, max_val]
    Args:
        n_samples: number of samples to generate (should be a perfect square)
        max_val: maximum value for the grid points
    Returns:
        a tuple containing: the generated dataset (x_train)
            and the grid (grid_x, grid_y)
    """
    n_per_axis = int(jnp.sqrt(n_samples))
    x1 = jnp.linspace(0, max_val, n_per_axis)
    x2 = jnp.linspace(0, max_val, n_per_axis)
    grid_x, grid_y = jnp.meshgrid(x1, x2)
    x_train = jnp.column_stack((grid_x.ravel(), grid_y.ravel()))
    return x_train, (grid_x, grid_y)


def _make_loss_fn(
    loss_name: str,
    ts: jax.Array,
    *,
    integral: bool = False,
    n_points: int = 128,
    key: Optional[jax.Array] = None,
    spec: SpecType = "phi_xor_fast",
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    """Create the loss function and optional model wrapper for the given loss type."""
    # Build integral kwargs when MC sampling is requested.
    integral_kwargs = {}
    if integral:
        assert key is not None, "key is required when integral=True"
        integral_kwargs = dict(
            domain=BoxDomain(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
            n_points=n_points,
            key=key,
        )

    spec_kwargs = dict(
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
    )

    if spec == "phi_xor_fast":
        specification = phi_xor_fast
    elif spec == "phi_xor_ss":
        specification = phi_xor_ss
    else:
        raise ValueError(f"Unknown specification: {spec!r}")

    if loss_name == "relu":
        loss_fn = make_relu_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "sigmoid":
        loss_fn = make_sigmoid_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "silu":
        loss_fn = make_silu_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "logsumexp":
        loss_fn = make_logsumexp_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "softmax":
        loss_fn = make_softmax_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "leaky_relu":
        loss_fn = make_leaky_relu_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "elu":
        loss_fn = make_elu_loss(
            specification=specification, ts=ts, **spec_kwargs, **integral_kwargs
        )
        wrap_model = None
    elif loss_name == "slack_relu":
        loss_fn = slack_relu_ic_loss(specification=specification, ts=ts, **spec_kwargs)
        wrap_model = SlackModel
    elif loss_name == "softrelu":
        loss_fn = make_softrelu_loss(specification=specification, ts=ts, **spec_kwargs)
        wrap_model = SlackModel
    elif loss_name == "slack_softmax":
        loss_fn = slack_softmax_loss(specification=specification, ts=ts, **spec_kwargs)
        wrap_model = SlackModel
    else:
        raise ValueError(f"Unknown loss function: {loss_name!r}")
    return loss_fn, wrap_model, specification, spec_kwargs


def train_model(
    biosyst: ClassModel,
    x_train: jax.Array,
    loss_name: str,
    *,
    spec: SpecType = "phi_xor_fast",
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
    lr: float = 0.05,
    epochs: int = 1200,
    show: bool = False,
    integral: bool = False,
    n_points: int = 128,
    log_interval: int = 0,
    key: jax.Array,
) -> Tuple[NFC, List[float]]:
    """
    Trains the simulator model given a dataset with parameter sets and
    a list of robustness functions.

    Args:
        biosyst: the NFC to train
        x_train: dataset with the values for x.
        loss_name: which loss function to use (xor_ss, sigmoid, slack_relu).
        integral: use Monte-Carlo integral loss.
        n_points: number of MC sample points when integral is True.
        key: random key to use for training.
    Returns:

    """
    x_diff = x_train[:, 1] - x_train[:, 0]
    y_train = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)

    ts = jnp.arange(0, 20, 1.0)
    x_plot = jnp.linspace(0, 1, 128)
    xs, ys = jnp.meshgrid(x_plot, x_plot)
    x_plot = jnp.stack([xs.flatten(), ys.flatten()], axis=-1)
    grid = (xs, ys)

    loss_key, key = jax.random.split(key)
    loss_fn, wrap_model, specification, spec_kwargs = _make_loss_fn(
        loss_name,
        ts,
        integral=integral,
        n_points=n_points,
        key=loss_key,
        spec=spec,
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
    )

    trainable = wrap_model(biosyst) if wrap_model is not None else biosyst

    learning_rate_schedule = optax.exponential_decay(
        init_value=lr,  # Starting learning rate
        transition_steps=1000,  # Number of steps before each decay
        decay_rate=0.5,  # Multiplicative factor of decay
        staircase=True,  # If True, learning rate decays in discrete steps
    )

    optimizer = optax.adabelief(learning_rate=learning_rate_schedule)

    try:
        trained_model, loss_traj = train(
            trainable,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=x_train.shape[0],
            x_train=x_train,
            log_interval=log_interval,
            plot_xs=x_plot,
            plot_grid=grid,
            ts=ts,
            specification=specification,
            specification_kwargs=spec_kwargs,
            loss_name=loss_name,
            key=key,
        )
        # Unwrap if we used a model wrapper
        if wrap_model is not None:
            trained_syst = trained_model.model
        else:
            trained_syst = trained_model
    except KeyboardInterrupt:
        trained_syst = biosyst
        loss_traj = [float(jnp.inf)]

    def _count_satisfied(x_batch):
        def _run_single(xi):
            y_ss_i, _ = trained_syst.simulate(
                x=xi,
                ts=ts,
                to_ss=False,
                stiff=True,
                max_steps=int(1e6),
                rtol=1e-6,
                atol=1e-6,
                progress_bar=False,
            )
            x_traj = jnp.array([[xi[0], xi[1]]])
            x_traj = jnp.repeat(x_traj, repeats=y_ss_i.shape[0], axis=0)
            y_out = y_ss_i[:, -1][:, None]
            traj = jnp.concatenate([x_traj, y_out], axis=1)
            return specification(traj, **spec_kwargs)

        ros = jax.vmap(_run_single)(x_batch)
        satisfied = ros >= 0.0

        return satisfied

    satisfied = _count_satisfied(x_train)
    satisfied_count = satisfied.sum()
    print(
        f"Specification satisfied on {satisfied_count}/{satisfied.shape[0]} initial conditions."
    )
    n = int(jnp.sqrt(satisfied.shape[0]))
    print(satisfied.reshape((n, n)))

    if show:
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()

        y_out, _ = jax.vmap(
            lambda _xi: trained_syst.simulate(
                x=_xi,
                ts=None,
                to_ss=True,
                stiff=True,
                max_steps=int(1e6),
                rtol=1e-4,
                atol=1e-4,
                progress_bar=False,
            )
        )(x_train)

        plot(
            x1_grid=grid[0],
            x2_grid=grid[1],
            y=y_train.reshape(grid[0].shape),
            title="Target",
            ax=axes[0],
            colorbar=True,
        )
        plot(
            x1_grid=grid[0],
            x2_grid=grid[1],
            y=y_out.reshape(grid[0].shape),
            title="Predicted",
            ax=axes[1],
            colorbar=True,
        )
        plt.show()

    return trained_syst, loss_traj


def run_training(
    layer_sizes,
    x_train,
    k1,
    k2,
    w,
    epochs: int,
    lr: float,
    loss_name: str,
    show=False,
    integral=False,
    n_points=128,
    wandb_project: Optional[str] = None,
    seed_idx: int = 0,
    log_interval: int = 0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    arch_str = "-".join(str(l) for l in layer_sizes)
    integral_tag = "integral" if integral else "batch"
    run_name = f"{loss_name}_{integral_tag}_arch{arch_str}_seed{seed_idx}"

    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "loss": loss_name,
            "integral": integral,
            "n_points": n_points,
            "epochs": epochs,
            "lr": lr,
            "arch": layer_sizes,
            "seed_idx": seed_idx,
        },
    )

    sk1, sk2 = jax.random.split(key, num=2)
    print("Dataset shape: ", x_train.shape)
    nfc = MoormanNFC(
        n_inputs=2,
        layer_sizes=layer_sizes,
        gamma=1000,
        beta=1.0,
        k=k1,
        key=sk1,
    )
    biosyst = ClassModel(nfc=nfc, k=k2, delta=1.0, w=w)

    trained_biosyst, loss_traj = train_model(
        biosyst,
        x_train,
        loss_name=loss_name,
        spec=spec,
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
        epochs=epochs,
        lr=lr,
        show=show,
        integral=integral,
        n_points=n_points,
        log_interval=log_interval,
        key=sk2,
    )

    wandb.finish()

    return trained_biosyst, loss_traj


def repeat_training(
    layer_sizes,
    x_train,
    n,
    epochs=1200,
    lr=0.05,
    loss_name="xor_ss",
    save_path=None,
    show=False,
    integral=False,
    n_points=128,
    wandb_project: Optional[str] = None,
    log_interval: int = 0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    keys = jax.random.split(key, num=n)

    info = {}
    all_params = []
    for i, ki in enumerate(keys):
        trained_sim, loss_traj = run_training(
            layer_sizes,
            epochs=epochs,
            lr=lr,
            loss_name=loss_name,
            x_train=x_train,
            key=ki,
            show=show,
            integral=integral,
            n_points=n_points,
            wandb_project=wandb_project,
            seed_idx=i,
            log_interval=log_interval,
            w=5.0,
            k1=0.8,
            k2=0.8,
            spec=spec,
            semantics=semantics,
            dgmsr_p=dgmsr_p,
            smooth_temperature=smooth_temperature,
        )

        info.update(
            {f"{i}": [len(loss_traj), loss_traj[-1] if loss_traj else float(jnp.inf)]}
        )
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(trained_sim, eqx.is_inexact_array)
        )
        flat_params = jnp.concatenate([l.flatten() for l in leaves])
        all_params.append(flat_params)

    # Log 3D point cloud of learned parameters to wandb
    if wandb_project and all_params:
        n_models = len(all_params)
        all_points = []
        for i, params in enumerate(all_params):
            n_usable = ((len(params) + 3) // 3) * 3
            pad = n_usable - len(params)
            params = jnp.concatenate([params, jnp.zeros(pad)])
            pts = params.reshape(-1, 3)

            # Color: fixed hue (blue), varying brightness across models
            v = 0.4 + 0.6 * i / max(n_models - 1, 1)
            h = 0.4 + 0.6 * i / max(n_models - 1, 1)
            rgb = matplotlib.colors.hsv_to_rgb([h, 0.8, v])
            rgb_255 = (jnp.array(rgb) * 255).astype(jnp.int32)
            colors = jnp.tile(rgb_255, (pts.shape[0], 1))
            all_points.append(jnp.concatenate([pts, colors], axis=1))

        point_cloud = jnp.concatenate(all_points, axis=0)
        integral_tag = "integral" if integral else "nointegral"
        arch_str = str(list(layer_sizes))
        run_name = f"{loss_name}_{integral_tag}_arch{arch_str}_params3d"
        wandb.init(project=wandb_project, name=run_name)
        wandb.log({"params_3d": wandb.Object3D.from_numpy(np.array(point_cloud))})
        wandb.finish()

    return info


def analyze_layer_sizes(
    layer_sizes_list,
    x_train,
    epochs=1000,
    lr=0.05,
    loss_name="xor_ss",
    n=10,
    save=False,
    show=False,
    integral=False,
    n_points=128,
    wandb_project: Optional[str] = None,
    log_interval: int = 0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    all_info = []
    for layer_sizes in layer_sizes_list:
        arch_str = "-".join(str(l) for l in layer_sizes)
        print(f"******* Training for architecture {arch_str} ********")

        experiment_name = f"runs_arch_{arch_str}"

        model_save_path = os.path.join(SS_CLASS_EXP_RESULTS_PATH, experiment_name)

        _, key = jax.random.split(key, num=2)
        info = repeat_training(
            layer_sizes,
            x_train=x_train,
            n=n,
            key=key,
            epochs=epochs,
            lr=lr,
            loss_name=loss_name,
            save_path=model_save_path,
            show=show,
            integral=integral,
            n_points=n_points,
            wandb_project=wandb_project,
            log_interval=log_interval,
            spec=spec,
            semantics=semantics,
            dgmsr_p=dgmsr_p,
            smooth_temperature=smooth_temperature,
        )

        info_df = pd.DataFrame(info, columns=["Steps", "final loss"])

        if save:
            df_path = os.path.join(model_save_path, "runs.csv")
            info_df.to_csv(df_path)
        all_info.append(info_df)

    return all_info


def main(cfg: TrainConfig):
    key = jax.random.PRNGKey(cfg.seed)
    key = jax.random.split(key, num=5)[-1]

    archs = [(2, 1), (2, 1, 1)]

    x_train, _ = create_dataset(n_samples=cfg.n_samples, max_val=1.0)

    info = analyze_layer_sizes(
        archs,
        x_train=x_train,
        epochs=cfg.epochs,
        lr=cfg.lr,
        loss_name=cfg.loss,
        save=cfg.save,
        show=cfg.show,
        integral=cfg.integral,
        n_points=cfg.n_points,
        wandb_project=cfg.wandb_project,
        log_interval=cfg.log_interval,
        n=cfg.n_seeds,
        key=key,
        spec=cfg.spec,
        semantics=cfg.semantics,
        dgmsr_p=cfg.dgmsr_p,
        smooth_temperature=cfg.smooth_temperature,
    )

    import code

    code.interact(local=locals())


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
