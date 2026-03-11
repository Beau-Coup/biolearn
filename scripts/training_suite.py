"""Shared training infrastructure for biolearn experiments."""

import traceback
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

import wandb
from biolearn.losses.base import _robustnesses
from biolearn.models.base import SimulateConfig

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


def dataloader(
    arrays: jt.PyTree, batch_size: int, *, key: jt.PRNGKeyArray
) -> Iterator[jt.PyTree]:
    """Infinite mini-batch iterator."""
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


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _clip_pytree(
    pytree: jt.PyTree,
    clip_filter_fn: Optional[Callable[[jt.PyTree], Union[Any, Sequence[Any]]]],
    min_val: jt.ScalarLike,
    max_val: jt.ScalarLike,
):
    if clip_filter_fn is None:
        return jax.tree_util.tree_map(
            lambda x: jnp.clip(x, min_val, max_val),
            pytree,
            is_leaf=eqx.is_inexact_array,
        )
    return eqx.tree_at(
        clip_filter_fn,
        pytree,
        replace_fn=lambda _x: jnp.clip(_x, min_val, max_val),
        is_leaf=eqx.is_inexact_array,
    )


def adam(
    lr: jt.ScalarLike,
    transition_steps: int = 10,
    decay_rate: float = 0.9,
    transition_begin: int = 50,
):
    """Adam optimizer with non-negativity constraints and LR schedule."""

    def non_negative():
        def init_fn(params):
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

    scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=transition_begin,
    )

    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
        non_negative(),
    )

    return optimizer


# ---------------------------------------------------------------------------
# Gradient / parameter utilities
# ---------------------------------------------------------------------------


def compute_grad_mag(grads):
    """Scalar L2 gradient magnitude."""
    squared_grads = jax.tree_util.tree_map(
        lambda g: jnp.sum(g**2), grads, is_leaf=eqx.is_inexact_array
    )
    total_grad = sum(
        jax.tree_util.tree_leaves(squared_grads, is_leaf=eqx.is_inexact_array)
    )
    return jnp.sqrt(total_grad)


def compute_difference(pytree1: jt.PyTree, pytree2: jt.PyTree) -> jt.PyTree:
    """L1 parameter delta between two identically structured pytrees."""
    pytree1_params = eqx.filter(pytree1, eqx.is_array)
    pytree2_params = eqx.filter(pytree2, eqx.is_array)
    pytree_sub = jax.tree_util.tree_map(
        lambda x, y: jnp.abs(x - y).sum(),
        pytree1_params,
        pytree2_params,
        is_leaf=eqx.is_inexact_array,
    )
    return sum(jax.tree_util.tree_leaves(pytree_sub, is_leaf=eqx.is_inexact_array))


def freeze_parameter_mask(nfc, parameters: List[str]):
    """Trainable/frozen partition mask — True only for unfrozen inexact arrays."""
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


# ---------------------------------------------------------------------------
# Contour plot logging
# ---------------------------------------------------------------------------


def _log_contour_plots(
    model,
    plot_xs,
    plot_grid,
    ts,
    specification,
    specification_kwargs,
    epoch,
    loss_name,
    elem_loss_fn=None,
    boundary_xs=None,
):
    """Compute robustness over a grid and log contour plots to wandb."""
    ros = _robustnesses(model, plot_xs, ts, specification, **specification_kwargs)
    grid_x, grid_y = plot_grid
    ros_grid = jax.nn.hard_tanh(ros.reshape(grid_x.shape))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = 0.1
    n_levels = 30

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

    if elem_loss_fn is not None:
        loss_grid = elem_loss_fn(ros_grid)
        cf1 = axes[1].contourf(
            grid_x, grid_y, loss_grid, levels=n_levels, cmap="RdBu_r", extend="both"
        )
        fig.colorbar(cf1, ax=axes[1])
        axes[1].set_title(f"Loss (epoch {epoch})")
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")

    if boundary_xs is not None:
        b = np.array(boundary_xs)
        for ax in axes:
            ax.scatter(
                b[:, 0], b[:, 1], c="black", s=6, alpha=0.6, linewidths=0, zorder=7
            )

    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({"contour_plots": wandb.Image(fig)}, commit=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


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
    specification_kwargs: Optional[dict] = None,
    loss_name: str = "",
    early_stop: Optional[float] = None,
    x_holdout: Optional[jt.Array] = None,
    boundary_xs: Optional[jt.Array] = None,
    elem_loss_fn: Optional[Callable] = None,
    config: SimulateConfig = SimulateConfig(),
    *,
    key: jt.PRNGKeyArray,
) -> Tuple[jt.PyTree, List[float]]:
    """Generic training loop for any Equinox PyTree model."""
    if batch_size > x_train.shape[0]:
        raise ValueError("Batch size cannot be larger than the dataset size.")

    freeze_mask = eqx.is_inexact_array if freeze_mask is None else freeze_mask

    def get_loss(_diff_model, _static_model, _x_batch):
        _joint_model = eqx.combine(_diff_model, _static_model)
        _ys_dummy = jnp.zeros((_x_batch.shape[0],), dtype=float)
        return loss_fn(_joint_model, _x_batch, _ys_dummy)

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
        except Exception:
            tqdm.write(f"Stopped due to error raised: {epochs} epochs")
            traceback.print_exc()
            break

        loss_val = float(loss)
        loss_traj.append(loss_val)

        if x_holdout is not None and specification is not None and ts is not None:
            holdout_ros = _robustnesses(
                model, x_holdout, ts, specification, **specification_kwargs
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "holdout_mean_robustness": float(jnp.mean(holdout_ros)),
                        "holdout_frac_satisfied": float(jnp.mean(holdout_ros >= 0.0)),
                    },
                    commit=False,
                )

        if wandb.run is not None:
            wandb.log(
                {
                    "loss": loss_val,
                    "grad_mag": float(grad_mag),
                    "delta_mag": float(delta_mag),
                }
            )

        if early_stop is not None and loss_val <= early_stop:
            if verbose:
                tqdm.write(
                    f"Early stopping: loss reached {loss_val:.6g} <= "
                    f"{float(early_stop):.6g} at epoch {epoch_idx + 1}."
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
                elem_loss_fn=elem_loss_fn,
                boundary_xs=boundary_xs,
            )

        pbar.set_description(
            f"Loss: {loss:.4f}, Delta Mag: {delta_mag:.4f}, Grad Mag: {grad_mag:.4f}"
        )
    else:
        if verbose:
            tqdm.write(f"Reached the end of the {epochs} epochs")

    return model, loss_traj


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------


def run_suite(
    run_one_fn: Callable,
    n_seeds: int,
    *,
    key: jax.Array,
) -> dict:
    """Run ``run_one_fn(seed_idx, key)`` for n_seeds and collect results."""
    keys = jax.random.split(key, n_seeds)
    info = {}
    for i, ki in enumerate(keys):
        model, loss_traj = run_one_fn(i, ki)
        info[str(i)] = [len(loss_traj), loss_traj[-1] if loss_traj else float("inf")]
    return info
