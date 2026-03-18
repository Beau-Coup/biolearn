"""Shared training config and loss creation for biolearn experiment scripts."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import dataclasses
from typing import Literal, Optional

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import jax.numpy as jnp
import optax
from training_suite import adam, train

import wandb
from biolearn.losses import (
    BoxDomain,
    SlackModel,
    activations,
    make_loss,
    make_softrelu_loss,
    slack_relu_ic_loss,
)
from biolearn.specifications.common import BaseSpec
from biolearn.tasks import Task
from biolearn.utils import sample_hypercube_faces

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


@dataclasses.dataclass
class TrainConfig:
    """Train XOR classification with different loss functions."""

    loss: LossType
    """Loss function to use for training."""
    spec: SpecType = "phi_xor_fast"
    """STL specification to evaluate on trajectories."""
    early_stop: Optional[float] = None
    """Stop early when the scalar loss is <= this value."""
    optimizer: Literal["adam", "adabelief"] = "adabelief"
    """Which optimizer to use for training."""
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
    """Number of training points sampled uniformly from the task domain."""
    show: bool = False
    """Show plots after training."""
    save: bool = True
    """Save results to disk."""
    integral: bool = False
    """Use Monte-Carlo integral loss: each loss call samples its own points
    from the task domain instead of using the fixed training dataset."""
    n_points: int = 128
    """Number of interior points sampled per loss evaluation when --integral is set."""
    wandb_project: str = "biolearn"
    """Weights & Biases project name."""
    log_interval: int = 100
    """Log contour plots to wandb every N epochs."""
    n_holdout: int = 0
    """Number of holdout initial conditions to track per epoch (0 = disabled)."""
    boundary_frac: float = 0.0
    """Fraction of points placed on domain boundary faces via
    sample_hypercube_faces. Applies to the training dataset (non-integral)
    and to each loss evaluation (--integral). 0 disables boundary sampling."""
    batch_size: int = 64
    """Mini-batch size for shuffled training."""


def make_task_loss(loss_name, spec, ts, **kwargs):
    """Create a loss function from a name and a specification callable.

    Returns ``(loss_fn, wrap_model_cls_or_None)``.

    Simple activation losses use the ``activations`` dict and ``make_loss``.
    Slack-based losses return ``SlackModel`` as the wrapper class.
    """
    if loss_name in activations:
        return make_loss(activations[loss_name], spec, ts, **kwargs), None
    if loss_name == "slack_relu":
        return slack_relu_ic_loss(specification=spec, ts=ts, **kwargs), SlackModel
    if loss_name == "slack_softmax":
        return slack_softmax_loss(specification=spec, ts=ts, **kwargs), SlackModel
    if loss_name == "softrelu":
        return make_softrelu_loss(specification=spec, ts=ts, **kwargs), SlackModel
    raise ValueError(f"Unknown loss: {loss_name!r}")


def create_dataset(n_samples: int, max_val=1.0, boundary_frac: float = 0.0, key=None):
    """Create a 2D grid dataset for training.

    Args:
        n_samples: number of samples to generate (should be a perfect square)
        max_val: maximum value for the grid points
        boundary_frac: fraction of samples to draw from boundary edges (0 = disabled)
        key: PRNG key required when boundary_frac > 0

    Returns:
        a tuple containing: the generated dataset (x_train)
            and the grid (grid_x, grid_y)
    """
    n_per_axis = int(jnp.sqrt(n_samples))
    x1 = jnp.linspace(0, max_val, n_per_axis)
    x2 = jnp.linspace(0, max_val, n_per_axis)
    grid_x, grid_y = jnp.meshgrid(x1, x2)
    x_train = jnp.column_stack((grid_x.ravel(), grid_y.ravel()))

    if boundary_frac > 0:
        assert key is not None, "key is required when boundary_frac > 0"
        n_per_edge = round(n_samples * boundary_frac) // 4
        k1, k2, k3, k4 = jax.random.split(key, 4)
        t1 = jax.random.uniform(k1, (n_per_edge,), maxval=max_val)
        t2 = jax.random.uniform(k2, (n_per_edge,), maxval=max_val)
        t3 = jax.random.uniform(k3, (n_per_edge,), maxval=max_val)
        t4 = jax.random.uniform(k4, (n_per_edge,), maxval=max_val)
        zeros = jnp.zeros(n_per_edge)
        ones = jnp.full(n_per_edge, max_val)
        boundary_pts = jnp.concatenate(
            [
                jnp.stack([t1, zeros], axis=1),  # bottom edge (y=0)
                jnp.stack([t2, ones], axis=1),  # top edge (y=max_val)
                jnp.stack([zeros, t3], axis=1),  # left edge (x=0)
                jnp.stack([ones, t4], axis=1),  # right edge (x=max_val)
            ],
            axis=0,
        )
        n_replace = boundary_pts.shape[0]
        x_train = jnp.concatenate([x_train[:-n_replace], boundary_pts], axis=0)

    return x_train, (grid_x, grid_y)


def train_one(
    task: Task,
    cfg: TrainConfig,
    spec: BaseSpec,
    seed_idx: int,
    key: jax.Array,
):
    """Train a single seed on any task."""
    task_name = type(task).__name__
    run_name = f"{task_name}_{cfg.loss}_seed{seed_idx}"

    wandb.init(
        entity="biolearn",
        project=cfg.wandb_project,
        name=run_name,
        config={
            "loss": cfg.loss,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "seed_idx": seed_idx,
        },
    )

    loss_key, train_key = jax.random.split(key)

    # Build loss kwargs (integral support)
    loss_kwargs: dict = {}
    if cfg.integral:
        n_boundary_points = round(cfg.n_points * cfg.boundary_frac)
        loss_kwargs.update(
            domain=BoxDomain(task.domain_low, task.domain_hi),
            n_points=cfg.n_points,
            key=loss_key,
            n_boundary_points=n_boundary_points,
        )

    loss_fn, wrap_cls = make_task_loss(
        cfg.loss, spec.evaluate, task.ts, traj_fn=task.traj_fn, **loss_kwargs
    )
    trainable = wrap_cls(task.model) if wrap_cls else task.model

    lr_schedule = optax.exponential_decay(
        init_value=cfg.lr,
        transition_steps=600,
        decay_rate=0.5,
        staircase=True,
    )
    if cfg.optimizer == "adam":
        optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        optimizer = optax.adabelief(learning_rate=lr_schedule)

    optimizer = adam(cfg.lr)

    # Generate training data from task domain
    dataset_key, boundary_key, train_key = jax.random.split(train_key, 3)
    x_train = jax.random.uniform(
        dataset_key,
        (cfg.n_samples, task.domain_low.shape[0]),
        minval=task.domain_low,
        maxval=task.domain_hi,
    )
    if cfg.boundary_frac > 0:
        n_boundary = round(cfg.n_samples * cfg.boundary_frac)
        d = task.domain_low.shape[0]
        n_faces = d * 2 ** (d - 1)
        n_per_face = max(1, n_boundary // n_faces)
        boundary_pts = sample_hypercube_faces(
            boundary_key,
            task.domain_low,
            task.domain_hi,
            n_per_face=n_per_face,
            max_k=1,
        )
        x_train = jnp.concatenate([x_train[: -boundary_pts.shape[0]], boundary_pts])

    # Build specification kwargs with traj_fn
    specification_kwargs = {"traj_fn": task.traj_fn}

    # Holdout set from the task's built-in eval set
    x_holdout = task.eval_set

    # Contour plot grid for 2D tasks only
    is_2d = task.domain_low.shape[0] == 2
    plot_xs = plot_grid = None
    if cfg.log_interval > 0 and is_2d:
        plot_xs, plot_grid = create_dataset(n_samples=cfg.n_samples)

    trained_model, loss_traj = train(
        trainable,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        x_train=x_train,
        early_stop=cfg.early_stop,
        log_interval=cfg.log_interval,
        plot_xs=plot_xs,
        plot_grid=plot_grid,
        ts=task.ts,
        specification=spec.evaluate,
        specification_kwargs=specification_kwargs,
        loss_name=cfg.loss,
        x_holdout=x_holdout,
        key=train_key,
    )

    if isinstance(trained_model, SlackModel):
        print(f"SlackReLU optimized margin C={float(trained_model.slack):.6g}")
        trained_model = trained_model.model

    # Post-training evaluation on the task's eval set
    task.model = trained_model
    eval_ros = task.evaluate_robustness()
    sat_rate = float(jnp.mean(eval_ros >= 0.0))
    mean_rob = float(jnp.mean(eval_ros))
    print(f"Eval: satisfaction={sat_rate:.3f}, mean_robustness={mean_rob:.4f}")
    if wandb.run is not None:
        wandb.log({"eval_sat_rate": sat_rate, "eval_mean_robustness": mean_rob})

    wandb.finish()
    return trained_model, loss_traj
