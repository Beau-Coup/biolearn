"""Training script for the HK25 fast_produce specification using BioGNN."""

import dataclasses
from typing import List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tyro
from training_suite import run_suite, train

import wandb
from biolearn.losses import *
from biolearn.losses.slack_relu import SlackModel
from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
from biolearn.specifications.hk25 import fast_produce

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

# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------

DEFAULT_GRAPH = [
    (0, 1, EdgeType.Activation),  # x1 -> x2
    (0, 3, EdgeType.Inhibition),  # x1 |- x4
    (3, 1, EdgeType.Activation),  # x4 -> x2
    (3, 2, EdgeType.Inhibition),  # x4 |- x3
    (4, 1, EdgeType.Activation),  # x5 -> x2
    (4, 2, EdgeType.Activation),  # x5 -> x3
    (4, 5, EdgeType.Activation),  # x5 -> x6
]

# ---------------------------------------------------------------------------
# Trajectory builder — y_trace IS the (T, 6) trajectory
# ---------------------------------------------------------------------------


def _hk25_traj_fn(x0, y_trace):
    """For hk25, the full simulation output is already the trajectory."""
    return y_trace


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HK25Config:
    """Train HK25 fast_produce specification with a BioGNN model."""

    loss: LossType
    """Loss function to use for training."""
    epochs: int = 1000
    """Number of training epochs."""
    lr: float = 0.05
    """Learning rate."""
    n_seeds: int = 3
    """Number of random seeds."""
    n_samples: int = 256
    """Number of 6D initial conditions."""
    seed: int = 0
    """Random seed."""
    wandb_project: str = "biolearn-hk25"
    """Weights & Biases project name."""
    log_interval: int = 100
    """Log to wandb every N epochs."""
    early_stop: Optional[float] = None
    """Stop when loss <= this value."""
    semantics: Literal["dgmsr", "smooth", "classical", "agm"] = "dgmsr"
    """PySTL robustness semantics."""
    dgmsr_p: int = 3
    """D-GMSR 'p' parameter."""
    smooth_temperature: float = 1.0
    """Temperature for smooth semantics."""
    hill_coefficient: float = 2.0
    """Hill coefficient for BioGNN edges."""
    n_holdout: int = 0
    """Holdout points to track per epoch (0 = disabled)."""


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------


def _make_hk25_loss_fn(
    loss_name: str,
    ts: jax.Array,
    *,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    """Create loss function for hk25, injecting _hk25_traj_fn."""
    spec_kwargs = dict(
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
    )
    traj_kwargs = dict(traj_fn=_hk25_traj_fn)

    if loss_name == "relu":
        loss_fn = make_relu_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "sigmoid":
        loss_fn = make_sigmoid_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "silu":
        loss_fn = make_silu_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "logsumexp":
        loss_fn = make_logsumexp_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "softmax":
        loss_fn = make_softmax_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "leaky_relu":
        loss_fn = make_leaky_relu_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "elu":
        loss_fn = make_elu_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = None
    elif loss_name == "slack_relu":
        loss_fn = slack_relu_ic_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = SlackModel
    elif loss_name == "softrelu":
        loss_fn = make_softrelu_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = SlackModel
    elif loss_name == "slack_softmax":
        loss_fn = slack_softmax_loss(
            specification=fast_produce, ts=ts, **spec_kwargs, **traj_kwargs
        )
        wrap_model = SlackModel
    else:
        raise ValueError(f"Unknown loss function: {loss_name!r}")

    return loss_fn, wrap_model


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def train_model_hk25(
    biosyst: BioGnnModel,
    x_train: jax.Array,
    loss_name: str,
    *,
    epochs: int = 1000,
    lr: float = 0.05,
    log_interval: int = 0,
    early_stop: Optional[float] = None,
    n_holdout: int = 0,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
    key: jax.Array,
) -> Tuple[BioGnnModel, List[float]]:
    ts = jnp.arange(0, 25, 1.0)

    holdout_key, train_key = jax.random.split(key)
    x_holdout = (
        jax.random.uniform(holdout_key, (n_holdout, 6)) if n_holdout > 0 else None
    )

    loss_fn, wrap_model = _make_hk25_loss_fn(
        loss_name,
        ts,
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
    )

    trainable = wrap_model(biosyst) if wrap_model is not None else biosyst

    lr_schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=600,
        decay_rate=0.5,
        staircase=True,
    )
    optimizer = optax.adabelief(learning_rate=lr_schedule)

    spec_kwargs = dict(
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
    )

    try:
        trained_model, loss_traj = train(
            trainable,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=x_train.shape[0],
            x_train=x_train,
            log_interval=log_interval,
            ts=ts,
            specification=fast_produce,
            specification_kwargs=spec_kwargs,
            early_stop=early_stop,
            x_holdout=x_holdout,
            key=train_key,
        )
        if wrap_model is not None:
            trained_syst = trained_model.model
        else:
            trained_syst = trained_model
    except KeyboardInterrupt:
        trained_syst = biosyst
        loss_traj = [float(jnp.inf)]

    return trained_syst, loss_traj


def run_training_hk25(
    loss_name: str,
    x_train: jax.Array,
    epochs: int,
    lr: float,
    wandb_project: Optional[str],
    seed_idx: int = 0,
    log_interval: int = 0,
    early_stop: Optional[float] = None,
    n_holdout: int = 0,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
    hill_coefficient: float = 2.0,
    *,
    key: jax.Array,
) -> Tuple[BioGnnModel, List[float]]:
    run_name = f"hk25_{loss_name}_seed{seed_idx}"
    wandb.init(
        entity="biolearn",
        project=wandb_project,
        name=run_name,
        config={
            "loss": loss_name,
            "epochs": epochs,
            "lr": lr,
            "seed_idx": seed_idx,
            "hill_coefficient": hill_coefficient,
        },
    )

    model_key, train_key = jax.random.split(key)
    gnn = BioGNN(DEFAULT_GRAPH, hill_coefficient)
    biosyst = BioGnnModel(model=gnn)

    trained, loss_traj = train_model_hk25(
        biosyst,
        x_train,
        loss_name,
        epochs=epochs,
        lr=lr,
        log_interval=log_interval,
        early_stop=early_stop,
        n_holdout=n_holdout,
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
        key=train_key,
    )

    wandb.finish()
    return trained, loss_traj


def main(cfg: HK25Config):
    key = jax.random.PRNGKey(cfg.seed)
    data_key, suite_key = jax.random.split(key)

    x_train = jax.random.uniform(data_key, (cfg.n_samples, 6))
    print(f"Dataset shape: {x_train.shape}")

    def run_one(seed_idx, ki):
        return run_training_hk25(
            loss_name=cfg.loss,
            x_train=x_train,
            epochs=cfg.epochs,
            lr=cfg.lr,
            wandb_project=cfg.wandb_project,
            seed_idx=seed_idx,
            log_interval=cfg.log_interval,
            early_stop=cfg.early_stop,
            n_holdout=cfg.n_holdout,
            semantics=cfg.semantics,
            dgmsr_p=cfg.dgmsr_p,
            smooth_temperature=cfg.smooth_temperature,
            hill_coefficient=cfg.hill_coefficient,
            key=ki,
        )

    info = run_suite(run_one, cfg.n_seeds, key=suite_key)
    print("Results:", info)


if __name__ == "__main__":
    main(tyro.cli(HK25Config))
