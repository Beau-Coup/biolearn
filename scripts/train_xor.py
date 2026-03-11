import dataclasses
import os
from typing import Callable, List, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tyro

import wandb
from biolearn import (
    NFC,
    BioModel,
    MoormanNFC,
)
from biolearn.losses import *
from biolearn.losses.slack_relu import SlackModel
from biolearn.specifications.ss_classification import phi_xor_fast, phi_xor_ss
from scripts.training_suite import (
    run_suite,
    train,
)

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
    early_stop: Optional[float] = None
    """Stop early when the scalar loss is <= this value."""
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
    n_holdout: int = 0
    """Number of holdout initial conditions to track per epoch (0 = disabled)."""
    boundary_frac: float = 0.0
    """Fraction of samples drawn from domain boundary edges (0 = disabled)."""


class ClassModel(BioModel):
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


def create_dataset(n_samples: int, max_val=1.0, boundary_frac: float = 0.0, key=None):
    """
    Create a 2D grid dataset for training.
    The dataset is created by generating a grid of points in the range [0, max_val]
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
    boundary_frac: float = 0.0,
):
    """Create the loss function and optional model wrapper for the given loss type."""
    # Build integral kwargs when MC sampling is requested.
    integral_kwargs = {}
    if integral:
        assert key is not None, "key is required when integral=True"
        n_boundary_points = round(n_points * boundary_frac)
        integral_kwargs = dict(
            domain=BoxDomain(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
            n_points=n_points,
            key=key,
            n_boundary_points=n_boundary_points,
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
    early_stop: Optional[float] = None,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
    lr: float = 0.05,
    epochs: int = 1200,
    show: bool = False,
    integral: bool = False,
    n_points: int = 128,
    log_interval: int = 0,
    n_holdout: int = 0,
    boundary_frac: float = 0.0,
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

    bvis_key, holdout_key, loss_key, key = jax.random.split(key, 4)

    x_boundary_plot = None
    if boundary_frac > 0:
        n_bvis_per_edge = 10
        bk1, bk2, bk3, bk4 = jax.random.split(bvis_key, 4)
        t1 = jax.random.uniform(bk1, (n_bvis_per_edge,))
        t2 = jax.random.uniform(bk2, (n_bvis_per_edge,))
        t3 = jax.random.uniform(bk3, (n_bvis_per_edge,))
        t4 = jax.random.uniform(bk4, (n_bvis_per_edge,))
        z = jnp.zeros(n_bvis_per_edge)
        o = jnp.ones(n_bvis_per_edge)
        x_boundary_plot = jnp.concatenate(
            [
                jnp.stack([t1, z], axis=1),
                jnp.stack([t2, o], axis=1),
                jnp.stack([z, t3], axis=1),
                jnp.stack([o, t4], axis=1),
            ],
            axis=0,
        )

    if n_holdout > 0:
        if boundary_frac > 0:
            n_holdout_boundary = round(n_holdout * boundary_frac)
            n_holdout_interior = n_holdout - n_holdout_boundary
            int_key, bnd_key = jax.random.split(holdout_key)
            x_holdout_int = jax.random.uniform(int_key, (n_holdout_interior, 2))
            if n_holdout_boundary > 0:
                n_hb_per_edge = n_holdout_boundary // 4
                hbk1, hbk2, hbk3, hbk4 = jax.random.split(bnd_key, 4)
                th1 = jax.random.uniform(hbk1, (n_hb_per_edge,))
                th2 = jax.random.uniform(hbk2, (n_hb_per_edge,))
                th3 = jax.random.uniform(hbk3, (n_hb_per_edge,))
                th4 = jax.random.uniform(hbk4, (n_hb_per_edge,))
                zh = jnp.zeros(n_hb_per_edge)
                oh = jnp.ones(n_hb_per_edge)
                x_holdout_bnd = jnp.concatenate(
                    [
                        jnp.stack([th1, zh], axis=1),
                        jnp.stack([th2, oh], axis=1),
                        jnp.stack([zh, th3], axis=1),
                        jnp.stack([oh, th4], axis=1),
                    ],
                    axis=0,
                )
                x_holdout = jnp.concatenate([x_holdout_int, x_holdout_bnd], axis=0)
            else:
                x_holdout = x_holdout_int
        else:
            x_holdout = jax.random.uniform(holdout_key, (n_holdout, 2))
    else:
        x_holdout = None

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
        boundary_frac=boundary_frac,
    )

    trainable = wrap_model(biosyst) if wrap_model is not None else biosyst

    learning_rate_schedule = optax.exponential_decay(
        init_value=lr,  # Starting learning rate
        transition_steps=600,  # Number of steps before each decay
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
            elem_loss_fn=_get_elementwise_loss(loss_name, trainable),
            early_stop=early_stop,
            x_holdout=x_holdout,
            boundary_xs=x_boundary_plot,
            key=key,
        )
        if loss_name == "slack_relu":
            print(f"SlackReLU optimized margin C={float(trained_model.slack):.6g}")
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
    n_holdout: int = 0,
    boundary_frac: float = 0.0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    early_stop: Optional[float] = None,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    arch_str = "-".join(str(l) for l in layer_sizes)
    integral_tag = "integral" if integral else "batch"
    run_name = f"{loss_name}_{integral_tag}_arch{arch_str}_seed{seed_idx}"

    wandb.init(
        entity="biolearn",
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
        early_stop=early_stop,
        semantics=semantics,
        dgmsr_p=dgmsr_p,
        smooth_temperature=smooth_temperature,
        epochs=epochs,
        lr=lr,
        show=show,
        integral=integral,
        n_points=n_points,
        log_interval=log_interval,
        n_holdout=n_holdout,
        boundary_frac=boundary_frac,
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
    n_holdout: int = 0,
    boundary_frac: float = 0.0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    early_stop: Optional[float] = None,
    semantics: str = "dgmsr",
    dgmsr_p: int = 3,
    smooth_temperature: float = 1.0,
):
    all_params = []

    def _run_one(seed_idx, ki):
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
            seed_idx=seed_idx,
            log_interval=log_interval,
            n_holdout=n_holdout,
            boundary_frac=boundary_frac,
            w=5.0,
            k1=0.8,
            k2=0.8,
            spec=spec,
            early_stop=early_stop,
            semantics=semantics,
            dgmsr_p=dgmsr_p,
            smooth_temperature=smooth_temperature,
        )
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(trained_sim, eqx.is_inexact_array)
        )
        flat_params = jnp.concatenate([l.flatten() for l in leaves])
        print(flat_params)
        all_params.append(flat_params)
        return trained_sim, loss_traj

    info = run_suite(_run_one, n, key=key)

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
        wandb.init(entity="biolearn", project=wandb_project, name=run_name)
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
    n_holdout: int = 0,
    boundary_frac: float = 0.0,
    *,
    key,
    spec: SpecType = "phi_xor_fast",
    early_stop: Optional[float] = None,
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
            n_holdout=n_holdout,
            boundary_frac=boundary_frac,
            spec=spec,
            early_stop=early_stop,
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

    dataset_key, key = jax.random.split(key)
    x_train, _ = create_dataset(
        n_samples=cfg.n_samples,
        max_val=1.0,
        boundary_frac=cfg.boundary_frac,
        key=dataset_key,
    )

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
        n_holdout=cfg.n_holdout,
        n=cfg.n_seeds,
        boundary_frac=cfg.boundary_frac,
        key=key,
        spec=cfg.spec,
        early_stop=cfg.early_stop,
        semantics=cfg.semantics,
        dgmsr_p=cfg.dgmsr_p,
        smooth_temperature=cfg.smooth_temperature,
    )

    import code

    code.interact(local=locals())


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
