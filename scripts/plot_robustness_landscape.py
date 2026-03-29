"""Visualize the robustness landscape of a model over parameter space.

Supports all system types: nfc, hill, quadrotor, laub.

Two visualization modes:
  - random_dirs: Li et al. 2018-style random normalized direction planes
  - coord_slice: sweep two specific parameter indices

Usage:
  uv run python -m scripts.plot_robustness_landscape --help
  uv run python -m scripts.plot_robustness_landscape --system nfc --grid-size 11 --n-test 8
  uv run python -m scripts.plot_robustness_landscape --system hill --grid-size 5 --n-test 4
  uv run python -m scripts.plot_robustness_landscape --system quadrotor --grid-size 5 --n-test 4
  uv run python -m scripts.plot_robustness_landscape --mode coord_slice --param-i 0 --param-j 6
  uv run python -m scripts.plot_robustness_landscape --model-type buffer --freeze-mlp
  uv run python -m scripts.plot_robustness_landscape --checkpoint results/run/best_model.eqx
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

from biolearn.models.base import BioModel, SimulateConfig
from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
from biolearn.models.laub import LLModel, LaubLoomis
from biolearn.models.nfc import NFC, MoormanNFC
from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.hk25 import FastProduce
from biolearn.specifications.laub import StableConverge
from biolearn.specifications.quadrotor import HeightMaintain
from biolearn.specifications.ss_classification import PhiXorFast

# ---------------------------------------------------------------------------
# MLP and BufferModel (copied from train_scratch.py to avoid __init__.py issues)
# ---------------------------------------------------------------------------


class MLP(eqx.Module):
    layers: list

    def __init__(self, key: jax.Array, state_size: int, hidden_size: int = 32):
        k1, k2, k3 = jr.split(key, 3)
        out_layer = eqx.nn.Linear(hidden_size, state_size, key=k3)
        self.layers = [
            eqx.nn.Linear(state_size, hidden_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            jax.nn.relu,
            out_layer,
        ]

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class BufferModel(eqx.Module):
    """A module wrapper that adds a residual NN to the ODE."""

    nominal_model: BioModel
    residual_model: MLP
    slack: jax.Array

    def __init__(self, key: jax.Array, nominal: BioModel):
        self.nominal_model = nominal
        state_size = math.prod(nominal.shape)
        residual_model = MLP(key, state_size=state_size, hidden_size=64)
        self.residual_model = jax.tree_util.tree_map(
            lambda x: x * 1e-4 if eqx.is_inexact_array(x) else x,
            residual_model,
        )
        self.slack = jnp.array(0.01)

    def _step(self, t, y, args):
        mlp_out = self.residual_model(y.flatten()).reshape(y.shape)
        return self.nominal_model.diffrax_step(t, y, args) + mlp_out * y

    def simulate(
        self,
        x: jax.Array,
        ts: jax.Array,
        x_ts=None,
        config: SimulateConfig = SimulateConfig(),
    ):
        if isinstance(self.nominal_model, NFC):
            interp = NFC._handle_inputs(x, x_ts)
            args = (interp,)
            y0 = jnp.zeros(self.nominal_model.shape)
        else:
            args = None
            y0 = x

        solver = diffrax.Kvaerno5() if config.stiff else diffrax.Tsit5()

        stepsize_controller = diffrax.PIDController(
            pcoeff=0.3 if config.stiff else 0.0,
            icoeff=0.3 if config.stiff else 1.0,
            dcoeff=0,
            rtol=config.rtol,
            atol=config.atol,
            dtmax=config.max_stepsize,
            dtmin=1e-10,
        )
        term = diffrax.ODETerm(self._step)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=config.max_steps,
            throw=config.throw,
            args=args,
        )
        return solution.ys, solution


# ---------------------------------------------------------------------------
# Parameter flatten / unflatten
# ---------------------------------------------------------------------------


def flatten_params(model, filter_fn=eqx.is_inexact_array):
    """Extract trainable params as a single 1D array plus reconstruction info."""
    params, static = eqx.partition(model, filter_fn)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    sizes = [l.size for l in leaves]
    flat = jnp.concatenate([l.ravel() for l in leaves])
    return flat, (static, treedef, shapes, sizes)


def unflatten_params(flat, info):
    """Reconstruct model from a 1D parameter vector."""
    static, treedef, shapes, sizes = info
    splits = np.cumsum(sizes[:-1])
    arrays = jnp.split(flat, splits)
    leaves = [a.reshape(s) for a, s in zip(arrays, shapes)]
    params = jax.tree_util.tree_unflatten(treedef, leaves)
    return eqx.combine(params, static)


# ---------------------------------------------------------------------------
# Parameter name generation (for coordinate-slice axis labels)
# ---------------------------------------------------------------------------


def param_names(model, filter_fn=eqx.is_inexact_array):
    """Generate human-readable names for each scalar in the flat param vector."""
    params, _ = eqx.partition(model, filter_fn)
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    names = []
    for path, leaf in leaves_with_path:
        key_str = ".".join(str(k) for k in path)
        if leaf.ndim == 0:
            names.append(key_str)
        else:
            for idx in np.ndindex(leaf.shape):
                idx_str = "[" + ",".join(str(i) for i in idx) + "]"
                names.append(key_str + idx_str)
    return names


# ---------------------------------------------------------------------------
# Random direction generation (Li et al. 2018)
# ---------------------------------------------------------------------------


def generate_random_directions(key, theta_0, sizes):
    """Generate two filter-normalized orthogonal directions in param space."""
    D = theta_0.shape[0]
    k1, k2 = jr.split(key)
    d1_raw = jr.normal(k1, (D,))
    d2_raw = jr.normal(k2, (D,))

    def filter_normalize(d, theta_0):
        """Normalize direction per-leaf to match parameter norms."""
        splits = np.cumsum(sizes[:-1])
        d_parts = jnp.split(d, splits)
        t_parts = jnp.split(theta_0, splits)
        normed = []
        for dp, tp in zip(d_parts, t_parts):
            t_norm = jnp.linalg.norm(tp)
            d_norm = jnp.linalg.norm(dp)
            scale = jnp.where(d_norm > 1e-12, t_norm / d_norm, 1.0)
            normed.append(dp * scale)
        return jnp.concatenate(normed)

    d1 = filter_normalize(d1_raw, theta_0)

    # Gram-Schmidt orthogonalization
    d2_orth = d2_raw - (jnp.dot(d2_raw, d1) / jnp.dot(d1, d1)) * d1
    d2 = filter_normalize(d2_orth, theta_0)

    return d1, d2


# ---------------------------------------------------------------------------
# Per-system configuration
# ---------------------------------------------------------------------------

HILL_GRAPH = [
    (0, 1, EdgeType.Activation),  # x1 -> x2
    (2, 3, EdgeType.Inhibition),  # x3 -| x4
    (3, 0, EdgeType.Inhibition),  # x4 -| x1
    (3, 1, EdgeType.Activation),  # x4 -> x2
    (4, 1, EdgeType.Activation),  # x5 -> x2
    (4, 2, EdgeType.Activation),  # x5 -> x3
    (4, 5, EdgeType.Activation),  # x5 -> x6
]


def setup_system(system: str, key: jax.Array):
    """Return (model, spec, traj_fn, sim_cfg, ts, low, high, is_nfc) for a system."""
    if system == "nfc":
        model = MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=key)
        spec = PhiXorFast()
        traj_fn = ss_to_traj_xor
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=False,
            max_steps=int(3e4),
            rtol=1e-4,
            atol=1e-5,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts = jnp.arange(0.0, 20.0, 1.0)
        low = jnp.array([0.1, 0.1])
        high = jnp.array([0.9, 0.9])
        is_nfc = True
    elif system == "hill":
        model = BioGnnModel(BioGNN(key, HILL_GRAPH, 2.0))
        spec = FastProduce()
        traj_fn = ss_to_traj_hill
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=False,
            max_steps=int(2e4),
            rtol=1e-3,
            atol=1e-4,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts = jnp.arange(0.0, 15.0, 1.0)
        low = jnp.array([0.0, 0.0, 0.01, 0.01, 0.99, 0.99])
        high = jnp.array([0.2, 0.2, 0.04, 0.04, 1.0, 1.0])
        is_nfc = False
    elif system == "quadrotor":
        model = QuadModel(Quadrotor(key))
        spec = HeightMaintain()
        traj_fn = ss_to_traj_q
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=False,
            throw=False,
            max_steps=int(3e4),
            rtol=1e-3,
            atol=1e-4,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts = jnp.arange(0.0, 5.0, 1.0)
        low = jnp.array(
            [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, -0.02, 0.0, -0.02, 0.0, -0.02]
        )
        high = jnp.array(
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02]
        )
        is_nfc = False
    elif system == "laub":
        model = LLModel(LaubLoomis(key))
        spec = StableConverge()
        traj_fn = ss_to_traj_laub
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=False,
            max_steps=4096,
            rtol=1e-3,
            atol=1e-4,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts = jnp.arange(0.0, 20.0, 0.5)
        center = jnp.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
        low = center - 0.1
        high = center + 0.1
        is_nfc = False
    else:
        raise ValueError(f"Unknown system: {system}")
    return model, spec, traj_fn, sim_cfg, ts, low, high, is_nfc


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------


def make_test_ics(
    n_test: int, key: jax.Array, low: jax.Array, high: jax.Array, is_nfc: bool
) -> jax.Array:
    """Create test initial conditions.

    For NFC: 2D meshgrid (n_test per axis, total = n_test^2).
    For hill/quadrotor: random uniform sampling (total = n_test^2).
    """
    if is_nfc:
        xs = jnp.linspace(low[0], high[0], n_test)
        ys = jnp.linspace(low[1], high[1], n_test)
        g1, g2 = jnp.meshgrid(xs, ys)
        return jnp.stack([g1.ravel(), g2.ravel()], axis=-1)
    else:
        dim = low.shape[0]
        return jr.uniform(key, (n_test**2, dim), minval=low, maxval=high)


def ss_to_traj_xor(y_trace, x):
    """Convert NFC output to (T, 3) trajectory for PhiXorFast."""
    y_traj = y_trace[..., -1, 0][..., None]  # (T, 1) — last node, first species
    x_traj = jnp.ones_like(y_traj) * x[None, :]  # (T, 2)
    return jnp.concatenate([x_traj, y_traj], axis=-1)  # (T, 3)


def ss_to_traj_hill(y_trace, x):
    """Hill model trajectory is the state directly: (T, 6)."""
    return y_trace


def ss_to_traj_q(y_trace, _):
    """Quadrotor trajectory extracts height and vertical rate: (T, 2)."""
    return y_trace[..., 4:6]


def ss_to_traj_laub(y_trace, _):
    """Laub-Loomis trajectory is the full 7-D state."""
    return y_trace


def build_eval_fn(info, x_test, spec, traj_fn, sim_cfg, ts):
    """Build a JIT-compiled function: flat_params -> (N,) robustness array."""

    @jax.jit
    def eval_rhos(flat_params):
        model = unflatten_params(flat_params, info)
        sim = partial(model.simulate, ts=ts, config=sim_cfg)

        def run_one(x0):
            y_trace, _ = sim(x=x0)
            traj = traj_fn(y_trace, x0)
            rho = spec.evaluate(traj)
            return jnp.where(jnp.isfinite(rho), rho, -1.0)

        return jax.vmap(run_one)(x_test)

    return eval_rhos


def aggregate(rhos, metric: str):
    """Aggregate per-IC robustness into a single scalar."""
    if metric == "sat_frac":
        return jnp.mean(rhos > 0)
    elif metric == "mean":
        return jnp.mean(rhos)
    elif metric == "min":
        return jnp.min(rhos)
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "sat_frac": r"Sat.\ fraction",
    "mean": r"$\bar{\rho}$",
    "min": r"$\min \rho$",
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def plot_landscape(
    alpha_vals,
    beta_vals,
    grid,
    metric,
    label_x,
    label_y,
    title,
    output_path,
    show,
):
    A, B = np.meshgrid(alpha_vals, beta_vals, indexing="ij")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    if metric == "sat_frac":
        levels = np.linspace(0, 1, 31)
        boundary = [0.5]
    else:
        vmax = (
            np.nanmax(np.abs(grid[np.isfinite(grid)]))
            if np.any(np.isfinite(grid))
            else 1.0
        )
        vmax = min(vmax, 0.5)
        levels = np.linspace(-vmax, vmax, 31)
        boundary = [0.0]

    cf = ax.contourf(A, B, grid, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(A, B, grid, levels=boundary, colors=["#c0392b"], linewidths=1.5)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(METRIC_LABELS.get(metric, metric))
    ax.plot(0, 0, "k*", markersize=8, zorder=10)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class PlotArgs:
    system: Literal["nfc", "hill", "quadrotor", "laub"] = "nfc"
    """System type to analyze."""
    seed: int = 42
    """Random seed for model initialization."""
    checkpoint: str | None = None
    """Path to .eqx checkpoint. Loads trained parameters instead of random init."""
    mode: Literal["random_dirs", "coord_slice"] = "random_dirs"
    """Visualization mode."""
    grid_size: int = 31
    """Grid points per axis."""
    alpha_range: float = 1.0
    """Half-width of sweep (log-space units for coord_slice, normalized for random_dirs)."""
    param_i: int = 0
    """First parameter index (coord_slice mode)."""
    param_j: int = 1
    """Second parameter index (coord_slice mode)."""
    dir_seed: int = 0
    """Seed for random directions."""
    n_test: int = 16
    """IC grid points per axis (total ICs = n_test^2)."""
    metric: Literal["mean", "sat_frac", "min"] = "sat_frac"
    """Aggregation metric over initial conditions."""
    model_type: Literal["nominal", "buffer"] = "nominal"
    """Model type: bare nominal model or BufferModel wrapping it."""
    freeze_mlp: bool = False
    """BufferModel only: sweep only nominal parameters, keep MLP fixed."""
    buffer_seed: int = 99
    """Seed for BufferModel MLP init."""
    output: str = "figures/robustness_landscape.pdf"
    """Output file path."""
    show: bool = False
    """Call plt.show()."""


def main():
    args = tyro.cli(PlotArgs)
    key = jr.key(args.seed)
    key, subkey = jr.split(key)

    # --- System setup ---
    nominal_model, spec, traj_fn, sim_cfg, ts, low, high, is_nfc = setup_system(
        args.system, subkey
    )

    # --- Model setup (optionally wrap in BufferModel, optionally load checkpoint) ---
    if args.checkpoint is not None:
        if args.model_type == "buffer":
            template = BufferModel(jr.key(args.buffer_seed), nominal_model)
        else:
            template = nominal_model
        model = eqx.tree_deserialise_leaves(args.checkpoint, template)
        print(f"Loaded checkpoint from {args.checkpoint}")
    elif args.model_type == "buffer":
        model = BufferModel(jr.key(args.buffer_seed), nominal_model)
    else:
        model = nominal_model

    # --- Filter for parameter sweep ---
    if args.model_type == "buffer" and args.freeze_mlp:
        # Only sweep nominal model parameters, keep MLP fixed
        def nominal_filter(model):
            mask = jax.tree_util.tree_map(lambda _: False, model)
            nom_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model.nominal_model)
            return eqx.tree_at(lambda m: m.nominal_model, mask, nom_mask)

        filter_fn = nominal_filter(model)
    else:
        filter_fn = eqx.is_inexact_array

    # --- Flatten parameters ---
    theta_0, info = flatten_params(model, filter_fn)
    _, _, _, sizes = info
    D = theta_0.shape[0]
    names = param_names(model, filter_fn)
    print(f"Parameter space dimension: {D}")

    # --- Test ICs ---
    key, subkey = jr.split(key)
    x_test = make_test_ics(args.n_test, subkey, low, high, is_nfc)
    print(f"Test ICs: {x_test.shape[0]} points, dim={x_test.shape[1]}")

    # --- Evaluation function ---
    eval_fn = build_eval_fn(info, x_test, spec, traj_fn, sim_cfg, ts)

    # --- Grid setup ---
    alphas = np.linspace(-args.alpha_range, args.alpha_range, args.grid_size)
    betas = np.linspace(-args.alpha_range, args.alpha_range, args.grid_size)

    if args.mode == "random_dirs":
        d1, d2 = generate_random_directions(jr.key(args.dir_seed), theta_0, sizes)
        label_x = r"$\epsilon_1$"
        label_y = r"$\epsilon_2$"
    else:
        assert args.param_i < D and args.param_j < D, f"param indices must be < {D}"
        d1 = None
        d2 = None
        label_x = rf"$\theta_{{{args.param_i}}}$ ({names[args.param_i]})"
        label_y = rf"$\theta_{{{args.param_j}}}$ ({names[args.param_j]})"

    # --- Grid sweep ---
    robustness_grid = np.full((args.grid_size, args.grid_size), np.nan)
    total = args.grid_size

    pbar = tqdm(total=total, desc="Sweep")

    for i, alpha in enumerate(alphas):
        if args.mode == "random_dirs":
            theta = theta_0 + alpha * d1 + betas[:, None] * d2  # pyright: ignore
        else:
            theta = jnp.tile(theta_0, (args.grid_size, 1))
            theta = theta.at[:, args.param_i].set(theta_0[args.param_i] + alpha)
            theta = theta.at[:, args.param_j].set(theta_0[args.param_j] + betas)

        try:
            rhos = jax.vmap(eval_fn)(theta)
            robustness_grid[i] = jax.vmap(partial(aggregate, metric=args.metric))(rhos)
        except Exception:
            robustness_grid[i] = np.nan
        pbar.update(1)
    pbar.close()

    # --- Plot ---
    system_names = {"nfc": "NFC", "hill": "Hill", "quadrotor": "Quadrotor", "laub": "Laub-Loomis"}
    title = rf"Robustness landscape -- {system_names[args.system]} ({D}D)"

    plot_landscape(
        alphas,
        betas,
        robustness_grid,
        args.metric,
        label_x,
        label_y,
        title,
        args.output,
        args.show,
    )


if __name__ == "__main__":
    main()
