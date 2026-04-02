"""Visualize model trajectories colored by robustness, with spec overlay.

Supports all system types: nfc, hill, quadrotor, laub.

Usage:
  uv run python scripts/plot_trajectories.py --help
  uv run python scripts/plot_trajectories.py --system nfc --n-ics 20
  uv run python scripts/plot_trajectories.py --system quadrotor --n-ics 16
  uv run python scripts/plot_trajectories.py --system hill --n-ics 10
  uv run python scripts/plot_trajectories.py --system laub --n-ics 16
  uv run python scripts/plot_trajectories.py --system nfc --checkpoint results/run/best_model.eqx
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
from pathlib import Path
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from biolearn.models.base import BioModel, SimulateConfig
from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
from biolearn.models.laub import LaubLoomis, LLModel
from biolearn.models.nfc import NFC, MoormanNFC
from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.hk25 import FastProduce
from biolearn.specifications.laub import StableConverge
from biolearn.specifications.quadrotor import HeightMaintain
from biolearn.specifications.ss_classification import PhiXorFast

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)

# ---------------------------------------------------------------------------
# MLP and BufferModel (copied from train_scratch.py)
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

    def simulate(self, x, ts, x_ts=None, config=SimulateConfig()):
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
# Per-system configuration
# ---------------------------------------------------------------------------

HILL_GRAPH = [
    (0, 1, EdgeType.Activation),
    (2, 3, EdgeType.Inhibition),
    (3, 0, EdgeType.Inhibition),
    (3, 1, EdgeType.Activation),
    (4, 1, EdgeType.Activation),
    (4, 2, EdgeType.Activation),
    (4, 5, EdgeType.Activation),
]


def setup_system(system: str, key: jax.Array):
    """Return (model, spec, sim_cfg, ts_plot, ts_rob, low, high, is_nfc)."""
    if system == "nfc":
        model = MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=key)
        spec = PhiXorFast()
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=False,
            max_steps=4096,
            rtol=1e-4,
            atol=1e-5,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts_plot = jnp.arange(0.0, 20.0, 0.1)
        ts_rob = jnp.arange(0.0, 20.0, 1.0)
        low = jnp.array([0.1, 0.1])
        high = jnp.array([0.9, 0.9])
        is_nfc = True
    elif system == "hill":
        model = BioGnnModel(BioGNN(key, HILL_GRAPH, 2.0))
        spec = FastProduce()
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=True,
            throw=False,
            max_steps=4096,
            rtol=1e-5,
            atol=1e-6,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts_plot = jnp.arange(0.0, 5.5, 0.05)
        ts_rob = jnp.arange(0.0, 25.0, 1.0)
        low = jnp.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.9])
        high = jnp.array([0.4, 0.4, 0.4, 0.4, 1.0, 1.0])
        is_nfc = False
    elif system == "quadrotor":
        model = QuadModel(Quadrotor(key))
        spec = HeightMaintain()
        sim_cfg = SimulateConfig(
            to_ss=False,
            stiff=False,
            throw=False,
            max_steps=4096,
            rtol=1e-3,
            atol=1e-4,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts_plot = jnp.arange(0.0, 5.0, 0.02)
        ts_rob = jnp.arange(0.0, 5.0, 1.0)
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
        ts_plot = jnp.arange(0.0, 20.0, 0.1)
        ts_rob = jnp.arange(0.0, 20.0, 1.0)
        center = jnp.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
        low = center - 0.1
        high = center + 0.1
        is_nfc = False
    else:
        raise ValueError(f"Unknown system: {system}")
    return model, spec, sim_cfg, ts_plot, ts_rob, low, high, is_nfc


# ---------------------------------------------------------------------------
# Trajectory conversion
# ---------------------------------------------------------------------------


def ss_to_traj_xor(y_trace, x):
    y_traj = y_trace[..., -1, 0][..., None]
    x_traj = jnp.ones_like(y_traj) * x[None, :]
    return jnp.concatenate([x_traj, y_traj], axis=-1)


def ss_to_traj_hill(y_trace, x):
    return y_trace


def ss_to_traj_q(y_trace, _):
    return y_trace[..., 4:6]


def ss_to_traj_laub(y_trace, x):
    return y_trace


TRAJ_FNS = {
    "nfc": ss_to_traj_xor,
    "hill": ss_to_traj_hill,
    "quadrotor": ss_to_traj_q,
    "laub": ss_to_traj_laub,
}


# ---------------------------------------------------------------------------
# Spec overlay drawing (minimal gray dashed lines only)
# ---------------------------------------------------------------------------


_SHADE = dict(color="red", alpha=0.08, zorder=0, linewidth=0)
_BORDER = dict(color="red", alpha=0.3, lw=0.5, ls="--")


def _shade_above(ax, y):
    """Shade forbidden region above y, clipped to current ylim."""
    _, hi = ax.get_ylim()
    if hi > y:
        ax.axhspan(y, hi, **_SHADE)
        ax.axhline(y, **_BORDER)


def _shade_below(ax, y):
    """Shade forbidden region below y, clipped to current ylim."""
    lo, _ = ax.get_ylim()
    if lo < y:
        ax.axhspan(lo, y, **_SHADE)
        ax.axhline(y, **_BORDER)


def draw_spec_overlay_nfc(axes, ts):
    # err > 0.1 forbidden (must eventually stay below)
    _shade_above(axes[0], 0.1)


def draw_spec_overlay_quadrotor(axes, ts):
    t_end = float(ts[-1])
    ts_np = np.array(ts)
    # h > 1.4 forbidden always
    _shade_above(axes[0], 1.4)
    # h < 0.9 forbidden from t=1 onward
    mask = ts_np >= 1.0
    lo = axes[0].get_ylim()[0]
    axes[0].fill_between(ts_np, lo, 0.9, where=mask, **_SHADE)
    axes[0].hlines(0.9, xmin=1.0, xmax=t_end, **_BORDER)
    # |hdot| > 0.1 forbidden from t=3 onward
    mask3 = ts_np >= 3.0
    hi = axes[1].get_ylim()[1]
    lo = axes[1].get_ylim()[0]
    axes[1].fill_between(ts_np, 0.1, hi, where=mask3, **_SHADE)
    axes[1].fill_between(ts_np, lo, -0.1, where=mask3, **_SHADE)
    axes[1].hlines(0.1, xmin=3.0, xmax=t_end, **_BORDER)
    axes[1].hlines(-0.1, xmin=3.0, xmax=t_end, **_BORDER)


def draw_spec_overlay_hill(axes, ts):
    # x_i < 1.5 always for all 4 species (single axis now)
    _shade_above(axes[0], 1.5)


def draw_spec_overlay_laub(axes, ts):
    # x4 < 4.5 always
    _shade_above(axes[0], 4.5)
    axes[0].axhline(0.45, ls="-.", color="C1", lw=1.0, alpha=0.5)
    axes[0].axhline(0.25, ls="--", color="C1", lw=1.0)
    axes[0].axhline(0.45, ls="--", color="C2", lw=1.0, alpha=0.5)
    axes[0].axhline(0.65, ls="--", color="C2", lw=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_axes(axes, ts=None):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.locator_params(axis="y", nbins=4)
        if ts is not None:
            ax.set_xlim(float(ts[0]), float(ts[-1]))


_BOUNDS_STYLE = dict(color="black", lw=0.6, ls="-", alpha=0.8)


def _overlay_bounds(axes, csv_path: str, system: str):
    """Overlay upper/lower bounds from a CSV onto the plot axes."""
    df = pd.read_csv(csv_path)
    if system == "quadrotor":
        t = df["Time_s"].values
        for ax, lo_col, hi_col in [
            (axes[0], "Min_h", "Max_h"),
            (axes[1], "Min_h_dot", "Max_h_dot"),
        ]:
            if lo_col in df.columns:
                ax.plot(t, df[lo_col].values, **_BOUNDS_STYLE)
            if hi_col in df.columns:
                ax.plot(t, df[hi_col].values, **_BOUNDS_STYLE)
    elif system == "hill":
        t = df["Time"].values
        ax = axes[0]
        for j in range(6):
            lo_col, hi_col = f"Min_state_{j + 1}", f"Max_state_{j + 1}"
            style = {**_BOUNDS_STYLE}
            if lo_col in df.columns:
                ax.plot(t, df[lo_col].values, **style)
            if hi_col in df.columns:
                ax.plot(t, df[hi_col].values, **style)
    elif system == "laub":
        t = df["Time"].values
        ax = axes[0]
        for j in range(1, 8):
            lo_col, hi_col = f"Min_state_{j}", f"Max_state_{j}"
            style = {**_BOUNDS_STYLE}
            if lo_col in df.columns:
                ax.plot(t, df[lo_col].values, **style)
            if hi_col in df.columns:
                ax.plot(t, df[hi_col].values, **style)
    elif system == "nfc":
        t = df["Time_s"].values
        if "Min_err" in df.columns:
            axes[0].plot(t, df["Min_err"].values, **_BOUNDS_STYLE)
        if "Max_err" in df.columns:
            axes[0].plot(t, df["Max_err"].values, **_BOUNDS_STYLE)


# ---------------------------------------------------------------------------
# Per-system plot layout
# ---------------------------------------------------------------------------


def plot_nfc(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.4))
    axes = [ax]
    for i in range(len(all_traces)):
        y_trace, x0 = all_traces[i], all_ics[i]
        traj = ss_to_traj_xor(y_trace, x0)
        x_diff = traj[:, 1] - traj[:, 0]
        y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
        err = jnp.abs(y_true - traj[:, 2])
        color = cmap(norm(float(rhos[i])))
        ax.plot(np.array(ts), np.array(err), color=color, alpha=0.6, linewidth=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Error $|y - y^*|$")
    draw_spec_overlay_nfc(axes, ts)
    _clean_axes(axes, ts)
    return fig, axes


def plot_quadrotor(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, (ax_h, ax_v) = plt.subplots(
        2,
        1,
        figsize=(3.5, 2.0),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    axes = [ax_h, ax_v]
    traces = jnp.array(all_traces)
    print(f"Maximum across traces: {traces[:, :, 4].max()}")
    three = jnp.searchsorted(ts, jnp.array([3.0]))[0]
    print(f"Velocity across traces: {traces[:, three:, 5].max()}")
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        ax_h.plot(
            np.array(ts), np.array(y_trace[:, 4]), color=color, alpha=0.3, linewidth=0.4
        )
        ax_v.plot(
            np.array(ts), np.array(y_trace[:, 5]), color=color, alpha=0.3, linewidth=0.4
        )
    ax_h.set_ylabel(r"$h$ (m)")
    ax_h.set_ylim(top=2.0)
    ax_v.set_ylabel(r"$\dot{h}$ (m/s)")
    ax_v.set_xlabel("Time (s)")
    draw_spec_overlay_quadrotor(axes, ts)
    _clean_axes(axes, ts)
    ax_v.set_xticks(np.arange(0, 6, 1.0))
    return fig, axes


def _robustness_alpha(rho_val, rhos_arr):
    """Map robustness to alpha: more robust = more saturated."""
    rho_min, rho_max = float(rhos_arr.min()), float(rhos_arr.max())
    span = min(max(rho_max - rho_min, 1e-6), 1.0)
    frac = rho_val / rho_max
    # We're plotting positive robs
    return 0.15 + 0.65 * frac  # range [0.15, 0.85]


# Base colors for hill species (tab10 palette)
_HILL_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def plot_hill(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0))
    axes = [ax]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        alpha = _robustness_alpha(float(rhos[i]), rhos) * 0.8
        for j in range(6):
            ax.plot(
                np.array(ts),
                np.array(y_trace[:, j]),
                color=_HILL_COLORS[j],
                alpha=alpha,
                linewidth=0.5,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration")
    draw_spec_overlay_hill(axes, ts)
    _clean_axes(axes, ts)
    return fig, axes


_LAUB_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def plot_laub(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0))
    axes = [ax]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        alpha = _robustness_alpha(float(rhos[i]), rhos) * 0.7
        for j in range(7):
            ax.plot(
                np.array(ts),
                np.array(y_trace[:, j]),
                color=_LAUB_COLORS[j],
                alpha=alpha,
                linewidth=0.2,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration")
    draw_spec_overlay_laub(axes, ts)
    _clean_axes(list(axes), ts)
    return fig, list(axes)


PLOT_FNS = {
    "nfc": plot_nfc,
    "hill": plot_hill,
    "quadrotor": plot_quadrotor,
    "laub": plot_laub,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    system: Literal["nfc", "hill", "quadrotor", "laub"] = "nfc"
    """System type."""
    seed: int = 42
    """Random seed."""
    checkpoint: str | None = None
    """Path to .eqx checkpoint (BufferModel or nominal)."""
    model_type: Literal["nominal", "buffer"] = "nominal"
    """Model type."""
    buffer_seed: int = 99
    """Seed for BufferModel MLP init."""
    n_ics: int = 32
    """Number of initial conditions to sample."""
    bounds_csv: str | None = None
    """CSV with upper/lower bounds to overlay. Columns depend on system."""
    traj_csv_dir: str | None = None
    """Directory with per-state trajectory CSVs (e.g. matlab/). Overrides simulation."""
    output: str = "figures/trajectories.pdf"
    """Output file path."""
    show: bool = False
    """Call plt.show()."""
    parameter_vary: bool = False
    """Plot a variation over the parameters for a single initial condition."""
    perturb_scale: float = 0.3
    """Scale of Gaussian perturbation to model parameters (for --parameter-vary)."""
    ic_grid: bool = False
    """Plot robustness heatmap over a 2D grid of initial conditions."""
    ic_grid_size: int = 25
    """Number of grid points per axis for IC grid."""
    ic_grid_dims: tuple[int, int] = (4, 5)
    """State indices for the two IC grid axes (default: h, dh)."""


def perturb_model(model, key, scale):
    """Add Gaussian noise to all learnable parameters of a model."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(flat_params))
    noisy = [p + scale * jr.normal(k, p.shape) for p, k in zip(flat_params, keys)]
    noisy_params = jax.tree_util.tree_unflatten(tree_def, noisy)
    return eqx.combine(noisy_params, static)


def plot_ic_grid(points, rho_grid, dim_labels):
    """Plot robustness heatmap over 2D IC grid."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

    rho_min = rho_grid.min()
    rho_max = rho_grid.max()

    if rho_min < 0:
        norm = mcolors.TwoSlopeNorm(vmin=rho_min, vcenter=0.0, vmax=rho_max)
        colors = ["#e31a1c", "#ff7f00", "#33a02c"]
        cmap = mcolors.LinearSegmentedColormap.from_list("RedBlackGreen", colors)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=max(rho_max, 0.01))
        cmap = cm.Greens

    cf = ax.scatter(
        np.array(points[:, 0]),
        np.array(points[:, 1]),
        s=2.0,
        c=rho_grid,
        cmap=cmap,
        norm=norm,
    )
    # ax.contour(
    #     np.array(grid_h),
    #     np.array(grid_v),
    #     rho_grid,
    #     levels=[0.0],
    #     colors="k",
    #     linewidths=1.0,
    # )
    # fig.colorbar(cf, ax=ax, label="Robustness")
    # ax.set_xlabel(dim_labels[0])
    # ax.set_ylabel(dim_labels[1])
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.margins(0.01)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    return fig, [ax]


def load_csv_trajectories(
    csv_dir: str, system: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load per-state CSV files and assemble into full trajectories.

    Returns (ts, traces) where ts is shape (T,) and traces is a list of
    arrays each of shape (T, n_states).
    """
    if system == "hill":
        prefix, n_states = "hill_traj_state", 6
    elif system == "laub":
        prefix, n_states = "laubloomis_traj_state", 7
    else:
        raise ValueError(f"CSV trajectories not supported for system {system!r}")

    dfs = []
    for s in range(1, n_states + 1):
        path = Path(csv_dir) / f"{prefix}_{s}.csv"
        dfs.append(pd.read_csv(path))

    ts = dfs[0]["Time"].values
    n_traj = len([c for c in dfs[0].columns if c.startswith("traj_")])
    traces = []
    for j in range(1, n_traj + 1):
        cols = [df[f"traj_{j}"].values for df in dfs]
        traces.append(np.stack(cols, axis=-1))  # (T, n_states)
    return ts, traces


def main():
    args = tyro.cli(Args)
    key = jr.key(args.seed)
    key, subkey = jr.split(key)

    # --- Load from CSV or simulate ---
    if args.traj_csv_dir is not None:
        # Load pre-computed trajectories from CSV files
        ts_csv, all_traces_np = load_csv_trajectories(args.traj_csv_dir, args.system)
        ics = np.zeros((len(all_traces_np), all_traces_np[0].shape[-1]))  # dummy ICs

        # Setup spec and get ts_plot limit from setup_system
        _, spec, _, ts_plot_cfg, _, _, _, _ = setup_system(args.system, subkey)
        t_plot_max = float(ts_plot_cfg[-1])

        # Truncate CSV data to plotting time range
        plot_mask = ts_csv <= t_plot_max + 1e-9
        ts_plot = jnp.array(ts_csv[plot_mask])
        all_traces = [jnp.array(t[plot_mask]) for t in all_traces_np]

        # Use full CSV data (not truncated) for robustness evaluation at dt=1.0
        dt_csv = float(ts_csv[1] - ts_csv[0])
        step = max(1, round(1.0 / dt_csv))
        traj_fn = TRAJ_FNS[args.system]
        rhos = []
        for i in range(len(all_traces_np)):
            traj_rob = jnp.array(all_traces_np[i][::step])
            traj = traj_fn(traj_rob, ics[i])
            rho = spec.evaluate(traj)
            rho = float(jnp.where(jnp.isfinite(rho), rho, -1.0))
            rhos.append(rho)
        rhos = np.array(rhos)
    else:
        # --- System setup ---
        nominal_model, spec, sim_cfg, ts_plot, ts_rob, low, high, is_nfc = setup_system(
            args.system, subkey
        )

        # --- Model setup ---
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

        traj_fn = TRAJ_FNS[args.system]

        if args.ic_grid:
            # --- IC grid robustness heatmap ---
            model = perturb_model(model, subkey, args.perturb_scale)
            d0, d1 = args.ic_grid_dims
            center = (high + low) / 2
            lo = jnp.array([low[d0], low[d1]])
            hi = jnp.array([high[d0], high[d1]])
            rho_grid = np.zeros(args.ic_grid_size)
            points = jr.uniform(key, (args.ic_grid_size, 2), minval=lo, maxval=hi)
            for i, p in enumerate(points):
                x0 = center.at[d0].set(p[0]).at[d1].set(p[1])
                y_rob, _ = model.simulate(x=x0, ts=ts_rob, config=sim_cfg)
                traj = traj_fn(y_rob, x0)
                rho = spec.evaluate(traj)
                rho_grid[i] = float(jnp.where(jnp.isfinite(rho), rho, -1.0))

            state_names = [
                "pn",
                "dpn",
                "pe",
                "dpe",
                "h",
                "dh",
                "phi",
                "dphi",
                "theta",
                "dtheta",
                "psi",
                "dpsi",
            ]
            dim_labels = (state_names[d0], state_names[d1])
            fig, axes = plot_ic_grid(points, rho_grid, dim_labels)
            n_sat = np.sum(rho_grid > 0)
            n_total = rho_grid.size
            print(f"Satisfied: {n_sat}/{n_total} ({100 * n_sat / n_total:.0f}%)")
            print(
                f"Robustness: min={rho_grid.min():.4f}, mean={rho_grid.mean():.4f}, "
                f"max={rho_grid.max():.4f}"
            )
            if args.bounds_csv is not None:
                _overlay_bounds(axes, args.bounds_csv, args.system)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output, dpi=300, bbox_inches="tight")
            if args.show:
                plt.show()
            plt.close(fig)
            print(f"Saved to {args.output}")
            return

        if args.parameter_vary:
            # --- Parameter perturbation from single IC ---
            x0 = (low + high) / 2
            ics = jnp.tile(x0, (args.n_ics, 1))
            all_traces, rhos = [], []
            for i in range(args.n_ics):
                key, subkey = jr.split(key)
                perturbed = perturb_model(model, subkey, args.perturb_scale)
                y_plot, _ = perturbed.simulate(x=x0, ts=ts_plot, config=sim_cfg)
                y_rob, _ = perturbed.simulate(x=x0, ts=ts_rob, config=sim_cfg)
                all_traces.append(y_plot)
                traj = traj_fn(y_rob, x0)
                rho = spec.evaluate(traj)
                rho = float(jnp.where(jnp.isfinite(rho), rho, -1.0))
                rhos.append(rho)
            rhos = np.array(rhos)
        else:
            # --- Sample ICs ---
            key, subkey = jr.split(key)
            ics = jr.uniform(
                subkey, (args.n_ics, low.shape[0]), minval=low, maxval=high
            )

            # --- Simulate (fine dt for plotting) and evaluate robustness (dt=1.0) ---
            sim_plot = partial(model.simulate, ts=ts_plot, config=sim_cfg)
            sim_rob = partial(model.simulate, ts=ts_rob, config=sim_cfg)

            all_traces = []
            rhos = []
            for i in range(args.n_ics):
                x0 = ics[i]
                y_plot, _ = sim_plot(x=x0)
                y_rob, _ = sim_rob(x=x0)
                all_traces.append(y_plot)
                traj = traj_fn(y_rob, x0)
                rho = spec.evaluate(traj)
                rho = float(jnp.where(jnp.isfinite(rho), rho, -1.0))
                rhos.append(rho)
            rhos = np.array(rhos)

    n_sat = np.sum(rhos > 0)
    print(f"Satisfied: {n_sat}/{len(rhos)} ({100 * n_sat / len(rhos):.0f}%)")
    print(
        f"Robustness: min={rhos.min():.4f}, mean={rhos.mean():.4f}, "
        f"max={rhos.max():.4f}"
    )

    # --- Adaptive color normalization ---
    rho_min, rho_max = float(rhos.min()), float(rhos.max())
    if rho_min >= 0:
        norm = mcolors.Normalize(vmin=0, vmax=max(rho_max, 0.01))
        cmap = cm.Greens
    elif rho_max <= 0:
        norm = mcolors.Normalize(vmin=min(rho_min, -0.01), vmax=0)
        cmap = cm.Reds_r
    else:
        vabs = max(abs(rho_min), abs(rho_max))
        norm = mcolors.TwoSlopeNorm(vmin=rho_min, vcenter=0, vmax=rho_max)

        colors = ["#e31a1c", "#ff7f00", "#33a02c"]
        cmap = mcolors.LinearSegmentedColormap.from_list("RedBlackGreen", colors)

    # --- Plot ---
    plot_fn = PLOT_FNS[args.system]
    fig, axes = plot_fn(ts_plot, all_traces, ics, rhos, cmap, norm)

    # --- Optional bounds overlay from CSV ---
    if args.bounds_csv is not None:
        _overlay_bounds(axes, args.bounds_csv, args.system)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
