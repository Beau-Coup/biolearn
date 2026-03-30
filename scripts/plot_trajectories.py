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
            max_steps=int(3e4),
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
            max_steps=int(2e4),
            rtol=1e-3,
            atol=1e-4,
            max_stepsize=0.5,
            progress_bar=False,
        )
        ts_plot = jnp.arange(0.0, 25.0, 0.1)
        ts_rob = jnp.arange(0.0, 25.0, 1.0)
        low = jnp.array([0.0, 0.0, 0.01, 0.01, 0.99, 0.99])
        high = jnp.array([0.2, 0.2, 0.04, 0.04, 1.0, 1.0])
        is_nfc = False
    elif system == "quadrotor":
        model = QuadModel(Quadrotor(key))
        spec = HeightMaintain()
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
    # x_i < 1.5 always for all 4 species
    for i in range(min(len(axes), 4)):
        _shade_above(axes[i], 1.5)


def draw_spec_overlay_laub(axes, ts):
    # x4 < 4.5 always
    _shade_above(axes[2], 4.5)


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
        t = df["Time_s"].values
        for j, name in enumerate(["x1", "x2", "x3", "x4"]):
            lo_col, hi_col = f"Min_{name}", f"Max_{name}"
            if j < len(axes) and lo_col in df.columns:
                axes[j].plot(t, df[lo_col].values, **_BOUNDS_STYLE)
            if j < len(axes) and hi_col in df.columns:
                axes[j].plot(t, df[hi_col].values, **_BOUNDS_STYLE)
    elif system == "laub":
        t = df["Time_s"].values
        state_names = ["x1", "x3", "x4"]
        for j, name in enumerate(state_names):
            lo_col, hi_col = f"Min_{name}", f"Max_{name}"
            if j < len(axes) and lo_col in df.columns:
                axes[j].plot(t, df[lo_col].values, **_BOUNDS_STYLE)
            if j < len(axes) and hi_col in df.columns:
                axes[j].plot(t, df[hi_col].values, **_BOUNDS_STYLE)
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


def plot_hill(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(3.5, 3.2),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    species_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        for j in range(4):
            axes[j].plot(
                np.array(ts),
                np.array(y_trace[:, j]),
                color=color,
                alpha=0.6,
                linewidth=0.4,
            )
    for j in range(4):
        axes[j].set_ylabel(species_labels[j])
    axes[-1].set_xlabel("Time (s)")
    draw_spec_overlay_hill(axes, ts)
    _clean_axes(list(axes), ts)
    return fig, list(axes)


def plot_laub(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.5, 2.6),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    state_labels = [r"$x_1$", r"$x_3$", r"$x_4$"]
    state_indices = [0, 2, 3]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        for j, si in enumerate(state_indices):
            axes[j].plot(
                np.array(ts),
                np.array(y_trace[:, si]),
                color=color,
                alpha=0.2,
                linewidth=0.4,
            )
    for j in range(3):
        axes[j].set_ylabel(state_labels[j])
    axes[-1].set_xlabel("Time (s)")
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
    output: str = "figures/trajectories.pdf"
    """Output file path."""
    show: bool = False
    """Call plt.show()."""


def main():
    args = tyro.cli(Args)
    key = jr.key(args.seed)
    key, subkey = jr.split(key)

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

    # --- Sample ICs ---
    key, subkey = jr.split(key)
    ics = jr.uniform(subkey, (args.n_ics, low.shape[0]), minval=low, maxval=high)

    # --- Simulate (fine dt for plotting) and evaluate robustness (dt=1.0) ---
    traj_fn = TRAJ_FNS[args.system]
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
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cmap = cm.RdYlGn

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
