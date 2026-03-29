"""Visualize model trajectories colored by robustness, with spec overlay.

Supports all system types: nfc, hill, quadrotor.

Usage:
  uv run python scripts/plot_trajectories.py --help
  uv run python scripts/plot_trajectories.py --system nfc --n-ics 20
  uv run python scripts/plot_trajectories.py --system quadrotor --n-ics 16
  uv run python scripts/plot_trajectories.py --system hill --n-ics 10
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
import tyro

from biolearn.models.base import BioModel, SimulateConfig
from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
from biolearn.models.laub import LLModel, LaubLoomis
from biolearn.models.nfc import NFC, MoormanNFC
from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.hk25 import FastProduce
from biolearn.specifications.laub import StableConverge
from biolearn.specifications.quadrotor import HeightMaintain
from biolearn.specifications.ss_classification import PhiXorFast

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

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
            dcoeff=0, rtol=config.rtol, atol=config.atol,
            dtmax=config.max_stepsize, dtmin=1e-10,
        )
        term = diffrax.ODETerm(self._step)
        solution = diffrax.diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=0.001, y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=config.max_steps, throw=config.throw, args=args,
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
    """Return (model, spec, sim_cfg, ts, low, high, is_nfc) for a system."""
    if system == "nfc":
        model = MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=key)
        spec = PhiXorFast()
        sim_cfg = SimulateConfig(
            to_ss=False, stiff=True, throw=False, max_steps=int(3e4),
            rtol=1e-4, atol=1e-5, max_stepsize=0.5, progress_bar=False,
        )
        ts = jnp.arange(0.0, 20.0, 0.5)
        low = jnp.array([0.1, 0.1])
        high = jnp.array([0.9, 0.9])
        is_nfc = True
    elif system == "hill":
        model = BioGnnModel(BioGNN(key, HILL_GRAPH, 2.0))
        spec = FastProduce()
        sim_cfg = SimulateConfig(
            to_ss=False, stiff=True, throw=False, max_steps=int(2e4),
            rtol=1e-3, atol=1e-4, max_stepsize=0.5, progress_bar=False,
        )
        ts = jnp.arange(0.0, 15.0, 0.5)
        low = jnp.array([0.0, 0.0, 0.01, 0.01, 0.99, 0.99])
        high = jnp.array([0.2, 0.2, 0.04, 0.04, 1.0, 1.0])
        is_nfc = False
    elif system == "quadrotor":
        model = QuadModel(Quadrotor(key))
        spec = HeightMaintain()
        sim_cfg = SimulateConfig(
            to_ss=False, stiff=False, throw=False, max_steps=int(3e4),
            rtol=1e-3, atol=1e-4, max_stepsize=0.5, progress_bar=False,
        )
        ts = jnp.arange(0.0, 5.0, 0.05)
        low = jnp.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4,
                          0.0, -0.02, 0.0, -0.02, 0.0, -0.02])
        high = jnp.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                           0.0, 0.02, 0.0, 0.02, 0.0, 0.02])
        is_nfc = False
    elif system == "laub":
        model = LLModel(LaubLoomis(key))
        spec = StableConverge()
        sim_cfg = SimulateConfig(
            to_ss=False, stiff=True, throw=False, max_steps=4096,
            rtol=1e-3, atol=1e-4, max_stepsize=0.5, progress_bar=False,
        )
        ts = jnp.arange(0.0, 20.0, 0.5)
        center = jnp.array([1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45])
        low = center - 0.1
        high = center + 0.1
        is_nfc = False
    else:
        raise ValueError(f"Unknown system: {system}")
    return model, spec, sim_cfg, ts, low, high, is_nfc


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
# Spec overlay drawing
# ---------------------------------------------------------------------------


def draw_spec_overlay_nfc(axes, ts):
    ax = axes[0]
    ax.axhline(0.1, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.axhspan(0.0, 0.05, alpha=0.05, color="green")
    ax.axvline(5.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.text(5.2, 0.08, r"$t_1{=}5$", fontsize=5, color="gray")
    ax.text(0.2, 0.105, r"$\epsilon_1{=}0.1$", fontsize=5, color="red", alpha=0.7)
    ax.text(0.2, 0.055, r"$\epsilon_2{=}0.05$", fontsize=5, color="red", alpha=0.7)


def draw_spec_overlay_quadrotor(axes, ts):
    ax_h, ax_v = axes[0], axes[1]
    ax_h.axhline(1.4, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_h.axhline(0.9, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_h.axhspan(0.9, 1.4, alpha=0.05, color="green")
    ax_h.axhline(1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_v.axhline(0.1, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_v.axhline(-0.1, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_v.axhspan(-0.1, 0.1, alpha=0.05, color="green")


def draw_spec_overlay_hill(axes, ts):
    thresholds = {0: [0.2], 1: [0.3], 2: [0.3, 0.5], 3: [0.6, 0.9]}
    for i in range(min(len(axes), 4)):
        axes[i].axhline(1.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        if i in thresholds:
            for val in thresholds[i]:
                axes[i].axhline(val, color="red", linestyle="--",
                                linewidth=0.5, alpha=0.7)


def draw_spec_overlay_laub(axes, ts):
    # x4 (axes[2]): must stay below 0.4; trigger at 3.0
    axes[2].axhline(0.4, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
    axes[2].axhspan(0.0, 0.4, alpha=0.04, color="green")
    axes[2].axhline(3.0, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)


# ---------------------------------------------------------------------------
# Per-system plot layout
# ---------------------------------------------------------------------------


def plot_nfc(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.2))
    axes = [ax]
    for i in range(len(all_traces)):
        y_trace, x0 = all_traces[i], all_ics[i]
        traj = ss_to_traj_xor(y_trace, x0)
        x_diff = traj[:, 1] - traj[:, 0]
        y_true = jax.nn.relu(x_diff - 0.1) + jax.nn.relu(-x_diff - 0.1)
        err = jnp.abs(y_true - traj[:, 2])
        color = cmap(norm(float(rhos[i])))
        ax.plot(np.array(ts), np.array(err), color=color, alpha=0.6, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Error $|y - y^*|$")
    draw_spec_overlay_nfc(axes, ts)
    return fig, axes


def plot_quadrotor(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, (ax_h, ax_v) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)
    axes = [ax_h, ax_v]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        ax_h.plot(np.array(ts), np.array(y_trace[:, 4]),
                  color=color, alpha=0.6, linewidth=0.5)
        ax_v.plot(np.array(ts), np.array(y_trace[:, 5]),
                  color=color, alpha=0.6, linewidth=0.5)
    ax_h.set_ylabel(r"Height $h$ (m)")
    ax_v.set_ylabel(r"Velocity $\dot{h}$ (m/s)")
    ax_v.set_xlabel("Time (s)")
    draw_spec_overlay_quadrotor(axes, ts)
    return fig, axes


def plot_hill(ts, all_traces, all_ics, rhos, cmap, norm):
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.5), sharex=True)
    species_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        for j in range(4):
            axes[j].plot(np.array(ts), np.array(y_trace[:, j]),
                         color=color, alpha=0.6, linewidth=0.5)
    for j in range(4):
        axes[j].set_ylabel(species_labels[j])
    axes[-1].set_xlabel("Time (s)")
    draw_spec_overlay_hill(axes, ts)
    return fig, list(axes)


def plot_laub(ts, all_traces, all_ics, rhos, cmap, norm):
    """Plot Laub-Loomis x1, x3, x4 (three constrained states)."""
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True)
    state_labels = [r"$x_1$", r"$x_3$", r"$x_4$"]
    state_indices = [0, 2, 3]
    for i in range(len(all_traces)):
        y_trace = all_traces[i]
        color = cmap(norm(float(rhos[i])))
        for j, si in enumerate(state_indices):
            axes[j].plot(np.array(ts), np.array(y_trace[:, si]),
                         color=color, alpha=0.6, linewidth=0.5)
    for j in range(3):
        axes[j].set_ylabel(state_labels[j])
    axes[-1].set_xlabel("Time (s)")
    draw_spec_overlay_laub(axes, ts)
    return fig, list(axes)


PLOT_FNS = {"nfc": plot_nfc, "hill": plot_hill, "quadrotor": plot_quadrotor, "laub": plot_laub}


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
    output: str = "figures/trajectories.pdf"
    """Output file path."""
    show: bool = False
    """Call plt.show()."""


def main():
    args = tyro.cli(Args)
    key = jr.key(args.seed)
    key, subkey = jr.split(key)

    # --- System setup ---
    nominal_model, spec, sim_cfg, ts, low, high, is_nfc = setup_system(
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

    # --- Simulate and evaluate ---
    traj_fn = TRAJ_FNS[args.system]
    sim = partial(model.simulate, ts=ts, config=sim_cfg)

    all_traces = []
    rhos = []
    for i in range(args.n_ics):
        x0 = ics[i]
        y_trace, _ = sim(x=x0)
        all_traces.append(y_trace)
        traj = traj_fn(y_trace, x0)
        rho = spec.evaluate(traj)
        rho = float(jnp.where(jnp.isfinite(rho), rho, -1.0))
        rhos.append(rho)
    rhos = np.array(rhos)

    n_sat = np.sum(rhos > 0)
    print(f"Satisfied: {n_sat}/{len(rhos)} ({100 * n_sat / len(rhos):.0f}%)")
    print(f"Robustness: min={rhos.min():.4f}, mean={rhos.mean():.4f}, "
          f"max={rhos.max():.4f}")

    # --- Colormap: robustness value ---
    vabs = max(abs(rhos.min()), abs(rhos.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = cm.RdBu_r

    # --- Plot ---
    plot_fn = PLOT_FNS[args.system]
    fig, axes = plot_fn(ts, all_traces, ics, rhos, cmap, norm)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label(r"Robustness $\rho$")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
