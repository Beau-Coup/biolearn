"""Plot quadrotor height trajectories under parameter perturbation.

Fixes a single IC at the center of the domain, perturbs model parameters,
and plots height traces colored by robustness.

Usage:
  uv run python scripts/plot_perturbed_quadrotor.py --help
  uv run python scripts/plot_perturbed_quadrotor.py --n-ics 30 --perturb-scale 0.3
  uv run python scripts/plot_perturbed_quadrotor.py --checkpoint results/run/best_model.eqx
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tyro

from biolearn.models.base import SimulateConfig
from biolearn.models.quadrotor import QuadModel, Quadrotor
from biolearn.specifications.quadrotor import HeightMaintain

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

_SHADE = dict(color="red", alpha=0.08, zorder=0, linewidth=0)
_BORDER = dict(color="red", alpha=0.3, lw=0.5, ls="--")


def perturb_model(model, key, scale):
    """Add Gaussian noise to all learnable parameters of a model."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(flat_params))
    noisy = [p + scale * jr.normal(k, p.shape) for p, k in zip(flat_params, keys)]
    noisy_params = jax.tree_util.tree_unflatten(tree_def, noisy)
    return eqx.combine(noisy_params, static)


def ss_to_traj_q(y_trace, _):
    return y_trace[..., 4:6]


@dataclass
class Args:
    seed: int = 42
    """Random seed."""
    checkpoint: str | None = None
    """Path to .eqx checkpoint."""
    n_ics: int = 32
    """Number of perturbed models to simulate."""
    perturb_scale: float = 0.3
    """Scale of Gaussian perturbation to model parameters."""
    output: str = "figures/quad_perturbed.pdf"
    """Output file path."""
    show: bool = False
    """Call plt.show()."""


def main():
    args = tyro.cli(Args)
    key = jr.key(args.seed)
    key, subkey = jr.split(key)

    # --- Setup ---
    model = QuadModel(Quadrotor(subkey))
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

    if args.checkpoint is not None:
        model = eqx.tree_deserialise_leaves(args.checkpoint, model)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # --- Simulate perturbed models from center IC ---
    x0 = (low + high) / 2
    all_heights = []
    rhos = []
    for i in range(args.n_ics):
        key, subkey = jr.split(key)
        perturbed = perturb_model(model, subkey, args.perturb_scale)
        y_plot, _ = perturbed.simulate(x=x0, ts=ts_plot, config=sim_cfg)
        y_rob, _ = perturbed.simulate(x=x0, ts=ts_rob, config=sim_cfg)
        all_heights.append(y_plot[:, 4])  # height only
        traj = ss_to_traj_q(y_rob, x0)
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

    # --- Color map ---
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

    # --- Plot height only ---
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.4))
    ts_np = np.array(ts_plot)
    for i in range(len(all_heights)):
        color = cmap(norm(float(rhos[i])))
        ax.plot(ts_np, np.array(all_heights[i]), color=color, alpha=0.4, linewidth=0.5)
    ax.set_ylabel(r"$h$ (m)")
    ax.set_xlabel("Time (s)")

    # Spec overlay: h < 1.4 always, h > 0.9 from t=1
    t_end = float(ts_plot[-1])
    _, hi = ax.get_ylim()
    if hi > 1.4:
        ax.axhspan(1.4, hi, **_SHADE)
        ax.axhline(1.4, **_BORDER)
    mask = ts_np >= 1.0
    lo_ylim = ax.get_ylim()[0]
    ax.fill_between(ts_np, lo_ylim, 0.9, where=mask, **_SHADE)
    ax.hlines(0.9, xmin=1.0, xmax=t_end, **_BORDER)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.locator_params(axis="y", nbins=4)
    ax.set_xlim(float(ts_plot[0]), t_end)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
