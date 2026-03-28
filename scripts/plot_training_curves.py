"""Plot training curves from one or more result directories.

Usage:
  uv run python scripts/plot_training_curves.py
  uv run python scripts/plot_training_curves.py --dirs results/hill_15be1c9_*
  uv run python scripts/plot_training_curves.py --output figures/my_plot.pdf
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


@dataclass
class Args:
    dirs: list[str] = field(default_factory=list)
    """Result directories to plot. If empty, plots all under results/."""
    output: str = "figures/training_curves.pdf"
    """Output file path."""
    smooth: int = 1
    """Smoothing window size (1 = no smoothing)."""


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def _load_slack(run_dir: Path, run_args: dict) -> float | None:
    """Load final slack value from best_model.eqx checkpoint."""
    best_model_path = run_dir / "best_model.eqx"
    if not best_model_path.exists():
        return None
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        from biolearn.models.hill import BioGNN, BioGnnModel, EdgeType
        from biolearn.models.nfc import MoormanNFC
        from biolearn.models.quadrotor import QuadModel, Quadrotor

        # Reconstruct template to match tree structure
        system = run_args.get("system", "nfc")
        key = jr.key(0)
        if system == "nfc":
            nominal = MoormanNFC(2, [2, 1], gamma=1000.0, k=0.8, key=key)
        elif system == "hill":
            graph = [
                (0, 1, EdgeType.Activation), (2, 3, EdgeType.Inhibition),
                (3, 0, EdgeType.Inhibition), (3, 1, EdgeType.Activation),
                (4, 1, EdgeType.Activation), (4, 2, EdgeType.Activation),
                (4, 5, EdgeType.Activation),
            ]
            nominal = BioGnnModel(BioGNN(key, graph, 2.0))
        elif system == "quadrotor":
            nominal = QuadModel(Quadrotor(key))
        else:
            return None

        # Import BufferModel from plot_robustness_landscape (same copy)
        import math
        import diffrax
        from biolearn.models.base import BioModel, SimulateConfig
        from biolearn.models.nfc import NFC

        class _MLP(eqx.Module):
            layers: list
            def __init__(self, key, state_size, hidden_size=32):
                k1, k2, k3 = jr.split(key, 3)
                self.layers = [
                    eqx.nn.Linear(state_size, hidden_size, key=k1),
                    jax.nn.relu,
                    eqx.nn.Linear(hidden_size, hidden_size, key=k2),
                    jax.nn.relu,
                    eqx.nn.Linear(hidden_size, state_size, key=k3),
                ]
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class _BM(eqx.Module):
            nominal_model: BioModel
            residual_model: _MLP
            slack: jax.Array
            def __init__(self, key, nominal):
                self.nominal_model = nominal
                ss = math.prod(nominal.shape)
                rm = _MLP(key, state_size=ss, hidden_size=64)
                self.residual_model = jax.tree_util.tree_map(
                    lambda x: x * 1e-4 if eqx.is_inexact_array(x) else x, rm)
                self.slack = jnp.array(0.01)

        template = _BM(jr.key(99), nominal)
        loaded = eqx.tree_deserialise_leaves(str(best_model_path), template)
        return float(jax.nn.relu(loaded.slack))
    except Exception:
        return None


def main():
    args = tyro.cli(Args)

    if args.dirs:
        run_dirs = [Path(d) for d in args.dirs]
    else:
        run_dirs = sorted(Path("results").iterdir())

    run_dirs = [d for d in run_dirs if (d / "training_curves.npz").exists()]

    if not run_dirs:
        print("No result directories with training_curves.npz found.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for run_dir in run_dirs:
        curves = np.load(run_dir / "training_curves.npz")
        losses = curves["losses"]  # (num_instantiations, num_epochs)

        # Find best model index from summary
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            best_idx = summary.get("best_model_idx", 0)
        else:
            best_idx = 0

        # Load run args
        args_path = run_dir / "args.json"
        if args_path.exists():
            with open(args_path) as f:
                run_args = json.load(f)
        else:
            run_args = {}

        for i in range(losses.shape[0]):
            y = _smooth(losses[i], args.smooth)
            x = np.arange(len(y))
            is_best = (i == best_idx)
            ax.plot(x, y, alpha=0.9 if is_best else 0.25,
                    linewidth=0.8 if is_best else 0.5, color="C0")

        # Annotate slack for best run
        slack = _load_slack(run_dir, run_args)
        if slack is not None:
            ax.text(0.97, 0.97, rf"$s={slack:.3f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2, linewidth=0.4)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
