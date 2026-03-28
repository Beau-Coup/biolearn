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

    has_rhos = any(
        "rhos" in np.load(d / "training_curves.npz").files for d in run_dirs
    )
    nrows = 2 if has_rhos else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(3.5, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax_loss = axes[0]
    ax_rho = axes[1] if has_rhos else None

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

        for i in range(losses.shape[0]):
            y = _smooth(losses[i], args.smooth)
            x = np.arange(len(y))
            is_best = (i == best_idx)
            ax_loss.plot(x, y, alpha=0.9 if is_best else 0.25,
                         linewidth=0.8 if is_best else 0.5, color="C0")

        # Plot per-epoch mean robustness if available
        if ax_rho is not None and "rhos" in curves.files:
            rhos = curves["rhos"]  # (num_instantiations, num_epochs, n_samples)
            for i in range(rhos.shape[0]):
                mean_rho = np.mean(rhos[i], axis=-1)  # (num_epochs,)
                y = _smooth(mean_rho, args.smooth)
                x = np.arange(len(y))
                is_best = (i == best_idx)
                ax_rho.plot(x, y, alpha=0.9 if is_best else 0.25,
                            linewidth=0.8 if is_best else 0.5, color="C1")

    ax_loss.set_yscale("symlog", linthresh=1e-3)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.2, linewidth=0.4)

    if ax_rho is not None:
        ax_rho.axhline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
        ax_rho.set_ylabel(r"Mean $\rho$")
        ax_rho.grid(True, alpha=0.2, linewidth=0.4)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
