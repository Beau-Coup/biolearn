"""Plot training curves from one or more result directories.

Usage:
  uv run python scripts/plot_training_curves.py
  uv run python scripts/plot_training_curves.py --dirs results/hill_15be1c9_*
  uv run python scripts/plot_training_curves.py --output figures/my_plot.pdf
  uv run python scripts/plot_training_curves.py --metrics mean sat_rate min
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

RhoMetric = Literal["mean", "sat_rate", "min"]


@dataclass
class Args:
    dirs: list[str] = field(default_factory=list)
    """Result directories to plot. If empty, plots all under results/."""
    output: str = "figures/training_curves.pdf"
    """Output file path."""
    smooth: int = 1
    """Smoothing window size (1 = no smoothing)."""
    metrics: list[RhoMetric] = field(default_factory=lambda: ["mean"])
    """Robustness metrics to plot: mean, sat_rate, min, or any combination."""


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def _compute_metric(rhos: np.ndarray, metric: RhoMetric) -> np.ndarray:
    """Compute per-epoch scalar from rhos array of shape (num_epochs, n_samples)."""
    if metric == "mean":
        return np.mean(rhos, axis=-1)
    elif metric == "sat_rate":
        return np.mean(rhos > 0, axis=-1)
    elif metric == "min":
        return np.min(rhos, axis=-1)
    raise ValueError(f"Unknown metric: {metric}")


METRIC_LABELS = {
    "mean": r"Mean $\rho$",
    "sat_rate": r"Sat.\ rate",
    "min": r"$\min \rho$",
}

METRIC_COLORS = {
    "mean": "C1",
    "sat_rate": "C2",
    "min": "C3",
}


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
    metrics = args.metrics if has_rhos else []
    n_metrics = len(metrics)

    # Layout: loss always present; metrics in additional panels
    # 1 metric  -> 2 rows, 1 col
    # 2 metrics -> 2 rows, 1 col (loss + 2 metrics stacked... actually 3 rows)
    # 3 metrics -> 2x2 grid (loss + 3 metrics)
    # 0 metrics -> 1 row
    n_panels = 1 + n_metrics
    if n_panels <= 2:
        nrows, ncols = n_panels, 1
    else:
        nrows, ncols = 2, 2

    fig, axes_arr = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 1.6 * nrows),
        squeeze=False,
    )
    # Flatten to list for easy indexing
    axes_flat = axes_arr.ravel().tolist()

    # Hide unused panels (e.g. 3 panels in 2x2 grid)
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    ax_loss = axes_flat[0]
    metric_axes = {m: axes_flat[1 + i] for i, m in enumerate(metrics)}

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

        if "rhos" in curves.files:
            rhos = curves["rhos"]  # (num_instantiations, num_epochs, n_samples)
            for metric in metrics:
                ax = metric_axes[metric]
                color = METRIC_COLORS[metric]
                for i in range(rhos.shape[0]):
                    vals = _compute_metric(rhos[i], metric)
                    y = _smooth(vals, args.smooth)
                    x = np.arange(len(y))
                    is_best = (i == best_idx)
                    ax.plot(x, y, alpha=0.9 if is_best else 0.25,
                            linewidth=0.8 if is_best else 0.5, color=color)

    ax_loss.set_yscale("symlog", linthresh=1e-3)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.2, linewidth=0.4)

    for metric, ax in metric_axes.items():
        if metric != "sat_rate":
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.2, linewidth=0.4)

    # Set xlabel on bottom row axes
    for ax in axes_arr[-1]:
        if ax.get_visible():
            ax.set_xlabel("Epoch")

    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
