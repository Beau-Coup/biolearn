"""Plot training curves from one or more result directories.

Usage:
  uv run python -m scripts.plot_training_curves
  uv run python -m scripts.plot_training_curves --dirs results/hill_15be1c9_*
  uv run python -m scripts.plot_training_curves --output figures/my_plot.png
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    dirs: list[str] = field(default_factory=list)
    """Result directories to plot. If empty, plots all under results/."""
    output: str = "figures/training_curves.png"
    """Output file path."""
    figsize: tuple[float, float] = (10, 6)
    """Figure size in inches."""
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

    fig, ax = plt.subplots(figsize=args.figsize)

    for run_dir in run_dirs:
        curves = np.load(run_dir / "training_curves.npz")
        losses = curves["losses"]  # (num_instantiations, num_epochs)

        args_path = run_dir / "args.json"
        if args_path.exists():
            with open(args_path) as f:
                run_args = json.load(f)
        else:
            run_args = {}

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = None

        system = run_args.get("system", "?")
        loss_fn = run_args.get("loss", "?")
        lr = run_args.get("lr", "?")
        commit = run_args.get("commit", run_dir.name.split("_")[1] if "_" in run_dir.name else "?")

        label_parts = [system, f"loss={loss_fn}", f"lr={lr}", commit]
        if summary:
            sat = summary.get("best_sat", "?")
            total = summary.get("total_test_points", "?")
            label_parts.append(f"sat={sat}/{total}")
        label = " | ".join(label_parts)

        for i in range(losses.shape[0]):
            y = _smooth(losses[i], args.smooth)
            x = np.arange(len(y))
            line_label = label if i == 0 else None
            alpha = 0.4 if losses.shape[0] > 1 and i > 0 else 0.9
            ax.plot(x, y, label=line_label, alpha=alpha)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
