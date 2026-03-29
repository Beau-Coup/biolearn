"""Plot L2 norm of residual network parameters over training.

Usage:
  uv run python scripts/plot_residual_norm.py --dirs results/hill_*
  uv run python scripts/plot_residual_norm.py --dir results/hill_67f7f7b_1774736330
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
    """Result directories. If empty, uses all under results/."""
    output: str = "figures/residual_norm.pdf"
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

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    found_any = False

    for run_dir in run_dirs:
        curves = np.load(run_dir / "training_curves.npz")

        if "res_norms" not in curves.files:
            print(f"Skipping {run_dir.name} (no res_norms, re-run training)")
            continue

        res_norms = curves["res_norms"]  # (num_instantiations, num_epochs)
        found_any = True

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            best_idx = summary.get("best_model_idx", 0)
        else:
            best_idx = 0

        for i in range(res_norms.shape[0]):
            y = _smooth(np.sqrt(res_norms[i]), args.smooth)
            x = np.arange(len(y))
            is_best = (i == best_idx)
            ax.plot(x, y, alpha=0.9 if is_best else 0.25,
                    linewidth=0.8 if is_best else 0.5, color="C0")

    if not found_any:
        print("No runs with res_norms found. Re-run training to save them.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|\theta_{\mathrm{res}}\|_2$")
    ax.grid(True, alpha=0.2, linewidth=0.4)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
