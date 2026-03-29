"""Compare training runs across loss functions.

Plots training loss and satisfaction rate over epochs, grouped by loss function.
Each loss function gets a distinct color; the best run per group is highlighted.

Usage:
  uv run python scripts/plot_compare.py --dirs results/hill_*
  uv run python scripts/plot_compare.py --dirs results/quadrotor_* --smooth 5
  uv run python scripts/plot_compare.py  # all results/
"""

import json
from collections import defaultdict
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
    output: str = "figures/compare.pdf"
    """Output file path."""
    smooth: int = 1
    """Smoothing window size (1 = no smoothing)."""
    group_by: str = "loss"
    """args.json field to group runs by."""


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&")


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

    # --- Load and group runs ---
    groups: dict[str, list[dict]] = defaultdict(list)

    for run_dir in run_dirs:
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
            summary = {}

        curves = np.load(run_dir / "training_curves.npz")

        group_key = str(run_args.get(args.group_by, "unknown"))
        best_idx = summary.get("best_model_idx", 0)

        groups[group_key].append({
            "losses": curves["losses"],  # (num_inst, num_epochs)
            "rhos": curves["rhos"] if "rhos" in curves.files else None,
            "best_idx": best_idx,
            "run_dir": str(run_dir),
            "summary": summary,
        })

    # --- Determine layout ---
    has_rhos = any(
        run["rhos"] is not None
        for runs in groups.values()
        for run in runs
    )
    nrows = 2 if has_rhos else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(3.5, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax_loss = axes[0]
    ax_sat = axes[1] if has_rhos else None

    # --- Plot each group ---
    colors = [f"C{i}" for i in range(10)]

    for gi, (group_name, runs) in enumerate(sorted(groups.items())):
        color = colors[gi % len(colors)]
        label_placed = False

        for run in runs:
            losses = run["losses"]
            best_idx = run["best_idx"]

            for i in range(losses.shape[0]):
                y = _smooth(losses[i], args.smooth)
                x = np.arange(len(y))
                is_best = (i == best_idx)
                label = _tex_escape(group_name) if (is_best and not label_placed) else None
                ax_loss.plot(x, y, color=color, alpha=0.8 if is_best else 0.15,
                             linewidth=0.8 if is_best else 0.4, label=label)
                if label:
                    label_placed = True

            # Satisfaction rate from rhos
            if ax_sat is not None and run["rhos"] is not None:
                rhos = run["rhos"]  # (num_inst, num_epochs, n_samples)
                for i in range(rhos.shape[0]):
                    sat_rate = np.mean(rhos[i] > 0, axis=-1)  # (num_epochs,)
                    y = _smooth(sat_rate, args.smooth)
                    x = np.arange(len(y))
                    is_best = (i == best_idx)
                    ax_sat.plot(x, y, color=color,
                                alpha=0.8 if is_best else 0.15,
                                linewidth=0.8 if is_best else 0.4)

    ax_loss.set_yscale("symlog", linthresh=1e-3)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax_loss.grid(True, alpha=0.2, linewidth=0.4)

    if ax_sat is not None:
        ax_sat.set_ylabel(r"Sat.\ rate")
        ax_sat.set_ylim(-0.05, 1.05)
        ax_sat.grid(True, alpha=0.2, linewidth=0.4)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
