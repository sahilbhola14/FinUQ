"""
Per-iteration error bound comparison plot for Block Jacobi Poisson solver.
CSV format: iter, det, mprea, vprea (each column is the infinity norm of the
per-iteration error bound of that type, see save_per_iteration_bounds)
Filename:   poisson_per_iteration_bounds_<prec>_prec_chol_<prec_cholesky>.csv
Author: Sahil Bhola, University of Michigan, 2026
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("../journal.mplstyle")

BOUND_TYPES = ["det", "mprea", "vprea"]
BOUND_LABELS = {
    "det": "Deterministic",
    "mprea": "Mean-informed",
    "vprea": "Variance-informed",
}
BOUND_LINESTYLES = {"det": "-", "mprea": "--", "vprea": ":"}

parser = argparse.ArgumentParser(
    description="Compare per-iteration Block Jacobi error bounds"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=".",
    help="Directory containing the per-iteration bound CSV files",
)
args = parser.parse_args()


def find_bound_files(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "poisson_per_iteration_bounds_*_prec_chol_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No per-iteration bound CSVs matched: {pattern}")
    return files


def parse_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(
        r"poisson_per_iteration_bounds_(?P<prec>\w+)_prec_chol_(?P<chol>\w+)\.csv",
        base,
    )
    if m:
        return f"prec={m.group('prec')}  chol={m.group('chol')}"
    return base


def main() -> None:
    files = find_bound_files(args.data_dir)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, path in enumerate(files):
        df = pd.read_csv(path)
        label = parse_label(path)
        color = colors[idx % len(colors)]

        # compare the individual bound types against each other
        for bound_type in BOUND_TYPES:
            ax.semilogy(
                df["iter"],
                df[bound_type],
                linestyle=BOUND_LINESTYLES[bound_type],
                color=color,
                linewidth=1.3,
                label=f"{label}  {BOUND_LABELS[bound_type]}",
            )

        # overall infinity norm: worst case across all three bound types
        overall_inf_norm = df[BOUND_TYPES].max(axis=1)
        ax.semilogy(
            df["iter"],
            overall_inf_norm,
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=f"{label}  $\\ell_\\infty$ (max over bound types)",
        )

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(r"Error bound ($\ell_\infty$ norm)", fontsize=13)
    ax.set_title("Block Jacobi per-iteration error bounds", fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    savename = "poisson_per_iteration_bounds.png"
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    print(f"Saved: {savename}")


if __name__ == "__main__":
    main()
