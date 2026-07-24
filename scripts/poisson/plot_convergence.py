"""
Convergence plot for Block Jacobi Poisson solver.
CSV format: iter, error
Filename:   poisson_convergence_<prec>_prec_chol_<prec_cholesky>_zeta_<zeta>.csv
Author: Sahil Bhola, University of Michigan, 2026
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

plt.style.use("../journal.mplstyle")

parser = argparse.ArgumentParser(description="Block Jacobi convergence plot")
parser.add_argument(
    "--data_dir",
    type=str,
    default=".",
    help="Directory containing the convergence CSV files",
)
parser.add_argument(
    "--etol",
    type=float,
    default=None,
    help="If given, draw a horizontal dashed line at this tolerance",
)
args = parser.parse_args()


def find_convergence_files(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "poisson_convergence_*_prec_chol_*_zeta_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No convergence CSVs matched: {pattern}")
    return files


def parse_label(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(
        r"poisson_convergence_(?P<prec>\w+)_prec_chol_(?P<chol>\w+)_zeta_"
        r"(?P<zeta>[0-9.]+)\.csv",
        base,
    )
    if m:
        return f"prec={m.group('prec')}  chol={m.group('chol')}  "
        "$\\zeta$={m.group('zeta')}"
    return base


LINESTYLES = ["-", "--", "-.", ":"]


def main() -> None:
    files = find_convergence_files(args.data_dir)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, path in enumerate(files):
        df = pd.read_csv(path)
        label = parse_label(path)
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.semilogy(df["iter"], df["error"], linestyle=ls, label=label)

    if args.etol is not None:
        ax.axhline(
            args.etol,
            color="k",
            linestyle="--",
            linewidth=0.8,
            label=f"etol={args.etol:.0e}",
        )

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(r"Relative $\ell_2$ error", fontsize=13)
    ax.set_title("Block Jacobi convergence", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    savename = "poisson_convergence.png"
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    print(f"Saved: {savename}")


if __name__ == "__main__":
    main()
