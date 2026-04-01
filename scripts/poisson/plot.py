"""
Contour plot of Poisson equation solution from launch_jacobi_solver CSV output.
CSV format: i_x, i_y, value
Filename:   poisson_solution_<prec>_prec_chol_<prec_cholesky>_zeta_<zeta>.csv
Author: Sahil Bhola, University of Michigan, 2026
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("../journal.mplstyle")

prec_options = ["double", "single", "half"]

parser = argparse.ArgumentParser(description="Poisson solution contour plot")
parser.add_argument(
    "--prec",
    choices=prec_options,
    default="double",
    help="Precision used in the solver",
)
parser.add_argument(
    "--prec_cholesky",
    choices=prec_options,
    default="double",
    help="Precision used for Cholesky decomposition",
)
parser.add_argument(
    "--zeta",
    type=float,
    default=None,
    help="Zeta value (if omitted, all matching CSVs are plotted)",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=".",
    help="Directory containing the CSV files",
)
parser.add_argument(
    "--levels",
    type=int,
    default=20,
    help="Number of contour levels",
)
args = parser.parse_args()


def get_filename(prec: str, prec_cholesky: str, zeta: float) -> str:
    return f"poisson_solution_{prec}_prec_chol_{prec_cholesky}_zeta_{zeta:.6f}.csv"


def find_csv_files(
    prec: str, prec_cholesky: str, zeta: float | None, data_dir: str
) -> list[str]:
    if zeta is not None:
        path = os.path.join(data_dir, get_filename(prec, prec_cholesky, zeta))
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return [path]
    pattern = os.path.join(
        data_dir, f"poisson_solution_{prec}_prec_chol_{prec_cholesky}_zeta_*.csv"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs matched: {pattern}")
    return files


def load_grid(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CSV and return (x_coords, y_coords, 2-D value grid)."""
    df = pd.read_csv(path)
    Nx = df["i_x"].max() + 1
    Ny = df["i_y"].max() + 1
    # internal node spacing on the unit square
    x = np.linspace(1 / (Nx + 1), Nx / (Nx + 1), Nx)
    y = np.linspace(1 / (Ny + 1), Ny / (Ny + 1), Ny)
    # pivot: rows = i_y, cols = i_x
    grid = df.pivot(index="i_y", columns="i_x", values="value").to_numpy()
    return x, y, grid


def plot_contour(path: str, ax: plt.Axes) -> None:
    x, y, grid = load_grid(path)
    X, Y = np.meshgrid(x, y)

    cf = ax.contourf(X, Y, grid, levels=args.levels, cmap="viridis")
    ax.contour(X, Y, grid, levels=args.levels, colors="k", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label=r"$u(x,y)$")

    # parse prec, prec_cholesky, zeta from filename for the title
    base = os.path.basename(path)
    zeta_str = base.split("zeta_")[-1].replace(".csv", "")
    prec_str = base.split("_prec_")[0].split("solution_")[-1]
    chol_str = base.split("_chol_")[-1].split("_zeta_")[0]
    ax.set_title(
        rf"Poisson solution  |  {prec_str}  |  chol: {chol_str}  |  $\zeta={zeta_str}$",
        fontsize=14,
    )
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=11)


def main() -> None:
    files = find_csv_files(args.prec, args.prec_cholesky, args.zeta, args.data_dir)
    n = len(files)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        layout="compressed",
    )
    axs = np.array(axs).reshape(nrows, ncols)

    for idx, path in enumerate(files):
        r, c = divmod(idx, ncols)
        plot_contour(path, axs[r, c])

    # hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r, c].set_visible(False)

    savename = f"poisson_contour_{args.prec}_prec_chol_{args.prec_cholesky}.png"
    plt.savefig(savename, dpi=150)
    print(f"Saved: {savename}")


if __name__ == "__main__":
    main()
