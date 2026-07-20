"""
Compare sequential and block dot-product backward error results.
Author: Sahil Bhola, University of Michigan, 2026
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

plt.style.use("../journal.mplstyle")

parser = argparse.ArgumentParser(
    description="compare sequential and block dot-product results"
)

dist_options = ["Normal", "ZeroOne", "MinusOnePlusOne", "PowTwo", "Ones"]
prec_options = ["Double", "Single", "Half"]
COLORS = {
    "backward_error_mean": "k",
    "backward_error_max": "k",
    "gamma_det": "#1B6F6A",
    "gamma_mprea": "red",
    "gamma_vprea_uniform": "blue",
    "gamma_vprea_beta": "goldenrod",
}
REFERENCE_COLOR = "0.7"
BETA_LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]

parser.add_argument(
    "--dist",
    nargs="+",
    choices=dist_options,
    default=["ZeroOne", "MinusOnePlusOne"],
    help="Plotting distribution",
)
parser.add_argument(
    "--prec",
    type=str,
    choices=prec_options,
    default="Half",
    help="Plotting precision",
)
parser.add_argument(
    "--alpha",
    nargs="+",
    type=float,
    default=[1.6, 1.7, 1.8],
    help="Beta bound model alpha value for each comparison curve",
)
parser.add_argument(
    "--beta", type=float, default=2.0, help="Beta bound model beta value"
)
parser.add_argument("--confidence", type=float, default=0.99, help="Bound confidence")
parser.add_argument(
    "--sequential_folder",
    type=Path,
    default=Path("sequential"),
    help="Folder containing sequential dot-product CSV results",
)
parser.add_argument(
    "--block_folder",
    type=Path,
    default=Path("block"),
    help="Folder containing block dot-product CSV results",
)
parser.add_argument(
    "--tile_size",
    type=int,
    default=64,
    help="Block tile size used in the block filename",
)
parser.add_argument(
    "--output_folder",
    type=Path,
    default=Path("."),
    help="Folder for the generated comparison figure",
)
parser.add_argument(
    "--reference_offset",
    type=float,
    default=20.0,
    help="Multiplicative offset applied to the sqrt(log_2(n) / n) reference line",
)
args = parser.parse_args()


def pretty_dist(dist):
    if dist == "MinusOnePlusOne":
        return "U(-1,1)"
    if dist == "ZeroOne":
        return "U(0,1)"
    if dist == "Normal":
        return "N(0,1)"
    if dist == "PowTwo":
        return "U(1,2)"
    return dist


def get_filename(kind, model="uniform", dist="ZeroOne", alpha=None, beta=None):
    assert kind in ["sequential", "block"]
    assert model.lower() in ["uniform", "beta"]

    if kind == "sequential":
        prefix = "backward_error_result_dot_product"
    else:
        prefix = f"backward_error_result_block_tile_size_{args.tile_size}_dot_product"

    base = (
        f"{prefix}_"
        f"{args.prec.lower()}_prec_"
        f"distribution_{pretty_dist(dist)}_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"bound_model_{model.lower()}"
    )

    if model.lower() == "beta":
        base += f"_a_{alpha:0.5f}_b_{beta:0.5f}"

    folder = args.sequential_folder if kind == "sequential" else args.block_folder
    return folder / f"{base}.csv"


def load_backward_error_data(kind, model, dist, alpha=None, beta=None):
    return pd.read_csv(get_filename(kind, model, dist, alpha, beta))


def get_savefig_name():
    dist_name = "_".join(
        pretty_dist(dist).replace(",", "").replace("(", "").replace(")", "")
        for dist in args.dist
    )
    return (
        args.output_folder
        / f"backward_error_comparison_dot_product_{args.prec.lower()}_prec_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"tile_size_{args.tile_size}_"
        f"dist_{dist_name}.png"
    )


def plot_method_for_distribution(dist, kind, ax):
    df_uniform = load_backward_error_data(kind, "uniform", dist)
    n = df_uniform["n"]

    ax.plot(
        n,
        df_uniform["backward_error_mean"],
        color=COLORS["backward_error_mean"],
        linestyle="-",
        marker="X",
        label=r"$\varepsilon_{bwd}^{mean}$",
    )
    ax.plot(
        n,
        df_uniform["backward_error_max"],
        color=COLORS["backward_error_max"],
        linestyle="--",
        marker="s",
        label=r"$\varepsilon_{bwd}^{max}$",
    )

    for key, label in [
        ("gamma_det", "DREA"),
        ("gamma_mprea", "MPREA"),
        ("gamma_vprea", r"VPREA ($\mathcal{U}$-model)"),
    ]:
        color_key = key if key != "gamma_vprea" else "gamma_vprea_uniform"
        ax.plot(n, df_uniform[key], color=COLORS[color_key], label=label)

    for ii, alpha in enumerate(args.alpha):
        df_beta = load_backward_error_data(
            kind, "beta", dist, alpha=alpha, beta=args.beta
        )
        beta_linestyle = BETA_LINESTYLES[ii % len(BETA_LINESTYLES)]
        ax.plot(
            n,
            df_beta["gamma_vprea"],
            color=COLORS["gamma_vprea_beta"],
            linestyle=beta_linestyle,
            linewidth=2.0,
            label=rf"VPREA ($\beta$; $\alpha$={alpha:.2f})",
        )

    ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Vector size, $n$")
    ax.set_ylabel(r"$\varepsilon_{bwd}$")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[10**0.5]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[10**0.5]))
    ax.minorticks_on()
    if args.prec.lower() == "single":
        ax.set_ylim(bottom=1e-10, top=1e-2)

    return n, df_uniform


def main():
    fig, axs = plt.subplots(
        len(args.dist),
        2,
        figsize=(12.5, 5.2 * len(args.dist)),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    if len(args.dist) == 1:
        axs = [axs]

    column_titles = ["Sequential", "Block"]
    for row, dist in enumerate(args.dist):
        for col, kind in enumerate(["sequential", "block"]):
            ax = axs[row][col]
            n, df_uniform = plot_method_for_distribution(dist, kind, ax)
            if row == 0:
                ax.set_title(rf"{column_titles[col]}")
            # if col == 0:
            #     ax.text(
            #         -0.28,
            #         0.5,
            #         pretty_dist(dist),
            #         transform=ax.transAxes,
            #         rotation=90,
            #         va="center",
            #         ha="center",
            #     )
            if row == 1 and kind == "block":
                ref_shape = np.sqrt(np.log2(n) / n)
                ref_scale = (
                    df_uniform["backward_error_mean"].iloc[0] / ref_shape.iloc[0]
                )
                ref_values = args.reference_offset * ref_scale * ref_shape
                ax.plot(
                    n,
                    ref_values,
                    color=REFERENCE_COLOR,
                    linestyle=":",
                    linewidth=2.0,
                    label=r"$O(\sqrt{\log_2(n) / n})$",
                )
                annotate_idx = max(len(n) - 2, 0)
                ax.annotate(
                    r"$O(\sqrt{\log_2(n) / n})$",
                    xy=(n.iloc[annotate_idx], ref_values.iloc[annotate_idx]),
                    xytext=(-100, 30),
                    textcoords="offset points",
                    color=REFERENCE_COLOR,
                )
            ax.label_outer()

    axs[0][0].legend(ncol=1, loc="best", fontsize=15)

    plt.savefig(get_savefig_name())


if __name__ == "__main__":
    main()
