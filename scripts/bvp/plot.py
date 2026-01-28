"""
plotting utils for dot product
Author: Sahil Bhola, University of Michigan, 2026
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse

plt.style.use("../journal.mplstyle")
parser = argparse.ArgumentParser(description="dot product plotting config")

dist_options = ["Normal", "ZeroOne", "MinusOnePlusOne", "PowTwo", "Ones"]
prec_options = ["Double", "Single", "Half"]
COLORS = {
    "DREA": "#1B6F6A",
    "MPREA": "red",
    "VPREA_U": "blue",
    "VPREA_beta": "goldenrod",
}
LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]


parser.add_argument(
    "--dist",
    nargs="+",
    choices=dist_options,
    default=["ZeroOne", "MinusOnePlusOne"],
    help="Plotting distribution",
)
parser.add_argument(
    "--prec",
    nargs="+",
    choices=prec_options,
    default=["Single", "Half"],
    help="Plotting precision",
)
parser.add_argument(
    "--num_samples", type=int, default=1, help="Number of samples for hte forward error"
)
parser.add_argument(
    "--alpha",
    nargs="+",
    default=[2.0],
    help="Beta bound model alpha value for each confidence",
)
parser.add_argument(
    "--beta", type=float, default=2.0, help="Beta bound model beta value"
)
parser.add_argument("--confidence", type=float, default=0.99, help="Bound confidence")
args = parser.parse_args()


def get_filename(
    experiment="backward", model="uniform", prec="Single", alpha=None, beta=None
):
    assert model.lower() in ["uniform", "beta"]
    # prefix
    if experiment == "backward":
        prefix = "backward_error_result"
    elif experiment == "forward":
        prefix = f"forward_error_result_num_samples_{args.num_samples}"
    else:
        prefix = "invalid"
    base = (
        f"{prefix}_bvp_"
        f"{prec.lower()}_prec_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"bound_model_{model.lower()}"
    )

    if model.lower() == "beta":
        base += f"_a_{alpha:0.5f}_b_{beta:0.5f}"

    return base + ".csv"


def get_savefig_name(experiment="backward"):
    # prefix
    if experiment == "backward":
        prefix = "backward_error"
    elif experiment == "forward":
        prefix = f"forward_error_vector_size_{args.vector_size}"
    else:
        prefix = "invalid"
    base = (
        f"{prefix}_result_bvp_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"beta_dist_params"
    )

    base += f"_b_{args.beta:0.5f}"

    return base + ".png"


def get_backward_error_data(model, prec, alpha=None, beta=None):
    # get filename
    filename = get_filename("backward", model, prec, alpha, beta)
    # load dataframe
    df = pd.read_csv(filename)
    return df


def get_forward_error_data(model, prec, alpha=None, beta=None):
    # get filename
    filename = get_filename("forward", model, prec, alpha, beta)
    # load dataframe
    df = pd.read_csv(filename)
    return df


def empirical_cdf(samples):
    """
    Compute the empirical CDF from 1D samples.

    Args:
        samples: 1D array-like of samples

    Returns:
        cdf_func: A callable function that returns CDF values for given x values
        x_sorted: Sorted sample values (for reference)
        cdf_values: Corresponding CDF values at each sorted sample
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)

    # Sort samples
    x_sorted = np.sort(samples)

    # Compute empirical CDF values (0 to 1)
    cdf_values = np.arange(1, n + 1) / n

    # Create interpolation function
    # For values outside the range, use step function behavior
    cdf_func = interpolate.interp1d(
        x_sorted, cdf_values, kind="next", fill_value=(0, 1), bounds_error=False
    )

    return {"function": cdf_func, "x": x_sorted, "F": cdf_values}


def pretty_dist(dist):
    if dist == "MinusOnePlusOne":
        return "U(-1,1)"
    elif dist == "ZeroOne":
        return "U(0,1)"
    elif dist == "Normal":
        return "N(0,1)"
    elif dist == "PowTwo":
        return "U(1,2)"
    else:
        return dist


def plot_backward_error_given_precision(prec, ax):
    print(f"Plotting backward error for alpha: {args.alpha}")
    # uniform data
    df_uniform = get_backward_error_data(model="uniform", prec=prec)
    n = df_uniform["n"]
    backward_error_mean = df_uniform["backward_error_mean"]
    backward_error_max = df_uniform["backward_error_max"]
    gamma_det = df_uniform["gamma_det"]
    gamma_mprea = df_uniform["gamma_mprea"]
    gamma_vprea_u = df_uniform["gamma_vprea"]

    ax.plot(
        n,
        backward_error_mean,
        label=r"$\varepsilon_{bwd}^{mean}$",
        color="k",
        linestyle="-",
        marker="X",
    )
    ax.plot(
        n,
        backward_error_max,
        label=r"$\varepsilon_{bwd}^{max}$",
        color="k",
        linestyle="--",
        marker="s",
    )
    # ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0, linestyle="-")
    ax.plot(n, gamma_det, label=r"DREA", color=COLORS["DREA"])
    ax.plot(n, gamma_mprea, label=r"MPREA", color=COLORS["MPREA"])
    ax.plot(
        n, gamma_vprea_u, label=r"VPREA ($\mathcal{U}$-model)", color=COLORS["VPREA_U"]
    )
    for ii, alpha in enumerate(args.alpha):
        df_beta = get_backward_error_data(
            model="beta", prec=prec, alpha=alpha, beta=args.beta
        )
        gamma_vprea_beta = df_beta["gamma_vprea"]
        ax.plot(
            n,
            gamma_vprea_beta,
            label=rf"VPREA ($\beta$-model; $\alpha$={alpha:.2f})",
            color=COLORS["VPREA_beta"],
            linestyle=LINESTYLES[ii],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Discretization intervals, $M$", size=27)
    ax.set_ylabel(r"$\varepsilon_{bwd}$", size=27)
    if prec.lower() == "single":
        ax.set_ylim(bottom=1e-8, top=1e-5)
    elif prec.lower() == "half":
        ax.set_ylim(bottom=1e-4, top=1e-1)
    ax.set_xlim(left=10, right=10000)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=25,
    )

    return ax


def plot_forward_error_given_precision(prec, ax):
    print(f"Plotting forward error for alpha: {args.alpha}")
    assert len(args.alpha) == 1
    # uniform data
    df_uniform = get_forward_error_data(model="uniform", prec=prec)
    df_beta = get_forward_error_data(
        model="beta", prec=prec, alpha=args.alpha[0], beta=args.beta
    )

    n = df_uniform["n"]
    forward_error = df_uniform["forward_error"]
    forward_error_model_u = df_uniform["forward_error_model"]
    forward_error_model_beta = df_beta["forward_error_model"]

    gamma_det = df_uniform["gamma_det"]
    gamma_mprea = df_uniform["gamma_mprea"]
    gamma_vprea_u = df_uniform["gamma_vprea"]
    gamma_vprea_beta = df_beta["gamma_vprea"]

    ax.plot(
        n,
        forward_error,
        label=r"$\varepsilon_{fwd}^{true}$",
        color="k",
        linestyle="-",
        marker="X",
    )
    ax.plot(
        n,
        forward_error_model_u,
        label=r"$\varepsilon_{fwd}^{\mathcal{U}-model}$",
        color=COLORS["VPREA_U"],
        linestyle="--",
        marker="s",
    )

    ax.plot(
        n,
        forward_error_model_beta,
        label=r"$\varepsilon_{fwd}^{\beta-model}$",
        color=COLORS["VPREA_beta"],
        linestyle="--",
        marker="s",
    )

    # # ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0, linestyle="-")
    ax.plot(n, gamma_det, label=r"DREA", color=COLORS["DREA"])
    ax.plot(n, gamma_mprea, label=r"MPREA", color=COLORS["MPREA"])
    ax.plot(
        n, gamma_vprea_u, label=r"VPREA ($\mathcal{U}$-model)", color=COLORS["VPREA_U"]
    )
    ax.plot(
        n,
        gamma_vprea_beta,
        label=rf"VPREA ($\beta$-model; $\alpha$={args.alpha[0]:.2f})",
        color=COLORS["VPREA_beta"],
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Discretization intervals, $M$")
    ax.set_ylabel(r"$\varepsilon_{fwd}$")
    if prec.lower() == "single":
        ax.set_ylim(bottom=1e-8)
    elif prec.lower() == "half":
        ax.set_ylim(bottom=1e-4)
    # ax.set_xlim(left=10, right=10000)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=25,
    )

    return ax


def plot_backward_error():
    fig, axs = plt.subplots(
        1,
        len(args.prec),
        figsize=(16, 8),
        # sharex=True,
        # sharey=True,
        layout="compressed",
    )
    for ii, ax in enumerate(axs):
        plot_backward_error_given_precision(args.prec[ii].lower(), ax=ax)
        # ax.label_outer()
        if ii == 0:
            ax.legend(ncol=2, loc="upper left", fontsize=20)
            ax.set_title(r"Single-precison, $\mathrm{fp}32$", fontsize=22)
        if ii == 1:
            ax.set_ylabel(None)
            ax.set_title(r"Half-precison, $\mathrm{fp}16$", fontsize=22)

    savename = get_savefig_name(experiment="backward")
    plt.savefig(savename)


def plot_forward_error():
    fig, axs = plt.subplots(
        1,
        len(args.prec),
        figsize=(6.2 * len(args.prec), 5.5),
        # sharex=True,
        # sharey=True,
        layout="compressed",
    )
    for ii, ax in enumerate(axs):
        plot_forward_error_given_precision(args.prec[ii].lower(), ax=ax)
        # ax.label_outer()
        if ii == 0:
            ax.legend(loc="best")
            ax.set_title(r"Single-precison, $\mathrm{fp}32$")
        if ii == 1:
            ax.set_ylabel(None)
            ax.set_title(r"Half-precison, $\mathrm{fp}16$")

    # savename = get_savefig_name(experiment="backward")
    plt.savefig("test.png")


if __name__ == "__main__":
    # plot_backward_error()
    plot_forward_error()
