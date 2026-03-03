"""
plotting utils for dot product
Author: Sahil Bhola, University of Michigan, 2026
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse
from matplotlib.ticker import LogLocator

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
    type=str,
    choices=prec_options,
    default="Single",
    help="Plotting precision",
)
parser.add_argument(
    "--vector_size", type=int, default=1000, help="Vector size for forward error"
)
parser.add_argument(
    "--alpha",
    nargs="+",
    default=[1.6, 1.7, 1.8],
    help="Beta bound model alpha value for each confidence",
)
parser.add_argument(
    "--beta", type=float, default=2.0, help="Beta bound model beta value"
)
parser.add_argument("--confidence", type=float, default=0.99, help="Bound confidence")
args = parser.parse_args()


def get_filename(
    experiment="backward", model="uniform", dist="ZeroOne", alpha=None, beta=None
):
    assert model.lower() in ["uniform", "beta"]
    # prefix
    if experiment == "backward":
        prefix = "backward_error_result"
    elif experiment == "forward":
        prefix = f"forward_error_result_vector_size_{args.vector_size}"
    else:
        prefix = "invalid"
    base = (
        f"{prefix}_dot_product_"
        f"{args.prec.lower()}_prec_"
        f"distribution_{pretty_dist(dist)}_"
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
        f"{prefix}_result_dot_product_"
        f"{args.prec.lower()}_prec_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"beta_dist_params"
    )

    base += f"_b_{args.beta:0.5f}"

    return base + ".png"


def get_backward_error_data(model, dist, alpha=None, beta=None):
    # get filename
    filename = get_filename("backward", model, dist, alpha, beta)
    # load dataframe
    df = pd.read_csv(filename)
    return df


def get_forward_error_data(model, dist, alpha=None, beta=None):
    # get filename
    filename = get_filename("forward", model, dist, alpha, beta)
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


def plot_backward_error_given_data_distribution(dist, ax):
    # uniform data
    df_uniform = get_backward_error_data(model="uniform", dist=dist)
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
    ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0, linestyle="-")
    ax.plot(n, gamma_det, label=r"DREA", color=COLORS["DREA"])
    ax.plot(n, gamma_mprea, label=r"MPREA", color=COLORS["MPREA"])
    ax.plot(
        n, gamma_vprea_u, label=r"VPREA ($\mathcal{U}$-model)", color=COLORS["VPREA_U"]
    )
    for ii, alpha in enumerate(args.alpha):
        df_beta = get_backward_error_data(
            model="beta", dist=dist, alpha=alpha, beta=args.beta
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
    ax.set_xlabel(r"Vector size, $n$")
    ax.set_ylabel(r"$\varepsilon_{bwd}$")
    if args.prec.lower() == "single":
        # ax.set_xlim(left=10, right=10**6)
        # ax.set_xlim(left=10)
        ax.set_ylim(bottom=1e-8, top=1e2)
    elif args.prec.lower() == "half":
        pass
        # ax.set_xlim(left=10, right=10**5)
    # Major ticks at 10^k
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))

    # ONE minor tick per decade
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[10**0.5]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[10**0.5]))
    ax.minorticks_on()

    return ax


def plot_backward_error():
    fig, axs = plt.subplots(
        1,
        len(args.dist),
        figsize=(6.2 * len(args.dist), 5.5),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    for ii, ax in enumerate(axs):
        plot_backward_error_given_data_distribution(args.dist[ii], ax=ax)
        ax.label_outer()
        if ii == 0:
            ax.legend(ncol=1, loc="best")

    savename = get_savefig_name(experiment="backward")
    plt.savefig(savename)


def plot_forward_error_cdf_given_data_distribution(dist, ax, alpha_plot=1.6):
    print(f"Plotting forward error cdf for alpha: {alpha_plot}")
    # uniform data
    df_uniform = get_forward_error_data(model="uniform", dist=dist)
    # n = df_uniform["n"]
    forward_error = df_uniform["forward_error"]
    forward_error_model_u = df_uniform["forward_error_model"]
    gamma_det = df_uniform["gamma_det"]
    gamma_mprea = df_uniform["gamma_mprea"]
    gamma_vprea_u = df_uniform["gamma_vprea"]

    df_beta = get_forward_error_data(
        model="beta", dist=dist, alpha=alpha_plot, beta=args.beta
    )
    forward_error_model_beta = df_beta["forward_error_model"]
    gamma_vprea_beta = df_beta["gamma_vprea"]

    # compute the CDF
    forward_error_cdf = empirical_cdf(forward_error)
    forward_error_model_u_cdf = empirical_cdf(forward_error_model_u)
    forward_error_model_beta_cdf = empirical_cdf(forward_error_model_beta)
    gamma_det_cdf = empirical_cdf(gamma_det)
    gamma_mprea_cdf = empirical_cdf(gamma_mprea)
    gamma_vprea_u_cdf = empirical_cdf(gamma_vprea_u)
    gamma_vprea_beta_cdf = empirical_cdf(gamma_vprea_beta)

    # plot
    ax.plot(
        forward_error_cdf["x"],
        forward_error_cdf["F"],
        label=r"$\varepsilon_{fwd}^{true}$",
        color="k",
    )
    ax.plot(
        forward_error_model_u_cdf["x"],
        forward_error_model_u_cdf["F"],
        label=r"$\varepsilon_{fwd}^{\mathcal{U}-model}$",
        color="b",
        linestyle="--",
    )
    ax.plot(
        forward_error_model_beta_cdf["x"],
        forward_error_model_beta_cdf["F"],
        label=r"$\varepsilon_{fwd}^{\beta-model}$",
        color="goldenrod",
        linestyle="--",
    )
    ax.plot(
        gamma_det_cdf["x"],
        gamma_det_cdf["F"],
        label=r"DREA",
        color=COLORS["DREA"],
    )
    ax.plot(
        gamma_mprea_cdf["x"],
        gamma_mprea_cdf["F"],
        label=r"MPREA",
        color=COLORS["MPREA"],
    )
    ax.plot(
        gamma_vprea_u_cdf["x"],
        gamma_vprea_u_cdf["F"],
        label=r"VPREA ($\mathcal{U}$-model)",
        color=COLORS["VPREA_U"],
    )
    ax.plot(
        gamma_vprea_beta_cdf["x"],
        gamma_vprea_beta_cdf["F"],
        label=rf"VPREA ($\beta$-model; $\alpha$={alpha_plot:.2f})",
        color=COLORS["VPREA_beta"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_{fwd}$")
    ax.set_ylabel(r"$F_{\varepsilon_{fwd}}(\varepsilon_{fwd})$")
    ax.set_xlim(left=1e-3)
    return ax


def plot_forward_error_cdf():
    fig, axs = plt.subplots(
        1,
        len(args.dist),
        figsize=(7 * len(args.dist), 4),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    if len(args.dist) == 1:
        axs = [axs]
    for ii, ax in enumerate(axs):
        plot_forward_error_cdf_given_data_distribution(
            args.dist[ii], ax=ax, alpha_plot=1.6
        )
        ax.label_outer()
        if ii == 0:
            ax.legend(loc="best")

    savename = get_savefig_name(experiment="forward")
    plt.savefig(savename)


if __name__ == "__main__":
    plot_backward_error()
    # plot_forward_error_cdf()
