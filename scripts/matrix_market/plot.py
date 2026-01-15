"""
plotting utils for matrix market
Author: Sahil Bhola, University of Michigan, 2026
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse

plt.style.use("../journal.mplstyle")
parser = argparse.ArgumentParser(description="matrix product plotting config")

dist_options = ["Normal", "ZeroOne", "MinusOnePlusOne", "PowTwo", "Ones"]
prec_options = ["Double", "Single", "Half"]
COLORS = {
    "DREA": "#1B6F6A",
    "MPREA": "red",
    "VPREA_U": "blue",
    "VPREA_beta": "goldenrod",
}
LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]
MARKERSTYLES = ["o", "v", "s", "P"]


parser.add_argument(
    "--dist",
    type=list,
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
    type=list,
    default=[2.0],
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
        f"{prefix}_matvec_product_"
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
        prefix = experiment
    base = (
        f"{prefix}_result_matvec_product_"
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


def get_forward_error_data(model="uniform"):
    # get filename
    filename = get_filename("forward", model)
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

    return cdf_func, x_sorted, cdf_values


def plot_forward_error_cdf():
    # uniform data
    df_uniform = get_forward_error_data(model="uniform")
    # beta model
    df_beta = get_forward_error_data(model="beta")
    # get cdfs
    F_forward_error = empirical_cdf(df_uniform["forward_error"])

    F_forward_error_uniform_model = empirical_cdf(df_uniform["forward_error_model"])
    F_forward_error_beta_model = empirical_cdf(df_beta["forward_error_model"])

    F_forward_error_gamma_det = empirical_cdf(df_uniform["gamma_det"])
    F_forward_error_gamma_mprea = empirical_cdf(df_uniform["gamma_mprea"])

    F_forward_error_gamma_vprea_uniform = empirical_cdf(df_uniform["gamma_vprea"])
    F_forward_error_gamma_vprea_beta = empirical_cdf(df_beta["gamma_vprea"])

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="compressed")
    ax.plot(
        F_forward_error[1], F_forward_error[2], label=r"$e_{fwd}^{true}$", color="k"
    )
    ax.plot(
        F_forward_error_uniform_model[1],
        F_forward_error_uniform_model[2],
        label=r"$e_{fwd}^{\mathcal{U}-model}$",
        color="b",
        linestyle="--",
    )
    ax.plot(
        F_forward_error_beta_model[1],
        F_forward_error_beta_model[2],
        label=r"$e_{fwd}^{\beta-model}$",
        color="m",
        linestyle="--",
    )
    ax.plot(
        F_forward_error_gamma_det[1],
        F_forward_error_gamma_det[2],
        label=r"DREA",
        color="g",
    )
    ax.plot(
        F_forward_error_gamma_mprea[1],
        F_forward_error_gamma_mprea[2],
        label=r"MPREA",
        color="r",
    )
    ax.plot(
        F_forward_error_gamma_vprea_uniform[1],
        F_forward_error_gamma_vprea_uniform[2],
        label=r"VPREA ($\mathcal{U}$-model)",
        color="b",
    )
    ax.plot(
        F_forward_error_gamma_vprea_beta[1],
        F_forward_error_gamma_vprea_beta[2],
        label=r"VPREA ($\beta$-model)",
        color="m",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$e_{fwd}$")
    ax.set_ylabel(r"$F_{e_{fwd}}(e_{fwd})$")
    ax.legend()

    savefig_name = get_savefig_name("forward")
    # plt.savefig(savefig_name)
    plt.savefig("forward_test.png")
    print(f"forward error figure saved to: {savefig_name}")


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


def plot_backward_error_given_data_distribution(dist, ax, x_data="n"):
    # uniform data
    df_uniform = get_backward_error_data(model="uniform", dist=dist)
    n = df_uniform[x_data]
    assert x_data in ["n", "nnz_to_size_ratio"]
    backward_error_mean = df_uniform["backward_error_mean"]
    backward_error_max = df_uniform["backward_error_max"]
    gamma_det = df_uniform["gamma_det"]
    gamma_mprea = df_uniform["gamma_mprea"]
    gamma_vprea_u = df_uniform["gamma_vprea"]

    # add jitter
    backward_error_mean = backward_error_mean.clip(lower=2e-8)
    backward_error_max = backward_error_max.clip(lower=2e-8)

    alpha_ref = 0.9

    s_prob = 150
    s_ref = 75

    marker_style = {
        "DREA": "o",
        "MPREA": "P",
        "VPREA_u": "v",
        "VPREA_beta": "*",
    }

    ax.scatter(
        n,
        backward_error_mean,
        label=r"$e_{bwd}^{mean}$",
        color="k",
        alpha=alpha_ref,
        marker="X",
        s=s_ref,
    )
    ax.scatter(
        n,
        backward_error_max,
        label=r"$e_{bwd}^{max}$",
        color="k",
        alpha=alpha_ref,
        marker="s",
        s=s_ref,
    )
    # ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0, linestyle="-")
    ax.scatter(
        n,
        gamma_det,
        label=r"DREA",
        color=COLORS["DREA"],
        marker=marker_style["DREA"],
        s=s_ref,
    )
    ax.scatter(
        n,
        gamma_mprea,
        label=r"MPREA",
        color=COLORS["MPREA"],
        marker=marker_style["MPREA"],
        s=s_prob,
        zorder=3,
    )
    ax.scatter(
        n,
        gamma_vprea_u,
        label=r"VPREA ($\mathcal{U}$-model)",
        color=COLORS["VPREA_U"],
        marker=marker_style["VPREA_u"],
        s=s_prob,
        zorder=3,
    )
    for ii, alpha in enumerate(args.alpha):
        df_beta = get_backward_error_data(
            model="beta", dist=dist, alpha=alpha, beta=args.beta
        )
        gamma_vprea_beta = df_beta["gamma_vprea"]
        ax.scatter(
            n,
            gamma_vprea_beta,
            label=rf"VPREA ($\beta$-model; $\alpha$={alpha:.2f})",
            color=COLORS["VPREA_beta"],
            marker=marker_style["VPREA_beta"],
            s=s_prob,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()

    if x_data == "n":
        ax.set_xlabel(r"Matrix dimension, $n$")
    elif x_data == "nnz_to_size_ratio":
        # ax.set_xlabel(r"$\frac{\#\text{ non-zero entries}}{n^2}$")
        ax.set_xlabel(r"Matrix density ($\mathrm{nnz}(\mathbf{A}) / n^2$)")
    ax.set_ylabel(r"$e_{bwd}$")
    if args.prec.lower() == "single":
        # ax.set_xlim(left=10, right=10**6)
        if x_data == "n":
            ax.set_xlim(left=10, right=10000)
        else:
            ax.set_xlim(left=1e-4, right=1)

        ax.set_ylim(bottom=1e-8, top=1e-1)

    elif args.prec.lower() == "half":
        ax.set_xlim(left=10, right=10**5)

    return ax


def plot_backward_error(x_data="n"):
    fig, axs = plt.subplots(
        1,
        len(args.dist),
        figsize=(6 * len(args.dist), 5),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    for ii, ax in enumerate(axs):
        plot_backward_error_given_data_distribution(args.dist[ii], ax=ax, x_data=x_data)
        ax.label_outer()
        if ii == 0:
            ax.legend(ncol=2, loc="best")

    savename = get_savefig_name(experiment=f"backward_error_vs_{x_data}")
    plt.savefig(savename)


if __name__ == "__main__":
    # plot_backward_error()
    plot_backward_error(x_data="nnz_to_size_ratio")
    plot_backward_error(x_data="n")
    # plot_forward_error_cdf()
