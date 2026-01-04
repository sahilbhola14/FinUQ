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

parser.add_argument(
    "--dist",
    type=str,
    choices=dist_options,
    default="MinusOnePlusOne",
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
    "--alpha", type=float, default=2.01, help="Beta bound model alpha value"
)
parser.add_argument(
    "--beta", type=float, default=2.0, help="Beta bound model beta value"
)
parser.add_argument("--confidence", type=float, default=0.97, help="Bound confidence")
args = parser.parse_args()


def get_filename(experiment="backward", model="uniform"):
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
        f"distribution_{pretty_dist(args.dist)}_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"bound_model_{model.lower()}"
    )

    if model.lower() == "beta":
        base += f"_a_{args.alpha:0.5f}_b_{args.beta:0.5f}"

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
        f"distribution_{pretty_dist(args.dist)}_"
        f"bound_confidence_{args.confidence:0.5f}_"
        f"beta_dist_params"
    )

    base += f"_a_{args.alpha:0.5f}_b_{args.beta:0.5f}"

    return base + ".png"


def get_backward_error_data(model="uniform"):
    # get filename
    filename = get_filename("backward", model)
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


def plot_backward_error_vs_n():
    # uniform data
    df_uniform = get_backward_error_data(model="uniform")
    # beta model
    df_beta = get_backward_error_data(model="beta")
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="compressed")
    ax.plot(
        df_uniform["n"],
        df_uniform["backward_error_mean"],
        label=r"$e_{bwd}^{mean}$",
        color="k",
        linestyle="-",
        marker="^",
    )
    ax.plot(
        df_uniform["n"],
        df_uniform["backward_error_max"],
        label=r"$e_{bwd}^{max}$",
        color="k",
        linestyle="--",
        marker="v",
    )
    ax.plot(df_uniform["n"], df_uniform["gamma_det"], label=r"DREA", color="green")
    ax.plot(df_uniform["n"], df_uniform["gamma_mprea"], label=r"MPREA", color="red")
    ax.plot(
        df_uniform["n"],
        df_uniform["gamma_vprea"],
        label=r"VPREA ($\mathcal{U}$-model)",
        color="blue",
    )
    ax.plot(
        df_uniform["n"],
        df_beta["gamma_vprea"],
        label=r"VPREA ($\beta$-model)",
        color="blue",
        linestyle="--",
    )
    ax.set_xlabel("Vector size, $n$")
    ax.set_ylabel(r"$e_{bwd}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    savefig_name = get_savefig_name("backward")
    plt.savefig(savefig_name)
    print(f"backward error figure saved to: {savefig_name}")


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
    plt.savefig("test.png")
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


if __name__ == "__main__":
    # plot_backward_error_vs_n()
    plot_forward_error_cdf()
