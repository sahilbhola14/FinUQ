"""
plotting utils for dot product
Author: Sahil Bhola, University of Michigan, 2026
"""
import pandas as pd
import matplotlib.pyplot as plt
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
        prefix = "backward_error"
    elif experiment == "forward":
        prefix = "forward_error"
    else:
        prefix = "invalid"
    base = (
        f"{prefix}_result_dot_product_"
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
        prefix = "forward_error"
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


def plot_backward_error_vs_n():
    # uniform data
    df_uniform = get_backward_error_data(model="uniform")
    # beta model
    df_beta = get_backward_error_data(model="beta")
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout="compressed")
    ax.plot(
        df_uniform["n"],
        df_uniform["backward_mean"],
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
        label=r"VPREA (Uniform model)",
        color="blue",
    )
    ax.plot(
        df_uniform["n"],
        df_beta["gamma_vprea"],
        label=r"VPREA ($\beta$ model)",
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
    plot_backward_error_vs_n()
    # plot_backward_error_vs_n()
    # plot_forward_error_cdf()
