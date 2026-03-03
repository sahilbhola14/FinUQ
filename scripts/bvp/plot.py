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
    type=float,
    default=1.6,
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
    linewidth = 4
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
    ax.plot(n, gamma_det, label=r"DREA", color=COLORS["DREA"], linewidth=linewidth)
    ax.plot(n, gamma_mprea, label=r"MPREA", color=COLORS["MPREA"], linewidth=linewidth)
    ax.plot(
        n,
        gamma_vprea_u,
        label=r"VPREA ($\mathcal{U}$-model)",
        color=COLORS["VPREA_U"],
        linewidth=linewidth,
    )
    plot_alpha = [args.alpha] if isinstance(args.alpha, float) else args.alpha
    for ii, alpha in enumerate(plot_alpha):
        df_beta = get_backward_error_data(
            model="beta", prec=prec, alpha=alpha, beta=args.beta
        )
        gamma_vprea_beta = df_beta["gamma_vprea"]
        ax.plot(
            n,
            gamma_vprea_beta,
            label=rf"VPREA ($\beta$-model; $\alpha$={alpha:.2f})",
            color=COLORS["VPREA_beta"],
            linestyle="-.",
            linewidth=linewidth,
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


def plot_backward_error():
    fig, axs = plt.subplots(
        1,
        len(args.prec),
        figsize=(12, 5),
        # sharex=True,
        # sharey=True,
        layout="compressed",
    )
    for ii, ax in enumerate(axs):
        plot_backward_error_given_precision(args.prec[ii].lower(), ax=ax)
        # ax.label_outer()
        if ii == 0:
            # ax.legend(ncol=2, loc="upper left", fontsize=20)
            ax.set_title(r"Single-precison, $\mathrm{fp}32$", fontsize=22)
        if ii == 1:
            ax.set_ylabel(None)
            ax.set_title(r"Half-precison, $\mathrm{fp}16$", fontsize=22)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
        frameon=True,
        fontsize=20,
        bbox_to_anchor=(1.17, 0.83),
    )

    savename = get_savefig_name(experiment="backward")
    plt.savefig(savename)


def plot_forward_error_cdf_for_multiple_discretizations_given_precision(prec, axs):
    linewidth = 4
    df_uniform = get_forward_error_data(model="uniform", prec=prec)
    alpha_val = args.alpha[0] if isinstance(args.alpha, list) else args.alpha
    df_beta = get_forward_error_data(
        model="beta", prec=prec, alpha=alpha_val, beta=args.beta
    )
    n_all = sorted(set(df_uniform["n"]))
    assert len(axs) <= len(n_all)
    print(f"Plotting forward error cdf for alpha: {alpha_val}")

    if len(n_all) == len(axs):
        n_plot = n_all
    else:
        idx = np.linspace(0, len(n_all) - 1, len(axs), dtype=int)
        n_plot = [n_all[i] for i in idx]

    for ax, n in zip(axs, n_plot):
        df_uniform_n = df_uniform[df_uniform["n"] == n]
        df_beta_n = df_beta[df_beta["n"] == n]

        forward_error_cdf = empirical_cdf(df_uniform_n["forward_error"])
        forward_error_model_u_cdf = empirical_cdf(df_uniform_n["forward_error_model"])
        forward_error_model_beta_cdf = empirical_cdf(df_beta_n["forward_error_model"])
        gamma_det_cdf = empirical_cdf(df_uniform_n["gamma_det"])
        gamma_mprea_cdf = empirical_cdf(df_uniform_n["gamma_mprea"])
        gamma_vprea_u_cdf = empirical_cdf(df_uniform_n["gamma_vprea"])
        gamma_vprea_beta_cdf = empirical_cdf(df_beta_n["gamma_vprea"])

        ax.plot(
            forward_error_cdf["x"],
            forward_error_cdf["F"],
            label=r"$\varepsilon_{fwd}^{true}$",
            color="k",
            linewidth=linewidth,
        )
        ax.plot(
            forward_error_model_u_cdf["x"],
            forward_error_model_u_cdf["F"],
            label=r"$\varepsilon_{fwd}^{\mathcal{U}-model}$",
            color=COLORS["VPREA_U"],
            linestyle="--",
            linewidth=linewidth,
        )
        ax.plot(
            forward_error_model_beta_cdf["x"],
            forward_error_model_beta_cdf["F"],
            label=r"$\varepsilon_{fwd}^{\beta-model}$",
            color=COLORS["VPREA_beta"],
            linestyle="--",
            linewidth=linewidth,
        )
        ax.plot(
            gamma_det_cdf["x"],
            gamma_det_cdf["F"],
            label=r"DREA",
            color=COLORS["DREA"],
            linewidth=linewidth,
        )
        ax.plot(
            gamma_mprea_cdf["x"],
            gamma_mprea_cdf["F"],
            label=r"MPREA",
            color=COLORS["MPREA"],
            linewidth=linewidth,
        )
        ax.plot(
            gamma_vprea_u_cdf["x"],
            gamma_vprea_u_cdf["F"],
            label=r"VPREA ($\mathcal{U}$-model)",
            color=COLORS["VPREA_U"],
            linewidth=linewidth,
        )
        ax.plot(
            gamma_vprea_beta_cdf["x"],
            gamma_vprea_beta_cdf["F"],
            label=rf"VPREA ($\beta$-model; $\alpha$={alpha_val:.2f})",
            color=COLORS["VPREA_beta"],
            linewidth=linewidth,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\varepsilon_{fwd}$", size=25)
        ax.set_ylabel(r"$F_{\varepsilon_{fwd}}(\varepsilon_{fwd})$", size=25)
        ax.set_title(rf"$M={n}$", size=22)
        ax.tick_params(axis="both", which="major", labelsize=22)

    return axs


def plot_forward_error_cdf_for_multiple_discretizations(n_discrete_plot=2):
    fig, axs = plt.subplots(
        n_discrete_plot,
        len(args.prec),
        figsize=(6 * len(args.prec), 4.2 * n_discrete_plot),
        layout="compressed",
        sharex="col",
        sharey="row",
    )
    if n_discrete_plot == 1 and len(args.prec) == 1:
        axs = np.array([[axs]])
    elif n_discrete_plot == 1:
        axs = np.expand_dims(axs, axis=0)
    elif len(args.prec) == 1:
        axs = np.expand_dims(axs, axis=1)
    for ii, prec in enumerate(args.prec):
        plot_forward_error_cdf_for_multiple_discretizations_given_precision(
            prec.lower(), axs[:, ii]
        )

        for _ax in axs[:, ii]:
            _ax.label_outer()
            _ax.set_ylim(1e-2, None)

    for ii, prec in enumerate(args.prec):

        if prec.lower() == "single":
            if args.num_samples == 100:
                axs[0, ii].set_xlim(1e-8, 1e-4)
            else:
                axs[0, ii].set_xlim(1e-8, 1e-3)
        elif prec.lower() == "half":
            if args.num_samples == 100:
                axs[0, ii].set_xlim(1e-4, 1e0)
            else:
                axs[0, ii].set_xlim(1e-4, 1e1)

    # Get handles from one representative axis
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(1.17, 0.74),
        fontsize=20,
    )

    plt.savefig(f"forward_error_cdf_num_samples_{args.num_samples}.png")


def plot_forward_error_vs_discretization_intervals_fixed_precision(prec, ax):
    df_uniform = get_forward_error_data(model="uniform", prec=prec)
    alpha_val = args.alpha[0] if isinstance(args.alpha, list) else args.alpha
    df_beta = get_forward_error_data(
        model="beta", prec=prec, alpha=alpha_val, beta=args.beta
    )
    print(f"Plotting forward error vs discretization for alpha: {alpha_val}")

    ax.plot(
        df_uniform["n"],
        df_uniform["forward_error"],
        label=r"$\varepsilon_{fwd}^{true}$",
        color="k",
        linestyle="-",
        marker="X",
    )
    ax.plot(
        df_uniform["n"],
        df_uniform["forward_error_model"],
        label=r"$\varepsilon_{fwd}^{\mathcal{U}\mathrm{-model}}$",
        color=COLORS["VPREA_U"],
        linestyle="--",
        marker="s",
    )
    ax.plot(
        df_beta["n"],
        df_beta["forward_error_model"],
        label=r"$\varepsilon_{fwd}^{\beta\mathrm{-model}}$",
        color=COLORS["VPREA_beta"],
        linestyle="--",
        marker="s",
    )
    ax.plot(
        df_uniform["n"], df_uniform["gamma_det"], label="DREA", color=COLORS["DREA"]
    )
    ax.plot(
        df_uniform["n"],
        df_uniform["gamma_mprea"],
        label="MPREA",
        color=COLORS["MPREA"],
    )
    ax.plot(
        df_uniform["n"],
        df_uniform["gamma_vprea"],
        label=r"VPREA ($\mathcal{U}$-model)",
        color=COLORS["VPREA_U"],
    )
    ax.plot(
        df_beta["n"],
        df_beta["gamma_vprea"],
        label=rf"VPREA ($\beta$-model; $\alpha$={alpha_val:.2f})",
        color=COLORS["VPREA_beta"],
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Discretization intervals, $M$")
    ax.set_ylabel(r"$\varepsilon_{fwd}$")
    ax.tick_params(axis="both", which="major", labelsize=18)

    return ax


def plot_forward_error_vs_discretization_intervals():
    fig, axs = plt.subplots(
        1,
        len(args.prec),
        figsize=(5.5 * len(args.prec), 4),
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    if len(args.prec) == 1:
        axs = [axs]
    for (ax, prec) in zip(axs, args.prec):
        plot_forward_error_vs_discretization_intervals_fixed_precision(prec.lower(), ax)
        ax.label_outer()
    axs[0].legend(ncol=2, loc="best", fontsize=12)
    plt.savefig(f"forward_error_vs_discretization_num_samples_{args.num_samples}.png")


def plot_all_forward_error():
    # plot forward error cdf for various discretization intervals
    plot_forward_error_cdf_for_multiple_discretizations()
    # plot forward error vs number of discretization intervals
    # (for fixed Monte carlo samples)
    # plot_forward_error_vs_discretization_intervals()


if __name__ == "__main__":
    # plot_backward_error()
    plot_all_forward_error()
