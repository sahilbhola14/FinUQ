"""
plotting utils for comparing gamma
Author: Sahil Bhola, University of Michigan, 2026
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.style.use("../journal.mplstyle")
parser = argparse.ArgumentParser(description="compare plotting config")
parser.add_argument(
    "--confidence", type=list, default=[0.99], help="confidence to plot"
)
parser.add_argument(
    "--alpha",
    type=list,
    default=[2.001, 2.01, 2.1],
    help="Beta bound model alpha value for each confidence",
)
parser.add_argument(
    "--beta", type=float, default=2.0, help="Beta bound model beta value"
)
args = parser.parse_args()


def get_filename(model, confidence, prec, alpha, beta):
    assert model.lower() in ["uniform", "beta"]
    assert prec.lower() in ["double", "single", "half"]
    base = (
        f"gamma_"
        f"{prec.lower()}_prec_"
        f"confidence_{confidence:0.3f}_"
        f"{model.lower()}"
    )

    if model.lower() == "beta":
        assert alpha is not None, "alpha must not be none"
        assert beta is not None, "beta must not be none"
        base += f"_a_{alpha:0.3f}_b_{beta:0.3f}"

    return base + ".csv"


def get_data(model, confidence, prec, alpha=None, beta=None):
    filename = get_filename(model, confidence, prec, alpha, beta)
    df = pd.read_csv(filename)
    return df


def plot_gamma_vs_n(prec, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), layout="compressed")

    linestyles = ["solid", "dashed", "dotted"]
    colors = {
        "DREA": "#000000",
        "MPREA": "red",
        "VPREA_U": "blue",
        "VPREA_beta": "goldenrod",
    }
    cc, aa = np.meshgrid(args.confidence, args.alpha)
    cc = cc.ravel("F")
    aa = aa.ravel("F")
    df_uniform = get_data(model="uniform", confidence=args.confidence[0], prec=prec)
    ax.plot(
        df_uniform["n"], df_uniform["gamma_det"], color=colors["DREA"], label="DREA"
    )
    for ii, confidence in enumerate(args.confidence):
        # get data
        df_uniform = get_data(model="uniform", confidence=confidence, prec=prec)

        if len(args.confidence) == 1:
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_mprea"],
                color=colors["MPREA"],
                label=r"MPREA",
                linestyle=linestyles[ii],
            )
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_vprea"],
                color=colors["VPREA_U"],
                label=r"VPREA ($\mathcal{{U}}$-model)",
                linestyle=linestyles[ii],
            )
        else:
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_mprea"],
                color=colors["MPREA"],
                label=rf"MPREA ($\zeta$={confidence: .3f})",
                linestyle=linestyles[ii],
            )
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_vprea"],
                color=colors["VPREA_U"],
                label=rf"VPREA ($\mathcal{{U}}$-model; $\zeta$={confidence: .3f})",
                linestyle=linestyles[ii],
            )

        for jj, alpha in enumerate(args.alpha):
            df_beta = get_data(
                model="beta",
                confidence=confidence,
                prec=prec,
                alpha=alpha,
                beta=args.beta,
            )
            if len(args.confidence) == 1:
                ax.plot(
                    df_beta["n"],
                    df_beta["gamma_vprea"],
                    color=colors["VPREA_beta"],
                    label=rf"VPREA ($\beta$-model; $\alpha$={alpha: .3f})",
                    linestyle=linestyles[jj],
                )
            else:
                ax.plot(
                    df_beta["n"],
                    df_beta["gamma_vprea"],
                    color=colors["VPREA_beta"],
                    label=(
                        rf"VPREA ($\beta$-model; "
                        rf"$\alpha$={alpha: .3f}, "
                        rf"$\zeta$={confidence: .3f})"
                    ),
                    linestyle=linestyles[jj],
                )
    ax.axhline(1.0, color="0.7", alpha=0.5, linewidth=2.0, linestyle="-")
    # ax.axvline(10, color="0.7", alpha=0.5, linewidth=3.0, linestyle="--")
    # ax.axvline(5, color="0.7", alpha=0.5, linewidth=3.0, linestyle=":")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Bound for $|\theta_n|$")
    ax.minorticks_off()
    ax.legend(loc="lower right")

    # n-range matching your plot
    n_ref = np.array(df_uniform["n"])

    # choose reference constants (pick something visually sensible)
    if prec == "Single":
        C_lin = 1e-6
        C_sqrt = 3e-8
    elif prec == "Half":
        C_lin = 1e-2
        C_sqrt = 3e-4
    else:
        C_lin = 1e-3
        C_sqrt = 1e-3

    ax.plot(
        n_ref,
        C_lin * n_ref,
        color="0.7",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=r"$\mathcal{O}(n)$",
    )

    ax.plot(
        n_ref,
        C_sqrt * np.sqrt(n_ref),
        color="0.7",
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label=r"$\mathcal{O}(\sqrt{n})$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def compare_gamma():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5), layout="compressed", sharex=True)
    # single precision
    plot_gamma_vs_n("Single", ax=ax[0])
    plot_gamma_vs_n("Half", ax=ax[1])

    ax[0].get_legend().remove()
    ax[0].text(
        5e1,
        1.5,
        r"$|\theta_n| \leq 1$",
        color="grey",
        fontsize=15,
        ha="right",
        va="bottom",
    )
    ax[0].text(
        100,
        3e-3,
        r"$\mathcal{O}(n)$",
        color="grey",
        fontsize=15,
        ha="left",
        va="top",
    )
    ax[0].text(
        1e6,
        2e-5,
        r"$\mathcal{O}(\sqrt{n})$",
        color="grey",
        fontsize=15,
        ha="left",
        va="top",
    )

    # ax[0].text(
    #     1.5,
    #     1e1 * 6e-2,  # near top
    #     r"$n = 5$",
    #     color="grey",
    #     fontsize=15,
    #     ha="left",
    #     va="top",
    #     rotation=90,
    # )

    ax[0].set_title(r"Single-precison, $\mathrm{fp}32$")
    ax[1].set_title(r"Half-precison, $\mathrm{fp}16$")
    ax[0].set_ylim(bottom=1e-7, top=1e1)
    ax[1].set_ylim(bottom=1e-4, top=1e1)
    ax[1].set_ylabel("")
    for _ax in ax:
        _ax.set_xlim(left=1, right=1e8)
    if len(args.confidence) == 1:
        plt.savefig(f"gamma_vs_n_confidence_{args.confidence[0]}_beta_{args.beta}.png")
    else:
        plt.savefig(f"gamma_vs_n_vary_confidence_beta_{args.beta}.png")


if __name__ == "__main__":
    compare_gamma()
