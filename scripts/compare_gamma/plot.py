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
    "--prec",
    type=str,
    choices=["Double", "Single", "Half"],
    default="Single",
    help="Plotting precision",
)
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


def get_filename(model, confidence, alpha, beta):
    assert model.lower() in ["uniform", "beta"]
    base = (
        f"gamma_"
        f"{args.prec.lower()}_prec_"
        f"confidence_{confidence: 0.3f}_"
        f"{model.lower()}"
    )

    if model.lower() == "beta":
        assert alpha is not None, "alpha must not be none"
        assert beta is not None, "beta must not be none"
        base += f"_a_{alpha: 0.3f}_b_{beta: 0.3f}"

    return base + ".csv"


def get_data(model, confidence, alpha=None, beta=None):
    filename = get_filename(model, confidence, alpha, beta)
    df = pd.read_csv(filename)
    return df


def plot_gamma_vs_n():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), layout="compressed")
    linestyles = ["solid", "dashed", "dashdot"]
    cc, aa = np.meshgrid(args.confidence, args.alpha)
    cc = cc.ravel("F")
    aa = aa.ravel("F")
    df_uniform = get_data(model="uniform", confidence=args.confidence[0])
    ax.plot(df_uniform["n"], df_uniform["gamma_det"], color="k", label="DREA")
    for ii, confidence in enumerate(args.confidence):
        # get data
        df_uniform = get_data(model="uniform", confidence=confidence)

        if len(args.confidence) == 1:
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_mprea"],
                color="r",
                label=r"MPREA",
                linestyle=linestyles[ii],
            )
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_vprea"],
                color="b",
                label=r"VPREA ($\mathcal{{U}}$-model)",
                linestyle=linestyles[ii],
            )
        else:
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_mprea"],
                color="r",
                label=rf"MPREA ($\zeta$={confidence: .3f})",
                linestyle=linestyles[ii],
            )
            ax.plot(
                df_uniform["n"],
                df_uniform["gamma_vprea"],
                color="b",
                label=rf"VPREA ($\mathcal{{U}}$-model; $\zeta$={confidence: .3f})",
                linestyle=linestyles[ii],
            )

        for jj, alpha in enumerate(args.alpha):
            df_beta = get_data(
                model="beta", confidence=confidence, alpha=alpha, beta=args.beta
            )
            if len(args.confidence) == 1:
                ax.plot(
                    df_beta["n"],
                    df_beta["gamma_vprea"],
                    color="g",
                    label=rf"VPREA ($\beta$-model; $\alpha$={alpha: .3f})",
                    linestyle=linestyles[jj],
                )
            else:
                ax.plot(
                    df_beta["n"],
                    df_beta["gamma_vprea"],
                    color="g",
                    label=(
                        rf"VPREA ($\beta$-model; "
                        rf"$\alpha$={alpha: .3f}, "
                        rf"$\zeta$={confidence: .3f})"
                    ),
                    linestyle=linestyles[jj],
                )
    ax.axhline(1.0, color="grey", alpha=0.5, linewidth=3.0)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\gamma_n$")
    ax.set_ylim(top=1e1)
    if len(args.confidence) == 1:
        plt.savefig(f"gamma_vs_n_confidence_{args.confidence[0]}_beta_{args.beta}.png")
    else:
        plt.savefig(f"gamma_vs_n_vary_confidence_beta_{args.beta}.png")


if __name__ == "__main__":
    plot_gamma_vs_n()
