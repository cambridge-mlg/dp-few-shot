from scipy.interpolate import interp1d
from plotting_utils import tidy_plot, get_long_param_name
from obtain_data import obtain_results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.title_fontsize": 10}, style="whitegrid")


def plot_multipliers(df_plot: pd.DataFrame):

    fig, ax = plt.subplots(1, 1, figsize=(4, 2), facecolor="white")
    sns.lineplot(data=df_plot[df_plot["epsilon"] != "∞"], x="epsilon", y="Multiplier", hue="dataset", hue_order=[
                 "SVHN", "CIFAR-10", "CIFAR-100"], legend="full", ax=ax, estimator="average", errorbar=None, sort=True, marker="o", linewidth=2, markersize=6)
    tidy_plot(fig, [ax])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, markerscale=2)
    sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 0))
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r"$ϵ$={}".format(eps) for eps in ["1", "2", "4", "8"]])
    ax.set_xlabel("")
    ax.set_ylim([0.9, 60])
    ax.set_xlim([-0.3, 3.3])
    ax.minorticks_off()
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(bottom=True, left=True)

    return fig, ax


def interpolate_data(df, learnable_params, datasets):
    interpolated = list()
    for i, d in enumerate(datasets):
        for e in df["epsilon"].unique():
            # get median values at 1, 5, 10, 25, 50, 100, 250, 500
            median_df = df[(df["dataset"] == d) & (df["epsilon"] == e) & (df["learnable_params"] == learnable_params)
                           ].groupby(["epsilon", "shots"])["accuracy"].median().reset_index()

            # interpolate other values
            f = interp1d(median_df["shots"], median_df["accuracy"], kind="linear")
            xnew = np.linspace(1, max(median_df["shots"]), num=max(median_df["shots"]), endpoint=True)
            intermediate_df = pd.DataFrame(np.array([xnew, f(xnew)]).T, columns=["shots", "accuracy"])
            intermediate_df["epsilon"] = e
            intermediate_df["learnable_params"] = learnable_params
            intermediate_df["dataset"] = d
            interpolated.append(intermediate_df)

    all_interpolated = pd.concat(interpolated)
    return all_interpolated


def compute_multipliers(df, learnable_params, datasets, shots):
    dfs = list()
    for k, d in enumerate(datasets):
        for i, t in enumerate(shots):
            # get median non-DP performance
            non_private_performance = df[(df["dataset"] == d) & (df["epsilon"] == "∞") & (
                df["learnable_params"] == learnable_params) & (df["shots"] == t)]["accuracy"].median()
            sorted_df = df[(df["accuracy"] >= non_private_performance) & (df["dataset"] == d) & (
                df["learnable_params"] == learnable_params)].sort_values(by=["shots"]).drop_duplicates(subset=["epsilon"])
            sorted_df["Multiplier"] = sorted_df["shots"] / sorted_df[sorted_df["epsilon"] == "∞"]["shots"].values[0]
            dfs.append(sorted_df)

    return pd.concat(dfs).sort_values(by="epsilon", ascending=True)


def plot_interpolation(df, feature_extractor, learnable_params, shots, title=False):
    datasets = ["SVHN", "CIFAR-10", "CIFAR-100"]

    # interpolate data that is not at S = [1, 5, 10, 25, 50, 100, 250, 500]
    interpolated_data = interpolate_data(df[(df["feature_extractor"] == feature_extractor)], learnable_params, datasets)

    # obtain multipliers
    multiplier_data = compute_multipliers(interpolated_data, learnable_params, datasets, shots)

    fig, ax = plot_multipliers(multiplier_data)

    if title:
        fe = "VIT-B" if feature_extractor == "vit-b-16" else "R-50"
        ax.set_title("{} ({}) for non-private $S={}$".format(fe, get_long_param_name(learnable_params), shots[0]))

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    df = obtain_results()
    for s in [5, 10]:
        for f in ["vit-b-16", "BiT-M-R50x1"]:
            for p in ["all", "film", "none"]:
                fig, ax = plot_interpolation(df, f, p, [s], title=True)
                if not (f == "vit-b-16" and p == "all"):
                    ax.get_legend().remove()
                fig.savefig("plots/interpolate/interpolated_shots_tradeoff_{}_{}_{}.pdf".format(f, p, s), bbox_inches='tight')

    for s in [5]:
        for f in ["vit-b-16"]:
            for p in ["film"]:
                fig, axes = plot_interpolation(df, f, p, [s], title=False)
                fig.savefig("plots/interpolate/interpolated_shots_tradeoff_{}_{}_{}_paper.pdf".format(f, p, s), bbox_inches='tight')
