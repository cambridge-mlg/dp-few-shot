import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from obtain_data import obtain_results
from plotting_utils import (errorbar_min_max, get_overlap_title,
                            set_log_shot_scales, set_shot_axis_y, tidy_plot)

pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.title_fontsize": 10}, style="whitegrid")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def make_plot_eps(df, datasets, eps, feature_extractor):
    params = ["all", "film", "none"]
    df = df[(df["feature_extractor"] == feature_extractor)]
    fig, axes = plt.subplots(1, len(params), figsize=(8.5, 4/16*8.5), sharey=True, facecolor="white")
    for i, d in enumerate(datasets):
        df_eps = df[(df["dataset"] == d) & (df["epsilon"].isin(eps)) & (df["learnable_params"].isin(params))]

        # filter best config out of All, FiLM and Head
        rows = []
        for e in df["epsilon"].unique():
            for s in df_eps["shots"].unique():
                best_config = df_eps[(df_eps["shots"] == s) & (df_eps["epsilon"] == e)].groupby("learnable_params")["accuracy"].mean(
                ).reset_index().sort_values(by="accuracy", ascending=False)["learnable_params"].values[0]
                rows.append(df_eps[(df_eps["shots"] == s) & (df_eps["epsilon"] == e) & (df_eps["learnable_params"] == best_config)])
        df_combined = pd.concat(rows)

        g = sns.lineplot(data=df_combined, x="shots", y="accuracy", estimator="median", hue="epsilon", markersize=3, linewidth=1, errorbar=errorbar_min_max, err_style="band",
                         marker="o", ax=axes[i], legend="full", markers=["D", "X", "o"], hue_order=eps, palette=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(eps)])
        axes[i].set_title(get_overlap_title(d))

        # log scale
        set_log_shot_scales(axes[i])
    tidy_plot(fig, axes)

    handles, labels = axes[0].get_legend_handles_labels()
    labels[4] = r"$\infty$"
    axes[0].legend(handles=handles[::-1], labels=labels[::-1])
    for i, ax in enumerate(axes):
        if i == 0:
            sns.move_legend(axes[0], "upper left", bbox_to_anchor=(-0.5, 1.05), title=r"$ϵ$")
            set_shot_axis_y(axes[0])
        else:
            if ax.get_legend():
                ax.get_legend().remove()
        ax.tick_params(bottom=True, left=i == 0, length=3)
    fig.tight_layout(w_pad=0.1)
    return fig


if __name__ == "__main__":
    df = obtain_results()
    for f in ["vit-b-16", "BiT-M-R50x1"]:
        fig = make_plot_eps(df, ["CIFAR-10", "CIFAR-100", "SVHN"], ["1", "2", "4", "8", "∞"], f)
        fig.savefig("plots/best_config_shots_all_datasets_{}.pdf".format(f), bbox_inches='tight')
