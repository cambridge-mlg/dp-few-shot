from plotting_utils import errorbar_min_max, tidy_plot, set_log_shot_scales, get_overlap_title, set_shot_axis_y
from obtain_data import obtain_results
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.title_fontsize": 10}, style="whitegrid")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def make_plot_eps(df, datasets, eps):
    params = ["all", "film", "none"]
    fig, axes = plt.subplots(1, len(params), figsize=(8.5, 1.8), sharey=True, facecolor="white")
    for i, d in enumerate(datasets):
        df_eps = df[(df["dataset"] == d) & (df["epsilon"].isin(eps)) & (df["learnable_params"].isin(params))]
        df_eps["feature_extractor"] = df_eps["feature_extractor"].apply(lambda x: "VIT-B" if x == "vit-b-16" else "R-50")

        # filter best config out of All, FiLM and Head
        rows = []
        for s in df_eps["shots"].unique():
            for e in df_eps["epsilon"].unique():
                best_config = df_eps[(df_eps["shots"] == s) & (df_eps["epsilon"] == e)].groupby("learnable_params")["accuracy"].mean(
                ).reset_index().sort_values(by="accuracy", ascending=False)["learnable_params"].values[0]
                rows.append(df_eps[(df_eps["shots"] == s) & (df_eps["learnable_params"] == best_config)])
        df_combined = pd.concat(rows)
        set_shot_axis_y(axes[i])
        axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
        set_log_shot_scales(axes[i])
        g = sns.lineplot(data=df_combined, x="shots", y="accuracy", estimator="median", hue="feature_extractor", hue_order=[
                         "VIT-B", "R-50"], markersize=3, linewidth=1, errorbar=errorbar_min_max, err_style="band", marker="o", ax=axes[i], legend="full", markers=["D", "X", "o"], palette=plt.rcParams['axes.prop_cycle'].by_key()['color'][:2])
        axes[i].tick_params(bottom=True, left=i == 0)
        plt.setp(axes[i].get_xticklabels(), rotation=30, horizontalalignment='center')

        axes[i].set_title(get_overlap_title(d))
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend(handles=handles, labels=labels)
        axes[i].set_ylabel(r"$ϵ={}${} Accuracy (%)".format(eps[0] if not eps[0] == "∞" else "\infty", "\n"))
    tidy_plot(fig, axes)

    return fig, axes


if __name__ == "__main__":
    df = obtain_results()
    for eps in ["1", "2", "4", "8", "∞"]:
        fig, axes = make_plot_eps(df, ["CIFAR-10", "CIFAR-100", "SVHN"], [eps])
        for i, ax in enumerate(axes):
            if i == 0 and eps == "1":
                sns.move_legend(ax, "best")
            else:
                ax.get_legend().remove()
        for i, ax in enumerate(axes):
            if i != 0:
                ax.get_yaxis().label.set_visible(False)
        fig.tight_layout(w_pad=0.1)
        if eps == "∞":
            eps = "-1"
        fig.savefig("plots/fe_cp/epsilon_shots_fe_cp_{}.pdf".format(eps))
