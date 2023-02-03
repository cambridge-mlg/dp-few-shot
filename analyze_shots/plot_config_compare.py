import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from obtain_data import obtain_results
from plotting_utils import (errorbar_min_max, get_long_param_name,
                            get_overlap_title, set_log_shot_scales,
                            set_shot_axis_y, tidy_plot)

pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.title_fontsize": 10}, style="whitegrid")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_main_plot(df, datasets: str, eps_list=["1", "2", "4", "8", "∞"], learnable_params=["all", "film", "none"]):
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.5, 4/16*8.5), sharey=True, facecolor="white")
    for i, d in enumerate(datasets):
        df_eps = df[(df["dataset"] == d) & (df["epsilon"].isin(eps_list)) & (df["learnable_params"].isin(learnable_params))]
        df_eps["learnable params"] = df_eps["learnable_params"].apply(lambda x: get_long_param_name(x))
        g = sns.lineplot(data=df_eps, x="shots", y="accuracy", estimator="median", hue="learnable params", style="epsilon", markersize=4, linewidth=2, style_order=eps_list,
                         errorbar=errorbar_min_max, err_style="band", ax=axes[i], legend="full", markers=["o", "X"], hue_order=["All", "Head", "FiLM", ], palette=["C0", "C2", "C1"])

        # legend
        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            labels[4] = ""
            # switch Head and FiLM for legend
            handles[2], handles[3] = handles[3], handles[2]
            labels[2], labels[3] = labels[3], labels[2]

            handles[5], handles[6] = handles[6], handles[5]
            labels[5], labels[6] = r"$ϵ$={}".format(labels[6]), r"$ϵ$={}".format(labels[5])
            axes[i].legend(handles=handles[1:], labels=labels[1:], markerscale=1)
            sns.move_legend(g, "upper left", bbox_to_anchor=(-0.60, 1.05))
        else:
            if axes[i].get_legend():
                axes[i].get_legend().remove()

        axes[i].set_title(get_overlap_title(d))
        axes[i].tick_params(bottom=True, left=i == 0, length=3)

        set_shot_axis_y(axes[i])
        # log scale
        set_log_shot_scales(axes[i])
        axes[i].set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        tidy_plot(fig, axes)
    fig.tight_layout(w_pad=0.1)

    axes[0].set_yticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, "", 100], minor=False)
    return fig, axes


def add_insert(fig, axes, eps, feature_extractor):
    for i, d in enumerate(["CIFAR-10", "CIFAR-100"]):
        axin = axes[i].inset_axes([0.7115, 0.02, 0.257, 0.6])
        axin.set_xscale("log")
        df_fe = df[(df["dataset"] == d) & (df["epsilon"].isin(eps) & (df["feature_extractor"] == feature_extractor))]
        df_fe["learnable params"] = df_fe["learnable_params"].apply(lambda x: get_long_param_name(x))
        sns.lineplot(data=df_fe, x="shots", y="accuracy", estimator="median", hue="learnable params", style="epsilon", markersize=4, linewidth=2, style_order=eps,
                     errorbar=errorbar_min_max, err_style="band", ax=axin, legend=None, markers=["o", "X"], hue_order=["All", "Head", "FiLM"], palette=["C0", "C2", "C1"])
        axin.set_yticks([80, 90, 95, 100])
        axin.set_xticks([100, 250, 500])
        axin.set_ylim([92, 100] if d == "CIFAR-10" else [80, 95])
        axin.set_xlim([95, 550])
        axin.set_ylabel("")
        axin.set_xlabel("")
        axin.set_xticklabels([], minor=True)
        axin.set_yticklabels([], minor=True)
        axin.set_xticklabels([], minor=False)
        axin.set_yticklabels([], minor=False)
        tidy_plot(fig, [axin])
        axes[i].indicate_inset_zoom(axin, edgecolor="black")


def make_plot_eps(df, datasets, eps, feature_extractor):
    params = ["all", "film", "none"]
    df_plot = df[(df["feature_extractor"] == feature_extractor)]
    fig, axes = plot_main_plot(df_plot, datasets, learnable_params=params, eps_list=eps)

    if feature_extractor == "vit-b-16":
        add_insert(fig, axes, eps, feature_extractor)

    return fig


if __name__ == "__main__":
    for f in ["vit-b-16", "BiT-M-R50x1"]:
        df = obtain_results()
        fig = make_plot_eps(df, ["CIFAR-10", "CIFAR-100", "SVHN"], ["2", "∞"], f)
        fig.savefig("plots/compare_config_shots_2_inf_all_datasets_{}.pdf".format(f), bbox_inches='tight')
