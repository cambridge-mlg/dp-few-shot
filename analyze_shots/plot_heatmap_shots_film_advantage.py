import seaborn as sns

sns.set(rc={"pdf.fonttype":42, "ps.fonttype":42, "font.size":10,"axes.titlesize":10,"axes.labelsize":10, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize":10, "legend.title_fontsize": 10},style="whitegrid")
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from obtain_data import obtain_results
from plotting_utils import get_overlap_title, tidy_plot


def plot_heatmap(datas, titles):
    fig, axes = plt.subplots(1, len(datas), sharex=True, figsize=(8.5,2.5),  gridspec_kw={'width_ratios': [1.2, 1.1, 1.4]})

    x_labels = ["1", "5", "10", "25", "50", "100", "250", "500"]
    y_labels = ["1", "2", "4", "8", '$\infty$']
    
    for i, ax in enumerate(axes):
        heatmap = sns.heatmap(datas[i], cbar=i==2, ax=ax, annot=True, fmt='.1f', cmap=matplotlib.cm.seismic, vmin=-50, vmax=50)

        if i == 0:
            ax.set_yticklabels(y_labels)
            ax.set_ylabel('$ϵ$')
        else:
            ax.get_yaxis().set_visible(False)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('shots')
        ax.tick_params(bottom=True, left=True)
        ax.set_title(get_overlap_title(titles[i]))
    fig.tight_layout(pad=0.5, w_pad=0.1)
    return fig, axes

def create_data(feature_extractor, relative=False):
    df = obtain_results()
    df["n_seeds"] = df.groupby(['classifier', 'feature_extractor', 'learnable_params', 'shots', 'dataset', 'epsilon', "optimizer"])['seed'].transform('count')
    df = df[df["n_seeds"] >= 3]
    datas = []
    for ds in ["CIFAR-10", "CIFAR-100", "SVHN"]:
        head = df[(df["dataset"] == ds) & (df["feature_extractor"] == feature_extractor) & (df["learnable_params"] == "none")].pivot_table(index=["epsilon"],  columns=["shots"], values="accuracy", aggfunc="mean").values
        film = df[(df["dataset"] == ds) & (df["feature_extractor"] == feature_extractor) & (df["learnable_params"] == "film")].pivot_table(index=["epsilon"],  columns=["shots"], values="accuracy", aggfunc="mean").values
        if relative:
            datas.append((film-head)/film*100)
        else:
            datas.append(film-head)
    return datas

def main():
    for f in ["vit-b-16", "BiT-M-R50x1"]:
        for relative in [False]:
            datas = create_data(f, relative=relative)
            fig, ax = plot_heatmap(datas, ["CIFAR-10", "CIFAR-100", "SVHN"])
            computation = "relative" if relative else "absolute"
            tidy_plot(fig, ax)
            #fig.suptitle("{} ({})".format(f,computation))
            fig.savefig("plots/heatmaps/heatmap_shots_film_advantage_{}_{}.pdf".format(f, computation))

if __name__ == "__main__":
    main()