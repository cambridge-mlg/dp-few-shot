import matplotlib


def tidy_plot(fig, axes):
    for ax in axes:
        for _, spine in ax.spines.items():
            spine.set_visible(True)  # You have to first turn them on
            spine.set_color('black')
            spine.set_linewidth(1)


def errorbar_min_max(v):
    return [min(v), max(v)]


def set_log_shot_scales(ax):
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 25, 50, 100, 250, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("shots", labelpad=0)


def set_shot_axis_y(ax):
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.set_ylim([0, 100])
    ax.set_ylabel("Accuracy (%)", labelpad=0)


def get_overlap_title(ds):
    if ds == "CIFAR-10":
        return "CIFAR-10 (high DDO)"
    elif ds == "CIFAR-100":
        return "CIFAR-100 (medium DDO)"
    elif ds == "SVHN":
        return "SVHN (low DDO)"


def get_long_param_name(short_name):
    names = ["All", "FiLM", "Head"]
    if short_name == "all":
        return names[0]
    elif short_name == "film":
        return names[1]
    elif short_name == "none":
        return names[2]
