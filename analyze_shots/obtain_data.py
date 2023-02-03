import pandas as pd


def filter_eps(eps):
    if eps == "None":
        return "∞"
    elif float(eps) < 1:
        return "1"
    elif float(eps) < 2:
        return "2"
    elif float(eps) < 4:
        return "4"
    elif float(eps) < 8:
        return "8"
    else:
        ValueError()


COLUMNS = ["classifier", "feature_extractor", "learnable_params", "examples_per_class",
           "dataset", "eps", "delta", "accuracy", "seed", "total_steps", "sample_rate"]
COLUMNS += ["noise_multiplier", "train_batch_size", "learning_rate", "epochs",
            "max_grad_norm", "runtime", "optimizer", "slurm_job_id", "timestamp", "git_hash"]
COLUMNS += ["tune_params", "epochs_lb", "epochs_ub", "train_batch_size_lb", "train_batch_size_ub",
            "max_grad_norm_lb", "max_grad_norm_ub", "learning_rate_lb", "learning_rate_ub"]
COLUMNS += ["number_of_trials", "max_physical_batch_size", "number_of_gpus"]


def filter_optimizers(df, filter=True):
    # pick SGD for all params non private
    if not filter:
        return df
    else:
        return df[~((df["learnable_params"] == "all") & (df["epsilon"] == "∞") & (df["optimizer"] == "adam")) & ~(~((df["learnable_params"] == "all") & (df["epsilon"] == "∞")) & (df["optimizer"] == "sgd"))]


def rename_datasets(df):
    df["dataset"] = df["dataset"].apply(lambda x: "CIFAR-10" if x == "cifar10" else x)
    df["dataset"] = df["dataset"].apply(lambda x: "CIFAR-100" if x == "cifar100" else x)
    df["dataset"] = df["dataset"].apply(lambda x: "SVHN" if x == "svhn_cropped" else x)
    return df


def obtain_results():
    df = pd.read_csv("few_shot_results.csv", header=None, names=COLUMNS)
    df["epsilon"] = df["eps"].apply(filter_eps)
    df["shots"] = df["examples_per_class"]
    df["private"] = df["epsilon"].apply(lambda x: True if x != "∞" else False)
    df.drop_duplicates(subset=['classifier', 'feature_extractor', 'learnable_params', 'shots',
                       'dataset', 'epsilon', "optimizer", 'seed'], keep='last', inplace=True)
    df.sort_values(by=["epsilon"], inplace=True)
    df["accuracy"] = df["accuracy"] * 100
    df = df[df["shots"].isin([1, 5, 10, 25, 50, 100, 250, 500])]
    df = filter_optimizers(df)
    df = rename_datasets(df)
    return df
