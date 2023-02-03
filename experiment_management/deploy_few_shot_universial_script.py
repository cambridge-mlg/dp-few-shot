#!/usr/bin/env python3

import argparse
import textwrap
from os import path
from subprocess import call


def parse_command_line():
    parser = argparse.ArgumentParser()

    # experiment params
    parser.add_argument('--dataset', help='dataset for testing', choices=["cifar10", "cifar100", "svhn_cropped"])
    parser.add_argument("--feature_extractor", default="BiT-M-R50x1", choices=['vit-b-16', 'BiT-M-R50x1'], help="Feature extractor to use.")
    parser.add_argument("--classifier", choices=['linear'], default='linear', help="Which classifier to use.")
    parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                        help="Which feature extractor parameters to learn.")
    parser.add_argument("--examples_per_class", type=int, default=None,
                        help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
    parser.add_argument("--target_epsilon", type=float, default=8.0, help="Maximum value of epsilon allowed.")

    # memory options
    parser.add_argument("--max_physical_batch_size", type=int, default=400, help="Maximum physical batch size")
    parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="test batch size.")

    # seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for determinstic results.")
    # optimizer
    parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')

    # hyperparameter ranges
    parser.add_argument("--epochs_lb", type=int, default=1, help="LB of fine-tune epochs.")
    parser.add_argument("--epochs_ub", type=int, default=200, help="UB of fine-tune epochs.")
    parser.add_argument("--train_batch_size_lb", type=int, default=10, help="LB of Batch size.")
    parser.add_argument("--train_batch_size_ub", type=int, default=None, help="UB of Batch size.")
    parser.add_argument("--max_grad_norm_lb", type=float, default=0.2, help="LB of maximum gradient norm.")
    parser.add_argument("--max_grad_norm_ub", type=float, default=10.0, help="UB of maximum gradient norm.")
    parser.add_argument("--learning_rate_lb", type=float, default=1e-7, help="LB of learning rate")
    parser.add_argument("--learning_rate_ub", type=float,  default=1e-2, help="UB of learning rate")

    # optuna options
    parser.add_argument("--save_optuna_study", dest="save_optuna_study", default=True, action="store_true",
                        help="If true, save optuna studies.")
    parser.add_argument("--number_of_trials", type=int, default=20, help="The number of trials for optuna")

    # cluster options
    parser.add_argument("--job_maximum_hours", type=float, default=0.1, help="The maximum hours the job is running.")
    parser.add_argument("--number_of_gpus", type=int, default=1, help="The number of gpus used.")

    args = parser.parse_args()
    return args


def get_run_option_string(args):
    options = list()
    # cluster params
    options.append("--download_path_for_tensorflow_datasets {} -c {}".format(TF_DATASET_PATH, CHECKPOINT_DIR))

    # experiment params
    options.append("--dataset {} --feature_extractor {} ".format(args.dataset, args.feature_extractor))
    if args.target_epsilon != -1:
        options.append("--private")
    options.append("--learnable_params {} --examples_per_class {} --tune_params".format(args.learnable_params, args.examples_per_class))
    options.append("--target_epsilon {}".format(args.target_epsilon))

    # memory options
    options.append("--max_physical_batch_size {} --test_batch_size {}".format(args.max_physical_batch_size, args.test_batch_size))

    # seed
    options.append("--seed {}".format(args.seed))

    # optimizer
    options.append("--optimizer {}".format(args.optimizer))

    # hyperparameter ranges
    options.append("--epochs_lb {} --epochs_ub {}".format(args.epochs_lb, args.epochs_ub))
    # set train_batch_size_ub to all data samples if None

    options.append("--train_batch_size_lb {} --train_batch_size_ub {}".format(args.train_batch_size_lb, _determine_batch_size_ub(args)))
    options.append("--max_grad_norm_lb {} --max_grad_norm_ub {}".format(args.max_grad_norm_lb, args.max_grad_norm_ub))
    options.append("--learning_rate_lb {} --learning_rate_ub {}".format(args.learning_rate_lb, args.learning_rate_ub))

    # optuna options
    if args.save_optuna_study:
        options.append("--save_optuna_study")
    options.append("--number_of_trials {}".format(args.number_of_trials))

    return " ".join(options)


def _determine_batch_size_ub(args):
    if args.train_batch_size_ub is not None:
        return args.train_batch_size_ub
    else:
        if args.examples_per_class is None:  # few shot
            return 1000
        elif args.examples_per_class == -1:
            return _get_size_of_dataset(args.dataset)
        else:
            return args.examples_per_class * _get_number_of_classes(args.dataset)


def _get_size_of_dataset(dataset):
    if dataset in ["cifar10", "cifar100"]:
        return 50000
    elif dataset == "svhn_cropped":
        return 73257
    else:
        ValueError("dataset unknown")


def _get_number_of_classes(dataset):
    if dataset in ["cifar10", "svhn_cropped"]:
        return 10
    elif dataset == "cifar100":
        return 100
    else:
        ValueError("dataset unknown")


def _get_run_script(number_of_gpus):
    if number_of_gpus == 1:
        return "run.py"
    else:
        return "run_distributed.py"


def create_job_name(args):

    if args.dataset == "cifar10":
        prefix = "c10"
    elif args.dataset == "cifar100":
        prefix = "c100"
    elif args.dataset == "svhn_cropped":
        prefix = "sv"
    else:
        raise ValueError("Jobname is not defined for dataset")

    if args.learnable_params == "none":
        prefix += ".h"
    elif args.learnable_params == "film":
        prefix += ".f"
    elif args.learnable_params == "all":
        prefix += ".a"
    else:
        raise ValueError("Jobname is not defined for params")

    if args.examples_per_class is not None:
        examples_per_class = args.examples_per_class
    else:
        examples_per_class = "VTAB"

    return "{}_{}_{}_{}_{}_{}".format(prefix, int(args.target_epsilon), examples_per_class, args.seed, args.feature_extractor, args.optimizer)


def parse_job_time(args):
    if args.job_maximum_hours <= 0:
        raise ValueError("Job time invalid")
    hours = int(args.job_maximum_hours)
    minutes = int(args.job_maximum_hours % 1 * 60)
    return "{}:{:02d}:00".format(hours, minutes)


def compute_memory(args):
    if args.dataset == "cifar100":
        return str(args.number_of_gpus * 95000)
    else:
        return str(args.number_of_gpus * 45000)


if __name__ == "__main__":
    # CLUSTER_SPECIFIC_PATHS
    with open("dp-fsl/hyperparam_optimization/deploy_config.txt") as file:
        lines = [line.rstrip() for line in file]

    TF_DATASET_PATH = lines[0]
    CHECKPOINT_DIR = lines[1]
    JOB_SCRIPT_SAVE_LOCATION = lines[2]
    LOG_SAVE_LOCATION = lines[3]
    SCRIPT_LOCATION = lines[4]
    ENV_ACTIVATION = lines[5]
    SLURM_ACCOUNT = lines[6]

    F_STRING = """
    #!/bin/bash
    ############## This section states the requirements the job requires:
    #SBATCH --job-name={job_name:s}
    #SBATCH -o {job_log_location:s}
    #SBATCH --account={slurm_account:s}
    #SBATCH --partition=gpu
    #SBATCH --time={job_max_time:s}
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={number_of_cpus:s}
    #SBATCH --mem={memory:s}
    #SBATCH --gres=gpu:v100:{number_of_gpus:s}
    ############## Here starts the actual UNIX commands and payload: 
    module purge
    module load git
    {env_activation:s}
    cd {script_location:s}
    python {run_script:s} {run_option_string:s}
    """

    # remove empty first line and indent
    F_STRING = "\n".join(F_STRING.split("\n")[1:])
    F_STRING = textwrap.dedent(F_STRING)

    args = parse_command_line()
    job_name = create_job_name(args)

    format_args = {
        "run_script": _get_run_script(args.number_of_gpus),
        "run_option_string": get_run_option_string(args),
        "job_name": job_name,
        "job_max_time": parse_job_time(args),
        "job_log_location": path.join(LOG_SAVE_LOCATION, "{:s}_ID_%j.log".format(job_name)),
        "script_location": SCRIPT_LOCATION,
        "env_activation": ENV_ACTIVATION,
        "slurm_account": SLURM_ACCOUNT,
        "number_of_gpus": str(args.number_of_gpus),
        "number_of_cpus": str(args.number_of_gpus * 10),
        "memory": compute_memory(args),
    }

    job_script_name = "job_{:s}.sh".format(job_name)
    job_script_path = path.join(JOB_SCRIPT_SAVE_LOCATION, job_script_name)

    with open(job_script_path, "w") as jobscript:
        jobscript.write(F_STRING.format(**format_args))

    call(["sbatch", job_script_path])
