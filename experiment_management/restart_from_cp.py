import argparse
import fnmatch
import os
import re
from pathlib import Path
from subprocess import call

from deploy_few_shot_universial_script import create_job_name, parse_job_time


def _get_cp_path(directory, args):
    latest_file = None
    for file in os.listdir(directory):
        pkl_name = "{}_{}_{}_{}_{}_{}_{}_{}_*_study.pkl".format(args.dataset, args.optimizer, args.target_epsilon, args.examples_per_class,
                                                                args.seed, args.classifier, args.feature_extractor, args.learnable_params)
        if fnmatch.fnmatch(file, pkl_name):
            latest_file = file

    if latest_file is None:
        raise ValueError("File not found.")
    return os.path.join(directory, latest_file)


def _parse_command_line():
    parser = argparse.ArgumentParser()

    # experiment params
    parser.add_argument("--feature_extractor", default="BiT-M-R50x1", choices=['vit-b-16', 'BiT-M-R50x1'], help="Feature extractor to use.")
    parser.add_argument("--classifier", choices=['linear'], default='linear', help="Which classifier to use.")
    parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                        help="Which feature extractor parameters to learn.")
    parser.add_argument("--examples_per_class", type=int, default=None,
                        help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
    parser.add_argument("--target_epsilon", type=float, default=8.0, help="Maximum value of epsilon allowed.")
    parser.add_argument('--dataset', help='dataset for testing', choices=["cifar10", "cifar100", "svhn_cropped"])
    parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')

    # memory options
    parser.add_argument("--max_physical_batch_size", type=int, default=40, help="Maximum physical batch size")

    # seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for determinstic results.")

    parser.add_argument("--job_maximum_hours", type=float, default=0.1, help="The maximum hours the job is running.")
    args = parser.parse_args()
    return args


def _get_old_jobfile(directory, args):
    previous_script_name = "job_{:s}.sh".format(create_job_name(args))
    job_file_path = os.path.join(directory, previous_script_name)
    if not os.path.exists(job_file_path):
        raise ValueError("File {} does not exist.".format(job_file_path))
    else:
        with open(job_file_path, "r") as file:
            data = file.read()
        return data


def _replace_optuna_args(old_job_file, cp_path):
    if "--save_optuna_study" not in old_job_file:
        raise ValueError("--save_optuna_study missing from job file")
    new_job_file = old_job_file.replace("--save_optuna_study", "--optuna_starting_checkpoint {}".format(cp_path))
    return new_job_file


def _replace_time(old_job_file, args):
    new_job_file = re.sub(r"--time=.*", "--time {}".format(parse_job_time(args)), old_job_file)
    return new_job_file


def _replace_distributed_run(old_job_file):
    new_job_file = old_job_file.replace("python run_distributed.py", "python run.py")
    new_job_file = new_job_file.replace("--gres=gpu:v100:2", "--gres=gpu:v100:1")
    new_job_file = new_job_file.replace("--gres=gpu:v100:3", "--gres=gpu:v100:1")
    new_job_file = new_job_file.replace("--gres=gpu:v100:4", "--gres=gpu:v100:1")
    return new_job_file


if __name__ == "__main__":
    DIRECTORY = Path('hyperparam_optimization/experiment_scripts/restart_config.txt').read_text()
    args = _parse_command_line()
    cp_path = _get_cp_path(DIRECTORY+"optuna", args)
    if not os.path.exists(cp_path):
        raise ValueError("File {} does not exist.".format(cp_path))
    else:
        old_job_file = _get_old_jobfile(DIRECTORY+"submitted_jobs", args)
        new_job_file = _replace_optuna_args(old_job_file, cp_path)
        new_job_file = _replace_time(new_job_file, args)
        new_job_file = _replace_distributed_run(new_job_file)
        new_script_name = create_job_name(args).replace(".sh", "_rerun.sh")
        new_job_path = os.path.join(DIRECTORY+"submitted_jobs", new_script_name)
        with open(new_job_path, "w") as jobscript:
            jobscript.write(new_job_file)

        call(["sbatch", new_job_path])
