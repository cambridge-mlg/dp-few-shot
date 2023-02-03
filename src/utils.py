import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import csv
import numpy as np
import random
import subprocess

class Logger():
    def __init__(self, checkpoint_dir, log_file_name, distributed_rank = None):
        # only print and log on first worker
        self.distributed_rank = distributed_rank
        if distributed_rank is None or distributed_rank == 0:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            log_file_path = os.path.join(checkpoint_dir, log_file_name)
            self.file = None
            if os.path.isfile(log_file_path):
                self.file = open(log_file_path, "a", buffering=1)
            else:
                self.file = open(log_file_path, "w", buffering=1)

    def __del__(self):
        if self.distributed_rank is None or self.distributed_rank == 0:
            self.file.close()

    def log(self, message):
        if self.distributed_rank is None or self.distributed_rank == 0:
            self.file.write(message + '\n')

    def print_and_log(self, message):
        if self.distributed_rank is None or self.distributed_rank == 0:
            print(message, flush=True)
            self.log(message)


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, torch.argmax(logits, dim=-1)).float())


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())


def limit_tensorflow_memory_usage(gpu_memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
                )
        except RuntimeError as e:
            print(e)


class CsvWriter:
    def __init__(self, file_path, header):
        self.file = open(file_path, 'w', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

    def __del__(self):
        self.file.close()

    def write_row(self, row):
        self.writer.writerow(row)


def get_mean_percent_and_95_confidence_interval(list_of_quantities):
    if len(list_of_quantities) > 1:
        mean_percent = np.array(list_of_quantities).mean() * 100.0
        confidence_interval = (196.0 * np.array(list_of_quantities).std()) / np.sqrt(len(list_of_quantities))
        return mean_percent, confidence_interval
    else:
        return list_of_quantities[0] * 100.0, 0.0


def get_mean_percent(list_of_quantities):
    mean_percent = np.array(list_of_quantities).mean() * 100.0
    return mean_percent


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

import sys


class LogFiles:
    def __init__(self, checkpoint_dir, resume, test_mode):
        self._checkpoint_dir = checkpoint_dir
        if not self._verify_checkpoint_dir(resume, test_mode):
            sys.exit()
        if not test_mode and not resume:
            os.makedirs(self.checkpoint_dir)
        self._best_validation_model_path = os.path.join(checkpoint_dir, 'best_validation.pt')
        self._fully_trained_model_path = os.path.join(checkpoint_dir, 'fully_trained.pt')

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def best_validation_model_path(self):
        return self._best_validation_model_path

    @property
    def fully_trained_model_path(self):
        return self._fully_trained_model_path

    def _verify_checkpoint_dir(self, resume, test_mode):
        checkpoint_dir_is_ok = True
        if resume:  # verify that the checkpoint directory and file exists
            if not os.path.exists(self.checkpoint_dir):
                print("Can't resume from checkpoint. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            if not os.path.isfile(checkpoint_file):
                print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
                checkpoint_dir_is_ok = False

        elif test_mode:
            if not os.path.exists(self.checkpoint_dir):
                print("Can't test. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

        else:
            if os.path.exists(self.checkpoint_dir):
                print("Checkpoint directory ({}) already exits.".format(self.checkpoint_dir), flush=True)
                print("If starting a new training run, specify a directory that does not already exist.", flush=True)
                print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
                checkpoint_dir_is_ok = False

        return checkpoint_dir_is_ok


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def get_batch_indices(index, last_element, batch_size):
    batch_start_index = index * batch_size
    batch_end_index = batch_start_index + batch_size
    if batch_end_index > last_element:
        batch_end_index = last_element
    return batch_start_index, batch_end_index


def compute_features_by_batch(images, feature_extractor, batch_size):
    features = []
    num_images = images.size(0)
    num_batches = int(np.ceil(float(num_images) / float(batch_size)))
    for batch in range(num_batches):
        batch_start_index, batch_end_index = get_batch_indices(batch, num_images, batch_size)
        features.append(feature_extractor(images[batch_start_index: batch_end_index]))
    return torch.vstack(features)


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]

def get_git_revision_short_hash() -> str:
    """
    Returns current git hash
    Returns
    -------
    str
        short git hash
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_slurm_job_id() -> str:
    """
    Returns the slurm job id.
    """
    return os.environ.get("SLURM_JOB_ID", "")
