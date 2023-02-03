import argparse
import gc
import os.path
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import torch
import torch.distributed as dist
from dataset import dataset_map
from model import DpFslLinear
from opacus import PrivacyEngine
from opacus.distributed import \
    DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tf_dataset_reader import TfDatasetReader
from torch.utils.data import DataLoader, TensorDataset
from utils import (CsvWriter, Logger, compute_accuracy_from_predictions,
                   cross_entropy_loss, get_git_revision_short_hash,
                   get_mean_percent, get_slurm_job_id,
                   limit_tensorflow_memory_usage, predict_by_max_logit,
                   set_seeds, shuffle)

# custom alphas: alpha has to be > 1 and we add a bit more alphas than the default (max. 255 instead of 63)
CUSTOM_ALPHAS = [1.01, 1.05] + [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self, distributed_rank=None):
        self.distributed_rank = distributed_rank
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, 'log.txt', self.distributed_rank)
        self.start_time = datetime.now()
        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        if self.distributed_rank is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.distributed_rank)
        self.loss = cross_entropy_loss
        self.csv_writer = CsvWriter(
            file_path=os.path.join(self.args.checkpoint_dir, 'results.csv'),
            header=['dataset', 'accuracy']
        )
        self.print_parameter_count = True
        self.eps = None
        self.delta = None
        self.train_images = None
        self.train_labels = None
        self.num_classes = None
        self.git_revision_hash = get_git_revision_short_hash()

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', help='Dataset to use.', choices=dataset_map.keys(), default="small_set")
        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.003, help="Learning rate.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
                            help="Directory to save checkpoint to.")
        parser.add_argument("--epochs", "-e", type=int, default=400, help="Number of fine-tune epochs.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=200, help="Batch size.")
        parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--save_model", dest="save_model", default=False,
                            action="store_true", help="If true, save the fine tuned model.")

        # differential privacy options
        parser.add_argument("--private", dest="private", default=False, action="store_true",
                            help="If true, use differential privacy.")
        parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Noise multiplier.")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--target_epsilon", type=float, default=10.0, help="Maximum value of epsilon allowed.")
        parser.add_argument("--max_physical_batch_size", type=int, default=400, help="Maximum physical batch size")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")

        # tuning params
        parser.add_argument("--tune_params", dest="tune_params", default=False, action="store_true",
                            help="If true, tune hyper-parameters.")
        parser.add_argument("--epochs_lb", type=int, default=20, help="LB of fine-tune epochs.")
        parser.add_argument("--epochs_ub", type=int, default=200, help="UB of fine-tune epochs.")
        parser.add_argument("--train_batch_size_lb", type=int, default=10, help="LB of Batch size.")
        parser.add_argument("--train_batch_size_ub", type=int, default=1000, help="UB of Batch size.")
        parser.add_argument("--max_grad_norm_lb", type=float, default=0.2, help="LB of maximum gradient norm.")
        parser.add_argument("--max_grad_norm_ub", type=float, default=10.0, help="UB of maximum gradient norm.")
        parser.add_argument("--learning_rate_lb", type=float, default=1e-7, help="LB of learning rate")
        parser.add_argument("--learning_rate_ub", type=float,  default=1e-2, help="UB of learning rate")
        parser.add_argument("--save_optuna_study", dest="save_optuna_study", default=True, action="store_true",
                            help="If true, save optuna studies.")
        parser.add_argument("--number_of_trials", type=int, default=20, help="The number of trials for optuna")
        parser.add_argument("--optuna_starting_checkpoint", default=None,
                            help="Path of a optuna checkpoint from which to start the study again. (Updates the number of trials)")
        args = parser.parse_args()
        return args

    def is_master_process(self):
        # only execute on master process (either when non distributed or on the master process when distributed)
        return self.distributed_rank is None or self.distributed_rank == 0

    def run(self):
        if self.distributed_rank is not None and not self.args.private:
            raise NotImplementedError("Non-private distributed training has not been implemented.")

        # seeding
        set_seeds(self.args.seed)

        limit_tensorflow_memory_usage(2048)

        self.logger.print_and_log("")  # add a blank line

        datasets = dataset_map[self.args.dataset]
        summary_dict = self._get_summary_dict()

        for dataset in datasets:
            if dataset['enabled'] is False:
                continue

            if self.args.examples_per_class == -1:
                context_set_size = -1  # this is the use the entire training set case
            elif (self.args.examples_per_class is not None) and (dataset['name'] != 'oxford_iiit_pet'):  # bug in pets
                context_set_size = self.args.examples_per_class * dataset['num_classes']  # few-shot case
            else:
                context_set_size = 1000  # VTAB1000

            self.num_classes = dataset['num_classes']

            self.dataset_reader = TfDatasetReader(
                dataset=dataset['name'],
                task=dataset['task'],
                context_batch_size=context_set_size,
                target_batch_size=self.args.test_batch_size,
                path_to_datasets=self.args.download_path_for_tensorflow_datasets,
                num_classes=dataset['num_classes'],
                image_size=224 if 'vit' in self.args.feature_extractor else dataset['image_size'],
                examples_per_class=self.args.examples_per_class if self.args.examples_per_class != -1 else None,
                examples_per_class_seed=self.args.seed,
                tfds_seed=self.args.seed,
                device=self.device,
                osr=False
            )

            # create the training dataset
            self.train_images, self.train_labels = self.dataset_reader.get_context_batch()

            # tune hyper-parameters
            if self.args.tune_params:
                # we are loading the study here and update the number of trials to avoid issues in the distributed case
                if self.args.optuna_starting_checkpoint is not None:
                    study, self.args.number_of_trials = self.resume_from_optuna_checkpoint()
                else:
                    study = None

                if self.is_master_process():
                    if study is None:  # if not resuming study
                        sampler = optuna.samplers.TPESampler(seed=self.args.seed)
                        study = optuna.create_study(study_name="dp_fsl", direction="maximize", sampler=sampler)
                    for _ in range(self.args.number_of_trials):
                        study.optimize(self.objective, n_trials=1)
                        if self.args.save_optuna_study:
                            self.checkpoint_optuna_study(study)
                else:
                    # following: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
                    for _ in range(self.args.number_of_trials):
                        try:
                            self.objective(None)
                        except optuna.TrialPruned:
                            pass

                if self.is_master_process():
                    print("Best trial:")
                    trial = study.best_trial

                    print("Value: ", trial.value)

                    print("Params: ")
                    for key, value in trial.params.items():
                        print("{}: {}".format(key, value))

                    if self.args.private:
                        self.args.max_grad_norm = trial.params['max_grad_norm']
                    self.args.train_batch_size = trial.params['batch_size']
                    self.args.learning_rate = trial.params['learning_rate']
                    self.args.epochs = trial.params['epochs']

            # add barrier to prevent timeout in distributed case
            if self.distributed_rank is not None:
                dist.barrier()

            # final train and test on full test set
            accuracy, _, privacy_engine = self.train_test(
                train_images=self.train_images,
                train_labels=self.train_labels,
                num_classes=self.num_classes,
                test_set_reader=self.dataset_reader,
                validate=False
            )

            if self.is_master_process():

                if summary_dict is not None:
                    summary_dict['all'].append(accuracy)
                    summary_dict[dataset['category']].append(accuracy)

                self.print_metrics(dataset, accuracy, validation=False)

                self.csv_writer.write_row(
                    [
                        "{0:}".format(dataset['name']),
                        "{0:3.1f}".format(accuracy * 100.00)
                    ]
                )

        if self.is_master_process():
            if summary_dict is not None:
                for key in summary_dict:
                    acc = get_mean_percent(summary_dict[key])
                    self.logger.print_and_log("{0}: {1:3.1f}".format(key, acc))
                    self.csv_writer.write_row(
                        [
                            "{0:}".format(key),
                            "{0:3.1f}".format(acc)
                        ]
                    )
            self.logger.print_and_log("Time Taken = {}".format(datetime.now() - self.start_time))
            result_file = os.path.join(self.args.checkpoint_dir, "few_shot_results.csv")

            self._save_final_run_to_csv(result_file, accuracy, privacy_engine)

        # add barrier to prevent timeout in distributed case
        if self.distributed_rank is not None:
            dist.barrier()

    def objective(self, single_trial):
        if self.distributed_rank is not None:
            # following: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
            trial = optuna.integration.TorchDistributedTrial(single_trial)
        else:
            trial = single_trial

        if self.args.private:
            self.args.max_grad_norm = trial.suggest_float(
                'max_grad_norm',
                self.args.max_grad_norm_lb,
                self.args.max_grad_norm_ub
            )

        self.args.train_batch_size = trial.suggest_int(
            'batch_size',
            self.args.train_batch_size_lb,
            self.args.train_batch_size_ub
        )

        self.args.learning_rate = trial.suggest_float(
            'learning_rate',
            self.args.learning_rate_lb,
            self.args.learning_rate_ub
        )
        self.args.epochs = trial.suggest_int(
            'epochs',
            self.args.epochs_lb,
            self.args.epochs_ub
        )

        accuracy, eps, _ = self.train_test(
            train_images=self.train_images,
            train_labels=self.train_labels,
            num_classes=self.num_classes,
            test_set_reader=None,
            validate=True
        )

        return accuracy

    def train_test(self, train_images, train_labels, num_classes, test_set_reader=None, validate=False):

        # seed when not doing hyperparameter tuning
        if not validate:
            set_seeds(self.args.seed)

        batch_size = self.args.train_batch_size

        if validate:  # tune hyper-parameters
            if self.args.examples_per_class is not None:
                train_images, train_labels = shuffle(train_images, train_labels)

            train_partition_size = int(0.7 * len(train_labels))
            train_loader = DataLoader(
                TensorDataset(train_images[:train_partition_size], train_labels[:train_partition_size]),
                batch_size=min(batch_size, train_partition_size),
                shuffle=True
            )

            val_loader = DataLoader(
                TensorDataset(train_images[train_partition_size:], train_labels[train_partition_size:]),
                batch_size=self.args.test_batch_size,
                shuffle=False
            )
        else:  # testing
            train_loader_generator = torch.Generator()
            train_loader_generator.manual_seed(self.args.seed)
            self.start_time_final_run = datetime.now()
            train_loader = DataLoader(
                TensorDataset(train_images, train_labels),
                batch_size=batch_size if self.args.private else min(self.args.train_batch_size,
                                                                    self.args.max_physical_batch_size),
                shuffle=True,
                generator=train_loader_generator
            )

        if self.args.classifier == 'linear':
            model = DpFslLinear(
                feature_extractor_name=self.args.feature_extractor,
                num_classes=num_classes,
                learnable_params=self.args.learnable_params
            )
        else:
            print("Invalid classifier option.")
            sys.exit()

        # print parameters, but only once
        if self.print_parameter_count:
            self.get_parameter_count(model)
            self.print_parameter_count = False

        if self.distributed_rank is not None:
            model = DPDDP(model)

        model = model.to(self.device)

        if self.args.classifier == 'linear':
            self.eps, self.delta, privacy_engine = self.fine_tune_batch(model=model, train_loader=train_loader)
            if validate:
                accuracy = (self.validate_linear(model=model, val_loader=val_loader)).cpu()
            else:
                accuracy = (self.test_linear(model=model, dataset_reader=test_set_reader)).cpu()
        else:
            print("Invalid classifier option.")
            sys.exit()

        # free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy, self.eps, privacy_engine

    def fine_tune_batch(self, model, train_loader):
        model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        delta = None
        if self.args.private:
            delta = 1.0 / (len(train_loader.dataset))
            privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=self.args.secure_rng)

            seeded_noise_generator = torch.Generator(device=self.device)
            seeded_noise_generator.manual_seed(self.args.seed)

            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=self.args.target_epsilon,
                epochs=self.args.epochs,
                target_delta=delta,
                max_grad_norm=self.args.max_grad_norm,
                noise_generator=seeded_noise_generator if not self.args.secure_rng else None,
                alphas=CUSTOM_ALPHAS)

        if self.args.private:
            for epoch in range(self.args.epochs):
                with BatchMemoryManager(
                        data_loader=train_loader,
                        max_physical_batch_size=self.args.max_physical_batch_size,
                        optimizer=optimizer
                ) as new_train_loader:
                    for batch_images, batch_labels in new_train_loader:
                        batch_images = batch_images.to(self.device)
                        batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                        optimizer.zero_grad()
                        torch.set_grad_enabled(True)
                        logits = model(batch_images)
                        loss = self.loss(logits, batch_labels)
                        loss.backward()
                        del logits
                        optimizer.step()

        else:
            for epoch in range(self.args.epochs):
                for batch_images, batch_labels in train_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                    optimizer.zero_grad()
                    torch.set_grad_enabled(True)
                    batch_size = len(batch_labels)
                    num_sub_batches = self._get_number_of_sub_batches(batch_size)
                    for batch in range(num_sub_batches):
                        batch_start_index, batch_end_index = self._get_sub_batch_indices(batch, batch_size)
                        logits = model(batch_images[batch_start_index: batch_end_index])
                        loss = self.loss(logits, batch_labels[batch_start_index: batch_end_index])
                        loss.backward()
                        del logits
                    optimizer.step()

        eps = None
        if self.args.private:
            # we need to call the accountant directly because we want to pass custom alphas
            eps = privacy_engine.accountant.get_epsilon(delta=delta, alphas=CUSTOM_ALPHAS)
            return eps, delta, privacy_engine
        else:
            return eps, delta, None

    def test_linear(self, model, dataset_reader):
        model.eval()

        with torch.no_grad():
            labels = []
            predictions = []
            test_set_size = dataset_reader.get_target_dataset_length()
            num_batches = int(np.ceil(float(test_set_size) / float(self.args.test_batch_size)))
            for batch in range(num_batches):
                batch_images, batch_labels = dataset_reader.get_target_batch()
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def validate_linear(self, model, val_loader):
        model.eval()

        with torch.no_grad():
            labels = []
            predictions = []
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def get_parameter_count(self, model):
        model_param_count = sum(p.numel() for p in model.parameters())
        self.logger.print_and_log("Model Parameter Count = {}".format(model_param_count))
        trainable_model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.print_and_log("Model Trainable Parameter Count = {}".format(trainable_model_param_count))

        feature_extractor_param_count = sum(p.numel() for p in model.feature_extractor.parameters())
        self.logger.print_and_log("Feature Extractor Parameter Count = {}".format(feature_extractor_param_count))
        trainable_feature_extractor_param_count = sum(p.numel() for p in model.feature_extractor.parameters() if p.requires_grad)
        self.logger.print_and_log("Feature Extractor Trainable Parameter Count = {}".format(trainable_feature_extractor_param_count))

        if self.args.classifier == 'linear':
            head_param_count = sum(p.numel() for p in model.head.parameters())
        else:
            head_param_count = 0
        self.logger.print_and_log("Head Parameter Count = {}".format(head_param_count))
        if self.args.classifier == 'linear':
            trainable_head_param_count = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        else:
            trainable_head_param_count = 0
        self.logger.print_and_log("Head Trainable Parameter Count = {}".format(trainable_head_param_count))

    def print_metrics(self, dataset, accuracy, validation):
        if validation:
            self.logger.print_and_log("Validation")
        else:
            self.logger.print_and_log("Test")
        if self.args.private:
            if dataset['task'] is None:
                self.logger.print_and_log(
                    '{0:}: accuracy = {1:3.1f}, (eps, delta) = ({2:3.3f}, {3:.2E})'.format(
                        dataset['name'], accuracy * 100.0, self.eps, self.delta
                    )
                )
            else:
                self.logger.print_and_log(
                    '{0:} {1:}: accuracy = {2:3.1f}, (eps, delta) = ({3:3.3f}, {4:.2E})'.format(
                        dataset['name'], dataset['task'], accuracy * 100.0, self.eps, self.delta
                    )
                )
        else:
            if dataset['task'] is None:
                self.logger.print_and_log('{0:}: {1:3.1f}'.format(dataset['name'], accuracy * 100.0))
            else:
                self.logger.print_and_log('{0:} {1:}: {2:3.1f}'.format(dataset['name'], dataset['task'],
                                                                       accuracy * 100.0))

    def extract_accounting_history(self, privacy_engine):
        total_steps = 0
        sample_rate = None
        noise_multiplier = None
        # extract steps and sample_rate from opacus privacy_engine
        for (entry_noise_multiplier, entry_sample_rate, entry_num_steps) in privacy_engine.accountant.history:
            if sample_rate is None:
                sample_rate = entry_sample_rate
            elif sample_rate != entry_sample_rate or total_steps > 0:
                raise ValueError("Sample rates do not match or sample rate is None.")
            total_steps += entry_num_steps
            if noise_multiplier is None:
                noise_multiplier = entry_noise_multiplier
            elif noise_multiplier != entry_noise_multiplier or total_steps > 0:
                raise ValueError("noise_multiplier do not match or noise_multiplier is None.")
        return total_steps, sample_rate, noise_multiplier

    def checkpoint_optuna_study(self, study):
        pkl_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_study.pkl".format(self.args.dataset, self.args.optimizer, self.args.target_epsilon, self.args.examples_per_class,
                                                                 self.args.seed, self.args.classifier, self.args.feature_extractor, self.args.learnable_params, get_slurm_job_id())
        directory = os.path.join(self.args.checkpoint_dir, "optuna")
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(study, os.path.join(directory, pkl_name))

    def resume_from_optuna_checkpoint(self):
        self.logger.print_and_log("\nResuming from loaded study at {}".format(self.args.optuna_starting_checkpoint))
        existing_study = joblib.load(self.args.optuna_starting_checkpoint)

        number_of_trails = len(existing_study.get_trials())
        self.logger.print_and_log("Found {} trials in study".format(number_of_trails))

        remaining_trials = self.args.number_of_trials - number_of_trails
        self.logger.print_and_log("Updating remaining trails from {} to {}\n".format(self.args.number_of_trials, remaining_trials))

        return existing_study, remaining_trials

    def _get_number_of_sub_batches(self, task_size):
        num_batches = int(np.ceil(float(task_size) / float(self.args.max_physical_batch_size)))
        if num_batches > 1 and (task_size % self.args.max_physical_batch_size == 1):
            num_batches -= 1

        return num_batches

    def _get_sub_batch_indices(self, index, total_batch_size):
        batch_start_index = index * self.args.max_physical_batch_size
        batch_end_index = batch_start_index + self.args.max_physical_batch_size
        if batch_end_index == (total_batch_size - 1):  # avoid batch size of 1
            batch_end_index = total_batch_size
        if batch_end_index > total_batch_size:
            batch_end_index = total_batch_size
        return batch_start_index, batch_end_index

    def _save_final_run_to_csv(self, result_file, accuracy, privacy_engine):
        if self.args.private:
            total_steps, sample_rate, noise_multiplier = self.extract_accounting_history(privacy_engine)
        else:
            total_steps, sample_rate, noise_multiplier = None, None, None

        with open(result_file, 'a') as file:
            for c in [self.args.classifier, self.args.feature_extractor, self.args.learnable_params, self.args.examples_per_class]:
                file.write(str(c)+",")
            for c in [self.args.dataset, self.eps, self.delta, accuracy.item(), self.args.seed, total_steps, sample_rate, noise_multiplier]:
                file.write(str(c)+",")
            for c in [self.args.train_batch_size, self.args.learning_rate, self.args.epochs, self.args.max_grad_norm, datetime.now() - self.start_time_final_run, self.args.optimizer]:
                file.write(str(c)+",")
            # add slurm job_id and time
            file.write(get_slurm_job_id()+","+datetime.now().strftime("%Y-%m-%d %H:%M")+",")
            # add git revision hash
            file.write(self.git_revision_hash+",")
            # add hyperparameter tuning information
            for c in [self.args.tune_params, self.args.epochs_lb, self.args.epochs_ub, self.args.train_batch_size_lb, self.args.train_batch_size_ub]:
                file.write(str(c)+",")
            for c in [self.args.max_grad_norm_lb, self.args.max_grad_norm_ub, self.args.learning_rate_lb, self.args.learning_rate_ub, self.args.number_of_trials]:
                file.write(str(c)+",")
            # add cluster information
            for c in [self.args.max_physical_batch_size, torch.cuda.device_count()]:
                file.write(str(c)+",")
            file.write('\n')

    def _get_summary_dict(self):
        if self.args.dataset == "vtab_1000":
            summary_dict = {
                'all': [],
                'natural': [],
                'specialized': [],
                'structured': [],
            }
        elif self.args.dataset == "vtab_natural":
            summary_dict = {
                'all': [],
                'natural': [],
            }
        elif self.args.dataset == "vtab_structured":
            summary_dict = {
                'all': [],
                'structured': [],
            }
        elif self.args.dataset == "vtab_specialized":
            summary_dict = {
                'all': [],
                'specialized': [],
            }
        else:
            summary_dict = None

        return summary_dict


if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
