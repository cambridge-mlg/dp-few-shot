# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py

import numpy as np
import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from dataset import dataset_map
from utils import Logger, limit_tensorflow_memory_usage,\
    compute_accuracy_from_predictions, predict_by_max_logit, cross_entropy_loss, shuffle, set_seeds
from tf_dataset_reader import TfDatasetReader
from datetime import datetime
from model import DpFslLinear
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import gc
import sys
import warnings
from lira import convert_logit_to_prob, calculate_statistic, log_loss
import pickle


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, 'log.txt')
        self.start_time = datetime.now()
        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss = cross_entropy_loss
        self.print_parameter_count = True
        self.eps = None
        self.delta = None
        self.tune_images = None
        self.tune_labels = None
        self.num_classes = None

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

        # LiRA options
        parser.add_argument("--num_shadow_models", type=int, default=256,
                            help="Number of shadow models to train tfor the LiRA attack.")

        args = parser.parse_args()
        return args

    def init_model(self, num_classes):
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

        model = model.to(self.device)

        return model

    def run(self):
        # seeding
        set_seeds(self.args.seed)

        limit_tensorflow_memory_usage(2048)

        self.logger.print_and_log("")  # add a blank line

        datasets = dataset_map[self.args.dataset]

        for dataset in datasets:
            if dataset['enabled'] is False:
                continue

            self.num_classes = dataset['num_classes']

            self.dataset_reader = TfDatasetReader(
                dataset=dataset['name'],
                task=dataset['task'],
                context_batch_size=1000,
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
            train_images, train_labels = self.dataset_reader.get_context_batch()

            self.logger.print_and_log("{}".format(dataset['name']))

            self.run_lira(
                x=train_images,
                y=train_labels,
                dataset_reader=self.dataset_reader
            )

    def train_test(
            self,
            train_images,
            train_labels,
            num_classes,
            test_set_reader=None,
            save_model_name=None):

        batch_size = self.args.train_batch_size

        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(self.args.seed)
        self.start_time_final_run = datetime.now()
        train_loader = DataLoader(
            TensorDataset(train_images, train_labels),
            batch_size=batch_size if self.args.private else min(self.args.train_batch_size, self.args.max_physical_batch_size),
            shuffle=True,
            generator=train_loader_generator
        )

        model = self.init_model(num_classes=num_classes)

        if self.args.classifier == 'linear':
            self.eps, self.delta = self.fine_tune_batch(model=model, train_loader=train_loader)
            if test_set_reader is not None:  # use test set for testing
                accuracy = (self.test_linear(model=model, dataset_reader=test_set_reader)).cpu()
            else:
                accuracy = 0.0  # don't test
        else:
            print("Invalid classifier option.")
            sys.exit()

        if save_model_name is not None:
            self.save_model(model=model, file_name=save_model_name)

        # free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy, self.eps

    def fine_tune_batch(self, model, train_loader):
        model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        delta = None
        if self.args.private:
            delta = 1.0 / (len(train_loader.dataset))
            privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=self.args.secure_rng)

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
                noise_generator=seeded_noise_generator if not self.args.secure_rng else None)

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
                    logits = model(batch_images)
                    loss = self.loss(logits, batch_labels)
                    loss.backward()
                    del logits
                    optimizer.step()

        eps = None
        if self.args.private:
            eps = privacy_engine.get_epsilon(delta=delta)

        return eps, delta

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

    def run_lira(self, x, y, dataset_reader):
        # Sample weights are set to `None` by default, but can be changed here.
        sample_weight = None
        n = x.shape[0]
        delta = 1.0 / float(n / 2)

        # Train the target and shadow models. We will use one of the model in `models`
        # as target and the rest as shadow.
        # Here we use the same architecture and optimizer. In practice, they might
        # differ between the target and shadow models.
        in_indices = []  # a list of in-training indices for all models
        stat = []  # a list of statistics for all models
        losses = []  # a list of losses for all models
        for i in range(self.args.num_shadow_models + 1):
            model_name = f'model{i}.pt'

            # Generate a binary array indicating which example to include for training
            in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))

            if not os.path.exists(self._get_model_path(model_name)):  # train the model
                model_train_images = x[in_indices[-1]]
                model_train_labels = y[in_indices[-1]]
                model_train_images = model_train_images.to(self.device)
                model_train_labels = model_train_labels.to(self.device)

                accuracy, eps = self.train_test(
                    train_images=model_train_images,
                    train_labels=model_train_labels,
                    num_classes=self.num_classes,
                    test_set_reader=dataset_reader if i == 0 else None,
                    save_model_name=model_name  # save the model, so we can load it and get challenge example losses
                )

                self.logger.print_and_log(
                    f'Trained model #{i} with {in_indices[-1].sum()} examples. Accuracy = {accuracy}. Epsilon = {eps}'
                )

            # Get the statistics of the current model.
            idx = self.init_model(num_classes=self.num_classes)
            self.load_model(model=idx, file_name=model_name)
            s, _ = self.get_stat_and_loss_aug(idx, x, y.numpy(), sample_weight)
            stat.append(s)

            # Avoid OOM
            gc.collect()

        # save stat and in_indices
        with open(os.path.join(self.args.checkpoint_dir, 'stat_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(stat, f)
        with open(os.path.join(self.args.checkpoint_dir, 'in_indices_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(in_indices, f)

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

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), os.path.join(self.args.checkpoint_dir, file_name))

    def load_model(self, model, file_name):
        model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, file_name)))

    def _get_model_path(self, file_name):
        return os.path.join(self.args.checkpoint_dir, file_name)

    def get_stat_and_loss_aug(self,
                              model,
                              x,
                              y,
                              sample_weight=None,
                              batch_size=4096):
        """A helper function to get the statistics and losses.

        Here we get the statistics and losses for the images.

        Args:
            model: model to make prediction
            x: samples
            y: true labels of samples (integer valued)
            sample_weight: a vector of weights of shape (n_samples, ) that are
                assigned to individual samples. If not provided, then each sample is
                given unit weight. Only the LogisticRegressionAttacker and the
                RandomForestAttacker support sample weights.
            batch_size: the batch size for model.predict

        Returns:
            the statistics and cross-entropy losses
        """
        losses, stat = [], []
        data = x.to(self.device)
        data_size = len(data)
        num_sub_batches = self._get_number_of_sub_batches(data_size, self.args.test_batch_size)
        for batch in range(num_sub_batches):
            batch_start_index, batch_end_index = self._get_sub_batch_indices(batch, data_size, self.args.test_batch_size)
            with torch.no_grad():
                logits = model(data[batch_start_index: batch_end_index]).cpu().numpy()
            prob = convert_logit_to_prob(logits)
            losses.append(log_loss(y[batch_start_index: batch_end_index], prob, sample_weight=sample_weight))
            stat.append(calculate_statistic(prob, y[batch_start_index: batch_end_index], sample_weight=sample_weight, is_logits=False))
        return np.expand_dims(np.concatenate(stat), axis=1), np.expand_dims(np.concatenate(losses), axis=1)

    def _get_number_of_sub_batches(self, task_size, sub_batch_size):
        num_batches = int(np.ceil(float(task_size) / float(sub_batch_size)))
        if num_batches > 1 and (task_size % sub_batch_size == 1):
            num_batches -= 1
        return num_batches

    def _get_sub_batch_indices(self, index, task_size, sub_batch_size):
        batch_start_index = index * sub_batch_size
        batch_end_index = batch_start_index + sub_batch_size
        if batch_end_index == (task_size - 1):  # avoid batch size of 1
            batch_end_index = task_size
        if batch_end_index > task_size:
            batch_end_index = task_size
        return batch_start_index, batch_end_index


if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
