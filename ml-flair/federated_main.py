# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import atexit
import functools
import os
import tensorflow as tf
import tensorflow_federated as tff
import time
import asyncio
from absl import app
from absl import flags
from absl import logging
from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib
from typing import Any, Callable, Optional, Dict

import flair_data, flair_metrics, flair_model

# Defining optimizer flags
flags.DEFINE_float('client_learning_rate', 0.1, 'Client local learning rate')
flags.DEFINE_float(
    'client_clipnorm', 10.0,
    'Max L2 norm for gradient of each weight. '
    'This is used to prevent gradient explosion in client local training')
flags.DEFINE_float('server_learning_rate', 0.1, 'Server learning_rate')

# Federated training hyperparameters
flags.DEFINE_integer('client_epochs_per_round', 2,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('client_batch_size', 16, 'Batch size on the clients.')
flags.DEFINE_integer('clients_per_round', 200,
                     'How many clients to sample per round.')
flags.DEFINE_integer('clients_per_thread', 50,
                     'How many clients to sample per thread.')
flags.DEFINE_integer('client_datasets_random_seed', None,
                     'Random seed for client sampling.')
# Training loop configuration
flags.DEFINE_integer('total_rounds', 5000, 'Number of total training rounds.')
flags.DEFINE_integer(
    'rounds_per_eval', 10,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('max_elements_per_client', 512,
                     'Max number of training examples to use per client.')
flags.DEFINE_integer('eval_batch_size', 512,
                     'Batch size when evaluating on central datasets.')
flags.DEFINE_boolean('use_latest_checkpoint', False, 
                     'whether to proceed from latest checkpoint.')
flags.DEFINE_float('best_val_loss', None, 'include best val loss between runs.')
# Model configuration
flags.DEFINE_string('restore_model_path', None, 'Path to pretrained model.')
flags.DEFINE_string(
    'save_model_dir', './', 'Path to directory for saving model.')
flags.DEFINE_string('model_type', 'resnet18', 'Architecture to use.')
flags.DEFINE_string('adaptation', 'all', 'Adaptation used: all, film, head.')
# Data configuration
flags.DEFINE_string('dataset_name', 'flair', 'Dataset name')
flags.DEFINE_string('tfrecords_dir', None, 'Path to FLAIR tfrecords.')
flags.DEFINE_integer('image_height', 224, 'Height of input image.')
flags.DEFINE_integer('image_width', 224, 'Width of input image.')
flags.DEFINE_boolean('use_fine_grained_labels', False,
                     'use_fine_grained_labels.')
flags.DEFINE_string(
    'binary_label', None,
    'If set, train a binary classification model on the provided binary label.')
# Differential privacy configuration
flags.DEFINE_float('epsilon', 0.0, 'DP epsilon.')
flags.DEFINE_float('l2_norm_clip', 0.1, 'DP clipping bound.')
flags.DEFINE_float(
    'target_unclipped_quantile', 0.1,
    'Quantile for adaptive clipping bound. Value 0 turns off adaptive clipping')
flags.DEFINE_integer(
    'simulated_clients_per_round', None,
    'A simulated `clients_per_round` for experimenting DP more efficiently.'
    'If set larger than `clients_per_round`, the DP noise scale will be the '
    'same as if training with `simulated_clients_per_round` clients when only '
    '`clients_per_round` clients are actually sampled. See detailed description'
    ' in Section 5.1 of https://arxiv.org/abs/2207.08869')

FLAGS = flags.FLAGS


def get_dataset():

    if FLAGS.dataset_name == 'flair':
        image_shape = (256, 256, 3)
        label_to_index = flair_data.load_label_to_index(
            os.path.join(FLAGS.tfrecords_dir, "label_to_index.json"),
            FLAGS.use_fine_grained_labels)
        num_labels = len(label_to_index)

        binary_label_index = None
        if FLAGS.binary_label is not None:
            binary_label_index = label_to_index[FLAGS.binary_label]

        train_fed_data, val_fed_data, test_fed_data = flair_data.load_tfrecords_data(
            FLAGS.tfrecords_dir,
            image_shape=image_shape,
            num_labels=num_labels,
            use_fine_grained_labels=FLAGS.use_fine_grained_labels,
            binary_label_index=binary_label_index)

        if binary_label_index is not None:
            num_labels = 1
    elif FLAGS.dataset_name == 'cifar100':
        image_shape = (32, 32, 3)
        train_fed_data, test_fed_data = tff.simulation.datasets.cifar100.load_data(cache_dir=FLAGS.tfrecords_dir)
        _, val_fed_data = tff.simulation.datasets.cifar100.load_data(cache_dir=FLAGS.tfrecords_dir)
        num_labels = 100
    elif FLAGS.dataset_name == 'emnist':
        image_shape = (28, 28, 3)
        train_fed_data, test_fed_data = tff.simulation.datasets.emnist.load_data(only_digits=False,
                                                                                 cache_dir=FLAGS.tfrecords_dir)
        _, val_fed_data = tff.simulation.datasets.emnist.load_data(only_digits=False,
                                                                   cache_dir=FLAGS.tfrecords_dir)
        num_labels = 62
        
    return image_shape, num_labels, train_fed_data, val_fed_data, test_fed_data


def preprocess_dataset(train_fed_data, val_fed_data, test_fed_data):
    def preprocess_fn(data: tf.data.Dataset,
                      is_training: bool) -> tf.data.Dataset:
        """Preprocesses `tf.data.Dataset` by shuffling and batching."""
        if is_training:
            data = data.shuffle(FLAGS.max_elements_per_client, 
                                seed=FLAGS.client_datasets_random_seed)
            # Repeat data by client epochs and batch
            dataset = data.take(FLAGS.max_elements_per_client).repeat(
                FLAGS.client_epochs_per_round).batch(FLAGS.client_batch_size)
        else:
            dataset = data.batch(FLAGS.eval_batch_size)

        if FLAGS.dataset_name == 'cifar100':
            dataset = dataset.map(lambda x: (x['image'], tf.one_hot(x['label'], 100)))

        elif FLAGS.dataset_name == 'emnist':
            dataset = dataset.map(lambda x: (tf.tile(x['pixels'][..., None]*255, tf.constant([1, 1, 1, 3], tf.int32)),
                                             tf.one_hot(x['label'], 62)))
        return dataset

    train_fed_data = train_fed_data.preprocess(
        functools.partial(preprocess_fn, is_training=True))
    input_spec = train_fed_data.element_type_structure
    val_data = preprocess_fn(
        val_fed_data.create_tf_dataset_from_all_clients(), is_training=False)
    test_data = preprocess_fn(
        test_fed_data.create_tf_dataset_from_all_clients(), is_training=False)

    return input_spec, train_fed_data, val_data, test_data


def load_resnet50(model, adaptation, model_path, model_builder):
    if adaptation != 'film':
        model.load_weights(model_path, skip_mismatch=True, by_name=True)
    else:
        model_pt = model_builder()
        model_pt.load_weights(model_path, skip_mismatch=True, by_name=True)
        weights = model_pt.get_weights()
        variable_names = [v.name for v in model_pt.weights]

        weights_film = weights[:1]
        weights_names = variable_names[:1]

        last_gn_idx = 1
        num_gn = 0

        current_index = 1

        while not 'dense' in variable_names[current_index]:
            if 'group_norm' in variable_names[current_index]:
                if num_gn == 0:
                    last_gn_idx = current_index
                    num_gn += 1
                elif num_gn == 5:
                    weights_film += weights[current_index - 1: current_index + 1]
                    weights_film += weights[last_gn_idx: current_index - 1]

                    weights_names += variable_names[current_index - 1: current_index + 1]
                    weights_names += variable_names[last_gn_idx: current_index - 1]
                    num_gn = 0
                else:
                    num_gn += 1

            elif num_gn == 0:
                weights_film += [weights[current_index]]
                weights_names += [variable_names[current_index]]

            current_index += 1

        weights_names += variable_names[-4:]
        weights_film += weights[-4:]

        # model.save_weights('Flair-FiLM-BiT-M-R50x1.tf')
        model.set_weights(weights_film)


def main(argv):

    loop = asyncio.get_event_loop()

    if len(argv) > 1:
        raise app.UsageError('Expected no command-line arguments, '
                             'got: {}'.format(argv))

    gpu_devices = tf.config.list_logical_devices('GPU')
    if len(gpu_devices) > 0:
        tff.backends.native.set_local_python_execution_context(
            default_num_clients=FLAGS.clients_per_round,
            max_fanout=2 * FLAGS.clients_per_round,
            server_tf_device=tf.config.list_logical_devices('CPU')[0],
            client_tf_devices=gpu_devices,
            clients_per_thread=FLAGS.clients_per_thread)

    os.makedirs(FLAGS.save_model_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file('logs.txt', FLAGS.save_model_dir)
    f_val = open(os.path.join(FLAGS.save_model_dir, f"results.txt"), 'a')

    client_optimizer_fn = lambda: tf.keras.optimizers.SGD(
        FLAGS.client_learning_rate, clipnorm=FLAGS.client_clipnorm)
    server_optimizer_fn = lambda: tf.keras.optimizers.Adam(
        FLAGS.server_learning_rate, epsilon=0.01)

    image_shape, num_labels, train_fed_data, val_fed_data, test_fed_data = get_dataset()

    logging.info(
        "{} training users, {} validating users".format(
            len(train_fed_data.client_ids), len(val_fed_data.client_ids)))

    input_spec, train_fed_data, val_data, test_data = preprocess_dataset(train_fed_data,
                                                                         val_fed_data,
                                                                         test_fed_data)

    model_update_aggregation_factory = None
    if FLAGS.epsilon > 0.0:
        # Setup TFF with differential privacy
        n = len(train_fed_data.client_ids)
        if FLAGS.simulated_clients_per_round is not None:
            assert FLAGS.simulated_clients_per_round >= FLAGS.clients_per_round
            batch_size = FLAGS.simulated_clients_per_round
        else:
            batch_size = FLAGS.clients_per_round

        # Compute central DP noise scale added to aggregated model updates
        noise_multiplier = compute_noise_from_budget_lib.compute_noise(
            n=n,
            batch_size=batch_size,
            target_epsilon=FLAGS.epsilon,
            epochs=FLAGS.total_rounds,
            delta=1 / (n ** 1.1),
            noise_lbd=1e-5)

        # Simulate the noise level of large cohort with small cohort
        if FLAGS.simulated_clients_per_round is not None:
            noise_multiplier = (noise_multiplier /
                                FLAGS.simulated_clients_per_round *
                                FLAGS.clients_per_round)

        logging.info("DP noise multiplier: {:.2f}".format(noise_multiplier))
        if FLAGS.target_unclipped_quantile == 0.0:
            model_update_aggregation_factory = tff.aggregators. \
                DifferentiallyPrivateFactory.gaussian_fixed(
                    noise_multiplier=noise_multiplier,
                    clients_per_round=FLAGS.clients_per_round,
                    clip=FLAGS.l2_norm_clip)
        else:
            logging.info("Use adaptive clipping for L2 norm clip")
            model_update_aggregation_factory = tff.aggregators. \
                DifferentiallyPrivateFactory.gaussian_adaptive(
                    noise_multiplier=noise_multiplier,
                    clients_per_round=FLAGS.clients_per_round,
                    initial_l2_norm_clip=FLAGS.l2_norm_clip,
                    target_unclipped_quantile=FLAGS.target_unclipped_quantile)

        # Add DP related metrics to report
        model_update_aggregation_factory = tff.learning.add_debug_measurements(
            model_update_aggregation_factory)

    def iterative_process_builder(
            model_fn: Callable[[], tff.learning.Model]
    ) -> tff.templates.IterativeProcess:
        """Creates an iterative process using a given TFF `model_fn`."""
        if FLAGS.epsilon > 0.0:
            return tff.learning.algorithms.build_unweighted_fed_avg(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            model_aggregator=model_update_aggregation_factory,
            use_experimental_simulation_loop=True)
        else:
            return tff.learning.algorithms.build_weighted_fed_avg(
                model_fn=model_fn,
                client_optimizer_fn=client_optimizer_fn,
                server_optimizer_fn=server_optimizer_fn,
                model_aggregator=model_update_aggregation_factory,
                use_experimental_simulation_loop=True)


    model_func = flair_model.get_nn_constructor_fn(FLAGS.model_type)
    model_builder = functools.partial(
        model_func,
        input_shape=image_shape,
        num_classes=num_labels,
        pretrained=FLAGS.restore_model_path is not None,
        adaptation=FLAGS.adaptation)

    if FLAGS.dataset_name == 'flair':
        loss_builder = functools.partial(
            tf.keras.losses.BinaryCrossentropy, from_logits=True)

        metrics_builder = functools.partial(
            flair_metrics.metrics_builder, num_labels=num_labels)
    else:
        loss_builder = functools.partial(
            tf.keras.losses.CategoricalCrossentropy, from_logits=True)
        metrics_builder = functools.partial(
            flair_metrics.cross_entropy_metrics_builder, train=False)

    def tff_model_fn() -> tff.learning.Model:
        """Wraps a tensorflow model to TFF model."""
        return tff.learning.from_keras_model(keras_model=model_builder(),
                                             input_spec=input_spec,
                                             loss=loss_builder(),
                                             metrics=metrics_builder())

    iterative_process = iterative_process_builder(tff_model_fn)

    # training_process accepts client ids as input
    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        train_fed_data.dataset_computation, iterative_process)

    client_ids_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            train_fed_data.client_ids,
            replace=False,
            random_seed=FLAGS.client_datasets_random_seed),
        size=FLAGS.clients_per_round)
    # We convert the output to a list (instead of an np.ndarray) so that it can
    # be used as input to the iterative process.
    client_sampling_fn = lambda x: list(client_ids_fn(x))

    # Build central Keras model for evaluation
    strategy = tf.distribute.MirroredStrategy()
    # To prevent OSError: [Errno 9] Bad file descriptor
    # https://github.com/tensorflow/tensorflow/issues/50487
    atexit.register(strategy._extended._collective_ops._pool.close)

    # Open a strategy scope.
    with strategy.scope():
        eval_model = model_builder()
        eval_model.compile(loss=loss_builder(), metrics=metrics_builder())

    def evaluation_fn(state, eval_data: tf.data.Dataset) -> Dict:
        """Evaluate TFF model state on `eval_data`"""
        state.global_model_weights.assign_weights_to(eval_model)
        eval_metrics = eval_model.evaluate(
            eval_data,
            verbose=0,
            batch_size=FLAGS.eval_batch_size,
            return_dict=True)
        return flair_metrics.flatten_metrics(eval_metrics)

    logging.info('Training model:')
    logging.info(model_builder().summary())

    state = training_process.initialize()
    if FLAGS.restore_model_path is not None:
        logging.info("Loading pretrained weights from {}".format(
            FLAGS.restore_model_path))
        pretrained_model = model_builder()
    
        if FLAGS.model_type == 'resnet18':
            pretrained_model.load_weights(
                FLAGS.restore_model_path, skip_mismatch=True, by_name=True)
        elif FLAGS.model_type == 'resnet50':
            model_builder_non_film = functools.partial(
                model_func,
                input_shape=image_shape,
                num_classes=num_labels,
                pretrained=FLAGS.restore_model_path is not None,
                adaptation='head')
            load_resnet50(pretrained_model, FLAGS.adaptation, FLAGS.restore_model_path, model_builder_non_film)
        pretrained_model_weights = tff.learning.ModelWeights.from_model(pretrained_model)
        state = iterative_process.set_model_weights(state, pretrained_model_weights)

    round_num = 0
    loop_start_time = time.time()
    best_val_loss = float('inf')
    if FLAGS.best_val_loss is not None:
        best_val_loss = FLAGS.best_val_loss

    program_manager = tff.program.FileProgramStateManager(root_dir=FLAGS.save_model_dir)

    if FLAGS.use_latest_checkpoint:
        state, round_num = loop.run_until_complete(program_manager.load_latest(state))
    else:
        loop.run_until_complete(program_manager.save(state, 0))

    save_model_path = os.path.join(
        FLAGS.save_model_dir, f"{FLAGS.model_type}_federated_{num_labels}labels.tf")

    # Main training loop
    while round_num < FLAGS.total_rounds:
        data_prep_start_time = time.time()
        sampled_clients = client_sampling_fn(round_num)
        metrics = {'prepare datasets secs': time.time() - data_prep_start_time}

        result = training_process.next(state, sampled_clients)
        state, round_metrics = result.state, result.metrics
        metrics.update(flair_metrics.flatten_metrics(round_metrics))
        logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
            round_num, (time.time() - loop_start_time) / (round_num + 1)))

        if (round_num + 1) % FLAGS.rounds_per_eval == 0:
            # Compute evaluation metrics
            val_metrics = evaluation_fn(state, val_data)
            metrics.update({'val ' + k: v for k, v in val_metrics.items()})
            # Save model if current iteration has better val metrics
            current_val_loss = float(val_metrics["loss"])
            if current_val_loss < best_val_loss:
                logging.info(f"Saving current best model to {save_model_path}")
                eval_model.save(save_model_path)
                best_val_loss = current_val_loss
            loop.run_until_complete(program_manager.save(state, round_num + 1))
            print('Best val loss {}:'.format(best_val_loss), file=f_val)

        metrics['duration of iteration'] = time.time() - data_prep_start_time
        flair_metrics.print_metrics(metrics, iteration=round_num, f_val=f_val)
        round_num += 1

    # eval_model.load_weights(save_model_path, by_name=True)
    eval_model.load_weights(save_model_path)
    # final dev evaluation
    logging.info("Evaluating best model on val set.")
    val_metrics = eval_model.evaluate(
        val_data, batch_size=FLAGS.eval_batch_size, return_dict=True)
    val_metrics = {'final val ' + k: v for k, v in
                   flair_metrics.flatten_metrics(val_metrics).items()}
    flair_metrics.print_metrics(val_metrics, f_val=f_val)

    # final test evaluation
    logging.info("Evaluating best model on test set.")
    test_metrics = eval_model.evaluate(
        test_data, batch_size=FLAGS.eval_batch_size, return_dict=True)
    test_metrics = {'final test ' + k: v for k, v in
                    flair_metrics.flatten_metrics(test_metrics).items()}
    flair_metrics.print_metrics(test_metrics, f_val=f_val)
    f_val.close()


if __name__ == '__main__':
    app.run(main)
