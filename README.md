# On the Efficacy of Differentially Private Few-shot Image Classification

This repository contains the code to reproduce the experiments carried out in [On the Efficacy of Differentially Private Few-shot Image Classification](https://arxiv.org/pdf/2302.01190.pdf).

The code has been authored by: Marlon Tobaben, Aliaksandra Shysheya, and John Bronskill.

## Dependencies
This code requires the following:
* Python 3.8 or greater
* PyTorch 1.11 or greater (most of the code is written in PyTorch)
* opacus 1.3 or greater
* prv_accountant 0.2.0 or greater
* optuna 3.0 or greater
* TensorFlow 2.8 or greater (for reading VTAB datasets)
* TensorFlow Datasets 4.5.2 or greater (for reading VTAB datasets)
* Tensorflow Federated 0.30.0 (for Federated Learning experiments)
* Tensorflow Privacy 0.8.0 or greater (for Federated Learning)
* Tensorflow Addons 0.18.0 or greater (for Federated Learning)
* Tensorflow Probability 0.15.0 or greater (for Federated Learning)

## Source Code Libraries
In this work codebase, we rely on the following open source code libraries, some of which we have modified:
- TIMM (for the PyTorch VIT-B implementation): Copyright 2020 Ross Wightman https://github.com/rwightman/pytorch-image-models
- Big Transfer (for the R-50 implementation): Copyright 2020 Google LLC https://github.com/google-research/big_transfer
- Tensorflow Privacy (for the LiRA implementation): Copyright 2022, The TensorFlow Authors https://github.com/tensorflow/privacy
- ML-FLAIR (for the federated learning experiments): Copyright 2020 Apple Inc. https://github.com/apple/ml-flair
- vit-keras (for the tensorFlow VIT-B implementation used in the Federated Learning Experiments): Copyright 2020 Fausto Morales https://github.com/faustomorales/vit-keras

## GPU Requirements
The experiments in the paper are executed on NVIDIA V100 GPUs with 40 GB or a single NVIDIA A100 GPU with 80 GB of memory. Additional information for the centralized experiments (Section 4):
* Larger batch sizes: The code allows for setting a `--max_physical_batch_size` to allow for larger logical batch sizes then what would fit in the GPU memory.
* The experiments under DP allow for training using multiple GPUs (see `src/run_distributed.py`).

## Centralized experiments (Section 4)
### Installation
The following steps will take a considerable length of time and disk space.

1. Clone or download this repository.
2. Install the dependencies listed above.
3. The experiments use datasets obtained from [TensorFlow Datasets](https://www.tensorflow.org/datasets).
   The majority of these are downloaded and pre-processed upon first use. However, the
   [Diabetic Retinopathy](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection)
   and [Resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) datasets need to be
   downloaded manually. Click on the links for details.
4. Switch to the ```src``` directory in this repo and download the BiT pretrained model:

   ```wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz```

### Usage
Switch to the ```src``` directory in this repo and execute `python run.py` (use `python run_distributed.py` for multi-gpu). There are several options that have to be set via the command line. A list is below:

- general experiment options:
    ```
    --feature_extractor <BiT-M-R50x1,vit-b-16> 
    --learnable_params <all,film,none>
    --examples_per_class <number of examples per class, `-1` means the whole training set, `None` enables the VTAB split>
    --seed <for reproducibility, e.g., 0>
    --optimizer <adam,sgd> 
    ```

- DP options
    ```
    --private --target_epsilon <1,2,4,8>
    ```
    (leave ``--private`` away and set ``--target_epsilon -1`` for non-DP)

- dataset:
    ```
    --dataset <caltech101,cifar10,cifar100,clevr_count,clevr_distance,diabetic_retinopathy_detection,dmlab,dsprites_location,dsprites_orientation,dtd,eurosat,kitti,oxford_flowers,oxford_iiit_pet,patch_camelyon,resisc45,smallnorb_azimuth,smallnorb_elevation,sun397,svhn_cropped> 
    ```

- setup options:
    ```
    --download_path_for_tensorflow_datasets <path to dataset>
    --checkpoint_dir <path to checkpoint directory>
    --max_physical_batch_size  <for running under constrained memory> 
    --test_batch_size <for running under constrained memory> 
    ```

- hyperparameter optimization:
    ```
    --tune_params --number_of_trials 20 --save_optuna_study 
    --epochs_lb 1 --epochs_ub 200 
    --train_batch_size_lb 10 --train_batch_size_ub 10000
    --max_grad_norm_lb 0.2 --max_grad_norm_ub 10.0
    --learning_rate_lb 1e-07 --learning_rate_ub 0.01
    ```

The files to plot the Figures from Section 4.1 are in the folder ```analyze_shots```. We also provide scripts that might be helpful if you are running the Experiments of Section 4.1 on Slurm, which is a workload manager used by many large computing clusters, in ```experiment_management```. 

## Membership Inference Attacks (Section 4.3)
1. Train the shadow and target models:
Switch to the ```src``` directory in this repo and execute `python train_lira.py`. The options are the same as for `run.py` except that the hyperparameter optimization options are not available and there is one additional option:
    ```
    --num_shadow_models 256
    ```
   For each configuration two files will be saved off in the checkpoint directory: `in_indices_<learnable_parameters>_<examples_per_class>_<epsilon>.pkl` and `stat_<learnable_parameters>_<examples_per_class>_<epsilon>.pkl`.
2. Compute the LiRA scores:
   Switch to the ```src``` directory in this repo and execute `python process_lira.py`. There is only one option:
    ```
    --data_path <path to a directory that contains all the in_indices*.pkl and stat_*.pkl input files for all the configurations and where the output score*.pkl files will be written>
    ```    
3. Plot the results:
   Switch to the ```analyze_lira``` directory in this repo and execute `python plot_lira.py`. There is only one option:
    ```
    --data_path <path to a directory that contains all the score*.pkl input files for all the configurations>
    ```
    Output plots and data files will be written to a `results` directory located under the ```analyze_lira``` directory.

## Federated learning experiments (Section 5)
### Installation
1. Clone or download this repository.
2. Install the dependencies listed above.
3. Download the FLAIR dataset following the instructions in the [FLAIR repo](https://github.com/apple/ml-flair). 
   The CIFAR-100 and Federated MNIST datasets are obtained from [TensorFlow Datasets](https://www.tensorflow.org/datasets) and they are downloaded and pre-processed upon first use.
4. Switch to the ```ml-flair``` directory in this repo and download the pretrained model you want to use: 

   R-18: ```wget -O /path/to/model https://docs-assets.developer.apple.com/ml-research/datasets/flair/models/resnet18.h5```
   
   R-50: ```wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.h5```
   
   VIT-B: is downloaded automatically upon first use. 

### Usage

Switch to the ```ml-flair``` directory in this repo and execute `python federated_main.py`. There are several options that have to be set via the command line. A list is below:

- model configuration options:
    ```
    --model_type <resnet18, resnet50, vitb16> 
    --adaptation <all,film,head>
    --restore_model_path <path to the pretrained model>
   ```
  
- federated training hyper-parameters:
    ```
    --client_learning_rate 
    --server_learning_rate
    --client_epochs_per_round <number of epochs over the client's data> 
    --total_rounds <number of training rounds>
    --clients_per_round <number of clients to sample in one round>
    --client_batch_size
    ```
  
- dataset options:
  ```
    --dataset_name <flair, cifar100, emnist> 
    --tfrecords_dir <path to dataset>
   ```
  
- DP options
    ```
    --epsilon <0 for non-DP>
    --target_unclipped_quantile <quantile for adaptive clipping>
    --simulated_clients_per_round  
    ```

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our [paper](https://arxiv.org/pdf/2302.01190.pdf).
```
@inproceedings{tobaben2023efficacy,
  title={On the Efficacy of Differentially Private Few-shot Image Classification},
  author={Tobaben, Marlon and Shysheya, Aliaksandra and Bronskill, John and Paverd, Andrew and Tople, Shruti and Zanella-Beguelin, Santiago and Turner, Richard E and Honkela, Antti},
  journal={arXiv preprint arXiv:2302.01190},
  year={2023}
}
```
