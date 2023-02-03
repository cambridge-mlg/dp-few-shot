import os
import socket
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from optuna.exceptions import ExperimentalWarning

from run import Learner

# these lines follow from https://opacus.ai/tutorials/ddp_tutorial

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def launch(rank, world_size, port):
    setup(rank, world_size, port)
    with warnings.catch_warnings():
        # irrelevant for research but relevant for production
        warnings.filterwarnings("ignore", message=r".*Secure RNG turned off.*")
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        # We are aware some features of optuna are experimental
        warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")

        distributed_learner = Learner(rank)
        distributed_learner.run()

    cleanup()


if __name__ == '__main__':
    port = get_free_port()
    world_size = torch.cuda.device_count()
    mp.spawn(
        launch,
        args=(world_size, port),
        nprocs=world_size,
        join=True
    )
