"""Miscellaneous utility functions for training."""

import numpy as np
from loguru import logger
from torch import nn
import torch.distributed as dist
from typing import Optional, Tuple

import sleap_io as sio
from sleap_nn.data.providers import get_max_instances


def is_distributed_initialized():
    """Check if distributed processes are initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_rank():
    """Return the rank of the current process if torch.distributed is initialized."""
    return dist.get_rank() if is_distributed_initialized() else None


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


def check_memory(
    labels: sio.Labels,
    max_hw: Tuple[int, int],
    model_type: str,
    input_scaling: float,
    crop_size: Optional[int],
):
    """Return memory required for caching the image samples."""
    if model_type == "centered_instance":
        num_samples = len(labels) * get_max_instances(labels)
        img = (labels[0].image / 255.0).astype(np.float32)
        img_mem = (crop_size**2) * img.shape[-1] * img.itemsize * num_samples

        return img_mem

    num_lfs = len(labels)
    img = (labels[0].image / 255.0).astype(np.float32)
    h, w = max_hw[0] * input_scaling, max_hw[1] * input_scaling
    img_mem = h * w * img.shape[-1] * img.itemsize * num_lfs

    return img_mem
