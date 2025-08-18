"""Miscellaneous utility functions for data processing."""

from typing import Tuple, List, Any, Optional
import torch
from omegaconf import DictConfig
import sleap_io as sio
from sleap_nn.config.utils import get_model_type_from_cfg
import psutil
import numpy as np
from sleap_nn.data.providers import get_max_instances


def ensure_list(x: Any) -> List[Any]:
    """Convert the input into a list if it is not already."""
    if not isinstance(x, list):
        return [x]
    return x


def make_grid_vectors(
    image_height: int, image_width: int, output_stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make sampling grid vectors from image dimensions.

    This is a useful function for creating the x- and y-vectors that define a sampling
    grid over an image space. These vectors can be used to generate a full meshgrid or
    for equivalent broadcasting operations.

    Args:
        image_height: Height of the image grid that will be sampled, specified as a
            scalar integer.
        image_width: width of the image grid that will be sampled, specified as a
            scalar integer.
        output_stride: Sampling step size, specified as a scalar integer. This can be
            used to specify a sampling grid that has a smaller shape than the image
            grid but with values span the same range. This can be thought of as the
            reciprocal of the output scale, i.e., it will induce subsampling when set to
            values greater than 1.

    Returns:
        Tuple of grid vectors (xv, yv). These are tensors of dtype tf.float32 with
        shapes (grid_width,) and (grid_height,) respectively.

        The grid dimensions are calculated as:
            grid_width = image_width // output_stride
            grid_height = image_height // output_stride
    """
    xv = torch.arange(0, image_width, step=output_stride, dtype=torch.float32)
    yv = torch.arange(0, image_height, step=output_stride, dtype=torch.float32)
    return xv, yv


def expand_to_rank(
    x: torch.Tensor, target_rank: int, prepend: bool = True
) -> torch.Tensor:
    """Expand a tensor to a target rank by adding singleton dimensions in PyTorch.

    Args:
        x: Any `torch.Tensor` with rank <= `target_rank`. If the rank is higher than
            `target_rank`, the tensor will be returned with the same shape.
        target_rank: Rank to expand the input to.
        prepend: If True, singleton dimensions are added before the first axis of the
            data. If False, singleton dimensions are added after the last axis.

    Returns:
        The expanded tensor of the same dtype as the input, but with rank `target_rank`.
        The output has the same exact data as the input tensor and will be identical if
        they are both flattened.
    """
    n_singleton_dims = max(target_rank - x.dim(), 0)
    singleton_dims = [1] * n_singleton_dims
    if prepend:
        new_shape = singleton_dims + list(x.shape)
    else:
        new_shape = list(x.shape) + singleton_dims
    return x.reshape(new_shape)


def gaussian_pdf(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the PDF of an unnormalized 0-centered Gaussian distribution.

    Args:
        x: A tensor of dtype torch.float32 with values to compute the PDF for.
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        A tensor of the same shape as `x`, but with values of a PDF of an unnormalized
        Gaussian distribution. Values of 0 have an unnormalized PDF value of 1.0.
    """
    return torch.exp(-(x**2) / (2 * sigma**2))


def check_memory(
    labels: sio.Labels,
) -> int:
    """Return memory required for caching the image samples from a single labels object.

    Args:
        labels: A `sleap_io.Labels` object containing the labels for a single dataset.

    Returns:
        Memory in bytes required to cache the image samples from the labels object.
    """
    imgs_bytes = []
    for label in labels:
        if label.image is not None:
            img = label.image
            img_bytes = img.nbytes
            imgs_bytes.append(img_bytes)
        else:
            raise ValueError(
                "Labels object contains a label with no image data, which is required for training."
            )
    img_mem = sum(imgs_bytes)
    return img_mem


def check_cache_memory(
    train_labels: List[sio.Labels],
    val_labels: List[sio.Labels],
    memory_buffer: float = 0.2,
) -> bool:
    """Check memory requirements for in-memory caching dataset pipeline.

    Args:
        train_labels: List of `sleap_io.Labels` objects for training data.
        val_labels: List of `sleap_io.Labels` objects for validation data.
        memory_buffer: Fraction of the total image memory required for caching that
            should be reserved as a buffer.

    Returns:
        bool: True if the total memory required for caching is within available system
            memory, False otherwise.
    """
    train_cache_memory_final = 0
    val_cache_memory_final = 0
    for train, val in zip(train_labels, val_labels):
        train_cache_memory = check_memory(train)
        val_cache_memory = check_memory(val)
        train_cache_memory_final += train_cache_memory
        val_cache_memory_final += val_cache_memory

    total_cache_memory = train_cache_memory_final + val_cache_memory_final
    total_cache_memory += memory_buffer * total_cache_memory  # memory required in bytes
    available_memory = psutil.virtual_memory().available  # available memory in bytes

    if total_cache_memory > available_memory:
        return False
    return True
