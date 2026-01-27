"""Miscellaneous utility functions for data processing."""

from typing import Tuple, List, Any, Optional
import sys
import torch
from omegaconf import DictConfig
import sleap_io as sio
from sleap_nn.config.utils import get_model_type_from_cfg
import psutil
import numpy as np
from loguru import logger
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


def estimate_cache_memory(
    train_labels: List[sio.Labels],
    val_labels: List[sio.Labels],
    num_workers: int = 0,
    memory_buffer: float = 0.2,
) -> dict:
    """Estimate memory requirements for in-memory caching dataset pipeline.

    This function calculates the total memory needed for caching images, accounting for:
    - Raw image data size
    - Python object overhead (dictionary keys, numpy array wrappers)
    - DataLoader worker memory overhead (Copy-on-Write duplication on Unix systems)
    - General memory buffer for training overhead

    When using DataLoader with num_workers > 0, worker processes are spawned via fork()
    on Unix systems. While Copy-on-Write (CoW) initially shares memory, Python's reference
    counting can trigger memory page duplication when workers access cached data.

    Args:
        train_labels: List of `sleap_io.Labels` objects for training data.
        val_labels: List of `sleap_io.Labels` objects for validation data.
        num_workers: Number of DataLoader worker processes. When > 0, additional memory
            overhead is estimated for worker process duplication.
        memory_buffer: Fraction of memory to reserve as buffer for training overhead
            (model weights, activations, gradients, etc.). Default: 0.2 (20%).

    Returns:
        dict: Memory estimation breakdown with keys:
            - 'raw_cache_bytes': Raw image data size in bytes
            - 'python_overhead_bytes': Estimated Python object overhead
            - 'worker_overhead_bytes': Estimated memory for DataLoader workers
            - 'buffer_bytes': Memory buffer for training overhead
            - 'total_bytes': Total estimated memory requirement
            - 'available_bytes': Available system memory
            - 'sufficient': True if total <= available, False otherwise
    """
    # Calculate raw image cache size
    train_cache_bytes = 0
    val_cache_bytes = 0
    num_train_samples = 0
    num_val_samples = 0

    for train, val in zip(train_labels, val_labels):
        train_cache_bytes += check_memory(train)
        val_cache_bytes += check_memory(val)
        num_train_samples += len(train)
        num_val_samples += len(val)

    raw_cache_bytes = train_cache_bytes + val_cache_bytes
    total_samples = num_train_samples + num_val_samples

    # Python object overhead: dict keys, numpy array wrappers, tuple keys
    # Estimate ~200 bytes per sample for Python object overhead
    python_overhead_per_sample = 200
    python_overhead_bytes = total_samples * python_overhead_per_sample

    # Worker memory overhead
    # When num_workers > 0, workers are forked or spawned depending on platform.
    # Default start methods (Python 3.8+):
    #   - Linux: fork (Copy-on-Write, partial memory duplication)
    #   - macOS: spawn (full dataset copy to each worker, changed in Python 3.8)
    #   - Windows: spawn (full dataset copy to each worker)
    worker_overhead_bytes = 0
    if num_workers > 0:
        if sys.platform == "linux":
            # Linux uses fork() with Copy-on-Write by default
            # Estimate 25% duplication per worker due to Python refcounting
            # triggering CoW page copies
            worker_overhead_bytes = int(raw_cache_bytes * 0.25 * num_workers)
            if num_workers >= 4:
                logger.info(
                    f"Using in-memory caching with {num_workers} DataLoader workers. "
                    f"Estimated additional memory for workers: "
                    f"{worker_overhead_bytes / (1024**3):.2f} GB"
                )
        else:
            # macOS (darwin) and Windows use spawn - dataset is copied to each worker
            # Since Python 3.8, macOS defaults to spawn due to fork safety issues
            # With caching enabled, we avoid pickling labels_list, but the cache
            # dict is still part of the dataset and gets copied to each worker
            worker_overhead_bytes = int(raw_cache_bytes * 0.5 * num_workers)
            platform_name = "macOS" if sys.platform == "darwin" else "Windows"
            logger.warning(
                f"Using in-memory caching with {num_workers} DataLoader workers on {platform_name}. "
                f"Memory usage may be significantly higher than estimated (~{worker_overhead_bytes / (1024**3):.1f} GB extra) "
                f"due to spawn-based multiprocessing. "
                f"Consider using disk caching or num_workers=0 for large datasets."
            )

    # Memory buffer for training overhead (model, gradients, activations)
    subtotal = raw_cache_bytes + python_overhead_bytes + worker_overhead_bytes
    buffer_bytes = int(subtotal * memory_buffer)

    total_bytes = subtotal + buffer_bytes
    available_bytes = psutil.virtual_memory().available

    return {
        "raw_cache_bytes": raw_cache_bytes,
        "python_overhead_bytes": python_overhead_bytes,
        "worker_overhead_bytes": worker_overhead_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "sufficient": total_bytes <= available_bytes,
        "num_samples": total_samples,
    }


def check_cache_memory(
    train_labels: List[sio.Labels],
    val_labels: List[sio.Labels],
    memory_buffer: float = 0.2,
    num_workers: int = 0,
) -> bool:
    """Check memory requirements for in-memory caching dataset pipeline.

    This function determines if the system has sufficient memory for in-memory
    image caching, accounting for DataLoader worker processes.

    Args:
        train_labels: List of `sleap_io.Labels` objects for training data.
        val_labels: List of `sleap_io.Labels` objects for validation data.
        memory_buffer: Fraction of memory to reserve as buffer. Default: 0.2 (20%).
        num_workers: Number of DataLoader worker processes. When > 0, additional memory
            overhead is estimated for worker process duplication.

    Returns:
        bool: True if the total memory required for caching is within available system
            memory, False otherwise.
    """
    estimate = estimate_cache_memory(
        train_labels=train_labels,
        val_labels=val_labels,
        num_workers=num_workers,
        memory_buffer=memory_buffer,
    )

    if not estimate["sufficient"]:
        total_gb = estimate["total_bytes"] / (1024**3)
        available_gb = estimate["available_bytes"] / (1024**3)
        raw_gb = estimate["raw_cache_bytes"] / (1024**3)
        logger.info(
            f"Memory check failed: need ~{total_gb:.2f} GB "
            f"(raw cache: {raw_gb:.2f} GB, {estimate['num_samples']} samples), "
            f"available: {available_gb:.2f} GB"
        )

    return estimate["sufficient"]
