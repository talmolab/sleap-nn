"""Miscellaneous utility functions for data processing."""

from typing import Tuple, List, Any, Optional
import torch
from omegaconf import DictConfig
import sleap_io as sio
from sleap_nn.config.utils import get_model_type_from_cfg
import psutil
import numpy as np
from sleap_nn.data.providers import get_max_instances
#new import for rotating calipers
from scipy.spatial import ConvexHull


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
    max_hw: Tuple[int, int],
    model_type: str,
    input_scaling: float,
    crop_size: Optional[int],
):
    """Return memory required for caching the image samples from a single labels object."""
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


def check_cache_memory(train_labels, val_labels, config: DictConfig):
    """Check memory requirements for in-memory caching dataset pipeline."""
    train_cache_memory_final = 0
    val_cache_memory_final = 0
    model_type = get_model_type_from_cfg(config)
    for train, val in zip(train_labels, val_labels):
        train_cache_memory = check_memory(
            train,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            model_type=model_type,
            input_scaling=config.data_config.preprocessing.scale,
            crop_size=config.data_config.preprocessing.crop_hw[0],
        )
        val_cache_memory = check_memory(
            val,
            max_hw=(
                config.data_config.preprocessing.max_height,
                config.data_config.preprocessing.max_width,
            ),
            model_type=model_type,
            input_scaling=config.data_config.preprocessing.scale,
            crop_size=config.data_config.preprocessing.crop_hw[0],
        )
        train_cache_memory_final += train_cache_memory
        val_cache_memory_final += val_cache_memory

    total_cache_memory = train_cache_memory_final + val_cache_memory_final
    total_cache_memory += 0.1 * total_cache_memory  # memory required in bytes
    available_memory = psutil.virtual_memory().available  # available memory in bytes

    if total_cache_memory > available_memory:
        return False
    return True

def rotating_calipers(points):
    """
    Computes the convex hull of a set of points using the rotating calipers method.
    
    Args:
        points (np.ndarray): An array of shape (N, 2) representing the points.
        
    Returns:
        np.ndarray: Indices of the points that form the convex hull.
    """
    
    # Remove NaN values and check if there are enough valid points
    valid_points = points[~np.isnan(points).any(axis=1)]
        
    
    # Determine the convex hull using scipy's ConvexHull
    hull = ConvexHull(valid_points)
    hull_points = valid_points[hull.vertices]

    min_area = np.inf #intialize minimum area to infinity
    best_box = None #to store the best bounding box found

    # Iterate through each edge of the convex hull
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        # Compute the angle of the edge
        edge = p2 - p1
        angle = -np.arctan2(edge[1], edge[0])

        # Rotate points to align with the edge
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]])
        rotated_points = (hull_points - p1) @ rotation_matrix.T

        # Compute the bounding box of the rotated points
        xmin = np.min(rotated_points[:, 0])
        xmax = np.max(rotated_points[:, 0])
        ymin = np.min(rotated_points[:, 1])
        ymax = np.max(rotated_points[:, 1])
        area = (xmax - xmin) * (ymax - ymin)

        # Update the best bounding box if the area is smaller
        if area < min_area:
            min_area = area
            #rectangle corners in rotated coordinates
            box = np.array([[xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax]])
            #rotate back to original coordinates
            best_box = (box @ rotation_matrix) + p1

    return best_box