"""Handle cropping of instances."""

from typing import Tuple, Dict, Optional
import math
import numpy as np
import sleap_io as sio
import torch
from kornia.geometry.transform import crop_and_resize


def compute_augmentation_padding(
    bbox_size: float,
    rotation_max: float = 0.0,
    scale_max: float = 1.0,
) -> int:
    """Compute padding needed to accommodate augmentation transforms.

    When rotation and scaling augmentations are applied, the bounding box of an
    instance can expand beyond its original size. This function calculates the
    padding needed to ensure the full instance remains visible after augmentation.

    Args:
        bbox_size: The size of the instance bounding box (max of width/height).
        rotation_max: Maximum absolute rotation angle in degrees. For symmetric
            rotation ranges like [-180, 180], pass 180.
        scale_max: Maximum scaling factor. For scale range [0.9, 1.1], pass 1.1.

    Returns:
        Padding in pixels to add around the bounding box (total, not per side).
    """
    if rotation_max == 0.0 and scale_max <= 1.0:
        return 0

    # For a square bbox rotated by angle θ, the new bbox has side length:
    # L' = L * (|cos(θ)| + |sin(θ)|)
    # Maximum expansion occurs at 45°: L' = L * sqrt(2)
    # For arbitrary angle: we use the worst case within the rotation range
    rotation_rad = math.radians(min(abs(rotation_max), 90))
    rotation_factor = abs(math.cos(rotation_rad)) + abs(math.sin(rotation_rad))

    # For angles > 45°, the factor increases, max at 45° = sqrt(2)
    # But for angles approaching 90°, it goes back to 1
    # Worst case in any range including 45° is sqrt(2)
    if abs(rotation_max) >= 45:
        rotation_factor = math.sqrt(2)

    # Combined expansion factor
    expansion_factor = rotation_factor * max(scale_max, 1.0)

    # Total padding needed (both sides)
    expanded_size = bbox_size * expansion_factor
    padding = expanded_size - bbox_size

    return int(math.ceil(padding))


def find_max_instance_bbox_size(labels: sio.Labels) -> float:
    """Find the maximum bounding box dimension across all instances in labels.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.

    Returns:
        The maximum bounding box dimension (max of width or height) across all instances.
    """
    max_length = 0.0
    for lf in labels:
        for inst in lf.instances:
            if not inst.is_empty:
                pts = inst.numpy()
                diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
                diff_x = 0 if np.isnan(diff_x) else diff_x
                max_length = np.maximum(max_length, diff_x)
                diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
                diff_y = 0 if np.isnan(diff_y) else diff_y
                max_length = np.maximum(max_length, diff_y)
    return float(max_length)


def find_instance_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    min_crop_size: Optional[int] = None,
) -> int:
    """Compute the size of the largest instance bounding box from labels.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin padding.
        maximum_stride: Ensure that the returned crop size is divisible by this value.
            Useful for ensuring that the crop size will not be truncated in a given
            architecture.
        min_crop_size: The crop size set by the user.

    Returns:
        An integer crop size denoting the length of the side of the bounding boxes that
        will contain the instances when cropped. The returned crop size will be larger
        or equal to the input `min_crop_size`.

        This accounts for stride and padding when ensuring divisibility.
    """
    # Check if user-specified crop size is divisible by max stride
    min_crop_size = 0 if min_crop_size is None else min_crop_size
    if (min_crop_size > 0) and (min_crop_size % maximum_stride == 0):
        return min_crop_size

    # Calculate crop size
    min_crop_size_no_pad = min_crop_size - padding
    max_length = 0.0
    for lf in labels:
        for inst in lf.instances:
            if not inst.is_empty:  # only if at least one point is not nan
                pts = inst.numpy()
                diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
                diff_x = 0 if np.isnan(diff_x) else diff_x
                max_length = np.maximum(max_length, diff_x)
                diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
                diff_y = 0 if np.isnan(diff_y) else diff_y
                max_length = np.maximum(max_length, diff_y)
                max_length = np.maximum(max_length, min_crop_size_no_pad)

    max_length += float(padding)
    crop_size = math.ceil(max_length / float(maximum_stride)) * maximum_stride

    return int(crop_size)


def make_centered_bboxes(
    centroids: torch.Tensor, box_height: int, box_width: int
) -> torch.Tensor:
    """Create centered bounding boxes around centroid.

    To be used with `kornia.geometry.transform.crop_and_resize`in the following
    (clockwise) order: top-left, top-right, bottom-right and bottom-left.

    Args:
        centroids: A tensor of centroids with shape (n_centroids, 2), where n_centroids is the
            number of centroids, and the last dimension represents x and y coordinates.
        box_height: The desired height of the bounding boxes.
        box_width: The desired width of the bounding boxes.

    Returns:
        torch.Tensor: A tensor containing bounding box coordinates for each centroid.
            The output tensor has shape (n_centroids, 4, 2), where n_centroids is the number
            of centroids, and the second dimension represents the four corner points of
            the bounding boxes, each with x and y coordinates. The order of the corners
            follows a clockwise arrangement: top-left, top-right, bottom-right, and
            bottom-left.
    """
    half_h = box_height / 2
    half_w = box_width / 2

    # Get x and y values from the centroids tensor.
    x = centroids[..., 0]
    y = centroids[..., 1]

    # Calculate the corner points.
    top_left = torch.stack([x - half_w, y - half_h], dim=-1)
    top_right = torch.stack([x + half_w, y - half_h], dim=-1)
    bottom_left = torch.stack([x - half_w, y + half_h], dim=-1)
    bottom_right = torch.stack([x + half_w, y + half_h], dim=-1)

    # Get bounding box.
    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=-2)

    offset = torch.tensor([[+0.5, +0.5], [-0.5, +0.5], [-0.5, -0.5], [+0.5, -0.5]]).to(
        corners.device
    )

    return corners + offset


def generate_crops(
    image: torch.Tensor,
    instance: torch.Tensor,
    centroid: torch.Tensor,
    crop_size: Tuple[int],
) -> Dict[str, torch.Tensor]:
    """Generate cropped image for the given centroid.

    Args:
        image: Input source image. (n_samples, C, H, W)
        instance: Keypoints for the instance to be cropped. (n_nodes, 2)
        centroid: Centroid of the instance to be cropped. (2)
        crop_size: (height, width) of the crop to be generated.

    Returns:
        A dictionary with cropped images, bounding box for the cropped instance, keypoints and
        centroids adjusted to the crop.
    """
    box_size = crop_size

    # Generate bounding boxes from centroid.
    instance_bbox = torch.unsqueeze(
        make_centered_bboxes(centroid, box_size[0], box_size[1]), 0
    )  # (n_samples=1, 4, 2)

    # Generate cropped image of shape (n_samples, C, crop_H, crop_W)
    instance_image = crop_and_resize(
        image,
        boxes=instance_bbox,
        size=box_size,
    )

    # Access top left point (x,y) of bounding box and subtract this offset from
    # position of nodes.
    point = instance_bbox[0][0]
    center_instance = (instance - point).unsqueeze(0)  # (n_samples=1, n_nodes, 2)
    centered_centroid = (centroid - point).unsqueeze(0)  # (n_samples=1, 2)

    cropped_sample = {
        "instance_image": instance_image,
        "instance_bbox": instance_bbox,
        "instance": center_instance,
        "centroid": centered_centroid,
    }

    return cropped_sample
