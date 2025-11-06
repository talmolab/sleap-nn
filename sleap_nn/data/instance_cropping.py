"""Handle cropping of instances."""

from typing import Tuple, Dict, Optional
import math
import numpy as np
import sleap_io as sio
import torch
from kornia.geometry.transform import crop_and_resize, warp_affine

def apply_egocentric_rotation(
    image: torch.Tensor,
    instance: torch.Tensor,
    centroid: torch.Tensor,
    orientation_anchor_inds: Union[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply egocentric rotation to align centroid->orientation_anchor vector with x-axis.

    Args:
        image: Input image. Shape: (1, C, H, W) - singleton batch dimension for kornia
        instance: Input keypoints for a single instance. Shape: (n_nodes, 2)
        centroid: Centroid point. Shape: (2,)
        orientation_anchor_inds: Index or list of indices (in priority order) of nodes to use
            for orientation alignment (e.g., head/snout). The function will try nodes in order
            and use the first one that is not NaN. The vector from centroid to this point will
            be aligned with the positive x-axis. If a single int is provided, it's treated as
            a single-element list.

    Returns:
        Tuple of (rotated_image, rotated_instance, rotated_centroid, rotation_angle) all rotated
        so that the centroid->orientation_anchor vector points along positive x-axis.
        - rotated_image: Shape (1, C, H, W)
        - rotated_instance: Shape (n_nodes, 2)
        - rotated_centroid: Shape (2,) - same as input (centroid doesn't move)
        - rotation_angle: Rotation angle applied in radians (torch.Tensor scalar)
    """
    # Convert single int to list for uniform handling
    if isinstance(orientation_anchor_inds, int):
        orientation_anchor_inds = [orientation_anchor_inds]
    
    # Try nodes in priority order, use first one that's not NaN
    orientation_anchor = None
    for anchor_ind in orientation_anchor_inds:
        candidate_anchor = instance[anchor_ind, :]  # (2,)
        # Check if this candidate is valid (not NaN)
        if not torch.isnan(candidate_anchor).any():
            orientation_anchor = candidate_anchor
            break
    
    # If no valid orientation anchor found, no rotation needed
    if orientation_anchor is None:
        return image, instance, centroid, torch.tensor(0.0, dtype=image.dtype, device=image.device)

    # Compute vector from centroid to orientation anchor
    direction_vector = orientation_anchor - centroid  # (2,)

    # Check if vector has zero length (shouldn't rotate in this case)
    vector_length = torch.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
    if vector_length < 1e-6:
        return image, instance, centroid, torch.tensor(0.0, dtype=image.dtype, device=image.device)
    
    # Compute angle to align with x-axis (positive x-axis = 0 degrees)
    # atan2(y, x) gives angle from x-axis in radians
    # In image coordinates: +x is right, +y is down
    # We want to rotate so that direction_vector points along +x axis (to the right)
    current_angle = torch.atan2(direction_vector[1], direction_vector[0])  # Current angle from +x axis
    target_angle = torch.tensor(0.0, dtype=image.dtype, device=image.device)  # We want it to point along +x axis (to the right)
    rotation_angle_rad = target_angle - current_angle  # Angle to rotate (counter-clockwise is positive)
    rotation_angle_deg = torch.rad2deg(rotation_angle_rad)
    
    # Get image dimensions for rotation center
    _, C, H, W = image.shape

    # Rotate image around centroid using affine transformation
    # To rotate around point (cx, cy), we need: translate -> rotate -> translate_back
    # The affine matrix is: M = T_back @ R @ T_to_origin

    cx, cy = centroid[0].item(), centroid[1].item()
    cos_a = torch.cos(rotation_angle_rad).item()
    sin_a = torch.sin(rotation_angle_rad).item()

    # Create 2x3 affine transformation matrix
    # For rotation around (cx, cy), the matrix is:
    # [cos(θ)  -sin(θ)  -cx*cos(θ) + cy*sin(θ) + cx]
    # [sin(θ)   cos(θ)  -cx*sin(θ) - cy*cos(θ) + cy]
    affine_matrix = torch.tensor([
        [cos_a, -sin_a, -cx * cos_a + cy * sin_a + cx],
        [sin_a, cos_a, -cx * sin_a - cy * cos_a + cy]
    ], dtype=image.dtype, device=image.device).unsqueeze(0)  # (1, 2, 3)

    # Apply affine transformation
    rotated_image = warp_affine(
        image,
        affine_matrix,
        dsize=(H, W),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    
    # Create rotation matrix for keypoints: [cos(θ) -sin(θ); sin(θ) cos(θ)]
    # This rotates counter-clockwise by rotation_angle_rad
    # Reuse cos_a and sin_a computed above for consistency
    rotation_matrix = torch.tensor(
        [[cos_a, -sin_a],
         [sin_a, cos_a]],
        dtype=image.dtype,
        device=image.device,
    )
    
    # Rotate keypoints around centroid (centroid stays fixed)
    instance_relative = instance - centroid  # (n_nodes, 2) - relative to centroid
    rotated_instance_relative = torch.matmul(instance_relative, rotation_matrix.T)  # (n_nodes, 2)
    rotated_instance = rotated_instance_relative + centroid  # (n_nodes, 2) - back to absolute coords
    
    # Centroid doesn't move (we rotated around it)
    rotated_centroid = centroid

    return rotated_image, rotated_instance, rotated_centroid, rotation_angle_rad

def find_instance_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    input_scaling: float = 1.0,
    min_crop_size: Optional[int] = None,
) -> int:
    """Compute the size of the largest instance bounding box from labels.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin padding.
        maximum_stride: Ensure that the returned crop size is divisible by this value.
            Useful for ensuring that the crop size will not be truncated in a given
            architecture.
        input_scaling: Float factor indicating the scale of the input images if any
            scaling will be done before cropping.
        min_crop_size: The crop size set by the user.

    Returns:
        An integer crop size denoting the length of the side of the bounding boxes that
        will contain the instances when cropped. The returned crop size will be larger
        or equal to the input `min_crop_size`.

        This accounts for stride, padding and scaling when ensuring divisibility.
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
                pts *= input_scaling
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
