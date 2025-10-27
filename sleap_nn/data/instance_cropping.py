"""Handle cropping of instances."""

from typing import Tuple, Dict, Optional
import math
import numpy as np
import sleap_io as sio
import torch
from kornia.geometry.transform import crop_and_resize
from sleap_nn.data.utils import rotating_calipers
import kornia


def get_cropped_img(image: torch.Tensor, instance: torch.Tensor, head_idx: int):
    """Crop and rotate an image using the oriented bounding box (OBB) of a given instance.

    This function performs a padding-aware crop around the instance keypoints using the minimum-area
    rotating calipers OBB. It then aligns the longest edge with the x-axis, warps the image and keypoints
    accordingly, and applies a conditional 180° rotation if the head is facing left. The output is a
    torch-native equivalent of OpenCV's getAffineTransform + warpAffine behavior.

    Args:
        image (torch.Tensor): A float tensor of shape (C, H, W), representing an RGB image.
        instance (torch.Tensor): A float tensor of shape (N, 2), representing keypoint coordinates of one instance.
        head_idx (int): Index of the head keypoint, used to determine leftward orientation.

    Returns:
        cropped_image (torch.Tensor): Cropped and rotated image of shape (C, H, W), aligned to face +x.
        adjusted_kpts (torch.Tensor): Keypoints of shape (N, 2), transformed to match the cropped image coordinates.
        src_pts (torch.Tensor): Three source points from the padded OBB used for affine transformation (3, 2).
        dst_pts (torch.Tensor): Three target points in the crop destination space used for affine warping (3, 2).
        rotated (bool): True if the instance was rotated 180° to face the positive x-axis, otherwise False.
    """
    # Define padding
    pad = 32

    # ensure dtype
    image = image.float()
    device = image.device
    instance = instance.to(device)

    # Get OBB from keypoints
    obb_coords = rotating_calipers(instance)

    # Find longest edge and roll OBB
    dists = torch.norm(obb_coords - torch.roll(obb_coords, shifts=-1, dims=0), dim=1)
    max_index = torch.argmax(dists)
    obb_coords = torch.roll(obb_coords, shifts=max_index.item(), dims=0)

    # Compute padded OBB by expanding each corner outward from center
    center = obb_coords.mean(dim=0, keepdims=True)
    vecs = obb_coords - center
    norms = torch.norm(vecs, dim=1).unsqueeze(1)  # shape: (4, 1)
    norms = torch.where(
        norms == 0, torch.ones_like(norms), norms
    )  # avoid division by zero

    # Find the OBB edge closest to the x-axis (smallest absolute angle)
    best_idx = 0
    min_abs_angle = float("inf")
    for i in range(4):
        edge = obb_coords[(i + 1) % 4] - obb_coords[i]
        angle = torch.atan2(edge[1], edge[0])
        if abs(angle) < min_abs_angle:
            min_abs_angle = abs(angle)
            best_idx = i

    # Roll so this edge is [0] -> [1]
    obb_coords = torch.roll(obb_coords, shifts=-best_idx, dims=0)
    edge = obb_coords[1] - obb_coords[0]
    angle = torch.atan2(edge[1], edge[0])

    # If the edge points left, reverse the OBB
    if edge[0] < 0:
        obb_coords = obb_coords[::-1]
        edge = obb_coords[1] - obb_coords[0]
        angle = torch.atan2(edge[1], edge[0])

    # Defining the width/height based on the obb coordinates
    width = torch.norm(obb_coords[1] - obb_coords[0])
    height = torch.norm(obb_coords[3] - obb_coords[0])

    # If the crop is taller than wide, rotate OBB by 90 deg to make it horizontal
    if height > width:
        obb_coords = torch.roll(obb_coords, shifts=-1, dims=0)  # rotate OBB 90 degrees
        edge = obb_coords[1] - obb_coords[0]
        angle = torch.atan2(edge[1], edge[0])
        if edge[0] < 0:
            obb_coords = torch.flip(obb_coords, dims=[0])
            edge = obb_coords[1] - obb_coords[0]
            angle = torch.atan2(edge[1], edge[0])
        width = torch.norm(obb_coords[1] - obb_coords[0])
        height = torch.norm(obb_coords[3] - obb_coords[0])

    # Add padding to the final crop dimensions
    width += pad * 2
    height += pad * 2

    # Build affine from OBB -> crop box
    src_pts = (
        obb_coords[:3].clone().to(dtype=torch.float32, device=device)
    )  # using corners of OBB

    # rectangular region we want to map the OBB onto
    dst_pts = torch.tensor(
        [[pad, pad], [width - pad, pad], [width - pad, height - pad]],
        dtype=torch.float32,
        device=device,
    )

    ones = torch.ones((3, 1), device=device)
    src = torch.cat(
        [src_pts, ones], dim=1
    )  # appending 1s to the source points to compute affine transformation

    # solves least squares system giving the affine that best maps src_pts -> dst_pts
    affine_matrix = torch.linalg.lstsq(src, dst_pts).solution.T

    # Warp the image with the affine transform
    cropped_image = kornia.geometry.transform.warp_affine(
        image.unsqueeze(0), affine_matrix.unsqueeze(0), dsize=(int(height), int(width))
    )[0]

    # Warp the keypoints with the same affine
    kp_homo = torch.cat(
        [instance.to(device), torch.ones((instance.shape[0], 1), device=device)], dim=1
    )
    adjusted_kpts = (affine_matrix @ kp_homo.T).T

    # Define head/body keypoints
    head_x = adjusted_kpts[head_idx, 0]
    body_center_x = adjusted_kpts[:, 0][~torch.isnan(adjusted_kpts[:, 0])].mean()

    # Rotate 180° if facing left (by comparing the head keypoint to the body center keypoints)
    rotated = False
    if head_x < body_center_x:
        rotated = True
        # Rotate image 180°
        cropped_image = torch.rot90(cropped_image, k=2, dims=[1, 2])

        adjusted_kpts[:, 0] = cropped_image.shape[2] - adjusted_kpts[:, 0]
        adjusted_kpts[:, 1] = cropped_image.shape[1] - adjusted_kpts[:, 1]

    return cropped_image, adjusted_kpts, src_pts, dst_pts, rotated


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
