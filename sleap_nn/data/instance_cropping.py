"""Handle cropping of instances."""

from typing import Iterator, Tuple, Dict, Optional
import math
import numpy as np
import sleap_io as sio
import torch
from kornia.geometry.transform import crop_and_resize
from torch.utils.data.datapipes.datapipe import IterDataPipe
from sleap_nn.data.resizing import find_padding_for_stride


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
            max_length = np.maximum(
                max_length, np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
            )
            max_length = np.maximum(
                max_length, np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
            )
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
        image: Input source image.
        instance: Keypoints for the instance to be cropped.
        centroid: Centroid of the instance to be cropped.
        crop_size: (height, width) of the crop to be generated.

    Returns:
        A dictionary with cropped images, bounding box for the cropped instance, keypoints and
        centroids adjusted to the crop.
    """
    box_size = crop_size

    # Generate bounding boxes from centroid.
    instance_bbox = torch.unsqueeze(
        make_centered_bboxes(centroid, box_size[0], box_size[1]), 0
    )  # (n_samples, 4, 2)

    # Generate cropped image of shape (n_samples, C, crop_H, crop_W)
    instance_image = crop_and_resize(
        image,
        boxes=instance_bbox,
        size=box_size,
    )

    # Access top left point (x,y) of bounding box and subtract this offset from
    # position of nodes.
    point = instance_bbox[0][0]
    center_instance = (instance - point).unsqueeze(0)  # (n_samples, n_nodes, 2)
    centered_centroid = (centroid - point).unsqueeze(0)  # (n_samples, 2)

    cropped_sample = {
        "instance_image": instance_image,
        "instance_bbox": instance_bbox,
        "instance": center_instance,
        "centroid": centered_centroid,
    }

    return cropped_sample


class InstanceCropper(IterDataPipe):
    """IterDataPipe for cropping instances.

    This IterDataPipe will produce examples that are instance cropped.

    Attributes:
        source_dp: The previous `IterDataPipe` with samples that contain an `instances` key.
        crop_hw: Height and width of the crop in pixels.

    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        crop_hw: Tuple[int, int],
    ) -> None:
        """Initialize InstanceCropper with the source `IterDataPipe`."""
        self.source_dp = source_dp
        self.crop_hw = crop_hw

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate instance cropped examples."""
        for ex in self.source_dp:
            image = ex["image"]  # (n_samples, C, H, W)
            instances = ex["instances"]  # (n_samples, n_instances, n_nodes, 2)
            centroids = ex["centroids"]  # (n_samples, n_instances, 2)
            del ex["instances"]
            del ex["centroids"]
            del ex["image"]
            for cnt, (instance, centroid) in enumerate(zip(instances[0], centroids[0])):
                if cnt == ex["num_instances"]:
                    break
                box_size = (self.crop_hw[0], self.crop_hw[1])

                # Generate bounding boxes from centroid.
                instance_bbox = torch.unsqueeze(
                    make_centered_bboxes(centroid, box_size[0], box_size[1]), 0
                )  # (n_samples, 4, 2)

                # Generate cropped image of shape (n_samples, C, crop_H, crop_W)
                instance_image = crop_and_resize(
                    image,
                    boxes=instance_bbox,
                    size=box_size,
                )

                # Access top left point (x,y) of bounding box and subtract this offset from
                # position of nodes.
                point = instance_bbox[0][0]
                center_instance = instance - point
                centered_centroid = centroid - point

                instance_example = {
                    "instance_image": instance_image,  # (n_samples, C, crop_H, crop_W)
                    "instance_bbox": instance_bbox,  # (n_samples, 4, 2)
                    "instance": center_instance.unsqueeze(0),  # (n_samples, n_nodes, 2)
                    "centroid": centered_centroid.unsqueeze(0),  # (n_samples, 2)
                }
                ex.update(instance_example)

                yield ex
