"""Handle cropping of instances."""
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Optional
import sleap_io as sio
from kornia.geometry.transform import crop_and_resize
import numpy as np
import torch

def make_centered_bboxes(
    centroids: torch.Tensor, box_height: int, box_width: int
) -> torch.Tensor:
    """Create centered bounding boxes around centroid.

    To be used with `kornia.geometry.transform.crop_and_resize`in the following (clockwise)
    order: top-left, top-right, bottom-right and bottom-left.
    """
    half_h = box_height / 2
    half_w = box_width / 2

    # Get x and y values from the centroids tensor
    x = centroids[..., 0]
    y = centroids[..., 1]

    # Calculate the corner points
    top_left = torch.stack([x - half_w, y - half_h], dim=-1)
    top_right = torch.stack([x + half_w, y - half_h], dim=-1)
    bottom_left = torch.stack([x - half_w, y + half_h], dim=-1)
    bottom_right = torch.stack([x + half_w, y + half_h], dim=-1)

    # Get bounding box
    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=-2)

    return corners

def normalize_bboxes(
    bboxes: torch.Tensor, image_height: int, image_width: int
) -> torch.Tensor:
    """Normalize bounding box coordinates to the range [0, 1].

    This is useful for transforming points for PyTorch operations that require
    normalized image coordinates.

    Args:
        bboxes: Tensor of shape (n_bboxes, 4) and dtype torch.float32, where the last axis
            corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes.
        image_height: Scalar integer indicating the height of the image.
        image_width: Scalar integer indicating the width of the image.

    Returns:
        Tensor of the normalized points of the same shape as `bboxes`.

        The normalization applied to each point is `x / (image_width - 1)` and
        `y / (image_width - 1)`.

    See also: unnormalize_bboxes
    """
    # Compute normalizing factor of shape (1, 4).
    factor = (
        torch.tensor(
            [[image_height, image_width, image_height, image_width]], dtype=torch.float32
        ) 
        - 1
    )

    # Normalize and return.
    normalized_bboxes = bboxes / factor
    return normalized_bboxes

class InstanceCropper(IterDataPipe):
    """Datapipe for cropping instances.

    This DataPipe will produce examples that are instance cropped.

    Attributes:
        source_dp: The previous `DataPipe` with samples that contain an `instances` key.
        crop_width: Width of the crop in pixels
        crop_height: Height of the crop in pixels
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        crop_width: int,
        crop_height: int,
    ):
        """Initialize InstanceCropper with the source `DataPipe."""
        self.source_dp = source_dp
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __iter__(self):
        """Generate instance cropped examples."""
        for example in self.source_dp:
            image = example["image"]  # (frames, channels, height, width)
            instances = example["instances"]  # (frames, n_instances, n_nodes, 2)
            centroids = example["centroids"]  # (frames, n_instances, 2)
            for instance, centroid in zip(instances[0], centroids[0]):
                # Generate bounding boxes from centroid.
                bbox = torch.unsqueeze(
                    make_centered_bboxes(centroid, self.crop_height, self.crop_width), 0
                )  # (frames, 4, 2)

                box_size = (self.crop_height, self.crop_width)

                # Generate cropped image of shape (frames, channels, crop_height, crop_width)
                instance_image = crop_and_resize(
                    image,
                    boxes=bbox,
                    size=box_size,
                )

                # Access top left point (x,y) of bounding box and subtract this offset from
                # position of nodes.
                point = bbox[0][0]
                center_instance = instance - point

                instance_example = {
                    "instance_image": instance_image,  # (frames, channels, crop_height, crop_width)
                    "bbox": bbox,  # (frames, 4, 2)
                    "instance": center_instance,  # (n_instances, 2)
                }
                yield instance_example