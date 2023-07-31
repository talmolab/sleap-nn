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

                print(f" instance_image shape - {instance_image.shape}")
                print(f" bbox shape - {bbox.shape}")
                print(f" instance shape - {instance.shape}")
                instance_example = {
                    "instance_image": instance_image,  # (frames, channels, crop_height, crop_width)
                    "bbox": bbox[0],  # (frames, 4, 2)
                    "instance": center_instance,  # (n_instances, 2)
                }
                yield instance_example
