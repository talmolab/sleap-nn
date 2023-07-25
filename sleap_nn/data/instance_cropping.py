"""Handle cropping of instances."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl
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
    The coordinates must be in the x, y order.
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


class InstanceCropper(dp.IterDataPipe):
    """Datapipe for cropping instances.

    This DataPipe will produce examples that are instance cropped.

    Attributes:
        source_dp: The previous `DataPipe` with samples that contain an `instances` key.
        crop_width: Width of the crop in pixels
        crop_height: Height of the crop in pixels
    """

    def __init__(
        self,
        source_dp: dp.IterDataPipe,
        crop_width: int,
        crop_height: int,
    ):
        """Initialize InstanceCropper with the source `DataPipe."""
        self.source_dp = source_dp
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __iter__(self):
        """Add `"centroids"` key to example."""
        for example in self.source_dp:
            image = example["image"]
            instances = example["instances"]
            centroids = example["centroids"]
            for instance, centroid in zip(instances[0], centroids[0]):
                print(instance)
                print(centroid)
                bboxes = torch.unsqueeze(
                    make_centered_bboxes(centroid, self.crop_height, self.crop_width), 0
                )

                print(f"make  {bboxes}")

                # Crop.
                box_size = (self.crop_width, self.crop_height)
                print(f" boxes {bboxes.shape}")
                print(f" image {image.shape}")
                print(f" box_size {box_size}")

                instance_images = crop_and_resize(
                    image,
                    boxes=bboxes,
                    size=box_size,
                )

                point = bboxes[0][0]
                center_instances = torch.sub(instance, point)

                example = {
                    "instance_image": instance_images,
                    "image": image,
                    "bbox": bboxes,
                    "instances": instances,
                    "centered_instances": center_instances,
                    "centroids": centroids,
                }
                # TODO return instance-wise examples
                yield example
