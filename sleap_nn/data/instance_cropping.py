"""Handle cropping of instances."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl
from typing import Optional
import sleap_io as sio
from kornia.geometry.transoform import crop_and_resize
import numpy as np
import torch


def make_centered_bboxes(
    centroids: torch.Tensor, box_height: int, box_width: int
) -> torch.Tensor:
    delta = (
        torch.Tensor(
            [[-box_height + 1, -box_width + 1, box_height - 1, box_width - 1]],
            torch.float32,
        )
        * 0.5
    )
    bboxes = torch.gather(centroids, [1, 0, 1, 0], dim=-1) + delta
    return bboxes


def normalize_bboxes(
    bboxes: torch.Tensor, image_height: int, image_width: int
) -> torch.Tensor:
    factor = (
        torch.Tensor([[image_height, image_width, image_height, image_width]]).to(
            torch.float32
        )
        - 1
    )

    # Normalize and return.
    normalized_bboxes = bboxes / factor
    return normalized_bboxes


def crop_bboxes(image: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:

    indices = [[0, 0], [0, 1]]
    y1x1 = bboxes.masked_select(indices)
    indices = [[0, 2], [0, 3]]
    y2x2 = bboxes.masked_select(indices)

    box_size = torch.round((y2x2 - y1x1) + 1).int()  # (height, width)

    # Normalize bounding boxes.
    image_height = image.shape[0]
    image_width = image.shape[1]
    normalized_bboxes = normalize_bboxes(
        bboxes, image_height=image_height, image_width=image_width
    )

    # Crop.
    crops = crop_and_resize(
        tensor=torch.unsqueeze(image, 0),
        boxes=normalized_bboxes,

        size=box_size
        mode="bilinear"
    )

    # Cast back to original dtype and return.
    crops = crops.to(image.dtype)
    return crops


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
            bboxes = make_centered_bboxes(
                example["centroids"], self.crop_height, self.crop_width
            )

            instance_images = crop_bboxes(example["image"], bboxes)

            bboxes_x1y1 = torch.gather(bboxes, [1, 0], dim=1)
            n_instances = bboxes.shape[0]
            all_instances = torch.repeat_interleave(
                example["instances"].unsqueeze(0),
                n_instances,
                dim=0,
            )
            all_instances = all_instances - bboxes_x1y1.view(n_instances, 1, 1, 2)