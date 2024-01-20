"""Handle cropping of instances."""
from typing import Iterator, Tuple, Dict

import torch
from kornia.geometry.transform import crop_and_resize
from torch.utils.data.datapipes.datapipe import IterDataPipe


def make_centered_bboxes(
    centroids: torch.Tensor, box_height: int, box_width: int
) -> torch.Tensor:
    """Create centered bounding boxes around centroid.

    To be used with `kornia.geometry.transform.crop_and_resize`in the following
    (clockwise) order: top-left, top-right, bottom-right and bottom-left.

    Args:
        centroids: A tensor of centroids with shape (channels, 2), where channels is the
            number of centroids, and the last dimension represents x and y coordinates.
        box_height: The desired height of the bounding boxes.
        box_width: The desired width of the bounding boxes.

    Returns:
        torch.Tensor: A tensor containing bounding box coordinates for each centroid.
            The output tensor has shape (channels, 4, 2), where channels is the number
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

    offset = torch.tensor([[+0.5, +0.5], [-0.5, +0.5], [-0.5, -0.5], [+0.5, -0.5]])

    return corners + offset


class InstanceCropper(IterDataPipe):
    """IterDataPipe for cropping instances.

    This IterDataPipe will produce examples that are instance cropped.

    Attributes:
        source_dp: The previous `IterDataPipe` with samples that contain an `instances` key.
        crop_hw: Height and Width of the crop in pixels
    """

    def __init__(self, source_dp: IterDataPipe, crop_hw: Tuple[int, int]) -> None:
        """Initialize InstanceCropper with the source `IterDataPipe`."""
        self.source_dp = source_dp
        self.crop_hw = crop_hw

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate instance cropped examples."""
        for ex in self.source_dp:
            image = ex["image"]  # (B=1, C, H, W)
            instances = ex.pop("instances")  # (B=1, num_instances, num_nodes, 2)
            centroids = ex.pop("centroids")  # (B=1, num_instances, 2)
            for instance, centroid in zip(instances[0], centroids[0]):
                # Generate bounding boxes from centroid.
                instance_bbox = torch.unsqueeze(
                    make_centered_bboxes(centroid, self.crop_hw[0], self.crop_hw[1]), 0
                )  # (B=1, 4, 2)

                box_size = (self.crop_hw[0], self.crop_hw[1])

                # Generate cropped image of shape (B=1, C, crop_H, crop_W)
                instance_image = crop_and_resize(
                    image,
                    boxes=instance_bbox,
                    size=box_size,
                )

                # Access top left point (x,y) of bounding box and subtract this offset from
                # position of nodes.
                point = instance_bbox[0][0]
                center_instance = instance - point

                instance_example = {
                    "instance_image": instance_image,  # (B=1, C, crop_H, crop_W)
                    "instance_bbox": instance_bbox,  # (B=1, 4, 2)
                    "instance": center_instance.unsqueeze(0),  # (B=1, num_nodes, 2)
                    "centroid": centroid.unsqueeze(0),  # (B=1, 2)
                }
                ex.update(instance_example)
                yield ex
