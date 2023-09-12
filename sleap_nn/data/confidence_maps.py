"""Generate confidence maps."""
from typing import Dict, Optional

import sleap_io as sio
import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe

from sleap_nn.data.utils import make_grid_vectors


def make_confmaps(
    points: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Make confidence maps from a set of points from a single instance.

    Args:
        points: A tensor of points of shape `(n_nodes, 2)` and dtype `torch.float32` where
            the last axis corresponds to (x, y) pixel coordinates on the image. These
            can contain NaNs to indicate missing points.
        xv: Sampling grid vector for x-coordinates of shape `(grid_width,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape `(grid_height,)` and dtype
            `torch.float32`. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        sigma: Standard deviation of the 2D Gaussian distribution sampled to generate
            confidence maps.

    Returns:
        Confidence maps as a tensor of shape `(n_nodes, grid_height, grid_width)` of
        dtype `torch.float32`.
    """
    x = torch.reshape(points[:, 0], (-1, 1, 1))
    y = torch.reshape(points[:, 1], (-1, 1, 1))
    cm = torch.exp(
        -(
            (torch.reshape(xv, (1, 1, -1)) - x) ** 2
            + (torch.reshape(yv, (1, -1, 1)) - y) ** 2
        )
        / (2 * sigma**2)
    )

    # Replace NaNs with 0.
    cm = torch.where(torch.isnan(cm), 0.0, cm)
    return cm


class ConfidenceMapGenerator(IterDataPipe):
    """DataPipe for generating confidence maps.

    This DataPipe will generate confidence maps for examples from the input pipeline.
    Input examples must contain image of shape (frames, channels, crop_height, crop_width)
    and instance of shape (n_instances, 2).

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain an instance and
            an image.
        sigma: The standard deviation of the Gaussian distribution that is used to
            generate confidence maps.
        output_stride: The relative stride to use when generating confidence maps.
            A larger stride will generate smaller confidence maps.
        instance_key: The name of the key where the instance points (n_instances, 2) are.
        image_key: The name of the key where the image (frames, channels, crop_height, crop_width) is.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        sigma: int = 1.5,
        output_stride: int = 1,
        instance_key: str = "instance",
        image_key: str = "instance_image",
    ) -> None:
        """Initialize ConfidenceMapGenerator with input `DataPipe`, sigma, and output stride."""
        self.source_dp = source_dp
        self.sigma = sigma
        self.output_stride = output_stride
        self.instance_key = instance_key
        self.image_key = image_key

    def __iter__(self) -> Dict[str, torch.Tensor]:
        """Generate confidence maps for each example."""
        for example in self.source_dp:
            instance = example[self.instance_key]
            width = example[self.image_key].shape[-1]
            height = example[self.image_key].shape[-2]

            xv, yv = make_grid_vectors(height, width, self.output_stride)

            confidence_maps = make_confmaps(
                instance, xv, yv, self.sigma
            )  # (n_nodes, height, width)

            example["confidence_maps"] = confidence_maps
            yield example
