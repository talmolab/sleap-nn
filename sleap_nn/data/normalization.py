"""This module implements data pipeline blocks for normalization operations."""

from typing import Dict, Iterator

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe
import torchvision.transforms.v2.functional as F


class Normalizer(IterDataPipe):
    """IterDataPipe for applying normalization.

    This IterDataPipe will normalize the image from `uint8` to `float32` and scale the
    values to the range `[0, 1]` and converts to grayscale/ rgb.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        is_rgb: bool = False,
    ) -> None:
        """Initialize the `IterDataPipe`."""
        self.source_dp = source_dp
        self.is_rgb = is_rgb

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the normalized image."""
        for ex in self.source_dp:
            image = ex["image"]
            if not torch.is_floating_point(image):
                image = image.to(torch.float32) / 255.0

            # convert to rgb
            if self.is_rgb and image.shape[-3] != 3:
                image = F.to_grayscale(image, num_output_channels=3)

                image = torch.concatenate([image, image, image], dim=-3)

            # convert to grayscale
            if not self.is_rgb and image.shape[-3] != 1:
                image = F.rgb_to_grayscale(image, num_output_channels=1)

            ex["image"] = image

            yield ex
