"""This module implements data pipeline blocks for normalization operations."""

from typing import Dict, Iterator

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe
import torchvision.transforms.v2.functional as F


def convert_to_grayscale(image: torch.Tensor):
    """Convert given image to Grayscale image (single-channel).

    This functions converts the input image to grayscale only if the given image is not
    a single-channeled image.

    Args:
        image: Tensor image of shape (..., 3, H, W)

    Returns:
        Tensor image of shape (..., 1, H, W).
    """
    if image.shape[-3] != 1:
        image = F.rgb_to_grayscale(image, num_output_channels=1)
    return image


def convert_to_rgb(image: torch.Tensor):
    """Convert given image to RGB image (three-channel image).

    This functions converts the input image to RGB only if the given image is not
    a RGB image.

    Args:
        image: Tensor image of shape (..., 1, H, W)

    Returns:
        Tensor image of shape (..., 3, H, W).
    """
    if image.shape[-3] != 3:
        image = image.repeat(1, 3, 1, 1)
    return image


def apply_normalization(image: torch.Tensor):
    """Normalize image tensor."""
    if not torch.is_floating_point(image):
        image = image.to(torch.float32) / 255.0
    return image


class Normalizer(IterDataPipe):
    """IterDataPipe for applying normalization.

    This IterDataPipe will normalize the image from `uint8` to `float32` and scale the
    values to the range `[0, 1]` and converts to grayscale/ rgb.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
    ) -> None:
        """Initialize the `IterDataPipe`."""
        self.source_dp = source_dp
        self.ensure_rgb = ensure_rgb
        self.ensure_grayscale = ensure_grayscale

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the normalized image."""
        for ex in self.source_dp:
            image = ex["image"]
            if not torch.is_floating_point(image):
                image = image.to(torch.float32) / 255.0

            # convert to rgb
            if self.ensure_rgb:
                image = convert_to_rgb(image)

            # convert to grayscale
            elif self.ensure_grayscale:
                image = convert_to_grayscale(image)

            ex["image"] = image  # (n_samples, channels, height, width)

            yield ex
