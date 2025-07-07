"""This module implements data pipeline blocks for normalization operations."""

import torch
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
