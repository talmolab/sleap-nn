"""This module implements data pipeline blocks for normalization operations."""

import torch
import torchvision.transforms.v2.functional as F


def normalize_on_gpu(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor on GPU after transfer.

    This function is called in the model's forward() method after the image has been
    transferred to GPU. It converts uint8 images to float32 and normalizes to [0, 1].

    By performing normalization on GPU after transfer, we reduce PCIe bandwidth by 4x
    (transferring 1 byte/pixel as uint8 instead of 4 bytes/pixel as float32). This
    provides up to 17x speedup for the transfer+normalization stage.

    This function handles two cases:
    1. uint8 tensor with values in [0, 255] -> convert to float32 and divide by 255
    2. float32 tensor with values in [0, 255] (e.g., from preprocessing that cast to
       float32 without normalizing) -> divide by 255

    Args:
        image: Tensor image that may be uint8 or float32 with values in [0, 255] range.

    Returns:
        Float32 tensor normalized to [0, 1] range.
    """
    if not torch.is_floating_point(image):
        # uint8 -> float32 normalized
        image = image.float() / 255.0
    elif image.max() > 1.0:
        # float32 but not normalized (values > 1 indicate [0, 255] range)
        image = image / 255.0
    return image


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


def apply_normalization(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor from uint8 [0, 255] to float32 [0, 1].

    This function is used during training data preprocessing where augmentation
    operations (kornia) require float32 input.

    For inference, normalization is deferred to GPU via `normalize_on_gpu()` in the
    model's forward() method to reduce PCIe bandwidth.

    Args:
        image: Tensor image (typically uint8 with values in [0, 255]).

    Returns:
        Float32 tensor normalized to [0, 1] range.
    """
    if not torch.is_floating_point(image):
        image = image.to(torch.float32) / 255.0
    return image
