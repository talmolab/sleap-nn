"""This module implements image resizing and padding."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvf
from loguru import logger


def find_padding_for_stride(
    image_height: int, image_width: int, max_stride: int
) -> Tuple[int, int]:
    """Compute padding required to ensure image is divisible by a stride.

    This function is useful for determining how to pad images such that they will not
    have issues with divisibility after repeated pooling steps.

    Args:
        image_height: Scalar integer specifying the image height (rows).
        image_width: Scalar integer specifying the image height (columns).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by.

    Returns:
        A tuple of (pad_height, pad_width), integers with the number of pixels that the
        image would need to be padded by to meet the divisibility requirement.
    """
    # The outer-most modulo handles edge case when image_height % max_stride == 0
    pad_height = (max_stride - (image_height % max_stride)) % max_stride
    pad_width = (max_stride - (image_width % max_stride)) % max_stride
    return pad_height, pad_width


def apply_pad_to_stride(image: torch.Tensor, max_stride: int) -> torch.Tensor:
    """Pad an image to meet a max stride constraint.

    This is useful for ensuring there is no size mismatch between an image and the
    output tensors after multiple downsampling and upsampling steps.

    Args:
        image: Single image tensor of shape (..., channels, height, width).
        max_stride: Scalar integer specifying the maximum stride that the image must be
            divisible by. This is the ratio between the length of the image and the
            length of the smallest tensor it is converted to. This is typically
            `2 ** n_down_blocks`, where `n_down_blocks` is the number of 2-strided
            reduction layers in the model.

    Returns:
        The input image with 0-padding applied to the bottom and/or right such that the
        new shape's height and width are both divisible by `max_stride`.
    """
    if max_stride > 1:
        image_height, image_width = image.shape[-2:]
        pad_height, pad_width = find_padding_for_stride(
            image_height=image_height,
            image_width=image_width,
            max_stride=max_stride,
        )

        if pad_height > 0 or pad_width > 0:
            image = F.pad(
                image,
                (0, pad_width, 0, pad_height),
                mode="constant",
            ).to(torch.float32)
    return image


def resize_image(image: torch.Tensor, scale: float):
    """Rescale an image by a scale factor.

    Args:
        image: Single image tensor of shape (..., channels, height, width).
        scale: Factor to resize the image dimensions by, specified as a float
            scalar.

    Returns:
        The resized image tensor of the same dtype but scaled height and width.
    """
    img_height, img_width = image.shape[-2:]
    new_size = [int(img_height * scale), int(img_width * scale)]
    image = tvf.resize(image, size=new_size)
    return image


def apply_resizer(image: torch.Tensor, instances: torch.Tensor, scale: float = 1.0):
    """Rescale image and keypoints by a scale factor.

    Args:
        image: Image tensor of shape (..., channels, height, width)
        instances: Keypoints tensor.
        scale: Factor to resize the image dimensions by, specified as a float
            scalar. Default: 1.0.

    Returns:
        Tuple with resized image and corresponding keypoints.
    """
    if scale != 1.0:
        image = resize_image(image, scale)
        instances = instances * scale
    return image, instances


def apply_sizematcher(
    image: torch.Tensor,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
):
    """Apply scaling and padding to image to (max_height, max_width) shape."""
    img_height, img_width = image.shape[-2:]
    # pad images to max_height and max_width
    if max_height is None:
        max_height = img_height
    if max_width is None:
        max_width = img_width
    if img_height != max_height or img_width != max_width:
        hratio = max_height / img_height
        wratio = max_width / img_width

        if hratio > wratio:
            eff_scale_ratio = wratio
            target_h = int(round(img_height * wratio))
            target_w = int(round(img_width * wratio))
        else:
            eff_scale_ratio = hratio
            target_w = int(round(img_width * hratio))
            target_h = int(round(img_height * hratio))

        image = tvf.resize(image, size=(target_h, target_w))

        pad_height = max_height - target_h
        pad_width = max_width - target_w

        image = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode="constant",
        ).to(torch.float32)

        return image, eff_scale_ratio
    else:
        return image, 1.0
