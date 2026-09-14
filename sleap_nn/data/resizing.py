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
            )
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


_SIZEMATCHER_WARNED_KEYS: set = set()


def _warn_size_mismatch(
    img_height: int, img_width: int, max_height: int, max_width: int
) -> None:
    """Emit a one-time warning per unique (input, target) size combination.

    `apply_sizematcher` is called per frame, so dedup is required to avoid
    flooding the log when every frame of a video hits the same size mismatch.
    """
    key = (int(img_height), int(img_width), int(max_height), int(max_width))
    if key in _SIZEMATCHER_WARNED_KEYS:
        return
    _SIZEMATCHER_WARNED_KEYS.add(key)
    direction = (
        "downscaled"
        if (img_height > max_height or img_width > max_width)
        else "upscaled"
    )
    logger.warning(
        f"Input image size ({img_height}x{img_width}, HxW) does not match "
        f"the configured max_height/max_width ({max_height}x{max_width}); "
        f"frames will be {direction} and padded on every call, which is "
        f"slower than running on natively-sized inputs. To eliminate this "
        f"overhead, set max_height/max_width in your config to match the "
        f"input dimensions."
    )


def compute_eff_scale(
    img_hw: Tuple[int, int],
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> float:
    """Return the scale `apply_sizematcher` would apply to a frame of this size.

    The size matcher fits each frame into ``(max_height, max_width)`` preserving
    aspect ratio, so the scale is the *smaller* of the two ratios. This is the
    single definition of that factor: `apply_sizematcher` calls it to derive the
    ``eff_scale`` it returns, and the crop-size helpers in
    `sleap_nn.data.instance_cropping` call it to measure labels in the same space
    the crops are taken in. Keeping one implementation is what stops the sizing
    and the cropping from drifting apart.

    Args:
        img_hw: The frame's ``(height, width)`` in its native resolution.
        max_hw: The configured ``(max_height, max_width)``. ``None``, or a
            ``None`` in either slot, means "no target for that axis", which
            leaves the frame at its native size on that axis.

    Returns:
        The scale factor, ``1.0`` when the frame already matches the target.
    """
    img_height, img_width = img_hw
    max_height, max_width = (None, None) if max_hw is None else max_hw
    if max_height is None:
        max_height = img_height
    if max_width is None:
        max_width = img_width
    if img_height == max_height and img_width == max_width:
        return 1.0
    return min(max_height / img_height, max_width / img_width)


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
        _warn_size_mismatch(img_height, img_width, max_height, max_width)
        eff_scale_ratio = compute_eff_scale(
            (img_height, img_width), (max_height, max_width)
        )
        target_h = int(round(img_height * eff_scale_ratio))
        target_w = int(round(img_width * eff_scale_ratio))

        image = tvf.resize(image, size=(target_h, target_w))

        pad_height = max_height - target_h
        pad_width = max_width - target_w

        image = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode="constant",
        )

        return image, eff_scale_ratio
    else:
        return image, 1.0
