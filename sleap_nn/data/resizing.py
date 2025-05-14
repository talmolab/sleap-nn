"""This module implements image resizing and padding."""

from typing import Dict, Iterator, Optional, Tuple, List, Union

import torch
from sleap_nn.data.providers import LabelsReaderDP, VideoReader
import torchvision.transforms.v2.functional as tvf
from torch.utils.data.datapipes.datapipe import IterDataPipe
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
        A tuple with the input image with 0-padding applied to the bottom and/or right such that the
        new shape's height and width are both divisible by `max_stride` and (pad_width_left, pad_height_top)
        to shift the ground-truth keypoints according to the padded image.
    """
    if max_stride > 1:
        image_height, image_width = image.shape[-2:]
        pad_height, pad_width = find_padding_for_stride(
            image_height=image_height,
            image_width=image_width,
            max_stride=max_stride,
        )

        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left

        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top

        if pad_height > 0 or pad_width > 0:
            image = tvf.pad(
                image,
                (pad_width_left, pad_height_top, pad_width_right, pad_height_bottom),
                0,
                "constant",
            ).to(torch.float32)
    return image, (pad_width_left, pad_height_top)


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


def apply_padding(
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

        pad_height = max_height - img_height
        pad_width = max_width - img_width

        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left

        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top

        image = tvf.pad(
            image,
            (pad_width_left, pad_height_top, pad_width_right, pad_height_bottom),
            0,
            "constant",
        ).to(torch.float32)

        return image, (pad_width_left, pad_height_top)
    else:
        return image, (0, 0)


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

        pad_width_left = pad_width // 2
        pad_width_right = pad_width - pad_width_left

        pad_height_top = pad_height // 2
        pad_height_bottom = pad_height - pad_height_top

        image = tvf.pad(
            image,
            (pad_width_left, pad_height_top, pad_width_right, pad_height_bottom),
            0,
            "constant",
        ).to(torch.float32)

        return image, eff_scale_ratio, (pad_width_left, pad_height_top)
    else:
        return image, 1.0, (0, 0)


class Resizer(IterDataPipe):
    """IterDataPipe for resizing images.

    This IterDataPipe will produce examples containing the resized image and original
    shape of the image before resizing is applied.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        scale: Factor to resize the image dimensions by, specified as a float
            scalar.
        keep_original: True if original image should be retained.
        image_key: Key for the image to be scaled. One of ["image", "instance_image"]
        instances_key: Key for the instances to be corrected to input scale. One of
            ["instances", "instance"]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        scale: int = 1.0,
        keep_original: bool = False,
        image_key: str = "image",
        instances_key: str = "instances",
    ):
        """Initialize labels attribute of the class."""
        self.source_datapipe = source_datapipe
        self.scale = scale
        self.keep_original = keep_original
        self.image_key = image_key
        self.instances_key = instances_key

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the resized image and `orig_size` key to represent the original shape of the source image."""
        for ex in self.source_datapipe:
            # Rescaling
            if self.keep_original:
                ex["original_image"] = ex["image"]
            if self.scale != 1.0:
                ex[self.image_key] = resize_image(ex[self.image_key], self.scale)
                ex[self.instances_key] = ex[self.instances_key] * self.scale
            yield ex


class PadToStride(IterDataPipe):
    """IterDataPipe to pad images based on max stride.

    This IterDataPipe will produce examples containing the padded image and original
    shape of the image before resizing is applied.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        max_stride: Maximum stride in a model that the images must be divisible by.
            If > 1, this will pad the bottom and right of the images to ensure they meet
            this divisibility criteria. Padding is applied after the scaling specified
            in the `scale` attribute.
        image_key: Key for the image to be scaled. One of ["image", "instance_image"]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        max_stride: int = 1,
        image_key: str = "image",
    ):
        """Initialize labels attribute of the class."""
        self.source_datapipe = source_datapipe
        self.max_stride = max_stride
        self.image_key = image_key

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the resized image and `orig_size` key to represent the original shape of the source image."""
        for ex in self.source_datapipe:
            ex[self.image_key], _ = apply_pad_to_stride(
                ex[self.image_key], self.max_stride
            )
            yield ex


class SizeMatcher(IterDataPipe):
    """IterDataPipe for padding smaller images to same shape.

    This IterDataPipe will produce examples containing the resized image and original
    shape of the image before padding is applied.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        provider: Data Provider.
        max_height: Maximum height the image should be padded to. If not provided, the
                    original image size will be retained.
        max_width: Maximum width the image should be padded to. If not provided, the
                    original image size will be retained.
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        provider: Optional[Union[LabelsReaderDP, VideoReader]] = None,
        max_height: Optional[int] = None,
        max_width: Optional[int] = None,
    ):
        """Initialize labels attribute of the class."""
        self.source_datapipe = source_datapipe
        if max_height is None and max_width is None:
            if provider is not None:
                max_height, max_width = provider.max_height_and_width
        self.max_width = max_width
        self.max_height = max_height

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the resized image and `orig_size` key to represent the original shape of the source image."""
        for ex in self.source_datapipe:
            img_height, img_width = ex["image"].shape[-2:]
            # pad images to max_height and max_width
            if self.max_height is None:
                self.max_height = img_height
            if self.max_width is None:
                self.max_width = img_width
            pad_height = self.max_height - img_height
            pad_width = self.max_width - img_width
            if pad_height < 0:
                message = f"Max height {self.max_height} should be greater than the current image height: {img_height}"
                logger.error(message)
                raise Exception(message)
            if pad_width < 0:
                message = f"Max width {self.max_width} should be greater than the current image width: {img_width}"
                logger.error(message)
                raise Exception(message)
            ex["image"] = F.pad(
                ex["image"],
                (0, pad_width, 0, pad_height),
                mode="constant",
            ).to(torch.float32)

            yield ex
