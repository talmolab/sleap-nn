"""This module implements image resizing and padding."""

from typing import Dict, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data.datapipes.datapipe import IterDataPipe


class SizeMatcher(IterDataPipe):
    """IterDataPipe for resizing and padding images.

    This IterDataPipe will produce examples containing the resized image and original
    shape of the image before padding/ resizing is applied.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        max_height: Maximum height the image should be padded to. If not provided, the
                    original image size will be retained.
        max_width: Maximum width the image should be padded to. If not provided, the
                    original image size will be retained.
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        max_height: int = None,
        max_width: int = None,
    ):
        """Initialize labels attribute of the class."""
        self.source_datapipe = source_datapipe
        self.max_width = max_width
        self.max_height = max_height

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the resized image and `orig_size` key to
        represent the original shape of the source image."""
        for ex in self.source_datapipe:
            img_height, img_width = ex["image"].shape[-2:]
            # pad images to max_height and max_width
            if self.max_height is not None:  # only if user provides
                pad_height = self.max_height - img_height
                pad_width = self.max_width - img_width
                if pad_height < 0:
                    raise Exception(
                        f"Max height {self.max_height} should be greater than the current image height: {img_height}"
                    )
                if pad_width < 0:
                    raise Exception(
                        f"Max width {self.max_width} should be greater than the current image width: {img_width}"
                    )
                ex["image"] = F.pad(
                    ex["image"],
                    (0, pad_width, 0, pad_height),
                    mode="constant",
                ).to(torch.float32)

            ex["orig_size"] = torch.Tensor([img_height, img_width])

            yield ex
