"""This module implements data pipeline blocks for normalization operations."""

from typing import Dict, Iterator

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe


class Normalizer(IterDataPipe):
    """IterDataPipe for applying normalization.

    This IterDataPipe will normalize the image from `uint8` to `float32` and scale the
    values to the range `[0, 1]`.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
    ) -> None:
        """Initialize the `IterDataPipe`."""
        self.source_dp = source_dp

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the augmented image and instance."""
        for ex in self.source_dp:
            if not torch.is_floating_point(ex["image"]):
                ex["image"] = ex["image"].to(torch.float32) / 255.0
            yield ex
