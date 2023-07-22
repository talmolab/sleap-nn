"""This module implements data pipeline blocks for normalization operations."""
import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe


class Normalizer(IterDataPipe):
    """DataPipe for applying normalization.

    This DataPipe will normalize the image from `uint8` to `float32` and scale the
    values to the range `[0, 1]`.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
    ):
        """Initialize the block."""
        self.source_dp = source_dp

    def __iter__(self):
        """Return an example dictionary with the augmented image and instance."""
        for ex in self.source_dp:
            if not torch.is_floating_point(ex["image"]):
                ex["image"] = ex["image"].to(torch.float32) / 255.0
            yield ex
