"""Centered-instance ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class CenteredInstanceONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for centered-instance models.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        output_stride: int = 4,
        input_scale: float = 1.0,
    ):
        """Initialize centered instance ONNX wrapper.

        Args:
            model: Centered instance model for pose estimation.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
        """
        super().__init__(model)
        self.output_stride = output_stride
        self.input_scale = input_scale

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centered-instance inference on crops."""
        image = self._normalize_uint8(image)
        if self.input_scale != 1.0:
            height = int(image.shape[-2] * self.input_scale)
            width = int(image.shape[-1] * self.input_scale)
            image = F.interpolate(
                image, size=(height, width), mode="bilinear", align_corners=False
            )

        confmaps = self._extract_tensor(
            self.model(image), ["centered", "instance", "confmap"]
        )
        peaks, values = self._find_global_peaks(confmaps)
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "peaks": peaks,
            "peak_vals": values,
        }
