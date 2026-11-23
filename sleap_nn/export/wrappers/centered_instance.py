"""Centered-instance ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from sleap_nn.export.wrappers.base import (
    BaseExportWrapper,
    antialiased_resize,
    input_scale_resize_rule,
)


class CenteredInstanceONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for centered-instance models.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        output_stride: int = 4,
        input_scale: float = 1.0,
        peak_threshold: float = 0.2,
    ):
        """Initialize centered instance ONNX wrapper.

        Args:
            model: Centered instance model for pose estimation.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
            peak_threshold: Minimum confidence for a peak to be considered valid.
        """
        super().__init__(model)
        self.output_stride = output_stride
        self.input_scale = input_scale
        self._resize_rule = input_scale_resize_rule(input_scale)
        self.peak_threshold = peak_threshold

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centered-instance inference on crops."""
        image = self._normalize_uint8(image)
        # Resize by input_scale as checkpoint inference does: antialiased, to
        # (int(H * scale), int(W * scale)) of each crop, with H and W read from the
        # input's shape when the graph runs (``antialiased_resize``).
        if self.input_scale != 1.0:
            image = antialiased_resize(image, *self._resize_rule)

        confmaps = self._extract_tensor(
            self.model(image), ["centered", "instance", "confmap"]
        )
        peaks, values = self._find_global_peaks(confmaps, self.peak_threshold)
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "peaks": peaks,
            "peak_vals": values,
        }
