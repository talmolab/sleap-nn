"""Centroid ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from sleap_nn.export.wrappers.base import (
    BaseExportWrapper,
    antialiased_resize,
    input_scale_resize_rule,
)


class CentroidONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for centroid models.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        max_instances: int = 20,
        output_stride: int = 2,
        input_scale: float = 1.0,
        peak_threshold: float = 0.2,
    ):
        """Initialize centroid ONNX wrapper.

        Args:
            model: Centroid detection model.
            max_instances: Maximum number of instances to detect.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
            peak_threshold: Minimum confidence for a peak to be considered valid.
        """
        super().__init__(model)
        self.max_instances = max_instances
        self.output_stride = output_stride
        self.input_scale = input_scale
        self._resize_rule = input_scale_resize_rule(input_scale)
        self.peak_threshold = peak_threshold

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centroid inference and return fixed-size outputs."""
        image = self._normalize_uint8(image)
        # Resize by input_scale as checkpoint inference does: antialiased, to
        # (int(H * scale), int(W * scale)) of each frame, with H and W read from the
        # input's shape when the graph runs (``antialiased_resize``).
        if self.input_scale != 1.0:
            image = antialiased_resize(image, *self._resize_rule)

        confmaps = self._extract_tensor(self.model(image), ["centroid", "confmap"])
        peaks, values, valid = self._find_topk_peaks(
            confmaps, self.max_instances, self.peak_threshold
        )
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "centroids": peaks,
            "centroid_vals": values,
            "instance_valid": valid,
        }
