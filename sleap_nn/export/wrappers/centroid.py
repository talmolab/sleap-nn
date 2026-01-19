"""Centroid ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


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
    ):
        """Initialize centroid ONNX wrapper.

        Args:
            model: Centroid detection model.
            max_instances: Maximum number of instances to detect.
            output_stride: Output stride for confidence maps.
            input_scale: Input scaling factor.
        """
        super().__init__(model)
        self.max_instances = max_instances
        self.output_stride = output_stride
        self.input_scale = input_scale

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run centroid inference and return fixed-size outputs."""
        image = self._normalize_uint8(image)
        if self.input_scale != 1.0:
            height = int(image.shape[-2] * self.input_scale)
            width = int(image.shape[-1] * self.input_scale)
            image = F.interpolate(
                image, size=(height, width), mode="bilinear", align_corners=False
            )

        confmaps = self._extract_tensor(self.model(image), ["centroid", "confmap"])
        peaks, values, valid = self._find_topk_peaks(confmaps, self.max_instances)
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "centroids": peaks,
            "centroid_vals": values,
            "instance_valid": valid,
        }
