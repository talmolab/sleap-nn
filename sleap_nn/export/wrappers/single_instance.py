"""Single-instance ONNX wrapper."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class SingleInstanceONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for single-instance models.

    This wrapper handles full-frame inference assuming a single instance per frame.
    For each body part (channel), it finds the global maximum in the confidence map.

    Expects input images as uint8 tensors in [0, 255].

    Attributes:
        model: The trained backbone model that outputs confidence maps.
        output_stride: Output stride of the model (e.g., 4 means confmaps are 1/4 the
            input resolution).
        input_scale: Factor to scale input images before inference.
    """

    def __init__(
        self,
        model: nn.Module,
        output_stride: int = 4,
        input_scale: float = 1.0,
    ):
        """Initialize the single-instance wrapper.

        Args:
            model: The trained backbone model.
            output_stride: Output stride of the model. Default: 4.
            input_scale: Factor to scale input images. Default: 1.0.
        """
        super().__init__(model)
        self.output_stride = output_stride
        self.input_scale = input_scale

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run single-instance inference.

        Args:
            image: Input image tensor of shape (batch, channels, height, width).
                Expected as uint8 [0, 255] values.

        Returns:
            Dictionary with:
                peaks: Peak coordinates of shape (batch, n_nodes, 2) in (x, y) format.
                peak_vals: Peak confidence values of shape (batch, n_nodes).
        """
        # Normalize uint8 [0, 255] to float32 [0, 1]
        image = self._normalize_uint8(image)

        # Apply input scaling if needed
        if self.input_scale != 1.0:
            height = int(image.shape[-2] * self.input_scale)
            width = int(image.shape[-1] * self.input_scale)
            image = F.interpolate(
                image, size=(height, width), mode="bilinear", align_corners=False
            )

        # Run model to get confidence maps: (batch, n_nodes, height, width)
        confmaps = self._extract_tensor(
            self.model(image), ["single", "instance", "confmap"]
        )

        # Find global peak for each channel: (batch, n_nodes, 2), (batch, n_nodes)
        peaks, values = self._find_global_peaks(confmaps)

        # Scale peaks from confmap coordinates to image coordinates
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "peaks": peaks,
            "peak_vals": values,
        }
