"""ONNX wrapper for bottom-up multiclass (supervised ID) models."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class BottomUpMultiClassONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for bottom-up multiclass (supervised ID) models.

    This wrapper handles models that output both confidence maps for keypoint
    detection and class maps for identity classification. Unlike PAF-based
    bottom-up models, multiclass models use class maps to assign identity to
    each detected peak, then group peaks by identity.

    The wrapper performs:
    1. Peak detection in confidence maps (GPU)
    2. Class probability sampling at peak locations (GPU)
    3. Returns fixed-size tensors for CPU-side grouping

    Expects input images as uint8 tensors in [0, 255].

    Attributes:
        model: The underlying PyTorch model.
        n_nodes: Number of keypoint nodes in the skeleton.
        n_classes: Number of identity classes.
        max_peaks_per_node: Maximum number of peaks to detect per node.
        cms_output_stride: Output stride of the confidence map head.
        class_maps_output_stride: Output stride of the class maps head.
        input_scale: Scale factor applied to input images before inference.
    """

    def __init__(
        self,
        model: nn.Module,
        n_nodes: int,
        n_classes: int = 2,
        max_peaks_per_node: int = 20,
        cms_output_stride: int = 4,
        class_maps_output_stride: int = 8,
        input_scale: float = 1.0,
    ):
        """Initialize the wrapper.

        Args:
            model: The underlying PyTorch model.
            n_nodes: Number of keypoint nodes.
            n_classes: Number of identity classes (e.g., 2 for male/female).
            max_peaks_per_node: Maximum peaks per node to detect.
            cms_output_stride: Output stride of confidence maps.
            class_maps_output_stride: Output stride of class maps.
            input_scale: Scale factor for input images.
        """
        super().__init__(model)
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.max_peaks_per_node = max_peaks_per_node
        self.cms_output_stride = cms_output_stride
        self.class_maps_output_stride = class_maps_output_stride
        self.input_scale = input_scale

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run bottom-up multiclass inference.

        Args:
            image: Input image tensor of shape (batch, channels, height, width).
                   Expected to be uint8 in [0, 255].

        Returns:
            Dictionary with keys:
                - "peaks": Detected peak coordinates (batch, n_nodes, max_peaks, 2).
                    Coordinates are in input image space (x, y).
                - "peak_vals": Peak confidence values (batch, n_nodes, max_peaks).
                - "peak_mask": Boolean mask for valid peaks (batch, n_nodes, max_peaks).
                - "class_probs": Class probabilities at each peak location
                    (batch, n_nodes, max_peaks, n_classes).

            Postprocessing on CPU uses `classify_peaks_from_maps()` to group
            peaks by identity using Hungarian matching.
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

        batch_size = image.shape[0]

        # Forward pass
        out = self.model(image)

        # Extract outputs
        # Note: Use "classmaps" as a single hint to avoid "map" matching "confmaps"
        confmaps = self._extract_tensor(out, ["confmap", "multiinstance"])
        class_maps = self._extract_tensor(out, ["classmaps", "classmapshead"])

        # Find top-k peaks per node
        peaks, peak_vals, peak_mask = self._find_topk_peaks_per_node(
            confmaps, self.max_peaks_per_node
        )

        # Scale peaks to input image space
        peaks = peaks * self.cms_output_stride

        # Sample class maps at peak locations
        class_probs = self._sample_class_maps_at_peaks(class_maps, peaks, peak_mask)

        # Scale peaks for output (accounting for input scale)
        if self.input_scale != 1.0:
            peaks = peaks / self.input_scale

        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "peak_mask": peak_mask,
            "class_probs": class_probs,
        }

    def _sample_class_maps_at_peaks(
        self,
        class_maps: torch.Tensor,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample class map values at peak locations.

        Args:
            class_maps: Class maps of shape (batch, n_classes, height, width).
            peaks: Peak coordinates in cms_output_stride space,
                shape (batch, n_nodes, max_peaks, 2) in (x, y) order.
            peak_mask: Boolean mask for valid peaks (batch, n_nodes, max_peaks).

        Returns:
            Class probabilities at each peak location,
            shape (batch, n_nodes, max_peaks, n_classes).
        """
        batch_size, n_classes, cm_height, cm_width = class_maps.shape
        _, n_nodes, max_peaks, _ = peaks.shape
        device = peaks.device

        # Initialize output tensor
        class_probs = torch.zeros(
            (batch_size, n_nodes, max_peaks, n_classes),
            device=device,
            dtype=class_maps.dtype,
        )

        # Convert peak coordinates to class map space
        # peaks are in full image space (after cms_output_stride scaling)
        peaks_cm = peaks / self.class_maps_output_stride

        # Clamp coordinates to valid range
        peaks_cm_x = peaks_cm[..., 0].clamp(0, cm_width - 1)
        peaks_cm_y = peaks_cm[..., 1].clamp(0, cm_height - 1)

        # Use grid_sample for bilinear interpolation
        # Normalize coordinates to [-1, 1] for grid_sample
        grid_x = (peaks_cm_x / (cm_width - 1)) * 2 - 1
        grid_y = (peaks_cm_y / (cm_height - 1)) * 2 - 1

        # Reshape for grid_sample: (batch, n_nodes * max_peaks, 1, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid_flat = grid.reshape(batch_size, n_nodes * max_peaks, 1, 2)

        # Sample class maps: (batch, n_classes, n_nodes * max_peaks, 1)
        sampled = F.grid_sample(
            class_maps,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Reshape to (batch, n_nodes, max_peaks, n_classes)
        sampled = sampled.squeeze(-1)  # (batch, n_classes, n_nodes * max_peaks)
        sampled = sampled.permute(0, 2, 1)  # (batch, n_nodes * max_peaks, n_classes)
        sampled = sampled.reshape(batch_size, n_nodes, max_peaks, n_classes)

        # Apply softmax to get probabilities (optional - depends on training)
        # For now, return raw values as the grouping function expects logits
        class_probs = sampled

        # Mask invalid peaks
        class_probs = class_probs * peak_mask.unsqueeze(-1).float()

        return class_probs
