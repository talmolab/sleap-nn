"""Top-down ONNX wrapper."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class TopDownONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for top-down (centroid + centered-instance) inference.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        centroid_model: nn.Module,
        instance_model: nn.Module,
        max_instances: int = 20,
        crop_size: Tuple[int, int] = (192, 192),
        centroid_output_stride: int = 2,
        instance_output_stride: int = 4,
        centroid_input_scale: float = 1.0,
        instance_input_scale: float = 1.0,
        n_nodes: int = 1,
    ) -> None:
        """Initialize top-down ONNX wrapper.

        Args:
            centroid_model: Centroid detection model.
            instance_model: Instance pose estimation model.
            max_instances: Maximum number of instances to detect.
            crop_size: Size of instance crops (height, width).
            centroid_output_stride: Centroid model output stride.
            instance_output_stride: Instance model output stride.
            centroid_input_scale: Centroid input scaling factor.
            instance_input_scale: Instance input scaling factor.
            n_nodes: Number of skeleton nodes.
        """
        super().__init__(centroid_model)
        self.centroid_model = centroid_model
        self.instance_model = instance_model
        self.max_instances = max_instances
        self.crop_size = crop_size
        self.centroid_output_stride = centroid_output_stride
        self.instance_output_stride = instance_output_stride
        self.centroid_input_scale = centroid_input_scale
        self.instance_input_scale = instance_input_scale
        self.n_nodes = n_nodes

        crop_h, crop_w = crop_size
        y_crop = torch.linspace(-1, 1, crop_h, dtype=torch.float32)
        x_crop = torch.linspace(-1, 1, crop_w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_crop, x_crop, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("base_grid", base_grid, persistent=False)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run top-down inference and return fixed-size outputs."""
        image = self._normalize_uint8(image)
        batch_size, channels, height, width = image.shape

        scaled_image = image
        if self.centroid_input_scale != 1.0:
            scaled_h = int(height * self.centroid_input_scale)
            scaled_w = int(width * self.centroid_input_scale)
            scaled_image = F.interpolate(
                scaled_image,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )

        centroid_out = self.centroid_model(scaled_image)
        centroid_cms = self._extract_tensor(centroid_out, ["centroid", "confmap"])

        centroids, centroid_vals, instance_valid = self._find_topk_peaks(
            centroid_cms, self.max_instances
        )
        centroids = centroids * (
            self.centroid_output_stride / self.centroid_input_scale
        )

        crops = self._extract_crops(image, centroids)
        crops_flat = crops.reshape(
            batch_size * self.max_instances,
            channels,
            self.crop_size[0],
            self.crop_size[1],
        )

        if self.instance_input_scale != 1.0:
            scaled_h = int(self.crop_size[0] * self.instance_input_scale)
            scaled_w = int(self.crop_size[1] * self.instance_input_scale)
            crops_flat = F.interpolate(
                crops_flat,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )

        instance_out = self.instance_model(crops_flat)
        instance_cms = self._extract_tensor(
            instance_out, ["centered", "instance", "confmap"]
        )

        crop_peaks, crop_peak_vals = self._find_global_peaks(instance_cms)
        crop_peaks = crop_peaks * (
            self.instance_output_stride / self.instance_input_scale
        )

        crop_peaks = crop_peaks.reshape(batch_size, self.max_instances, self.n_nodes, 2)
        peak_vals = crop_peak_vals.reshape(batch_size, self.max_instances, self.n_nodes)

        crop_offset = centroids.unsqueeze(2) - image.new_tensor(
            [self.crop_size[1] / 2.0, self.crop_size[0] / 2.0]
        )
        peaks = crop_peaks + crop_offset

        invalid_mask = ~instance_valid
        centroids = centroids.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
        centroid_vals = centroid_vals.masked_fill(invalid_mask, 0.0)
        peaks = peaks.masked_fill(invalid_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        peak_vals = peak_vals.masked_fill(invalid_mask.unsqueeze(-1), 0.0)

        return {
            "centroids": centroids,
            "centroid_vals": centroid_vals,
            "peaks": peaks,
            "peak_vals": peak_vals,
            "instance_valid": instance_valid,
        }

    def _extract_crops(
        self,
        image: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract crops around centroids using grid_sample."""
        batch_size, channels, height, width = image.shape
        crop_h, crop_w = self.crop_size
        n_instances = centroids.shape[1]

        scale_x = crop_w / width
        scale_y = crop_h / height
        scale = image.new_tensor([scale_x, scale_y])
        base_grid = self.base_grid.to(device=image.device, dtype=image.dtype)
        scaled_grid = base_grid * scale

        scaled_grid = scaled_grid.unsqueeze(0).unsqueeze(0)
        scaled_grid = scaled_grid.expand(batch_size, n_instances, -1, -1, -1)

        norm_centroids = torch.zeros_like(centroids)
        norm_centroids[..., 0] = (centroids[..., 0] / (width - 1)) * 2 - 1
        norm_centroids[..., 1] = (centroids[..., 1] / (height - 1)) * 2 - 1
        offset = norm_centroids.unsqueeze(2).unsqueeze(2)

        sample_grid = scaled_grid + offset

        image_expanded = image.unsqueeze(1).expand(-1, n_instances, -1, -1, -1)
        image_flat = image_expanded.reshape(
            batch_size * n_instances, channels, height, width
        )
        grid_flat = sample_grid.reshape(batch_size * n_instances, crop_h, crop_w, 2)

        crops_flat = F.grid_sample(
            image_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        crops = crops_flat.reshape(batch_size, n_instances, channels, crop_h, crop_w)
        return crops
