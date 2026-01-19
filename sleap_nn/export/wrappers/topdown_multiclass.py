"""ONNX wrapper for top-down multiclass (supervised ID) models."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class TopDownMultiClassONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for top-down multiclass (supervised ID) models.

    This wrapper handles models that output both confidence maps for keypoint
    detection and class logits for identity classification. It runs on instance
    crops (centered around detected centroids).

    Expects input images as uint8 tensors in [0, 255].

    Attributes:
        model: The underlying PyTorch model (centered instance + class vectors heads).
        output_stride: Output stride of the confmap head.
        input_scale: Scale factor applied to input images before inference.
        n_classes: Number of identity classes.
    """

    def __init__(
        self,
        model: nn.Module,
        output_stride: int = 2,
        input_scale: float = 1.0,
        n_classes: int = 2,
    ):
        """Initialize the wrapper.

        Args:
            model: The underlying PyTorch model.
            output_stride: Output stride of the confidence maps.
            input_scale: Scale factor for input images.
            n_classes: Number of identity classes (e.g., 2 for male/female).
        """
        super().__init__(model)
        self.output_stride = output_stride
        self.input_scale = input_scale
        self.n_classes = n_classes

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run top-down multiclass inference on crops.

        Args:
            image: Input image tensor of shape (batch, channels, height, width).
                   Expected to be uint8 in [0, 255].

        Returns:
            Dictionary with keys:
                - "peaks": Predicted peak coordinates (batch, n_nodes, 2) in (x, y).
                - "peak_vals": Peak confidence values (batch, n_nodes).
                - "class_logits": Raw class logits (batch, n_classes).

            The class assignment is done on CPU using Hungarian matching
            via `get_class_inds_from_vectors()`.
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

        # Forward pass
        out = self.model(image)

        # Extract outputs
        confmaps = self._extract_tensor(out, ["centered", "instance", "confmap"])
        class_logits = self._extract_tensor(out, ["class", "vector"])

        # Find global peaks (one per node)
        peaks, peak_vals = self._find_global_peaks(confmaps)

        # Scale peaks back to input coordinates
        peaks = peaks * (self.output_stride / self.input_scale)

        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "class_logits": class_logits,
        }


class TopDownMultiClassCombinedONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for combined centroid + multiclass instance models.

    This wrapper combines a centroid detection model with a centered instance
    multiclass model. It performs:
    1. Centroid detection on full images
    2. Cropping around each centroid using vectorized grid_sample
    3. Instance keypoint detection + identity classification on each crop

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        centroid_model: nn.Module,
        instance_model: nn.Module,
        max_instances: int = 20,
        crop_size: tuple = (192, 192),
        centroid_output_stride: int = 4,
        instance_output_stride: int = 2,
        centroid_input_scale: float = 1.0,
        instance_input_scale: float = 1.0,
        n_nodes: int = 13,
        n_classes: int = 2,
    ):
        """Initialize the combined wrapper.

        Args:
            centroid_model: Model for centroid detection.
            instance_model: Model for instance keypoints + class prediction.
            max_instances: Maximum number of instances to detect.
            crop_size: Size of crops around centroids (height, width).
            centroid_output_stride: Output stride of centroid model.
            instance_output_stride: Output stride of instance model.
            centroid_input_scale: Input scale for centroid model.
            instance_input_scale: Input scale for instance model.
            n_nodes: Number of keypoint nodes per instance.
            n_classes: Number of identity classes.
        """
        super().__init__(centroid_model)  # Primary model is centroid
        self.instance_model = instance_model
        self.max_instances = max_instances
        self.crop_size = crop_size
        self.centroid_output_stride = centroid_output_stride
        self.instance_output_stride = instance_output_stride
        self.centroid_input_scale = centroid_input_scale
        self.instance_input_scale = instance_input_scale
        self.n_nodes = n_nodes
        self.n_classes = n_classes

        # Pre-compute base grid for crop extraction (same as TopDownONNXWrapper)
        crop_h, crop_w = crop_size
        y_crop = torch.linspace(-1, 1, crop_h, dtype=torch.float32)
        x_crop = torch.linspace(-1, 1, crop_w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_crop, x_crop, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("base_grid", base_grid, persistent=False)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run combined top-down multiclass inference.

        Args:
            image: Input image tensor of shape (batch, channels, height, width).
                   Expected to be uint8 in [0, 255].

        Returns:
            Dictionary with keys:
                - "centroids": Detected centroids (batch, max_instances, 2).
                - "centroid_vals": Centroid confidence values (batch, max_instances).
                - "peaks": Instance peaks (batch, max_instances, n_nodes, 2).
                - "peak_vals": Peak values (batch, max_instances, n_nodes).
                - "class_logits": Class logits per instance (batch, max_instances, n_classes).
                - "instance_valid": Validity mask (batch, max_instances).
        """
        # Normalize input
        image = self._normalize_uint8(image)
        batch_size, channels, height, width = image.shape

        # Apply centroid input scaling
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

        # Centroid detection
        centroid_out = self.model(scaled_image)
        centroid_cms = self._extract_tensor(centroid_out, ["centroid", "confmap"])
        centroids, centroid_vals, instance_valid = self._find_topk_peaks(
            centroid_cms, self.max_instances
        )
        centroids = centroids * (
            self.centroid_output_stride / self.centroid_input_scale
        )

        # Extract crops using vectorized grid_sample (same as TopDownONNXWrapper)
        crops = self._extract_crops(image, centroids)
        crops_flat = crops.reshape(
            batch_size * self.max_instances,
            channels,
            self.crop_size[0],
            self.crop_size[1],
        )

        # Apply instance input scaling if needed
        if self.instance_input_scale != 1.0:
            scaled_h = int(self.crop_size[0] * self.instance_input_scale)
            scaled_w = int(self.crop_size[1] * self.instance_input_scale)
            crops_flat = F.interpolate(
                crops_flat,
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
            )

        # Instance model forward (batch all crops)
        instance_out = self.instance_model(crops_flat)
        instance_cms = self._extract_tensor(
            instance_out, ["centered", "instance", "confmap"]
        )
        instance_class = self._extract_tensor(instance_out, ["class", "vector"])

        # Find peaks in all crops
        crop_peaks, crop_peak_vals = self._find_global_peaks(instance_cms)
        crop_peaks = crop_peaks * (
            self.instance_output_stride / self.instance_input_scale
        )

        # Reshape to batch x instances x nodes x 2
        crop_peaks = crop_peaks.reshape(batch_size, self.max_instances, self.n_nodes, 2)
        peak_vals = crop_peak_vals.reshape(batch_size, self.max_instances, self.n_nodes)

        # Reshape class logits
        class_logits = instance_class.reshape(
            batch_size, self.max_instances, self.n_classes
        )

        # Transform peaks from crop coordinates to full image coordinates
        crop_offset = centroids.unsqueeze(2) - image.new_tensor(
            [self.crop_size[1] / 2.0, self.crop_size[0] / 2.0]
        )
        peaks = crop_peaks + crop_offset

        # Zero out invalid instances
        invalid_mask = ~instance_valid
        centroids = centroids.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
        centroid_vals = centroid_vals.masked_fill(invalid_mask, 0.0)
        peaks = peaks.masked_fill(invalid_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        peak_vals = peak_vals.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
        class_logits = class_logits.masked_fill(invalid_mask.unsqueeze(-1), 0.0)

        return {
            "centroids": centroids,
            "centroid_vals": centroid_vals,
            "peaks": peaks,
            "peak_vals": peak_vals,
            "class_logits": class_logits,
            "instance_valid": instance_valid,
        }

    def _extract_crops(
        self,
        image: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract crops around centroids using grid_sample.

        This is the same vectorized implementation as TopDownONNXWrapper.
        """
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
