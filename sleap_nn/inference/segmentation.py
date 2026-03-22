"""Inference utilities for bottom-up instance segmentation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import lightning as L

from sleap_nn.data.normalization import normalize_on_gpu
from sleap_nn.data.resizing import apply_pad_to_stride
from sleap_nn.inference.peak_finding import find_local_peaks_rough


def group_instances_from_offsets(
    foreground: torch.Tensor,
    center_heatmap: torch.Tensor,
    offsets: torch.Tensor,
    fg_threshold: float = 0.5,
    peak_threshold: float = 0.2,
    output_stride: int = 2,
) -> List[Dict]:
    """Group foreground pixels into instances using center-offset predictions.

    Args:
        foreground: Foreground probability map. Shape: (1, 1, H, W).
        center_heatmap: Center heatmap. Shape: (1, 1, H, W).
        offsets: Offset field (dx, dy). Shape: (1, 2, H, W).
        fg_threshold: Threshold for foreground binarization.
        peak_threshold: Minimum peak value for center detection.
        nms_kernel_size: Kernel size for non-maximum suppression.
        output_stride: Stride of the output maps relative to the input image.

    Returns:
        List of dicts, each with:
            - "mask": (H, W) boolean numpy array (at output stride resolution)
            - "center": (x, y) tuple in original pixel coordinates
            - "score": float confidence score (peak value)
    """
    h, w = foreground.shape[-2:]

    # 1. Threshold foreground
    fg_binary = foreground[0, 0] > fg_threshold  # (H, W)

    if fg_binary.sum() == 0:
        return []

    # 2. Find centers via local peak finding
    peaks, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks_rough(
        center_heatmap,
        threshold=peak_threshold,
    )

    if len(peaks) == 0:
        return []

    # peaks are in (x, y) format at output stride resolution
    # Convert to original pixel coords
    centers = peaks.float()  # (N, 2) in output stride pixel coords

    # 3. For each foreground pixel, compute predicted center
    fg_coords = torch.nonzero(
        fg_binary, as_tuple=False
    )  # (M, 2) as (row, col) = (y, x)
    fg_y = fg_coords[:, 0]
    fg_x = fg_coords[:, 1]

    # Get offsets at foreground pixels
    dx = offsets[0, 0, fg_y, fg_x]  # (M,)
    dy = offsets[0, 1, fg_y, fg_x]  # (M,)

    # Pixel coordinates in original resolution
    pixel_x = fg_x.float() * output_stride + output_stride / 2.0
    pixel_y = fg_y.float() * output_stride + output_stride / 2.0

    # Predicted centers for each foreground pixel
    pred_center_x = pixel_x + dx  # (M,)
    pred_center_y = pixel_y + dy  # (M,)

    # 4. Assign each foreground pixel to nearest detected center
    # centers are already in original pixel coordinates (from peak finding * output_stride)
    center_x = centers[:, 0] * output_stride + output_stride / 2.0  # (N,)
    center_y = centers[:, 1] * output_stride + output_stride / 2.0  # (N,)

    # Compute distances: (M, N)
    dist_x = pred_center_x.unsqueeze(1) - center_x.unsqueeze(0)
    dist_y = pred_center_y.unsqueeze(1) - center_y.unsqueeze(0)
    dists = dist_x**2 + dist_y**2

    assignments = dists.argmin(dim=1)  # (M,) index into centers

    # 5. Build per-instance masks
    instances = []
    for i in range(len(centers)):
        member_mask = assignments == i
        if member_mask.sum() == 0:
            continue

        instance_mask = torch.zeros((h, w), dtype=torch.bool)
        instance_mask[fg_y[member_mask], fg_x[member_mask]] = True

        instances.append(
            {
                "mask": instance_mask.numpy(),
                "center": (center_x[i].item(), center_y[i].item()),
                "score": peak_vals[i].item() if i < len(peak_vals) else 0.0,
            }
        )

    return instances


class BottomUpSegmentationInferenceModel(L.LightningModule):
    """Inference model for bottom-up instance segmentation.

    Wraps a trained model and post-processing into a single forward pass.

    Attributes:
        torch_model: Callable model that returns head output dict.
        fg_threshold: Threshold for foreground binarization.
        peak_threshold: Minimum peak value for center detection.
        output_stride: Stride of the model output maps.
        input_scale: Factor to resize input images by before inference.
        max_stride: Maximum stride for input padding.
    """

    def __init__(
        self,
        torch_model,
        fg_threshold: float = 0.5,
        peak_threshold: float = 0.2,
        output_stride: int = 2,
        input_scale: float = 1.0,
        max_stride: int = 16,
    ):
        """Initialize the inference model."""
        super().__init__()
        self.torch_model = torch_model
        self.fg_threshold = fg_threshold
        self.peak_threshold = peak_threshold
        self.output_stride = output_stride
        self.input_scale = input_scale
        self.max_stride = max_stride

    def forward(self, batch: Dict) -> List[List[Dict]]:
        """Run inference on a batch of images.

        Args:
            batch: Dict with "image" key. Shape: (B, 1, C, H, W) or (B, C, H, W).

        Returns:
            List of instance lists (one per batch element). Each instance is a dict
            with "mask", "center", and "score" keys.
        """
        images = batch["image"]
        if images.dim() == 5:
            images = images.squeeze(1)

        images = images.to(self.device)

        if self.max_stride > 1:
            images = apply_pad_to_stride(images, self.max_stride)

        output = self.torch_model(images.unsqueeze(1))

        foreground = output["SegmentationHead"]
        center_heatmap = output["InstanceCenterHead"]
        offsets = output["CenterOffsetHead"]

        batch_results = []
        for b in range(foreground.shape[0]):
            instances = group_instances_from_offsets(
                foreground=foreground[b : b + 1],
                center_heatmap=center_heatmap[b : b + 1],
                offsets=offsets[b : b + 1],
                fg_threshold=self.fg_threshold,
                peak_threshold=self.peak_threshold,
                output_stride=self.output_stride,
            )
            batch_results.append(instances)

        return batch_results
