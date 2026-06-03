"""Inference utilities for bottom-up instance segmentation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L


def find_center_peaks(
    center_heatmap: torch.Tensor, threshold: float = 0.2, kernel_size: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find instance-center peaks robustly (plateau-aware).

    Strict-greater non-maximum suppression (``find_local_peaks_rough``) drops a
    peak whose maximum spans 2+ tied pixels — which happens routinely for the
    synthetic center heatmap when a centroid lands exactly between grid points
    (the ``+stride/2`` convention). This detector instead keeps every pixel
    equal to its neighborhood max (``>=``) and collapses each connected
    plateau of tied maxima to a single representative (its argmax pixel), so a
    flat-topped peak yields exactly one center.

    Args:
        center_heatmap: ``(1, 1, H, W)`` instance-center heatmap.
        threshold: Minimum peak value.
        kernel_size: Odd window size for the max-pool NMS. Larger values suppress
            nearby duplicate centers (a lever against over-segmentation from a
            single instance producing two close center peaks). Default ``3``.

    Returns:
        ``(peaks, vals)`` where ``peaks`` is ``(N, 2)`` float ``(x, y)`` in grid
        (output-stride) coordinates and ``vals`` is ``(N,)``.
    """
    from scipy.ndimage import label as cc_label

    hm = center_heatmap[0, 0]
    pad = int(kernel_size) // 2
    pooled = F.max_pool2d(
        hm[None, None], kernel_size=int(kernel_size), stride=1, padding=pad
    )[0, 0]
    cand = (hm >= pooled) & (hm > threshold)  # local maxima incl. plateaus
    if not bool(cand.any()):
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,))

    hm_np = hm.detach().cpu().numpy()
    labels, n = cc_label(cand.detach().cpu().numpy())
    peaks: List[Tuple[float, float]] = []
    vals: List[float] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        comp_vals = hm_np[ys, xs]
        k = int(comp_vals.argmax())
        peaks.append((float(xs[k]), float(ys[k])))  # (x, y) grid coords
        vals.append(float(comp_vals[k]))
    return (
        torch.tensor(peaks, dtype=torch.float32),
        torch.tensor(vals, dtype=torch.float32),
    )


def group_instances_from_offsets(
    foreground: torch.Tensor,
    center_heatmap: torch.Tensor,
    offsets: torch.Tensor,
    fg_threshold: float = 0.5,
    peak_threshold: float = 0.2,
    output_stride: int = 2,
    max_instances: Optional[int] = None,
    center_nms_kernel: int = 3,
    mask_cleanup: bool = False,
) -> List[Dict]:
    """Group foreground pixels into instances using center-offset predictions.

    Args:
        foreground: Foreground probability map. Shape: (1, 1, H, W).
        center_heatmap: Center heatmap. Shape: (1, 1, H, W).
        offsets: Offset field (dx, dy). Shape: (1, 2, H, W).
        fg_threshold: Threshold for foreground binarization.
        peak_threshold: Minimum peak value for center detection.
        output_stride: Stride of the output maps relative to the input image.
        max_instances: Optional cap on the number of instances per frame. When
            more centers than this are detected, only the ``max_instances``
            highest-scoring (peak-value) centers are kept before grouping.
            ``None`` keeps all detected centers.
        center_nms_kernel: Odd window size for center-peak NMS (passed to
            :func:`find_center_peaks`). Larger merges nearby duplicate centers.
            Default ``3`` (no change vs. the original behavior).
        mask_cleanup: When ``True``, post-process each per-instance mask by
            keeping only its largest connected component and filling interior
            holes (suppresses speckle/fragments). Default ``False``.

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

    # 2. Find centers via plateau-robust local peak finding
    peaks, peak_vals = find_center_peaks(
        center_heatmap, threshold=peak_threshold, kernel_size=center_nms_kernel
    )

    if len(peaks) == 0:
        return []

    # Cap to the top-``max_instances`` centers by peak value, mirroring the
    # confidence-truncation other bottom-up layers apply (BottomUpLayer /
    # CentroidLayer). Below the cap, all centers are kept.
    if max_instances is not None and len(peaks) > int(max_instances):
        peak_vals, keep = torch.topk(peak_vals, int(max_instances))
        peaks = peaks[keep]

    # peaks are in (x, y) format at output stride resolution
    centers = peaks.float()  # (N, 2) in output stride grid coords

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

        mask_np = instance_mask.numpy()
        if mask_cleanup:
            mask_np = _clean_instance_mask(mask_np)
            if not mask_np.any():
                continue

        instances.append(
            {
                "mask": mask_np,
                "center": (center_x[i].item(), center_y[i].item()),
                "score": peak_vals[i].item() if i < len(peak_vals) else 0.0,
            }
        )

    return instances


def _clean_instance_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill interior holes.

    Suppresses speckle and fragments left by the per-pixel offset grouping.
    Operates at output-stride resolution; a no-op for an already-clean mask.
    """
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label as cc_label

    labels, n = cc_label(mask)
    if n > 1:
        # Keep the largest component (component 0 is background).
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    return binary_fill_holes(mask)


class BottomUpSegmentationInferenceModel(L.LightningModule):
    """Inference model for bottom-up instance segmentation.

    Wraps a trained model and post-processing into a single forward pass.
    Input images should already be padded to stride before being passed to this
    model (handled by the predictor's ``_run_inference_on_batch``).

    Attributes:
        torch_model: Callable model that returns head output dict.
        fg_threshold: Threshold for foreground binarization.
        peak_threshold: Minimum peak value for center detection.
        output_stride: Stride of the model output maps.
        min_mask_area: Minimum mask area (original-image pixels) carried through
            to ``SegmentationLayer`` to drop tiny spurious masks. ``0`` disables
            it. Not applied here (``forward`` returns output-stride masks for
            training visualization); see ``SegmentationLayer.postprocess``.
        max_instances: Optional cap on instances per frame (highest-scoring
            centers kept). Carried through to ``SegmentationLayer``; ``None``
            keeps all detected centers.
        center_nms_kernel: Odd window size for center-peak NMS. Default ``3``.
        mask_cleanup: Keep-largest-CC + hole-fill per mask. Default ``False``.
    """

    def __init__(
        self,
        torch_model,
        fg_threshold: float = 0.5,
        peak_threshold: float = 0.2,
        output_stride: int = 2,
        input_scale: float = 1.0,
        min_mask_area: int = 0,
        max_instances: Optional[int] = None,
        center_nms_kernel: int = 3,
        mask_cleanup: bool = False,
    ):
        """Initialize the inference model."""
        super().__init__()
        self.torch_model = torch_model
        self.fg_threshold = fg_threshold
        self.peak_threshold = peak_threshold
        self.output_stride = output_stride
        self.input_scale = input_scale
        self.min_mask_area = int(min_mask_area)
        self.max_instances = max_instances
        self.center_nms_kernel = int(center_nms_kernel)
        self.mask_cleanup = bool(mask_cleanup)

    def forward(self, batch: Dict) -> List[List[Dict]]:
        """Run inference on a batch of images.

        Args:
            batch: Dict with "image" key. Shape: (B, C, H, W). Images should
                already be padded to the model's max stride.

        Returns:
            List of instance lists (one per batch element). Each instance is a dict
            with "mask", "center", and "score" keys.
        """
        images = batch["image"]
        if images.dim() == 5:
            images = images.squeeze(1)

        images = images.to(self.device)

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
                max_instances=self.max_instances,
                center_nms_kernel=self.center_nms_kernel,
                mask_cleanup=self.mask_cleanup,
            )
            batch_results.append(instances)

        return batch_results
