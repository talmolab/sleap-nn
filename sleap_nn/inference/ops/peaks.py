"""Peak finding ops for confidence maps.

Functions provided:

- :func:`morphological_dilation` — pure-PyTorch 3×3 dilation used as the
  non-maximum suppression kernel below.
- :func:`integral_regression` — sub-pixel refinement by integrating expected
  coordinates over a local patch.
- :func:`find_global_peaks_rough` / :func:`find_global_peaks` — single peak
  per (sample, channel) for centroid / single-instance confmaps.
- :func:`find_local_peaks_rough` / :func:`find_local_peaks` — multi-peak
  finding for bottom-up confmaps.

PR 1 is a pure relocation — every function below preserves the exact
signature and behavior of its old home in ``sleap_nn/inference/peak_finding.py``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sleap_nn.inference.ops.crops import crop_bboxes, make_centered_bboxes


def morphological_dilation(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply morphological dilation using max pooling.

    Pure-PyTorch replacement for ``kornia.morphology.dilation``. For
    non-maximum suppression, this computes the max of 8 neighbors (excluding
    the center pixel — the kernel is zero at center, one elsewhere).

    Args:
        image: Input tensor of shape ``(B, 1, H, W)``.
        kernel: Dilation kernel (3×3 expected for NMS).

    Returns:
        Dilated tensor with the same shape as the input.
    """
    padded = F.pad(image, (1, 1, 1, 1), mode="constant", value=float("-inf"))

    # Shape after the two unfolds: (B, 1, H, W, 3, 3)
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
    b, c, h, w, _, _ = patches.shape
    patches = patches.reshape(b, c, h, w, -1)

    kernel_flat = kernel.reshape(-1).to(patches.device)
    kernel_mask = kernel_flat > 0

    patches_masked = patches.clone()
    patches_masked[..., ~kernel_mask] = float("-inf")

    return patches_masked.max(dim=-1)[0]


def integral_regression(
    cms: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute regression by integrating over the confidence maps on a grid.

    Args:
        cms: Confidence maps with shape ``(samples, channels, height, width)``.
        xv: ``float32`` x-grid vector of coordinates to sample.
        yv: ``float32`` y-grid vector of coordinates to sample.

    Returns:
        ``(x_hat, y_hat)`` regressed coordinates per channel, each of shape
        ``(samples, channels)``.
    """
    z = torch.sum(cms, dim=[2, 3]).to(cms.device)
    xv = xv.to(cms.device)
    yv = yv.to(cms.device)

    x_hat = torch.sum(xv.view(1, 1, 1, -1) * cms, dim=[2, 3]) / z
    y_hat = torch.sum(yv.view(1, 1, -1, 1) * cms, dim=[2, 3]) / z
    return x_hat, y_hat


def find_global_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the global maximum for each sample and channel.

    Args:
        cms: ``(samples, channels, height, width)``.
        threshold: Peaks below this are replaced with NaN.

    Returns:
        ``(peak_points, peak_vals)`` where ``peak_points`` is
        ``(samples, channels, 2)`` in ``(x, y)`` order and ``peak_vals`` is
        ``(samples, channels)``.
    """
    max_values, _max_indices_y = torch.max(cms, dim=2, keepdim=True)
    max_values, max_indices_x = torch.max(max_values, dim=3, keepdim=True)
    max_indices_x = max_indices_x.squeeze(dim=(2, 3))

    amax_values, _amax_indices_x = torch.max(cms, dim=3, keepdim=True)
    amax_values, amax_indices_y = torch.max(amax_values, dim=2, keepdim=True)
    amax_indices_y = amax_indices_y.squeeze(dim=(2, 3))

    peak_points = torch.cat(
        [max_indices_x.unsqueeze(-1), amax_indices_y.unsqueeze(-1)], dim=-1
    ).to(torch.float32)
    max_values = max_values.squeeze(-1).squeeze(-1)

    below_threshold_mask = max_values < threshold
    peak_points[below_threshold_mask] = float("nan")
    max_values[below_threshold_mask] = float(0)
    return peak_points, max_values


def find_global_peaks(
    cms: torch.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find global peaks with optional refinement.

    Args:
        cms: ``(samples, channels, height, width)`` confidence maps.
        threshold: Peaks below this are NaN-padded.
        refinement: ``None`` returns grid-aligned peaks; ``"integral"`` runs
            sub-pixel integral refinement on a small patch around each peak.
        integral_patch_size: Side length of the refinement patch.

    Returns:
        ``(peak_points, peak_vals)`` as in :func:`find_global_peaks_rough`.
    """
    rough_peaks, peak_vals = find_global_peaks_rough(cms, threshold=threshold)

    if refinement is None or torch.isnan(rough_peaks).all():
        return rough_peaks, peak_vals
    if refinement != "integral":
        return rough_peaks, peak_vals

    crop_size = integral_patch_size

    samples = cms.size(0)
    channels = cms.size(1)
    rough_peaks = rough_peaks.view(samples * channels, 2)

    valid_idx = torch.where(~torch.isnan(rough_peaks[:, 0]))[0]
    valid_peaks = rough_peaks[valid_idx]

    bboxes = make_centered_bboxes(
        valid_peaks, box_height=crop_size, box_width=crop_size
    )

    cms = torch.reshape(cms, [samples * channels, 1, cms.size(2), cms.size(3)])
    cm_crops = crop_bboxes(cms, bboxes, valid_idx)

    gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
    dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
    offsets = torch.cat([dx_hat, dy_hat], dim=1)

    refined_peaks = rough_peaks.clone()
    refined_peaks[valid_idx] += offsets

    return refined_peaks.reshape(samples, channels, 2), peak_vals


def find_local_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find local maxima via non-maximum suppression.

    Args:
        cms: ``(samples, channels, height, width)``.
        threshold: Peaks below this are dropped.

    Returns:
        ``(peak_points, peak_vals, peak_sample_inds, peak_channel_inds)``:
        ``peak_points`` is ``(n_peaks, 2)`` in ``(x, y)`` order;
        ``peak_vals`` is ``(n_peaks,)``;
        the index tensors are ``(n_peaks,)`` ``int32``.
    """
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)

    height = cms.size(2)
    width = cms.size(3)
    channels = cms.size(1)
    flat_img = cms.reshape(-1, 1, height, width)

    max_img = morphological_dilation(flat_img, kernel.to(flat_img.device))
    max_img = max_img.reshape(-1, channels, height, width)

    argmax_and_thresh_img = (cms > max_img) & (cms > threshold)

    peak_subs = torch.stack(
        torch.where(argmax_and_thresh_img.permute(0, 2, 3, 1)), axis=-1
    )
    peak_vals = cms[peak_subs[:, 0], peak_subs[:, 3], peak_subs[:, 1], peak_subs[:, 2]]
    peak_points = peak_subs[:, [2, 1]].to(torch.float32)
    peak_sample_inds = peak_subs[:, 0].to(torch.int32)
    peak_channel_inds = peak_subs[:, 3].to(torch.int32)
    return peak_points, peak_vals, peak_sample_inds, peak_channel_inds


def find_local_peaks(
    cms: torch.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find local peaks with optional refinement.

    Same return shape as :func:`find_local_peaks_rough`. ``refinement``
    accepts ``None`` (no refinement) or ``"integral"``.
    """
    rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds = (
        find_local_peaks_rough(cms, threshold=threshold)
    )

    if rough_peaks.size(0) == 0 or refinement is None:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds
    if refinement != "integral":
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    crop_size = integral_patch_size

    bboxes = make_centered_bboxes(
        rough_peaks, box_height=crop_size, box_width=crop_size
    )

    samples = cms.size(0)
    channels = cms.size(1)
    cms = torch.reshape(cms, [samples * channels, 1, cms.size(2), cms.size(3)])
    box_sample_inds = (peak_sample_inds * channels) + peak_channel_inds

    cm_crops = crop_bboxes(cms, bboxes, sample_inds=box_sample_inds)

    gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
    dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
    offsets = torch.cat([dx_hat, dy_hat], dim=1)

    refined_peaks = rough_peaks + offsets
    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds
