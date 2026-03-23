"""Generate ground truth tensors for instance segmentation from SegmentationMask objects."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def generate_foreground_mask(
    masks: List[np.ndarray],
    img_hw: Tuple[int, int],
    output_stride: int = 2,
) -> torch.Tensor:
    """Generate binary foreground mask as union of all instance masks.

    Args:
        masks: List of 2D boolean arrays (H, W), one per instance.
        img_hw: Original image size as (height, width).
        output_stride: Stride for downsampling the output mask.

    Returns:
        Tensor of shape (1, 1, H/s, W/s) with float32 values in [0, 1].
    """
    height, width = img_hw
    out_h = height // output_stride
    out_w = width // output_stride

    if len(masks) == 0:
        return torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

    # Union of all masks at original resolution
    union = np.zeros((height, width), dtype=bool)
    for m in masks:
        # Handle masks that may be different sizes than image
        mh, mw = m.shape
        h_end = min(mh, height)
        w_end = min(mw, width)
        union[:h_end, :w_end] |= m[:h_end, :w_end]

    # Convert to tensor and downsample via area interpolation
    fg = torch.from_numpy(union.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if output_stride > 1:
        fg = F.interpolate(fg, size=(out_h, out_w), mode="area")
    # Threshold back to binary-ish (area interpolation produces soft values)
    fg = (fg > 0.5).float()

    return fg  # (1, 1, H/s, W/s)


def generate_center_heatmap(
    masks: List[np.ndarray],
    img_hw: Tuple[int, int],
    output_stride: int = 2,
    sigma: float = 10.0,
    centers: Optional[List[Tuple[float, float]]] = None,
) -> torch.Tensor:
    """Generate Gaussian heatmap at each instance mask centroid.

    Args:
        masks: List of 2D boolean arrays (H, W), one per instance.
        img_hw: Original image size as (height, width).
        output_stride: Stride for downsampling the output.
        sigma: Standard deviation of the Gaussian in pixels (at original resolution).
        centers: Pre-computed list of (x, y) centroid coordinates. If None, centroids
            will be computed from masks via ``_compute_mask_centroids``.

    Returns:
        Tensor of shape (1, 1, H/s, W/s) with float32 values.
    """
    height, width = img_hw
    out_h = height // output_stride
    out_w = width // output_stride
    xv = torch.arange(out_w, dtype=torch.float32) * output_stride + output_stride / 2.0
    yv = torch.arange(out_h, dtype=torch.float32) * output_stride + output_stride / 2.0

    if len(masks) == 0:
        return torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

    # Compute centroids of each mask if not provided
    if centers is None:
        centers = _compute_mask_centroids(masks)  # (N, 2) in (x, y) pixel coords

    # Build heatmap as max of Gaussians
    heatmap = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)
    xv_grid = xv.reshape(1, 1, 1, -1)  # (1, 1, 1, W/s)
    yv_grid = yv.reshape(1, 1, -1, 1)  # (1, 1, H/s, 1)
    scaled_sigma = sigma * output_stride

    for cx, cy in centers:
        g = torch.exp(
            -((xv_grid - cx) ** 2 + (yv_grid - cy) ** 2) / (2 * scaled_sigma**2)
        )
        heatmap = torch.maximum(heatmap, g)

    return heatmap  # (1, 1, H/s, W/s)


def generate_center_offsets(
    masks: List[np.ndarray],
    img_hw: Tuple[int, int],
    output_stride: int = 2,
    centers: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate per-pixel offset vectors pointing to each pixel's instance center.

    Args:
        masks: List of 2D boolean arrays (H, W), one per instance.
        img_hw: Original image size as (height, width).
        output_stride: Stride for downsampling the output.
        centers: Pre-computed list of (x, y) centroid coordinates. If None, centroids
            will be computed from masks via ``_compute_mask_centroids``.

    Returns:
        Tuple of:
            offsets: Tensor of shape (1, 2, H/s, W/s) with (dx, dy) offset vectors.
                Only defined on foreground pixels; background pixels are 0.
            weight_mask: Tensor of shape (1, 1, H/s, W/s) binary mask indicating
                where offset loss should be computed (foreground pixels).
    """
    height, width = img_hw
    out_h = height // output_stride
    out_w = width // output_stride

    offsets = torch.zeros((1, 2, out_h, out_w), dtype=torch.float32)
    weight_mask = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

    if len(masks) == 0:
        return offsets, weight_mask

    if centers is None:
        centers = _compute_mask_centroids(masks)

    # Sort by area descending so smaller instances overwrite larger ones in overlaps
    areas = [m.sum() for m in masks]
    sorted_indices = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)

    # Create coordinate grids at output stride resolution
    # Grid values are in original pixel coordinates
    yy = torch.arange(out_h, dtype=torch.float32) * output_stride + output_stride / 2.0
    xx = torch.arange(out_w, dtype=torch.float32) * output_stride + output_stride / 2.0
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing="xy")  # both (out_h, out_w)

    for idx in sorted_indices:
        m = masks[idx]
        # Downsample mask
        m_tensor = (
            torch.from_numpy(m[:height, :width].astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if output_stride > 1:
            m_ds = F.interpolate(m_tensor, size=(out_h, out_w), mode="area")
        else:
            m_ds = m_tensor
        m_binary = m_ds[0, 0] > 0.5  # (out_h, out_w)

        cx, cy = centers[idx]

        # Offset = center - pixel_coord (so pixel + offset = center)
        dx = cx - grid_x  # (out_h, out_w)
        dy = cy - grid_y

        # Only set offsets for this instance's foreground pixels
        offsets[0, 0][m_binary] = dx[m_binary]
        offsets[0, 1][m_binary] = dy[m_binary]
        weight_mask[0, 0][m_binary] = 1.0

    return offsets, weight_mask


def _compute_mask_centroids(masks: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Compute the centroid (center of mass) of each binary mask.

    Args:
        masks: List of 2D boolean arrays.

    Returns:
        List of (x, y) centroid coordinates in pixel space.
    """
    centers = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            # Empty mask — use image center as fallback
            cy, cx = m.shape[0] / 2.0, m.shape[1] / 2.0
        else:
            cx = float(xs.mean())
            cy = float(ys.mean())
        centers.append((cx, cy))
    return centers
