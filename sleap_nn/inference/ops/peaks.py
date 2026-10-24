"""Peak finding ops for confidence maps.

Functions provided:

- :func:`morphological_dilation` — pure-PyTorch 3×3 dilation (max over the
  8 neighbors).
- :func:`morphological_dilation_split` — the same 8-neighbor max, split into
  the neighbors before and after each pixel in raster order.
- :func:`integral_regression` — sub-pixel refinement by integrating expected
  coordinates over a local patch.
- :func:`find_global_peaks_rough` / :func:`find_global_peaks` — single peak
  per (sample, channel) for centroid / single-instance confmaps.
- :func:`find_local_peaks_rough` / :func:`find_local_peaks` — multi-peak
  finding for bottom-up confmaps.
- :func:`quarter_offset_refinement` — sub-pixel refinement of global peaks
  (``refinement="local"``).

PR 1 is a pure relocation — every function below preserves the exact
signature and behavior of its old home in ``sleap_nn/inference/peak_finding.py``.
"""

import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sleap_nn.inference.ops.crops import crop_bboxes, make_centered_bboxes


def morphological_dilation(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Compute the per-pixel max over the 8-neighborhood (excluding center).

    The ``kernel`` argument is preserved for API compatibility but is
    currently ignored — the 8-neighbor pattern is hardcoded as shifted slices
    of a padded image combined with elementwise ``torch.maximum``, which
    exports to ONNX (PR 5 of #508 rewrote the original ``Tensor.unfold``
    formulation that the legacy ONNX exporter rejected).

    Args:
        image: Input tensor of shape ``(B, 1, H, W)``.
        kernel: Legacy 3×3 NMS kernel; unused. Kept so existing callers
            continue to work without modification.

    Returns:
        Same shape as ``image``; each output pixel is the max over its
        eight neighbors in the input (out-of-image neighbors are ``-inf``,
        i.e. pad-with-minimum).
    """
    del kernel  # see docstring
    return torch.maximum(*morphological_dilation_split(image))


def morphological_dilation_split(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the 8-neighbor max split into earlier and later neighbors.

    The 8 neighbors of each pixel are split by raster order (row, then column):
    the 4 that come before it (NW, N, NE, W) and the 4 that come after it
    (E, SW, S, SE). The elementwise max of the two outputs equals
    :func:`morphological_dilation`.

    Args:
        image: Input tensor of shape ``(..., H, W)``.

    Returns:
        ``(before, after)``, each the same shape as ``image``: the max over the
        4 earlier and over the 4 later neighbors (out-of-image neighbors are
        ``-inf``).
    """
    padded = F.pad(image, (1, 1, 1, 1), mode="constant", value=float("-inf"))
    return _neighbor_max_split(padded)


def _neighbor_max_split(padded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(before, after)`` neighbor maxima of the image inside a 1-pixel border.

    Pairwise ``torch.maximum`` of shifted views never materializes the shifts
    (stacking them costs ~1.5x the time and twice the memory on CUDA), and
    exports to ONNX as ``Max``.
    """
    p = padded
    before = torch.maximum(
        torch.maximum(p[..., :-2, :-2], p[..., :-2, 1:-1]),  # NW, N
        torch.maximum(p[..., :-2, 2:], p[..., 1:-1, :-2]),  # NE, W
    )
    after = torch.maximum(
        torch.maximum(p[..., 1:-1, 2:], p[..., 2:, :-2]),  # E, SW
        torch.maximum(p[..., 2:, 1:-1], p[..., 2:, 2:]),  # S, SE
    )
    return before, after


# (row, col) offsets of the 8 neighbors of a pixel (row 0 of the table) and of
# the 8 neighbors of each of those neighbors (rows 1-8), relative to the pixel.
_NEIGHBORS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
_NEIGHBOR_TABLE = torch.tensor(
    [[(r + dr, c + dc) for dr, dc in _NEIGHBORS] for r, c in [(0, 0)] + _NEIGHBORS]
)  # (9, 8, 2)


@functools.lru_cache(maxsize=16)
def _gather_layout(
    device: torch.device, channels: int, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flat-index strides and neighbor offsets into a ``(S, C, H, W)`` tensor.

    Cached so that a call costs a lookup rather than a host-to-device copy.

    Returns:
        ``(strides, offsets)``: ``strides`` ``(4,)`` maps ``(sample, row, col,
        channel)`` to a flat index; ``offsets`` ``(9, 8)`` are the flat offsets
        of ``_NEIGHBOR_TABLE``.
    """
    strides = torch.tensor([channels * height * width, width, 1, height * width])
    offsets = _NEIGHBOR_TABLE[..., 0] * width + _NEIGHBOR_TABLE[..., 1]
    return strides.to(device), offsets.to(device)


@functools.lru_cache(maxsize=16)
def _flat_top_tables(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lookup tables for :func:`_flat_tops`, indexed by an 8-bit tie pattern.

    Bit ``k`` of a pattern is set if neighbor ``k`` (in ``_NEIGHBORS`` order)
    of a candidate equals it; the flat top is the candidate plus those
    neighbors. All neighbors are after the candidate in raster order.

    Returns:
        ``(bits, fits, offsets, outside)``: ``bits`` ``(8,)`` turns tie masks
        into patterns; ``fits`` ``(256,)``: the top fits in a 2x2 block;
        ``offsets`` ``(256, 2)``: ``(x, y)`` from the candidate to the mean of
        the top's pixels; ``outside`` ``(256, 64)``: neighbor ``j`` of neighbor
        ``k`` (at ``8 * k + j``) is outside the top while ``k`` is in it.
    """
    blocks = [{(0, 1), (1, 0), (1, 1)}, {(1, -1), (1, 0)}]  # E, S, SE / SW, S
    fits, offsets, outside = [], [], []
    for pattern in range(256):
        tied = {d for k, d in enumerate(_NEIGHBORS) if pattern >> k & 1}
        top = tied | {(0, 0)}
        fits.append(any(tied <= block for block in blocks))
        offsets.append(
            [sum(c for _, c in top) / len(top), sum(r for r, _ in top) / len(top)]
        )
        outside.append(
            [
                (r, c) in tied and (r + dr, c + dc) not in top
                for r, c in _NEIGHBORS
                for dr, dc in _NEIGHBORS
            ]
        )
    return (
        (2 ** torch.arange(8)).to(device),
        torch.tensor(fits).to(device),
        torch.tensor(offsets, dtype=torch.float32).to(device),
        torch.tensor(outside).to(device),
    )


def _flat_tops(
    padded: torch.Tensor, peak_subs: torch.Tensor, peak_vals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Check which candidate peaks have a flat top that fits in a 2x2 block.

    Each candidate is ``>=`` its 8 neighbors and ``>`` the 4 before it, so its
    equal neighbors come after it and, with it, form the flat top if the top
    fits in a 2x2 block. The candidate is kept if they do fit in one, and no
    pixel of the top has a neighbor ``>=`` it outside the top.

    Args:
        padded: ``(samples, channels, height + 4, width + 4)`` confidence maps
            with a 2-pixel ``-inf`` border.
        peak_subs: ``(n_peaks, 4)`` ``(sample, row, col, channel)`` of the
            candidates, in unpadded coordinates.
        peak_vals: ``(n_peaks,)`` candidate values.

    Returns:
        ``(keep, offsets)``: ``keep`` ``(n_peaks,)`` bool; ``offsets``
        ``(n_peaks, 2)`` ``(x, y)`` from the candidate to the mean of its flat
        top's pixel coordinates (zero for a one-pixel top).
    """
    _, channels, height, width = padded.shape
    strides, offsets = _gather_layout(padded.device, channels, height, width)
    bits, fits, top_offsets, outside = _flat_top_tables(padded.device)
    center = (peak_subs * strides).sum(1) + (2 * width + 2)
    neighbors = padded.reshape(-1)[center[:, None, None] + offsets]  # (n, 9, 8)
    ge = neighbors >= peak_vals[:, None, None]
    pattern = (ge[:, 0] * bits).sum(1)  # the candidate's equal neighbors
    leaks = (ge[:, 1:].flatten(1) & outside[pattern]).any(1)
    return fits[pattern] & ~leaks, top_offsets[pattern]


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
    # Drop dims one at a time so the ONNX exporter can lower each Squeeze
    # node independently (a single ``dim=(2, 3)`` argument is not supported).
    max_indices_x = max_indices_x.squeeze(3).squeeze(2)

    amax_values, _amax_indices_x = torch.max(cms, dim=3, keepdim=True)
    amax_values, amax_indices_y = torch.max(amax_values, dim=2, keepdim=True)
    amax_indices_y = amax_indices_y.squeeze(3).squeeze(2)

    peak_points = torch.cat(
        [max_indices_x.unsqueeze(-1), amax_indices_y.unsqueeze(-1)], dim=-1
    ).to(torch.float32)
    max_values = max_values.squeeze(-1).squeeze(-1)

    # Below-threshold positions get NaN coords + zero value. We use
    # ``torch.where`` rather than boolean-mask in-place assignment so this
    # function exports to ONNX cleanly (PR 5 of #508).
    below_threshold_mask = max_values < threshold
    peak_points = torch.where(
        below_threshold_mask.unsqueeze(-1).expand_as(peak_points),
        torch.full_like(peak_points, float("nan")),
        peak_points,
    )
    max_values = torch.where(
        below_threshold_mask, torch.zeros_like(max_values), max_values
    )
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
        refinement: ``None`` (or ``"none"``) returns grid-aligned peaks;
            ``"integral"`` runs sub-pixel integral refinement on a small patch
            around each peak; ``"local"`` moves each peak a quarter pixel towards
            its higher neighbor along each axis (:func:`quarter_offset_refinement`).
        integral_patch_size: Side length of the refinement patch.

    Returns:
        ``(peak_points, peak_vals)`` as in :func:`find_global_peaks_rough`.

    Raises:
        ValueError: If ``refinement`` is not one of the names above.
    """
    refinement = _check_refinement(refinement)
    rough_peaks, peak_vals = find_global_peaks_rough(cms, threshold=threshold)

    if refinement is None or torch.isnan(rough_peaks).all():
        return rough_peaks, peak_vals
    if refinement == "local":
        return quarter_offset_refinement(cms, rough_peaks), peak_vals

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


#: Peak refinement names the peak finders accept (``None`` also disables it).
REFINEMENT_METHODS = ("integral", "local", "none")


def _check_refinement(refinement: Optional[str]) -> Optional[str]:
    """Normalize a refinement name: ``None`` / ``"none"`` -> ``None``.

    Raises:
        ValueError: On a name that is not in :data:`REFINEMENT_METHODS`. Unknown
            names used to fall through silently as no refinement.
    """
    if refinement is None or str(refinement).lower() == "none":
        return None
    if refinement not in REFINEMENT_METHODS:
        raise ValueError(
            f"Unknown peak refinement {refinement!r}; expected one of None, "
            "'none', 'integral' or 'local'."
        )
    return refinement


def _gather_neighborhood(
    cms: torch.Tensor, rough_peaks: torch.Tensor, radius: int = 1
) -> torch.Tensor:
    """Gather the ``(2r+1) x (2r+1)`` neighborhood of each global peak.

    Out-of-bounds pixels repeat the nearest border pixel (replicate padding).

    Args:
        cms: ``(samples, channels, height, width)``.
        rough_peaks: ``(samples, channels, 2)`` integer-valued ``(x, y)`` grid
            peaks. NaN rows give the patch at ``(0, 0)``; callers keep them NaN.
        radius: Neighborhood radius ``r``.

    Returns:
        ``(samples, channels, 2r+1, 2r+1)`` patches, rows along ``y``.
    """
    samples, channels, height, width = cms.shape
    peaks = torch.nan_to_num(rough_peaks, nan=0.0).to(torch.long)
    offs = torch.arange(-radius, radius + 1, device=cms.device)
    xs = (peaks[..., 0, None] + offs).clamp(0, width - 1)  # (S, C, k)
    ys = (peaks[..., 1, None] + offs).clamp(0, height - 1)  # (S, C, k)
    idx = ys[..., :, None] * width + xs[..., None, :]  # (S, C, k, k)
    k = 2 * radius + 1
    flat = cms.reshape(samples, channels, height * width)
    patches = torch.gather(flat, 2, idx.reshape(samples, channels, k * k))
    return patches.reshape(samples, channels, k, k)


def quarter_offset_refinement(
    cms: torch.Tensor, rough_peaks: torch.Tensor
) -> torch.Tensor:
    """Refine global peaks with the quarter-offset rule (``refinement="local"``).

    Each coordinate moves 0.25 confmap pixels towards its higher neighbor
    (``x += 0.25 * sign(cm[y, x+1] - cm[y, x-1])``, same for ``y``), as in
    SLEAP's ``"local"`` refinement and the Hourglass/SimpleBaseline decoders.

    Args:
        cms: ``(samples, channels, height, width)`` confidence maps.
        rough_peaks: ``(samples, channels, 2)`` grid peaks from
            :func:`find_global_peaks_rough` (NaN for missing).

    Returns:
        ``(samples, channels, 2)`` refined ``(x, y)`` peaks (NaN preserved).
    """
    p = _gather_neighborhood(cms, rough_peaks, radius=1).to(torch.float32)
    dx = torch.sign(p[..., 1, 2] - p[..., 1, 0])
    dy = torch.sign(p[..., 2, 1] - p[..., 0, 1])
    return rough_peaks + 0.25 * torch.stack([dx, dy], dim=-1)


def _quarter_offset_local_peaks(
    cms: torch.Tensor,
    peaks: torch.Tensor,
    sample_inds: torch.Tensor,
    channel_inds: torch.Tensor,
) -> torch.Tensor:
    """Quarter-offset refinement (:func:`quarter_offset_refinement`) of local peaks.

    A flat-top peak's coordinate can already be fractional (the mean of the tied
    pixels, see :func:`find_local_peaks_rough`); that axis is kept as is.

    Args:
        cms: ``(samples, channels, height, width)`` confidence maps.
        peaks: ``(n_peaks, 2)`` ``(x, y)`` peaks.
        sample_inds: ``(n_peaks,)`` sample index of each peak.
        channel_inds: ``(n_peaks,)`` channel index of each peak.

    Returns:
        ``(n_peaks, 2)`` refined ``(x, y)`` peaks.
    """
    height, width = cms.shape[-2:]
    s, c = sample_inds.long(), channel_inds.long()
    x, y = peaks[:, 0].long(), peaks[:, 1].long()

    def _at(xx, yy):
        return cms[s, c, yy.clamp(0, height - 1), xx.clamp(0, width - 1)].float()

    dx = torch.sign(_at(x + 1, y) - _at(x - 1, y))
    dy = torch.sign(_at(x, y + 1) - _at(x, y - 1))
    offsets = 0.25 * torch.stack([dx, dy], dim=-1)
    on_grid = peaks == torch.floor(peaks)
    return peaks + torch.where(on_grid, offsets, torch.zeros_like(offsets))


def find_local_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find local maxima via non-maximum suppression.

    A pixel is a peak if it is above ``threshold`` and greater than all 8 of
    its neighbors. A flat top of equal pixels (common with fp16 confidence
    maps) is one peak if it fits in a 2x2 block (2 pixels, an L of 3, or all
    4) and is higher than every pixel around it; its point is the mean of the
    pixels' coordinates, so it can be fractional. A wider flat top, or one
    next to a higher pixel, gives no peak. See #781.

    Args:
        cms: ``(samples, channels, height, width)``.
        threshold: Peaks below this are dropped.

    Returns:
        ``(peak_points, peak_vals, peak_sample_inds, peak_channel_inds)``:
        ``peak_points`` is ``(n_peaks, 2)`` in ``(x, y)`` order;
        ``peak_vals`` is ``(n_peaks,)``;
        the index tensors are ``(n_peaks,)`` ``int32``.
    """
    padded = F.pad(cms, (2, 2, 2, 2), mode="constant", value=float("-inf"))
    before_img, after_img = _neighbor_max_split(padded[..., 1:-1, 1:-1])
    # >= all 8 neighbors and > the 4 before it: the first pixel of a flat top.
    argmax_and_thresh_img = (cms > before_img) & (cms >= after_img) & (cms > threshold)

    peak_subs = torch.stack(
        torch.where(argmax_and_thresh_img.permute(0, 2, 3, 1)), axis=-1
    )
    s, y, x, c = peak_subs.unbind(1)
    peak_vals = cms[s, c, y, x]
    peak_points = peak_subs[:, [2, 1]].to(torch.float32)
    tied = peak_vals == after_img[s, c, y, x]
    # Ties are rare in fp32, so the flat-top check only runs when there is one.
    # Traced or compiled graphs always run it (it keeps every untied peak).
    tracing = torch.jit.is_tracing() or torch.compiler.is_compiling()
    if tracing or bool(tied.any()):
        keep, offsets = _flat_tops(padded, peak_subs, peak_vals)
        keep = torch.nonzero(keep)[:, 0]
        peak_subs, peak_vals = peak_subs[keep], peak_vals[keep]
        peak_points = peak_points[keep] + offsets[keep]
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
    accepts ``None`` / ``"none"`` (no refinement), ``"integral"`` or
    ``"local"`` (quarter-pixel offset towards the higher neighbor, see
    :func:`quarter_offset_refinement`); other names raise ``ValueError``.
    """
    refinement = _check_refinement(refinement)
    rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds = (
        find_local_peaks_rough(cms, threshold=threshold)
    )

    if rough_peaks.size(0) == 0 or refinement is None:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds
    if refinement == "local":
        refined_peaks = _quarter_offset_local_peaks(
            cms, rough_peaks, peak_sample_inds, peak_channel_inds
        )
        return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    crop_size = integral_patch_size

    # A flat top's rough peak can be fractional; refine around the pixel it
    # floors to, since the crops are taken on the pixel grid.
    anchors = torch.floor(rough_peaks)
    bboxes = make_centered_bboxes(anchors, box_height=crop_size, box_width=crop_size)

    samples = cms.size(0)
    channels = cms.size(1)
    cms = torch.reshape(cms, [samples * channels, 1, cms.size(2), cms.size(3)])
    box_sample_inds = (peak_sample_inds * channels) + peak_channel_inds

    cm_crops = crop_bboxes(cms, bboxes, sample_inds=box_sample_inds)

    gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
    dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
    offsets = torch.cat([dx_hat, dy_hat], dim=1)

    refined_peaks = anchors + offsets
    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds
