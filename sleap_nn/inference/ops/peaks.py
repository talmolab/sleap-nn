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
- :func:`dark_refinement` / :func:`quarter_offset_refinement` — sub-pixel
  refinement of global peaks (``refinement="dark"`` / ``"local"``).
- :func:`resolve_refinement` — turns a user-facing refinement name (including
  ``"auto"``) into what a given peak finder runs.
- :func:`fuse_flip_confmaps` — flip-test averaging of confmaps predicted on an
  image and its horizontal mirror.

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
    dark_blur_sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find global peaks with optional refinement.

    Args:
        cms: ``(samples, channels, height, width)`` confidence maps.
        threshold: Peaks below this are NaN-padded.
        refinement: ``None`` (or ``"none"``) returns grid-aligned peaks;
            ``"integral"`` runs sub-pixel integral refinement on a small patch
            around each peak; ``"local"`` moves each peak a quarter pixel towards
            its higher neighbor along each axis (:func:`quarter_offset_refinement`);
            ``"dark"`` runs DARK (:func:`dark_refinement`). ``"auto"`` is not
            accepted here: resolve it first with :func:`resolve_refinement`.
        integral_patch_size: Side length of the refinement patch.
        dark_blur_sigma: Std (confmap pixels) of the DARK modulation blur; set
            it to the head's training ``sigma``.

    Returns:
        ``(peak_points, peak_vals)`` as in :func:`find_global_peaks_rough`.

    Raises:
        ValueError: If ``refinement`` is not one of the names above.
    """
    if refinement == "none":
        refinement = None
    if refinement not in (None, "integral", "local", "dark"):
        raise ValueError(
            f"Unknown peak refinement {refinement!r}; expected one of "
            "None, 'none', 'integral', 'local' or 'dark'."
        )
    rough_peaks, peak_vals = find_global_peaks_rough(cms, threshold=threshold)

    if refinement is None or torch.isnan(rough_peaks).all():
        return rough_peaks, peak_vals
    if refinement == "local":
        return quarter_offset_refinement(cms, rough_peaks), peak_vals
    if refinement == "dark":
        return dark_refinement(cms, rough_peaks, blur_sigma=dark_blur_sigma), peak_vals

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


#: User-facing refinement names. ``"auto"`` resolves per peak finder.
REFINEMENT_METHODS = ("auto", "dark", "integral", "local", "none")

#: Refinements that need one peak per channel (global peaks).
GLOBAL_PEAK_REFINEMENTS = ("dark", "local")


def resolve_refinement(
    refinement: Optional[str],
    auto: str = "integral",
    peaks: str = "global",
) -> Optional[str]:
    """Resolve a user-facing refinement name for one peak finder.

    ``"auto"`` picks the finder's default: ``"dark"`` for the top-down
    (centered-instance) keypoint stage, ``"integral"`` everywhere else.
    ``"dark"`` and ``"local"`` refine a single global peak per channel, so
    finders of local (multi-instance) peaks cannot run them:

    - ``peaks="local"`` (bottom-up keypoints) raises,
    - ``peaks="centroid"`` (centroid stage, alone or in a top-down pipeline)
      falls back to ``"integral"``: ``--integral_refinement dark`` on a
      top-down model refines the keypoints, not the centroids.

    Args:
        refinement: One of :data:`REFINEMENT_METHODS`, or ``None`` (no
            refinement).
        auto: What ``"auto"`` resolves to.
        peaks: ``"global"``, ``"local"`` or ``"centroid"`` (see above).

    Returns:
        The refinement to pass to :func:`find_global_peaks` /
        :func:`find_local_peaks`: ``None``, ``"integral"``, ``"local"`` or
        ``"dark"``.

    Raises:
        ValueError: On an unknown name, or ``"dark"`` / ``"local"`` with
            ``peaks="local"``.
    """
    if refinement is None:
        return None
    if refinement not in REFINEMENT_METHODS:
        raise ValueError(
            f"Unknown peak refinement {refinement!r}; expected one of "
            f"{', '.join(REFINEMENT_METHODS)}."
        )
    if refinement == "auto":
        refinement = auto
    if refinement == "none":
        return None
    if refinement in GLOBAL_PEAK_REFINEMENTS and peaks != "global":
        if peaks == "centroid":
            return "integral"
        raise ValueError(
            f"Peak refinement {refinement!r} refines one global peak per channel "
            "and is not available for multi-instance (bottom-up) peak finding; "
            "use 'integral' or 'none'."
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


def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Normalized 1D Gaussian of radius ``max(1, round(3 * sigma))``."""
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def dark_refinement(
    cms: torch.Tensor,
    rough_peaks: torch.Tensor,
    blur_sigma: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Refine global peaks with DARK (Zhang et al., CVPR 2020).

    The confmap is first modulated with a Gaussian blur of std ``blur_sigma``,
    then the sub-pixel mode is estimated with one Newton step on the log
    confmap at the integer peak: ``offset = -H^-1 g`` with central
    finite-difference gradient ``g`` and Hessian ``H``. For a noise-free
    Gaussian target the log confmap is exactly quadratic (blurring a Gaussian
    gives a Gaussian), so the estimate is exact wherever the keypoint falls
    between grid points. The blur suppresses the noise and flat tops of
    predicted maps; setting it to the training ``sigma`` is the paper's choice.

    The blur only matters around the peak, so it runs on a
    ``(3 + 2R) x (3 + 2R)`` patch (``R = round(3 * blur_sigma)``) rather than
    the whole map; with replicate padding the result is identical. DARK also
    rescales the blurred map to keep its maximum, which does not change the
    log-derivatives and is skipped.

    Offsets are dropped (zero) where the Hessian is not negative definite and
    clamped to +-1 confmap pixel.

    Args:
        cms: ``(samples, channels, height, width)`` confidence maps.
        rough_peaks: ``(samples, channels, 2)`` grid peaks from
            :func:`find_global_peaks_rough` (NaN for missing).
        blur_sigma: Std of the modulation blur in confmap pixels (``0`` skips).
        eps: Floor applied before the log.

    Returns:
        ``(samples, channels, 2)`` refined ``(x, y)`` peaks (NaN preserved).
    """
    samples, channels = rough_peaks.shape[:2]
    if blur_sigma > 0:
        k1 = _gaussian_kernel_1d(blur_sigma, cms.device)
        r = (k1.numel() - 1) // 2
        p = _gather_neighborhood(cms, rough_peaks, radius=1 + r).to(torch.float32)
        p = p.reshape(samples * channels, 1, 3 + 2 * r, 3 + 2 * r)
        p = F.conv2d(p, k1.view(1, 1, 1, -1))
        p = F.conv2d(p, k1.view(1, 1, -1, 1))
        p = p.reshape(samples, channels, 3, 3)
    else:
        p = _gather_neighborhood(cms, rough_peaks, radius=1).to(torch.float32)
    p = torch.log(p.clamp(min=eps))
    dx = 0.5 * (p[..., 1, 2] - p[..., 1, 0])
    dy = 0.5 * (p[..., 2, 1] - p[..., 0, 1])
    dxx = p[..., 1, 2] - 2 * p[..., 1, 1] + p[..., 1, 0]
    dyy = p[..., 2, 1] - 2 * p[..., 1, 1] + p[..., 0, 1]
    dxy = 0.25 * (p[..., 2, 2] - p[..., 0, 2] - p[..., 2, 0] + p[..., 0, 0])
    det = dxx * dyy - dxy**2
    ok = (det > eps) & (dxx < 0)  # negative definite: a proper maximum
    safe_det = torch.where(ok, det, torch.ones_like(det))
    off_x = -(dyy * dx - dxy * dy) / safe_det
    off_y = -(dxx * dy - dxy * dx) / safe_det
    offsets = torch.stack([off_x, off_y], dim=-1)
    offsets = torch.where(ok[..., None], offsets, torch.zeros_like(offsets))
    return rough_peaks + offsets.clamp(-1.0, 1.0)


def flip_idx_from_symmetries(n_nodes: int, symmetric_inds) -> torch.Tensor:
    """Channel permutation ``(n_nodes,)`` swapping each symmetric node pair.

    Args:
        n_nodes: Number of confmap channels (skeleton nodes).
        symmetric_inds: Iterable of ``(i, j)`` node-index pairs (e.g. from
            :func:`sleap_nn.data.utils.get_symmetric_inds`). Repeated pairs (in
            either order) are fine.

    Returns:
        ``(n_nodes,)`` long tensor with ``flip_idx[i] = j`` and
        ``flip_idx[j] = i`` for each pair, identity elsewhere.

    Raises:
        ValueError: If a node is paired with two different nodes, or an index is
            out of range.
    """
    flip_idx = list(range(n_nodes))
    for a, b in symmetric_inds or []:
        a, b = int(a), int(b)
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            raise ValueError(
                f"Symmetry ({a}, {b}) is out of range for {n_nodes} nodes."
            )
        for i, j in ((a, b), (b, a)):
            if flip_idx[i] not in (i, j):
                raise ValueError(
                    f"Node {i} is symmetric to both node {flip_idx[i]} and node "
                    f"{j}; each node can have at most one symmetric partner."
                )
            flip_idx[i] = j
    return torch.tensor(flip_idx, dtype=torch.long)


def fuse_flip_confmaps(
    cms: torch.Tensor,
    cms_flipped: torch.Tensor,
    flip_idx: Optional[torch.Tensor] = None,
    output_stride: int = 1,
    input_width: Optional[int] = None,
) -> torch.Tensor:
    """Average confmaps with those of the horizontally mirrored input.

    ``cms_flipped`` is the model output for ``torch.flip(x, dims=[-1])``. It is
    mirrored back, its left/right channels are swapped with ``flip_idx``, it is
    resampled onto the grid of ``cms`` and the two are averaged.

    The resampling matters: sleap-nn samples confmap pixel ``j`` at input pixel
    ``j * stride``, and mirroring an input of width ``W`` maps ``x`` to
    ``W - 1 - x``. So mirrored-back pixel ``j`` sits at input pixel
    ``j * stride + d`` with ``d = W - 1 - (Wc - 1) * stride`` (``Wc`` = confmap
    width), i.e. ``stride - 1`` input pixels to the right when ``W = Wc *
    stride``. Averaging without the shift biases the decoded x by ``-d / 2``
    input pixels (1.5 px at stride 4). The shift is applied by linear
    interpolation along x with replicated borders.

    Args:
        cms: ``(S, C, H, Wc)`` confmaps of the original input.
        cms_flipped: ``(S, C, H, Wc)`` confmaps of the mirrored input, as
            predicted (this function mirrors them back).
        flip_idx: ``(C,)`` long permutation: output channel ``c`` is read from
            mirrored channel ``flip_idx[c]`` (see :func:`flip_idx_from_symmetries`).
            ``None`` keeps the channel order.
        output_stride: Confmap stride relative to the model input.
        input_width: Width ``W`` of the model input (after padding). ``None``
            assumes ``Wc * output_stride``.

    Returns:
        ``(S, C, H, Wc)`` fused confmaps on the grid of ``cms``.
    """
    back = torch.flip(cms_flipped, dims=[-1])
    if flip_idx is not None:
        back = back[:, flip_idx.to(back.device)]
    width = back.shape[-1]
    if input_width is None:
        input_width = width * output_stride
    # Mirrored-back pixel j holds the value of confmap position j + d (confmap
    # pixels); the fused grid needs position j, i.e. back[j - d].
    d = (input_width - 1 - (width - 1) * output_stride) / output_stride
    if d != 0:
        pos = torch.arange(width, dtype=torch.float32, device=back.device) - d
        lo = torch.floor(pos)
        frac = (pos - lo).to(back.dtype)
        i0 = lo.long().clamp(0, width - 1)
        i1 = (lo.long() + 1).clamp(0, width - 1)
        back = back[..., i0] * (1 - frac) + back[..., i1] * frac
    return 0.5 * (cms + back)


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
