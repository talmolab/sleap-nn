"""Inference utilities for bottom-up instance segmentation."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

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
    mask_cleanup_radius: int = 0,
    distance_gate_alpha: Optional[float] = None,
    distance_gate_iters: int = 3,
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
        mask_cleanup_radius: When ``mask_cleanup`` is on and this is ``> 0``,
            additionally apply a morphological open->close with an elliptical
            kernel of this radius (output-stride pixels) before keep-largest-CC.
            ``0`` (default) keeps the keep-largest-CC + fill-holes behavior.
        distance_gate_alpha: Adaptive distance-gate strength. When ``None``
            (default) every foreground pixel is assigned to its nearest center
            by ``argmin`` (the original, byte-for-byte behavior). When set, a
            pixel is dropped from its assigned instance if its offset-predicted
            center is farther than ``R_k = alpha * sqrt(area_k / pi)`` from that
            center, where ``area_k`` is the current pixel count of instance
            ``k``. The radius is re-estimated for ``distance_gate_iters``
            iterations (areas shrink as strays are gated out). This deletes
            stray foreground pixels and off-instance phantom peaks (a scale-free
            ``min_mask_area``) without affecting well-grouped pixels (their
            offset residual is small). Radii are expressed in original-pixel
            units (the same space as the ``dists`` below).
        distance_gate_iters: Number of adaptive re-estimation iterations for the
            distance gate (only used when ``distance_gate_alpha`` is not
            ``None``). Default ``3``.

    Returns:
        List of dicts, each with:
            - "mask": (H, W) boolean numpy array (at output stride resolution)
            - "center": (x, y) tuple in original pixel coordinates
            - "score": float confidence score (peak value)
    """
    # Run grouping device-consistently on CPU. find_center_peaks() routes through
    # scipy connected-components and returns CPU peak tensors, and the per-instance
    # masks below are built as CPU tensors (-> numpy), so a GPU input would mix
    # cuda/cpu devices inside this function. This is a no-op for the inference path
    # (already-CPU tensors) but is required for the GPU callers added in #649 —
    # epoch-end mask eval and training viz both pass cuda head tensors. The cost is
    # negligible (a few frames per epoch, not the training loop).
    foreground = foreground.detach().cpu()
    center_heatmap = center_heatmap.detach().cpu()
    offsets = offsets.detach().cpu()

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
    # Squared distance from each fg pixel to its assigned center (original px^2).
    dmin = dists.gather(1, assignments.unsqueeze(1)).squeeze(1)  # (M,)

    # Optional adaptive distance gate: drop a pixel whose offset-predicted center
    # is farther than R_k = alpha*sqrt(area_k/pi) from its assigned center. The
    # radius is re-estimated for ``distance_gate_iters`` passes so that areas (and
    # thus radii) shrink as stray pixels are gated out. ``distance_gate_alpha is
    # None`` keeps the original argmin behavior byte-for-byte (every pixel kept).
    keep = torch.ones_like(assignments, dtype=torch.bool)
    if distance_gate_alpha is not None:
        n_centers = len(centers)
        for _ in range(max(1, int(distance_gate_iters))):
            # area_k = current kept-pixel count assigned to center k (grid cells).
            counts = torch.zeros(n_centers, dtype=torch.long)
            counts.scatter_add_(
                0,
                assignments[keep],
                torch.ones(int(keep.sum().item()), dtype=torch.long),
            )
            # R_k in grid cells -> original px (* output_stride), then squared.
            r_grid = float(distance_gate_alpha) * torch.sqrt(counts.float() / math.pi)
            r_px2 = (r_grid * float(output_stride)) ** 2  # (N,)
            keep = dmin <= r_px2[assignments]

    # 5. Build per-instance masks
    instances = []
    for i in range(len(centers)):
        member_mask = (assignments == i) & keep
        if member_mask.sum() == 0:
            continue

        instance_mask = torch.zeros((h, w), dtype=torch.bool)
        instance_mask[fg_y[member_mask], fg_x[member_mask]] = True

        mask_np = instance_mask.numpy()
        if mask_cleanup:
            mask_np = _clean_instance_mask(mask_np, radius=mask_cleanup_radius)
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


def _clean_instance_mask(mask: np.ndarray, radius: int = 0) -> np.ndarray:
    """Keep the largest connected component and fill interior holes.

    Suppresses speckle and fragments left by the per-pixel offset grouping.
    Operates at output-stride resolution; a no-op for an already-clean mask.

    When ``radius > 0``, a morphological open->close with an elliptical
    structuring element of that radius (in output-stride pixels) runs FIRST:
    the open deletes isolated speckle and thin connectors (the noise that
    explodes the mask RLE), the close seals pinholes and small concavities.
    Keep-largest-CC + hole-fill then returns a single solid blob. ``radius == 0``
    reproduces the pre-morphology behavior byte-for-byte (the keep-largest-CC +
    fill-holes already shipped in #624).
    """
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label as cc_label

    if radius and radius > 0:
        import cv2

        ksize = 2 * int(radius) + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        u8 = mask.astype(np.uint8)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
        mask = u8.astype(bool)

    labels, n = cc_label(mask)
    if n > 1:
        # Keep the largest component (component 0 is background).
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    return binary_fill_holes(mask)


class CenteredInstanceMaskInferenceModel(L.LightningModule):
    """Stage-2 holder for top-down (crop-centered) instance segmentation (#622).

    A thin attribute bag (no real ``forward`` — the modern composed
    :class:`~sleap_nn.inference.layers.topdown_segmentation.TopDownSegmentationLayer`
    drives the run) carrying the per-crop seg model + the knobs the layer
    builder reads off it. Mirrors how the legacy ``FindInstancePeaks`` is used
    purely as an attribute holder by the modern top-down layer builder.

    Attributes:
        torch_model: Per-crop seg Lightning module; ``forward`` returns the
            ``SegmentationHead`` logits.
        output_stride: Head-map → crop-pixel stride.
        input_scale: Input scale the model was trained with (applied to crops).
        max_stride: Backbone max stride (crops padded to a multiple of it).
        fg_threshold: Foreground probability threshold for binarization.
        mask_output / polygon_epsilon: Output-packaging knobs (read by the
            layer builder and forwarded to ``Outputs.to_labels``).
    """

    def __init__(
        self,
        torch_model,
        output_stride: int = 2,
        input_scale: float = 1.0,
        max_stride: int = 1,
        fg_threshold: float = 0.5,
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
    ):
        """Stash the per-crop seg model + knobs."""
        super().__init__()
        self.torch_model = torch_model
        self.output_stride = int(output_stride)
        self.input_scale = float(input_scale)
        self.max_stride = int(max_stride)
        self.fg_threshold = float(fg_threshold)
        self.mask_output = str(mask_output)
        self.polygon_epsilon = float(polygon_epsilon)


class SemanticSegmentationInferenceModel(L.LightningModule):
    """Inference model for whole-frame semantic (foreground/background) segmentation.

    A lone :class:`~sleap_nn.architectures.heads.SegmentationHead` on the WHOLE
    frame (no crop, no instance grouping). The trained model's ``forward`` returns
    ``{"SegmentationHead": prob}`` with the sigmoid ALREADY applied (mirroring
    :class:`BottomUpSegmentationLightningModule`, whose foreground head is
    sigmoided in ``forward`` — required so tiled inference stitches probabilities,
    not logits). ``postprocess`` thresholds the foreground map into ONE mask per
    frame; there is no center/offset field and no ``group_instances_from_offsets``.

    Like :class:`BottomUpSegmentationInferenceModel`, this is primarily an
    attribute bag whose ``.torch_model`` + packaging knobs are read off it by
    ``_build_semantic_segmentation_layer`` (via ``getattr``); the real inference
    run is driven by the composed
    :class:`~sleap_nn.inference.layers.segmentation.SemanticSegmentationLayer`.
    ``forward`` is provided for training-viz / GPU mask-eval parity and emits raw
    output-stride masks (``min_mask_area`` / ``full_res_masks`` / packaging knobs
    are applied later, in the layer's ``postprocess``).

    Attributes:
        torch_model: Callable model returning ``{"SegmentationHead": prob}``
            (sigmoid already applied).
        fg_threshold: Foreground probability threshold for binarization.
        output_stride: Stride of the seg head map relative to the model input.
        input_scale: Input scale the model was trained with (applied to frames).
        min_mask_area: Minimum mask area (ORIGINAL-image pixels) carried through
            to ``SemanticSegmentationLayer`` to drop a tiny spurious mask. ``0``
            disables it. Not applied here (``forward`` returns output-stride masks).
        full_res_masks / mask_output / polygon_epsilon: Output-packaging knobs
            read by the layer builder and forwarded to the layer / ``Outputs.to_labels``.
    """

    def __init__(
        self,
        torch_model,
        fg_threshold: float = 0.5,
        output_stride: int = 2,
        input_scale: float = 1.0,
        min_mask_area: int = 0,
        full_res_masks: bool = False,
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
    ):
        """Stash the whole-frame seg model + packaging knobs."""
        super().__init__()
        self.torch_model = torch_model
        self.fg_threshold = float(fg_threshold)
        self.output_stride = int(output_stride)
        self.input_scale = float(input_scale)
        self.min_mask_area = int(min_mask_area)
        self.full_res_masks = bool(full_res_masks)
        self.mask_output = str(mask_output)
        self.polygon_epsilon = float(polygon_epsilon)

    def forward(self, batch: Dict) -> List[List[Dict]]:
        """Threshold the foreground map into ONE mask per batch element.

        Args:
            batch: Dict with an ``"image"`` key. Shape ``(B, C, H, W)``. Images
                should already be padded to the model's max stride.

        Returns:
            List (one per batch element) of instance lists. Each element is either
            empty (no foreground above threshold) or a single-item list
            ``[{"mask", "score"}]`` where ``mask`` is an ``(h, w)`` boolean numpy
            array at output-stride resolution and ``score`` is the mean foreground
            probability over the mask.
        """
        images = batch["image"]
        if images.dim() == 5:
            images = images.squeeze(1)

        images = images.to(self.device)

        output = self.torch_model(images.unsqueeze(1))

        foreground = output["SegmentationHead"]  # (B, 1, h, w), already sigmoid
        foreground = foreground.detach().cpu()

        batch_results: List[List[Dict]] = []
        for b in range(foreground.shape[0]):
            fg = foreground[b, 0]  # (h, w)
            fg_binary = fg > self.fg_threshold
            if not bool(fg_binary.any()):
                batch_results.append([])
                continue
            score = float(fg[fg_binary].mean())
            batch_results.append([{"mask": fg_binary.numpy(), "score": score}])

        return batch_results


# --------------------------------------------------------------------------- #
# Fragment-merge: RAG over candidate masks + greedy/multicut agglomeration.
#
# Failure mode: one animal is split into >=2 adjacent masks because two surviving
# center peaks (typically ~20 px apart along the body) each win a half. Neither
# ``min_mask_area`` (both halves are large) nor the distance gate (the winning
# peak is real and close) can fix this. ``merge_instances`` re-fuses the within-
# animal pieces while keeping two genuinely-touching distinct animals apart, by
# scoring each touching pair with a center-valley/ridge + offset-agreement
# affinity. The center-valley signal is load-bearing: a deep heatmap valley
# between two centers means "two animals" (do not merge); a high ridge means "one
# body" (merge). All of this is inert unless ``merge_fragments`` is enabled in
# the layer; ``group_instances_from_offsets`` itself is never changed by it.
# --------------------------------------------------------------------------- #
def _mask_pred_centers(
    mask: np.ndarray, offsets: np.ndarray, output_stride: int
) -> np.ndarray:
    """Offset-predicted center (x, y) px for each foreground pixel of ``mask``.

    Mirrors the grouping convention: ``pixel_px = grid * stride + stride/2`` then
    add the offset. Returns ``(M, 2)`` in original-pixel coordinates.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float64)
    dx = offsets[0, ys, xs]
    dy = offsets[1, ys, xs]
    px = xs.astype(np.float64) * output_stride + output_stride / 2.0
    py = ys.astype(np.float64) * output_stride + output_stride / 2.0
    return np.stack([px + dx, py + dy], axis=1)


def _contact_fraction(a: np.ndarray, b: np.ndarray, dilate_iters: int = 1) -> float:
    """Symmetric touch fraction of two masks: 0 if their dilations do not touch.

    ``(|dilate(A) & B| + |A & dilate(B)|) / min(area_a, area_b)``.

    ``dilate_iters`` is clamped to at least ``1``: the candidate masks coming out
    of :func:`group_instances_from_offsets` are MUTUALLY EXCLUSIVE (every fg pixel
    is argmin-assigned to exactly one center), so two abutting fragments of one
    animal never overlap. A raw-overlap contact test (no dilation) would therefore
    always report zero contact and silently disable the entire fragment-merge for
    exactly the split-animal case it targets. At least one dilation is required
    for the touch test to be meaningful.
    """
    from scipy.ndimage import binary_dilation

    iters = max(1, int(dilate_iters))
    da = binary_dilation(a, iterations=iters)
    db = binary_dilation(b, iterations=iters)
    overlap = int((da & b).sum() + (a & db).sum())
    if overlap == 0:
        return 0.0
    denom = max(1, min(int(a.sum()), int(b.sum())))
    return overlap / denom


def _center_valley_ridge(
    heatmap: np.ndarray,
    ca: Tuple[float, float],
    cb: Tuple[float, float],
    peak_a: float,
    peak_b: float,
    n_samples: int = 48,
) -> float:
    """Ridge score along the center-line between two centers (grid coords).

    Returns ``min_along_path / min(peak_a, peak_b)`` clipped to ``[0, 1]``: ~1
    means the heatmap stays high between the two centers (one body / ridge =>
    MERGE); ~0 means it dips to background (a valley => two animals => DON'T
    merge). The interior 70% of the segment is sampled so the peaks themselves
    are excluded.
    """
    h, w = heatmap.shape
    t = np.linspace(0.0, 1.0, n_samples)
    xs = ca[0] + (cb[0] - ca[0]) * t
    ys = ca[1] + (cb[1] - ca[1]) * t
    lo, hi = int(0.15 * n_samples), int(0.85 * n_samples)
    xs, ys = xs[lo:hi], ys[lo:hi]
    if len(xs) == 0:
        return 1.0
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    path_vals = heatmap[yi, xi]
    denom = max(1e-6, min(peak_a, peak_b))
    return float(np.clip(path_vals.min() / denom, 0.0, 1.0))


def _offset_agreement(pa: np.ndarray, pb: np.ndarray, output_stride: int) -> float:
    """Do two masks' pixels predict a SHARED center?

    ``pa``/``pb`` are the offset-predicted centers (px) of each mask's pixels. For
    a real fragment-split the two clouds cluster on the single true centroid
    (small separation relative to their spread => ~1); two distinct animals
    predict centers a body apart (=> ~0).
    """
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    sep = float(np.hypot(*(pa.mean(0) - pb.mean(0))))
    spread = float(0.5 * (pa.std(0).mean() + pb.std(0).mean()))
    scale = max(spread, float(output_stride))
    return float(np.exp(-(sep**2) / (2.0 * (2.0 * scale) ** 2)))


def _build_merge_rag(
    instances: List[Dict],
    center_heatmap: np.ndarray,
    offsets: np.ndarray,
    output_stride: int,
    *,
    dilate_iters: int = 1,
    w_valley: float = 1.0,
    w_offset: float = 0.25,
    contact_floor: float = 1e-3,
) -> Dict[Tuple[int, int], float]:
    """Region-adjacency graph over candidate masks: edge ``(i<j) -> affinity``.

    Edges exist only between masks whose dilations touch (non-touching pairs get
    no direct edge; transitive merges via touching chains are still possible).
    Affinity in ``[0, 1]``::

        affinity = contact_gate * (w_valley*ridge + w_offset*offset) / (w_valley+w_offset)

    where ``contact_gate = min(1, contact / 0.05)`` saturates so a firm touch does
    not over-weight, and the ridge/offset terms decide WHETHER to merge given
    contact. With ``w_valley == w_offset == 0`` the affinity collapses to the raw
    contact gate (a contact-only ablation that over-merges touching distinct
    animals — used by tests to show the valley signal is load-bearing).
    """
    n = len(instances)
    pred_centers = [
        _mask_pred_centers(inst["mask"], offsets, output_stride) for inst in instances
    ]
    edges: Dict[Tuple[int, int], float] = {}
    wsum = w_valley + w_offset
    for i in range(n):
        for j in range(i + 1, n):
            contact = _contact_fraction(
                instances[i]["mask"], instances[j]["mask"], dilate_iters
            )
            if contact <= contact_floor:
                continue
            contact_gate = min(1.0, contact / 0.05)
            if wsum <= 0:
                edges[(i, j)] = contact_gate
                continue
            # Invert the grid->pixel convention exactly (``px = grid*stride +
            # stride/2``; see line ~160 and ``_mask_pred_centers``) to recover the
            # grid coordinate the heatmap is indexed in. A bare ``center / stride``
            # would leave a half-cell (+0.5) offset on the sampled center-line.
            half = output_stride / 2.0
            ca = (
                (instances[i]["center"][0] - half) / output_stride,
                (instances[i]["center"][1] - half) / output_stride,
            )
            cb = (
                (instances[j]["center"][0] - half) / output_stride,
                (instances[j]["center"][1] - half) / output_stride,
            )
            ridge = _center_valley_ridge(
                center_heatmap,
                ca,
                cb,
                instances[i]["score"],
                instances[j]["score"],
            )
            offset = _offset_agreement(pred_centers[i], pred_centers[j], output_stride)
            blend = (w_valley * ridge + w_offset * offset) / wsum
            edges[(i, j)] = float(contact_gate * blend)
    return edges


def _union_groups(groups: List[set], instances: List[Dict]) -> List[Dict]:
    """Materialize merged masks from a partition (list of node-index sets).

    The merged mask is the OR of its members; the highest-scoring member's center
    and score are kept as the representative.
    """
    out = []
    for g in groups:
        members = sorted(g)
        best = max(members, key=lambda k: instances[k]["score"])
        mask = np.zeros_like(instances[members[0]]["mask"])
        for k in members:
            mask |= instances[k]["mask"]
        out.append(
            {
                "mask": mask,
                "center": instances[best]["center"],
                "score": instances[best]["score"],
            }
        )
    return out


def _merge_greedy_affinity(
    instances: List[Dict],
    edges: Dict[Tuple[int, int], float],
    *,
    thresholds: Sequence[float] = (0.85, 0.6, 0.4),
) -> List[Dict]:
    """Greedy decreasing-threshold agglomeration (Liu et al. ECCV'18 graph merge).

    In each phase, repeatedly contract the max-affinity live edge >= the phase
    threshold; the merged super-node's affinity to each neighbor is the MEAN over
    the contracted members' affinities. Union-find over node ids.
    """
    n = len(instances)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def key(a, b):
        return (a, b) if a < b else (b, a)

    # Live affinities keyed by (root_a, root_b) -> list of member affinities.
    aff: Dict[Tuple[int, int], List[float]] = {k: [v] for k, v in edges.items()}

    for thr in thresholds:
        while True:
            best_e = None
            best_v = -1.0
            for (a, b), vals in aff.items():
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                v = float(np.mean(vals))
                if v > best_v:
                    best_v = v
                    best_e = (ra, rb)
            if best_e is None or best_v < thr:
                break
            ra, rb = best_e
            parent[rb] = ra
            new_aff: Dict[Tuple[int, int], List[float]] = {}
            for (a, b), vals in aff.items():
                ca, cb = find(a), find(b)
                if ca == cb:
                    continue
                new_aff.setdefault(key(ca, cb), []).extend(vals)
            aff = new_aff

    groups: Dict[int, set] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return _union_groups(list(groups.values()), instances)


def _merge_multicut_greedy(
    instances: List[Dict],
    edges: Dict[Tuple[int, int], float],
    *,
    join_bias: float = 0.5,
) -> List[Dict]:
    """Greedy min-cost multicut / correlation clustering (GAEC-style).

    Edge cost = ``logit(affinity) - logit(join_bias)``: positive => attractive
    (want joined), negative => repulsive (want cut). Repeatedly contract the
    most-attractive edge while any positive-cost edge remains, summing parallel
    edge costs (correlation-clustering objective). No fixed instance count.
    """
    n = len(instances)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    cost: Dict[Tuple[int, int], float] = {
        k: logit(v) - logit(join_bias) for k, v in edges.items()
    }

    while True:
        best_e = None
        best_c = 0.0  # strictly > 0 to contract
        for (a, b), c in cost.items():
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if c > best_c:
                best_c = c
                best_e = (ra, rb)
        if best_e is None:
            break
        ra, rb = best_e
        parent[rb] = ra
        new_cost: Dict[Tuple[int, int], float] = {}
        for (a, b), c in cost.items():
            ca, cb = find(a), find(b)
            if ca == cb:
                continue
            k = key(ca, cb)
            new_cost[k] = new_cost.get(k, 0.0) + c
        cost = new_cost

    groups: Dict[int, set] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return _union_groups(list(groups.values()), instances)


def merge_instances(
    instances: List[Dict],
    center_heatmap: np.ndarray,
    offsets: np.ndarray,
    output_stride: int,
    *,
    method: str = "greedy",
    dilate_iters: int = 1,
    w_valley: float = 1.0,
    w_offset: float = 0.25,
    thresholds: Sequence[float] = (0.85, 0.6, 0.4),
    join_bias: float = 0.5,
) -> List[Dict]:
    """Fuse over-segmented fragments of one animal via a RAG over candidate masks.

    Builds a region-adjacency graph over the candidate masks (edge affinity =
    contact-gate * a center-valley-ridge / offset-agreement blend), then runs the
    chosen agglomeration. Two genuinely-touching distinct animals are kept apart
    by the valley term (a deep heatmap valley between their centers vetoes the
    merge). Operates at output-stride (grid) resolution, BEFORE upsample and
    ``min_mask_area``, on the dicts returned by
    :func:`group_instances_from_offsets`.

    Args:
        instances: ``{"mask", "center", "score"}`` dicts (grid-resolution masks,
            centers in original-pixel coords).
        center_heatmap: ``(h, w)`` center heatmap at grid resolution.
        offsets: ``(2, h, w)`` offset field ``(dx, dy)`` in original-pixel units.
        output_stride: Stride of the head maps relative to the model input.
        method: ``"greedy"`` (default, recommended) decreasing-threshold
            agglomeration, or ``"multicut"`` greedy correlation clustering.
        dilate_iters: Dilation iterations for the contact test (default ``1``).
        w_valley: Weight on the center-valley ridge term (default ``1.0``).
        w_offset: Weight on the offset-agreement term (default ``0.25``).
        thresholds: Decreasing affinity thresholds per greedy phase.
        join_bias: Multicut decision boundary (affinity > this => attractive).

    Returns:
        A NEW list of merged ``{"mask", "center", "score"}`` dicts (each merged
        mask is the OR of its members; the highest-scoring member is the
        representative). Returns ``instances`` unchanged when ``method == "none"``
        or fewer than two instances are present.
    """
    if method == "none" or len(instances) <= 1:
        return instances
    edges = _build_merge_rag(
        instances,
        center_heatmap,
        offsets,
        output_stride,
        dilate_iters=dilate_iters,
        w_valley=w_valley,
        w_offset=w_offset,
    )
    if method == "greedy":
        return _merge_greedy_affinity(instances, edges, thresholds=thresholds)
    if method == "multicut":
        return _merge_multicut_greedy(instances, edges, join_bias=join_bias)
    raise ValueError(f"unknown merge method {method!r}")


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
        mask_cleanup_radius: int = 0,
        distance_gate_alpha: Optional[float] = None,
        merge_fragments: bool = False,
        merge_method: str = "greedy",
        merge_thresholds: tuple = (0.85, 0.6, 0.4),
        merge_w_valley: float = 1.0,
        merge_w_offset: float = 0.25,
        merge_dilate: int = 1,
        full_res_masks: bool = False,
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
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
        self.mask_cleanup_radius = int(mask_cleanup_radius)
        # Increment-A fragment-merge / distance-gate knobs. Carried for
        # ``_build_bottomup_segmentation_layer`` (read via getattr) and applied in
        # ``SegmentationLayer.postprocess`` (NOT in this model's ``forward``, which
        # emits raw output-stride masks for training viz). All default to today's
        # behavior: ``distance_gate_alpha=None`` and ``merge_fragments=False``.
        self.distance_gate_alpha = (
            None if distance_gate_alpha is None else float(distance_gate_alpha)
        )
        self.merge_fragments = bool(merge_fragments)
        self.merge_method = str(merge_method)
        self.merge_thresholds = tuple(merge_thresholds)
        self.merge_w_valley = float(merge_w_valley)
        self.merge_w_offset = float(merge_w_offset)
        self.merge_dilate = int(merge_dilate)
        # Output-packaging knobs carried for ``_build_bottomup_segmentation_layer``
        # (read off this model via getattr). ``forward`` itself only emits
        # output-stride masks for training viz, so it consumes ``mask_cleanup_radius``
        # but not the packaging-time ``full_res_masks``/``mask_output``/``polygon_epsilon``.
        self.full_res_masks = bool(full_res_masks)
        self.mask_output = str(mask_output)
        self.polygon_epsilon = float(polygon_epsilon)

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
                mask_cleanup_radius=self.mask_cleanup_radius,
            )
            batch_results.append(instances)

        return batch_results
