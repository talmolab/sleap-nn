"""Bounding-box ground truth, augmentation geometry and training targets.

Shared by the ``bbox_*`` detection model types (``bbox_centernet``, ``bbox_ssd``):

* **Ground truth.** Boxes come from ``LabeledFrame.bboxes`` (``sio.UserBoundingBox``).
  :func:`add_bboxes_to_labels` can derive them in memory from pose instances or
  segmentation masks through sleap-io (``Instance.to_bbox`` /
  ``SegmentationMask.to_bbox``), so a pose dataset trains a detector unchanged.
* **Geometry.** A box rides the shared augmentation chain (size-match, scale, pad,
  rotate, scale, translate, flip) as a 4-corner pseudo-instance
  (:func:`bboxes_to_corners`) and is re-boxed afterwards as the corners' extent
  (:func:`corners_to_bboxes`), then clipped to the frame (:func:`clip_bboxes`).
* **Targets.** One encoder per detection style turns a frame's boxes into the dense
  tensors its loss reads: :class:`CenterNetTargetEncoder` (center heatmap + size and
  offset maps) and :class:`SSDTargetEncoder` (per-anchor class + box targets). The
  matching decoders live in :mod:`sleap_nn.inference.bbox`.

Box coordinates are ``(x1, y1, x2, y2)`` in pixels, with the same convention as
pose keypoints (a pixel's center is at its integer coordinate).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sleap_io as sio
import torch
from loguru import logger
from torchvision.ops import box_iou

from sleap_nn.config.model_config import DEFAULT_BBOX_CATEGORY

BBOX_SOURCES = ("auto", "bboxes", "instances", "masks")


# ---------------------------------------------------------------------------
# Ground truth from labels
# ---------------------------------------------------------------------------


def frame_user_bboxes(lf: sio.LabeledFrame) -> List[sio.BoundingBox]:
    """A frame's usable user boxes: not predicted and not degenerate."""
    return [
        b
        for b in (getattr(lf, "bboxes", None) or [])
        if not b.is_predicted and not b.is_empty and b.width > 0 and b.height > 0
    ]


def bbox_category_name(box: sio.BoundingBox) -> str:
    """The class name of a box; boxes without a category are ``"object"``."""
    category = getattr(box, "category", None)
    if category is None:
        return DEFAULT_BBOX_CATEGORY
    name = getattr(category, "name", category)
    return str(name) if name else DEFAULT_BBOX_CATEGORY


def labels_bbox_categories(labels_list: Sequence[sio.Labels]) -> List[str]:
    """Ordered category names of the user boxes in ``labels_list`` (first seen first).

    Returns ``["object"]`` when no box carries a category, so a class-agnostic
    dataset gets a single-class detector.
    """
    names: List[str] = []
    seen = set()
    for labels in labels_list:
        for lf in labels:
            for box in frame_user_bboxes(lf):
                name = bbox_category_name(box)
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names or [DEFAULT_BBOX_CATEGORY]


def resolve_bbox_source(labels: sio.Labels, source: str = "auto") -> str:
    """Resolve ``data_config.bbox_source`` for one labels file.

    ``auto`` picks ``bboxes`` when the file carries any user box, else ``instances``
    when it carries any user pose instance, else ``masks``.
    """
    if source not in BBOX_SOURCES:
        raise ValueError(f"bbox_source must be one of {BBOX_SOURCES}, got {source!r}.")
    if source != "auto":
        return source
    if any(frame_user_bboxes(lf) for lf in labels):
        return "bboxes"
    if any(lf.has_user_instances for lf in labels):
        return "instances"
    return "masks"


def _derived_frame_bboxes(
    lf: sio.LabeledFrame, source: str, padding: float
) -> List[sio.UserBoundingBox]:
    """User boxes derived from a frame's user instances or user masks."""
    boxes = []
    if source == "instances":
        for inst in lf.user_instances:
            if inst.is_empty:
                continue
            box = inst.to_bbox(mode="tight", padding=padding)
            if not box.is_empty and box.width > 0 and box.height > 0:
                boxes.append(box)
    elif source == "masks":
        for mask in getattr(lf, "masks", None) or []:
            if mask.is_predicted:
                continue
            box = mask.to_bbox(padding=padding)
            if not box.is_empty and box.width > 0 and box.height > 0:
                boxes.append(box)
    return boxes


def add_bboxes_to_labels(
    labels: sio.Labels, source: str = "auto", padding: float = 0.0
) -> str:
    """Give every frame of ``labels`` its ground-truth boxes for ``source``, in place.

    With ``bboxes`` the frames are left as they are. With ``instances`` / ``masks``
    each frame's user boxes are replaced (in memory) by boxes derived from its user
    pose instances / user masks through sleap-io, padded by ``padding`` pixels.
    Derived boxes keep the source's track and link back to it (``box.instance``).

    Args:
        labels: Labels to modify.
        source: One of ``auto``, ``bboxes``, ``instances``, ``masks`` (see
            :func:`resolve_bbox_source`).
        padding: Pixels added on every side of derived boxes.

    Returns:
        The resolved source.
    """
    resolved = resolve_bbox_source(labels, source)
    if resolved == "bboxes":
        return resolved
    n_boxes = 0
    for lf in labels:
        derived = _derived_frame_bboxes(lf, resolved, padding)
        lf.bboxes = [b for b in lf.bboxes if b.is_predicted] + derived
        n_boxes += len(derived)
    logger.info(
        f"Derived {n_boxes} ground-truth boxes from {resolved} "
        f"(padding={padding} px)."
    )
    return resolved


def frame_bbox_arrays(
    lf: sio.LabeledFrame, categories: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """A frame's user boxes as ``(n, 4)`` float32 xyxy and ``(n,)`` int64 class ids.

    Boxes whose category is not in ``categories`` are skipped (with a single-class
    detector every box is class 0).
    """
    index = {name: i for i, name in enumerate(categories)}
    single_class = len(categories) == 1
    xyxy, classes = [], []
    for box in frame_user_bboxes(lf):
        if box.is_rotated:
            x1, y1, x2, y2 = box.bounds
        else:
            x1, y1, x2, y2 = box.xyxy
        cls = 0 if single_class else index.get(bbox_category_name(box))
        if cls is None:
            continue
        xyxy.append([x1, y1, x2, y2])
        classes.append(cls)
    return (
        np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
    )


def bbox_size_stats(
    labels_list: Sequence[sio.Labels],
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """``sqrt(w * h)`` of every user box in ``labels_list``, in model-input pixels.

    Args:
        labels_list: Labels to measure.
        max_hw: The size matcher's ``(max_height, max_width)``; each frame's boxes
            are rescaled by the factor the size matcher applies to its video.
        scale: ``data_config.preprocessing.scale``.
    """
    from sleap_nn.data.resizing import compute_eff_scale

    sizes = []
    for labels in labels_list:
        for lf in labels:
            boxes = frame_user_bboxes(lf)
            if not boxes:
                continue
            eff = 1.0
            shape = getattr(lf.video, "shape", None)
            if max_hw is not None and shape is not None and None not in max_hw:
                if (shape[1], shape[2]) != tuple(max_hw):
                    eff = compute_eff_scale((shape[1], shape[2]), tuple(max_hw))
            sizes.extend(math.sqrt(b.width * b.height) * eff * scale for b in boxes)
    return np.asarray(sizes, dtype=np.float64)


def auto_anchor_sizes(
    box_sizes: np.ndarray, num_levels: int, strides: Sequence[int]
) -> List[float]:
    """Per-level SSD anchor sizes spanning the training boxes' size range.

    Geometrically spaced between the 5th and 95th percentile of ``sqrt(w * h)`` (a
    single level takes the median). Without boxes, falls back to RetinaNet's
    ``4 * stride``.
    """
    if box_sizes.size == 0:
        return [4.0 * s for s in strides]
    if num_levels == 1:
        return [float(np.median(box_sizes))]
    lo, hi = np.percentile(box_sizes, [5, 95])
    lo = max(float(lo), 1.0)
    hi = max(float(hi), lo * 1.01)
    return [float(v) for v in np.geomspace(lo, hi, num_levels)]


# ---------------------------------------------------------------------------
# Geometry through augmentation
# ---------------------------------------------------------------------------


def bboxes_to_corners(xyxy: np.ndarray) -> np.ndarray:
    """``(n, 4)`` xyxy boxes -> ``(n, 4, 2)`` corners (TL, TR, BR, BL)."""
    x1, y1, x2, y2 = (xyxy[:, i] for i in range(4))
    return np.stack(
        [
            np.stack([x1, y1], -1),
            np.stack([x2, y1], -1),
            np.stack([x2, y2], -1),
            np.stack([x1, y2], -1),
        ],
        axis=1,
    ).astype(np.float32)


def corners_to_bboxes(corners: torch.Tensor) -> torch.Tensor:
    """``(..., 4, 2)`` corners -> ``(..., 4)`` axis-aligned xyxy extent.

    After a rotation this is the rotated box's enclosing axis-aligned box (the
    standard treatment of boxes under rotation augmentation).
    """
    xs, ys = corners[..., 0], corners[..., 1]
    return torch.stack(
        [xs.min(-1).values, ys.min(-1).values, xs.max(-1).values, ys.max(-1).values],
        dim=-1,
    )


def clip_bboxes(
    xyxy: torch.Tensor,
    img_hw: Tuple[int, int],
    min_size: float = 2.0,
    min_visible_fraction: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip boxes to the frame and flag the ones still worth supervising.

    Args:
        xyxy: ``(n, 4)`` boxes.
        img_hw: ``(height, width)`` of the frame.
        min_size: Boxes narrower or shorter than this after clipping are dropped.
        min_visible_fraction: Boxes keeping less than this fraction of their area
            inside the frame (e.g. after a translation) are dropped.

    Returns:
        ``(clipped, keep)``: the clipped ``(n, 4)`` boxes and a ``(n,)`` bool mask.
    """
    h, w = img_hw
    finite = torch.isfinite(xyxy).all(-1)
    area = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(
        min=0
    )
    clipped = torch.stack(
        [
            xyxy[:, 0].clamp(0, w - 1),
            xyxy[:, 1].clamp(0, h - 1),
            xyxy[:, 2].clamp(0, w - 1),
            xyxy[:, 3].clamp(0, h - 1),
        ],
        dim=-1,
    )
    cw = clipped[:, 2] - clipped[:, 0]
    ch = clipped[:, 3] - clipped[:, 1]
    keep = (
        finite
        & (cw >= min_size)
        & (ch >= min_size)
        & (cw * ch >= min_visible_fraction * area)
    )
    return clipped, keep


# ---------------------------------------------------------------------------
# CenterNet targets
# ---------------------------------------------------------------------------


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet's size-adaptive Gaussian radius for a ``height x width`` box.

    The smallest of the three radii at which shifting the box's corners keeps IoU
    >= ``min_overlap`` with the original (CornerNet, Law & Deng 2018), exactly as in
    the CenterNet reference implementation.
    """
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / 2
    return min(r1, r2, r3)


def draw_gaussian(
    heatmap: torch.Tensor, cx: float, cy: float, sigma: float
) -> torch.Tensor:
    """Max-merge an unnormalized Gaussian (peak 1) centered at ``(cx, cy)``, in place.

    Args:
        heatmap: ``(H, W)`` map.
        cx: Center x in heatmap cells (an integer cell gives an exact peak of 1).
        cy: Center y in heatmap cells.
        sigma: Standard deviation in cells.
    """
    h, w = heatmap.shape
    radius = max(int(math.ceil(3 * sigma)), 1)
    x0, x1 = max(int(cx) - radius, 0), min(int(cx) + radius + 1, w)
    y0, y1 = max(int(cy) - radius, 0), min(int(cy) + radius + 1, h)
    if x0 >= x1 or y0 >= y1:
        return heatmap
    xs = torch.arange(x0, x1, dtype=heatmap.dtype)
    ys = torch.arange(y0, y1, dtype=heatmap.dtype)
    g = torch.exp(-((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], g)
    return heatmap


class CenterNetTargetEncoder:
    """Frame boxes -> CenterNet training targets.

    For each box, the center cell ``floor(center / stride)`` gets a Gaussian peak
    of 1 in its class's heatmap channel, the box size in grid cells in the size
    map, and the center's sub-cell position in the offset map; ``reg_mask`` marks
    the center cells (where size/offset are supervised). Two boxes whose centers
    fall in the same cell: the later one wins the regression targets (as in
    CenterNet).

    Attributes:
        num_classes: Heatmap channels.
        output_stride: Heatmap stride relative to the input.
        sigma: Fixed Gaussian sigma in input pixels, or ``None`` for the
            size-adaptive radius (:func:`gaussian_radius`).
        min_overlap: IoU for the size-adaptive radius.
    """

    def __init__(
        self,
        num_classes: int,
        output_stride: int,
        sigma: Optional[float] = None,
        min_overlap: float = 0.7,
    ):
        """Initialize the encoder."""
        self.num_classes = int(num_classes)
        self.output_stride = int(output_stride)
        self.sigma = sigma
        self.min_overlap = float(min_overlap)

    def __call__(
        self, boxes: torch.Tensor, classes: torch.Tensor, img_hw: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Encode ``(n, 4)`` boxes / ``(n,)`` classes for an ``img_hw`` input.

        Returns:
            ``center_heatmap (C, h, w)``, ``size_target (2, h, w)``,
            ``offset_target (2, h, w)`` and ``reg_mask (1, h, w)`` with
            ``h, w = ceil(img_hw / output_stride)``.
        """
        s = self.output_stride
        h, w = math.ceil(img_hw[0] / s), math.ceil(img_hw[1] / s)
        heatmap = torch.zeros((self.num_classes, h, w), dtype=torch.float32)
        size = torch.zeros((2, h, w), dtype=torch.float32)
        offset = torch.zeros((2, h, w), dtype=torch.float32)
        reg_mask = torch.zeros((1, h, w), dtype=torch.float32)
        for box, cls in zip(boxes.tolist(), classes.tolist()):
            x1, y1, x2, y2 = (v / s for v in box)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ix, iy = int(cx), int(cy)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            if self.sigma is not None:
                sigma = self.sigma / s
            else:
                radius = max(0, int(gaussian_radius(bh, bw, self.min_overlap)))
                sigma = (2 * radius + 1) / 6
            draw_gaussian(heatmap[cls], ix, iy, max(sigma, 1e-3))
            size[:, iy, ix] = torch.tensor([bw, bh])
            offset[:, iy, ix] = torch.tensor([cx - ix, cy - iy])
            reg_mask[0, iy, ix] = 1.0
        return {
            "center_heatmap": heatmap,
            "size_target": size,
            "offset_target": offset,
            "reg_mask": reg_mask,
        }


# ---------------------------------------------------------------------------
# SSD anchors, matching and box coding
# ---------------------------------------------------------------------------


def anchor_shapes(
    size: float, scales: Sequence[float], aspect_ratios: Sequence[float]
) -> List[Tuple[float, float]]:
    """``(w, h)`` of one level's anchors: every scale x aspect ratio (``h / w``)."""
    shapes = []
    for scale in scales:
        area = (size * scale) ** 2
        for ratio in aspect_ratios:
            w = math.sqrt(area / ratio)
            shapes.append((w, w * ratio))
    return shapes


def make_anchors(
    img_hw: Tuple[int, int],
    strides: Sequence[int],
    sizes: Sequence[float],
    scales: Sequence[float],
    aspect_ratios: Sequence[float],
) -> torch.Tensor:
    """All anchors of an ``img_hw`` input as ``(A, 4)`` xyxy, in head output order.

    Order: level (finest first), then row, column, anchor shape -- the order the
    SSD head flattens its per-level ``(B, A_l * K, h, w)`` maps in
    (:class:`sleap_nn.architectures.heads.SSDHead`). Anchor centers sit at
    ``(index + 0.5) * stride``.
    """
    all_anchors = []
    for stride, size in zip(strides, sizes):
        h, w = math.ceil(img_hw[0] / stride), math.ceil(img_hw[1] / stride)
        shapes = torch.tensor(
            anchor_shapes(size, scales, aspect_ratios), dtype=torch.float32
        )  # (a, 2)
        cy, cx = torch.meshgrid(
            (torch.arange(h, dtype=torch.float32) + 0.5) * stride,
            (torch.arange(w, dtype=torch.float32) + 0.5) * stride,
            indexing="ij",
        )
        centers = torch.stack([cx, cy], -1).reshape(-1, 1, 2)  # (h*w, 1, 2)
        half = shapes[None] / 2  # (1, a, 2)
        anchors = torch.cat([centers - half, centers + half], dim=-1)  # (h*w, a, 4)
        all_anchors.append(anchors.reshape(-1, 4))
    return torch.cat(all_anchors, dim=0)


def encode_boxes(
    boxes: torch.Tensor, anchors: torch.Tensor, variances: Sequence[float]
) -> torch.Tensor:
    """SSD box encoding of matched ``(A, 4)`` xyxy boxes relative to their anchors.

    ``((cx - acx) / aw / v0, (cy - acy) / ah / v1, log(w / aw) / v2, log(h / ah) / v3)``.
    """
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2
    acy = anchors[:, 1] + ah / 2
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3)
    bcx = boxes[:, 0] + bw / 2
    bcy = boxes[:, 1] + bh / 2
    v = variances
    return torch.stack(
        [
            (bcx - acx) / aw / v[0],
            (bcy - acy) / ah / v[1],
            torch.log(bw / aw) / v[2],
            torch.log(bh / ah) / v[3],
        ],
        dim=-1,
    )


def decode_boxes(
    deltas: torch.Tensor, anchors: torch.Tensor, variances: Sequence[float]
) -> torch.Tensor:
    """Inverse of :func:`encode_boxes`: ``(..., A, 4)`` deltas -> xyxy boxes."""
    aw = anchors[..., 2] - anchors[..., 0]
    ah = anchors[..., 3] - anchors[..., 1]
    acx = anchors[..., 0] + aw / 2
    acy = anchors[..., 1] + ah / 2
    v = variances
    cx = acx + deltas[..., 0] * v[0] * aw
    cy = acy + deltas[..., 1] * v[1] * ah
    # Clamp the log-size so an untrained head cannot overflow exp().
    bw = aw * torch.exp((deltas[..., 2] * v[2]).clamp(max=math.log(1000.0)))
    bh = ah * torch.exp((deltas[..., 3] * v[3]).clamp(max=math.log(1000.0)))
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


def match_anchors(
    boxes: torch.Tensor,
    anchors: torch.Tensor,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each anchor to a box (SSD/RetinaNet matching).

    An anchor is positive for the box it overlaps most if that IoU is >=
    ``pos_iou_threshold``, background if its best IoU is < ``neg_iou_threshold``, and
    ignored in between. Each box's highest-IoU anchor is forced positive for it, so
    no box goes unmatched however poorly the anchors fit.

    Returns:
        ``(matched_idx, state)``: ``(A,)`` index of each anchor's box, and ``(A,)``
        state ``1`` positive / ``0`` background / ``-1`` ignore.
    """
    num_anchors = anchors.shape[0]
    if boxes.shape[0] == 0:
        return (
            torch.zeros(num_anchors, dtype=torch.long),
            torch.zeros(num_anchors, dtype=torch.long),
        )
    iou = box_iou(anchors, boxes)  # (A, n)
    best_iou, matched_idx = iou.max(dim=1)
    state = torch.full((num_anchors,), -1, dtype=torch.long)
    state[best_iou < neg_iou_threshold] = 0
    state[best_iou >= pos_iou_threshold] = 1
    best_anchor = iou.argmax(dim=0)  # (n,)
    matched_idx[best_anchor] = torch.arange(boxes.shape[0])
    state[best_anchor] = 1
    return matched_idx, state


class SSDTargetEncoder:
    """Frame boxes -> SSD training targets.

    Attributes:
        num_classes: Foreground classes (logit 0 is background).
        strides: Pyramid level strides, finest first.
        sizes: Base anchor size per level (input pixels).
        scales: Anchor size multipliers.
        aspect_ratios: Anchor ``h / w`` ratios.
        pos_iou_threshold, neg_iou_threshold: Matching thresholds
            (:func:`match_anchors`).
        variances: Box-coding variances (:func:`encode_boxes`).
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int],
        sizes: Sequence[float],
        scales: Sequence[float],
        aspect_ratios: Sequence[float],
        pos_iou_threshold: float = 0.5,
        neg_iou_threshold: float = 0.4,
        variances: Sequence[float] = (0.1, 0.1, 0.2, 0.2),
    ):
        """Initialize the encoder."""
        if len(sizes) != len(strides):
            raise ValueError(
                f"One anchor size per pyramid level is required: got {len(sizes)} "
                f"sizes for {len(strides)} levels ({list(strides)})."
            )
        self.num_classes = int(num_classes)
        self.strides = [int(s) for s in strides]
        self.sizes = [float(s) for s in sizes]
        self.scales = [float(s) for s in scales]
        self.aspect_ratios = [float(r) for r in aspect_ratios]
        self.pos_iou_threshold = float(pos_iou_threshold)
        self.neg_iou_threshold = float(neg_iou_threshold)
        self.variances = [float(v) for v in variances]
        self._anchor_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    @property
    def anchors_per_location(self) -> int:
        """Anchors at each feature-map cell of every level."""
        return len(self.scales) * len(self.aspect_ratios)

    def anchors(self, img_hw: Tuple[int, int]) -> torch.Tensor:
        """``(A, 4)`` anchors of an ``img_hw`` input (cached per size)."""
        key = (int(img_hw[0]), int(img_hw[1]))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = make_anchors(
                key, self.strides, self.sizes, self.scales, self.aspect_ratios
            )
        return self._anchor_cache[key]

    def __call__(
        self, boxes: torch.Tensor, classes: torch.Tensor, img_hw: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Encode ``(n, 4)`` boxes / ``(n,)`` classes for an ``img_hw`` input.

        Returns:
            ``ssd_cls_target (A,)`` long -- ``0`` background, ``k + 1`` class ``k``,
            ``-1`` ignore -- and ``ssd_box_target (A, 4)`` encoded deltas (zero for
            non-positives).
        """
        anchors = self.anchors(img_hw)
        matched_idx, state = match_anchors(
            boxes, anchors, self.pos_iou_threshold, self.neg_iou_threshold
        )
        cls_target = torch.where(
            state == 1,
            classes[matched_idx] + 1 if boxes.shape[0] else state,
            state,
        ).long()
        box_target = torch.zeros((anchors.shape[0], 4), dtype=torch.float32)
        pos = state == 1
        if pos.any():
            box_target[pos] = encode_boxes(
                boxes[matched_idx[pos]], anchors[pos], self.variances
            )
        return {"ssd_cls_target": cls_target, "ssd_box_target": box_target}


def bbox_head_leaf(model_type: str, head_configs):
    """The head config leaf of a ``bbox_*`` model that holds its ``categories``."""
    if model_type == "bbox_centernet":
        return head_configs.bbox_centernet.center
    if model_type == "bbox_ssd":
        return head_configs.bbox_ssd.ssd
    raise ValueError(f"{model_type} is not a bounding-box model type.")


def bbox_head_categories(model_type: str, head_configs) -> List[str]:
    """Ordered class names of a ``bbox_*`` model (``["object"]`` when unset)."""
    categories = bbox_head_leaf(model_type, head_configs).categories
    return list(categories) if categories else [DEFAULT_BBOX_CATEGORY]


def get_bbox_target_encoder(model_type: str, head_configs):
    """The target encoder of a ``bbox_*`` model from its head config leaves."""
    from sleap_nn.config.model_config import ssd_level_strides

    if model_type == "bbox_centernet":
        center = head_configs.bbox_centernet.center
        return CenterNetTargetEncoder(
            num_classes=len(bbox_head_categories(model_type, head_configs)),
            output_stride=center.output_stride,
            sigma=center.sigma,
            min_overlap=center.min_overlap,
        )
    if model_type == "bbox_ssd":
        ssd = head_configs.bbox_ssd.ssd
        return SSDTargetEncoder(
            num_classes=len(bbox_head_categories(model_type, head_configs)),
            strides=ssd_level_strides(ssd),
            sizes=list(ssd.anchor_sizes),
            scales=list(ssd.anchor_scales),
            aspect_ratios=list(ssd.aspect_ratios),
            pos_iou_threshold=ssd.pos_iou_threshold,
            neg_iou_threshold=ssd.neg_iou_threshold,
            variances=list(ssd.box_variances),
        )
    raise ValueError(f"{model_type} is not a bounding-box model type.")
