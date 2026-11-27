"""Decoding for the bounding-box detectors (``bbox_centernet``, ``bbox_ssd``, ``bbox_detr``).

Each detection style has a pure-torch decoder that turns its raw head outputs into
per-frame boxes, scores and class ids in model-input pixels:

* :func:`decode_centernet` -- local maxima of the center heatmap (max-pool NMS), the
  box read from the size/offset maps at each peak.
* :func:`decode_ssd` -- per-anchor class probabilities and decoded deltas,
  thresholded and class-wise IoU-NMSed.
* :func:`decode_set` -- DETR set predictions: sigmoid scores over queries x
  classes, the top ``num_select`` pairs, no NMS. :func:`decode_set_keypoints` also
  returns each detection's keypoints (the experimental keypoint branch).

:class:`BBoxDecoder` binds a style and its parameters (built from a model's head
config) and is what both the LightningModules (val-time eval) and the inference
layer (:class:`sleap_nn.inference.layers.bbox.BBoxLayer`) call, so training metrics
and predictions come from the same code.

Decoded outputs are dense and padded: ``boxes (B, K, 4)`` xyxy, ``scores (B, K)``
and ``classes (B, K)``, sorted by descending score, with NaN boxes/scores and class
``-1`` in unused slots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from sleap_nn.data.bboxes import (
    SSDTargetEncoder,
    bbox_head_categories,
    decode_boxes,
)

# IoU-NMS default per style: SSD's anchors overlap so it needs NMS (0.45 as in the
# SSD paper); CenterNet's max-pool peak NMS already yields one peak per object.
# A DETR's one-to-one matching already yields one query per object.
DEFAULT_NMS_IOU = {"bbox_centernet": None, "bbox_ssd": 0.45, "bbox_detr": None}
DEFAULT_MAX_DETECTIONS = 100

_Decoded = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _pad_detections(
    per_frame: List[_Decoded], max_detections: int, device: torch.device
) -> _Decoded:
    """Stack ragged per-frame detections into padded ``(B, K, ...)`` tensors."""
    k = int(max_detections)
    b = len(per_frame)
    boxes = torch.full((b, k, 4), float("nan"), device=device)
    scores = torch.full((b, k), float("nan"), device=device)
    classes = torch.full((b, k), -1, dtype=torch.long, device=device)
    for i, (bx, sc, cl) in enumerate(per_frame):
        n = min(bx.shape[0], k)
        order = sc.argsort(descending=True)[:n]
        boxes[i, :n] = bx[order]
        scores[i, :n] = sc[order]
        classes[i, :n] = cl[order]
    return boxes, scores, classes


def _clip_to_image(boxes: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = img_hw
    return torch.stack(
        [
            boxes[..., 0].clamp(0, w - 1),
            boxes[..., 1].clamp(0, h - 1),
            boxes[..., 2].clamp(0, w - 1),
            boxes[..., 3].clamp(0, h - 1),
        ],
        dim=-1,
    )


def clip_boxes_to_frame(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    frame_hw: Union[Tuple[int, int], torch.Tensor],
) -> _Decoded:
    """Clip padded ``(B, K, 4)`` boxes to the original frame.

    The decoder clips to the padded model input; mapped back to the frame, a box
    can still reach into the stride padding or the size-matcher's letterbox. Boxes
    left with no area (predicted wholly outside the frame) become empty slots
    (NaN box and score, class ``-1``).

    ``frame_hw`` is one ``(height, width)`` for the batch, or a ``(B, 2)`` tensor
    of each frame's (a batch of frames of different sizes fitted to the model
    input before batching).
    """
    if isinstance(frame_hw, torch.Tensor):
        hw = frame_hw.to(device=boxes.device, dtype=boxes.dtype)
        h, w = hw[:, 0, None] - 1, hw[:, 1, None] - 1
    else:
        h, w = frame_hw[0] - 1, frame_hw[1] - 1
    clipped = torch.stack(
        [
            torch.clamp(boxes[..., 0], min=0).clamp(max=w),
            torch.clamp(boxes[..., 1], min=0).clamp(max=h),
            torch.clamp(boxes[..., 2], min=0).clamp(max=w),
            torch.clamp(boxes[..., 3], min=0).clamp(max=h),
        ],
        dim=-1,
    )
    valid = (
        (clipped[..., 2] > clipped[..., 0])
        & (clipped[..., 3] > clipped[..., 1])
        & torch.isfinite(scores)
    )
    clipped = clipped.masked_fill(~valid[..., None], float("nan"))
    scores = scores.masked_fill(~valid, float("nan"))
    classes = classes.masked_fill(~valid, -1)
    return clipped, scores, classes


def _nms(boxes, scores, classes, iou_threshold: Optional[float]):
    # Boxes clipped to nothing (predicted wholly outside the frame) are dropped.
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if iou_threshold is None or boxes.shape[0] == 0:
        return boxes, scores, classes
    keep = batched_nms(boxes.float(), scores.float(), classes, float(iou_threshold))
    return boxes[keep], scores[keep], classes[keep]


def decode_centernet(
    heatmap: torch.Tensor,
    size: torch.Tensor,
    offset: Optional[torch.Tensor],
    output_stride: int,
    score_threshold: float = 0.2,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    nms_kernel: int = 3,
    iou_threshold: Optional[float] = None,
    img_hw: Optional[Tuple[int, int]] = None,
) -> _Decoded:
    """Decode CenterNet head outputs into boxes.

    Args:
        heatmap: ``(B, C, h, w)`` center probabilities (sigmoid applied).
        size: ``(B, 2, h, w)`` box (width, height) in grid cells.
        offset: ``(B, 2, h, w)`` sub-cell center offsets, or ``None``.
        output_stride: Grid stride in input pixels.
        score_threshold: Minimum peak probability.
        max_detections: Peaks kept per frame (top-k before thresholding).
        nms_kernel: Max-pool window of the peak NMS.
        iou_threshold: Optional class-wise IoU-NMS on top of the peak NMS.
        img_hw: Model-input ``(H, W)`` to clip boxes to, or ``None``.

    Returns:
        Padded ``(boxes, scores, classes)`` in input pixels.
    """
    b, c, h, w = heatmap.shape
    pooled = F.max_pool2d(heatmap, nms_kernel, stride=1, padding=nms_kernel // 2)
    peaks = heatmap * (pooled == heatmap).to(heatmap.dtype)
    k = min(int(max_detections), c * h * w)
    scores, inds = peaks.reshape(b, -1).topk(k, dim=1)
    classes = inds // (h * w)
    cell = inds % (h * w)
    ys = (cell // w).to(heatmap.dtype)
    xs = (cell % w).to(heatmap.dtype)

    def gather(m):  # (B, 2, h, w) -> (B, k, 2)
        return m.reshape(b, 2, -1).gather(2, cell[:, None, :].expand(b, 2, k))

    wh = gather(size).permute(0, 2, 1).clamp(min=0)
    if offset is not None:
        off = gather(offset).permute(0, 2, 1)
        xs, ys = xs + off[..., 0], ys + off[..., 1]
    s = float(output_stride)
    boxes = torch.stack(
        [
            (xs - wh[..., 0] / 2) * s,
            (ys - wh[..., 1] / 2) * s,
            (xs + wh[..., 0] / 2) * s,
            (ys + wh[..., 1] / 2) * s,
        ],
        dim=-1,
    )
    if img_hw is not None:
        boxes = _clip_to_image(boxes, img_hw)

    per_frame = []
    for i in range(b):
        keep = scores[i] >= score_threshold
        per_frame.append(
            _nms(boxes[i][keep], scores[i][keep], classes[i][keep], iou_threshold)
        )
    return _pad_detections(per_frame, k, heatmap.device)


def decode_ssd(
    raw: torch.Tensor,
    anchors: torch.Tensor,
    variances: Sequence[float],
    score_threshold: float = 0.2,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    iou_threshold: Optional[float] = 0.45,
    pre_nms_top_k: int = 1000,
    img_hw: Optional[Tuple[int, int]] = None,
) -> _Decoded:
    """Decode SSD head outputs into boxes.

    Args:
        raw: ``(B, A, K + 4)`` head output (``K`` class logits incl. background).
        anchors: ``(A, 4)`` anchors in input pixels, in head output order.
        variances: Box-coding variances.
        score_threshold: Minimum class probability of a detection.
        max_detections: Detections kept per frame after NMS.
        iou_threshold: Class-wise IoU-NMS threshold (``None`` disables it).
        pre_nms_top_k: Highest-scoring (anchor, class) pairs entering NMS.
        img_hw: Model-input ``(H, W)`` to clip boxes to, or ``None``.

    Returns:
        Padded ``(boxes, scores, classes)`` in input pixels.
    """
    num_logits = raw.shape[-1] - 4
    probs = torch.softmax(raw[..., :num_logits].float(), dim=-1)[..., 1:]  # (B, A, C)
    boxes = decode_boxes(
        raw[..., num_logits:].float(), anchors.to(raw.device), variances
    )
    if img_hw is not None:
        boxes = _clip_to_image(boxes, img_hw)
    num_classes = probs.shape[-1]
    per_frame = []
    for i in range(raw.shape[0]):
        flat = probs[i].reshape(-1)  # (A * C,) anchor-major
        cand = (flat >= score_threshold).nonzero(as_tuple=True)[0]
        if cand.numel() > pre_nms_top_k:
            cand = cand[flat[cand].topk(pre_nms_top_k).indices]
        anchor_idx, cls = cand // num_classes, cand % num_classes
        bx, sc, cl = _nms(boxes[i][anchor_idx], flat[cand], cls, iou_threshold)
        per_frame.append(
            (bx[:max_detections], sc[:max_detections], cl[:max_detections])
        )
    return _pad_detections(per_frame, max_detections, raw.device)


def _set_boxes(
    pred_boxes: torch.Tensor, queries: torch.Tensor, img_hw: Tuple[int, int]
) -> torch.Tensor:
    """The ``(B, k)`` selected queries' boxes as ``xyxy`` input pixels, clipped."""
    b, k = queries.shape
    boxes = pred_boxes.float().gather(1, queries[..., None].expand(b, k, 4))
    cx, cy, w, h = boxes.unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    img_h, img_w = img_hw
    return _clip_to_image(
        boxes * boxes.new_tensor([img_w, img_h, img_w, img_h]), img_hw
    )


def _select_set_detections_batched(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    score_threshold: float,
    max_detections: int,
    iou_threshold: Optional[float],
    num_classes: int,
    img_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized batch selection + single-call batched NMS for DETR decoding.

    Replaces the per-frame ``nonzero()`` + ``batched_nms()`` loop with:
    - zero host syncs when ``iou_threshold is None``;
    - a single batch-wide ``batched_nms`` using ``b * num_classes + cls``
      offsets when ``iou_threshold is not None``.

    Returns:
        ``(sel, sel_valid)`` of shape ``(B, m_eff)`` where ``m_eff = min(k, m)``.
    """
    b, k = scores.shape
    m = int(max_detections)
    m_eff = min(k, m)
    if m_eff == 0:
        return (
            torch.empty((b, 0), dtype=torch.long, device=boxes.device),
            torch.empty((b, 0), dtype=torch.bool, device=boxes.device),
        )

    valid = (
        (scores >= score_threshold)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )
    if iou_threshold is not None:
        b_idx, j_idx = valid.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            kept_mask = valid
        else:
            flat_boxes = boxes[b_idx, j_idx].float()
            flat_scores = scores[b_idx, j_idx].float()
            flat_classes = classes[b_idx, j_idx].long()
            idxs = b_idx * max(1, int(num_classes)) + flat_classes
            # Normalize boxes to [0, 1] before batched_nms so its internal
            # ``idxs * (max_coord + 1)`` offset keeps full float32 precision
            # across large batches and many classes (IoU is scale-invariant).
            box_scale = float(max(img_hw[0], img_hw[1], 1))
            keep_flat = batched_nms(
                flat_boxes / box_scale,
                flat_scores,
                idxs,
                float(iou_threshold),
            )
            kept_mask = torch.zeros_like(valid)
            kept_mask[b_idx[keep_flat], j_idx[keep_flat]] = True
    else:
        kept_mask = valid

    sort_scores = scores.float().masked_fill(~kept_mask, float("-inf"))
    _, order = sort_scores.sort(dim=1, descending=True, stable=True)
    sel = order[:, :m_eff]
    sel_valid = kept_mask.gather(1, sel)
    return sel, sel_valid


def _gather_padded_batched(
    values: torch.Tensor,
    sel: torch.Tensor,
    sel_valid: torch.Tensor,
    m: int,
    fill,
) -> torch.Tensor:
    """Gather ``(B, k, ...)`` at vectorized ``sel`` indices, padded to ``m`` slots."""
    b = values.shape[0]
    m_eff = sel.shape[1]
    extra = values.shape[2:]
    if m_eff == 0:
        return torch.full(
            (b, m, *extra), fill, dtype=values.dtype, device=values.device
        )
    view_shape = (b, m_eff) + (1,) * len(extra)
    idx = sel.reshape(view_shape).expand(b, m_eff, *extra)
    mask = sel_valid.reshape(view_shape)
    gathered = values.gather(1, idx).masked_fill(~mask, fill)
    if m_eff == m:
        return gathered
    out = torch.full((b, m, *extra), fill, dtype=values.dtype, device=values.device)
    out[:, :m_eff] = gathered
    return out


def decode_set(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    img_hw: Tuple[int, int],
    score_threshold: float = 0.2,
    num_select: int = 300,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    iou_threshold: Optional[float] = None,
) -> _Decoded:
    """Decode DETR set predictions into boxes (RF-DETR's ``PostProcess``).

    Scores are the sigmoid of every (query, real class) logit; the extra last logit
    of the ``num_classes + 1`` is dropped. The ``num_select`` highest-scoring
    pairs are kept (one query can yield several classes), then thresholded. Boxes
    are normalized ``cxcywh`` relative to the model input and come out as ``xyxy``
    in its pixels.

    Args:
        pred_logits: ``(B, Q, C + 1)`` class logits.
        pred_boxes: ``(B, Q, 4)`` normalized ``cxcywh`` boxes.
        img_hw: Model-input ``(H, W)``; boxes are scaled to it and clipped.
        score_threshold: Minimum detection score.
        num_select: (query, class) pairs kept before thresholding.
        max_detections: Detections kept per frame.
        iou_threshold: Optional class-wise IoU-NMS (``None``, the default, keeps
            DETR's NMS-free output).

    Returns:
        Padded ``(boxes, scores, classes)`` in input pixels.
    """
    probs = pred_logits[..., :-1].float().sigmoid()
    b, q, c = probs.shape
    k = min(int(num_select), q * c)
    scores, inds = probs.reshape(b, -1).topk(k, dim=1)
    classes = inds % c
    boxes = _set_boxes(pred_boxes, inds // c, img_hw)
    sel, sel_valid = _select_set_detections_batched(
        boxes,
        scores,
        classes,
        score_threshold,
        max_detections,
        iou_threshold,
        num_classes=c,
        img_hw=img_hw,
    )
    m = int(max_detections)
    return (
        _gather_padded_batched(boxes, sel, sel_valid, m, float("nan")),
        _gather_padded_batched(scores, sel, sel_valid, m, float("nan")),
        _gather_padded_batched(classes, sel, sel_valid, m, -1),
    )


def decode_set_keypoints(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_keypoints: torch.Tensor,
    pred_keypoint_logits: torch.Tensor,
    img_hw: Tuple[int, int],
    score_threshold: float = 0.2,
    num_select: int = 300,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    iou_threshold: Optional[float] = None,
    visibility_threshold: float = 0.5,
) -> Tuple[torch.Tensor, ...]:
    """Decode a keypoint ``bbox_detr``'s set predictions: one detection per query.

    Unlike :func:`decode_set`, a query yields at most one detection, of its
    highest-scoring class, so each animal gives one instance (its points are the
    query's). The ``num_select`` best queries are then thresholded, filtered and
    capped exactly as in :func:`decode_set`; with one class the detections are
    the same.

    Args:
        pred_logits: ``(B, Q, C + 1)`` class logits.
        pred_boxes: ``(B, Q, 4)`` normalized ``cxcywh`` boxes.
        pred_keypoints: ``(B, Q, K, 2)`` normalized ``xy`` points.
        pred_keypoint_logits: ``(B, Q, K)`` visibility logits.
        img_hw: Model-input ``(H, W)``.
        score_threshold: Minimum detection score.
        num_select: Queries kept before thresholding.
        max_detections: Detections kept per frame.
        iou_threshold: Optional class-wise IoU-NMS.
        visibility_threshold: Points whose visibility probability is below this
            are NaN.

    Returns:
        ``(boxes, scores, classes, points, point_scores)``: as
        :func:`decode_set`, plus ``points (B, M, K, 2)`` in input pixels and
        ``point_scores (B, M, K)`` (visibility probabilities), NaN in unused
        slots.
    """
    probs = pred_logits[..., :-1].float().sigmoid()
    b, q, c = probs.shape
    n_kpts = pred_keypoints.shape[-2]
    best, best_class = probs.max(-1)
    k = min(int(num_select), q)
    scores, queries = best.topk(k, dim=1)
    classes = best_class.gather(1, queries)
    boxes = _set_boxes(pred_boxes, queries, img_hw)
    img_h, img_w = img_hw
    points = (
        pred_keypoints.float().gather(
            1, queries[..., None, None].expand(b, k, n_kpts, 2)
        )
        * pred_keypoints.new_tensor([img_w, img_h]).float()
    )
    vis = (
        pred_keypoint_logits.float()
        .sigmoid()
        .gather(1, queries[..., None].expand(b, k, n_kpts))
    )
    points = points.masked_fill((vis < visibility_threshold)[..., None], float("nan"))
    sel, sel_valid = _select_set_detections_batched(
        boxes,
        scores,
        classes,
        score_threshold,
        max_detections,
        iou_threshold,
        num_classes=c,
        img_hw=img_hw,
    )
    m = int(max_detections)
    return (
        _gather_padded_batched(boxes, sel, sel_valid, m, float("nan")),
        _gather_padded_batched(scores, sel, sel_valid, m, float("nan")),
        _gather_padded_batched(classes, sel, sel_valid, m, -1),
        _gather_padded_batched(points, sel, sel_valid, m, float("nan")),
        _gather_padded_batched(vis, sel, sel_valid, m, float("nan")),
    )


class BBoxDecoder:
    """A detection style's decoder bound to its parameters.

    Attributes:
        model_type: ``"bbox_centernet"``, ``"bbox_ssd"`` or ``"bbox_detr"``.
        categories: Ordered class names (class ids index into it).
        output_stride: CenterNet grid stride.
        score_threshold: Minimum detection score.
        iou_threshold: IoU-NMS threshold (``None`` disables it).
        max_detections: Detections kept per frame.
        anchor_encoder: SSD only -- the target encoder that owns the anchors.
        num_select: DETR only -- (query, class) pairs kept before thresholding.
        num_keypoints: DETR keypoint branch (experimental) -- keypoints per
            detection; ``0`` for a box detector.
        visibility_threshold: DETR keypoint branch -- points whose visibility
            probability is below this are written as missing.
    """

    num_keypoints = 0
    visibility_threshold = 0.5

    def __init__(
        self,
        model_type: str,
        categories: Sequence[str],
        output_stride: int = 1,
        score_threshold: float = 0.2,
        iou_threshold: Optional[float] = None,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        anchor_encoder: Optional[SSDTargetEncoder] = None,
        num_select: int = 300,
        num_keypoints: int = 0,
        visibility_threshold: float = 0.5,
    ):
        """Initialize the decoder."""
        if model_type not in DEFAULT_NMS_IOU:
            raise ValueError(f"{model_type} is not a bounding-box model type.")
        if model_type == "bbox_ssd" and anchor_encoder is None:
            raise ValueError("An SSD decoder needs its anchor_encoder.")
        self.model_type = model_type
        self.categories = list(categories)
        self.output_stride = int(output_stride)
        self.score_threshold = float(score_threshold)
        self.iou_threshold = iou_threshold
        self.max_detections = int(max_detections)
        self.anchor_encoder = anchor_encoder
        self.num_select = int(num_select)
        if num_keypoints:  # A box detector keeps the class defaults below.
            self.num_keypoints = int(num_keypoints)
            self.visibility_threshold = float(visibility_threshold)

    @classmethod
    def from_head_configs(
        cls,
        model_type: str,
        head_configs,
        score_threshold: float = 0.2,
        iou_threshold: Union[float, str, None] = "default",
        max_detections: Optional[int] = None,
    ) -> "BBoxDecoder":
        """Build the decoder of a trained model from its ``head_configs``.

        ``iou_threshold="default"`` picks the style's default (:data:`DEFAULT_NMS_IOU`).
        """
        from sleap_nn.data.bboxes import get_bbox_target_encoder

        if iou_threshold == "default":
            iou_threshold = DEFAULT_NMS_IOU[model_type]
        anchor_encoder, output_stride, num_select = None, 1, 300
        keypoint_kwargs = {}
        if model_type == "bbox_ssd":
            anchor_encoder = get_bbox_target_encoder(model_type, head_configs)
        elif model_type == "bbox_detr":
            from sleap_nn.architectures.detr.model import (
                detr_keypoints_config,
                detr_num_keypoints,
            )

            detr = head_configs.bbox_detr.detr
            num_select = detr.num_select
            keypoints = detr_keypoints_config(detr)
            if keypoints is not None:
                keypoint_kwargs = dict(
                    num_keypoints=detr_num_keypoints(detr),
                    visibility_threshold=float(keypoints.visibility_threshold),
                )
        else:
            output_stride = head_configs.bbox_centernet.center.output_stride
        return cls(
            model_type=model_type,
            categories=bbox_head_categories(model_type, head_configs),
            output_stride=output_stride,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections or DEFAULT_MAX_DETECTIONS,
            anchor_encoder=anchor_encoder,
            num_select=num_select,
            **keypoint_kwargs,
        )

    def decode_keypoints(
        self,
        outputs: Dict[str, torch.Tensor],
        img_hw: Tuple[int, int],
        score_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Decode a keypoint ``bbox_detr``'s outputs: boxes and their keypoints.

        Returns:
            ``(boxes, scores, classes, points, point_scores)`` in model-input
            pixels (:func:`decode_set_keypoints`).
        """
        thr = self.score_threshold if score_threshold is None else score_threshold
        k = self.max_detections if max_detections is None else int(max_detections)
        return decode_set_keypoints(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_keypoints"],
            outputs["pred_keypoint_logits"],
            img_hw=img_hw,
            score_threshold=thr,
            num_select=self.num_select,
            max_detections=k,
            iou_threshold=self.iou_threshold,
            visibility_threshold=self.visibility_threshold,
        )

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        img_hw: Tuple[int, int],
        score_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> _Decoded:
        """Decode a batch of head outputs (probabilities for CenterNet's heatmap).

        Args:
            outputs: The model's head-name -> tensor dict. CenterNet expects
                ``BBoxCenterHead`` as probabilities (the module ``forward`` applies
                the sigmoid); SSD expects the raw ``SSDHead`` tensor; DETR expects
                ``pred_logits`` and ``pred_boxes``.
            img_hw: Model-input ``(H, W)``.
            score_threshold: Per-call override of the score threshold.
            max_detections: Per-call override of the detections kept per frame.

        Returns:
            Padded ``(boxes, scores, classes)`` in model-input pixels.
        """
        thr = self.score_threshold if score_threshold is None else score_threshold
        k = self.max_detections if max_detections is None else int(max_detections)
        if self.model_type == "bbox_centernet":
            return decode_centernet(
                outputs["BBoxCenterHead"],
                outputs["BBoxSizeHead"],
                outputs.get("BBoxOffsetHead"),
                output_stride=self.output_stride,
                score_threshold=thr,
                max_detections=k,
                iou_threshold=self.iou_threshold,
                img_hw=img_hw,
            )
        if self.model_type == "bbox_detr":
            return decode_set(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                img_hw=img_hw,
                score_threshold=thr,
                num_select=self.num_select,
                max_detections=k,
                iou_threshold=self.iou_threshold,
            )
        return decode_ssd(
            outputs["SSDHead"],
            self.anchor_encoder.anchors(img_hw),
            self.anchor_encoder.variances,
            score_threshold=thr,
            max_detections=k,
            iou_threshold=self.iou_threshold,
            img_hw=img_hw,
        )


class BBoxInferenceModel(L.LightningModule):
    """Inference model of a ``bbox_*`` detector (an attribute bag, as for segmentation).

    Holds the trained module and its :class:`BBoxDecoder`; the inference run itself
    is driven by :class:`sleap_nn.inference.layers.bbox.BBoxLayer`, built from these
    attributes by the predictor factory.

    Attributes:
        torch_model: The trained LightningModule (its ``forward`` returns the head
            dict, with the CenterNet heatmap as probabilities).
        decoder: The style's :class:`BBoxDecoder`.
        input_scale: Input scale the model was trained with.
        output_stride: Finest head stride.
    """

    def __init__(
        self,
        torch_model,
        decoder: BBoxDecoder,
        input_scale: float = 1.0,
        output_stride: int = 1,
    ):
        """Stash the model and its decoder."""
        super().__init__()
        self.torch_model = torch_model
        self.decoder = decoder
        self.input_scale = float(input_scale)
        self.output_stride = int(output_stride)

    @property
    def peak_threshold(self) -> float:
        """The decoder's score threshold (the predictor's ``peak_threshold``)."""
        return self.decoder.score_threshold
