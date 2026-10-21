"""Decoding for the bounding-box detectors (``bbox_centernet``, ``bbox_ssd``).

Each detection style has a pure-torch decoder that turns its raw head outputs into
per-frame boxes, scores and class ids in model-input pixels:

* :func:`decode_centernet` -- local maxima of the center heatmap (max-pool NMS), the
  box read from the size/offset maps at each peak.
* :func:`decode_ssd` -- per-anchor class probabilities and decoded deltas,
  thresholded and class-wise IoU-NMSed.

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
DEFAULT_NMS_IOU = {"bbox_centernet": None, "bbox_ssd": 0.45}
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
    frame_hw: Tuple[int, int],
) -> _Decoded:
    """Clip padded ``(B, K, 4)`` boxes to the original frame.

    The decoder clips to the padded model input; mapped back to the frame, a box
    can still reach into the stride padding or the size-matcher's letterbox. Boxes
    left with no area (predicted wholly outside the frame) become empty slots
    (NaN box and score, class ``-1``).
    """
    h, w = frame_hw
    clipped = torch.stack(
        [
            boxes[..., 0].clamp(0, w - 1),
            boxes[..., 1].clamp(0, h - 1),
            boxes[..., 2].clamp(0, w - 1),
            boxes[..., 3].clamp(0, h - 1),
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


class BBoxDecoder:
    """A detection style's decoder bound to its parameters.

    Attributes:
        model_type: ``"bbox_centernet"`` or ``"bbox_ssd"``.
        categories: Ordered class names (class ids index into it).
        output_stride: CenterNet grid stride.
        score_threshold: Minimum detection score.
        iou_threshold: IoU-NMS threshold (``None`` disables it).
        max_detections: Detections kept per frame.
        anchor_encoder: SSD only -- the target encoder that owns the anchors.
    """

    def __init__(
        self,
        model_type: str,
        categories: Sequence[str],
        output_stride: int = 1,
        score_threshold: float = 0.2,
        iou_threshold: Optional[float] = None,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        anchor_encoder: Optional[SSDTargetEncoder] = None,
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
        anchor_encoder, output_stride = None, 1
        if model_type == "bbox_ssd":
            anchor_encoder = get_bbox_target_encoder(model_type, head_configs)
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
                the sigmoid); SSD expects the raw ``SSDHead`` tensor.
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
