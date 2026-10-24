"""``BBoxLayer`` — whole-frame bounding-box detection inference.

Wraps a trained ``bbox_centernet`` / ``bbox_ssd`` LightningModule (whose ``forward``
returns the decoder-ready head dict) and decodes its outputs with the model's
:class:`~sleap_nn.inference.bbox.BBoxDecoder` into ``Outputs.pred_bboxes`` /
``pred_bbox_scores`` / ``pred_bbox_classes`` in original-image pixels, which
``Outputs.to_labels`` packages as ``sio.PredictedBoundingBox`` on
``LabeledFrame.bboxes``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sleap_nn.inference.bbox import BBoxDecoder, clip_boxes_to_frame
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.coord import undo_eff_scale, undo_input_scale
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


class BBoxLayer(InferenceLayer):
    """Bounding-box detection prediction layer (CenterNet- or SSD-style).

    Args:
        backend: Runtime backend wrapping the detector's Lightning module.
        decoder: The model's :class:`BBoxDecoder` (style, anchors, NMS).
        output_stride: Finest head stride relative to the model input.
        max_stride: Backbone max stride (the input is padded to a multiple of it).
        preprocess_config: Standard preprocessing knobs.
        postprocess_config: ``peak_threshold`` is the detection score threshold
            and ``max_instances`` caps detections per frame (both overridable via
            ``Predictor.predict(peak_threshold=..., max_instances=...)``).
    """

    def __init__(
        self,
        backend: ModelBackend,
        decoder: BBoxDecoder,
        output_stride: int,
        max_stride: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Store the decoder and standard layer configs."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
            max_stride=max_stride,
        )
        self.decoder = decoder

    @property
    def categories(self) -> List[str]:
        """Ordered class names (``Outputs.pred_bbox_classes`` indexes into it)."""
        return self.decoder.categories

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode the head outputs into boxes in original-image pixels.

        Args:
            raw_out: Backend output dict (the module's decoder-ready head dict).
            info: Preprocessing metadata to map boxes back to the original frame.

        Returns:
            ``Outputs`` with ``pred_bboxes (B, K, 4)`` xyxy, ``pred_bbox_scores
            (B, K)`` and ``pred_bbox_classes (B, K)`` (NaN / ``-1`` padded, sorted
            by descending score).
        """
        boxes, scores, classes = self.decoder(
            raw_out,
            img_hw=tuple(info.processed_size),
            score_threshold=self.postprocess_config.peak_threshold,
            max_detections=self.postprocess_config.max_instances,
        )
        # Coord ladder on the two corners: model input -> original image.
        corners = boxes.reshape(boxes.shape[0], -1, 2)
        corners = undo_input_scale(corners, info.input_scale)
        corners = undo_eff_scale(corners, info.eff_scale)
        boxes = corners.reshape(boxes.shape)
        boxes, scores, classes = clip_boxes_to_frame(
            boxes, scores, classes, info.original_size
        )
        return Outputs(
            pred_bboxes=boxes.detach(),
            pred_bbox_scores=scores.detach(),
            pred_bbox_classes=classes.detach(),
            preprocess_info=info,
        )


class ExportedBBoxLayer(BBoxLayer):
    """``BBoxLayer`` for an ONNX/TRT-exported detector.

    The exported graph (:class:`~sleap_nn.export.wrappers.bbox.BBoxONNXWrapper`)
    holds only the network: it maps a model-input-sized uint8 frame to the
    decoder-ready head outputs. Everything else is this layer's -- the same
    ``preprocess`` chain as the checkpoint path (channel coercion, size matcher,
    ``input_scale`` resize, stride padding; a resize traced into the graph would be
    frozen to the trace input's size) and the same ``postprocess`` (decoder, coord
    ladder, clipping) -- so exported and native boxes match by construction. Only
    the frames handed to the backend differ: rounded to uint8 and without the
    ``n_samples`` axis, as the graph expects.
    """

    def preprocess(self, image):
        """Native preprocessing, emitted as the graph's ``(B, C, H, W)`` uint8 input."""
        x = self._to_4d_tensor(image).float()
        x, eff_scale, orig_hw = self._apply_full_preprocess(
            x, max_stride=self.max_stride, unsqueeze_n_samples=False
        )
        info = PreprocInfo(
            original_size=orig_hw,
            processed_size=tuple(x.shape[-2:]),
            eff_scale=eff_scale,
            input_scale=self.preprocess_config.scale,
            output_stride=self.output_stride,
        )
        return x.round().clamp(0, 255).to(torch.uint8), info


class BBoxStageLayer(BBoxLayer):
    """A box detector as the first stage of a top-down layer.

    Stands in for :class:`~sleap_nn.inference.layers.centroid.CentroidLayer` inside
    :class:`~sleap_nn.inference.layers.topdown.TopDownLayer` (and its multi-class /
    segmentation / embedding subclasses): ``predict`` returns each detection's box
    center as ``pred_centroids`` and its score as ``pred_centroid_values``, plus the
    boxes themselves (``pred_bboxes`` / ``pred_bbox_scores`` / ``pred_bbox_classes``)
    for box-scaled crops (``preprocess.crop_mode="bbox"``) and for packaging. The
    detector's ``peak_threshold`` (score threshold) and ``max_instances`` play the
    centroid stage's roles, so the per-stage threshold overrides route to it
    unchanged.

    With ``use_gt_boxes=True`` no detector runs: boxes are derived from the
    ground-truth instances passed to ``predict`` (the tight box of each instance's
    visible keypoints, padded by ``gt_bbox_padding`` pixels, as the centered-instance
    model's training boxes were), the box analog of the GT-centroid fallback.

    With ``use_given_boxes=True`` no detector runs either: each frame's boxes and
    scores are handed to ``predict`` (``Batch.frame_boxes``, e.g. the
    ``PredictedBoundingBox``es of an input ``.slp`` or a COCO detection file --
    ``sleap-nn predict --boxes``). Boxes scoring below ``box_score_threshold``
    are dropped; the rest seed one crop each, as a detector's would.

    Args:
        backend: The detector's backend (with ``use_gt_boxes``, any backend; it is
            never called).
        decoder: The detector's :class:`BBoxDecoder` (``None`` with
            ``use_gt_boxes``).
        output_stride: Finest head stride relative to the model input.
        max_stride: Backbone max stride.
        use_gt_boxes: Read boxes from ground-truth instances instead.
        gt_bbox_padding: Pixels added on every side of ground-truth boxes.
        use_given_boxes: Take each frame's boxes as given to ``predict``.
        box_score_threshold: With ``use_given_boxes``, drop boxes scoring below
            this.
        preprocess_config / postprocess_config: As for :class:`BBoxLayer`.
    """

    def __init__(
        self,
        backend: ModelBackend,
        decoder: Optional[BBoxDecoder],
        output_stride: int = 1,
        max_stride: int = 1,
        use_gt_boxes: bool = False,
        gt_bbox_padding: float = 0.0,
        use_given_boxes: bool = False,
        box_score_threshold: float = 0.0,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Store the detector (or the GT/given-box source) and its configs."""
        if use_gt_boxes and use_given_boxes:
            raise ValueError("use_gt_boxes and use_given_boxes are exclusive.")
        super().__init__(
            backend=backend,
            decoder=decoder,
            output_stride=output_stride,
            max_stride=max_stride,
            preprocess_config=preprocess_config,
            postprocess_config=postprocess_config,
        )
        self.use_gt_boxes = use_gt_boxes
        self.gt_bbox_padding = float(gt_bbox_padding)
        self.use_given_boxes = use_given_boxes
        self.box_score_threshold = float(box_score_threshold)

    @property
    def use_gt_centroids(self) -> bool:
        """Whether ``predict`` needs GT instances (the TopDownLayer contract)."""
        return self.use_gt_boxes

    @property
    def categories(self) -> List[str]:
        """Detector class names (``[]`` without a detector)."""
        return self.decoder.categories if self.decoder is not None else []

    def predict(
        self,
        image,
        instances: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        box_scores: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Detect boxes (or read GT/given boxes) and report their centers.

        Args:
            image: Full frames, as for :meth:`BBoxLayer.predict`.
            instances: ``(B, I, n_nodes, 2)`` GT keypoints in original pixels
                (NaN-padded); required with ``use_gt_boxes``.
            boxes: ``(B, I, 4)`` GT boxes aligned with ``instances`` (NaN for
                padding), e.g. ``Batch.instance_boxes``. With ``use_gt_boxes``
                they are used as given; without them boxes are derived from the
                keypoints. With ``use_given_boxes``: ``(B, K, 4)`` xyxy boxes of
                each frame (``Batch.frame_boxes``), required.
            box_scores: ``(B, K)`` scores of the given boxes (all ``1`` when
                ``None``); ``use_given_boxes`` only.

        Returns:
            ``Outputs`` with ``pred_centroids (B, K, 2)``, ``pred_centroid_values
            (B, K)``, ``pred_bboxes (B, K, 4)``, ``pred_bbox_scores (B, K)`` and
            ``pred_bbox_classes (B, K)`` in original-image pixels; empty slots are
            NaN (class ``-1``).
        """
        if self.use_given_boxes:
            boxes, scores, classes = self._given_boxes(image, boxes, box_scores)
        elif self.use_gt_boxes:
            if instances is None:
                raise ValueError(
                    "BBoxStageLayer(use_gt_boxes=True) requires `instances`."
                )
            if boxes is not None:
                boxes = boxes.float().to(instances.device)
                valid = torch.isfinite(boxes).all(dim=-1)
                scores = torch.ones(valid.shape, device=boxes.device).masked_fill(
                    ~valid, float("nan")
                )
                classes = torch.zeros(
                    valid.shape, dtype=torch.long, device=boxes.device
                ).masked_fill(~valid, -1)
            else:
                boxes, scores, classes = gt_boxes_from_instances(
                    instances, padding=self.gt_bbox_padding
                )
        else:
            out = super().predict(image)
            boxes, scores = out.pred_bboxes, out.pred_bbox_scores
            classes = out.pred_bbox_classes
        centers = torch.stack(
            [(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2],
            dim=-1,
        )
        return Outputs(
            pred_centroids=centers,
            pred_centroid_values=scores,
            pred_bboxes=boxes,
            pred_bbox_scores=scores,
            pred_bbox_classes=classes,
        )

    def _given_boxes(
        self,
        image,
        boxes: Optional[torch.Tensor],
        box_scores: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the given boxes and drop those below the score threshold."""
        n_frames = self._to_4d_tensor(image).shape[0]
        if boxes is None:
            # A frame batch with no box source at all: no detections.
            boxes = torch.full((n_frames, 1, 4), float("nan"))
        boxes = torch.as_tensor(boxes).float()
        if boxes.ndim != 3 or boxes.shape[0] != n_frames or boxes.shape[-1] != 4:
            raise ValueError(
                f"Given boxes must be (B={n_frames}, K, 4) xyxy; got "
                f"{tuple(boxes.shape)}."
            )
        if box_scores is None:
            scores = torch.ones(boxes.shape[:2], device=boxes.device)
        else:
            scores = torch.as_tensor(box_scores).float().to(boxes.device)
        valid = torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores)
        valid &= scores >= self.box_score_threshold
        boxes = boxes.masked_fill(~valid[..., None], float("nan"))
        scores = scores.masked_fill(~valid, float("nan"))
        classes = torch.zeros(valid.shape, dtype=torch.long, device=boxes.device)
        return boxes, scores, classes.masked_fill(~valid, -1)


def gt_boxes_from_instances(
    instances: torch.Tensor, padding: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tight, padded boxes of GT instances' visible keypoints.

    Matches ``sio.Instance.to_bbox(mode="tight", padding=padding)``, the derivation
    of the centered-instance model's training boxes for ``bbox_source="instances"``.

    Args:
        instances: ``(B, I, n_nodes, 2)`` keypoints, NaN for missing nodes and
            padded instance slots.
        padding: Pixels added on every side.

    Returns:
        ``(boxes (B, I, 4), scores (B, I), classes (B, I))``: score ``1`` and class
        ``0`` for instances with a visible keypoint, NaN / ``-1`` otherwise.
    """
    pts = instances.float()
    visible = ~torch.isnan(pts).any(dim=-1)  # (B, I, N)
    big = torch.finfo(pts.dtype).max
    lo = torch.where(visible[..., None], pts, torch.full_like(pts, big)).amin(dim=-2)
    hi = torch.where(visible[..., None], pts, torch.full_like(pts, -big)).amax(dim=-2)
    valid = visible.any(dim=-1)  # (B, I)
    boxes = torch.cat([lo - padding, hi + padding], dim=-1)
    boxes = boxes.masked_fill(~valid[..., None], float("nan"))
    scores = torch.ones(valid.shape, device=pts.device).masked_fill(
        ~valid, float("nan")
    )
    classes = torch.zeros(valid.shape, dtype=torch.long, device=pts.device)
    classes = classes.masked_fill(~valid, -1)
    return boxes, scores, classes
