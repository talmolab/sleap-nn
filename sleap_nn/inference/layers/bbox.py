"""``BBoxLayer`` — whole-frame bounding-box detection inference.

Wraps a trained ``bbox_centernet`` / ``bbox_ssd`` LightningModule (whose ``forward``
returns the decoder-ready head dict) and decodes its outputs with the model's
:class:`~sleap_nn.inference.bbox.BBoxDecoder` into ``Outputs.pred_bboxes`` /
``pred_bbox_scores`` / ``pred_bbox_classes`` in original-image pixels, which
``Outputs.to_labels`` packages as ``sio.PredictedBoundingBox`` on
``LabeledFrame.bboxes``.
"""

from __future__ import annotations

from typing import List, Optional

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
