"""ONNX export wrapper for the bounding-box detectors (``bbox_centernet``, ``bbox_ssd``, ``bbox_detr``)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from sleap_nn.export.wrappers.base import BaseExportWrapper

# Graph output names per detector style, in the order the wrapper returns them.
BBOX_EXPORT_OUTPUTS = {
    "bbox_centernet": ["BBoxCenterHead", "BBoxSizeHead", "BBoxOffsetHead"],
    "bbox_ssd": ["SSDHead"],
    "bbox_detr": ["pred_logits", "pred_boxes"],
}


class BBoxONNXWrapper(BaseExportWrapper):
    """Wrap a detector for export: uint8 image -> decoder-ready head outputs.

    Unlike the pose wrappers, decoding is NOT baked into the graph: the output is
    exactly what the detector's LightningModule ``forward`` returns (CenterNet's
    center heatmap as probabilities plus raw size/offset maps; SSD's raw
    per-anchor logits + deltas; DETR's final-layer ``pred_logits`` / ``pred_boxes``,
    without the training-only auxiliary and encoder outputs), and
    :class:`~sleap_nn.inference.layers.bbox.ExportedBBoxLayer` decodes it
    with the same :class:`~sleap_nn.inference.bbox.BBoxDecoder` as the
    checkpoint path. That keeps exported and native predictions identical
    (anchors, NMS and thresholds live in one place) and keeps the graph free of
    size-dependent anchor constants, so its spatial axes stay dynamic.

    Expects ``(B, C, H, W)`` uint8 images (or float in ``[0, 255]``) that are
    already at the model's input size (size-matched, ``input_scale``-resized and
    padded to the backbone's max stride). ``ExportedBBoxLayer`` does that on the
    host with the checkpoint path's own preprocessing, because a resize traced into
    the graph would be frozen to the size of the trace input.
    """

    def __init__(self, model: nn.Module, model_type: str):
        """Initialize.

        Args:
            model: The underlying ``Model`` (backbone + detector heads).
            model_type: ``"bbox_centernet"``, ``"bbox_ssd"`` or ``"bbox_detr"``.
        """
        super().__init__(model)
        if model_type not in BBOX_EXPORT_OUTPUTS:
            raise ValueError(f"{model_type} is not a bounding-box model type.")
        self.model_type = model_type

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the network and return the decoder-ready head outputs."""
        out = self.model(self._normalize_uint8(image))
        if self.model_type == "bbox_centernet":
            return {
                "BBoxCenterHead": torch.sigmoid(out["BBoxCenterHead"]),
                "BBoxSizeHead": out["BBoxSizeHead"],
                "BBoxOffsetHead": out["BBoxOffsetHead"],
            }
        if self.model_type == "bbox_detr":
            return {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        return {"SSDHead": out["SSDHead"]}
