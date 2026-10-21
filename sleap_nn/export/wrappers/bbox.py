"""ONNX export wrapper for the bounding-box detectors (``bbox_centernet``, ``bbox_ssd``)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper

# Graph output names per detector style, in the order the wrapper returns them.
BBOX_EXPORT_OUTPUTS = {
    "bbox_centernet": ["BBoxCenterHead", "BBoxSizeHead", "BBoxOffsetHead"],
    "bbox_ssd": ["SSDHead"],
}


class BBoxONNXWrapper(BaseExportWrapper):
    """Wrap a detector for export: uint8 image -> decoder-ready head outputs.

    Unlike the pose wrappers, decoding is NOT baked into the graph: the output is
    exactly what the detector's LightningModule ``forward`` returns (CenterNet's
    center heatmap as probabilities plus raw size/offset maps; SSD's raw
    per-anchor logits + deltas), and
    :class:`~sleap_nn.inference.layers.exported.ExportedBBoxLayer` decodes it
    with the same :class:`~sleap_nn.inference.bbox.BBoxDecoder` as the
    checkpoint path. That keeps exported and native predictions identical
    (anchors, NMS and thresholds live in one place) and keeps the graph free of
    size-dependent anchor constants, so its spatial axes stay dynamic.

    Expects ``(B, C, H, W)`` uint8 images (or float in ``[0, 255]``) whose
    ``input_scale``-resized size is divisible by the backbone's max stride (the
    exported layer pads frames to guarantee it).
    """

    def __init__(self, model: nn.Module, model_type: str, input_scale: float = 1.0):
        """Initialize.

        Args:
            model: The underlying ``Model`` (backbone + detector heads).
            model_type: ``"bbox_centernet"`` or ``"bbox_ssd"``.
            input_scale: Input scaling factor applied before the network.
        """
        super().__init__(model)
        if model_type not in BBOX_EXPORT_OUTPUTS:
            raise ValueError(f"{model_type} is not a bounding-box model type.")
        self.model_type = model_type
        self.input_scale = input_scale

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the network and return the decoder-ready head outputs."""
        image = self._normalize_uint8(image)
        if self.input_scale != 1.0:
            height = int(image.shape[-2] * self.input_scale)
            width = int(image.shape[-1] * self.input_scale)
            image = F.interpolate(
                image, size=(height, width), mode="bilinear", align_corners=False
            )
        out = self.model(image)
        if self.model_type == "bbox_centernet":
            return {
                "BBoxCenterHead": torch.sigmoid(out["BBoxCenterHead"]),
                "BBoxSizeHead": out["BBoxSizeHead"],
                "BBoxOffsetHead": out["BBoxOffsetHead"],
            }
        return {"SSDHead": out["SSDHead"]}
