"""Model modules of the ``bbox_detr`` (RF-DETR-style) box detector.

The layout follows HF transformers' ``models/rf_detr`` port so RF-DETR weights load
by renaming: a backbone with tapped token maps (:mod:`.backbones`), a single-level
projector (:mod:`.projector`), ``grid_sample`` deformable attention
(:mod:`.deformable_attention`), the decoder (:mod:`.decoder`) and the two-stage
proposals and prediction heads (:mod:`.model`). Training losses live in
:mod:`sleap_nn.training.detr`.
"""

from sleap_nn.architectures.detr.backbones import (
    DETR_BACKBONES,
    DetrBackbone,
    TinyDetrBackbone,
    build_detr_backbone,
    register_detr_backbone,
)
from sleap_nn.architectures.detr.decoder import (
    MLP,
    Decoder,
    DecoderLayer,
    GroupSelfAttention,
    refine_boxes,
    sine_position_embedding,
)
from sleap_nn.architectures.detr.deformable_attention import (
    MultiScaleDeformableAttention,
    multi_scale_deformable_attention,
)
from sleap_nn.architectures.detr.model import (
    BBoxDetrHead,
    build_bbox_detr_head,
    token_proposals,
)
from sleap_nn.architectures.detr.projector import C2FLayer, ScaleProjector

__all__ = [
    "BBoxDetrHead",
    "build_bbox_detr_head",
    "build_detr_backbone",
    "C2FLayer",
    "Decoder",
    "DecoderLayer",
    "DETR_BACKBONES",
    "DetrBackbone",
    "GroupSelfAttention",
    "MLP",
    "multi_scale_deformable_attention",
    "MultiScaleDeformableAttention",
    "refine_boxes",
    "register_detr_backbone",
    "ScaleProjector",
    "sine_position_embedding",
    "TinyDetrBackbone",
    "token_proposals",
]
