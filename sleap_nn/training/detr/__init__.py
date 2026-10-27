"""Set-prediction training for DETR-style box detectors (``bbox_detr``).

Box ops, the Hungarian set matcher (with Group DETR) and the RF-DETR set criterion
(IoU-aware BCE + L1 + GIoU, deep supervision over decoder and encoder outputs).

Conventions shared by the matcher and criterion:

- ``outputs`` is a dict with ``pred_logits`` ``(B, Q, num_classes + 1)`` sigmoid
  logits and ``pred_boxes`` ``(B, Q, 4)`` normalized ``cxcywh`` boxes, optionally
  ``aux_outputs`` (a list of such dicts, one per earlier decoder layer) and
  ``enc_outputs`` (one such dict for the two-stage encoder proposals).
- ``targets`` is a list of ``B`` dicts with ``labels`` ``(N_i,)`` int64 class ids in
  ``[0, num_classes)`` and ``boxes`` ``(N_i, 4)`` normalized ``cxcywh`` boxes.
- With Group DETR, ``Q = G * Q_g``: queries ``[g * Q_g, (g + 1) * Q_g)`` form group
  ``g`` and are matched to the targets independently of the other groups.
"""

from sleap_nn.training.detr.box_ops import (
    box_area,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    generalized_box_iou,
)
from sleap_nn.training.detr.criterion import (
    LOSS_TERMS,
    SetCriterion,
    register_loss_term,
)
from sleap_nn.training.detr.matcher import HungarianMatcher

__all__ = [
    "box_area",
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "elementwise_box_iou",
    "elementwise_generalized_box_iou",
    "generalized_box_iou",
    "HungarianMatcher",
    "LOSS_TERMS",
    "SetCriterion",
    "register_loss_term",
]
