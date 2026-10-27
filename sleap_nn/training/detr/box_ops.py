"""Box conversions and (generalized) IoU for DETR set losses.

Pairwise ops broadcast over leading batch dimensions: ``(..., N, 4)`` against
``(..., M, 4)`` gives ``(..., N, M)``. Elementwise ops compare matched pairs
``(N, 4)`` against ``(N, 4)`` and give ``(N,)`` without building the ``N x N``
matrix. IoU ops take ``xyxy`` boxes.

Numerics follow ``rfdetr`` (``utilities/box_ops.py``): ``cxcywh -> xyxy`` clamps
negative widths and heights to 0, and union / enclosing areas are clamped at
``1e-7`` only to keep degenerate (zero-area) boxes finite, so identical
non-degenerate boxes give an IoU of exactly 1.
"""

from typing import Tuple

import torch

_AREA_EPS = 1e-7


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x0, y0, x1, y1)``.

    Negative widths and heights are clamped to 0, so the result is always a valid
    (possibly empty) corner box.

    Args:
        boxes: Boxes ``(..., 4)`` in ``cxcywh`` format.

    Returns:
        Boxes ``(..., 4)`` in ``xyxy`` format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    half_w = 0.5 * w.clamp(min=0.0)
    half_h = 0.5 * h.clamp(min=0.0)
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(x0, y0, x1, y1)`` boxes to ``(cx, cy, w, h)``.

    Args:
        boxes: Boxes ``(..., 4)`` in ``xyxy`` format.

    Returns:
        Boxes ``(..., 4)`` in ``cxcywh`` format.
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of ``xyxy`` boxes.

    Args:
        boxes: Boxes ``(..., 4)`` in ``xyxy`` format.

    Returns:
        Areas ``(...,)``.
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and union between two sets of ``xyxy`` boxes.

    Args:
        boxes1: Boxes ``(..., N, 4)``.
        boxes2: Boxes ``(..., M, 4)``; leading dims broadcast with ``boxes1``.

    Returns:
        A tuple ``(iou, union)``, each ``(..., N, M)``.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[..., :, None] + area2[..., None, :] - inter
    return inter / union.clamp(min=_AREA_EPS), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise generalized IoU (https://giou.stanford.edu) of ``xyxy`` boxes.

    Args:
        boxes1: Boxes ``(..., N, 4)``.
        boxes2: Boxes ``(..., M, 4)``; leading dims broadcast with ``boxes1``.

    Returns:
        GIoU ``(..., N, M)`` in ``[-1, 1]``.
    """
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=_AREA_EPS)


def _check_same_length(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Refuse unequal lengths so a length-1 side cannot silently broadcast."""
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            "Elementwise box ops need boxes1 and boxes2 of the same length, got "
            f"{boxes1.shape[0]} and {boxes2.shape[0]}."
        )


def elementwise_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IoU and union of matched ``xyxy`` box pairs.

    Equal to the diagonal of :func:`box_iou` at ``O(N)`` memory.

    Args:
        boxes1: Boxes ``(N, 4)``.
        boxes2: Boxes ``(N, 4)``.

    Returns:
        A tuple ``(iou, union)``, each ``(N,)``.

    Raises:
        ValueError: If the two sets have different lengths.
    """
    _check_same_length(boxes1, boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return inter / union.clamp(min=_AREA_EPS), union


def elementwise_generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """Generalized IoU of matched ``xyxy`` box pairs.

    Equal to the diagonal of :func:`generalized_box_iou` at ``O(N)`` memory.

    Args:
        boxes1: Boxes ``(N, 4)``.
        boxes2: Boxes ``(N, 4)``.

    Returns:
        GIoU ``(N,)`` in ``[-1, 1]``.

    Raises:
        ValueError: If the two sets have different lengths.
    """
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:, 0] * wh[:, 1]
    return iou - (enclosing - union) / enclosing.clamp(min=_AREA_EPS)
