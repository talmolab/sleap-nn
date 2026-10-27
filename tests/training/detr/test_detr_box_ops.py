"""Tests for `sleap_nn.training.detr.box_ops`."""

import pytest
import torch

from sleap_nn.training.detr.box_ops import (
    box_area,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    generalized_box_iou,
)


def _random_xyxy(n, generator):
    cxcywh = torch.cat(
        [
            torch.rand(n, 2, generator=generator),
            0.05 + torch.rand(n, 2, generator=generator),
        ],
        dim=-1,
    )
    return box_cxcywh_to_xyxy(cxcywh)


def test_cxcywh_xyxy_round_trip():
    """Conversions invert each other for valid boxes."""
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.0, 0.3]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    torch.testing.assert_close(xyxy[0], torch.tensor([0.4, 0.3, 0.6, 0.7]))
    torch.testing.assert_close(box_xyxy_to_cxcywh(xyxy), boxes)


def test_cxcywh_to_xyxy_clamps_negative_size():
    """Negative widths/heights collapse to an empty box at the center."""
    xyxy = box_cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, -0.2, 0.4]]))
    torch.testing.assert_close(xyxy, torch.tensor([[0.5, 0.3, 0.5, 0.7]]))


def test_box_iou_known_values():
    """IoU and union of hand-computed pairs."""
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    iou, union = box_iou(a, b)
    torch.testing.assert_close(iou, torch.tensor([[1 / 7, 1.0, 0.0]]))
    torch.testing.assert_close(union, torch.tensor([[7.0, 4.0, 5.0]]))


def test_generalized_box_iou_known_values():
    """GIoU is IoU minus the empty fraction of the enclosing box."""
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 0.0, 3.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    # Disjoint: iou 0, enclosing 3, union 2 -> -1/3.
    torch.testing.assert_close(generalized_box_iou(a, b), torch.tensor([[-1 / 3, 1.0]]))


def test_pairwise_ops_broadcast_over_batch():
    """Batched pairwise ops equal per-batch calls."""
    g = torch.Generator().manual_seed(0)
    b1 = torch.stack([_random_xyxy(5, g) for _ in range(3)])
    b2 = torch.stack([_random_xyxy(4, g) for _ in range(3)])
    iou, union = box_iou(b1, b2)
    giou = generalized_box_iou(b1, b2)
    assert iou.shape == giou.shape == (3, 5, 4)
    for i in range(3):
        torch.testing.assert_close(iou[i], box_iou(b1[i], b2[i])[0])
        torch.testing.assert_close(union[i], box_iou(b1[i], b2[i])[1])
        torch.testing.assert_close(giou[i], generalized_box_iou(b1[i], b2[i]))


def test_elementwise_matches_pairwise_diagonal():
    """Elementwise IoU/GIoU equal the diagonal of the pairwise matrices."""
    g = torch.Generator().manual_seed(1)
    b1, b2 = _random_xyxy(7, g), _random_xyxy(7, g)
    iou, union = elementwise_box_iou(b1, b2)
    torch.testing.assert_close(iou, torch.diag(box_iou(b1, b2)[0]))
    torch.testing.assert_close(union, torch.diag(box_iou(b1, b2)[1]))
    torch.testing.assert_close(
        elementwise_generalized_box_iou(b1, b2),
        torch.diag(generalized_box_iou(b1, b2)),
    )


def test_identical_boxes_have_unit_iou():
    """No epsilon leaks into non-degenerate boxes."""
    boxes = torch.tensor([[0.1, 0.2, 0.4, 0.9]])
    assert elementwise_box_iou(boxes, boxes)[0].item() == 1.0
    assert elementwise_generalized_box_iou(boxes, boxes).item() == 1.0


def test_degenerate_boxes_stay_finite():
    """Zero-area boxes give finite IoU/GIoU instead of 0/0."""
    point = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert torch.isfinite(box_iou(point, point)[0]).all()
    assert torch.isfinite(generalized_box_iou(point, point)).all()
    assert torch.isfinite(elementwise_generalized_box_iou(point, point)).all()


def test_elementwise_rejects_length_mismatch():
    """A length-1 side does not silently broadcast."""
    with pytest.raises(ValueError, match="same length"):
        elementwise_box_iou(torch.zeros(1, 4), torch.zeros(3, 4))


def test_box_area():
    """Area of xyxy boxes over leading dims."""
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.5, 2.0]]])
    torch.testing.assert_close(box_area(boxes), torch.tensor([[6.0, 0.5]]))
