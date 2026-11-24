"""Tests for `sleap_nn.training.detr.criterion`."""

import pytest
import torch

from sleap_nn.training.detr import criterion as criterion_module
from sleap_nn.training.detr.box_ops import (
    box_cxcywh_to_xyxy,
    elementwise_box_iou,
)
from sleap_nn.training.detr.criterion import (
    DEFAULT_WEIGHT_DICT,
    LOSS_TERMS,
    SetCriterion,
    register_loss_term,
)

NUM_CLASSES = 2


def _boxes(*shape, generator):
    return torch.cat(
        [
            0.1 + 0.8 * torch.rand(*shape, 2, generator=generator),
            0.05 + 0.4 * torch.rand(*shape, 2, generator=generator),
        ],
        dim=-1,
    )


def _layer(batch_size, num_queries, generator):
    return {
        "pred_logits": torch.randn(
            batch_size, num_queries, NUM_CLASSES + 1, generator=generator
        ),
        "pred_boxes": _boxes(batch_size, num_queries, generator=generator),
    }


def _case(sizes=(2, 3), num_queries=6, aux_layers=2, enc=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    outputs = _layer(len(sizes), num_queries, g)
    if aux_layers:
        outputs["aux_outputs"] = [
            _layer(len(sizes), num_queries, g) for _ in range(aux_layers)
        ]
    if enc:
        outputs["enc_outputs"] = _layer(len(sizes), num_queries, g)
    targets = [
        {
            "labels": torch.randint(0, NUM_CLASSES, (n,), generator=g),
            "boxes": _boxes(n, generator=g),
        }
        for n in sizes
    ]
    return outputs, targets


def _reference_ia_bce(logits, pred_boxes, targets, indices, num_boxes):
    """Loop-based IA-BCE in the -pos*log(p) - neg*log(1-p) form."""
    prob = logits.sigmoid()
    pos = torch.zeros_like(prob)
    neg = prob**2
    for b, (rows, cols) in enumerate(indices):
        for q, t in zip(rows.tolist(), cols.tolist()):
            c = targets[b]["labels"][t]
            iou, _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[b, q][None]),
                box_cxcywh_to_xyxy(targets[b]["boxes"][t][None]),
            )
            quality = max(float(prob[b, q, c] ** 0.25 * iou[0] ** 0.75), 0.01)
            pos[b, q, c] = quality
            neg[b, q, c] = 1 - quality
    loss = -pos * prob.log() - neg * (1 - prob).log()
    return loss.sum() / num_boxes


def test_loss_keys_per_layer():
    """Final, aux (_i) and encoder (_enc) layers each get every loss key."""
    outputs, targets = _case(aux_layers=2)
    losses = SetCriterion(NUM_CLASSES)(outputs, targets)
    base = ["loss_ce", "loss_bbox", "loss_giou", "cardinality_error"]
    expected = {k + s for k in base for s in ["", "_0", "_1", "_enc"]}
    assert set(losses) == expected
    assert all(torch.isfinite(v) for v in losses.values())


def test_ia_bce_matches_reference():
    """IA-BCE equals a direct loop over matched queries."""
    outputs, targets = _case(aux_layers=0, enc=False)
    criterion = SetCriterion(NUM_CLASSES, group_detr=2).train()
    losses = criterion(outputs, targets)
    indices = criterion.matcher(outputs, targets, group_detr=2)
    num_boxes = criterion.num_boxes(targets, torch.device("cpu"))
    expected = _reference_ia_bce(
        outputs["pred_logits"], outputs["pred_boxes"], targets, indices, num_boxes
    )
    torch.testing.assert_close(losses["loss_ce"], expected)


def test_ia_bce_is_stable_for_extreme_logits():
    """The logsigmoid form stays finite where log(p) / log(1 - p) would not."""
    outputs, targets = _case(aux_layers=0, enc=False)
    outputs["pred_logits"] = outputs["pred_logits"] * 200
    losses = SetCriterion(NUM_CLASSES)(outputs, targets)
    assert torch.isfinite(losses["loss_ce"])


def test_box_losses_are_zero_for_perfect_matches():
    """Predicting the targets exactly gives zero L1 and GIoU loss."""
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.4]]),
        }
    ]
    logits = torch.full((1, 3, NUM_CLASSES + 1), -6.0)
    logits[0, 0, 0] = logits[0, 1, 1] = 6.0
    boxes = torch.cat([targets[0]["boxes"], torch.tensor([[0.5, 0.5, 0.1, 0.1]])])[None]
    losses = SetCriterion(NUM_CLASSES)(
        {"pred_logits": logits, "pred_boxes": boxes}, targets
    )
    assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["loss_giou"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["cardinality_error"].item() == 0.0


def test_box_losses_known_value():
    """L1 and GIoU are summed over matches and divided by num_boxes."""
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    ]
    outputs = {
        "pred_logits": torch.zeros(1, 1, NUM_CLASSES + 1),
        "pred_boxes": torch.tensor([[[0.6, 0.5, 0.2, 0.2]]]),
    }
    losses = SetCriterion(NUM_CLASSES)(outputs, targets)
    assert losses["loss_bbox"].item() == pytest.approx(0.1)
    # Overlap 0.1 x 0.2, union 0.06, enclosing 0.3 x 0.2: GIoU = 1/3 - 0.
    assert losses["loss_giou"].item() == pytest.approx(1 - 1 / 3)


def test_num_boxes_counts_groups_in_training_only():
    """num_boxes = targets x groups in training, targets in eval, at least 1."""
    _, targets = _case(sizes=(2, 3))
    criterion = SetCriterion(NUM_CLASSES, group_detr=13)
    cpu = torch.device("cpu")
    assert criterion.train().num_boxes(targets, cpu).item() == 65
    assert criterion.eval().num_boxes(targets, cpu).item() == 5
    _, empty = _case(sizes=(0, 0))
    assert criterion.train().num_boxes(empty, cpu).item() == 1


def test_num_boxes_is_averaged_across_ranks(monkeypatch):
    """The normalizer is all-reduced and divided by the world size."""
    dist = criterion_module.dist
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    # Pretend the other three ranks hold 1, 2 and 9 targets x 2 groups.
    monkeypatch.setattr(dist, "all_reduce", lambda t: t.add_(24.0))
    _, targets = _case(sizes=(2, 3))
    num_boxes = (
        SetCriterion(NUM_CLASSES, group_detr=2)
        .train()
        .num_boxes(targets, torch.device("cpu"))
    )
    assert num_boxes.item() == pytest.approx((10 + 24) / 4)


def test_groups_scale_matches_and_normalizer():
    """Duplicating a group's queries keeps the mean loss per group the same."""
    outputs, targets = _case(aux_layers=0, enc=False)
    one = SetCriterion(NUM_CLASSES, group_detr=1).train()(outputs, targets)
    doubled = {k: torch.cat([v, v], dim=1) for k, v in outputs.items()}
    two = SetCriterion(NUM_CLASSES, group_detr=2).train()(doubled, targets)
    for key in ["loss_ce", "loss_bbox", "loss_giou"]:
        torch.testing.assert_close(two[key], one[key])


def test_empty_targets():
    """No targets: box losses are 0, classification still pushes logits down."""
    outputs, targets = _case(sizes=(0, 0))
    criterion = SetCriterion(NUM_CLASSES, group_detr=2).train()
    losses = criterion(outputs, targets)
    assert losses["loss_bbox"].item() == 0.0
    assert losses["loss_giou"].item() == 0.0
    expected = (
        outputs["pred_logits"].sigmoid() ** 2
        * -torch.log1p(-outputs["pred_logits"].sigmoid())
    ).sum()
    torch.testing.assert_close(losses["loss_ce"], expected)


def test_gradients():
    """Weighted losses reach logits and boxes; cardinality has no graph."""
    outputs, targets = _case(aux_layers=1)
    for layer in [outputs, *outputs["aux_outputs"], outputs["enc_outputs"]]:
        layer["pred_logits"].requires_grad_(True)
        layer["pred_boxes"].requires_grad_(True)
    criterion = SetCriterion(NUM_CLASSES, group_detr=2).train()
    losses = criterion(outputs, targets)
    assert not losses["cardinality_error"].requires_grad
    criterion.total_loss(losses).backward()
    for layer in [outputs, *outputs["aux_outputs"], outputs["enc_outputs"]]:
        assert layer["pred_logits"].grad.abs().sum() > 0
        assert layer["pred_boxes"].grad.abs().sum() > 0


def test_ia_bce_target_is_detached():
    """The soft target t carries no gradient into the boxes."""
    outputs, targets = _case(aux_layers=0, enc=False)
    outputs["pred_logits"].requires_grad_(True)
    outputs["pred_boxes"].requires_grad_(True)
    criterion = SetCriterion(NUM_CLASSES, losses=["labels"])
    losses = criterion(outputs, targets)
    (grad,) = torch.autograd.grad(
        losses["loss_ce"], outputs["pred_boxes"], allow_unused=True
    )
    assert grad is None


def test_total_loss_weights():
    """total_loss = sum of weight x term over layers; logged-only keys excluded."""
    outputs, targets = _case(aux_layers=1)
    criterion = SetCriterion(NUM_CLASSES)
    losses = criterion(outputs, targets)
    expected = sum(
        DEFAULT_WEIGHT_DICT[k] * losses[k + s]
        for k in DEFAULT_WEIGHT_DICT
        for s in ["", "_0", "_enc"]
    )
    torch.testing.assert_close(criterion.total_loss(losses), expected)

    custom = SetCriterion(NUM_CLASSES, weight_dict={"loss_giou": 3.0})
    torch.testing.assert_close(
        custom.total_loss(losses),
        3.0 * (losses["loss_giou"] + losses["loss_giou_0"] + losses["loss_giou_enc"]),
    )
    with pytest.raises(ValueError, match="weight"):
        custom.total_loss({"cardinality_error": torch.tensor(1.0)})


def test_registry_plugs_in_custom_terms():
    """A registered term runs on every layer with the layer's matches."""
    name = "test_matched_count"

    @register_loss_term(name)
    def matched_count(criterion, outputs, targets, indices, num_boxes):
        return {"loss_matched": sum(len(r) for r, _ in indices) / num_boxes}

    try:
        outputs, targets = _case(sizes=(2, 3), aux_layers=1)
        criterion = SetCriterion(
            NUM_CLASSES, losses=["boxes", name], group_detr=2
        ).train()
        losses = criterion(outputs, targets)
        assert {"loss_matched", "loss_matched_0", "loss_matched_enc"} <= set(losses)
        assert "loss_ce" not in losses
        assert losses["loss_matched"].item() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="already registered"):
            register_loss_term(name)(matched_count)
    finally:
        LOSS_TERMS.pop(name)


def test_rejects_unknown_terms_and_bad_logit_width():
    """Unknown term names and logits not num_classes + 1 wide raise."""
    with pytest.raises(ValueError, match="Unknown DETR loss term"):
        SetCriterion(NUM_CLASSES, losses=["labels", "masks"])
    outputs, targets = _case(aux_layers=0, enc=False)
    with pytest.raises(ValueError, match="num_classes \\+ 1"):
        SetCriterion(NUM_CLASSES + 1)(outputs, targets)
