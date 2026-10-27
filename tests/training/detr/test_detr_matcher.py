"""Tests for `sleap_nn.training.detr.matcher`."""

import itertools

import pytest
import torch

from sleap_nn.training.detr.matcher import (
    HungarianMatcher,
    focal_classification_cost,
    sanitize_cost_matrix,
)


def _random_case(sizes, num_queries, num_logits=3, seed=0):
    g = torch.Generator().manual_seed(seed)

    def boxes(*shape):
        return torch.cat(
            [
                0.1 + 0.8 * torch.rand(*shape, 2, generator=g),
                0.05 + 0.4 * torch.rand(*shape, 2, generator=g),
            ],
            dim=-1,
        )

    outputs = {
        "pred_logits": torch.randn(len(sizes), num_queries, num_logits, generator=g),
        "pred_boxes": boxes(len(sizes), num_queries),
    }
    targets = [
        {
            "labels": torch.randint(0, num_logits - 1, (n,), generator=g),
            "boxes": boxes(n),
        }
        for n in sizes
    ]
    return outputs, targets


def _brute_force_cost(matcher, outputs, targets, image):
    """Unbatched reference cost for one image."""
    from sleap_nn.training.detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

    logits = outputs["pred_logits"][image]
    boxes = outputs["pred_boxes"][image]
    t = targets[image]
    class_cost = focal_classification_cost(logits[:, t["labels"]])
    bbox_cost = (boxes[:, None] - t["boxes"][None]).abs().sum(-1)
    giou = generalized_box_iou(
        box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(t["boxes"])
    )
    return (
        matcher.cost_class * class_cost
        + matcher.cost_bbox * bbox_cost
        - matcher.cost_giou * giou
    )


def test_batched_cost_matches_per_image_cost():
    """Padding targets to a batch does not change real entries."""
    matcher = HungarianMatcher()
    outputs, targets = _random_case([2, 5, 1], num_queries=6)
    cost, valid = matcher.cost_matrix(outputs, targets)
    assert cost.shape == (3, 6, 5)
    assert valid[:, 0].sum(-1).tolist() == [2, 5, 1]
    for i, t in enumerate(targets):
        n = len(t["labels"])
        torch.testing.assert_close(
            cost[i, :, :n], _brute_force_cost(matcher, outputs, targets, i)
        )


def test_matching_is_optimal():
    """The assignment has the minimum total cost over all one-to-one assignments."""
    matcher = HungarianMatcher()
    outputs, targets = _random_case([3], num_queries=5, seed=3)
    ((rows, cols),) = matcher(outputs, targets)
    cost = _brute_force_cost(matcher, outputs, targets, 0)
    best = min(
        sum(cost[q, t] for t, q in enumerate(perm))
        for perm in itertools.permutations(range(5), 3)
    )
    torch.testing.assert_close(cost[rows, cols].sum(), best)
    assert rows.dtype == cols.dtype == torch.int64


def test_perfect_predictions_match_their_targets():
    """Queries equal to the targets (in shuffled order) are matched to them."""
    targets = [
        {
            "labels": torch.tensor([0, 1, 0]),
            "boxes": torch.tensor(
                [[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.3], [0.8, 0.7, 0.1, 0.2]]
            ),
        }
    ]
    order = torch.tensor([2, 0, 1])
    logits = torch.full((1, 4, 3), -5.0)
    logits[0, torch.arange(3), targets[0]["labels"][order]] = 5.0
    boxes = torch.cat(
        [targets[0]["boxes"][order], torch.tensor([[0.5, 0.5, 0.9, 0.9]])]
    )[None]
    ((rows, cols),) = HungarianMatcher()(
        {"pred_logits": logits, "pred_boxes": boxes}, targets
    )
    assert dict(zip(rows.tolist(), cols.tolist())) == {0: 2, 1: 0, 2: 1}


def test_groups_are_matched_independently():
    """Each group's matches equal matching that group's queries alone."""
    matcher = HungarianMatcher()
    group_size, groups = 4, 3
    outputs, targets = _random_case([3, 2], num_queries=group_size * groups)
    indices = matcher(outputs, targets, group_detr=groups)
    for image, (rows, cols) in enumerate(indices):
        n = len(targets[image]["labels"])
        assert len(rows) == groups * n
        for g in range(groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            group_out = {k: v[:, sl] for k, v in outputs.items()}
            g_rows, g_cols = matcher(group_out, targets)[image]
            torch.testing.assert_close(rows[g * n : (g + 1) * n], g_rows + sl.start)
            torch.testing.assert_close(cols[g * n : (g + 1) * n], g_cols)


def test_more_targets_than_group_queries():
    """Each group matches min(group size, targets) pairs."""
    outputs, targets = _random_case([6], num_queries=8)
    rows, cols = HungarianMatcher()(outputs, targets, group_detr=2)[0]
    assert len(rows) == 8
    assert len(set(rows.tolist())) == 8
    for g in range(2):
        assert len(set(cols[g * 4 : (g + 1) * 4].tolist())) == 4


def test_empty_targets():
    """Images without targets get empty matches; an all-empty batch too."""
    matcher = HungarianMatcher()
    outputs, targets = _random_case([0, 2, 0], num_queries=6)
    indices = matcher(outputs, targets, group_detr=2)
    assert [len(rows) for rows, _ in indices] == [0, 4, 0]

    outputs, targets = _random_case([0, 0], num_queries=6)
    for rows, cols in matcher(outputs, targets, group_detr=3):
        assert rows.numel() == cols.numel() == 0
        assert rows.dtype == torch.int64


def test_non_finite_costs_are_guarded():
    """NaN/inf predictions do not crash and lose to finite ones."""
    matcher = HungarianMatcher()
    outputs, targets = _random_case([2], num_queries=4)
    outputs["pred_boxes"][0, 0] = float("nan")
    outputs["pred_logits"][0, 1] = float("inf")
    rows, cols = matcher(outputs, targets)[0]
    assert sorted(rows.tolist()) == [2, 3]

    # Everything non-finite still yields a full assignment.
    outputs["pred_boxes"][:] = float("nan")
    rows, _ = matcher(outputs, targets)[0]
    assert len(rows) == 2


def test_sanitize_cost_matrix():
    """The sentinel exceeds every valid finite cost, even all-negative ones."""
    cost = torch.tensor([[-3.0, float("nan")], [float("inf"), -1.0]])
    out = sanitize_cost_matrix(cost, torch.ones_like(cost, dtype=torch.bool))
    assert torch.isfinite(out).all()
    assert out[0, 1] > -1.0 and out[1, 0] > -1.0
    assert sanitize_cost_matrix(torch.ones(2, 2), torch.ones(2, 2).bool()).eq(1).all()
    all_bad = sanitize_cost_matrix(
        torch.full((2, 2), float("nan")), torch.ones(2, 2).bool()
    )
    assert torch.isfinite(all_bad).all()
    # A sentinel that would overflow is clamped to the largest finite value.
    huge = torch.tensor([[3e38, float("inf")]])
    out = sanitize_cost_matrix(huge, torch.ones_like(huge, dtype=torch.bool))
    assert out[0, 1] == torch.finfo(torch.float32).max


def test_focal_classification_cost_values():
    """Logit 0 gives the closed-form focal cost; confident positives cost less."""
    cost = focal_classification_cost(torch.tensor([0.0, 4.0, -4.0]))
    log2 = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(cost[0], 0.25 * 0.25 * log2 - 0.75 * 0.25 * log2)
    assert cost[1] < cost[0] < cost[2]


def test_rejects_bad_groups_and_weights():
    """Queries must split into groups; not all costs may be zero."""
    outputs, targets = _random_case([1], num_queries=5)
    with pytest.raises(ValueError, match="divisible"):
        HungarianMatcher()(outputs, targets, group_detr=2)
    with pytest.raises(ValueError, match="2 targets for a batch of 1"):
        HungarianMatcher()(outputs, targets * 2)
    with pytest.raises(ValueError, match="non-zero"):
        HungarianMatcher(cost_class=0, cost_bbox=0, cost_giou=0)


def test_no_grad():
    """Matching never builds a graph."""
    outputs, targets = _random_case([2], num_queries=4)
    outputs["pred_logits"].requires_grad_(True)
    rows, _ = HungarianMatcher()(outputs, targets)[0]
    assert not rows.requires_grad
