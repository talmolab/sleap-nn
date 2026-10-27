"""Tests for `sleap_nn.training.detr.matcher`."""

import itertools

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from sleap_nn.training.detr.criterion import SetCriterion
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


@pytest.mark.parametrize("make_case", ["random", "tied"])
def test_compact_cost_equals_padded_cost_exactly(make_case):
    """The unpadded per-target cost has the padded cost's values, bit for bit."""
    matcher = HungarianMatcher()
    case = _random_case if make_case == "random" else _tied_case
    outputs, targets = case([2, 0, 5, 1], num_queries=7, seed=4)
    cost, _ = matcher.cost_matrix(outputs, targets)
    image_idx = torch.cat(
        [torch.full((len(t["labels"]),), i) for i, t in enumerate(targets)]
    )
    compact = matcher._compact_cost(
        outputs,
        image_idx,
        torch.cat([t["labels"] for t in targets]),
        torch.cat([t["boxes"] for t in targets]),
    )
    row = 0
    for i, t in enumerate(targets):
        n = len(t["labels"])
        assert torch.equal(compact[row : row + n], cost[i, :, :n].T)
        row += n


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


def _reference_match(matcher, outputs, targets, group_detr=1):
    """The original one-layer path: scipy on the padded host cost, per image/group."""
    cost, valid = matcher.cost_matrix(outputs, targets)
    if not torch.isfinite(cost.masked_select(valid)).all():
        cost = sanitize_cost_matrix(cost, valid)
    cost = cost.cpu().numpy()
    group_size = cost.shape[1] // group_detr
    indices = []
    for i, t in enumerate(targets):
        n = len(t["labels"])
        query_idx, target_idx = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for g in range(group_detr if n else 0):
            start = g * group_size
            rows, cols = linear_sum_assignment(cost[i, start : start + group_size, :n])
            query_idx.append(rows + start)
            target_idx.append(cols)
        indices.append(
            (
                torch.as_tensor(np.concatenate(query_idx), dtype=torch.int64),
                torch.as_tensor(np.concatenate(target_idx), dtype=torch.int64),
            )
        )
    return indices


def _assert_same_matches(actual, expected):
    assert len(actual) == len(expected)
    for (rows, cols), (ref_rows, ref_cols) in zip(actual, expected):
        assert rows.dtype == cols.dtype == torch.int64
        assert torch.equal(rows, ref_rows)
        assert torch.equal(cols, ref_cols)


def _tied_case(sizes, num_queries, seed=0):
    """Predictions and targets drawn from a few values, so costs have many ties."""
    outputs, targets = _random_case(sizes, num_queries, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    levels = torch.tensor([0.2, 0.4, 0.6])
    pick = lambda *shape: levels[torch.randint(0, 3, shape, generator=g)]  # noqa: E731
    outputs["pred_boxes"] = pick(len(sizes), num_queries, 4)
    outputs["pred_logits"] = torch.randint(
        -1, 2, outputs["pred_logits"].shape, generator=g
    ).float()
    for t in targets:
        t["boxes"] = pick(len(t["labels"]), 4)
        t["labels"] = torch.zeros_like(t["labels"])
    return outputs, targets


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("make_case", [_random_case, _tied_case])
def test_match_layers_equals_reference_per_layer(seed, make_case):
    """Batched multi-layer matching gives exactly the original per-layer matches.

    Covers padded targets (mixed sizes, empty images, more targets than a group's
    queries), query groups, ties, and layers with equal and different query counts.
    """
    matcher = HungarianMatcher()
    sizes = [[0, 3, 7, 1], [5, 0], [2, 2, 2], [9, 1, 0, 4]][seed % 4]
    groups = [1, 3, 2][seed % 3]
    layers, targets = [], None
    # Equal query counts take the stacked cost pass, unequal ones the per-layer one.
    query_counts = [6, 6, 6] if seed % 2 else [6, 6, 4]
    for layer_seed, num_queries in enumerate(q * groups for q in query_counts):
        outputs, layer_targets = make_case(sizes, num_queries, seed=seed)
        if targets is None:
            targets = layer_targets
        layers.append(
            {
                k: v + 0.01 * layer_seed * (k == "pred_logits")
                for k, v in outputs.items()
            }
        )
    all_indices = matcher.match_layers(layers, targets, group_detr=groups)
    assert len(all_indices) == len(layers)
    for layer, indices in zip(layers, all_indices):
        expected = _reference_match(matcher, layer, targets, groups)
        _assert_same_matches(indices, expected)
        _assert_same_matches(matcher(layer, targets, group_detr=groups), expected)


def test_match_layers_sanitizes_each_layer_on_its_own():
    """A non-finite layer is sanitized with its own statistics, not the others'."""
    matcher = HungarianMatcher()
    clean, targets = _random_case([3, 2], num_queries=8, seed=1)
    bad = {k: v.clone() for k, v in clean.items()}
    bad["pred_boxes"][0, :3] = float("nan")
    bad["pred_logits"][1, 5] = float("inf")
    for layers in ([clean, bad], [bad, clean]):
        all_indices = matcher.match_layers(layers, targets, group_detr=2)
        for layer, indices in zip(layers, all_indices):
            _assert_same_matches(
                indices, _reference_match(matcher, layer, targets, group_detr=2)
            )


def test_match_layers_empty_and_validation():
    """All-empty targets and no layers short-circuit; bad layers still raise."""
    matcher = HungarianMatcher()
    outputs, targets = _random_case([0, 0], num_queries=4)
    out = matcher.match_layers([outputs, outputs], targets)
    assert [[r.numel() for r, _ in layer] for layer in out] == [[0, 0], [0, 0]]
    assert matcher.match_layers([], targets) == []
    good, targets = _random_case([1], num_queries=4)
    bad, _ = _random_case([1], num_queries=5)
    with pytest.raises(ValueError, match="divisible"):
        matcher.match_layers([good, bad], targets, group_detr=2)


def test_criterion_matches_every_layer_with_the_reference():
    """SetCriterion's losses equal those from per-layer reference matching."""
    num_queries, groups = 12, 3
    outputs, targets = _random_case([2, 0, 4], num_queries=num_queries, seed=7)
    for k in ("aux_outputs", "enc_outputs"):
        outputs.pop(k, None)
    aux = [_random_case([2, 0, 4], num_queries, seed=s)[0] for s in (8, 9)]
    enc = _random_case([2, 0, 4], num_queries, seed=10)[0]
    full = {**outputs, "aux_outputs": aux, "enc_outputs": enc}
    criterion = SetCriterion(num_classes=2, group_detr=groups).train()
    losses = criterion(full, targets)

    class Reference(HungarianMatcher):
        def match_layers(self, layers, targets, group_detr=1):
            return [_reference_match(self, l, targets, group_detr) for l in layers]

    reference = SetCriterion(num_classes=2, matcher=Reference(), group_detr=groups)
    expected = reference.train()(full, targets)
    assert losses.keys() == expected.keys()
    for key in losses:
        assert torch.equal(losses[key], expected[key]), key
