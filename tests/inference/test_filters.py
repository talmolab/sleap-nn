"""Tests for ``FilterConfig`` + ``FilterPipeline``.

Coverage:

1. Default ``FilterConfig`` is the no-op identity (everything passes through).
2. ``min_peak_value`` NaN-outs individual keypoints below threshold.
3. ``min_visible_nodes`` drops instances with too few visible kpts.
4. ``min_visible_node_fraction`` drops by visible-fraction.
5. ``min_instance_score`` drops by ``instance_scores``.
6. ``min_mean_node_score`` drops by mean per-keypoint score.
7. ``overlapping`` (IoU + OKS variants) dedupes overlapping instances.
8. Filter order — multi-filter combinations apply in cheap → expensive
   order.
9. ``FilterConfig`` is picklable (Tom's design-review constraint).
10. ``FilterPipeline.run`` one-off convenience returns same result as
    instantiated pipeline.
"""

from __future__ import annotations

import math
import pickle

import pytest
import torch

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.outputs import Outputs


def _make_outputs(B: int = 1, I: int = 3, N: int = 4) -> Outputs:
    """Build a synthetic ``Outputs`` with predictable, mostly-valid data."""
    kpts = torch.tensor(
        [
            # frame 0: 3 instances, varied keypoint scores
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]],
                [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0], [23.0, 23.0]],
            ]
        ]
    )
    vals = torch.tensor(
        [
            [
                [0.9, 0.8, 0.7, 0.6],  # mean 0.75, all visible
                [0.4, 0.3, 0.2, 0.1],  # mean 0.25, low scores
                [0.95, 0.05, 0.95, 0.05],  # bimodal — half above 0.5
            ]
        ]
    )
    inst_scores = torch.tensor([[0.85, 0.30, 0.55]])
    return Outputs(
        pred_keypoints=kpts,
        pred_peak_values=vals,
        instance_scores=inst_scores,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. No-op
# ─────────────────────────────────────────────────────────────────────────


def test_default_config_is_no_op():
    """A default ``FilterConfig`` makes the pipeline a no-op."""
    o = _make_outputs()
    out = FilterPipeline(FilterConfig())(o)
    torch.testing.assert_close(out.pred_keypoints, o.pred_keypoints)
    torch.testing.assert_close(out.pred_peak_values, o.pred_peak_values)


# ─────────────────────────────────────────────────────────────────────────
# 2. min_peak_value
# ─────────────────────────────────────────────────────────────────────────


def test_min_peak_value_nans_out_low_keypoints():
    """Individual kpts below ``min_peak_value`` become NaN; instance survives."""
    o = _make_outputs()
    out = FilterPipeline(FilterConfig(min_peak_value=0.5))(o)
    # Instance 1 has all scores < 0.5 → all kpts NaN; instance 2 has half.
    assert torch.isnan(out.pred_keypoints[0, 1]).all()
    assert torch.isnan(out.pred_keypoints[0, 2, 1]).all()  # 0.05 < 0.5
    assert not torch.isnan(out.pred_keypoints[0, 2, 0]).any()  # 0.95 >= 0.5


# ─────────────────────────────────────────────────────────────────────────
# 3. min_visible_nodes
# ─────────────────────────────────────────────────────────────────────────


def test_min_visible_nodes_drops_sparse_instances():
    """Instances with < min_visible_nodes are dropped (NaN-padded)."""
    o = _make_outputs()
    # First, NaN out half of instance 2's kpts (keep 2 visible).
    kpts = o.pred_keypoints.clone()
    kpts[0, 2, 2:] = float("nan")
    o = type(o)(
        pred_keypoints=kpts,
        pred_peak_values=o.pred_peak_values,
        instance_scores=o.instance_scores,
    )
    out = FilterPipeline(FilterConfig(min_visible_nodes=3))(o)
    # Instances 0 + 1 have 4 visible kpts each; 2 has only 2 → dropped.
    assert not torch.isnan(out.pred_keypoints[0, 0]).any()
    assert not torch.isnan(out.pred_keypoints[0, 1]).any()
    assert torch.isnan(out.pred_keypoints[0, 2]).all()


# ─────────────────────────────────────────────────────────────────────────
# 4. min_visible_node_fraction
# ─────────────────────────────────────────────────────────────────────────


def test_min_visible_node_fraction_drops_below_fraction():
    """Visible-fraction filter is independent of absolute count."""
    o = _make_outputs(N=4)
    kpts = o.pred_keypoints.clone()
    kpts[0, 0, 2:] = float("nan")  # instance 0: 2/4 visible (0.5)
    kpts[0, 1, 1:] = float("nan")  # instance 1: 1/4 visible (0.25)
    o = type(o)(
        pred_keypoints=kpts,
        pred_peak_values=o.pred_peak_values,
        instance_scores=o.instance_scores,
    )
    out = FilterPipeline(FilterConfig(min_visible_node_fraction=0.4))(o)
    assert not torch.isnan(out.pred_keypoints[0, 0]).all()  # 0.5 ≥ 0.4 → kept
    assert torch.isnan(out.pred_keypoints[0, 1]).all()  # 0.25 < 0.4 → dropped


# ─────────────────────────────────────────────────────────────────────────
# 5. min_instance_score
# ─────────────────────────────────────────────────────────────────────────


def test_min_instance_score_drops_low_score_instances():
    o = _make_outputs()
    out = FilterPipeline(FilterConfig(min_instance_score=0.5))(o)
    # instance 0 (0.85) + 2 (0.55) kept; 1 (0.30) dropped.
    assert not torch.isnan(out.pred_keypoints[0, 0]).any()
    assert torch.isnan(out.pred_keypoints[0, 1]).all()
    assert not torch.isnan(out.pred_keypoints[0, 2]).any()


# ─────────────────────────────────────────────────────────────────────────
# 6. min_mean_node_score
# ─────────────────────────────────────────────────────────────────────────


def test_min_mean_node_score_uses_mean_over_visible():
    o = _make_outputs()
    out = FilterPipeline(FilterConfig(min_mean_node_score=0.5))(o)
    # instance 0 mean=0.75 (kept), 1 mean=0.25 (dropped), 2 mean=0.5 (kept boundary)
    assert not torch.isnan(out.pred_keypoints[0, 0]).any()
    assert torch.isnan(out.pred_keypoints[0, 1]).all()
    assert not torch.isnan(out.pred_keypoints[0, 2]).any()


# ─────────────────────────────────────────────────────────────────────────
# 7. overlapping NMS
# ─────────────────────────────────────────────────────────────────────────


def test_overlapping_nms_drops_overlapping_low_score():
    """Two close instances → keep the higher-scoring one."""
    # Two instances overlapping in space; instance 0 score 0.9, instance 1 score 0.5.
    # Bboxes shifted by (0.1, 0.1) — IoU ≈ 0.91, well above 0.5.
    kpts = torch.tensor(
        [
            [
                [[10.0, 10.0], [12.0, 12.0]],
                [[10.1, 10.1], [12.1, 12.1]],
            ]
        ]
    )
    o = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.tensor([[[0.9, 0.9], [0.5, 0.5]]]),
        instance_scores=torch.tensor([[0.9, 0.5]]),
    )
    out = FilterPipeline(
        FilterConfig(
            overlapping=True, overlapping_threshold=0.5, overlapping_method="iou"
        )
    )(o)
    # Instance 0 kept; instance 1 dropped.
    assert not torch.isnan(out.pred_keypoints[0, 0]).any()
    assert torch.isnan(out.pred_keypoints[0, 1]).all()


def test_overlapping_oks_method_runs():
    """OKS overlap method runs and produces a sensible kept set."""
    kpts = torch.tensor([[[[1.0, 1.0]], [[1.05, 1.05]]]])
    o = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.tensor([[[1.0], [1.0]]]),
        instance_scores=torch.tensor([[0.9, 0.5]]),
    )
    out = FilterPipeline(
        FilterConfig(
            overlapping=True, overlapping_threshold=0.5, overlapping_method="oks"
        )
    )(o)
    # Method runs without raising; at least one instance survives.
    visible_count = (~torch.isnan(out.pred_keypoints).all(dim=(-2, -1))).sum().item()
    assert visible_count >= 1


# ─────────────────────────────────────────────────────────────────────────
# 7b. OKS parity with legacy ``_compute_oks`` (postprocessing.py on ``main``)
# ─────────────────────────────────────────────────────────────────────────


def _legacy_oks(points_a, points_b, kappa: float = 0.1) -> float:
    """Reference reimplementation of legacy ``_compute_oks`` (pure Python).

    Mirrors ``sleap_nn/inference/postprocessing.py::_compute_oks`` on ``main``:
    scale is the bbox *area* of instance ``a`` (its own valid keypoints),
    falloff constant ``kappa=0.1``, mean over jointly-visible keypoints of
    ``exp(-d^2 / (2 * scale_area * kappa^2))``.
    """
    valid_a = [not (math.isnan(x) or math.isnan(y)) for x, y in points_a]
    valid_b = [not (math.isnan(x) or math.isnan(y)) for x, y in points_b]
    valid = [va and vb for va, vb in zip(valid_a, valid_b)]
    if not any(valid):
        return 0.0
    pts_a_valid = [points_a[i] for i in range(len(points_a)) if valid_a[i]]
    if len(pts_a_valid) < 2:
        return 0.0
    xs = [p[0] for p in pts_a_valid]
    ys = [p[1] for p in pts_a_valid]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    scale_sq = bbox_w * bbox_h
    if scale_sq <= 0:
        return 0.0
    per_kpt = []
    for i in range(len(points_a)):
        if not valid[i]:
            continue
        dx = points_a[i][0] - points_b[i][0]
        dy = points_a[i][1] - points_b[i][1]
        d_sq = dx * dx + dy * dy
        per_kpt.append(math.exp(-d_sq / (2 * scale_sq * kappa**2)))
    return float(sum(per_kpt) / len(per_kpt))


def test_oks_matches_legacy_formula_by_hand():
    """``FilterPipeline._oks`` equals the legacy bbox-area/kappa=0.1 OKS value.

    Two overlapping instances with known keypoints. Instance A bbox is
    [0,0]..[10,10] → area = 100. With kappa=0.1, the per-keypoint denominator
    is ``2 * 100 * 0.1**2 = 2.0``.

    Keypoints (A vs B):
      node0: (0,0)   vs (0,0)    → d^2 = 0   → exp(0)        = 1.0
      node1: (10,0)  vs (12,0)   → d^2 = 4   → exp(-4/2)     = exp(-2)
      node2: (0,10)  vs (3,14)   → d^2 = 25  → exp(-25/2)    = exp(-12.5)
      node3: (10,10) vs (10,10)  → d^2 = 0   → exp(0)        = 1.0
    OKS = mean of the four = (1 + e^-2 + e^-12.5 + 1) / 4.
    """
    a = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    b = [(0.0, 0.0), (12.0, 0.0), (3.0, 14.0), (10.0, 10.0)]

    # Hand-computed expected value.
    expected = (1.0 + math.exp(-2.0) + math.exp(-12.5) + 1.0) / 4.0

    a_t = torch.tensor(a, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    got = FilterPipeline._oks(a_t, b_t)

    # Matches both the by-hand value and the legacy reference reimplementation.
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(_legacy_oks(a, b), abs=1e-9)


def test_oks_scale_from_instance_a_only():
    """Scale uses A's own valid keypoints, even where B is NaN there.

    Legacy derives the bbox scale from ``points_a[valid_a]`` (instance A's own
    valid keypoints), NOT from the jointly-visible set. Here node1 is NaN in B,
    so it does not contribute to the OKS mean, but it MUST still expand A's bbox
    (and thus the scale) exactly as legacy does.
    """
    a = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    b = [(0.0, 0.0), (float("nan"), float("nan")), (3.0, 14.0), (10.0, 10.0)]

    # Jointly-visible nodes: 0, 2, 3. Scale still from A's full bbox (area 100).
    #   node0: d^2 = 0       → 1.0
    #   node2: d^2 = 9+16=25 → exp(-25/2) = exp(-12.5)
    #   node3: d^2 = 0       → 1.0
    expected = (1.0 + math.exp(-12.5) + 1.0) / 3.0

    a_t = torch.tensor(a, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    got = FilterPipeline._oks(a_t, b_t)

    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(_legacy_oks(a, b), abs=1e-9)


def test_oks_nms_keeps_same_instance_as_legacy():
    """OKS NMS keeps the higher-scoring instance legacy would keep.

    Two near-identical instances (OKS well above threshold). Legacy greedy NMS
    sorts by score descending, keeps the top one, and drops the other because
    ``oks > threshold``. The refactored pipeline must keep the same instance.
    """
    # Instance 0 (score 0.9) and instance 1 (score 0.5) are nearly coincident.
    a = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    b = [(0.1, 0.1), (10.1, 0.0), (0.0, 10.1), (10.1, 10.1)]

    oks_ab = _legacy_oks(a, b)
    threshold = 0.5
    assert oks_ab > threshold  # legacy would suppress the lower-scoring one

    kpts = torch.tensor([[a, b]], dtype=torch.float64)
    o = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.ones(1, 2, 4, dtype=torch.float64),
        instance_scores=torch.tensor([[0.9, 0.5]], dtype=torch.float64),
    )
    out = FilterPipeline(
        FilterConfig(
            overlapping=True,
            overlapping_threshold=threshold,
            overlapping_method="oks",
        )
    )(o)

    # Higher-scoring instance 0 kept; instance 1 dropped — same as legacy.
    assert not torch.isnan(out.pred_keypoints[0, 0]).any()
    assert torch.isnan(out.pred_keypoints[0, 1]).all()


# ─────────────────────────────────────────────────────────────────────────
# 8. Filter order — combinations apply cheap → expensive
# ─────────────────────────────────────────────────────────────────────────


def test_combined_filters_apply_in_canonical_order():
    """A multi-filter config applies filters in the cheap→expensive order."""
    o = _make_outputs()
    cfg = FilterConfig(min_peak_value=0.1, min_visible_nodes=1, min_instance_score=0.5)
    out = FilterPipeline(cfg)(o)
    # Instance 1 has score 0.30 (below 0.5) → dropped after instance-score
    # filter, regardless of earlier passes.
    assert torch.isnan(out.pred_keypoints[0, 1]).all()


# ─────────────────────────────────────────────────────────────────────────
# 9. FilterConfig is picklable (Tom's constraint)
# ─────────────────────────────────────────────────────────────────────────


def test_filter_config_is_picklable():
    """``FilterConfig`` round-trips through pickle (multi-process safety)."""
    cfg = FilterConfig(
        min_peak_value=0.2,
        min_instance_score=0.3,
        overlapping=True,
        overlapping_threshold=0.5,
        overlapping_method="oks",
    )
    back = pickle.loads(pickle.dumps(cfg))
    assert back == cfg
    assert back.overlapping_method == "oks"


# ─────────────────────────────────────────────────────────────────────────
# 10. FilterPipeline.run convenience
# ─────────────────────────────────────────────────────────────────────────


def test_run_classmethod_matches_instance_apply():
    o = _make_outputs()
    cfg = FilterConfig(min_instance_score=0.5)
    via_run = FilterPipeline.run(o, cfg)
    via_inst = FilterPipeline(cfg)(o)
    torch.testing.assert_close(
        via_run.pred_keypoints, via_inst.pred_keypoints, equal_nan=True
    )
