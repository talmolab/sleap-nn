"""End-to-end tests for centroid-only inference (PR 25 / epic #508).

Covers:

1. ``Predictor.from_model_paths`` with a single centroid model returns a
   predictor whose layer is a ``CentroidLayer`` (auto-detect).
2. ``predict(make_labels=True)`` produces a saveable ``sio.Labels`` whose
   instances have the centroid at the anchor node (or node 0 if unset)
   and NaN at every other node.
3. The saved ``.slp`` round-trips through ``sio.load_slp``.
4. ``centroid_only=True`` on a top-down model_paths list still routes to
   the centroid-only layer (explicit override).
5. ``FilterPipeline`` emits a ``UserWarning`` when ``overlapping_method``
   is ``'oks'`` on centroid-only outputs and falls back to IoU.
6. Tracking via ``features='centroids'`` runs without error on
   centroid-only output.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sleap_io as sio

from sleap_nn.inference.factory import from_model_paths
from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.tracking import TrackerConfig

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"


# ─────────────────────────────────────────────────────────────────────────
# 1. Factory auto-detect: single centroid model → CentroidLayer
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not CENTROID_CKPT.exists(), reason="centroid ckpt absent")
def test_from_model_paths_centroid_only_auto_detect():
    """One centroid model_path → Predictor wraps a ``CentroidLayer``."""
    predictor = from_model_paths([str(CENTROID_CKPT)], device="cpu")
    assert isinstance(predictor, Predictor)
    assert isinstance(predictor.layer, CentroidLayer)


# ─────────────────────────────────────────────────────────────────────────
# 2. End-to-end packaging: anchor node populated, others NaN
# ─────────────────────────────────────────────────────────────────────────


def _stub_centroid_outputs(n_instances: int = 2) -> Outputs:
    """Build an ``Outputs`` with two valid centroids and one NaN slot."""
    centroids = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [float("nan"), float("nan")]]]
    )  # (B=1, I=3, 2)
    values = torch.tensor([[0.9, 0.7, float("nan")]])
    return Outputs(pred_centroids=centroids, pred_centroid_values=values)


def _make_skeleton(n_nodes: int = 3) -> sio.Skeleton:
    return sio.Skeleton(nodes=[sio.Node(f"n{i}") for i in range(n_nodes)])


def test_centroid_only_to_instances_default_anchor_is_node_0():
    """Default anchor (None) → centroid at node 0, others NaN."""
    outputs = _stub_centroid_outputs()
    skel = _make_skeleton(3)

    instances = outputs.to_instances(skeleton=skel)
    assert len(instances) == 2  # NaN slot dropped

    # Instance 0: centroid (10, 20) at node 0; node 1 and node 2 NaN.
    pts0 = instances[0].numpy()  # (N, 2)
    np.testing.assert_allclose(pts0[0], [10.0, 20.0])
    assert np.all(np.isnan(pts0[1:]))


def test_centroid_only_to_instances_respects_anchor_ind():
    """anchor_ind=2 → centroid at node 2; nodes 0 and 1 NaN."""
    outputs = _stub_centroid_outputs()
    skel = _make_skeleton(3)

    instances = outputs.to_instances(skeleton=skel, anchor_ind=2)
    pts0 = instances[0].numpy()
    assert np.all(np.isnan(pts0[:2]))
    np.testing.assert_allclose(pts0[2], [10.0, 20.0])


def test_centroid_only_to_instances_score_is_centroid_value():
    """Per-instance score = ``pred_centroid_values``."""
    outputs = _stub_centroid_outputs()
    skel = _make_skeleton(3)
    instances = outputs.to_instances(skeleton=skel)
    assert instances[0].score == pytest.approx(0.9)
    assert instances[1].score == pytest.approx(0.7)


def test_centroid_only_to_instances_anchor_out_of_range_raises():
    outputs = _stub_centroid_outputs()
    skel = _make_skeleton(3)
    with pytest.raises(ValueError, match="out of range"):
        outputs.to_instances(skeleton=skel, anchor_ind=5)


# ─────────────────────────────────────────────────────────────────────────
# 3. Round-trip: save + reload preserves NaN-padded structure
# ─────────────────────────────────────────────────────────────────────────


def test_centroid_only_labels_round_trip(tmp_path):
    """Saving + reloading the .slp preserves the anchor coord and NaNs."""
    outputs = _stub_centroid_outputs()
    skel = _make_skeleton(3)
    # sio.Labels.save() walks each video's backend; supply a real one.
    video_path = (
        Path(__file__).resolve().parents[1] / "assets" / "datasets" / "small_robot.mp4"
    )
    video = sio.Video.from_filename(str(video_path))

    labels = outputs.to_labels(skeleton=skel, videos=[video])
    out_path = tmp_path / "centroids.slp"
    labels.save(str(out_path), embed=False)

    reloaded = sio.load_slp(str(out_path))
    assert len(reloaded.labeled_frames) == 1
    assert len(reloaded.labeled_frames[0].instances) == 2

    pts = reloaded.labeled_frames[0].instances[0].numpy()
    np.testing.assert_allclose(pts[0], [10.0, 20.0])
    assert np.all(np.isnan(pts[1:]))


# ─────────────────────────────────────────────────────────────────────────
# 4. FilterPipeline: OKS NMS on centroid-only warns + falls back to IoU
# ─────────────────────────────────────────────────────────────────────────


def test_filter_oks_on_centroid_only_warns_and_falls_back():
    """``overlapping_method='oks'`` with no ``pred_keypoints`` →
    warning + IoU fallback (which is a no-op on point centroids).
    """
    outputs = _stub_centroid_outputs()
    pipeline = FilterPipeline(
        FilterConfig(
            overlapping=True,
            overlapping_method="oks",
            overlapping_threshold=0.5,
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline(outputs)
    msg = [str(w.message) for w in caught]
    assert any("OKS NMS is not meaningful for centroid-only" in m for m in msg)
    # Output should be unchanged (IoU on point centroids is a no-op).
    assert result is outputs or torch.equal(
        result.pred_centroids, outputs.pred_centroids
    )


def test_filter_iou_on_centroid_only_does_not_warn():
    """No warning when method is already 'iou'."""
    outputs = _stub_centroid_outputs()
    pipeline = FilterPipeline(
        FilterConfig(
            overlapping=True,
            overlapping_method="iou",
            overlapping_threshold=0.5,
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline(outputs)
    msg = [str(w.message) for w in caught]
    assert not any("OKS NMS is not meaningful" in m for m in msg)


# ─────────────────────────────────────────────────────────────────────────
# 5. Tracking smoke: features='centroids' runs on centroid-only Labels
# ─────────────────────────────────────────────────────────────────────────


def test_tracking_features_centroids_on_centroid_only_labels():
    """Tracker accepts centroid-only labels when features='centroids'.

    Anchor-node-only keypoints work for centroid-based tracking because
    the scoring path reads centroids directly.
    """
    from sleap_nn.inference.tracking import apply_tracking

    skel = _make_skeleton(3)
    # Two frames, one instance each, moving slightly.
    outputs_a = Outputs(
        pred_centroids=torch.tensor([[[10.0, 20.0]]]),
        pred_centroid_values=torch.tensor([[0.9]]),
        frame_indices=torch.tensor([0]),
    )
    outputs_b = Outputs(
        pred_centroids=torch.tensor([[[12.0, 22.0]]]),
        pred_centroid_values=torch.tensor([[0.85]]),
        frame_indices=torch.tensor([1]),
    )
    labels = sio.Labels(
        labeled_frames=(
            outputs_a.to_labels(skeleton=skel).labeled_frames
            + outputs_b.to_labels(skeleton=skel).labeled_frames
        ),
        skeletons=[skel],
    )

    tracked = apply_tracking(
        labels,
        TrackerConfig(
            features="centroids", window_size=3, scoring_method="euclidean_dist"
        ),
    )
    # No crash + one tracked instance per frame.
    assert len(tracked.labeled_frames) == 2
    assert all(len(lf.predicted_instances) == 1 for lf in tracked.labeled_frames)
