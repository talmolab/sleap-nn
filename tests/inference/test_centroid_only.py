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

from sleap_nn.inference.predictor import Predictor
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
    predictor = Predictor.from_model_paths([str(CENTROID_CKPT)], device="cpu")
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
# 4. FilterPipeline: overlap NMS is degenerate for points → warns + skips;
#    min_centroid_distance is the meaningful centroid de-dup knob.
# ─────────────────────────────────────────────────────────────────────────


def test_filter_overlapping_oks_on_centroid_only_warns_and_skips():
    """``overlapping`` on centroid-only output warns + leaves it unchanged."""
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
    assert any("centroid-only" in m and "min_centroid_distance" in m for m in msg)
    torch.testing.assert_close(
        result.pred_centroids, outputs.pred_centroids, equal_nan=True
    )


def test_filter_overlapping_iou_on_centroid_only_also_warns():
    """IoU overlap is equally degenerate for single points → also warns + skips."""
    outputs = _stub_centroid_outputs()
    pipeline = FilterPipeline(
        FilterConfig(
            overlapping=True, overlapping_method="iou", overlapping_threshold=0.5
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline(outputs)
    assert any("min_centroid_distance" in str(w.message) for w in caught)
    torch.testing.assert_close(
        result.pred_centroids, outputs.pred_centroids, equal_nan=True
    )


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


# ─────────────────────────────────────────────────────────────────────────
# 7. PR I keystone: single-node collapse + sio.Centroid emission
# ─────────────────────────────────────────────────────────────────────────

from sleap_nn.inference.centroid_convert import centroid_source_for_anchor

SLP_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "datasets"
    / "minimal_instance.pkg.slp"
)


def test_centroid_source_for_anchor_mapping():
    assert centroid_source_for_anchor(None) == "center_of_mass"
    assert centroid_source_for_anchor(2, ["a", "b", "c"]) == "anchor:c"
    assert centroid_source_for_anchor(1) == "anchor:1"
    assert centroid_source_for_anchor(2, None) == "anchor:2"


def test_outputs_to_labels_collapse_to_single_node():
    """collapse_skeleton → genuine 1-node 'centroid' instances (no NaN padding)."""
    outputs = _stub_centroid_outputs()
    labels = outputs.to_labels(
        skeleton=_make_skeleton(3), collapse_skeleton=sio.get_centroid_skeleton()
    )
    assert labels.skeletons[0].node_names == ["centroid"]
    lf = labels.labeled_frames[0]
    assert len(lf.instances) == 2
    for inst in lf.instances:
        pts = inst.numpy()
        assert pts.shape == (1, 2)
        assert not np.any(np.isnan(pts))
    np.testing.assert_allclose(lf.instances[0].numpy()[0], [10.0, 20.0])


def test_genuine_single_node_model_unaffected():
    """A genuinely 1-node skeleton (collapse_skeleton=None) is emitted verbatim."""
    outputs = _stub_centroid_outputs()
    skel1 = _make_skeleton(1)
    insts = outputs.to_instances(skeleton=skel1)
    assert len(insts) == 2
    assert insts[0].numpy().shape == (1, 2)
    np.testing.assert_allclose(insts[0].numpy()[0], [10.0, 20.0])


def test_outputs_to_labels_emit_centroid_mode():
    outputs = _stub_centroid_outputs()
    labels = outputs.to_labels(
        skeleton=_make_skeleton(3),
        collapse_skeleton=sio.get_centroid_skeleton(),
        emit_centroid="centroid",
        source="anchor:B",
    )
    lf = labels.labeled_frames[0]
    assert len(lf.instances) == 0
    assert len(lf.centroids) == 2
    c0 = lf.centroids[0]
    assert (c0.x, c0.y) == pytest.approx((10.0, 20.0))
    assert c0.score == pytest.approx(0.9)
    assert c0.source == "anchor:B"


def test_outputs_to_labels_emit_both():
    outputs = _stub_centroid_outputs()
    labels = outputs.to_labels(
        skeleton=_make_skeleton(3),
        collapse_skeleton=sio.get_centroid_skeleton(),
        emit_centroid="both",
    )
    lf = labels.labeled_frames[0]
    assert len(lf.instances) == 2
    assert len(lf.centroids) == 2
    np.testing.assert_allclose(
        lf.instances[0].numpy()[0], [lf.centroids[0].x, lf.centroids[0].y]
    )


def test_to_centroids_skips_nan_and_tags_source():
    outputs = _stub_centroid_outputs()
    cents = outputs.to_centroids(source="center_of_mass")
    assert len(cents) == 2  # NaN slot dropped
    assert all(c.source == "center_of_mass" for c in cents)
    assert cents[0].score == pytest.approx(0.9)


def test_predicted_centroid_round_trip(tmp_path):
    """save + reload preserves PredictedCentroid x/y/score/source."""
    outputs = _stub_centroid_outputs()
    video = sio.Video.from_filename(
        str(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "datasets"
            / "small_robot.mp4"
        )
    )
    labels = outputs.to_labels(
        skeleton=_make_skeleton(3),
        videos=[video],
        collapse_skeleton=sio.get_centroid_skeleton(),
        emit_centroid="centroid",
        source="anchor:B",
    )
    out_path = tmp_path / "centroids.slp"
    labels.save(str(out_path), embed=False)
    reloaded = sio.load_slp(str(out_path))
    cents = reloaded.labeled_frames[0].centroids
    assert len(cents) == 2
    assert cents[0].source == "anchor:B"
    assert cents[0].score == pytest.approx(0.9)
    np.testing.assert_allclose([cents[0].x, cents[0].y], [10.0, 20.0])


# ─────────────────────────────────────────────────────────────────────────
# 8. Centroid-aware filters
# ─────────────────────────────────────────────────────────────────────────


def test_filter_min_instance_score_centroid_only():
    outputs = _stub_centroid_outputs()  # values 0.9, 0.7, nan
    result = FilterPipeline(FilterConfig(min_instance_score=0.8))(outputs)
    cv = result.pred_centroid_values[0]
    assert not torch.isnan(cv[0])  # 0.9 kept
    assert torch.isnan(cv[1])  # 0.7 dropped
    assert torch.isnan(result.pred_centroids[0, 1]).all()


def test_filter_min_centroid_distance_nms():
    centroids = torch.tensor([[[10.0, 20.0], [11.0, 20.0], [100.0, 100.0]]])
    values = torch.tensor([[0.9, 0.6, 0.8]])
    outputs = Outputs(pred_centroids=centroids, pred_centroid_values=values)
    result = FilterPipeline(FilterConfig(min_centroid_distance=5.0))(outputs)
    cv = result.pred_centroid_values[0]
    assert not torch.isnan(cv[0])  # higher-scored of the close pair kept
    assert torch.isnan(cv[1])  # within 5px of a kept centroid → dropped
    assert not torch.isnan(cv[2])  # far centroid kept


def test_nan_out_where_clears_centroid_tensors():
    outputs = _stub_centroid_outputs()
    drop = torch.tensor([[True, False, False]])
    result = FilterPipeline._nan_out_where(drop, outputs)
    assert torch.isnan(result.pred_centroids[0, 0]).all()
    assert torch.isnan(result.pred_centroid_values[0, 0])
    assert not torch.isnan(result.pred_centroids[0, 1]).any()


# ─────────────────────────────────────────────────────────────────────────
# 9. End-to-end (real checkpoint): collapse, emit, top-down-not-collapsed
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_centroid_only_predict_collapses_to_single_node():
    pred = Predictor.from_model_paths([str(CENTROID_CKPT)], device="cpu")
    labels = pred.predict(str(SLP_FIXTURE), make_labels=True)
    assert labels.skeletons[0].node_names == ["centroid"]
    insts = [i for lf in labels.labeled_frames for i in lf.instances]
    assert insts, "expected at least one predicted centroid instance"
    for inst in insts:
        assert inst.numpy().shape == (1, 2)


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_centroid_only_predict_provider_source_is_saveable(tmp_path):
    """Regression #699: predicting from a pre-built ``Provider`` must attach the
    real source video to the output ``Labels``.

    The CLI wraps the input in a ``LabelsProvider`` whenever a frame-selection
    flag is set (``--only_suggested_frames``, ``--exclude_user_labeled``,
    ``--video_index`` …). Previously ``Predictor._make_provider`` dropped the
    videos for a pre-built provider (returned ``videos=None``), so the output
    ``Labels`` got a ``None`` placeholder video and ``sio.Labels.save`` crashed
    with ``AttributeError: 'NoneType' object has no attribute 'backend'``.
    """
    from sleap_nn.inference.providers import LabelsProvider

    labels_in = sio.load_slp(str(SLP_FIXTURE))
    provider = LabelsProvider(labels=labels_in, batch_size=4)
    # The provider surfaces its source videos for output packaging.
    assert provider.videos == list(labels_in.videos)
    assert provider.videos and all(v is not None for v in provider.videos)

    pred = Predictor.from_model_paths([str(CENTROID_CKPT)], device="cpu")
    labels = pred.predict(provider, make_labels=True)

    # Output references the real video, not a None placeholder.
    assert labels.videos and all(v is not None for v in labels.videos)

    # The crash was inside Labels.save (it walks each video's backend); saving
    # and reloading must now succeed.
    out = tmp_path / "preds.slp"
    labels.save(str(out), embed=False)
    reloaded = sio.load_slp(str(out))
    assert reloaded.videos and all(v is not None for v in reloaded.videos)


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_centroid_only_predict_emit_both():
    pred = Predictor.from_model_paths(
        [str(CENTROID_CKPT)], device="cpu", emit_centroid="both"
    )
    labels = pred.predict(str(SLP_FIXTURE), make_labels=True)
    lf = next((f for f in labels.labeled_frames if f.instances or f.centroids), None)
    assert lf is not None
    assert len(lf.instances) >= 1
    assert len(lf.centroids) >= 1
    src = lf.centroids[0].source
    assert src == "center_of_mass" or src.startswith("anchor:")


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid+centered ckpts / slp fixture absent",
)
def test_topdown_not_collapsed():
    """Two-model top-down still emits the full multi-node skeleton (no collapse)."""
    pred = Predictor.from_model_paths(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)], device="cpu"
    )
    pkg = pred._resolve_centroid_packaging()
    assert pkg.collapse_skeleton is None
    assert pkg.emit_centroid == "instance"
    labels = pred.predict(str(SLP_FIXTURE), make_labels=True)
    assert labels.skeletons[0].node_names == ["A", "B"]


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_centroid_only_predict_to_file_collapses(tmp_path):
    """Streamed .slp matches in-memory: collapsed single-node 'centroid' skeleton."""
    pred = Predictor.from_model_paths([str(CENTROID_CKPT)], device="cpu")
    out = tmp_path / "stream.slp"
    pred.predict_to_file(str(SLP_FIXTURE), str(out))
    reloaded = sio.load_slp(str(out))
    assert reloaded.skeletons[0].node_names == ["centroid"]


# ─────────────────────────────────────────────────────────────────────────
# 10. Legacy path is closed for centroid-only
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_legacy_run_inference_lone_centroid_raises():
    from sleap_nn.predict import run_inference

    with pytest.raises(ValueError, match="predict"):
        run_inference(
            data_path=str(SLP_FIXTURE),
            model_paths=[str(CENTROID_CKPT)],
            make_labels=True,
        )


def test_cli_predict_has_centroid_output_option():
    """The new --centroid-output option is wired onto `predict`, and the stale
    'NaN-padded' wording is gone from --centroid_only help."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    result = CliRunner().invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--centroid-output" in result.output
    assert "NaN-padded" not in result.output


@pytest.mark.skipif(
    not (CENTROID_CKPT.exists() and SLP_FIXTURE.exists()),
    reason="centroid ckpt / slp fixture absent",
)
def test_cli_track_lone_centroid_errors():
    """Legacy `track` with a lone centroid model errors, pointing to `predict`."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    result = CliRunner().invoke(
        cli,
        ["track", "--data_path", str(SLP_FIXTURE), "--model_paths", str(CENTROID_CKPT)],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "predict" in str(result.exception)


# ─────────────────────────────────────────────────────────────────────────
# 11. Guard: centroid emission is incompatible with tracking
# ─────────────────────────────────────────────────────────────────────────


def test_cli_filter_min_centroid_distance_wired():
    """`--filter_min_centroid_distance` reaches FilterConfig and is registered
    on the `predict` command."""
    from sleap_nn.cli import _build_filter_config, predict

    cfg = _build_filter_config({"filter_min_centroid_distance": 8.0})
    assert cfg is not None
    assert cfg.min_centroid_distance == 8.0
    # Zero / absent -> no-op config (None) when nothing else is set.
    assert _build_filter_config({"filter_min_centroid_distance": 0.0}) is None

    assert "filter_min_centroid_distance" in {p.name for p in predict.params}


def test_emit_centroid_with_tracking_raises():
    """emit_centroid != 'instance' + tracking must hard-error (tracker drops
    sio.PredictedCentroid). The guard fires before any inference work."""
    pred = Predictor(
        layer=object(), emit_centroid="centroid", tracker_config=TrackerConfig()
    )
    with pytest.raises(ValueError, match="Tracking"):
        pred.predict("anything")

    pred_both = Predictor(
        layer=object(), emit_centroid="both", tracker_config=TrackerConfig()
    )
    with pytest.raises(ValueError, match="Tracking"):
        pred_both.predict("anything")
