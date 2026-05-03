"""Tests for the tracking integration in the new ``Predictor`` flow."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.inference.filters import FilterConfig
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def skeleton():
    return sio.Skeleton(nodes=["head", "tail"])


@pytest.fixture
def video():
    return sio.Video(filename="dummy.mp4")


def _make_labels(skeleton, video, frames=10, instances_per_frame=2, drift=1.0):
    """Synthetic labels: ``frames`` frames, each with ``instances_per_frame``
    predicted instances drifting linearly across frames so OKS tracking can
    associate them deterministically."""
    base_points = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],  # instance A
            [[100.0, 0.0], [110.0, 0.0]],  # instance B
        ],
        dtype=np.float32,
    )
    lfs: list[sio.LabeledFrame] = []
    for i in range(frames):
        insts: list[sio.PredictedInstance] = []
        for k in range(instances_per_frame):
            pts = base_points[k] + np.array([drift * i, drift * i], dtype=np.float32)
            insts.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts, skeleton=skeleton, score=0.9
                )
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=insts))
    return sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)


# ──────────────────────────────────────────────────────────────────────
# TrackerConfig — value type contract
# ──────────────────────────────────────────────────────────────────────


def test_tracker_config_defaults_and_pickle():
    cfg = TrackerConfig()
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.window_size == cfg.window_size == 5
    assert restored.candidates_method == "fixed_window"
    assert restored.scoring_method == "oks"
    assert restored.tracking_target_instance_count is None
    assert restored.post_connect_single_breaks is False


def test_tracker_config_is_frozen():
    cfg = TrackerConfig(window_size=10)
    with pytest.raises(attr_err()):
        cfg.window_size = 999  # type: ignore[misc]


def attr_err():
    """Frozen attrs raises ``FrozenInstanceError`` (subclass of AttributeError)."""
    import attrs

    return attrs.exceptions.FrozenInstanceError


# ──────────────────────────────────────────────────────────────────────
# apply_tracking — labels-in / labels-out parity vs run_tracker
# ──────────────────────────────────────────────────────────────────────


def test_apply_tracking_assigns_track_ids(skeleton, video):
    labels = _make_labels(skeleton, video, frames=5, instances_per_frame=2)
    cfg = TrackerConfig(window_size=5, candidates_method="fixed_window")
    out = apply_tracking(labels, cfg)
    assert len(out.labeled_frames) == 5
    for lf in out.labeled_frames:
        # Every instance should have a Track assigned (or, for first-frame
        # spawns, a fresh one). None means "untracked"; we want all tracked.
        for inst in lf.instances:
            assert inst.track is not None


def test_apply_tracking_assigns_consistent_track_ids_across_frames(skeleton, video):
    """A drifting two-instance scene should keep both tracks across the
    full window. Validates the ``Tracker`` glue is wired correctly: the
    same track names appear on the corresponding instance in every frame.
    """
    labels = _make_labels(skeleton, video, frames=8, instances_per_frame=2)
    out = apply_tracking(
        labels, TrackerConfig(window_size=5, candidates_method="fixed_window")
    )
    track_names_per_frame = [
        sorted(inst.track.name for inst in lf.instances) for lf in out.labeled_frames
    ]
    # Two stable tracks across all 8 frames.
    expected = sorted({n for names in track_names_per_frame for n in names})
    assert len(expected) == 2
    for names in track_names_per_frame:
        assert names == expected


def test_apply_tracking_post_connect_requires_target_count(skeleton, video):
    labels = _make_labels(skeleton, video, frames=3, instances_per_frame=1)
    cfg = TrackerConfig(post_connect_single_breaks=True)
    with pytest.raises(ValueError, match="tracking_target_instance_count"):
        apply_tracking(labels, cfg)


def test_apply_tracking_preserves_videos_and_skeletons(skeleton, video):
    labels = _make_labels(skeleton, video, frames=2, instances_per_frame=1)
    out = apply_tracking(labels, TrackerConfig())
    assert list(out.videos) == [video]
    assert list(out.skeletons) == [skeleton]


# ──────────────────────────────────────────────────────────────────────
# Predictor — tracker_config wiring
# ──────────────────────────────────────────────────────────────────────


class _StubLayer:
    """Minimal :class:`InferenceLayer`-shaped stub for unit tests."""

    def predict(self, image, **kwargs):  # pragma: no cover — not exercised here
        raise NotImplementedError


def test_predictor_accepts_tracker_config():
    cfg = TrackerConfig(window_size=7)
    pred = Predictor(layer=_StubLayer(), tracker_config=cfg)
    assert pred.tracker_config is cfg


def test_predictor_default_tracker_config_none():
    pred = Predictor(layer=_StubLayer())
    assert pred.tracker_config is None


def test_predictor_predict_streaming_raises_with_tracker_config():
    pred = Predictor(layer=_StubLayer(), tracker_config=TrackerConfig())

    class _Provider:
        def __iter__(self):
            return iter([])

    with pytest.raises(ValueError, match="tracker_config is not supported"):
        list(pred.predict_streaming(_Provider()))


def test_predictor_predict_no_make_labels_with_tracker_raises():
    pred = Predictor(layer=_StubLayer(), tracker_config=TrackerConfig())

    class _Provider:
        def __iter__(self):
            return iter([])

    with pytest.raises(ValueError, match="tracker_config requires make_labels"):
        pred.predict(_Provider(), make_labels=False)


def test_predictor_predict_applies_tracker_after_to_labels(
    skeleton, video, monkeypatch
):
    """End-to-end: stub the per-batch path + ``_to_labels`` so ``predict()``
    produces a known untracked Labels, then verify ``apply_tracking``
    runs and the returned Labels has tracks set."""
    untracked = _make_labels(skeleton, video, frames=4, instances_per_frame=2)

    pred = Predictor(
        layer=_StubLayer(),
        tracker_config=TrackerConfig(window_size=3),
    )

    class _Provider:
        def __iter__(self):
            return iter([])

    # Patch the class-level methods so ``predict()`` reaches the
    # post-_to_labels tracking hook without needing a real model.
    monkeypatch.setattr(Predictor, "_batch_iter", lambda self, provider: iter([]))
    monkeypatch.setattr(
        Predictor,
        "_to_labels",
        staticmethod(lambda outputs_list, skeleton, videos: untracked),
    )

    result = pred.predict(
        _Provider(), make_labels=True, skeleton=skeleton, videos=[video]
    )
    assert isinstance(result, sio.Labels)
    for lf in result.labeled_frames:
        for inst in lf.instances:
            assert inst.track is not None


def test_predictor_with_tracker_picklable_round_trip():
    pred = Predictor(
        layer=_StubLayer(),
        filter_config=FilterConfig(),
        tracker_config=TrackerConfig(window_size=11, max_tracks=4),
    )
    # Predictor itself isn't necessarily picklable (the layer holds a
    # torch model), but the tracker_config field must be.
    restored_cfg = pickle.loads(pickle.dumps(pred.tracker_config))
    assert restored_cfg.window_size == 11
    assert restored_cfg.max_tracks == 4
