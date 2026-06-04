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


def test_apply_tracking_progress_callback(skeleton, video):
    n_frames = 5
    labels = _make_labels(skeleton, video, frames=n_frames, instances_per_frame=2)
    calls: list[tuple[int, int]] = []
    out = apply_tracking(
        labels, TrackerConfig(), progress_callback=lambda p, t: calls.append((p, t))
    )
    assert len(out.labeled_frames) == n_frames
    assert len(calls) == n_frames
    assert calls[0] == (1, n_frames)
    assert calls[-1] == (n_frames, n_frames)


def test_apply_tracking_no_callback_is_silent(skeleton, video):
    labels = _make_labels(skeleton, video, frames=3, instances_per_frame=1)
    out = apply_tracking(labels, TrackerConfig())
    assert len(out.labeled_frames) == 3


def test_retrack_progress_callback(skeleton, video):
    n_frames = 4
    labels = _make_labels(skeleton, video, frames=n_frames, instances_per_frame=2)
    calls: list[tuple[int, int]] = []
    out = Predictor.retrack(
        labels,
        TrackerConfig(),
        progress_callback=lambda p, t: calls.append((p, t)),
    )
    assert len(out.labeled_frames) == n_frames
    assert len(calls) == n_frames
    assert calls[-1] == (n_frames, n_frames)


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
    """End-to-end: stub the per-batch path + ``to_labels`` so ``predict()``
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
    # post-to_labels tracking hook without needing a real model.
    monkeypatch.setattr(
        Predictor,
        "_batch_iter",
        lambda self, provider, progress_callback=None: iter([]),
    )
    monkeypatch.setattr(
        Predictor,
        "to_labels",
        lambda self, outputs_list, videos=None: untracked,
    )

    result = pred.predict(
        _Provider(), make_labels=True, skeleton=skeleton, videos=[video]
    )
    assert isinstance(result, sio.Labels)
    for lf in result.labeled_frames:
        for inst in lf.instances:
            assert inst.track is not None


def test_predictor_predict_clean_empty_frames_drops_empty(skeleton, video, monkeypatch):
    """``clean_empty_frames=True`` drops empty LabeledFrames after to_labels."""
    import sleap_io as sio

    pred_inst = sio.PredictedInstance.from_numpy(
        points_data=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        skeleton=skeleton,
        score=0.9,
    )
    lfs = [
        sio.LabeledFrame(video=video, frame_idx=0, instances=[]),
        sio.LabeledFrame(video=video, frame_idx=1, instances=[pred_inst]),
        sio.LabeledFrame(video=video, frame_idx=2, instances=[]),
    ]
    raw_labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)

    pred = Predictor(layer=_StubLayer())

    class _Provider:
        def __iter__(self):
            return iter([])

    monkeypatch.setattr(
        Predictor,
        "_batch_iter",
        lambda self, provider, progress_callback=None: iter([]),
    )
    monkeypatch.setattr(
        Predictor,
        "to_labels",
        lambda self, outputs_list, videos=None: raw_labels,
    )

    result = pred.predict(
        _Provider(),
        make_labels=True,
        skeleton=skeleton,
        videos=[video],
        clean_empty_frames=True,
    )
    # Frame 1 (with the instance) survives; frames 0 and 2 are dropped.
    assert [lf.frame_idx for lf in result.labeled_frames] == [1]


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


# ──────────────────────────────────────────────────────────────────────
# Single-node (centroid) tracking default resolution (#586)
# ──────────────────────────────────────────────────────────────────────


def _capture_from_config_kwargs(monkeypatch):
    """Patch ``Tracker.from_config`` to record the kwargs it receives.

    Returns a dict that ``apply_tracking`` fills in with the resolved
    ``features`` / ``scoring_method`` (and friends). The stub returns a
    Tracker whose ``track`` is a no-op so ``apply_tracking`` runs end-to-end
    without exercising real association logic.
    """
    import sleap_nn.tracking.tracker as tracker_mod

    captured: dict = {}

    class _NoopTracker:
        def track(self, untracked_instances, frame_idx, image=None):
            return list(untracked_instances)

    def _fake_from_config(cls, **kwargs):
        captured.update(kwargs)
        return _NoopTracker()

    monkeypatch.setattr(
        tracker_mod.Tracker,
        "from_config",
        classmethod(_fake_from_config),
    )
    return captured


def test_apply_tracking_single_node_resolves_centroid_defaults(video, monkeypatch):
    """1-node Skeleton(['centroid']) + non-explicit config → euclidean/centroids."""
    centroid_skel = sio.Skeleton(nodes=["centroid"])
    pts = np.array([[5.0, 5.0]], dtype=np.float32)
    lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            sio.PredictedInstance.from_numpy(
                points_data=pts, skeleton=centroid_skel, score=0.9
            )
        ],
    )
    labels = sio.Labels(videos=[video], skeletons=[centroid_skel], labeled_frames=[lf])
    captured = _capture_from_config_kwargs(monkeypatch)

    apply_tracking(
        labels,
        TrackerConfig(scoring_method_explicit=False, features_explicit=False),
    )

    assert captured["scoring_method"] == "euclidean_dist"
    assert captured["features"] == "centroids"


def test_apply_tracking_multi_node_keeps_defaults(skeleton, video, monkeypatch):
    """Multi-node skeleton + non-explicit config → keep oks/keypoints."""
    labels = _make_labels(skeleton, video, frames=1, instances_per_frame=1)
    captured = _capture_from_config_kwargs(monkeypatch)

    apply_tracking(
        labels,
        TrackerConfig(scoring_method_explicit=False, features_explicit=False),
    )

    assert captured["scoring_method"] == "oks"
    assert captured["features"] == "keypoints"


def test_apply_tracking_single_node_explicit_not_overridden(video, monkeypatch):
    """1-node skeleton + explicit scoring_method='oks' → NOT overridden."""
    centroid_skel = sio.Skeleton(nodes=["centroid"])
    pts = np.array([[5.0, 5.0]], dtype=np.float32)
    lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            sio.PredictedInstance.from_numpy(
                points_data=pts, skeleton=centroid_skel, score=0.9
            )
        ],
    )
    labels = sio.Labels(videos=[video], skeletons=[centroid_skel], labeled_frames=[lf])
    captured = _capture_from_config_kwargs(monkeypatch)

    # scoring_method explicit (default True) → keep 'oks'; features left
    # non-explicit → still resolve to 'centroids'.
    apply_tracking(
        labels,
        TrackerConfig(
            scoring_method="oks",
            scoring_method_explicit=True,
            features_explicit=False,
        ),
    )

    assert captured["scoring_method"] == "oks"
    assert captured["features"] == "centroids"


def test_build_tracker_config_explicit_sentinels():
    """_build_tracker_config records *_explicit from kwargs presence (#586)."""
    from sleap_nn.cli import _build_tracker_config

    # Both unset (CLI sentinel None) → not explicit, fall back to oks/keypoints.
    cfg = _build_tracker_config({"features": None, "scoring_method": None})
    assert cfg.features_explicit is False
    assert cfg.scoring_method_explicit is False
    assert cfg.features == "keypoints"
    assert cfg.scoring_method == "oks"

    # Both set explicitly → explicit, values forwarded verbatim.
    cfg2 = _build_tracker_config({"features": "bboxes", "scoring_method": "iou"})
    assert cfg2.features_explicit is True
    assert cfg2.scoring_method_explicit is True
    assert cfg2.features == "bboxes"
    assert cfg2.scoring_method == "iou"

    # candidates_method: unset (None) → not explicit; set → explicit.
    assert (
        _build_tracker_config({"candidates_method": None}).candidates_method_explicit
        is False
    )
    assert (
        _build_tracker_config(
            {"candidates_method": "local_queues"}
        ).candidates_method_explicit
        is True
    )


# ──────────────────────────────────────────────────────────────────────
# Mask-IoU tracking for bottom-up segmentation (#619)
# ──────────────────────────────────────────────────────────────────────


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


def _make_mask_labels(video, frames_pts, h=80, w=80, score=0.9):
    """Mask-only labels: ``frames_pts`` is a list (per frame) of (cy, cx) disk
    centers; each becomes a PredictedSegmentationMask. No skeleton, no
    predicted keypoint instances (a bottom-up segmentation output)."""
    lfs = []
    for i, pts in enumerate(frames_pts):
        masks = [
            sio.PredictedSegmentationMask.from_numpy(
                _disk(h, w, cy, cx, 10), score=score
            )
            for cy, cx in pts
        ]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, masks=masks))
    return sio.Labels(videos=[video], labeled_frames=lfs)


def _mask_cfg(**kw):
    """TrackerConfig with features/scoring left implicit so seg auto-default fires."""
    kw.setdefault("features_explicit", False)
    kw.setdefault("scoring_method_explicit", False)
    return TrackerConfig(**kw)


def _make_stride_mask_labels(video, frames_pts, h=80, w=80, score=0.9, stride=2):
    """Like ``_make_mask_labels`` but masks are stored at output-stride resolution
    (``scale=1/stride``), as the default inference path (``full_res_masks=False``)
    emits. Disk centers/radius are in IMAGE pixels."""
    yy, xx = np.ogrid[:h, :w]
    lfs = []
    for i, pts in enumerate(frames_pts):
        masks = []
        for cy, cx in pts:
            full = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 10**2
            small = full[::stride, ::stride]
            s = 1.0 / stride
            masks.append(
                sio.PredictedSegmentationMask.from_numpy(
                    small, scale=(s, s), score=score
                )
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, masks=masks))
    return sio.Labels(videos=[video], labeled_frames=lfs)


def test_apply_tracking_stride_masks_keep_identity(video):
    """Output-stride masks (scale!=1, the default path) must track by image-grid IoU.

    Regression (finalize-review blocker): `get_mask` used to crop stride-res
    `.data` with image-space bbox indices -> empty features -> IoU 1.0 for every
    pair -> arbitrary identity. A drifting two-lane scene must yield two stable
    tracks, exactly as the scale-1 case does.
    """
    frames = [[(20 + i, 20 + i), (60 + i, 60 + i)] for i in range(6)]
    labels = _make_stride_mask_labels(video, frames)
    # sanity: masks really are stride-stored
    assert tuple(labels.labeled_frames[0].masks[0].scale) != (1.0, 1.0)
    out = apply_tracking(labels, _mask_cfg(window_size=5))
    assert all(len(lf.masks) == 2 for lf in out.labeled_frames)
    lanes = {}
    for lf in out.labeled_frames:
        for m in lf.masks:
            assert m.track is not None
            lane = "A" if m.bbox[0] < 40 else "B"
            lanes.setdefault(lane, set()).add(m.track.name)
    assert all(len(names) == 1 for names in lanes.values())  # each lane = 1 track
    assert lanes["A"] != lanes["B"]  # and they are distinct
    # Decisive bug tell: with the old stride bug every feature decoded to area 0,
    # so every score was the degenerate empty-vs-empty IoU 1.0. The fix yields
    # real partial overlap for the drifting disks, so later-frame matches score
    # strictly below 1.0.
    later_scores = [m.tracking_score for lf in out.labeled_frames[1:] for m in lf.masks]
    assert later_scores and all(0.0 < s for s in later_scores)
    assert any(s < 0.999 for s in later_scores)


def test_apply_tracking_masks_derived_cap_e2e(video):
    """End-to-end: the mask-mode derived `max_tracks` cap actually caps tracks.

    Three non-overlapping lanes (an over-split stand-in). With a target count of 2
    (candidates_method left unset -> local_queues), exactly 2 tracks survive;
    uncapped, all 3 spawn. Exercises the REAL tracker, not a NoopTracker.
    """
    frames = [[(15, 15), (40, 40), (65, 65)] for _ in range(5)]
    labels = _make_mask_labels(video, frames)

    capped = apply_tracking(
        labels,
        _mask_cfg(candidates_method_explicit=False, tracking_target_instance_count=2),
    )
    n_capped = len(
        {m.track.name for lf in capped.labeled_frames for m in lf.masks if m.track}
    )
    assert n_capped == 2

    # reset tracks (apply_tracking mutates the shared mask objects' .track)
    for lf in labels.labeled_frames:
        for m in lf.masks:
            m.track = None
    uncapped = apply_tracking(labels, _mask_cfg(candidates_method_explicit=False))
    n_uncapped = len(
        {m.track.name for lf in uncapped.labeled_frames for m in lf.masks if m.track}
    )
    assert n_uncapped == 3


def test_apply_tracking_masks_preserved_and_stable(video):
    """Mask tracking preserves every mask and gives a moving pair stable ids."""
    # Two animals drifting; each frame's masks overlap only their predecessor.
    frames = [[(20 + i, 20 + i), (60 + i, 60 + i)] for i in range(6)]
    labels = _make_mask_labels(video, frames)
    out = apply_tracking(labels, _mask_cfg(window_size=5))

    assert len(out.labeled_frames) == 6
    # No masks dropped; instances stay empty (mask-only model).
    assert all(len(lf.masks) == 2 for lf in out.labeled_frames)
    assert all(len(lf.instances) == 0 for lf in out.labeled_frames)
    # Every tracked mask carries a track + finite tracking_score.
    all_masks = [m for lf in out.labeled_frames for m in lf.masks]
    assert all(m.track is not None for m in all_masks)
    assert all(m.tracking_score is not None for m in all_masks)
    # Exactly two stable tracks, one per spatial lane across all frames.
    lanes = {}
    for lf in out.labeled_frames:
        for m in lf.masks:
            lane = "A" if m.bbox[0] < 40 else "B"
            lanes.setdefault(lane, set()).add(m.track.name)
    assert all(len(names) == 1 for names in lanes.values())
    assert lanes["A"] != lanes["B"]


def test_apply_tracking_masks_disjoint_extra_new_track(video):
    """An extra non-overlapping mask spawns a new track id."""
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22), (60, 60)]])
    out = apply_tracking(labels, _mask_cfg(window_size=5))
    f1 = sorted(out.labeled_frames, key=lambda lf: lf.frame_idx)[1]
    by_lane = {("near" if m.bbox[0] < 40 else "far"): m.track.name for m in f1.masks}
    assert by_lane["near"] != by_lane["far"]


def test_apply_tracking_masks_auto_default(video, caplog):
    """Mask-only labels auto-resolve features='masks'/scoring='mask_iou'."""
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    out = apply_tracking(labels, _mask_cfg(window_size=5))
    tracked = [m for lf in out.labeled_frames for m in lf.masks]
    assert tracked and all(m.track is not None for m in tracked)


def test_apply_tracking_masks_bumps_default_window(video, monkeypatch):
    """Mask mode raises the default window_size; an explicit value is kept."""
    import sleap_nn.inference.tracking as trk
    import sleap_nn.tracking.tracker as trk_mod

    captured = {}
    real_from_config = trk_mod.Tracker.from_config

    def _spy(*args, **kwargs):
        captured["window_size"] = kwargs.get("window_size")
        return real_from_config(*args, **kwargs)

    monkeypatch.setattr(trk_mod.Tracker, "from_config", staticmethod(_spy))
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])

    # Default window (DEFAULT_WINDOW_SIZE) -> bumped to DEFAULT_MASK_WINDOW_SIZE.
    apply_tracking(labels, _mask_cfg(window_size=trk.DEFAULT_WINDOW_SIZE))
    assert captured["window_size"] == trk.DEFAULT_MASK_WINDOW_SIZE

    # An explicit non-default window is respected (the user's choice).
    apply_tracking(labels, _mask_cfg(window_size=8))
    assert captured["window_size"] == 8


def test_apply_tracking_masks_defaults_local_queues(video, monkeypatch):
    """Mask mode defaults candidates_method to local_queues when not explicit."""
    captured = _capture_from_config_kwargs(monkeypatch)
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    apply_tracking(labels, _mask_cfg(window_size=5, candidates_method_explicit=False))
    assert captured["candidates_method"] == "local_queues"


def test_apply_tracking_masks_explicit_candidates_method_respected(video, monkeypatch):
    """An explicit candidates_method is honored in mask mode (no override)."""
    captured = _capture_from_config_kwargs(monkeypatch)
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    apply_tracking(
        labels,
        _mask_cfg(
            window_size=5,
            candidates_method="fixed_window",
            candidates_method_explicit=True,
        ),
    )
    assert captured["candidates_method"] == "fixed_window"


def test_apply_tracking_masks_derives_max_tracks_from_target(video, monkeypatch):
    """Mask mode caps tracks at the known target count when max_tracks is unset.

    The cap is what lifts local_queues identity from ~0.52 to ~0.91 on the real
    over-segmented clip; deriving it from the target count makes
    --tracking_target_instance_count N enough.
    """
    captured = _capture_from_config_kwargs(monkeypatch)
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    apply_tracking(
        labels,
        _mask_cfg(
            window_size=5,
            candidates_method_explicit=False,
            tracking_target_instance_count=3,
        ),
    )
    assert captured["max_tracks"] == 3
    # An explicit max_tracks wins over the derived one.
    captured.clear()
    apply_tracking(
        labels,
        _mask_cfg(
            window_size=5,
            candidates_method_explicit=False,
            max_tracks=2,
            tracking_target_instance_count=3,
        ),
    )
    assert captured["max_tracks"] == 2


def test_apply_tracking_masks_explicit_masks_respected(video):
    """Explicit features='masks'+scoring='mask_iou' is honored (no override/raise)."""
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    cfg = TrackerConfig(
        window_size=5, features="masks", scoring_method="mask_iou"
    )  # explicit (both *_explicit default True)
    out = apply_tracking(labels, cfg)
    assert all(m.track is not None for lf in out.labeled_frames for m in lf.masks)


def test_apply_tracking_masks_explicit_incompatible_raises(video):
    """A seg model with explicit non-mask features fails fast (no silent drop)."""
    labels = _make_mask_labels(video, [[(20, 20)]])
    cfg = TrackerConfig(features="keypoints", scoring_method="oks")  # explicit
    with pytest.raises(ValueError, match="features='masks'"):
        apply_tracking(labels, cfg)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"use_flow": True}, "motion models"),
        ({"use_kalman": True, "tracking_target_instance_count": 2}, "motion models"),
        ({"tracking_clean_instance_count": 2}, "cull/clean"),
        (
            {"tracking_pre_cull_to_target": 1, "tracking_target_instance_count": 2},
            "cull/clean",
        ),
    ],
)
def test_apply_tracking_masks_forbidden_combos_raise(video, kw, match):
    """Motion models and pose-shaped cull/clean ops are rejected in mask mode."""
    labels = _make_mask_labels(video, [[(20, 20)], [(22, 22)]])
    with pytest.raises(ValueError, match=match):
        apply_tracking(labels, _mask_cfg(window_size=5, **kw))


def test_apply_tracking_masks_empty_frame_passthrough(video):
    """A mask-less frame mid-sequence passes through (masks=[]) without crashing."""
    labels = _make_mask_labels(video, [[(20, 20)], [], [(24, 24)]])
    out = apply_tracking(labels, _mask_cfg(window_size=5))
    by_idx = {lf.frame_idx: lf for lf in out.labeled_frames}
    assert len(by_idx[1].masks) == 0
    assert len(by_idx[0].masks) == 1 and len(by_idx[2].masks) == 1


def test_apply_tracking_masks_roundtrip_persistence(video, tmp_path):
    """Tracked mask track names + tracking scores survive save/load_slp."""
    labels = _make_mask_labels(video, [[(20, 20), (60, 60)], [(22, 22), (62, 62)]])
    out = apply_tracking(labels, _mask_cfg(window_size=5))
    path = tmp_path / "tracked_masks.slp"
    out.save(path.as_posix())
    reloaded = sio.load_slp(path.as_posix())
    masks = [m for lf in reloaded.labeled_frames for m in lf.masks]
    assert len(masks) == 4
    assert all(m.track is not None for m in masks)
    assert all(
        m.tracking_score is not None and np.isfinite(m.tracking_score) for m in masks
    )
