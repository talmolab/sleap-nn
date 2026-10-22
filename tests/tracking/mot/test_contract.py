"""Generic contract of every MOT tracker (`sleap_nn.tracking.mot.base`).

Parametrized over ``MOT_TRACKER_NAMES``: each test pins one clause of the
``BaseMOTTracker`` contract (never drops, stale labels cleared, empty frames,
time base, retroactive labeling, optional prepare/finalize, determinism, public
ids and naming, degenerate input, pose and mask carriers, score thresholds).
Scenes carry appearance vectors so appearance trackers run their full path.
"""

import gc
import json
import weakref

import numpy as np
import pytest
import sleap_io as sio
from _pytest.logging import LogCaptureFixture
from loguru import logger
from conftest import (
    build_tracker,
    count_id_switches,
    crossing_trajectories,
    distinct_names,
    linear_trajectories,
    make_mask,
    make_pose,
    make_scene,
    run_tracker,
    track_frames,
    track_names,
    with_gaps,
)

from sleap_nn.tracking.mot import (
    MOT_TRACKER_NAMES,
    MOT_TRACKERS,
    BaseMOTTracker,
    MOTParams,
    SORTTracker,
)
from sleap_nn.tracking.mot.calibration import frame_steps
from sleap_nn.tracking.mot.kalman import oru_replay
from sleap_nn.tracking.mot.track import TrackState

NAMES = MOT_TRACKER_NAMES


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru records into pytest's caplog."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def _trajectories(n_frames=24):
    """Three well-separated animals moving at constant velocity (px/frame)."""
    return linear_trajectories(
        [(40, 50), (40, 120), (60, 190)], [(3, 0), (3, 0.5), (2, -0.5)], n_frames
    )


def _scene(n_frames=24, **kwargs):
    kwargs.setdefault("embeddings", True)
    return make_scene(_trajectories(n_frames), **kwargs)


def _assert_labeled(obj):
    """``.track`` is a Track or None; ``.tracking_score`` a float in [0, 1]."""
    assert obj.track is None or isinstance(obj.track, sio.Track)
    assert isinstance(obj.tracking_score, float)
    assert 0.0 <= obj.tracking_score <= 1.0
    if obj.track is None:
        assert obj.tracking_score == 0.0


@pytest.mark.parametrize("name", NAMES)
def test_registry_class_attributes(name):
    """Every tracker is a BaseMOTTracker with its registry name and a filter."""
    tracker = build_tracker(name)
    assert isinstance(tracker, BaseMOTTracker)
    assert tracker.name == name
    assert tracker.kf is not None
    assert tracker._track_objects == {}
    json.dumps(tracker.resolved)


@pytest.mark.parametrize("name", NAMES)
def test_never_drops_and_preserves_order(name):
    """track() returns every input object, in order, each labeled or None."""
    scene = _scene(
        shuffle=True,
        duplicates=[(5, 0), (6, 1)],
        false_positives=[(7, (300.0, 20.0), 0.95), (8, (10.0, 230.0), 0.3)],
    )
    tracker = build_tracker(name)
    tracker.prepare(scene.frames)
    for frame_idx, objs in scene.frames:
        out = tracker.track(objs, frame_idx)
        assert isinstance(out, list)
        assert len(out) == len(objs)
        assert all(a is b for a, b in zip(out, objs))
        for obj in out:
            _assert_labeled(obj)
    tracker.finalize()
    for obj in scene.objects():
        _assert_labeled(obj)


@pytest.mark.parametrize("name", NAMES)
def test_tracks_well_separated_animals(name):
    """Three isolated animals with constant motion: three ids, no switch."""
    scene = _scene()
    names = run_tracker(build_tracker(name), scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    # Every detection after the confirmation window is tracked.
    assert all(n is not None for frame in names[5:] for n in frame)


@pytest.mark.parametrize("name", NAMES)
def test_clears_stale_tracks(name):
    """Labels from a previous run are cleared, also on never-tracked objects."""
    stale = sio.Track("stale")
    scene = _scene(stale_track=stale)
    run_tracker(build_tracker(name, min_hits=1000), scene.frames)
    for obj in scene.objects():
        assert obj.track is None
        assert obj.tracking_score == 0.0

    scene = _scene(stale_track=stale)
    run_tracker(build_tracker(name), scene.frames)
    assert all(obj.track is not stale for obj in scene.objects())


@pytest.mark.parametrize("name", NAMES)
def test_empty_frames(name):
    """Empty frames are fine anywhere (first, interleaved, whole video)."""
    scene = _scene(n_frames=20)
    frames = [(0, [])]
    for frame_idx, objs in scene.frames:
        frames.append((2 * frame_idx + 1, objs))
        frames.append((2 * frame_idx + 2, []))
    tracker = build_tracker(name)
    outputs = track_frames(tracker, frames)
    assert [len(o) for o in outputs] == [len(objs) for _, objs in frames]
    for obj in scene.objects():
        _assert_labeled(obj)

    tracker = build_tracker(name)
    outputs = track_frames(tracker, [(i, []) for i in range(10)])
    assert outputs == [[]] * 10
    assert tracker._track_objects == {}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("prepare", [True, False])
def test_frame_gap_keeps_identities(name, prepare):
    """A short gap in frame indices (frames time base) keeps every identity."""
    n_frames = 30
    traj = _trajectories(n_frames)
    keep = [t for t in range(n_frames) if not 12 <= t < 15]  # 3 missing frames
    scene = make_scene(traj[keep], frame_indices=keep, embeddings=True)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames, prepare=prepare)
    assert tracker.resolved["time_base"] == "frames"
    assert tracker.resolved["frame_step"] == 1
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_regular_stride_uses_frames_time_base(name):
    """A stride-5 video: frame_step 5, one step per processed frame."""
    n_frames = 24
    traj = _trajectories(n_frames)
    scene = make_scene(traj, frame_indices=[5 * t for t in range(n_frames)])
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["time_base"] == "frames"
    assert tracker.resolved["frame_step"] == 5
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_irregular_frames_use_steps_time_base(name):
    """Irregularly spaced frames: one step per processed frame (auto -> steps)."""
    n_frames = 24
    rng = np.random.default_rng(3)
    frame_indices = np.cumsum(rng.integers(1, 60, size=n_frames)).tolist()
    scene = make_scene(_trajectories(n_frames), frame_indices=frame_indices)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["time_base"] == "steps"
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_retroactive_labeling(name):
    """Tentative detections stay untracked until confirmation, then get the track."""
    scene = _scene(n_frames=12)
    tracker = build_tracker(name, min_hits=3)
    tracker.prepare(scene.frames)
    first_idx, first_objs = scene.frames[0]
    tracker.track(first_objs, first_idx)
    assert all(obj.track is None for obj in first_objs)
    for frame_idx, objs in scene.frames[1:]:
        tracker.track(objs, frame_idx)
    tracker.finalize()
    names = track_names(scene.frames)
    # Frame 0 detections carry the same track as the rest of their identity.
    for gid, name0 in zip(scene.gt[0], names[0]):
        assert name0 is not None
        later = {n for n, g in zip(names[-1], scene.gt[-1]) if g == gid}
        assert later == {name0}


@pytest.mark.parametrize("name", NAMES)
def test_works_without_prepare_or_finalize(name):
    """Library loops: track() alone works (uncalibrated)."""
    scene = _scene()
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames, prepare=False, finalize=False)
    assert tracker.calibration is None
    assert tracker.resolved["calibration"] is None
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    json.dumps(tracker.resolved)


@pytest.mark.parametrize("name", NAMES)
def test_prepare_empty_is_noop(name):
    """prepare([]) does nothing and tracking still works."""
    tracker = build_tracker(name)
    tracker.prepare([])
    assert tracker.calibration is None
    scene = _scene()
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("link", [False, True])
def test_finalize_is_idempotent(name, link):
    """A second finalize() changes nothing (with and without linking)."""
    scene = _scene()
    tracker = build_tracker(name, link_tracklets=link)
    names = run_tracker(tracker, scene.frames)
    resolved = json.dumps(tracker.resolved, sort_keys=True)
    objects = dict(tracker._track_objects)
    tracker.finalize()
    assert track_names(scene.frames) == names
    assert json.dumps(tracker.resolved, sort_keys=True) == resolved
    assert tracker._track_objects == objects
    assert ("linking" in tracker.resolved) == link


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name):
    """Two runs on identical (but distinct) scenes give identical results."""
    kwargs = dict(
        shuffle=True,
        position_noise=1.0,
        nan_node_prob=0.2,
        duplicates=[(4, 2)],
        false_positives=[(9, (200.0, 30.0), 0.9)],
    )
    scene_a, scene_b = _scene(**kwargs), _scene(**kwargs)
    tracker_a, tracker_b = build_tracker(name), build_tracker(name)
    names_a = run_tracker(tracker_a, scene_a.frames)
    names_b = run_tracker(tracker_b, scene_b.frames)
    assert names_a == names_b
    scores_a = [o.tracking_score for o in scene_a.objects()]
    scores_b = [o.tracking_score for o in scene_b.objects()]
    assert scores_a == scores_b
    assert json.dumps(tracker_a.resolved, sort_keys=True) == json.dumps(
        tracker_b.resolved, sort_keys=True
    )


@pytest.mark.parametrize("name", NAMES)
def test_track_objects_keyed_by_public_ids(name):
    """_track_objects holds public ids 0..k-1 only; labels use those objects."""
    # Single-frame false positives spawn tentative tracks that never confirm, so
    # internal tids run ahead of public ids.
    fps = [(t, (300.0, 10.0 + 8 * t), 0.95) for t in range(0, 20, 2)]
    scene = _scene(false_positives=fps)
    tracker = build_tracker(name)
    run_tracker(tracker, scene.frames)
    keys = sorted(tracker._track_objects)
    assert keys == list(range(len(keys)))
    assert len(keys) == 3
    for key, track in tracker._track_objects.items():
        assert track.name == f"track_{key}"
    values = list(tracker._track_objects.values())
    for obj in scene.objects():
        assert obj.track is None or any(obj.track is v for v in values)


@pytest.mark.parametrize("name", NAMES)
def test_track_name_offset(name):
    """Names are track_{public_id + track_name_offset}; keys stay public ids."""
    scene = _scene()
    tracker = build_tracker(name, track_name_offset=7)
    names = run_tracker(tracker, scene.frames)
    assert sorted(tracker._track_objects) == [0, 1, 2]
    assert distinct_names(names) == ["track_7", "track_8", "track_9"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("prepare", [True, False])
def test_degenerate_poses_do_not_crash(name, prepare):
    """NaN nodes, all-NaN poses and single visible nodes never raise."""
    scene = _scene(nan_node_prob=0.4, min_visible=1)
    invalid = []
    for t, (_, objs) in enumerate(scene.frames):
        if t % 3 == 0:
            bad = make_pose((150.0, 20.0), all_nan=True)
            objs.append(bad)
            invalid.append(bad)
        if t % 4 == 1:
            objs.append(make_pose((250.0, 220.0), nan_nodes=[0, 2, 3, 4]))
    tracker = build_tracker(name)
    run_tracker(tracker, scene.frames, prepare=prepare)
    for obj in scene.objects():
        _assert_labeled(obj)
    assert all(obj.track is None for obj in invalid)


@pytest.mark.parametrize("name", NAMES)
def test_single_node_skeleton(name):
    """Single-node (centroid-only) detections track with centroid similarity."""
    scene = make_scene(_trajectories(), n_nodes=1, embeddings=True)
    tracker = build_tracker(name, n_nodes=1)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["similarity"] == "centroid"
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_unknown_skeleton_size_resolves_from_detections(name):
    """n_nodes=None: the similarity kind resolves from the first detections."""
    scene = _scene()
    tracker = build_tracker(name, n_nodes=None)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.resolved["similarity"] == "iou"
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_mask_carrier(name):
    """Masks track like poses; empty masks are never tracked."""
    scene = make_scene(_trajectories(), carrier="mask", embeddings=True)
    empty = []
    for t, (_, objs) in enumerate(scene.frames):
        if t % 5 == 2:
            mask = make_mask((10.0, 10.0), empty=True)
            objs.append(mask)
            empty.append(mask)
    tracker = build_tracker(name, carrier="mask", n_nodes=None)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["similarity"] == "iou"
    gt = [
        ids + [-1] * (len(objs) - len(ids))
        for ids, (_, objs) in zip(scene.gt, scene.frames)
    ]
    assert count_id_switches(names, gt) == 0
    assert len(distinct_names(names)) == 3
    assert all(mask.track is None for mask in empty)
    for obj in scene.objects():
        _assert_labeled(obj)


@pytest.mark.parametrize("name", NAMES)
def test_scores_below_thresholds_are_not_tracked(name, caplog):
    """Detections scoring below every threshold never start a track.

    With the animal preset's per-video normalization (scores / video q95), a
    detection is low RELATIVE to the video: here two animals score 0.9 and one
    0.01. On raw scores (normalize_scores=False) a uniformly low video is not
    tracked at all, and finalize() says so.
    """
    scores = np.tile([0.9, 0.9, 0.01], (24, 1))
    scene = _scene(scores=scores)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["score_normalizer"] == pytest.approx(0.9)
    assert len(distinct_names(names)) == 2
    low = [obj for t, (_, objs) in enumerate(scene.frames) for obj in objs[2:]]
    assert all(obj.track is None and obj.tracking_score == 0.0 for obj in low)

    scene = _scene(scores=0.01)
    tracker = build_tracker(name, normalize_scores=False)
    names = run_tracker(tracker, scene.frames)
    assert distinct_names(names) == []
    assert tracker._track_objects == {}
    assert tracker.resolved["score_normalizer"] is None
    assert all(obj.tracking_score == 0.0 for obj in scene.objects())
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("no track was ever confirmed" in m for m in warnings)
    assert any("can never start a track" in m for m in warnings)


@pytest.mark.parametrize("name", NAMES)
def test_low_scale_scores_are_normalized_per_video(name):
    """A model whose scores all sit low (q95 0.45) still starts tracks under the
    animal preset: scores are divided by the video's q95. Without prepare()
    there is no calibration and no normalization."""
    scene = _scene(scores=0.45)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    assert tracker.params.normalize_scores
    assert tracker.resolved["score_normalizer"] == pytest.approx(0.45)
    assert tracker.calibration.score_quantiles[-1] == pytest.approx(0.45)  # raw
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    dets = tracker._detections(list(scene.frames[0][1]), 0)
    np.testing.assert_allclose(dets.scores, 1.0)

    tracker = build_tracker(name)
    run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.resolved["score_normalizer"] is None
    dets = tracker._detections(list(scene.frames[0][1]), 0)
    np.testing.assert_allclose(dets.scores, 0.45)


@pytest.mark.parametrize("name", NAMES)
def test_user_instances(name):
    """User instances (no scores: confidence 1.0) are tracked."""
    scene = _scene(user=True)
    names = run_tracker(build_tracker(name), scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_tracking_score_is_a_similarity(name):
    """Spawn detections score 1.0; matched ones their geometric similarity."""
    scene = _scene()
    run_tracker(build_tracker(name), scene.frames)
    assert all(obj.tracking_score == 1.0 for obj in scene.frames[0][1])
    later = [obj.tracking_score for _, objs in scene.frames[5:] for obj in objs]
    assert all(0.0 < s <= 1.0 for s in later)


@pytest.mark.parametrize("name", NAMES)
def test_resolved_is_json_safe(name):
    """``resolved`` (provenance) holds plain JSON types after a full run."""
    scene = _scene()
    tracker = build_tracker(name, link_tracklets=True)
    run_tracker(tracker, scene.frames)
    record = json.loads(json.dumps(tracker.resolved))
    assert record["tracker"] == name
    assert record["params"]["tracker"] == name
    assert record["time_base"] in ("frames", "steps")
    assert record["similarity"] in ("iou", "oks", "centroid")
    assert isinstance(record["calibration"], dict)
    assert isinstance(record["appearance"], dict)
    assert "linking" in record
    assert record["body_size_source"] == "measured"
    assert record["calibration"]["time_base"] == record["time_base"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("stride", [2, 3])
def test_strided_frames_without_prepare(name, stride, caplog):
    """No prepare(), every k-th frame: the stride is inferred online.

    It used to stay 1, so every processed frame was a k-step gap: tentative
    tracks died / never reached their streak and NOTHING was tracked, silently.
    """
    n_frames = 24
    scene = make_scene(
        _trajectories(n_frames),
        frame_indices=[stride * t for t in range(n_frames)],
        embeddings=True,
    )
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.calibration is None
    assert tracker.resolved["time_base"] == "frames"
    assert tracker.resolved["frame_step"] == stride
    assert tracker._step == n_frames - 1
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    assert all(n is not None for frame in names[5:] for n in frame)
    assert not any("no track was ever confirmed" in m for m in _warnings(caplog))

    # Same names as the calibrated run.
    scene_p = make_scene(
        _trajectories(n_frames),
        frame_indices=[stride * t for t in range(n_frames)],
        embeddings=True,
    )
    assert run_tracker(build_tracker(name), scene_p.frames) == names


@pytest.mark.parametrize("name", ["sort", "bytetrack"])
def test_irregular_frames_without_prepare_count_steps(name, caplog):
    """No prepare(), irregular frame indices: one step per processed frame
    (as prepare() would decide), with a warning."""
    n_frames = 24
    rng = np.random.default_rng(3)
    frame_indices = np.cumsum(rng.integers(1, 60, size=n_frames)).tolist()
    scene = make_scene(_trajectories(n_frames), frame_indices=frame_indices)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.resolved["time_base"] == "steps"
    # (The first deltas, before the irregularity shows, may count a few steps.)
    assert n_frames - 1 <= tracker._step <= n_frames + 2
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    assert any("irregularly spaced" in m for m in _warnings(caplog))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("prepare", [True, False])
def test_step_rounding_matches_calibration(name, prepare):
    """The tracker counts the same steps as the calibration / linker helpers,
    including half-stride deltas (one rounding rule: half to even)."""
    idx = [0, 2, 4, 6, 11, 13, 15, 17, 19, 21, 23, 25]
    traj = _trajectories(26)[idx]
    scene = make_scene(traj, frame_indices=idx, embeddings=True)
    tracker = build_tracker(name)
    run_tracker(tracker, scene.frames, prepare=prepare)
    expected = frame_steps(idx, "frames", 2)
    assert expected[4] == 5  # delta 5 at stride 2 -> 2 steps
    assert tracker._step == expected[-1]
    if prepare:
        cal = tracker.calibration
        assert (cal.time_base, cal.frame_step) == ("frames", 2)


@pytest.mark.parametrize("name", NAMES)
def test_backwards_frames_warn(name, caplog):
    """Two videos fed to ONE tracker (frame indices restart): a warning, from
    prepare() and from track() (once)."""
    scene = _scene(n_frames=12)
    frames = scene.frames + _scene(n_frames=12, seed=1).frames
    tracker = build_tracker(name)
    run_tracker(tracker, frames)
    warnings = _warnings(caplog)
    assert sum("increasing frame order" in m for m in warnings) == 2
    assert any("prepare() got" in m for m in warnings)

    caplog.clear()
    tracker = build_tracker(name)
    run_tracker(tracker, frames, prepare=False)
    assert sum("does not come after" in m for m in _warnings(caplog)) == 1


def _single_node_scene(**kwargs):
    """Two centroid-only animals 600 px apart moving 3 px/frame; animal 0 is not
    detected on frames 20-22 and a confident false positive appears 250 px from
    it on frame 21."""
    traj = linear_trajectories([(100, 100), (100, 700)], [(3, 0), (3, 0)], 40)
    traj = with_gaps(traj, {0: range(20, 23)})
    fp = (float(traj[19, 0, 0] + 9.0), float(traj[19, 0, 1] + 250.0))
    return make_scene(
        traj, n_nodes=1, false_positives=[(21, fp, 0.95)], embeddings=True, **kwargs
    )


@pytest.mark.parametrize("name", NAMES)
def test_single_node_gate_is_sized_by_motion(name):
    """Centroid-only: the gate is sized by the calibrated motion, not by the
    inter-animal spacing, so a far false positive does not take the identity
    of a briefly undetected animal."""
    scene = _single_node_scene()
    tracker = build_tracker(name, n_nodes=1)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["similarity"] == "centroid"
    assert tracker.calibration.body_size_source == "motion"
    assert tracker.resolved["body_size_source"] == "motion"
    assert tracker._body_size() < 50.0  # was the 600 px spacing
    fp = [obj for obj, g in zip(scene.frames[21][1], scene.gt[21]) if g < 0]
    assert len(fp) == 1
    assert fp[0].track is None or fp[0].track.name != names[19][0]
    assert count_id_switches(names, scene.gt) == 0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("n_animals", [1, 2])
@pytest.mark.parametrize("prepare", [True, False])
def test_single_node_fast_animals(name, n_animals, prepare):
    """Centroid-only animals at 40 px/frame are tracked, calibrated or not, alone
    or not (a fixed 50 px body size could not follow them)."""
    starts = [(20, 100), (20, 600)][:n_animals]
    traj = linear_trajectories(starts, [(40, 0)] * n_animals, 30)
    scene = make_scene(traj, n_nodes=1, embeddings=True)
    tracker = build_tracker(name, n_nodes=1)
    names = run_tracker(tracker, scene.frames, prepare=prepare)
    assert len(distinct_names(names)) == n_animals
    assert count_id_switches(names, scene.gt) == 0
    assert all(n is not None for frame in names[5:] for n in frame)
    assert tracker.resolved["body_size_source"] == "motion"
    assert tracker._body_size() >= 8 * 40.0 - 1e-6 or n_animals == 2


@pytest.mark.parametrize("prepare", [True, False])
def test_body_size_override(prepare):
    """MOTParams.body_size wins over every estimate (and reaches the calibration,
    which the identity linker reads)."""
    scene = make_scene(_trajectories(), n_nodes=1)
    tracker = build_tracker("sort", n_nodes=1, body_size=70.0)
    run_tracker(tracker, scene.frames, prepare=prepare)
    assert tracker._body_size() == 70.0
    assert tracker.resolved["body_size_source"] == "override"
    if prepare:
        assert tracker.calibration.body_size == 70.0
        assert tracker.calibration.body_size_source == "override"
    dets = tracker._detections(list(scene.frames[0][1]), 0)
    side = dets.raw_boxes[:, 2] - dets.raw_boxes[:, 0]
    np.testing.assert_allclose(side, 70.0)


def test_next_track_name_offset():
    """Per-video trackers chained with next_track_name_offset never reuse names."""
    tracker = build_tracker("sort", track_name_offset=4)
    assert tracker.next_track_name_offset == 4
    run_tracker(tracker, _scene().frames)
    assert tracker.next_track_name_offset == 7


# ---------------------------------------------------------------------------
# Base lifecycle hooks, exercised through SORT subclasses (independent of the
# other trackers' implementations).
class _AppearanceSORT(SORTTracker):
    """SORT with the base appearance memory switched on."""

    uses_appearance = True


class _OruSORT(SORTTracker):
    """SORT with OC-SORT's freeze / observation-centric re-update."""

    use_oru = True


class _DyingSORT(SORTTracker):
    """SORT with DeepSORT/ByteTrack tentative semantics."""

    tentative_dies_on_miss = True
    hits_start = 1
    confirm_on_streak = False


class _ZeroVelocitySORT(SORTTracker):
    """SORT zeroing the XYSR center velocity of unmatched tracks."""

    lost_zero_velocity = (4, 5)


class _HistorySORT(SORTTracker):
    """SORT keeping OC-SORT's observation history and velocity direction."""

    keeps_observation_history = True


def _make(cls, **overrides):
    return cls(MOTParams.for_tracker("sort", **overrides), carrier="pose", n_nodes=5)


def _one_animal(n_frames, velocity=(5.0, 0.0)):
    return linear_trajectories([(40.0, 120.0)], [velocity], n_frames)


def test_appearance_memory_seeded_and_quality_gated():
    """Spawn seeds the memory; isolated matches update it; overlapping ones don't."""
    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 10)
    scene = make_scene(traj, embeddings=True)
    tracker = _make(_AppearanceSORT)
    run_tracker(tracker, scene.frames, prepare=False)
    for track in tracker.tracks:
        assert not track.appearance.is_empty
        assert len(track.appearance.gallery) == 10  # seed + 9 gated updates

    # Crossing: frames where the boxes overlap (IoU >= 0.1) are not absorbed.
    scene = make_scene(crossing_trajectories(n_frames=30, offset=12.0), embeddings=True)
    tracker = _make(_AppearanceSORT)
    run_tracker(tracker, scene.frames, prepare=False)
    galleries = [len(t.appearance.gallery) for t in tracker.tracks]
    assert all(5 < g < 30 for g in galleries)


def test_appearance_gate_resolution():
    """auto: disabled without calibration; a float applies once vectors exist."""
    scene = make_scene(linear_trajectories([(40, 60)], [(3, 0)], 6), embeddings=True)
    tracker = _make(_AppearanceSORT)
    assert tracker._appearance_gate() is None  # no vectors seen yet
    run_tracker(tracker, scene.frames, prepare=False)
    assert tracker._appearance_gate("ema") is None  # "auto", uncalibrated
    assert tracker._appearance_margin() == 0.0
    dets = tracker._detections(list(scene.frames[-1][1]), 6)
    assert np.isnan(tracker._appearance_cos(tracker.tracks, dets)).all()
    assert tracker.resolved["appearance"]["enabled"] is False

    scene = make_scene(linear_trajectories([(40, 60)], [(3, 0)], 6), embeddings=True)
    tracker = _make(_AppearanceSORT, appearance_threshold=0.4)
    run_tracker(tracker, scene.frames, prepare=False)
    assert tracker._appearance_gate("ema") == 0.4
    assert tracker._appearance_gate("gallery_max") == 0.4
    dets = tracker._detections(list(scene.frames[-1][1]), 6)
    for statistic in ("ema", "gallery_max"):
        cos = tracker._appearance_cos(tracker.tracks, dets, statistic)
        assert cos.shape == (1, 1) and 0.9 < cos[0, 0] <= 1.0
    with pytest.raises(ValueError, match="statistic"):
        tracker._appearance_gate("mean")
    record = json.loads(json.dumps(tracker.resolved))
    assert record["appearance"]["enabled"] is True
    assert record["appearance"]["ema_cos_gate"] == 0.4

    # Trackers that do not use appearance never gate on it.
    tracker = _make(SORTTracker, appearance_threshold=0.4)
    run_tracker(tracker, scene.frames, prepare=False)
    assert tracker._appearance_gate() is None


def _states(tracker):
    return [
        (t.tid, int(t.state), t.hits, t.hit_streak, t.time_since_update, t.age)
        for t in tracker.tracks
    ]


@pytest.mark.parametrize("cls", [SORTTracker, _OruSORT, _DyingSORT, _ZeroVelocitySORT])
def test_frame_gap_equals_empty_frames(cls):
    """frames time base: a k-frame gap behaves exactly like k empty frames."""
    n_frames = 24
    traj = _one_animal(n_frames)
    gap = range(10, 13)
    dense = make_scene(with_gaps(traj, {0: gap}))
    keep = [t for t in range(n_frames) if t not in gap]
    sparse = make_scene(traj[keep], frame_indices=keep)
    a, b = _make(cls), _make(cls)
    names_a = run_tracker(a, dense.frames, prepare=False)
    names_b = run_tracker(b, sparse.frames, prepare=False)
    assert [n for frame in names_a for n in frame] == [
        n for frame in names_b for n in frame
    ]
    assert _states(a) == _states(b)
    assert a.n_removed == b.n_removed
    for ta, tb in zip(a.tracks, b.tracks):
        np.testing.assert_allclose(ta.mean, tb.mean, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ta.cov, tb.cov, rtol=1e-9, atol=1e-9)


def test_oru_freezes_at_first_miss_and_replays():
    """use_oru: the state is frozen at the first miss and replayed on re-match."""
    traj = with_gaps(_one_animal(20), {0: range(10, 13)})
    scene = make_scene(traj)
    tracker = _make(_OruSORT)
    for frame_idx, objs in scene.frames[:10]:
        tracker.track(objs, frame_idx)
    track = tracker.tracks[0]
    assert track.frozen is None and track.step == 9
    tracker.track([], 10)
    frozen = track.frozen
    assert frozen is not None and frozen[2] == 10
    np.testing.assert_allclose(frozen[0], track.mean)
    tracker.track([], 11)
    tracker.track([], 12)
    assert track.frozen is frozen
    last_obs = track.last_observation.copy()

    frame_idx, objs = scene.frames[13]
    dets = tracker._detections(list(objs), frame_idx)
    box = dets.boxes[0]
    mean, cov = oru_replay(
        tracker.kf, frozen[0], frozen[1], last_obs, box, n_gap=4, conf=0.9
    )
    mean, cov = tracker.kf.update(mean, cov, box, conf=0.9)
    tracker.track(objs, frame_idx)
    assert track.frozen is None and track.time_since_update == 0
    np.testing.assert_allclose(track.mean, mean)
    np.testing.assert_allclose(track.cov, cov)


def test_tentative_dies_on_miss():
    """tentative_dies_on_miss: a tentative track is removed at its first miss,
    including a miss inside a frame gap."""
    traj = with_gaps(_one_animal(6), {0: [2]})
    scene = make_scene(traj)
    tracker = _make(_DyingSORT, min_hits=3)
    run_tracker(tracker, scene.frames[:4], prepare=False)
    assert tracker.n_removed == 1
    assert [t.tid for t in tracker.tracks] == [1]

    keep = [0, 1, 3, 4]
    scene = make_scene(_one_animal(6)[keep], frame_indices=keep)
    tracker = _make(_DyingSORT, min_hits=3)
    run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.n_removed == 1 and [t.tid for t in tracker.tracks] == [1]
    assert all(obj.track is None for obj in scene.objects())

    # hits_start=1, min_hits=1: a spawn confirms immediately.
    scene = make_scene(_one_animal(2))
    tracker = _make(_DyingSORT, min_hits=1)
    frame_idx, objs = scene.frames[0]
    tracker.track(objs, frame_idx)
    assert tracker.tracks[0].state == TrackState.CONFIRMED
    assert objs[0].track.name == "track_0" and objs[0].tracking_score == 1.0


def test_lost_zero_velocity():
    """lost_zero_velocity: velocities of tracks unmatched at the previous step
    are zeroed before prediction; matched tracks keep theirs."""
    traj = with_gaps(_one_animal(14), {0: [10, 11]})
    scene = make_scene(traj)
    tracker = _make(_ZeroVelocitySORT)
    run_tracker(tracker, scene.frames[:11], prepare=False)
    track = tracker.tracks[0]
    assert track.state == TrackState.LOST
    assert track.mean[4] > 1.0  # first miss: predicted with its velocity
    tracker.track([], 11)
    np.testing.assert_allclose(track.mean[4:6], 0.0)


def test_observation_history_and_velocity():
    """OC-SORT observations are bounded to 2*delta_t+1 and give the direction;
    trackers that do not keep the history still keep the last observation."""
    scene = make_scene(_one_animal(15, velocity=(0.0, 4.0)))
    tracker = _make(_HistorySORT, delta_t=2)
    run_tracker(tracker, scene.frames, prepare=False)
    track = tracker.tracks[0]
    assert sorted(track.observations) == [10, 11, 12, 13, 14]
    np.testing.assert_allclose(track.velocity, [1.0, 0.0])  # (dy, dx)
    np.testing.assert_allclose(track.last_observation, track.observations[14])

    # Without the history (SORT): no observations / velocity, same tracking.
    plain = _make(SORTTracker, delta_t=2)
    run_tracker(plain, make_scene(_one_animal(15, velocity=(0.0, 4.0))).frames)
    other = plain.tracks[0]
    assert other.observations == {} and other.velocity is None
    np.testing.assert_array_equal(other.last_observation, track.last_observation)
    np.testing.assert_array_equal(other.mean, track.mean)
    assert _states(plain) == _states(tracker)

    # The spawn box is not an observation (OC-SORT reference).
    tracker = _make(_HistorySORT)
    frame_idx, objs = scene.frames[0]
    tracker.track(objs, frame_idx)
    assert tracker.tracks[0].observations == {}
    assert tracker.tracks[0].last_observation is None
    dets = tracker._detections(list(objs), 0)
    assert tracker._last_observation_similarity(tracker.tracks, dets).tolist() == [
        [0.0]
    ]


@pytest.mark.parametrize("kind", ["iou", "oks", "centroid"])
def test_similarity_matrices(kind):
    """Prediction and last-observation similarities: shapes, range, invalid=0."""
    scene = make_scene(linear_trajectories([(40, 60), (40, 180)], [(3, 0)] * 2, 5))
    tracker = _make(SORTTracker, similarity=kind)
    run_tracker(tracker, scene.frames, prepare=False)
    objs = [
        make_pose((55.0, 60.0)),
        make_pose((55.0, 180.0)),
        make_pose((0, 0), all_nan=True),
    ]
    dets = tracker._detections(objs, 5)
    for sim in (
        tracker._geometric_similarity(tracker.tracks, dets),
        tracker._last_observation_similarity(tracker.tracks, dets),
    ):
        assert sim.shape == (2, 3)
        assert np.isfinite(sim).all() and (sim >= 0).all() and (sim <= 1).all()
        assert sim[0, 0] > 0.5 and sim[1, 1] > 0.5
        assert sim[0, 1] < 0.05 and sim[1, 0] < 0.05
        assert (sim[:, 2] == 0).all()
    assert tracker._geometric_similarity([], dets).shape == (0, 3)
    empty = tracker._detections([], 6)
    assert tracker._geometric_similarity(tracker.tracks, empty).shape == (2, 0)


@pytest.mark.parametrize("name", NAMES)
def test_only_ocsort_family_keeps_observation_history(name):
    """The OC-SORT history (OCM velocity, `observations`) is kept exactly by
    the trackers that read it; every tracker keeps `last_observation`."""
    cls = MOT_TRACKERS[name]
    assert cls.keeps_observation_history == (name in ("ocsort", "deepocsort"))
    scene = _scene(n_frames=8)
    tracker = build_tracker(name)
    run_tracker(tracker, scene.frames)
    for track in tracker.tracks:
        assert track.last_observation is not None
        assert bool(track.observations) == cls.keeps_observation_history
        assert (track.velocity is not None) == cls.keeps_observation_history


def _churn_scene(n_frames=60, seed=0):
    """Three animals plus one confident one-frame false positive per frame."""
    rng = np.random.default_rng(seed)
    fps = [(t, tuple(rng.uniform(20, 300, size=2)), 0.95) for t in range(n_frames)]
    return _scene(n_frames=n_frames, false_positives=fps, seed=seed)


@pytest.mark.parametrize("name", NAMES)
def test_removed_tracks_are_not_retained(name):
    """Track turnover does not grow memory: removed tracks (with their Kalman
    state, members and appearance memory) are dropped, only counted."""
    scene = _churn_scene()
    tracker = build_tracker(name)
    tracker.prepare(scene.frames)
    refs = []
    for frame_idx, objs in scene.frames:
        tracker.track(objs, frame_idx)
        refs.extend(weakref.ref(t) for t in tracker.tracks)
    tracker.finalize()
    assert not hasattr(tracker, "finished")
    gc.collect()
    alive = {id(r()) for r in refs if r() is not None}
    assert alive == {id(t) for t in tracker.tracks}
    assert tracker.n_removed >= 15  # the false positives' tentative tracks
    assert len(tracker.tracks) < 3 + 35  # bounded by the buffer, not the video


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("link", [False, True])
def test_confirmed_tracks_keep_no_members(name, link):
    """`members` only holds a tentative track's detections (labeled when it
    confirms); processed frames are kept only for identity linking."""
    scene = _scene(n_frames=12)
    tracker = build_tracker(name, min_hits=3, link_tracklets=link)
    tracker.prepare(scene.frames)
    saw_tentative_members = False
    for frame_idx, objs in scene.frames:
        tracker.track(objs, frame_idx)
        for track in tracker.tracks:
            if track.public_id is None:
                saw_tentative_members |= len(track.members) > 0
            else:
                assert track.members == []
    assert saw_tentative_members
    assert len(tracker._seen_frames) == (len(scene.frames) if link else 0)
    tracker.finalize()
    assert ("linking" in tracker.resolved) == link
    # Retroactive labels were still written (frame 0 is tentative everywhere).
    assert all(obj.track is not None for obj in scene.frames[0][1])


_ODD_GAPS = [0, 2, 4, 6, 11, 13, 15, 17, 26, 28, 30, 32, 34, 36, 38]
_IRREGULAR = [0, 1, 5, 6, 14, 15, 33, 34, 35, 61, 62, 90, 91, 92, 140]


@pytest.mark.parametrize("name", ["sort", "bytetrack", "deepsort", "ocsort"])
@pytest.mark.parametrize("time_base", ["auto", "frames", "steps"])
@pytest.mark.parametrize("idx", [_ODD_GAPS, _IRREGULAR], ids=["odd_gaps", "irregular"])
def test_tracker_clock_is_the_calibration_clock(name, time_base, idx):
    """After prepare(), every frame's step equals the shared helper's
    `frame_steps(idx, cal.time_base, cal.frame_step)` for every time base (one
    clock for the tracker, the calibration and the identity linker)."""
    scene = make_scene(
        _trajectories(idx[-1] + 1)[idx], frame_indices=idx, embeddings=True
    )
    tracker = build_tracker(name, time_base=time_base)
    tracker.prepare(scene.frames)
    cal = tracker.calibration
    if time_base != "auto":
        assert cal.time_base == time_base
    steps = []
    for frame_idx, objs in scene.frames:
        tracker.track(objs, frame_idx)
        steps.append(tracker._step)
    expected = frame_steps(idx, cal.time_base, cal.frame_step)
    np.testing.assert_array_equal(steps, expected)
    assert (tracker.resolved["time_base"], tracker.resolved["frame_step"]) == (
        cal.time_base,
        int(cal.frame_step) if cal.time_base == "frames" else 1,
    )


@pytest.mark.parametrize("time_base", ["frames", "steps"])
def test_uncalibrated_clock_matches_frame_steps(time_base):
    """Without prepare(), an explicit time base counts the same steps as
    `frame_steps` with the stride inferred from the first gap."""
    idx = _ODD_GAPS
    scene = make_scene(_trajectories(idx[-1] + 1)[idx], frame_indices=idx)
    tracker = build_tracker("sort", time_base=time_base)
    steps = []
    for frame_idx, objs in scene.frames:
        tracker.track(objs, frame_idx)
        steps.append(tracker._step)
    np.testing.assert_array_equal(steps, frame_steps(idx, time_base, 2))
