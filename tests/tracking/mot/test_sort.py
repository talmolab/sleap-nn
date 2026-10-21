"""Tests for the SORT tracker (`sleap_nn.tracking.mot.sort`) and its base lifecycle."""

import numpy as np
import pytest
from conftest import (
    build_tracker,
    count_id_switches,
    crossing_trajectories,
    distinct_names,
    linear_trajectories,
    make_pose,
    make_scene,
    names_by_identity,
    run_tracker,
    static_trajectories,
    with_gaps,
)

from sleap_nn.tracking.mot import MOTParams, SORTTracker
from sleap_nn.tracking.mot.kalman import KalmanFilterXYSR, KalmanFilterXYWH
from sleap_nn.tracking.mot.track import TrackState


def _one_animal(n_frames=30, velocity=(4.0, 0.0)):
    """One animal moving at constant velocity from (40, 120)."""
    return linear_trajectories([(40.0, 120.0)], [velocity], n_frames)


@pytest.mark.parametrize("prepare", [True, False])
@pytest.mark.parametrize("carrier", ["pose", "mask"])
def test_crossing_animals_keep_identities(prepare, carrier):
    """Two animals crossing head-on (overlapping boxes) keep their ids."""
    scene = make_scene(crossing_trajectories(n_frames=30, offset=12.0), carrier=carrier)
    n_nodes = 5 if carrier == "pose" else None
    tracker = build_tracker("sort", carrier=carrier, n_nodes=n_nodes)
    names = run_tracker(tracker, scene.frames, prepare=prepare)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
    assert all(n is not None for frame in names for n in frame)


@pytest.mark.parametrize("kind", ["iou", "oks", "centroid"])
def test_similarity_kinds(kind):
    """Every geometric similarity kind tracks a crossing without a switch."""
    scene = make_scene(crossing_trajectories(n_frames=30, offset=16.0))
    tracker = build_tracker("sort", similarity=kind)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["similarity"] == kind
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2


def test_auto_similarity_resolution():
    """auto: iou for multi-node poses and masks, centroid for single-node poses."""
    assert build_tracker("sort").resolved["similarity"] == "iou"
    assert build_tracker("sort", n_nodes=1).resolved["similarity"] == "centroid"
    tracker = build_tracker("sort", carrier="mask", n_nodes=None)
    assert tracker.resolved["similarity"] == "iou"
    # oks needs keypoints: masks fall back to iou, single nodes to centroid.
    tracker = build_tracker("sort", carrier="mask", n_nodes=None, similarity="oks")
    assert tracker.resolved["similarity"] == "iou"
    tracker = build_tracker("sort", n_nodes=1, similarity="oks")
    assert tracker.resolved["similarity"] == "centroid"


def test_uses_xysr_filter_by_default():
    """SORT's own filter is XYSR; the `kalman` override swaps it."""
    tracker = build_tracker("sort")
    assert isinstance(tracker.kf, KalmanFilterXYSR)
    assert tracker.resolved["kalman"] == "xysr"
    tracker = build_tracker("sort", kalman="xywh")
    assert isinstance(tracker.kf, KalmanFilterXYWH)
    scene = make_scene(crossing_trajectories(n_frames=30))
    names = run_tracker(tracker, scene.frames)
    assert count_id_switches(names, scene.gt) == 0


def test_short_occlusion_keeps_identity():
    """An animal missing for fewer than track_buffer frames keeps its id."""
    traj = with_gaps(_one_animal(40), {0: range(15, 22)})
    scene = make_scene(traj)
    names = run_tracker(build_tracker("sort", track_buffer=30), scene.frames)
    seq = names_by_identity(names, scene.gt)[0]
    assert set(seq) == {"track_0"}


def test_long_occlusion_gets_new_identity():
    """An animal missing for more than track_buffer frames gets a new id."""
    traj = with_gaps(_one_animal(40), {0: range(15, 26)})  # 11 missing frames
    scene = make_scene(traj)
    tracker = build_tracker("sort", track_buffer=5)
    names = run_tracker(tracker, scene.frames)
    seq = names_by_identity(names, scene.gt)[0]
    assert seq[:15] == ["track_0"] * 15
    assert seq[15:] == ["track_1"] * len(seq[15:])
    assert [t.public_id for t in tracker.finished] == [0]


def test_gap_longer_than_buffer_expires_before_association():
    """frames time base: a static animal seen again after a long frame gap
    starts a new track (the gap's empty steps already removed the old one)."""
    traj = static_trajectories([(100.0, 100.0)], 20)
    frame_indices = list(range(10)) + list(range(30, 40))
    scene = make_scene(traj, frame_indices=frame_indices)
    tracker = build_tracker("sort", track_buffer=5, time_base="frames")
    names = run_tracker(tracker, scene.frames, prepare=False)
    seq = names_by_identity(names, scene.gt)[0]
    assert seq[:10] == ["track_0"] * 10
    assert set(seq[10:]) == {"track_1"}


def test_steps_time_base_ignores_frame_gaps():
    """time_base='steps': a frame gap counts one step whatever its length."""
    traj = _one_animal(20)
    frame_indices = list(range(10)) + list(range(110, 120))
    scene = make_scene(traj, frame_indices=frame_indices)
    tracker = build_tracker("sort", track_buffer=5, time_base="steps")
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert tracker.resolved["time_base"] == "steps"
    assert distinct_names(names) == ["track_0"]
    assert tracker._step == 19


def test_frame_gap_predicts_through_the_gap():
    """frames time base: the Kalman filter coasts through a frame gap, so a fast
    animal reappearing where it should be keeps its id."""
    n_frames = 40
    traj = _one_animal(n_frames, velocity=(12.0, 0.0))
    keep = [t for t in range(n_frames) if not 20 <= t < 24]
    scene = make_scene(traj[keep], frame_indices=keep)
    tracker = build_tracker("sort")
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert distinct_names(names) == ["track_0"]
    # 36 processed frames spanning 39 steps.
    assert tracker._step == 39


def test_tentative_track_survives_a_miss():
    """SORT family: a tentative track coasts over a miss instead of dying, and
    its streak restarts (min_hits=3 then needs 3 more matches)."""
    traj = with_gaps(_one_animal(8), {0: [2]})
    scene = make_scene(traj)
    tracker = build_tracker("sort", min_hits=3)
    frames = scene.frames
    for frame_idx, objs in frames[:2]:
        tracker.track(objs, frame_idx)
    frame_idx, objs = frames[2]
    assert objs == []  # frame 2: the animal is missed
    tracker.track(objs, frame_idx)
    assert [t.tid for t in tracker.tracks] == [0]
    assert tracker.tracks[0].state == TrackState.TENTATIVE
    assert tracker.tracks[0].time_since_update == 1
    for frame_idx, objs in frames[3:6]:  # streak 1, 2
        tracker.track(objs, frame_idx)
        if frame_idx < 5:
            assert tracker.tracks[0].state == TrackState.TENTATIVE
    assert [t.tid for t in tracker.tracks] == [0]
    assert tracker.tracks[0].state == TrackState.CONFIRMED  # streak 3 at frame 5
    tracker.finalize()
    # Retroactive labeling covers the frames before the miss too.
    assert {obj.track.name for _, objs in frames[:6] for obj in objs} == {"track_0"}


def test_confirms_on_four_consecutive_detections():
    """hits_start=0, min_hits=3: confirmation at the 4th consecutive detection."""
    scene = make_scene(_one_animal(6))
    tracker = build_tracker("sort", min_hits=3)
    for i, (frame_idx, objs) in enumerate(scene.frames[:4]):
        tracker.track(objs, frame_idx)
        state = tracker.tracks[0].state
        if i < 3:
            assert state == TrackState.TENTATIVE
            assert all(o.track is None for _, os in scene.frames[: i + 1] for o in os)
        else:
            assert state == TrackState.CONFIRMED
    labeled = [o.track for _, objs in scene.frames[:4] for o in objs]
    assert all(t is tracker._track_objects[0] for t in labeled)
    assert tracker.tracks[0].hits == 3 and tracker.tracks[0].hit_streak == 3


def test_min_hits_one_needs_one_match_after_spawn():
    """The spawn detection does not count: min_hits=1 confirms at the 2nd frame."""
    scene = make_scene(_one_animal(3))
    tracker = build_tracker("sort", min_hits=1)
    for i, state in enumerate([TrackState.TENTATIVE, TrackState.CONFIRMED]):
        frame_idx, objs = scene.frames[i]
        tracker.track(objs, frame_idx)
        assert tracker.tracks[0].state == state


def test_never_confirmed_track_stays_untracked():
    """A flickering detection never confirms and is never labeled."""
    traj = _one_animal(20)
    scene = make_scene(traj, false_positives=[(3, (300.0, 20.0), 0.9)])
    names = run_tracker(build_tracker("sort"), scene.frames)
    fp_names = [n for ns, g in zip(names, scene.gt) for n, gi in zip(ns, g) if gi < 0]
    assert fp_names == [None]
    assert distinct_names(names) == ["track_0"]


def test_low_score_detections_are_ignored():
    """score <= track_low_threshold: never matched, never spawns; the track
    coasts over it and resumes with the same id."""
    scores = np.full((20, 1), 0.9)
    scores[8] = 0.05
    scene = make_scene(_one_animal(20), scores=scores)
    tracker = build_tracker("sort", track_low_threshold=0.1)
    names = run_tracker(tracker, scene.frames)
    seq = names_by_identity(names, scene.gt)[0]
    assert seq[8] is None
    assert set(seq[:8] + seq[9:]) == {"track_0"}
    assert len(tracker._track_objects) == 1


def test_scores_between_low_and_new_match_but_do_not_spawn():
    """low < score < new_track_threshold: matches a track, never starts one."""
    scores = np.full((20, 2), 0.9)
    scores[:, 1] = 0.3  # animal 1 is always weak
    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 20)
    scene = make_scene(traj, scores=scores)
    tracker = build_tracker(
        "sort",
        track_low_threshold=0.1,
        track_high_threshold=0.5,
        new_track_threshold=0.5,
    )
    names = run_tracker(tracker, scene.frames)
    by_id = names_by_identity(names, scene.gt)
    assert set(by_id[0]) == {"track_0"}
    assert set(by_id[1]) == {None}


def test_ambiguous_assignment_uses_hungarian():
    """Side-by-side animals with overlapping boxes (no 1:1 valid pattern) are
    resolved by the global assignment."""
    traj = linear_trajectories([(40, 100), (40, 118)], [(3, 0), (3, 0)], 25)
    scene = make_scene(traj)
    tracker = build_tracker("sort")
    names = run_tracker(tracker, scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2


def test_tracking_score_is_matched_similarity():
    """Spawn detections score 1.0; matched ones the IoU of prediction vs box."""
    scene = make_scene(_one_animal(10))
    run_tracker(build_tracker("sort"), scene.frames)
    scores = [objs[0].tracking_score for _, objs in scene.frames]
    assert scores[0] == 1.0
    assert all(0.3 <= s <= 1.0 for s in scores[1:])
    # Constant velocity is learned: late predictions are near-perfect.
    assert scores[-1] > 0.9


def test_duplicates_do_not_steal_identity():
    """A one-frame duplicate detection does not take the animal's track."""
    scene = make_scene(_one_animal(20), duplicates=[(10, 0)])
    names = run_tracker(build_tracker("sort"), scene.frames)
    seq = names_by_identity(names, scene.gt)[0]
    assert set(seq) == {"track_0"}


def test_running_body_size_without_prepare():
    """Uncalibrated, degenerate (single-node) boxes use the running body size."""
    tracker = build_tracker("sort")
    assert tracker._body_size() == 50.0
    for t in range(3):
        tracker.track([make_pose((50.0 + 3 * t, 50.0))], t)
    # Five-node animal: 40 x 16 px raw extent.
    assert tracker._body_size() == pytest.approx(np.hypot(40.0, 16.0))
    single = make_pose((100.0, 100.0), nan_nodes=[0, 2, 3, 4])
    dets = tracker._detections([single], 3)
    side = dets.raw_boxes[0, 2] - dets.raw_boxes[0, 0]
    assert side == pytest.approx(np.hypot(40.0, 16.0))


def test_prepare_caches_detections_by_identity():
    """prepare() extracts once; track() reuses the cache only for the same objects."""
    scene = make_scene(_one_animal(5))
    tracker = build_tracker("sort")
    tracker.prepare(scene.frames)
    assert set(tracker._cache) == {0, 1, 2, 3, 4}
    cached = tracker._cache[0]
    assert tracker._detections(list(scene.frames[0][1]), 0) is cached
    other = [make_pose((40.0, 120.0))]
    assert tracker._detections(other, 1).sources == other
    assert 1 not in tracker._cache


def test_rejects_parameters_of_another_tracker():
    """A tracker refuses parameters resolved for another algorithm."""
    with pytest.raises(ValueError, match="sort"):
        SORTTracker(MOTParams.for_tracker("bytetrack"))
    with pytest.raises(ValueError, match="carrier"):
        SORTTracker(MOTParams.for_tracker("sort"), carrier="bbox")
