"""Tests for ByteTrack and BoT-SORT (`sleap_nn.tracking.mot.bytetrack`)."""

import numpy as np
import pytest
import sleap_io as sio
from conftest import (
    build_tracker,
    count_id_switches,
    distinct_names,
    identity_vectors,
    linear_trajectories,
    make_scene,
    run_tracker,
    static_trajectories,
    track_names,
    with_gaps,
)

from sleap_nn.tracking.mot import BoTSORTTracker, ByteTracker
from sleap_nn.tracking.mot.calibration import VideoCalibration
from sleap_nn.tracking.mot.track import TrackState

# Padded pose boxes of the conftest animal (heading 0) are 48 x 24 px, so a
# horizontal shift of dx px gives IoU (48 - dx) / (48 + dx).


def _one_animal(n_frames=30, velocity=(3.0, 0.0), start=(60.0, 120.0)):
    """One animal moving at constant velocity."""
    return linear_trajectories([start], [velocity], n_frames)


def _bounce(n_frames=24, speed=3.0, lane=4.0, meet=10):
    """Two animals approach along ``y`` in adjacent lanes, touch, then bounce back.

    Moving along the boxes' short side, the Kalman predictions overshoot after the
    bounce onto the OTHER animal: geometry alone swaps the identities.
    """
    t = np.arange(n_frames, dtype=np.float64)
    progress = np.where(t <= meet, t, 2 * meet - t)
    ya = 120.0 - speed * meet + speed * progress - 1.0
    yb = 120.0 + speed * meet - speed * progress + 1.0
    xa = np.full(n_frames, 160.0 - lane / 2)
    xb = np.full(n_frames, 160.0 + lane / 2)
    return np.stack([np.stack([xa, ya], -1), np.stack([xb, yb], -1)], axis=1)


def _attach(scene, vectors):
    """Attach ``vectors[gt_id]`` to every detection of the scene."""
    for (_, objs), ids in zip(scene.frames, scene.gt):
        for obj, gid in zip(objs, ids):
            obj.identity_embedding = sio.Embedding(
                np.asarray(vectors[gid], dtype=np.float32)
            )
    return scene


def _calibration(m_assoc=0.1, gate=0.5):
    """A calibration that enables appearance with the given gate and margin."""
    return VideoCalibration(
        status="ok",
        n_frames=24,
        n_detections=48,
        body_size=float(np.hypot(40.0, 16.0)),
        nn_distance=20.0,
        assoc_rank1=0.99,
        m_assoc=m_assoc,
        ema_cos_gate=gate,
        gallery_cos_gate=gate,
        appearance_for_association=True,
    )


def _scores_of(frames):
    """Tracking score of every object, per frame."""
    return [[o.tracking_score for o in objs] for _, objs in frames]


# ------------------------------------------------------------------ ByteTrack
@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_low_score_occlusion_keeps_identity(name):
    """An occluded animal (score 0.2) stays on its track through stage 2 ..."""
    n = 30
    scores = np.full((n, 1), 0.9)
    scores[10:16] = 0.2
    scene = make_scene(_one_animal(n), scores=scores)
    names = run_tracker(build_tracker(name, track_buffer=2), scene.frames)
    assert len(distinct_names(names)) == 1
    assert all(frame[0] is not None for frame in names)

    # ... and loses it when 0.2 is below the low threshold: the low-score
    # detections are ignored and the track expires during the occlusion.
    scene = make_scene(_one_animal(n), scores=scores)
    tracker = build_tracker(name, track_buffer=2, track_low_threshold=0.3)
    names = run_tracker(tracker, scene.frames)
    assert all(frame[0] is None for frame in names[10:16])
    assert names[9][0] is not None and names[20][0] is not None
    assert names[9][0] != names[20][0]
    assert len(distinct_names(names)) == 2


def test_low_stage_needs_confirmed_track_and_similarity():
    """Stage 2 serves tracks matched at the previous step, with its own cap."""
    # A track that missed the previous frame (LOST) is not offered low-score
    # detections; it is re-found by the next high-score detection.
    traj = with_gaps(_one_animal(16), {0: [10]})
    scores = np.full((16, 1), 0.9)
    scores[11] = 0.2
    scene = make_scene(traj, scores=scores)
    names = run_tracker(build_tracker("bytetrack"), scene.frames)
    assert names[10] == [] and names[11] == [None]
    assert len(distinct_names(names)) == 1
    assert names[9] == names[12]

    # A low-score detection 20 px off the prediction (IoU ~0.41) is rejected at
    # the default 0.5 low-stage similarity and accepted at 0.3.
    traj = static_trajectories([(100.0, 120.0)], 12)
    traj[10, 0, 0] += 20.0
    scores = np.full((12, 1), 0.9)
    scores[10] = 0.2
    for low_sim, tracked in [(0.5, False), (0.3, True)]:
        scene = make_scene(traj, scores=scores)
        tracker = build_tracker("bytetrack", low_stage_min_similarity=low_sim)
        names = run_tracker(tracker, scene.frames)
        assert (names[10][0] is not None) == tracked
        if tracked:
            assert names[10][0] == names[9][0]
            assert scene.frames[10][1][0].tracking_score == pytest.approx(
                28.0 / 68.0, abs=0.02
            )


def test_low_score_detections_never_spawn():
    """D_lo detections never start a track, whatever new_track_threshold says."""
    for new_track_threshold in (0.6, 0.2):
        scene = make_scene(_one_animal(12), scores=0.3)
        tracker = build_tracker("bytetrack", new_track_threshold=new_track_threshold)
        names = run_tracker(tracker, scene.frames)
        assert distinct_names(names) == []
        assert tracker.tracks == [] and tracker.finished == []


def test_new_tracks_need_new_track_threshold():
    """High-score detections start tracks only from new_track_threshold on."""
    scene = make_scene(_one_animal(12), scores=0.55)  # D_hi (> 0.5) but < 0.6
    tracker = build_tracker("bytetrack")
    assert distinct_names(run_tracker(tracker, scene.frames)) == []
    assert tracker.finished == []

    scene = make_scene(_one_animal(12), scores=0.65)
    names = run_tracker(build_tracker("bytetrack"), scene.frames)
    assert len(distinct_names(names)) == 1
    assert all(frame[0] is not None for frame in names)

    # Once tracked, a 0.55 detection still matches (stage 1 takes all of D_hi).
    scores = np.full((12, 1), 0.9)
    scores[6:] = 0.55
    scene = make_scene(_one_animal(12), scores=scores)
    names = run_tracker(build_tracker("bytetrack"), scene.frames)
    assert len(distinct_names(names)) == 1
    assert all(frame[0] is not None for frame in names)


def test_high_threshold_is_strict():
    """A score exactly at track_high_threshold is neither high nor low: ignored."""
    scores = np.full((14, 1), 0.9)
    scores[8] = 0.5
    scene = make_scene(_one_animal(14), scores=scores)
    tracker = build_tracker("bytetrack")
    names = run_tracker(tracker, scene.frames)
    assert names[8] == [None]
    assert len(distinct_names(names)) == 1  # re-found from LOST at frame 9
    assert names[7] == names[9]


def test_tentative_track_dies_on_first_miss():
    """A tentative track is removed at its first miss; its detection stays untracked."""
    traj = with_gaps(_one_animal(8), {0: [1]})
    scene = make_scene(traj)
    tracker = build_tracker("bytetrack", min_hits=2)
    names = run_tracker(tracker, scene.frames, finalize=False)
    assert [t.tid for t in tracker.finished] == [0]
    assert tracker.finished[0].state == TrackState.REMOVED
    assert names[0] == [None]
    assert names[1] == []
    assert all(frame == ["track_0"] for frame in names[2:])

    # A confirmed track survives the same miss as LOST and keeps its id.
    traj = with_gaps(_one_animal(8), {0: [3]})
    scene = make_scene(traj)
    names = run_tracker(build_tracker("bytetrack", min_hits=2), scene.frames)
    assert distinct_names(names) == ["track_0"]
    assert all(frame == ["track_0"] for frame in names if frame)


def test_tracking_score_is_unfused_geometry():
    """Score fusion changes the cost, not the reported (geometric) tracking score."""
    scene = make_scene(static_trajectories([(100.0, 120.0)], 8), scores=0.8)
    run_tracker(build_tracker("bytetrack", fuse_score=True), scene.frames)
    assert all(s == pytest.approx(1.0) for s in sum(_scores_of(scene.frames), []))


@pytest.mark.parametrize("fuse", [True, False])
def test_fuse_score(fuse):
    """IoU 1/3 with a 0.55 detection: fused (0.18 < 0.2) rejects, unfused accepts."""
    traj = static_trajectories([(100.0, 120.0)], 12)
    traj[10, 0, 0] += 24.0  # IoU 24/72 with the static prediction
    scores = np.full((12, 1), 0.9)
    scores[10] = 0.55
    scene = make_scene(traj, scores=scores)
    tracker = build_tracker("bytetrack", fuse_score=fuse)
    names = run_tracker(tracker, scene.frames)
    if fuse:
        assert names[10] == [None]
    else:
        assert names[10] == names[9]
        assert scene.frames[10][1][0].tracking_score == pytest.approx(1 / 3, abs=0.02)


def _two_static_tracks(**overrides):
    """A ByteTracker with two confirmed static tracks after frames 0-9."""
    scene = make_scene(static_trajectories([(100.0, 60.0), (100.0, 180.0)], 10))
    tracker = build_tracker("bytetrack", **overrides)
    run_tracker(tracker, scene.frames, prepare=False, finalize=False)
    t0, t1 = tracker.tracks
    assert t0.state == t1.state == TrackState.CONFIRMED
    return tracker, t0, t1


def _make_lost_duplicate(tracker, kept, dup, dx=2.0):
    """Turn ``dup`` into a LOST, motionless track predicted ``dx`` px off ``kept``."""
    dup.mean = np.array(kept.mean, dtype=np.float64).copy()
    dup.mean[0] += dx
    dup.mean[4:] = 0.0
    dup.cov = np.array(kept.cov, dtype=np.float64).copy()
    dup.state = TrackState.LOST
    dup.time_since_update = 1


def test_duplicate_removal_drops_younger_lost_track():
    """A LOST track predicted on a matched track's detection (younger) is removed."""
    tracker, t0, t1 = _two_static_tracks()
    _make_lost_duplicate(tracker, t0, t1)
    t1.start_frame = 5  # history 4 < t0's 10
    objs = make_scene(static_trajectories([(100.0, 60.0)], 1)).frames[0][1]
    tracker.track(objs, 10)
    assert t1.state == TrackState.REMOVED and t1 in tracker.finished
    assert tracker.tracks == [t0] and t0.state == TrackState.CONFIRMED
    assert objs[0].track is tracker._track_objects[t0.public_id]


def test_duplicate_removal_drops_younger_matched_track():
    """When the LOST duplicate is older, the matched track is removed instead and
    its detection stays untracked (and starts no track)."""
    tracker, t0, t1 = _two_static_tracks()
    _make_lost_duplicate(tracker, t0, t1)
    t1.start_frame = -50  # history 59 > t0's 10
    objs = make_scene(static_trajectories([(100.0, 60.0)], 1)).frames[0][1]
    tracker.track(objs, 10)
    assert t0.state == TrackState.REMOVED and t0 in tracker.finished
    assert tracker.tracks == [t1] and t1.state == TrackState.LOST
    assert objs[0].track is None and objs[0].tracking_score == 0.0

    # Without duplicate removal (threshold 1.0) both tracks live on.
    tracker, t0, t1 = _two_static_tracks(duplicate_similarity=1.0)
    _make_lost_duplicate(tracker, t0, t1)
    t1.start_frame = -50
    objs = make_scene(static_trajectories([(100.0, 60.0)], 1)).frames[0][1]
    tracker.track(objs, 10)
    assert tracker.tracks == [t0, t1]
    assert objs[0].track is tracker._track_objects[t0.public_id]


@pytest.mark.parametrize("dx,spawns", [(3.0, False), (6.0, True)])
def test_new_track_duplicating_a_lost_track_is_not_started(dx, spawns):
    """An unmatched detection on a LOST track's prediction (IoU > 0.85) does not
    start a track (it would be removed as the younger duplicate)."""
    traj = with_gaps(static_trajectories([(100.0, 120.0)], 12), {0: [10]})
    traj[11, 0, 0] += dx  # IoU 45/51 = 0.88 (dx 3) or 42/54 = 0.78 (dx 6)
    scene = make_scene(traj, scores=0.95)
    # Stage 1 demands IoU >= 0.95, so the shifted detection is never matched.
    tracker = build_tracker("bytetrack", min_match_similarity=0.95, fuse_score=False)
    names = run_tracker(tracker, scene.frames, finalize=False)
    assert len(distinct_names(names)) == 1
    assert names[11] == [None]
    assert tracker.tracks[0].state == TrackState.LOST
    assert len(tracker.tracks) == (2 if spawns else 1)


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("kalman", ["xyah", "xywh", "xysr"])
def test_lost_velocity_zeroing_follows_the_filter(name, kalman):
    """Size velocities of unmatched tracks are zeroed for every filter layout."""
    cls = ByteTracker if name == "bytetrack" else BoTSORTTracker
    tracker = build_tracker(name, kalman=kalman)
    assert tracker.lost_zero_velocity == cls.size_velocity[kalman]
    traj = with_gaps(
        linear_trajectories([(60.0, 120.0)], [(3.0, 0.5)], 20), {0: [10, 11]}
    )
    scene = make_scene(traj)
    run_tracker(tracker, scene.frames[:11], prepare=False, finalize=False)
    tracker.track([], 11)
    track = tracker.tracks[0]
    assert track.state == TrackState.LOST
    np.testing.assert_allclose(track.mean[list(tracker.lost_zero_velocity)], 0.0)
    assert track.mean[4] > 1.0  # the center velocity is kept
    run_tracker(tracker, scene.frames[12:], prepare=False)
    assert distinct_names(track_names(scene.frames)) == ["track_0"]


def test_bytetrack_defaults():
    """Paper/animal constants and class attributes."""
    tracker = build_tracker("bytetrack")
    p = tracker.params
    assert (p.track_high_threshold, p.track_low_threshold) == (0.5, 0.1)
    assert (p.new_track_threshold, p.min_match_similarity) == (0.6, 0.2)
    assert p.fuse_score and p.duplicate_similarity == 0.85
    assert tracker.kalman_kind == "xyah" and tracker.lost_zero_velocity == (7,)
    assert ByteTracker.tentative_dies_on_miss and not ByteTracker.uses_appearance
    botsort = build_tracker("botsort")
    assert botsort.kalman_kind == "xywh" and botsort.lost_zero_velocity == (6, 7)
    assert botsort.uses_appearance and not botsort.requires_vectors
    assert botsort.params.appearance_threshold == "auto"
    assert build_tracker("botsort", preset="paper").params.appearance_threshold == 0.5


# ------------------------------------------------------------------ BoT-SORT
def test_botsort_appearance_fixes_ambiguous_bounce():
    """Discriminative vectors keep identities where geometry alone swaps them."""
    geometry = make_scene(_bounce())
    names = run_tracker(
        build_tracker("botsort", appearance_threshold=0.5), geometry.frames
    )
    assert count_id_switches(names, geometry.gt) > 0
    names = run_tracker(build_tracker("bytetrack"), make_scene(_bounce()).frames)
    assert count_id_switches(names, geometry.gt) > 0

    scene = make_scene(_bounce(), embeddings=True)
    tracker = build_tracker("botsort", appearance_threshold=0.5)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
    assert all(n is not None for frame in names for n in frame)
    # The tracking score stays the geometric similarity.
    assert all(0.0 < s <= 1.0 for s in sum(_scores_of(scene.frames)[2:], []))

    # Same with a calibrated gate and margin.
    scene = make_scene(_bounce(), embeddings=True)
    tracker = build_tracker("botsort")
    tracker.calibration = _calibration(m_assoc=0.1)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert count_id_switches(names, scene.gt) == 0


def test_botsort_appearance_never_matches_outside_proximity_gate():
    """Identical vectors do not match a detection outside the geometric gate."""

    def scene_with_jump():
        traj = static_trajectories([(100.0, 120.0)], 12)
        traj[10:, 0, 0] += 70.0  # no overlap with the static prediction
        return _attach(make_scene(traj), identity_vectors(1))

    scene = scene_with_jump()
    tracker = build_tracker("botsort", appearance_threshold=0.5)
    names = run_tracker(tracker, scene.frames, finalize=False)
    assert tracker._appearance_gate("ema") == 0.5
    first = names[9][0]
    assert first is not None
    assert names[10][0] != first and names[11][0] != first
    assert tracker.tracks[0].state == TrackState.LOST

    # Control: without the proximity gate the same vectors force the match.
    scene = scene_with_jump()
    tracker = build_tracker(
        "botsort", appearance_threshold=0.5, proximity_threshold=0.0
    )
    names = run_tracker(tracker, scene.frames)
    assert names[10][0] == names[9][0]
    assert scene.frames[10][1][0].tracking_score == 0.0  # geometric similarity


def _run_bounce(tracker, vectors=None, prepare=False):
    scene = make_scene(_bounce())
    if vectors is not None:
        _attach(scene, vectors)
    names = run_tracker(tracker, scene.frames, prepare=prepare)
    return names, _scores_of(scene.frames)


def test_botsort_without_usable_appearance_is_geometric():
    """Uncalibrated 'auto', undecisive margins or no vectors: identical results."""
    reference = _run_bounce(build_tracker("botsort"))
    assert count_id_switches(reference[0], make_scene(_bounce()).gt) > 0
    discriminative = identity_vectors(2)

    # 'auto' without calibration disables appearance.
    tracker = build_tracker("botsort")
    assert _run_bounce(tracker, discriminative) == reference
    assert tracker.resolved["appearance"]["enabled"] is False

    # Calibration that disables appearance for association.
    tracker = build_tracker("botsort")
    tracker.calibration = VideoCalibration(status="insufficient")
    assert _run_bounce(tracker, discriminative) == reference

    # Vectors that barely differ (cosine margin ~0.014 < m_assoc 0.1).
    base = np.zeros(16)
    base[0] = 1.0
    similar = [base + 0.12 * v for v in identity_vectors(2, seed=1)]
    similar = [v / np.linalg.norm(v) for v in similar]
    tracker = build_tracker("botsort")
    tracker.calibration = _calibration(m_assoc=0.1)
    assert _run_bounce(tracker, similar) == reference

    # The paper preset has no margin: the same small differences decide.
    tracker = build_tracker("botsort", preset="paper", appearance_threshold=0.5)
    tracker.calibration = _calibration(m_assoc=0.1)
    names, _ = _run_bounce(tracker, similar)
    assert count_id_switches(names, make_scene(_bounce()).gt) == 0

    # BoT-SORT with appearance off is ByteTrack on BoT-SORT's filter and
    # thresholds (no track is lost here, so velocity zeroing does not differ).
    bytetrack = build_tracker(
        "bytetrack", kalman="xywh", track_high_threshold=0.6, new_track_threshold=0.7
    )
    assert _run_bounce(bytetrack) == reference


def test_botsort_no_vectors():
    """No vectors: strict mode raises up front, library mode tracks on geometry."""
    scene = make_scene(_one_animal(12))
    tracker = build_tracker("botsort")
    tracker.strict_vectors = True
    with pytest.raises(ValueError, match="bytetrack"):
        tracker.prepare(scene.frames)

    # Empty videos and videos with vectors pass the strict check.
    tracker = build_tracker("botsort")
    tracker.strict_vectors = True
    tracker.prepare([(i, []) for i in range(3)])
    tracker = build_tracker("botsort")
    tracker.strict_vectors = True
    names = run_tracker(tracker, make_scene(_one_animal(12), embeddings=True).frames)
    assert len(distinct_names(names)) == 1

    for prepare in (True, False):
        tracker = build_tracker("botsort")
        names = run_tracker(
            tracker, make_scene(_one_animal(12)).frames, prepare=prepare
        )
        assert len(distinct_names(names)) == 1
        assert tracker.resolved["appearance"]["enabled"] is False
