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
    make_pose,
    make_scene,
    pose_points,
    run_tracker,
    static_trajectories,
    track_names,
    with_gaps,
)
from loguru import logger

from sleap_nn.tracking.mot import BoTSORTTracker, ByteTracker
from sleap_nn.tracking.mot.association import iou_matrix
from sleap_nn.tracking.mot.bytetrack import decisive_cells, shared_node_iou
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


@pytest.fixture
def log_messages():
    """Capture loguru messages (WARNING and above) emitted during a test."""
    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(handler_id)


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
        tracker = build_tracker(
            "bytetrack", new_track_threshold=new_track_threshold, normalize_scores=False
        )
        names = run_tracker(tracker, scene.frames)
        assert distinct_names(names) == []
        assert tracker.tracks == [] and tracker.n_removed == 0


def test_new_tracks_need_new_track_threshold():
    """High-score detections start tracks only from new_track_threshold on."""
    scene = make_scene(_one_animal(12), scores=0.55)  # D_hi (> 0.5) but < 0.6
    tracker = build_tracker("bytetrack", normalize_scores=False)
    assert distinct_names(run_tracker(tracker, scene.frames)) == []
    assert tracker.n_removed == 0

    scene = make_scene(_one_animal(12), scores=0.65)
    names = run_tracker(
        build_tracker("bytetrack", normalize_scores=False), scene.frames
    )
    assert len(distinct_names(names)) == 1
    assert all(frame[0] is not None for frame in names)

    # Once tracked, a 0.55 detection still matches (stage 1 takes all of D_hi).
    scores = np.full((12, 1), 0.9)
    scores[6:] = 0.55
    scene = make_scene(_one_animal(12), scores=scores)
    names = run_tracker(
        build_tracker("bytetrack", normalize_scores=False), scene.frames
    )
    assert len(distinct_names(names)) == 1
    assert all(frame[0] is not None for frame in names)


def test_high_threshold_is_strict():
    """A score exactly at track_high_threshold is neither high nor low: ignored."""
    scores = np.full((14, 1), 0.9)
    scores[8] = 0.5
    scene = make_scene(_one_animal(14), scores=scores)
    tracker = build_tracker("bytetrack", normalize_scores=False)
    names = run_tracker(tracker, scene.frames)
    assert names[8] == [None]
    assert len(distinct_names(names)) == 1  # re-found from LOST at frame 9
    assert names[7] == names[9]


def test_tentative_track_dies_on_first_miss():
    """A tentative track is removed at its first miss; its detection stays untracked."""
    traj = with_gaps(_one_animal(8), {0: [1]})
    scene = make_scene(traj)
    tracker = build_tracker("bytetrack", min_hits=2)
    tracker.prepare(scene.frames)
    tracker.track(*scene.frames[0][::-1])
    (first,) = tracker.tracks
    for frame_idx, objs in scene.frames[1:]:
        tracker.track(objs, frame_idx)
    names = track_names(scene.frames)
    assert first.tid == 0 and first.state == TrackState.REMOVED
    assert first not in tracker.tracks and tracker.n_removed == 1
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
    tracker = build_tracker("bytetrack", fuse_score=fuse, normalize_scores=False)
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
    assert t1.state == TrackState.REMOVED and t1 not in tracker.tracks
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
    assert t0.state == TrackState.REMOVED and t0 not in tracker.tracks
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
    # Animal preset: per-video score normalization and an `xywh` filter (a
    # turning animal breaks `xyah`'s slowly varying aspect); paper: as published.
    assert p.normalize_scores and tracker.kalman_kind == "xywh"
    assert tracker.lost_zero_velocity == (7,)
    paper = build_tracker("bytetrack", preset="paper")
    assert not paper.params.normalize_scores and paper.kalman_kind == "xyah"
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


@pytest.mark.parametrize("bad", [None, "nan", "zero"])
def test_botsort_no_usable_vectors(bad):
    """No usable vector at all: BoT-SORT tracks on geometry alone.

    None, all NaN or all zero (the ``FrameDetections.has_embedding`` rule) give
    the same result whether or not the detections hold a (broken) vector.
    """
    reference = run_tracker(
        build_tracker("botsort"), make_scene(_one_animal(12)).frames
    )
    assert len(distinct_names(reference)) == 1
    for prepare in (True, False):
        scene = make_scene(_one_animal(12))
        if bad is not None:
            vec = np.full(16, np.nan if bad == "nan" else 0.0)
            _attach(scene, [vec])
        tracker = build_tracker("botsort")
        names = run_tracker(tracker, scene.frames, prepare=prepare)
        assert names == reference
        assert tracker._vectors_seen is False
        appearance = tracker.resolved["appearance"]
        assert appearance["enabled"] is False
        assert "no appearance vectors" in appearance["reason"]
    assert not hasattr(
        BoTSORTTracker(build_tracker("botsort").params), "strict_vectors"
    )


# ------------------------------------------------- duplicate removal (kinds)
def _huddle(n_frames=90, dropout=75):
    """Two single-node animals 150 px apart; B then rests 30 px from A.

    B misses one detection (frame ``dropout``) while resting there.
    """
    traj = np.zeros((n_frames, 2, 2))
    traj[:, 0] = (100.0, 120.0)
    traj[:, 1, 0] = np.concatenate(
        [
            np.full(50, 250.0),
            np.linspace(250.0, 130.0, 11),
            np.full(n_frames - 61, 130.0),
        ]
    )
    traj[:, 1, 1] = 120.0
    traj[dropout, 1] = np.nan
    return traj


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_single_node_huddle_is_not_a_duplicate(name):
    """Centroid similarity: a one-frame dropout in a huddle removes no track.

    Body 150 px, the animals rest 30 px apart: no identity switches either. The
    centroid kernel at 0.85 admits 0.285 body (43 px): it used to call the
    pair a duplicate and remove a real track. Duplicates are decided on the box
    IoU (body-sized pseudo boxes: 120/180 = 0.67 here).
    """
    scene = make_scene(_huddle(), n_nodes=1)
    tracker = build_tracker(name, n_nodes=1, body_size=150.0)
    names = run_tracker(tracker, scene.frames)
    assert tracker.resolved["similarity"] == "centroid"
    assert count_id_switches(names, scene.gt) == 0
    assert distinct_names(names) == ["track_0", "track_1"]
    assert all(n is not None for frame in names[5:] for n in frame)
    assert tracker.n_removed == 0


def _two_static_single_node_tracks():
    """A ByteTracker with two confirmed static single-node tracks (body 150 px)."""
    scene = make_scene(
        static_trajectories([(100.0, 60.0), (100.0, 400.0)], 10), n_nodes=1
    )
    tracker = build_tracker("bytetrack", n_nodes=1, body_size=150.0)
    run_tracker(tracker, scene.frames, prepare=False, finalize=False)
    t0, t1 = tracker.tracks
    assert t0.state == t1.state == TrackState.CONFIRMED
    return tracker, t0, t1


@pytest.mark.parametrize("dx,removed", [(3.0, True), (30.0, False)])
def test_single_node_duplicate_threshold_is_box_overlap(dx, removed):
    """A LOST single-node track ``dx`` px off a matched detection.

    It is a duplicate only when their (pseudo) boxes overlap above IoU 0.85.
    """
    tracker, t0, t1 = _two_static_single_node_tracks()
    _make_lost_duplicate(tracker, t0, t1, dx=dx)
    t1.start_frame = 5  # the younger one
    objs = make_scene(static_trajectories([(100.0, 60.0)], 1), n_nodes=1).frames[0][1]
    tracker.track(objs, 10)
    assert (t1.state == TrackState.REMOVED) == removed
    assert objs[0].track is tracker._track_objects[t0.public_id]
    # The centroid kernel (scale body / 2 = 75 px) would call both a duplicate.
    assert np.exp(-(dx**2) / (2 * 75.0**2)) > tracker.params.duplicate_similarity


# --------------------------------------------- partial (occluded) poses, IoU
def _partial_pose_frames(n_frames, occluded, hidden, low=0.3, lane=(60.0, 120.0)):
    """One animal moving 2 px/frame, occluded on the ``occluded`` frames.

    There it scores ``low`` and the ``hidden`` nodes are invisible.
    """
    frames = []
    for t in range(n_frames):
        occ = t in occluded
        center = (lane[0] + 2.0 * t, lane[1])
        pose = make_pose(
            center, score=low if occ else 0.9, nan_nodes=hidden if occ else ()
        )
        frames.append((t, [pose]))
    return frames


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("hidden", [(2,), (2, 3, 4)], ids=["no_tail", "head_thorax"])
@pytest.mark.parametrize("n_occluded", [10, 40])
def test_partial_low_score_pose_stays_on_track(name, hidden, n_occluded):
    """A low-score (D_lo) pose with its tail hidden matches its track in stage 2.

    Its visible-node box (IoU 0.42 with the full predicted box for "no tail") was
    rejected at the 0.5 low-stage cap: the track went LOST and, past
    ``track_buffer`` (40 frames), came back as a new identity. Pose IoU now also
    compares the nodes visible on both sides.
    """
    occluded = range(10, 10 + n_occluded)
    frames = _partial_pose_frames(20 + n_occluded, occluded, hidden)
    tracker = build_tracker(name)
    names = run_tracker(tracker, frames)
    assert tracker.resolved["similarity"] == "iou"
    assert distinct_names(names) == ["track_0"]
    assert all(frame == ["track_0"] for frame in names)
    assert all(frames[t][1][0].tracking_score > 0.5 for t in occluded)


def test_partial_poses_in_contact_keep_identities():
    """Two animals side by side (overlapping boxes); one turns partial, low-score.

    The shared-node IoU does not pull it onto its neighbor's track.
    """
    n = 30
    occluded = range(10, 20)
    frames = []
    for t in range(n):
        objs = []
        for k, y in enumerate((120.0, 138.0)):
            occ = k == 0 and t in occluded
            objs.append(
                make_pose(
                    (60.0 + 2.0 * t, y),
                    score=0.3 if occ else 0.9,
                    nan_nodes=(2,) if occ else (),
                )
            )
        frames.append((t, objs))
    gt = [[0, 1]] * n
    for name in ("bytetrack", "botsort"):
        names = run_tracker(build_tracker(name), frames)
        assert count_id_switches(names, gt) == 0
        assert len(distinct_names(names)) == 2
        assert all(n is not None for frame in names for n in frame)


def test_shared_node_iou():
    """Boxes from the nodes visible on both sides, padded and floored.

    Same rules as detection boxes; NaN where the shared nodes carry no extent.
    """
    full = pose_points((100.0, 120.0))
    no_tail = full.copy()
    no_tail[2] = np.nan
    head_thorax = full.copy()
    head_thorax[2:] = np.nan
    single = full.copy()
    single[1:] = np.nan
    trk = np.stack([full])
    det = np.stack([no_tail, head_thorax, single, full])
    iou = shared_node_iou(trk, det, box_padding=0.1, min_box_size=10.0)
    np.testing.assert_allclose(iou[0, :2], 1.0)  # the same part of the body
    assert np.isnan(iou[0, 2])  # one shared node: no extent
    assert iou[0, 3] == pytest.approx(1.0)
    # Collinear (axis-aligned) shared nodes need the floor to have a size.
    no_floor = shared_node_iou(trk, det, box_padding=0.1, min_box_size=None)
    assert np.isnan(no_floor[0, 1]) and no_floor[0, 0] == pytest.approx(1.0)

    # The full predicted box vs the no-tail detection box: IoU 0.42 < 0.5.
    def padded(kp):
        lo, hi = np.nanmin(kp, axis=0), np.nanmax(kp, axis=0)
        pad = 0.1 * (hi - lo).max()
        return np.concatenate([lo - pad, hi + pad])

    box_iou = iou_matrix(padded(full)[None], padded(no_tail)[None])[0, 0]
    assert box_iou == pytest.approx(480.0 / 1152.0)

    # Shifted by 6 px: (24 - 6) * 20 / (2 * 24 * 20 - 18 * 20) for the no-tail box.
    shifted = no_tail + np.array([6.0, 0.0])
    iou = shared_node_iou(trk, shifted[None], box_padding=0.1, min_box_size=10.0)
    assert iou[0, 0] == pytest.approx(18.0 / 30.0)
    assert shared_node_iou(trk[:0], det).shape == (0, 4)


# ---------------------------------------------------- BoT-SORT decisive cells
def test_decisive_cells_are_per_cell():
    """Only a row's AND column's top in-gate cosine, by the margin, decides."""
    cos = np.array([[0.62, 0.56], [0.0, 0.45]])
    gate = cos >= 0.5
    decisive = decisive_cells(cos, gate, margin=0.05)
    # T1's row margin (0.06) clears 0.05, but D2 is T1's runner-up.
    np.testing.assert_array_equal(decisive, [[True, False], [False, False]])
    assert not decisive_cells(cos, gate, margin=0.1).any()
    # Ties never decide; a lone in-gate cell does (infinite margin).
    tie = np.array([[0.8, 0.8]])
    assert not decisive_cells(tie, tie >= 0.5).any()
    lone = np.array([[0.8, np.nan]])
    np.testing.assert_array_equal(decisive_cells(lone, lone >= 0.5), [[True, False]])
    assert decisive_cells(np.zeros((0, 2)), np.zeros((0, 2), dtype=bool)).shape == (
        0,
        2,
    )


@pytest.mark.parametrize("cos22", [0.45, 0.51])
def test_botsort_runner_up_cell_is_not_discounted(cos22):
    """BoT-SORT keeps the identities geometry and appearance both prefer.

    Two static tracks 10 px apart; then D1 at +7 px and D2 at +5.5 px. Cosines
    T1-D1 0.62, T1-D2 0.56, T2-D1 0, T2-D2 ``cos22``; gate 0.5, m_assoc 0.05.
    Discounting T1's runner-up cell (T1-D2) made the swap cheaper. The paper
    preset (published min-fusion over every in-gate cell) still swaps.
    """
    e = np.eye(8)
    u1, u2 = e[0], e[1]
    v1 = 0.62 * e[0] + np.sqrt(1 - 0.62**2) * e[2]
    v2 = 0.56 * e[0] + cos22 * e[1]
    v2 = v2 + np.sqrt(1 - v2 @ v2) * e[3]

    def run(name, **overrides):
        tracker = build_tracker(name, **overrides)
        tracker.calibration = _calibration(m_assoc=0.05, gate=0.5)
        frames = [
            (
                t,
                [
                    make_pose((100.0, 120.0), embedding=u1),
                    make_pose((110.0, 120.0), embedding=u2),
                ],
            )
            for t in range(10)
        ]
        frames.append(
            (
                10,
                [
                    make_pose((107.0, 120.0), embedding=v1),
                    make_pose((115.5, 120.0), embedding=v2),
                ],
            )
        )
        names = run_tracker(tracker, frames, prepare=False)
        return names[9], names[10]

    before, after = run(
        "bytetrack", kalman="xywh", track_high_threshold=0.6, new_track_threshold=0.7
    )
    assert after == before
    before, after = run("botsort")
    assert after == before
    before, after = run("botsort", preset="paper")
    assert after == before[::-1]


# ------------------------------------------------------- score-gate warnings
def test_strict_high_threshold_warns(log_messages):
    """Scores never above track_high_threshold: no track can start.

    prepare() says so even though the scores clear new_track_threshold.
    """
    scene = make_scene(_one_animal(12), scores=0.7)
    tracker = build_tracker(
        "bytetrack",
        normalize_scores=False,
        track_high_threshold=0.8,
        new_track_threshold=0.3,
    )
    assert distinct_names(run_tracker(tracker, scene.frames)) == []
    assert any("track_high_threshold=0.8" in m for m in log_messages)

    # Normalized (animal preset): the same video starts its track, silently.
    log_messages.clear()
    scene = make_scene(_one_animal(12), scores=0.7)
    names = run_tracker(build_tracker("bytetrack"), scene.frames)
    assert len(distinct_names(names)) == 1
    assert not any("track_high_threshold" in m for m in log_messages)
