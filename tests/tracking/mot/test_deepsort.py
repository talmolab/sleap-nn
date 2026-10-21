"""DeepSORT and StrongSORT (`sleap_nn.tracking.mot.deepsort`).

Behavior on synthetic scenes (appearance keeps identities where geometry is
ambiguous, lost tracks come back through the cascade, the Kalman gate rejects far
detections, graceful degradation without usable appearance) plus white-box checks
of the matching rules (cascade order, StrongSORT's global motion-blended cost, NSA
Kalman updates, the appearance statistic of each tracker).
"""

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
    run_tracker,
    static_trajectories,
    track_frames,
    track_names,
    with_gaps,
)
from loguru import logger

from sleap_nn.tracking.mot import DeepSORTTracker, StrongSORTTracker
from sleap_nn.tracking.mot.appearance import AppearanceMemory
from sleap_nn.tracking.mot.association import INFEASIBLE_COST
from sleap_nn.tracking.mot.deepsort import GATING_THRESHOLD, MC_COST_ALLOWANCE
from sleap_nn.tracking.mot.kalman import KalmanFilterXYAH
from sleap_nn.tracking.mot.track import TrackState

NAMES = ["deepsort", "strongsort"]

# Vertical animals: the padded box is ~24 px wide and ~48 px tall, so the Kalman
# gate (scaled by the box height) admits a few px of unexpected motion.
VERTICAL = np.pi / 2


@pytest.fixture
def log_messages():
    """Capture loguru messages (INFO and above) emitted during a test."""
    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    yield messages
    logger.remove(handler_id)


def _unit(dim, *components):
    """A ``dim``-D vector with the given leading components (unit if they are)."""
    v = np.zeros(dim)
    v[: len(components)] = components
    return v


def _bounce(n_frames=30, speed=2.0, center=(160.0, 120.0)):
    """Two animals approach head-on, touch at the middle frame and turn back.

    A constant-velocity prediction expects them to pass through each other, so
    geometry alone swaps their identities at the turn.
    """
    half = n_frames // 2
    cx, cy = center
    d = np.abs(np.arange(n_frames) - half) * speed
    traj = np.empty((n_frames, 2, 2))
    traj[:, 0] = np.stack([cx - d, np.full(n_frames, cy)], axis=1)
    traj[:, 1] = np.stack([cx + d, np.full(n_frames, cy)], axis=1)
    return traj


def _confirmed_tracks(name, positions, vectors, **overrides):
    """A tracker holding one confirmed track per position (``min_hits=1``)."""
    overrides.setdefault("appearance_threshold", 0.5)
    tracker = build_tracker(name, min_hits=1, **overrides)
    objs = [
        make_pose(p, heading=VERTICAL, embedding=v) for p, v in zip(positions, vectors)
    ]
    tracker.track(objs, 0)
    assert [t.state for t in tracker.tracks] == [TrackState.CONFIRMED] * len(objs)
    return tracker


def _seed_memory(tracker, track, vec):
    """Replace a track's appearance memory by one seeded with ``vec``."""
    track.appearance = tracker._make_appearance_memory()
    track.appearance.seed(np.asarray(vec, dtype=np.float64))


# ------------------------------------------------------------------ behavior
@pytest.mark.parametrize("name", NAMES)
def test_bounce_keeps_identities_with_appearance(name):
    """Ambiguous geometry, discriminative vectors: no switch; SORT does not do better."""
    for seed in range(3):
        scene = make_scene(
            _bounce(), heading=VERTICAL, embeddings=True, position_noise=0.5, seed=seed
        )
        names = run_tracker(build_tracker(name, appearance_threshold=0.5), scene.frames)
        switches = count_id_switches(names, scene.gt)
        assert switches == 0
        assert len(distinct_names(names)) == 2

        scene = make_scene(
            _bounce(), heading=VERTICAL, embeddings=True, position_noise=0.5, seed=seed
        )
        sort_switches = count_id_switches(
            run_tracker(build_tracker("sort"), scene.frames), scene.gt
        )
        assert switches <= sort_switches


@pytest.mark.parametrize("name", NAMES)
def test_bounce_swaps_without_appearance(name):
    """Without vectors (geometric degradation) the same scene swaps.

    So appearance is what keeps the identities in the test above.
    """
    scene = make_scene(_bounce(), heading=VERTICAL, position_noise=0.5)
    names = run_tracker(build_tracker(name, appearance_threshold=0.5), scene.frames)
    assert count_id_switches(names, scene.gt) > 0


@pytest.mark.parametrize("name", NAMES)
def test_lost_track_rematched_after_gap_keeps_id(name):
    """A LOST track re-matches by appearance after a gap < track_buffer.

    Only the appearance stage can match it: the geometric stage takes tracks with
    ``time_since_update == 1`` only.
    """
    traj = linear_trajectories([(80, 120), (240, 120)], [(1, 0), (1, 0)], 30)
    gap = range(10, 18)  # 8 missing frames < track_buffer (30)
    scene = make_scene(
        with_gaps(traj, {0: gap}), heading=VERTICAL, embeddings=True, seed=2
    )
    tracker = build_tracker(name, appearance_threshold=0.5)
    track_frames(tracker, scene.frames[:18], finalize=False)
    lost = [t for t in tracker.tracks if t.state == TrackState.LOST]
    assert len(lost) == 1 and lost[0].time_since_update == 8
    lost_id = lost[0].public_id
    frame_idx, objs = scene.frames[18]
    tracker.track(objs, frame_idx)
    assert lost[0].state == TrackState.CONFIRMED and lost[0].public_id == lost_id
    assert objs[0].track is tracker._track_objects[lost_id]
    for frame_idx, objs in scene.frames[19:]:
        tracker.track(objs, frame_idx)
    tracker.finalize()
    names = track_names(scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2

    # The same reappearance with a foreign vector is not re-identified.
    scene = make_scene(
        with_gaps(traj, {0: gap}), heading=VERTICAL, embeddings=True, seed=2
    )
    foreign = identity_vectors(3, 16, seed=2)[2]  # orthogonal to both identities
    assert np.abs(scene.id_vectors @ foreign).max() < 1e-9
    for _, objs in scene.frames[18:]:
        objs[0].identity_embedding = sio.Embedding(foreign.astype(np.float32))
    names = run_tracker(build_tracker(name, appearance_threshold=0.5), scene.frames)
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_gate_rejects_far_detection_with_identical_appearance(name):
    """Identical vector but far outside the Kalman gate: no match, new track."""
    vec = _unit(16, 1.0)
    frames = [
        (t, [make_pose((100.0, 120.0), heading=VERTICAL, embedding=vec)])
        for t in range(5)
    ]
    far = make_pose((140.0, 120.0), heading=VERTICAL, embedding=vec)
    frames.append((5, [far]))
    tracker = build_tracker(name, appearance_threshold=0.5)
    track_frames(tracker, frames)
    assert far.track is None
    assert [t.state for t in tracker.tracks] == [
        TrackState.LOST,
        TrackState.TENTATIVE,
    ]

    # White box: the far cell is gated even though the cosine is 1.
    tracker = _confirmed_tracks(name, [(100.0, 120.0)], [vec])
    objs = [
        make_pose((101.0, 120.0), heading=VERTICAL, embedding=vec),
        make_pose((140.0, 120.0), heading=VERTICAL, embedding=vec),
    ]
    dets = tracker._detections(objs, 1)
    cols = np.arange(2)
    d2 = tracker._mahalanobis(tracker.tracks, dets, cols)
    assert d2[0, 0] < GATING_THRESHOLD < d2[0, 1]
    if name == "deepsort":
        cost = tracker._appearance_cost(tracker.tracks, dets, cols)
    else:
        cost = tracker._mc_cost(tracker.tracks, dets, cols)
    assert cost[0, 0] < 0.1 and cost[0, 1] >= 0.9 * INFEASIBLE_COST


def test_cascade_prefers_recently_updated_tracks():
    """DeepSORT serves cascade level 1 (tsu == 1) before older tracks.

    Even when an older (LOST) track would be the better appearance match. LOST
    tracks are in the cascade pool.
    """
    dim = 16
    tracker = _confirmed_tracks(
        "deepsort", [(100.0, 120.0), (160.0, 120.0)], [_unit(dim, 1.0)] * 2
    )
    recent, older = tracker.tracks
    older.mean, older.cov = recent.mean.copy(), recent.cov.copy()
    recent.time_since_update = 1
    older.time_since_update, older.state = 2, TrackState.LOST
    query = _unit(dim, 1.0)
    _seed_memory(tracker, recent, _unit(dim, 0.7, np.sqrt(1 - 0.49)))
    _seed_memory(tracker, older, _unit(dim, 0.95, 0.0, np.sqrt(1 - 0.9025)))
    det = make_pose((100.0, 120.0), heading=VERTICAL, embedding=query)
    dets = tracker._detections([det], 1)

    result = tracker._associate(dets)
    assert [(t.tid, j) for t, j, _ in result.matches] == [(recent.tid, 0)]

    # Out of the gate (cos 0.4 < 0.5) at level 1: level 2 takes the LOST track.
    _seed_memory(tracker, recent, _unit(dim, 0.4, np.sqrt(1 - 0.16)))
    result = tracker._associate(dets)
    assert [(t.tid, j) for t, j, _ in result.matches] == [(older.tid, 0)]


def test_strongsort_global_assignment():
    """StrongSORT solves one assignment over every confirmed track.

    So the better (older, LOST) appearance match wins over the more recent track.
    """
    dim = 16
    tracker = _confirmed_tracks(
        "strongsort", [(100.0, 120.0), (160.0, 120.0)], [_unit(dim, 1.0)] * 2
    )
    recent, older = tracker.tracks
    older.mean, older.cov = recent.mean.copy(), recent.cov.copy()
    recent.time_since_update = 1
    older.time_since_update, older.state = 2, TrackState.LOST
    _seed_memory(tracker, recent, _unit(dim, 0.7, np.sqrt(1 - 0.49)))
    _seed_memory(tracker, older, _unit(dim, 0.95, 0.0, np.sqrt(1 - 0.9025)))
    det = make_pose((100.0, 120.0), heading=VERTICAL, embedding=_unit(dim, 1.0))
    dets = tracker._detections([det], 1)
    result = tracker._associate(dets)
    assert [(t.tid, j) for t, j, _ in result.matches] == [(older.tid, 0)]


def test_strongsort_mc_cost_hand_checked():
    """MC cost: lambda * (1 - cos) + (1 - lambda) * d^2, gated cells at 1e5 first."""
    dim = 16
    tracker = _confirmed_tracks(
        "strongsort", [(100.0, 100.0)], [_unit(dim, 1.0)], appearance_threshold=0.7
    )
    track = tracker.tracks[0]
    # XYAH state at (100, 100), h = 40 -> R std (1/20) h = 2 px (variance 4); the
    # position covariance 5 gives S = 9 I on the center: d^2 = (dx^2 + dy^2) / 9.
    track.mean = np.array([100.0, 100.0, 0.5, 40.0, 0.0, 0.0, 0.0, 0.0])
    track.cov = np.diag([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _seed_memory(tracker, track, _unit(dim, 1.0))
    query = _unit(dim, 0.9, np.sqrt(1 - 0.81))  # cosine 0.9
    objs = [
        make_pose((103.0, 103.0), embedding=query),  # d^2 = 2
        make_pose((106.0, 106.0), embedding=query),  # d^2 = 8 > 5.9915: gated
        make_pose((103.0, 103.0)),  # no vector: d_app = 1
    ]
    dets = tracker._detections(objs, 1)
    cols = np.arange(3)
    np.testing.assert_allclose(
        tracker._mahalanobis(tracker.tracks, dets, cols), [[2.0, 8.0, 2.0]]
    )
    lam = tracker.params.mc_lambda
    assert lam == pytest.approx(0.98)
    expected = [
        0.98 * 0.1 + 0.02 * 2.0,  # 0.138
        0.98 * INFEASIBLE_COST + 0.02 * 8.0,
        0.98 * 1.0 + 0.02 * 2.0,  # 1.02
    ]
    cost = tracker._mc_cost(tracker.tracks, dets, cols)
    np.testing.assert_allclose(cost, [expected], rtol=1e-6)

    # Cap (1 - 0.7) + 0.05 = 0.35: only the first detection is matched.
    assert MC_COST_ALLOWANCE == 0.05
    track.time_since_update = 1
    result = tracker._associate(dets)
    assert [(t.tid, j) for t, j, _ in result.matches] == [(track.tid, 0)]


def test_appearance_statistics():
    """DeepSORT scores the gallery max, StrongSORT the EMA.

    The gallery max falls back to the EMA while the gallery is empty.
    """
    dim = 16
    u1, u2 = _unit(dim, 1.0), _unit(dim, 0.0, 1.0)
    costs = {}
    for name in NAMES:
        # mc_lambda = 1: StrongSORT's cost is its pure appearance term.
        tracker = _confirmed_tracks(name, [(100.0, 100.0)], [u1], mc_lambda=1.0)
        track = tracker.tracks[0]
        track.appearance = AppearanceMemory(ema_alpha=0.8, gallery_size=30)
        track.appearance.seed(u1)
        track.appearance.update(u2, gated=True)
        dets = tracker._detections(
            [make_pose((100.0, 100.0), heading=VERTICAL, embedding=u2)], 1
        )
        if name == "deepsort":
            costs[name] = tracker._appearance_cost(tracker.tracks, dets, np.arange(1))
        else:
            costs[name] = tracker._mc_cost(tracker.tracks, dets, np.arange(1))
    ema_cos = 0.2 / np.hypot(0.8, 0.2)
    assert DeepSORTTracker.appearance_statistic == "gallery_max"
    assert StrongSORTTracker.appearance_statistic == "ema"
    np.testing.assert_allclose(costs["deepsort"], [[0.0]], atol=1e-6)
    np.testing.assert_allclose(costs["strongsort"], [[1.0 - ema_cos]], atol=1e-6)

    # Empty gallery: the gallery statistic is the EMA.
    memory = AppearanceMemory(gallery_size=30)
    memory.seed(u1)
    memory.gallery.clear()
    np.testing.assert_allclose(memory.gallery_max_cos(u1[None]), [1.0])


def test_gallery_sizes():
    """DeepSORT galleries hold gallery_budget vectors; StrongSORT keeps gallery_size."""
    tracker = build_tracker("deepsort", gallery_budget=7, gallery_size=3)
    assert tracker._make_appearance_memory().gallery_size == 7
    tracker = build_tracker("strongsort", gallery_budget=7, gallery_size=3)
    assert tracker._make_appearance_memory().gallery_size == 3
    assert tracker._make_appearance_memory().ema_alpha == tracker.params.ema_alpha


@pytest.mark.parametrize("name", NAMES)
def test_spawn_seeds_the_gallery(name):
    """A new track's memory is seeded with its spawn vector (cascade-usable)."""
    vec = _unit(16, 0.0, 0.0, 1.0)
    tracker = build_tracker(name, appearance_threshold=0.5)
    tracker.track([make_pose((50.0, 50.0), embedding=vec)], 0)
    memory = tracker.tracks[0].appearance
    assert len(memory.gallery) == 1
    np.testing.assert_allclose(memory.ema, vec, atol=1e-6)


def test_strongsort_nsa_kalman_update():
    """StrongSORT's filter is NSA: the update uses R * (1 - score)^2."""
    assert build_tracker("deepsort").kf.nsa is False
    tracker = build_tracker("strongsort", score_mode="instance")
    assert tracker.params.nsa is True and tracker.kf.nsa is True
    assert isinstance(tracker.kf, KalmanFilterXYAH)

    mean, cov = tracker.kf.initiate(np.array([0.0, 0.0, 20.0, 40.0]))
    r = KalmanFilterXYAH()._measurement_var(mean)
    _, s = tracker.kf.project(mean, cov, conf=0.9)
    np.testing.assert_allclose(np.diag(s) - np.diag(cov)[:4], r * 0.01)
    # Confidence is clipped below 1, so R never vanishes; gating is un-scaled.
    _, s = tracker.kf.project(mean, cov, conf=1.0)
    np.testing.assert_allclose(np.diag(s) - np.diag(cov)[:4], r * 0.01**2)

    # Through the tracker: the match update is the NSA update with the det score.
    frames = [(t, [make_pose((60.0 + 3 * t, 80.0), score=0.7)]) for t in range(2)]
    tracker.track(frames[0][1], 0)
    track = tracker.tracks[0]
    pred_mean, pred_cov = tracker.kf.predict(track.mean, track.cov)
    tracker.track(frames[1][1], 1)
    box = tracker._detections(frames[1][1], 1).boxes[0]
    nsa_mean, nsa_cov = tracker.kf.update(pred_mean, pred_cov, box, conf=0.7)
    np.testing.assert_allclose(track.mean, nsa_mean)
    np.testing.assert_allclose(track.cov, nsa_cov)
    plain_mean, _ = KalmanFilterXYAH().update(pred_mean, pred_cov, box)
    assert not np.allclose(track.mean, plain_mean)


@pytest.mark.parametrize("name", NAMES)
def test_low_score_detections_are_ignored(name):
    """Scores <= track_low_threshold never match (even by appearance) or spawn."""
    vec = _unit(16, 1.0)
    tracker = _confirmed_tracks(
        name,
        [(100.0, 120.0)],
        [vec],
        preset="paper",
        appearance_threshold=0.5,
        score_mode="instance",
    )
    low = tracker.params.track_low_threshold
    assert low == pytest.approx(0.8 if name == "deepsort" else 0.6)
    tracker.tracks[0].time_since_update = 1
    objs = [
        make_pose((100.0, 120.0), heading=VERTICAL, embedding=vec, score=low),
        make_pose((200.0, 120.0), heading=VERTICAL, embedding=vec, score=0.95),
    ]
    result = tracker._associate(tracker._detections(objs, 1))
    assert result.matches == []
    assert result.no_spawn == {0}


# --------------------------------------------------------------- degradation
@pytest.mark.parametrize("name", NAMES)
def test_no_vectors_degrade_to_geometry(name, log_messages):
    """Vector-less input does not raise: the tracker degrades to geometry.

    The degradation is logged once. Rejecting such input up front is the caller's
    job, keyed on ``requires_vectors`` (apply_tracking / CLI).
    """
    cls = DeepSORTTracker if name == "deepsort" else StrongSORTTracker
    assert cls.requires_vectors is True and cls.uses_appearance is True
    traj = linear_trajectories([(40, 50), (40, 120), (60, 190)], [(3, 0)] * 3, 30)
    keep = [t for t in range(30) if not 12 <= t < 16]  # gap: tsu > 1 afterwards
    scene = make_scene(traj[keep], frame_indices=keep)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    appearance = tracker.resolved["appearance"]
    assert appearance["enabled"] is False
    assert "no appearance vectors" in appearance["reason"]
    degraded = [m for m in log_messages if "skipping the appearance stage" in m]
    assert len(degraded) == 1


def _static_scene(n_frames=400, collapsed=False):
    """Three static, well-separated animals with (separable or collapsed) vectors."""
    traj = static_trajectories([(60, 60), (60, 180), (200, 120)], n_frames)
    scene = make_scene(traj, embeddings=True, position_noise=1.0, seed=1)
    if collapsed:
        rng = np.random.default_rng(0)
        base = scene.id_vectors[0]
        for _, objs in scene.frames:
            for obj in objs:
                v = base + 0.05 * rng.standard_normal(base.shape)
                obj.identity_embedding = sio.Embedding(
                    (v / np.linalg.norm(v)).astype(np.float32)
                )
    return scene


@pytest.mark.parametrize("name", NAMES)
def test_calibration_disables_appearance(name, log_messages):
    """Calibration disables a non-separating embedding (auto threshold).

    The tracker then matches by geometry.
    """
    scene = _static_scene(collapsed=True)
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    cal = tracker.calibration
    assert cal.status == "ok" and cal.appearance_for_association is False
    assert tracker._appearance_gate(tracker.appearance_statistic) is None
    assert tracker.resolved["appearance"]["enabled"] is False
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3
    assert sum("skipping the appearance stage" in m for m in log_messages) == 1


@pytest.mark.parametrize("name", NAMES)
def test_calibrated_gate_matches_the_statistic(name):
    """Auto threshold, separable embedding: each tracker gates its own statistic.

    DeepSORT uses the calibrated gallery-max gate, StrongSORT the EMA gate.
    """
    scene = _static_scene()
    tracker = build_tracker(name)
    names = run_tracker(tracker, scene.frames)
    cal = tracker.calibration
    assert cal.appearance_for_association is True
    expected = cal.gallery_cos_gate if name == "deepsort" else cal.ema_cos_gate
    assert tracker._appearance_gate(tracker.appearance_statistic) == expected
    assert tracker.resolved["appearance"]["enabled"] is True
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 3


@pytest.mark.parametrize("name", NAMES)
def test_vectorless_track_matched_by_geometry(name):
    """A confirmed track without any vector is matched by geometry after a miss.

    With appearance on, such a track is not lost forever at its first miss.
    """
    traj = linear_trajectories([(80, 120), (240, 120)], [(1, 0), (1, 0)], 20)
    scene = make_scene(
        with_gaps(traj, {1: [8, 9]}), heading=VERTICAL, embeddings=True, seed=3
    )
    for (_, objs), ids in zip(scene.frames, scene.gt):
        for obj, gid in zip(objs, ids):
            if gid == 1:
                obj.identity_embedding = None
    tracker = build_tracker(name, appearance_threshold=0.5)
    names = run_tracker(tracker, scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
