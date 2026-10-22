"""Tests for OC-SORT / Deep OC-SORT (`sleap_nn.tracking.mot.ocsort`).

Geometry used throughout: a :data:`conftest.SKELETON` pose with heading 0 has a
40 x 16 px raw box, padded by 4 px per side to 48 x 24 px; two such boxes offset
by ``(dx, dy)`` have IoU ``(48 - dx)(24 - dy) / (2 * 1152 - (48 - dx)(24 - dy))``.
"""

import contextlib

import numpy as np
import pytest
import sleap_io as sio
from conftest import (
    SKELETON,
    build_tracker,
    count_id_switches,
    distinct_names,
    identity_vectors,
    linear_trajectories,
    make_pose,
    make_scene,
    pose_points,
    run_tracker,
    track_names,
    with_gaps,
)
from loguru import logger

import sleap_nn.tracking.mot.ocsort as ocsort_module
from sleap_nn.tracking.mot import DeepOCSORTTracker, MOTParams, OCSORTTracker
from sleap_nn.tracking.mot.association import adaptive_weight
from sleap_nn.tracking.mot.calibration import VideoCalibration
from sleap_nn.tracking.mot.kalman import oru_replay
from sleap_nn.tracking.mot.track import TrackState


@contextlib.contextmanager
def _warnings():
    """Collect loguru WARNING messages emitted inside the block."""
    messages = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _calibration(**kwargs):
    """A calibration with usable appearance statistics (overridable)."""
    values = dict(
        status="ok",
        n_trials=200,
        assoc_rank1=1.0,
        m_assoc=0.1,
        ema_cos_gate=0.5,
        gallery_cos_gate=0.5,
        mu_pos=0.9,
        mu_neg=0.1,
        appearance_for_association=True,
    )
    values.update(kwargs)
    return VideoCalibration(**values)


def _make(cls, name, **overrides):
    """A tracker of class ``cls`` with ``name``'s parameters (5-node poses)."""
    return cls(MOTParams.for_tracker(name, **overrides), carrier="pose", n_nodes=5)


def _track_all(tracker, frames):
    """Run ``track`` over ``frames`` (no prepare / finalize)."""
    for frame_idx, objs in frames:
        tracker.track(objs, frame_idx)


def _one_animal(n_frames, velocity=(3.0, 0.0), start=(40.0, 120.0)):
    return linear_trajectories([start], [velocity], n_frames)


# --------------------------------------------------------------------- ORU
@pytest.mark.parametrize("name", ["ocsort", "deepocsort"])
def test_oru_refinds_occluded_animal_and_replays(name):
    """An animal occluded for 10 steps while moving linearly is re-found by the same
    track; its state is the frozen state replayed along the virtual trajectory
    (then the real update), and the post-gap velocity matches the true one."""
    velocity = np.array([3.0, 1.0])
    traj = with_gaps(_one_animal(40, velocity=velocity), {0: range(12, 22)})
    scene = make_scene(traj)
    tracker = build_tracker(name)
    _track_all(tracker, scene.frames[:22])  # last frame of the gap: 21
    (track,) = tracker.tracks
    assert track.time_since_update == 10 and track.state == TrackState.LOST
    frozen_mean, frozen_cov, frozen_step = track.frozen
    assert frozen_step == 12  # frozen at the first miss
    last_obs = track.last_observation.copy()

    frame_idx, objs = scene.frames[22]
    box = tracker._detections(list(objs), frame_idx).boxes[0]
    mean, cov = oru_replay(
        tracker.kf, frozen_mean, frozen_cov, last_obs, box, n_gap=11, conf=0.9
    )
    mean, cov = tracker.kf.update(mean, cov, box, conf=0.9)
    tracker.track(objs, frame_idx)
    assert track.frozen is None and track.time_since_update == 0
    np.testing.assert_allclose(track.mean, mean)
    np.testing.assert_allclose(track.cov, cov)
    np.testing.assert_allclose(track.mean[4:6], velocity, rtol=0.02, atol=0.02)

    _track_all(tracker, scene.frames[23:])
    names = track_names(scene.frames)
    assert distinct_names(names) == ["track_0"]
    assert all(n == ["track_0"] for n in names[4:] if n)


def _growing_pose(t, velocity=(2.0, 0.0), growth=1.0):
    """A pose moving linearly whose body length grows ``growth`` px per frame."""
    center = np.array([60.0, 120.0]) + t * np.asarray(velocity)
    pts = pose_points(center, length=30.0 + growth * t)
    return sio.PredictedInstance.from_numpy(
        pts, skeleton=SKELETON, score=0.9, point_scores=np.full(5, 0.9)
    )


def test_deepocsort_oru_restores_size_velocity():
    """Deep OC-SORT stops the box-size velocity while a track is frozen; ORU
    replays the gap from the frozen state, so after a 10-step occlusion the
    re-found track's size velocity is close to the truth -- without ORU it stays
    near the stopped value."""
    gap = range(12, 22)

    def run(use_oru):
        cls = type("Probe", (DeepOCSORTTracker,), {"use_oru": use_oru})
        tracker = _make(cls, "deepocsort")
        for t in range(23):
            tracker.track([] if t in gap else [_growing_pose(t)], t)
        (track,) = tracker.tracks
        return track.mean

    # Padded box: w = 1.2 * length (+1.2 px/frame), h = 16 + 0.2 * length.
    truth = np.array([2.0, 0.0, 1.2, 0.2])
    with_oru, without_oru = run(True), run(False)
    np.testing.assert_allclose(with_oru[4:8], truth, atol=0.05)
    assert abs(without_oru[6] - truth[2]) > 0.3


@pytest.mark.parametrize("kalman", ["auto", "xysr", "xyah"])
def test_deepocsort_zeroes_size_velocity_only_on_xywh(kalman):
    """The frozen-track size-velocity stop applies to the [cx, cy, w, h] filter
    only (other filters keep their velocities and never index past their state)."""
    tracker = _make(DeepOCSORTTracker, "deepocsort", kalman=kalman)
    expected = (6, 7) if tracker.kalman_kind == "xywh" else ()
    assert tuple(tracker.lost_zero_velocity) == expected
    scene = make_scene(with_gaps(_one_animal(14), {0: [10, 11, 12]}))
    _track_all(tracker, scene.frames)
    assert len(tracker.tracks) == 1


# --------------------------------------------------------------------- OCM
def _crossing_with_burst(lane=4.0, speed=2.0, n_hist=8, burst=(8.0, 3.0)):
    """Two animals approaching head-on in lanes ``lane`` px apart; on the last
    frame they dart past each other (``burst`` px beyond the prediction) and the
    detector puts each one 1 px into the other's lane.

    The last frame's overlaps with the (exact) Kalman predictions prefer the
    swap: IoU 0.665 for each crossed pair vs 0.574 for each true pair. Seen from
    each track's observation 3 steps back, though, its true detection lies ahead
    along its motion and the other one behind: OC-SORT's momentum term reverses
    the decision.
    """
    a = linear_trajectories([(160 - speed * n_hist, 120.0)], [(speed, 0.0)], n_hist)
    b = linear_trajectories(
        [(160 + speed * n_hist, 120.0 + lane)], [(-speed, 0.0)], n_hist
    )
    # Both predictions are at x = 160 on the burst frame.
    burst_frame = np.array(
        [
            [
                [160.0 + burst[0], 120.0 + burst[1]],
                [160.0 - burst[0], 120.0 + lane - burst[1]],
            ]
        ]
    )
    traj = np.concatenate([np.concatenate([a, b], axis=1), burst_frame], axis=0)
    return make_scene(traj)


def test_ocm_resolves_head_on_crossing():
    """Head-on crossing where overlap alone swaps the animals: OC-SORT's
    direction term keeps the identities; without it (inertia 0) they swap."""
    scene = _crossing_with_burst()
    names = run_tracker(build_tracker("ocsort"), scene.frames)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2

    scene = _crossing_with_burst()
    names = run_tracker(build_tracker("ocsort", inertia=0.0), scene.frames)
    assert names[-1] == list(reversed(names[-2]))  # swapped on the burst frame
    assert count_id_switches(names, scene.gt) > 0


def _jump_scene(speed=4.0, jump=30.0, n_hist=8):
    """One animal moving along x; on the last frame it is detected ``jump`` px
    ahead of its Kalman prediction (and ``jump + speed`` past its last
    observation)."""
    traj = _one_animal(n_hist + 1, velocity=(speed, 0.0))
    traj[-1, 0, 0] += jump
    return make_scene(traj)


class _HostileOCSORT(OCSORTTracker):
    """OC-SORT whose stage-1 objective strongly prefers INADMISSIBLE pairs."""

    def _first_stage(self, tracks, dets, cols, sim):
        valid, _ = super()._first_stage(tracks, dets, cols, sim)
        return valid, np.where(valid, -10.0, 10.0)


@pytest.mark.parametrize("hostile", [False, True])
def test_admissibility_is_geometric_only(hostile):
    """A pair below min_match_similarity is never matched, however large the
    momentum (or any other objective) term."""
    # Jump of 30 px: IoU 0.23 with the prediction, 0.17 with the last
    # observation; dead ahead, so the momentum is maximal (+0.45 at inertia 1).
    scene = _jump_scene()
    tracker = _make(_HostileOCSORT if hostile else OCSORTTracker, "ocsort", inertia=1.0)
    _track_all(tracker, scene.frames)
    old, new = sorted(tracker.tracks, key=lambda t: t.tid)
    assert old.state == TrackState.LOST and old.time_since_update == 1
    assert new.state == TrackState.TENTATIVE and new.hits == 0
    assert scene.frames[-1][1][0].track is None


def test_one_to_one_pattern_bypasses_the_objective(monkeypatch):
    """Admissible pairs forming a 1:1 pattern are taken as is (no assignment
    solve, objective ignored); otherwise the masked solve runs."""
    calls = []
    original = ocsort_module.linear_assignment_masked

    def spy(score, valid):
        calls.append(np.asarray(valid).shape)
        return original(score, valid)

    monkeypatch.setattr(ocsort_module, "linear_assignment_masked", spy)

    # Well separated: every frame is a 1:1 pattern, so even a hostile objective
    # never matters and the solver is never called.
    traj = linear_trajectories([(40, 60), (40, 180)], [(4, 0), (4, 0)], 20)
    scene = make_scene(traj)
    tracker = _make(_HostileOCSORT, "ocsort")
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert calls == []
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
    # Matches carry their geometric similarity (identical boxes: 1.0).
    scores = [o.tracking_score for _, objs in scene.frames[5:] for o in objs]
    assert scores == pytest.approx([1.0] * len(scores), abs=1e-3)

    # Overlapping lanes: the pattern is not 1:1 and the masked solve decides.
    traj = linear_trajectories([(40, 120), (40, 126)], [(4, 0), (4, 0)], 12)
    scene = make_scene(traj)
    names = run_tracker(_make(OCSORTTracker, "ocsort"), scene.frames, prepare=False)
    assert calls and calls[0] == (2, 2)
    assert count_id_switches(names, scene.gt) == 0


# --------------------------------------------------------------------- OCR
@pytest.mark.parametrize("name", ["ocsort", "deepocsort"])
def test_ocr_recovers_from_last_observation(name):
    """An animal that stops while occluded reappears where it was last seen: its
    Kalman prediction has coasted away (IoU 0.23) but the last observation
    matches, so OCR recovers the track (SORT, prediction-only, does not)."""
    moving = _one_animal(10, velocity=(6.0, 0.0))
    stopped = np.repeat(moving[-1:], 9, axis=0)
    traj = with_gaps(np.concatenate([moving, stopped]), {0: range(10, 14)})
    scene = make_scene(traj)
    names = run_tracker(build_tracker(name), scene.frames, prepare=False)
    assert distinct_names(names) == ["track_0"]
    recovered = scene.frames[14][1][0]
    assert recovered.track.name == "track_0"
    # Scored by the last-observation similarity (identical boxes).
    assert recovered.tracking_score == pytest.approx(1.0)

    scene = make_scene(traj)
    names = run_tracker(build_tracker("sort"), scene.frames, prepare=False)
    assert len(distinct_names(names)) == 2


# -------------------------------------------------------------------- BYTE
def test_low_score_detections_only_with_use_byte():
    """Without use_byte low-score detections are ignored (never tracked, never
    spawn); with it they keep their track through the low-score stretch."""
    scores = np.full((20, 1), 0.9)
    scores[8:13] = 0.4  # track_low_threshold 0.1 < 0.4 <= track_high 0.6
    traj = _one_animal(20)

    scene = make_scene(traj, scores=scores)
    tracker = build_tracker("ocsort")
    names = run_tracker(tracker, scene.frames)
    assert distinct_names(names) == ["track_0"]
    assert all(n == [None] for n in names[8:13])
    assert tracker._next_tid == 1  # low-score detections spawned nothing

    scene = make_scene(traj, scores=scores)
    names = run_tracker(build_tracker("ocsort", use_byte=True), scene.frames)
    assert distinct_names(names) == ["track_0"]
    assert all(n == ["track_0"] for n in names[4:])


@pytest.mark.parametrize("use_byte", [False, True])
def test_byte_stage_runs_before_ocr(use_byte):
    """A track whose prediction matches a low-score detection and whose last
    observation matches a high-score one: BYTE (when on) takes the low-score
    detection first; otherwise OCR takes the high-score one."""
    traj = _one_animal(8, velocity=(8.0, 0.0))
    scene = make_scene(traj)
    tracker = build_tracker("ocsort", use_byte=use_byte)
    _track_all(tracker, scene.frames)
    # Prediction at x = 40 + 8 * 8 = 104; last observation at x = 96.
    low = make_pose((104.0, 120.0), score=0.4)
    high = make_pose((76.0, 120.0), score=0.9)  # IoU 0.26 / 0.41 (pred / last)
    tracker.track([high, low], 8)
    if use_byte:
        assert low.track.name == "track_0" and high.track is None
    else:
        assert high.track.name == "track_0" and low.track is None
        assert high.tracking_score == pytest.approx(28.0 / 68.0)


def test_deepocsort_has_no_byte_stage():
    """Deep OC-SORT associates high-score detections only (use_byte ignored)."""
    with _warnings() as messages:
        tracker = build_tracker("deepocsort", use_byte=True)
    assert any("use_byte" in m for m in messages)
    assert not tracker._byte_enabled()
    scores = np.full((20, 1), 0.9)
    scores[8:13] = 0.4
    scene = make_scene(_one_animal(20), scores=scores)
    names = run_tracker(tracker, scene.frames, prepare=False)
    assert all(n == [None] for n in names[8:13])


# ---------------------------------------------------------- Deep OC-SORT
def _ambiguous_crossing(embeddings):
    """Four animals for 301 frames (enough for the per-video calibration); two
    approach head-on in lanes 4 px apart and on frame 150 are detected in each
    other's lane, so overlap and momentum both prefer the swap."""
    traj = linear_trajectories(
        [(10, 120), (310, 124), (20, 30), (20, 210)],
        [(1, 0), (-1, 0), (1, 0), (1, 0)],
        301,
    )
    traj[150, 0] = (164, 124)
    traj[150, 1] = (156, 120)
    return make_scene(traj, embeddings=embeddings)


def test_deepocsort_appearance_resolves_ambiguous_crossing():
    """With discriminative vectors the calibrated appearance term keeps the
    identities where OC-SORT (and Deep OC-SORT without vectors) swap them."""
    scene = _ambiguous_crossing(embeddings=True)
    tracker = build_tracker("deepocsort")
    with _warnings() as messages:
        names = run_tracker(tracker, scene.frames)
    assert tracker.calibration.appearance_for_association
    assert tracker.resolved["appearance"]["enabled"]
    assert tracker.resolved["appearance"]["association_term"] == "standardized"
    assert not any("raw EMA cosine" in m for m in messages)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 4

    scene = _ambiguous_crossing(embeddings=True)
    names = run_tracker(build_tracker("ocsort"), scene.frames)
    assert count_id_switches(names, scene.gt) > 0

    scene = _ambiguous_crossing(embeddings=False)
    names = run_tracker(build_tracker("deepocsort"), scene.frames)
    assert count_id_switches(names, scene.gt) > 0


def test_deepocsort_adaptive_ema_memory():
    """Tracks get a score-adaptive EMA memory floored at track_high_threshold."""
    scene = make_scene(_one_animal(6), embeddings=True)
    tracker = build_tracker("deepocsort", preset="paper")
    run_tracker(tracker, scene.frames, prepare=False)
    memory = tracker.tracks[0].appearance
    assert memory.adaptive and memory.ema_alpha == 0.95
    assert memory.score_floor == tracker.params.track_high_threshold == 0.6


def _two_tracks(appearance_threshold=0.5, **overrides):
    """A Deep OC-SORT tracker with two confirmed, well-separated tracks."""
    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 8)
    scene = make_scene(traj, embeddings=True)
    tracker = build_tracker(
        "deepocsort", appearance_threshold=appearance_threshold, **overrides
    )
    tracker.calibration = _calibration()
    _track_all(tracker, scene.frames)
    return tracker, scene.id_vectors


def test_standardized_appearance():
    """E' maps the midpoint of mu_neg..mu_pos -> 0 and mu_pos -> +1, negative
    evidence is dropped (clipped at 0); missing -> 0; no usable calibration -> 0
    everywhere."""
    tracker, _ = _two_tracks()
    cos = np.array([[0.9, 0.1, 0.5, np.nan], [1.0, -0.5, 0.3, 0.7]])
    expected = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.5]])
    np.testing.assert_allclose(tracker._standardized_appearance(cos), expected)

    for calibration in (
        None,
        _calibration(mu_pos=None, mu_neg=None, status="insufficient"),
        _calibration(mu_pos=0.4, mu_neg=0.4),
        _calibration(mu_pos=float("nan")),
    ):
        tracker.calibration = calibration
        np.testing.assert_array_equal(tracker._standardized_appearance(cos), 0.0)


def test_association_term_falls_back_to_raw_cosine():
    """An enabled appearance gate without a usable calibrated spread (no
    calibration, too few trials, degenerate spread) uses the reference's raw EMA
    cosine instead of a neutral E' = 0; the calibrated spread standardizes; a
    disabled gate ("auto" without calibration) gives no term."""
    tracker, _ = _two_tracks()
    cos = np.array([[0.9, 0.1, 0.5, np.nan], [1.0, -0.5, 0.3, 0.7]])
    raw = np.nan_to_num(cos, nan=0.0)
    assert tracker._association_term() == "standardized"
    np.testing.assert_allclose(
        tracker._association_appearance(cos), tracker._standardized_appearance(cos)
    )
    for calibration in (
        None,
        _calibration(mu_pos=None, mu_neg=None, status="insufficient", n_trials=8),
        _calibration(mu_pos=0.4, mu_neg=0.4),
        _calibration(mu_pos=float("nan")),
    ):
        tracker.calibration = calibration
        assert tracker._association_term() == "raw_cosine"
        np.testing.assert_allclose(tracker._association_appearance(cos), raw)

    tracker = build_tracker("deepocsort")  # "auto", uncalibrated: disabled
    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 8)
    _track_all(tracker, make_scene(traj, embeddings=True).frames)
    assert tracker._association_term() == "off"
    np.testing.assert_array_equal(tracker._association_appearance(cos), 0.0)


def _bounce_scene(n_frames=30, speed=2.0, center=(160.0, 120.0), seed=0):
    """Two animals walking sideways (heading vertical) meet head-on and turn back.

    Too short for the per-video calibration (status "insufficient": no
    calibrated mu_pos / mu_neg), and geometry alone swaps the identities.
    """
    d = np.abs(np.arange(n_frames) - n_frames // 2) * speed
    traj = np.empty((n_frames, 2, 2))
    traj[:, 0] = np.stack([center[0] - d, np.full(n_frames, center[1])], axis=1)
    traj[:, 1] = np.stack([center[0] + d, np.full(n_frames, center[1])], axis=1)
    return make_scene(
        traj, heading=np.pi / 2, embeddings=True, position_noise=0.5, seed=seed
    )


@pytest.mark.parametrize("prepare", [True, False])
def test_explicit_threshold_uses_appearance_without_calibrated_spread(prepare):
    """Regression: an explicit appearance_threshold on a clip too short to
    calibrate (or without prepare()) silently reduced Deep OC-SORT to OC-SORT
    (E' = 0 everywhere) while provenance reported appearance as enabled. The
    raw-cosine term now keeps the identities, the fallback is logged once and
    recorded in the provenance."""
    scene = _bounce_scene()
    names = run_tracker(build_tracker("ocsort"), scene.frames, prepare=prepare)
    assert count_id_switches(names, scene.gt) > 0  # geometry alone swaps

    scene = _bounce_scene()
    tracker = build_tracker("deepocsort", appearance_threshold=0.5)
    with _warnings() as messages:
        names = run_tracker(tracker, scene.frames, prepare=prepare)
    if prepare:
        assert tracker.calibration.status == "insufficient"
        assert tracker.calibration.mu_pos is None
    else:
        assert tracker.calibration is None
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
    assert sum("raw EMA cosine" in m for m in messages) == 1
    summary = tracker.resolved["appearance"]
    assert summary["enabled"] and summary["association_term"] == "raw_cosine"
    assert "raw EMA cosine" in summary["reason"]

    # "auto" keeps its calibrated semantics: insufficient -> geometry only.
    scene = _bounce_scene()
    tracker = build_tracker("deepocsort")
    names_auto = run_tracker(tracker, scene.frames)
    assert tracker.resolved["appearance"]["enabled"] is False
    assert tracker.resolved["appearance"]["association_term"] == "off"
    assert count_id_switches(names_auto, scene.gt) > 0


def _stage1(tracker, objs, frame_idx=8):
    """Stage-1 ``(sim, valid, score, oc_score)`` of ``objs`` vs the live tracks."""
    dets = tracker._detections(list(objs), frame_idx)
    tracks = tracker._live()
    cols = np.arange(len(dets))
    sim = tracker._geometric_similarity(tracks, dets)
    valid, score = tracker._first_stage(tracks, dets, cols, sim)
    oc_valid, oc_score = OCSORTTracker._first_stage(tracker, tracks, dets, cols, sim)
    return sim, valid, score, oc_valid, oc_score


def test_appearance_term_and_missing_vectors():
    """Objective = OC-SORT's + W * E'; a detection without a vector is neutral
    (its column equals OC-SORT's), and disabled appearance changes nothing."""
    tracker, vecs = _two_tracks()
    objs = [
        make_pose((64.0, 60.0), embedding=vecs[0]),
        make_pose((64.0, 180.0)),  # no vector
    ]
    sim, valid, score, oc_valid, oc_score = _stage1(tracker, objs)
    np.testing.assert_array_equal(valid, oc_valid)
    np.testing.assert_allclose(score[:, 1], oc_score[:, 1])
    # Own vector (cos ~ 1 -> E' = +1) adds W >= w_assoc_emb; the other track's
    # (cos ~ 0 -> E' ~ -1) has no geometric overlap, hence no appearance term.
    assert score[0, 0] - oc_score[0, 0] >= tracker.params.w_assoc_emb - 0.1
    assert sim[1, 0] == 0.0 and score[1, 0] == oc_score[1, 0]

    # "auto" with a calibration that rejects appearance: plain OC-SORT objective.
    tracker = build_tracker("deepocsort")
    tracker.calibration = _calibration(appearance_for_association=False)
    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 8)
    _track_all(tracker, make_scene(traj, embeddings=True).frames)
    sim, valid, score, oc_valid, oc_score = _stage1(tracker, objs[:1])
    np.testing.assert_array_equal(score, oc_score)
    np.testing.assert_array_equal(valid, oc_valid)


@pytest.mark.parametrize("aw_off", [False, True])
def test_adaptive_weight_or_constant(aw_off):
    """aw_off: the appearance term is w_assoc_emb * E'; otherwise the adaptive
    weight on E' (larger where appearance discriminates)."""
    traj = linear_trajectories([(40, 120), (40, 132)], [(3, 0), (3, 0)], 8)
    scene = make_scene(traj, embeddings=True)
    tracker = build_tracker("deepocsort", appearance_threshold=0.5, aw_off=aw_off)
    tracker.calibration = _calibration()
    _track_all(tracker, scene.frames)
    vecs = scene.id_vectors
    objs = [
        make_pose((64.0, 120.0), embedding=vecs[0]),
        make_pose((64.0, 132.0), embedding=vecs[1]),
    ]
    sim, valid, score, _, oc_score = _stage1(tracker, objs)
    assert (sim > 0).all()  # every pair overlaps: appearance applies everywhere
    dets = tracker._detections(list(objs), 8)
    cos = tracker._appearance_cos(tracker._live(), dets, "ema")
    e = tracker._standardized_appearance(cos)
    w = tracker.params.w_assoc_emb
    if aw_off:
        np.testing.assert_allclose(score - oc_score, w * e)
    else:
        weight = adaptive_weight(e, aw_param=tracker.params.aw_param, base=w)
        np.testing.assert_allclose(score - oc_score, weight * e)
        assert (weight > w).all()


def _veto_frames(atypical_cos=0.1, jump=14.0):
    """Animal A alone, walking sideways (heading vertical: 16 px wide boxes) at 10
    px/frame; on frame 10 it moves ``jump`` px (too far from its last observation
    for OCR) and its crop is atypical (cosine ``atypical_cos`` to its identity:
    E' < 0). B and C stand 5 px apart, so each of their rows has two admissible
    pairs (no one-to-one shortcut); D leaves after frame 7 (a lost track: a
    spare row of zero-evidence cells)."""
    vecs = np.eye(16)
    atypical = atypical_cos * vecs[0] + np.sqrt(1 - atypical_cos**2) * vecs[15]
    frames, xa = [], 40.0
    for t in range(16):
        if t > 0:
            xa += jump if t == 10 else 10.0
        objs = [
            (
                make_pose((xa, 60.0), heading=np.pi / 2, embedding=atypical)
                if t == 10
                else make_pose((xa, 60.0), heading=np.pi / 2, embedding=vecs[0])
            ),
            make_pose((200.0, 160.0), heading=np.pi / 2, embedding=vecs[1]),
            make_pose((205.0, 160.0), heading=np.pi / 2, embedding=vecs[2]),
        ]
        if t < 8:
            objs.append(make_pose((280.0, 200.0), heading=np.pi / 2, embedding=vecs[3]))
        frames.append((t, objs))
    return frames


def test_appearance_never_vetoes_an_uncontested_pair():
    """Regression: a negative appearance term on the ONLY admissible pair of its
    row and column made the masked assignment drop it for a zero-evidence
    inadmissible cell (here the lost track D's) whenever an unrelated pair (B/C)
    stopped the one-to-one shortcut; OCR could not recover the jump, so A's
    detection went untracked. Appearance never subtracts: dissimilar pairs keep
    their OC-SORT score, contested or not."""
    frames = _veto_frames()
    tracker = build_tracker("deepocsort", preset="animal")
    tracker.calibration = _calibration()
    _track_all(tracker, frames[:10])

    captured = {}
    original = tracker._first_stage

    def spy(tracks, dets, cols, sim):
        valid, score = original(tracks, dets, cols, sim)
        _, oc_score = OCSORTTracker._first_stage(tracker, tracks, dets, cols, sim)
        cos = tracker._appearance_cos(tracks, dets, "ema")[:, cols]
        captured.update(
            rows=[t.tid for t in tracks],
            valid=valid,
            score=score,
            oc_score=oc_score,
            e=tracker._association_appearance(cos),
        )
        return valid, score

    tracker._first_stage = spy
    frame_idx, objs = frames[10]
    tracker.track(objs, frame_idx)

    valid, score, oc_score, e = (
        captured[k] for k in ("valid", "score", "oc_score", "e")
    )
    a, b = captured["rows"].index(0), captured["rows"].index(1)
    assert ocsort_module.one_to_one_pattern(valid) is None
    # A's pair is uncontested and dissimilar: no penalty, its OC-SORT score.
    assert valid[a].sum() == 1 and valid[:, 0].sum() == 1 and valid[a, 0]
    assert e[a, 0] == 0.0
    assert score[a, 0] == pytest.approx(oc_score[a, 0])
    # B's competing, dissimilar pair (B vs C's detection) is not penalized either.
    assert valid[b, 2] and e[b, 2] == 0.0
    assert score[b, 2] == pytest.approx(oc_score[b, 2])

    assert objs[0].track is not None and objs[0].track.name == "track_0"
    _track_all(tracker, frames[11:])
    names = track_names(frames)
    assert all(n[0] == "track_0" for n in names[5:])


def _rescue_frames(vec_a, vec_jump, vec_b=None, n_hist=8):
    """Animal A moving along y=120 at 4 px/frame (optionally B in the next lane);
    on the last frame A is detected 32 px past its prediction (IoU 0.2 with the
    prediction, 0.14 with its last observation)."""
    frames = []
    for t in range(n_hist):
        objs = [make_pose((40.0 + 4 * t, 120.0), embedding=vec_a)]
        if vec_b is not None:
            objs.append(make_pose((40.0 + 4 * t, 124.0), embedding=vec_b))
        frames.append((t, objs))
    last = [make_pose((40.0 + 4 * n_hist + 32.0, 120.0), embedding=vec_jump)]
    if vec_b is not None:
        last.append(make_pose((40.0 + 4 * n_hist, 124.0), embedding=vec_b))
    frames.append((n_hist, last))
    return frames


@pytest.mark.parametrize(
    "case, rescued",
    [
        ("decisive", True),
        ("rescue_off", False),
        ("other_identity", False),
        ("ambiguous_neighbor", False),
        ("ocsort", False),
    ],
)
def test_rescue_only_with_decisive_appearance(case, rescued):
    """A weak-geometry pair (0.15 <= IoU < 0.3) is admissible only when its
    appearance clears the gate and tops its row and column by m_assoc."""
    vecs = identity_vectors(2)
    vec_jump = vecs[1] if case == "other_identity" else vecs[0]
    vec_b = vecs[0] if case == "ambiguous_neighbor" else None
    frames = _rescue_frames(vecs[0], vec_jump, vec_b=vec_b)
    if case == "ocsort":
        tracker = build_tracker("ocsort")
    else:
        tracker = build_tracker(
            "deepocsort", appearance_threshold=0.5, rescue=case != "rescue_off"
        )
        tracker.calibration = _calibration()
    _track_all(tracker, frames)
    jumped = frames[-1][1][0]
    if rescued:
        assert jumped.track is not None and jumped.track.name == "track_0"
        assert jumped.tracking_score == pytest.approx(16.0 / 80.0, abs=0.02)
    else:
        assert jumped.track is None


def test_rescue_is_part_of_the_one_to_one_pattern(monkeypatch):
    """The 1:1 shortcut is evaluated on the rescue-extended admissibility: a lone
    rescued pair is taken without an assignment solve."""
    calls = []
    original = ocsort_module.linear_assignment_masked
    monkeypatch.setattr(
        ocsort_module,
        "linear_assignment_masked",
        lambda score, valid: calls.append(1) or original(score, valid),
    )
    vecs = identity_vectors(2)
    frames = _rescue_frames(vecs[0], vecs[0])
    tracker = build_tracker("deepocsort", appearance_threshold=0.5)
    tracker.calibration = _calibration()
    _track_all(tracker, frames)
    assert frames[-1][1][0].track.name == "track_0"
    assert calls == []


# ------------------------------------------------------------- no vectors
def test_deepocsort_without_vectors():
    """No vectors: an explicit appearance_threshold is an error in prepare();
    "auto" warns and tracks on geometry alone; without prepare() it warns once."""
    scene = make_scene(_one_animal(10))
    tracker = build_tracker("deepocsort", appearance_threshold=0.5)
    with pytest.raises(ValueError, match="appearance vector"):
        tracker.prepare(scene.frames)
    build_tracker("deepocsort", appearance_threshold=0.5).prepare(
        []
    )  # nothing to track
    build_tracker("deepocsort", appearance_threshold=0.5).prepare([(0, [])])

    traj = linear_trajectories([(40, 60), (40, 180)], [(3, 0), (3, 0)], 20)
    with _warnings() as messages:
        scene = make_scene(traj)
        names = run_tracker(build_tracker("deepocsort"), scene.frames)
    assert any("no appearance vectors" in m for m in messages)
    assert count_id_switches(names, scene.gt) == 0
    assert len(distinct_names(names)) == 2
    scene_oc = make_scene(traj)
    assert run_tracker(build_tracker("ocsort"), scene_oc.frames) == names

    with _warnings() as messages:
        scene = make_scene(traj)
        run_tracker(build_tracker("deepocsort"), scene.frames, prepare=False)
    assert sum("no appearance vectors" in m for m in messages) == 1

    # Vectors but "auto" without calibration: one warning, geometry only.
    with _warnings() as messages:
        scene = make_scene(traj, embeddings=True)
        tracker = build_tracker("deepocsort")
        run_tracker(tracker, scene.frames, prepare=False)
    assert sum("calibration" in m for m in messages) == 1
    assert tracker._appearance_gate("ema") is None
