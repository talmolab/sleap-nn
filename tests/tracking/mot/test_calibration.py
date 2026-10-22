"""Tests for `sleap_nn.tracking.mot.calibration` (VideoCalibration / calibrate)."""

import json
import time

import attrs
import numpy as np
import pytest
import sleap_io as sio
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.tracking.mot.calibration import (
    MAX_TRIAL_STRIDE,
    MIN_ASSOC_RANK1,
    MIN_MOTION_BODY,
    MIN_TRIALS,
    MOTION_BODY_FACTOR,
    NEG_COLLISION_WARN,
    VideoCalibration,
    calibrate,
    frame_steps,
    frame_stride,
    pair_geometry,
    resolve_body_size,
    resolve_time_base,
    score_normalizer,
    steps_between,
    stride_from_counts,
)
from sleap_nn.tracking.mot.detections import FrameDetections, extract_detections


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


def _frame(frame_idx, centroids, vectors=None, half=10.0, scores=0.9, n_visible=4):
    """FrameDetections of square bodies (side 2 * half) at `centroids`."""
    c = np.asarray(centroids, dtype=float).reshape(-1, 2)
    n = len(c)
    raw = np.concatenate([c - half, c + half], axis=1)
    pad = 0.1 * 2 * half
    if vectors is None:
        emb, has = None, np.zeros(n, dtype=bool)
    else:
        emb = np.asarray(vectors, dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        has = np.ones(n, dtype=bool)
    return FrameDetections(
        frame_idx=int(frame_idx),
        sources=[None] * n,
        boxes=raw + pad * np.array([-1.0, -1.0, 1.0, 1.0]),
        raw_boxes=raw,
        centroids=c.copy(),
        keypoints=np.zeros((n, 4, 2)),
        scores=np.broadcast_to(np.asarray(scores, dtype=float), (n,)).copy(),
        embeddings=emb,
        has_embedding=has,
        valid=np.ones(n, dtype=bool),
        n_visible=np.full(n, n_visible),
        n_nodes=4,
    )


def _video(
    n_frames=600,
    n_animals=4,
    dim=128,
    noise=0.03,
    centers=None,
    seed=0,
    frame_indices=None,
    step=2.0,
    spacing=200.0,
):
    """Random-walking, well-separated animals with identity-clustered vectors."""
    rng = np.random.default_rng(seed)
    start = np.stack(
        [np.arange(n_animals) * spacing + 100.0, np.full(n_animals, 300.0)], axis=1
    )
    walk = np.cumsum(rng.normal(0, step, (n_frames, n_animals, 2)), axis=0) + start
    if dim and centers is None:
        centers = rng.normal(size=(n_animals, dim))
    if frame_indices is None:
        frame_indices = range(n_frames)
    frames = []
    for f, fidx in enumerate(frame_indices):
        vecs = None
        if dim:
            vecs = centers + rng.normal(0, noise, (n_animals, dim)) * np.linalg.norm(
                centers, axis=1, keepdims=True
            )
        frames.append(_frame(fidx, walk[f], vecs))
    return frames


def test_separable_identities_calibrate_ok():
    """(a) Well-separated identity embeddings -> appearance on for both uses."""
    cal = calibrate(_video())

    assert cal.status == "ok"
    assert cal.n_frames == 600 and cal.n_detections == 2400
    assert cal.frame_step == 1 and cal.regular_stride
    np.testing.assert_allclose(cal.body_size, np.hypot(20, 20))
    assert cal.nn_distance > 100
    assert cal.n_anchors == 4
    assert 0 < cal.step_p95 <= cal.step_p99 < 10
    assert cal.n_trials >= 100
    assert cal.assoc_rank1 >= MIN_ASSOC_RANK1
    assert cal.appearance_for_association
    assert cal.mu_neg < cal.ema_cos_gate < cal.mu_pos
    assert cal.mu_neg < cal.gallery_cos_gate < cal.mu_pos
    assert cal.m_assoc >= 0.02
    assert cal.link_thresh is not None and cal.link_thresh < cal.mu_pos
    assert cal.link_margin is not None and cal.link_recall >= 0.2
    assert cal.appearance_for_linking
    assert cal.neg_collision_rate is not None
    assert cal.neg_collision_rate <= NEG_COLLISION_WARN
    assert len(cal.score_quantiles) == 5
    np.testing.assert_allclose(cal.score_quantiles, 0.9)


def test_collapsed_embeddings_disable_association_and_warn(caplog):
    """(b) Two animals per shared vector (a class/sex embedder) -> warning, off."""
    rng = np.random.default_rng(1)
    classes = rng.normal(size=(2, 128))
    centers = classes[[0, 0, 1, 1]]
    cal = calibrate(_video(centers=centers))

    assert cal.status == "ok"
    assert cal.assoc_rank1 < MIN_ASSOC_RANK1
    assert not cal.appearance_for_association
    assert not cal.appearance_for_linking
    # 2 of the 6 same-frame pairs are same-class -> ~1/3 collide.
    assert cal.neg_collision_rate > 0.2
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("class/sex" in r.getMessage() for r in warnings)
    infos = [r for r in caplog.records if r.levelname == "INFO"]
    assert any("MOT calibration" in r.getMessage() for r in infos)


def test_no_vectors_still_calibrates_geometry():
    """(c) No vectors -> status no_vectors; geometry is still measured."""
    cal = calibrate(_video(dim=0))

    assert cal.status == "no_vectors"
    assert cal.n_trials == 0 and cal.n_anchors == 4
    np.testing.assert_allclose(cal.body_size, np.hypot(20, 20))
    assert np.isfinite(cal.step_p95) and np.isfinite(cal.step_p99)
    assert cal.nn_distance > 100
    for field in ("assoc_rank1", "m_assoc", "ema_cos_gate", "link_thresh"):
        assert getattr(cal, field) is None
    assert cal.neg_collision_rate is None
    assert not cal.appearance_for_association and not cal.appearance_for_linking


def test_single_animal_is_insufficient():
    """(d) One animal -> no competitor -> insufficient; nn falls back to body."""
    cal = calibrate(_video(n_animals=1))

    assert cal.status == "insufficient"
    assert cal.n_trials == 0 and cal.n_anchors == 1
    assert cal.nn_distance == cal.body_size
    assert cal.assoc_rank1 is None and cal.link_margin is None
    assert not cal.appearance_for_association and not cal.appearance_for_linking


def test_too_few_trials_is_insufficient():
    """Fewer than MIN_TRIALS g=1 trials -> insufficient, appearance off."""
    # 40 frames x 2 animals: 78 split positions even at stride 1.
    cal = calibrate(_video(n_frames=40, n_animals=2))
    assert cal.status == "insufficient"
    assert 0 < cal.n_trials < MIN_TRIALS
    assert not cal.appearance_for_association


def test_short_clip_reaches_min_trials():
    """The trial stride adapts: a 400-frame clip of 2 animals is calibrated.

    With a fixed stride of 10 it had ~80 trials (< MIN_TRIALS) and appearance
    was always off for short clips.
    """
    cal = calibrate(_video(n_frames=400, n_animals=2))
    assert cal.status == "ok"
    assert cal.n_trials >= MIN_TRIALS
    assert cal.appearance_for_association and cal.appearance_for_linking

    from sleap_nn.tracking.mot.calibration import _trial_stride

    assert _trial_stride(798, 2000) == 1
    assert _trial_stride(12000, 2000) == 6
    assert _trial_stride(10**6, 2000) == MAX_TRIAL_STRIDE
    assert _trial_stride(0, 2000) == 1
    assert _trial_stride(50, 0) == MAX_TRIAL_STRIDE


@pytest.mark.parametrize("n_frames", [0, 5])
def test_empty_input_defaults(n_frames):
    """(e) No detections at all -> defaults, never raises."""
    frames = [FrameDetections.empty(i, n_nodes=4) for i in range(n_frames)]
    cal = calibrate(frames)

    assert cal.status == "no_vectors"
    assert cal.n_frames == n_frames and cal.n_detections == 0
    assert cal.frame_step == 1 and cal.regular_stride
    assert cal.body_size == 50.0 and cal.nn_distance == 50.0
    assert np.isnan(cal.step_p95) and cal.score_quantiles == ()
    assert cal.n_anchors == 0
    json.dumps(cal.as_dict())


def test_degenerate_inputs_do_not_raise():
    """Invalid detections, missing vectors and one-frame videos are handled."""
    frame = _frame(0, [[10, 10], [100, 100]], np.eye(2, 8))
    frame.valid[1] = False
    frame.has_embedding[0] = False
    frame.embeddings[0] = 0.0
    cal = calibrate([frame])
    # The only vector is on the invalid detection.
    assert cal.n_detections == 1 and cal.status == "no_vectors"

    frame = _frame(0, [[10, 10], [100, 100]], np.eye(2, 8))
    cal = calibrate([frame])
    assert cal.status == "insufficient" and cal.n_trials == 0

    # Non-finite vector values count as missing.
    frames = _video(n_frames=30, n_animals=2, dim=8)
    frames[3].embeddings[:] = np.nan
    calibrate(frames)


def test_determinism_with_subsampling():
    """(f) Same input and seed -> identical calibration (trials subsampled)."""
    frames = _video()
    a = calibrate(frames, max_trials=150, seed=3)
    b = calibrate(frames, max_trials=150, seed=3)
    assert a == b
    assert a.status == "ok" and a.n_trials == 150
    c = calibrate(frames, max_trials=150, seed=4)
    assert c.status == "ok" and c.n_trials == 150


def test_stride_two_is_regular():
    """(g) Every other frame -> frame_step 2, regular; steps are processed frames."""
    frames = _video(n_frames=300, frame_indices=range(0, 600, 2))
    cal = calibrate(frames)
    assert cal.frame_step == 2 and cal.regular_stride
    dense = calibrate(_video(n_frames=300))
    # Same walk (seed) -> same per-step displacement whatever the stride.
    np.testing.assert_allclose(cal.step_p95, dense.step_p95)


def test_irregular_frames_are_not_regular():
    """(g) A sparse, irregular frame list -> regular_stride False."""
    idx = [0, 3, 10, 11, 25, 40, 41, 70, 71, 120, 200, 201, 260]
    cal = calibrate(_video(n_frames=len(idx), frame_indices=idx))
    assert cal.frame_step == 1
    assert not cal.regular_stride


def _constant_velocity(frame_indices, speed=2.0):
    """Two animals moving `speed` px per FRAME along x."""
    frames = []
    for f in frame_indices:
        frames.append(_frame(f, [[100 + speed * f, 100], [100 + speed * f, 400]]))
    return frames


def test_step_statistics_are_per_step():
    """Displacements are divided by the steps between processed frames."""
    frames = _constant_velocity(list(range(100)) + list(range(106, 206)))
    cal = calibrate(frames)
    assert cal.frame_step == 1 and cal.regular_stride
    # The 6-frame gap moved 12 px = 6 steps of 2 px.
    np.testing.assert_allclose(cal.step_p99, 2.0)
    assert cal.n_anchors == 2

    # Deltas alternating 1 and 3 frames: still regular (half are frame_step 1),
    # so every per-step displacement is 2 px; forced step mode counts processed
    # frames instead (6 px over a 3-frame delta).
    idx = np.cumsum([0] + [1, 3] * 60)
    frames = _constant_velocity(idx)
    cal = calibrate(frames)
    assert cal.frame_step == 1 and cal.regular_stride
    np.testing.assert_allclose([cal.step_p95, cal.step_p99], 2.0)
    cal_steps = calibrate(frames, time_base="steps")
    np.testing.assert_allclose(cal_steps.step_p99, 6.0)

    with pytest.raises(ValueError, match="time_base"):
        calibrate(frames, time_base="seconds")


def _single_node_frames(n_frames, xs, speed=1.0, y=100.0):
    """Single-node poses at ``x + speed * f`` for each ``x`` in ``xs``."""
    skel = sio.Skeleton(nodes=["c"], name="one")
    frames = []
    for f in range(n_frames):
        objs = [
            sio.PredictedInstance.from_numpy(
                np.array([[100.0 + x + speed * f, y]]),
                skeleton=skel,
                score=0.9,
                point_scores=np.array([0.9]),
            )
            for x in xs
        ]
        frames.append(extract_detections(objs, f))
    return frames


def test_single_node_body_size_is_motion_derived():
    """(h) Single-node poses have no extent: the body size comes from motion.

    It used to fall back to the inter-animal spacing (80 px here), which sized
    the centroid gate by how sparse the arena is, not by the animal.
    """
    cal = calibrate(_single_node_frames(20, (0.0, 80.0, 200.0), speed=1.0))
    # NN distances: 80, 80, 120 -> median 80.
    np.testing.assert_allclose(cal.nn_distance, 80.0)
    assert cal.n_anchors == 3
    np.testing.assert_allclose(cal.step_p99, 1.0)
    # 8 x 1 px/step, floored at MIN_MOTION_BODY.
    assert cal.body_size_source == "motion"
    np.testing.assert_allclose(cal.body_size, MIN_MOTION_BODY)
    assert cal.status == "no_vectors"

    # Faster animals: MOTION_BODY_FACTOR x step_p99, capped at the spacing.
    cal = calibrate(_single_node_frames(20, (0.0, 300.0), speed=5.0))
    np.testing.assert_allclose(cal.body_size, MOTION_BODY_FACTOR * 5.0)
    cal = calibrate(_single_node_frames(20, (0.0, 30.0), speed=5.0))
    assert cal.body_size_source == "motion"
    np.testing.assert_allclose(cal.body_size, 30.0)

    # No anchor (too short to chain): the spacing, with a warning.
    cal = calibrate(_single_node_frames(5, (0.0, 80.0, 200.0)))
    assert cal.n_anchors == 0 and cal.body_size_source == "nn_distance"
    np.testing.assert_allclose(cal.body_size, 80.0)


def test_single_fast_animal_is_sized_from_motion():
    """One single-node animal at 40 px/frame: no spacing, so links are not
    distance-gated, and the motion sizes the body (it was 50 px, too small for
    the centroid similarity to follow it)."""
    cal = calibrate(_single_node_frames(30, (0.0,), speed=40.0))
    assert cal.n_anchors == 1
    np.testing.assert_allclose([cal.step_p95, cal.step_p99], 40.0)
    assert cal.body_size_source == "motion"
    np.testing.assert_allclose(cal.body_size, 320.0)
    assert cal.nn_distance == cal.body_size


def test_body_size_warning_and_override(caplog):
    """No extent and no motion -> WARNING; an explicit body size wins."""
    cal = calibrate(_single_node_frames(3, (0.0,)))
    assert cal.body_size_source == "default" and cal.body_size == 50.0
    assert any(
        "body size" in r.getMessage() and r.levelname == "WARNING"
        for r in caplog.records
    )
    cal = calibrate(_single_node_frames(20, (0.0, 80.0, 200.0)), body_size=33.0)
    assert cal.body_size_source == "override" and cal.body_size == 33.0
    # Measured sizes are unchanged.
    cal = calibrate(_video(n_frames=50, dim=0))
    assert cal.body_size_source == "measured"
    assert "(measured)" in cal.summary()


def test_resolve_body_size_chain():
    """measured > motion (clipped) > spacing > default."""
    nan = float("nan")
    assert resolve_body_size(40.0, 3.0, 500.0) == (40.0, "measured")
    assert resolve_body_size(nan, 3.0, 500.0) == (24.0, "motion")
    assert resolve_body_size(nan, 0.5, 500.0) == (MIN_MOTION_BODY, "motion")
    assert resolve_body_size(nan, 100.0, 500.0) == (500.0, "motion")
    assert resolve_body_size(nan, 3.0, nan) == (24.0, "motion")
    assert resolve_body_size(nan, nan, 500.0) == (500.0, "nn_distance")
    assert resolve_body_size(nan, nan, nan) == (50.0, "default")
    assert resolve_body_size(0.0, nan, 0.0) == (50.0, "default")


def test_score_normalizer():
    """q95 of the raw scores (floored), 1.0 without scores or calibration."""
    assert score_normalizer(None) == 1.0
    assert score_normalizer(VideoCalibration()) == 1.0
    cal = calibrate(_video(n_frames=30, dim=0))
    assert score_normalizer(cal) == pytest.approx(0.9)
    cal = VideoCalibration(score_quantiles=(0.0, 0.0, 0.0, 0.0, 0.0))
    assert score_normalizer(cal) == 1e-6


def test_step_helpers():
    """The one stride / rounding rule shared by trackers, calibration, linker."""
    assert stride_from_counts({}) == (1, True)
    assert stride_from_counts({2: 3, 5: 1}) == (2, True)
    assert stride_from_counts({1: 2, 3: 2}) == (1, True)  # tie -> smallest
    assert stride_from_counts({1: 1, 2: 1, 3: 1}) == (1, False)
    assert stride_from_counts({0: 9, -3: 2, 4: 1}) == (4, True)
    assert frame_stride([0, 3, 6, 9, 30]) == (3, True)
    assert frame_stride([5]) == (1, True)

    assert resolve_time_base("auto", True) == "frames"
    assert resolve_time_base("auto", False) == "steps"
    assert resolve_time_base("steps", True) == "steps"
    with pytest.raises(ValueError, match="time_base"):
        resolve_time_base("seconds", True)

    # max(1, round(delta / step)), half to even; non-positive deltas -> 1.
    assert steps_between(5, 2) == 2 and steps_between(7, 2) == 4
    assert steps_between(6, 2) == 3 and steps_between(1, 2) == 1
    assert steps_between(0, 1) == 1 and steps_between(-50, 1) == 1
    assert steps_between(10, 0) == 10
    np.testing.assert_array_equal(steps_between(np.array([2, 5, 9]), 2), [1, 2, 4])

    idx = [0, 2, 4, 6, 11, 13, 15, 17]
    np.testing.assert_array_equal(
        frame_steps(idx, "frames", 2), [0, 1, 2, 3, 5, 6, 7, 8]
    )
    np.testing.assert_array_equal(frame_steps(idx, "steps", 2), np.arange(8))
    assert frame_steps([], "frames").shape == (0,)
    with pytest.raises(ValueError, match="resolved"):
        frame_steps(idx, "auto")


def test_time_base_is_recorded():
    """VideoCalibration.time_base is the resolved time base."""
    cal = calibrate(_video(n_frames=30, dim=0))
    assert cal.time_base == "frames"
    idx = [0, 3, 10, 11, 25, 40, 41, 70, 71, 120, 200, 201, 260]
    assert calibrate(_video(n_frames=len(idx), frame_indices=idx)).time_base == (
        "steps"
    )
    frames = _video(n_frames=len(idx), frame_indices=idx)
    assert calibrate(frames, time_base="frames").time_base == "frames"
    # A regular video with a hole, forced to steps: steps count processed frames.
    idx = list(range(20)) + list(range(300, 320))
    cal = calibrate(_constant_velocity(idx), time_base="steps")
    assert cal.time_base == "steps" and cal.regular_stride
    np.testing.assert_array_equal(
        frame_steps(idx, cal.time_base, cal.frame_step), np.arange(40)
    )
    assert "time base steps" in cal.summary()


def _points(frame_idx, xs, pseudo_box_size=None):
    """Single-node pose detections at ``(x, 100)``, as a tracker extracts them."""
    skel = sio.Skeleton(nodes=["c"], name="one")
    objs = [
        sio.PredictedInstance.from_numpy(
            np.array([[float(x), 100.0]]),
            skeleton=skel,
            score=0.9,
            point_scores=np.array([0.9]),
        )
        for x in xs
    ]
    return extract_detections(objs, frame_idx, pseudo_box_size=pseudo_box_size)


def test_pair_geometry():
    """Online spacing / unambiguous links, with calibrate's rules."""
    a, b = _points(0, [100, 300]), _points(1, [104, 303])
    nn, links = pair_geometry(a, b)
    np.testing.assert_allclose(nn, [199.0, 199.0])
    np.testing.assert_allclose(np.sort(links), [3.0, 4.0])
    # Gate: links longer than max_dist are dropped.
    _, links = pair_geometry(a, b, max_dist=3.5)
    np.testing.assert_allclose(links, [3.0])
    # No previous frame: spacing only. A lone detection: no spacing, no gate.
    nn, links = pair_geometry(None, b)
    assert len(links) == 0 and len(nn) == 2
    nn, links = pair_geometry(_points(0, [0]), _points(1, [90]))
    assert len(nn) == 0
    np.testing.assert_allclose(links, [90.0])
    # Ambiguous: the runner-up is not 2x farther.
    _, links = pair_geometry(_points(0, [100, 130]), _points(1, [115, 140]))
    assert len(links) <= 1
    # Pseudo boxes (a tracker's body-size squares) do not make two close
    # animals duplicates: the spacing is measured as calibrate measures it.
    nn, _ = pair_geometry(None, _points(0, [100, 120], pseudo_box_size=200.0))
    np.testing.assert_allclose(nn, [20.0, 20.0])


def test_masks_and_sleap_io_vectors():
    """Masks count toward body size; sio identity_embedding vectors are used."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 16))
    frames = []
    for f in range(40):
        objs = []
        for k, x0 in enumerate((5, 60)):
            data = np.zeros((40, 100), dtype=bool)
            data[10:20, x0 + f % 3 : x0 + f % 3 + 10] = True
            mask = sio.PredictedSegmentationMask.from_numpy(data, score=0.8)
            mask.identity_embedding = sio.Embedding(
                centers[k] + 0.01 * rng.normal(size=16)
            )
            objs.append(mask)
        frames.append(extract_detections(objs, f, carrier="mask"))
    cal = calibrate(frames)
    np.testing.assert_allclose(cal.body_size, np.hypot(10, 10))
    assert cal.n_anchors == 2
    # Vectors were found (too few trials for "ok" in a 40-frame clip).
    assert cal.status == "insufficient" and cal.n_trials > 0
    np.testing.assert_allclose(cal.score_quantiles, 0.8)


def test_duplicates_and_crossings_cut_anchors():
    """Anchor chains are cut at duplicates and ambiguous (close) neighbors."""
    # Animal 0 has a duplicate detection 1 px away in every frame; animal 1 is
    # isolated.
    frames = [
        _frame(f, [[100 + f, 100], [101 + f, 100], [500, 500 + f]]) for f in range(30)
    ]
    cal = calibrate(frames)
    assert cal.n_anchors == 1

    # Two animals crossing paths: chains break around the crossing.
    frames = [_frame(f, [[100 + 4 * f, 100], [300 - 4 * f, 100]]) for f in range(51)]
    cal = calibrate(frames)
    assert cal.n_anchors >= 3


def test_as_dict_is_json_serializable_and_plain():
    """(i) as_dict(): JSON-native Python types, NaN -> None."""
    cal = calibrate(_video(n_frames=400, n_animals=3))
    assert cal.status == "ok"
    d = cal.as_dict()
    json.loads(json.dumps(d, allow_nan=False))
    assert set(d) == set(attrs.fields_dict(VideoCalibration))
    for value in d.values():
        assert value is None or type(value) in (str, int, float, bool, list)
    for name in attrs.fields_dict(VideoCalibration):
        value = getattr(cal, name)
        assert value is None or type(value) in (str, int, float, bool, tuple)

    default = VideoCalibration().as_dict()
    assert default["step_p95"] is None and default["score_quantiles"] == []
    json.dumps(default, allow_nan=False)
    assert "MOT calibration" in VideoCalibration().summary()
    assert "appearance ok" in cal.summary()


def test_perf_smoke():
    """(j) 3000 frames x 4 animals x 128-D vectors calibrate in well under 3 s."""
    frames = _video(n_frames=3000, n_animals=4)
    t0 = time.perf_counter()
    cal = calibrate(frames)
    elapsed = time.perf_counter() - t0
    assert cal.status == "ok"
    assert elapsed < 3.0, elapsed


def test_trial_statistics_match_appearance_memory():
    """The calibrated statistics are exactly those of `AppearanceMemory`."""
    from sleap_nn.tracking.mot import calibration as cal_mod
    from sleap_nn.tracking.mot.appearance import AppearanceMemory

    frames = _video(n_frames=80, n_animals=2, dim=8, noise=0.3)
    flat = cal_mod._flatten(frames)
    nn, dup, _ = cal_mod._same_frame_pairs(flat)
    nxt = cal_mod._anchor_links(flat, dup, 0.5 * np.median(nn))
    roots = cal_mod._chain_roots(nxt)
    anchor_of = np.unique(roots, return_inverse=True)[1]
    assert anchor_of.max() == 1  # two whole-video anchors
    step_idx = np.arange(len(frames))
    av = cal_mod._anchor_vectors(flat, anchor_of, 2, step_idx, 0.8)
    stats = cal_mod._run_trials(
        av,
        3,
        gallery_size=6,
        query_len=4,
        max_trials=1000,
        rng=np.random.default_rng(0),
    )
    # 2 x 77 candidate positions, far below max_trials: every one is a trial.
    stride = cal_mod._trial_stride(int(sum(av.lengths)) - 2 * 3, 1000)
    assert stride == 1

    def memory(anchor, upto):
        mem = AppearanceMemory(ema_alpha=0.8, gallery_size=6)
        for i in range(upto + 1):
            mem.update(av.vectors[av.offsets[anchor] + i])
        return mem

    # Trials in order: anchor 0 then 1, split positions 3, 3 + stride, ...
    expected = []
    for a in (0, 1):
        for j in range(3, int(av.lengths[a]), stride):
            # The query stops at the anchor's last vector.
            end = min(j + 4, int(av.lengths[a]))
            q = av.vectors[av.offsets[a] + j : av.offsets[a] + end]
            true, comp = memory(a, j - 3), memory(1 - a, j - 3)
            expected.append(
                (
                    true.ema_cos(q[0])[0],
                    true.gallery_max_cos(q[0])[0],
                    true.gallery_topk_mean(q, 5).mean(),
                    comp.ema_cos(q[0])[0],
                    comp.gallery_topk_mean(q, 5).mean(),
                )
            )
    expected = np.array(expected)
    assert len(stats) == len(expected)
    np.testing.assert_allclose(stats.true_ema, expected[:, 0], atol=1e-5)
    np.testing.assert_allclose(stats.true_gal, expected[:, 1], atol=1e-5)
    np.testing.assert_allclose(stats.true_link, expected[:, 2], atol=1e-5)
    np.testing.assert_allclose(
        np.concatenate(stats.comp_ema), expected[:, 3], atol=1e-5
    )
    np.testing.assert_allclose(
        np.concatenate(stats.comp_link), expected[:, 4], atol=1e-5
    )


def test_appearance_switch_skips_the_vector_work():
    """``appearance=False``: same geometry, no appearance part, status "skipped"."""
    frames = _video(n_frames=300, n_animals=3)
    full = calibrate(frames)
    geo = calibrate(frames, appearance=False)
    assert full.status == "ok" and geo.status == "skipped"
    for name in (
        "n_frames",
        "n_detections",
        "frame_step",
        "regular_stride",
        "time_base",
        "body_size",
        "body_size_source",
        "nn_distance",
        "step_p95",
        "step_p99",
        "score_quantiles",
        "n_anchors",
    ):
        assert getattr(geo, name) == getattr(full, name), name
    assert geo.n_trials == 0 and geo.link_thresh is None
    assert geo.neg_collision_rate is None and not geo.appearance_for_linking
    assert "appearance not calibrated (not used)" in geo.summary()


def test_link_rule_uses_the_linkers_gallery_length():
    """The linking rule ignores ``gallery_size`` (the linker scores 30-galleries).

    ``gallery_size`` only sets the ``g = 1`` gallery statistic (the tracker's
    memory, e.g. DeepSORT's ``gallery_budget``).
    """
    from sleap_nn.tracking.mot.calibration import LINK_GALLERY_LEN
    from sleap_nn.tracking.mot.linking import GALLERY_SIZE

    assert GALLERY_SIZE == LINK_GALLERY_LEN
    frames = _video(n_frames=400, n_animals=4, noise=0.3)
    ref = calibrate(frames)
    for gallery_size in (5, 100):
        cal = calibrate(frames, gallery_size=gallery_size)
        assert cal.link_thresh == ref.link_thresh
        assert cal.link_margin == ref.link_margin
        assert cal.link_thresh_by_len == ref.link_thresh_by_len
    assert calibrate(frames, gallery_size=5).gallery_cos_gate != ref.gallery_cos_gate
    # track_buffer=1: the long-gap trials are rerun with the linker's gallery.
    one = calibrate(frames, track_buffer=1, gallery_size=5)
    assert one.link_thresh == calibrate(frames, track_buffer=1).link_thresh


def test_link_rule_is_calibrated_per_query_length():
    """Shorter queries are noisier: their threshold is calibrated separately."""
    from sleap_nn.tracking.mot.calibration import LINK_QUERY_LEN, link_rule_tables

    cal = calibrate(_video(n_frames=600, n_animals=4, noise=0.3))
    assert cal.status == "ok"
    assert len(cal.link_thresh_by_len) == LINK_QUERY_LEN
    assert len(cal.link_margin_by_len) == LINK_QUERY_LEN
    assert cal.link_thresh_by_len[-1] == pytest.approx(cal.link_thresh)
    assert cal.link_margin_by_len[-1] == cal.link_margin
    # A single query vector has a much wider competitor distribution.
    assert cal.link_thresh_by_len[0] > cal.link_thresh_by_len[-1] + 0.02
    json.dumps(cal.as_dict(), allow_nan=False)

    thresh, margin = link_rule_tables(cal)
    np.testing.assert_allclose(thresh, cal.link_thresh_by_len)
    # Hand-built / older calibrations: the full-length rule for every length.
    flat = VideoCalibration(link_thresh=0.7, link_margin=0.05)
    thresh, margin = link_rule_tables(flat, 4)
    np.testing.assert_allclose(thresh, 0.7)
    np.testing.assert_allclose(margin, 0.05)
    # A length without a calibrated margin cannot be decided.
    gappy = VideoCalibration(
        link_thresh_by_len=(0.9, 0.8, 0.7), link_margin_by_len=(None, 0.1, 0.05)
    )
    thresh, margin = link_rule_tables(gappy, 5)
    assert np.isnan(thresh[0]) and np.isnan(margin[0])
    np.testing.assert_allclose(thresh[1:], [0.8, 0.7, 0.7, 0.7])
    assert np.isnan(link_rule_tables(None)[0]).all()


@pytest.mark.parametrize(
    "name, overrides, expected",
    [
        ("sort", {}, "skipped"),
        ("bytetrack", {}, "skipped"),
        ("sort", {"link_tracklets": True}, "ok"),
        ("sort", {"max_tracks": 4}, "ok"),
        ("botsort", {}, "ok"),
    ],
)
def test_prepare_calibrates_appearance_only_when_read(name, overrides, expected):
    """Geometry-only trackers without linking skip the appearance calibration."""
    from sleap_nn.tracking.mot import MOT_TRACKERS, MOTParams

    tracker = MOT_TRACKERS[name](
        MOTParams.for_tracker(name, **overrides), carrier="pose", n_nodes=4
    )
    objs = _sio_frames(_video(n_frames=300, n_animals=4))
    tracker.prepare(objs)
    assert tracker.calibration.status == expected


def test_prepare_calibrates_deepsorts_gallery_gate_on_its_budget(monkeypatch):
    """DeepSORT's gallery gate is calibrated on its own memory (gallery_budget)."""
    from sleap_nn.tracking.mot import MOT_TRACKERS, MOTParams, base

    seen = {}
    real = base.calibrate

    def spy(frames, **kwargs):
        seen.update(kwargs)
        return real(frames, **kwargs)

    monkeypatch.setattr(base, "calibrate", spy)
    params = MOTParams.for_tracker("deepsort", gallery_budget=100, gallery_size=7)
    tracker = MOT_TRACKERS["deepsort"](params, carrier="pose", n_nodes=4)
    tracker.prepare(_sio_frames(_video(n_frames=60, n_animals=2)))
    assert seen["gallery_size"] == 100 and seen["appearance"] is True
    params = MOTParams.for_tracker("botsort", gallery_size=7)
    tracker = MOT_TRACKERS["botsort"](params, carrier="pose", n_nodes=4)
    tracker.prepare(_sio_frames(_video(n_frames=60, n_animals=2)))
    assert seen["gallery_size"] == 7


def _sio_frames(frames):
    """``FrameDetections`` -> ``(frame_idx, [PredictedInstance])`` with vectors."""
    skel = sio.Skeleton(nodes=["a", "b", "c", "d"], name="sq")
    out = []
    for fd in frames:
        objs = []
        for k in range(len(fd)):
            x0, y0, x1, y1 = fd.raw_boxes[k]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            inst = sio.PredictedInstance.from_numpy(
                pts, skeleton=skel, score=0.9, point_scores=np.full(4, 0.9)
            )
            if fd.embeddings is not None:
                inst.identity_embedding = sio.Embedding(fd.embeddings[k])
            objs.append(inst)
        out.append((fd.frame_idx, objs))
    return out
