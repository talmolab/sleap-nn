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
    MIN_ASSOC_RANK1,
    NEG_COLLISION_WARN,
    VideoCalibration,
    calibrate,
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
    cal = calibrate(_video(n_frames=120, n_animals=2))
    assert cal.status == "insufficient"
    assert 0 < cal.n_trials < 100
    assert not cal.appearance_for_association


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


def test_single_node_body_size_falls_back_to_nn_distance():
    """(h) Single-node poses have no extent: body_size = nn_distance."""
    skel = sio.Skeleton(nodes=["c"], name="one")
    frames = []
    for f in range(20):
        objs = [
            sio.PredictedInstance.from_numpy(
                np.array([[100.0 + x + f, 100.0]]),
                skeleton=skel,
                score=0.9,
                point_scores=np.array([0.9]),
            )
            for x in (0.0, 80.0, 200.0)
        ]
        frames.append(extract_detections(objs, f))
    cal = calibrate(frames)
    # NN distances: 80, 80, 120 -> median 80.
    np.testing.assert_allclose(cal.nn_distance, 80.0)
    np.testing.assert_allclose(cal.body_size, 80.0)
    assert cal.n_anchors == 3
    assert cal.status == "no_vectors"


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

    def memory(anchor, upto):
        mem = AppearanceMemory(ema_alpha=0.8, gallery_size=6)
        for i in range(upto + 1):
            mem.update(av.vectors[av.offsets[anchor] + i])
        return mem

    # Trials in order: anchor 0 then 1, split positions 3, 13, 23, ...
    expected = []
    for a in (0, 1):
        for j in range(3, int(av.lengths[a]), 10):
            q = av.vectors[av.offsets[a] + j : av.offsets[a] + j + 4]
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
