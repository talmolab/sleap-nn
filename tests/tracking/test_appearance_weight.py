"""`appearance_weight`: appearance as a COMPLEMENTARY association cue.

The G4 measurement behind this knob: appearance-only association
(`features="embeddings"`) LOSES to geometry on dense continuous video, because
geometry is highly informative there. Blended with geometry at 0.15-0.5 it beats
either cue alone. So appearance ships as a weight on top of a geometric score,
not as a replacement — and at the default `0.0` it must be provably inert.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.tracking.tracker import Tracker

_SKEL = sio.Skeleton(nodes=["a", "b"], name="s")


def _instance(x, y, embedding=None, score=0.9):
    """A predicted instance at `(x, y)`, optionally carrying an appearance vector."""
    inst = sio.PredictedInstance.from_numpy(
        np.array([[x, y], [x + 1.0, y]]),
        skeleton=_SKEL,
        score=score,
        point_scores=np.ones(2),
    )
    if embedding is not None:
        vec = np.asarray(embedding, dtype=np.float32)
        inst.identity_embedding = sio.Embedding(vec)
    return inst


def _track_two_frames(appearance_weight, frame0, frame1, **kwargs):
    """Track two frames and return frame 1's assigned track names."""
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        window_size=3,
        features="keypoints",
        scoring_method="oks",
        appearance_weight=appearance_weight,
        **kwargs,
    )
    tracker.track(frame0, frame_idx=0)
    tracked = tracker.track(frame1, frame_idx=1)
    return [i.track.name if i.track else None for i in tracked]


# ─────────────────────────────────────────────────────────────────────────
# Inertness at the default
# ─────────────────────────────────────────────────────────────────────────
def test_default_is_zero_and_inert():
    """`0.0` must leave the score matrix untouched, embeddings present or not."""
    tracker = Tracker.from_config(features="keypoints", scoring_method="oks")
    assert tracker.appearance_weight == 0.0

    # Same detections, with and without embeddings: identical assignments.
    plain0 = [_instance(10, 10), _instance(100, 100)]
    plain1 = [_instance(11, 10), _instance(101, 100)]
    emb0 = [_instance(10, 10, [1, 0]), _instance(100, 100, [0, 1])]
    emb1 = [_instance(11, 10, [1, 0]), _instance(101, 100, [0, 1])]

    assert _track_two_frames(0.0, plain0, plain1) == _track_two_frames(0.0, emb0, emb1)


def test_geometry_only_is_unchanged_by_missing_embeddings():
    """With no embeddings anywhere, a blend cannot change the outcome."""
    frame0 = [_instance(10, 10), _instance(100, 100)]
    frame1 = [_instance(11, 10), _instance(101, 100)]

    assert _track_two_frames(0.0, frame0, frame1) == _track_two_frames(
        0.5, frame0, frame1
    )


# ─────────────────────────────────────────────────────────────────────────
# The blend itself
# ─────────────────────────────────────────────────────────────────────────
def test_appearance_breaks_a_geometric_tie():
    """Where geometry is ambiguous, appearance decides — the point of the cue.

    Two animals swap positions between frames, so geometry alone prefers the
    swap (each detection is nearest the OTHER animal's last position). Their
    appearance vectors are orthogonal and unambiguous, so a high weight must
    recover the true identities.
    """
    a_vec, b_vec = [1.0, 0.0], [0.0, 1.0]
    frame0 = [_instance(10, 10, a_vec), _instance(30, 10, b_vec)]
    # Positions swap; appearance does not.
    frame1 = [_instance(30, 10, a_vec), _instance(10, 10, b_vec)]

    geometry = _track_two_frames(0.0, frame0, frame1)
    appearance = _track_two_frames(0.9, frame0, frame1)

    # Geometry follows position, appearance follows identity, so they disagree.
    assert geometry != appearance
    # Under appearance, detection 0 (vector a) keeps animal a's original track.
    assert appearance[0] == geometry[1]
    assert appearance[1] == geometry[0]


def test_missing_embedding_keeps_the_geometric_score():
    """A detection with no vector must still match on geometry, not vanish.

    Blending toward NaN would make its cost infinite and spawn a spurious
    track — the failure mode this guards.
    """
    frame0 = [_instance(10, 10, [1.0, 0.0]), _instance(100, 100, [0.0, 1.0])]
    # Second detection carries NO embedding.
    frame1 = [_instance(11, 10, [1.0, 0.0]), _instance(101, 100)]

    names = _track_two_frames(0.5, frame0, frame1)

    assert all(
        n is not None for n in names
    ), "an embedding-less detection lost its track"
    assert len(set(names)) == 2, "two detections must hold two distinct tracks"
    # And it matched the SAME track geometry alone would have given it.
    assert names == _track_two_frames(0.0, frame0, frame1)


@pytest.mark.parametrize("weight", [0.15, 0.3, 0.5])
def test_recommended_weights_preserve_easy_assignments(weight):
    """In the regime the reframing recommends, unambiguous tracking is unchanged."""
    a_vec, b_vec = [1.0, 0.0], [0.0, 1.0]
    frame0 = [_instance(10, 10, a_vec), _instance(200, 200, b_vec)]
    frame1 = [_instance(11, 10, a_vec), _instance(201, 200, b_vec)]

    assert _track_two_frames(weight, frame0, frame1) == _track_two_frames(
        0.0, frame0, frame1
    )


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_out_of_range_weight_is_rejected(bad):
    with pytest.raises(ValueError, match=r"appearance_weight must be in \[0.0, 1.0\]"):
        Tracker.from_config(appearance_weight=bad)


def test_appearance_only_features_reject_the_blend():
    """`features="embeddings"` is already appearance-only; blending is incoherent."""
    with pytest.raises(ValueError, match="already appearance-only"):
        Tracker.from_config(
            features="embeddings", scoring_method="cosine_sim", appearance_weight=0.3
        )


def test_appearance_only_features_are_still_allowed_alone():
    """The appearance-only regime remains available for sparse / post-occlusion use."""
    tracker = Tracker.from_config(features="embeddings", scoring_method="cosine_sim")

    assert tracker.features == "embeddings"
    assert tracker.appearance_weight == 0.0


def test_appearance_only_logs_the_blend_recommendation(caplog):
    """Choosing appearance-only should point at the measured alternative."""
    from loguru import logger

    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        Tracker.from_config(features="embeddings", scoring_method="cosine_sim")
    finally:
        logger.remove(handler_id)

    joined = "".join(messages)
    assert "appearance_weight" in joined
    assert "continuous video" in joined


# ─────────────────────────────────────────────────────────────────────────
# Guards live at ONE choke point (review finding [6]/[8])
# ─────────────────────────────────────────────────────────────────────────
def test_from_config_rejects_embeddings_with_keypoint_metric():
    """`features='embeddings'` + a keypoint/box/mask metric must fail in from_config.

    The legacy `sleap-nn track` command builds `Tracker.from_config` directly, so
    the guard cannot live only in `apply_tracking`: without it, the class default
    `scoring_method='oks'` survived and `.track()` died inside `compute_oks` with
    `AxisError: axis -2 is out of bounds for array of dimension 1`.
    """
    with pytest.raises(ValueError, match="requires scoring_method='cosine_sim'"):
        Tracker.from_config(features="embeddings")
    # The two vector-valued metrics are accepted.
    for method in ("cosine_sim", "euclidean_dist"):
        assert Tracker.from_config(features="embeddings", scoring_method=method)


@pytest.mark.parametrize("motion", [{"use_flow": True}, {"use_kalman": True}])
def test_from_config_rejects_embeddings_with_motion_models(motion):
    """Motion models shift KEYPOINTS; `get_embedding` would pass the shifted pose
    through as the "embedding" and score finite garbage against a real vector."""
    with pytest.raises(ValueError, match="does not support motion models"):
        Tracker.from_config(
            features="embeddings",
            scoring_method="cosine_sim",
            max_tracks=2,
            **motion,
        )


@pytest.mark.parametrize("motion", [{"use_flow": True}, {"use_kalman": True}])
def test_appearance_weight_rejects_motion_models(motion):
    """`appearance_weight` + a motion model was ACCEPTED and then silently dropped:
    `from_config` only forwards the weight to the base `Tracker`, so the Flow /
    Kalman subclasses ran at weight 0 while reporting success."""
    with pytest.raises(ValueError, match="does not support motion models"):
        Tracker.from_config(
            features="keypoints",
            scoring_method="oks",
            appearance_weight=0.3,
            max_tracks=2,
            **motion,
        )


def test_appearance_weight_needs_a_scale_for_distance_scores():
    """`euclidean_dist` is negative PIXELS -- unbounded, and dataset-scaled.

    Blending cosine in [-1, 1] into it leaves the weight numerically inert at any
    realistic image scale, so it is mapped through `distance_to_similarity` first
    -- which needs a length scale. There is no universal default, so the scale is
    REQUIRED for that combination rather than guessed.
    """
    with pytest.raises(ValueError, match="requires euclidean_scale"):
        Tracker.from_config(
            features="centroids",
            scoring_method="euclidean_dist",
            appearance_weight=0.5,
        )
    with pytest.raises(ValueError, match="positive number of pixels"):
        Tracker.from_config(
            features="centroids",
            scoring_method="euclidean_dist",
            appearance_weight=0.5,
            euclidean_scale=0,
        )
    assert (
        Tracker.from_config(
            features="centroids",
            scoring_method="euclidean_dist",
            appearance_weight=0.5,
            euclidean_scale=25.0,
        ).euclidean_scale
        == 25.0
    )
    # Already-bounded metrics need no scale.
    for method in ("oks", "iou", "mask_iou", "cosine_sim"):
        assert Tracker.from_config(
            features="keypoints", scoring_method=method, appearance_weight=0.5
        )


def test_geometry_only_distance_run_ignores_the_scale():
    """A `euclidean_dist` run at weight 0 must be untouched by the kernel.

    The mapping lives inside `_blend_appearance`, which weight 0 never reaches, so
    the scores stay raw negative pixels. Asserted, since the kernel would otherwise
    change every centroid-tracking run's scores.
    """
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        features="centroids",
        scoring_method="euclidean_dist",
        euclidean_scale=25.0,  # set but inert at weight 0
    )
    tracker.track([_instance(10, 10, [1, 0])], frame_idx=0)
    query = [_instance(13, 10, [1, 0])]
    track_instances = tracker.get_features(query, 1, None)
    feature_dict = tracker.update_candidates(tracker.generate_candidates(), None)
    scores = tracker.get_scores(track_instances, feature_dict)
    # Raw negative distance (3 px apart), not a (0, 1] similarity.
    assert scores[0, 0] < 0
    np.testing.assert_allclose(scores[0, 0], -3.0, atol=1e-6)


def test_distance_kernel_is_monotone_bounded_and_nan_preserving():
    """The three properties the blend relies on."""
    from sleap_nn.tracking.tracker import distance_to_similarity

    d = np.sort(np.random.default_rng(0).uniform(0, 500, 200))
    sim = distance_to_similarity(-d, 40.0)
    # Monotone DECREASING in distance -> the geometric ordering is preserved
    # exactly, so the kernel changes the scale and nothing else.
    assert np.all(np.diff(sim) <= 0)
    assert sim.min() >= 0.0 and sim.max() <= 1.0
    np.testing.assert_allclose(distance_to_similarity(np.array([0.0]), 40.0), [1.0])
    assert np.isnan(distance_to_similarity(np.array([np.nan]), 40.0))[0]
    # No overflow warning at absurd distances.
    assert distance_to_similarity(np.array([-1e12]), 1.0)[0] == 0.0


def test_the_blend_is_live_on_a_distance_score():
    """The point of the kernel: the weight can now change an assignment.

    At pixel magnitudes it could not -- `(1-w)*(-20) + w*0.95` is dominated by the
    geometric term for every w, so the weight was silently inert. With ambiguous
    geometry (20 px vs 30 px) and appearance strongly preferring the farther
    candidate, weight 0.3 flips the choice.
    """
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        window_size=3,
        features="centroids",
        scoring_method="euclidean_dist",
        appearance_weight=0.3,
        euclidean_scale=50.0,
    )
    # Two tracks 10 px apart; the query sits nearer track 0 but LOOKS like track 1.
    tracker.track([_instance(0, 0, [1, 0]), _instance(50, 0, [0, 1])], frame_idx=0)
    tracked = tracker.track([_instance(20, 0, [0, 1])], frame_idx=1)
    assert [i.track.name for i in tracked] == ["track_1"]

    # Geometry alone would have taken the nearer one.
    geo = Tracker.from_config(
        candidates_method="fixed_window",
        window_size=3,
        features="centroids",
        scoring_method="euclidean_dist",
    )
    geo.track([_instance(0, 0, [1, 0]), _instance(50, 0, [0, 1])], frame_idx=0)
    assert [
        i.track.name for i in geo.track([_instance(20, 0, [0, 1])], frame_idx=1)
    ] == ["track_0"]


# ─────────────────────────────────────────────────────────────────────────
# The appearance cue reduces over the SAME candidate set as geometry
# ─────────────────────────────────────────────────────────────────────────
def test_appearance_respects_min_match_points():
    """A candidate the GEOMETRIC loop drops for `min_match_points` must not
    contribute to the appearance matrix either, or the two matrices reduce over
    different candidates and are not comparable before blending."""
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        features="keypoints",
        scoring_method="oks",
        appearance_weight=0.5,
        min_match_points=1,
    )
    # Frame 0: one full pose (2 valid points) and one with a single visible node.
    weak = sio.PredictedInstance.from_numpy(
        np.array([[100.0, 100.0], [np.nan, np.nan]]),
        skeleton=_SKEL,
        score=0.9,
        point_scores=np.ones(2),
    )
    weak.identity_embedding = sio.Embedding(np.array([0.0, 1.0], np.float32))
    tracker.track([_instance(10, 10, [1, 0]), weak], frame_idx=0)

    query = [_instance(11, 10, [1, 0])]
    track_instances = tracker.get_features(query, 1, None)
    feature_dict = tracker.update_candidates(tracker.generate_candidates(), None)
    appearance = tracker._appearance_scores(track_instances, feature_dict, np.nanmean)

    # Frame 0 seeded two tracks: track 0 from the full pose, track 1 from the weak
    # one. The geometric loop keeps track 0's candidate (2 valid points > 1) and
    # drops track 1's (1 valid point, not > 1), so the appearance matrix must do
    # the same: finite for track 0, NaN — "no evidence" — for track 1. Iterating
    # every candidate instead gave track 1 a real cosine, so the two matrices
    # reduced over different candidate sets.
    tracks = list(tracker.candidate.current_tracks)
    assert len(tracks) == 2, tracks
    assert appearance.shape == (1, 2)
    assert np.isfinite(appearance[0, 0]), "the surviving candidate lost its score"
    assert np.isnan(
        appearance[0, 1]
    ), "the min_match_points-filtered candidate still scored"

    # ...and the geometric matrix agrees, which is the property the blend needs.
    geometric = tracker.get_scores(track_instances, feature_dict)
    assert np.isfinite(geometric[0, 0]) and np.isnan(geometric[0, 1])


def test_appearance_scores_match_the_per_pair_reference():
    """The vectorized matmul must reproduce the per-pair `compute_cosine_sim` loop.

    (Rewritten for speed -- ~90x -- so equivalence is the thing to pin.)
    """
    from sleap_nn.tracking.utils import compute_cosine_sim, get_embedding

    rng = np.random.default_rng(0)
    tracker = Tracker.from_config(
        candidates_method="local_queues",
        window_size=3,
        features="keypoints",
        scoring_method="oks",
        appearance_weight=0.5,
    )
    for frame_idx in range(3):
        tracker.track(
            [
                _instance(10 + frame_idx, 10, rng.normal(size=8)),
                _instance(80 + frame_idx, 80, rng.normal(size=8)),
            ],
            frame_idx=frame_idx,
        )
    query = [
        _instance(13, 10, rng.normal(size=8)),
        _instance(83, 80, rng.normal(size=8)),
        _instance(50, 50),  # no embedding -> a whole NaN row
    ]
    track_instances = tracker.get_features(query, 3, None)
    feature_dict = tracker.update_candidates(tracker.generate_candidates(), None)

    got = tracker._appearance_scores(track_instances, feature_dict, np.nanmean)

    sources = tracker._source_detections(track_instances)
    expected = np.full(got.shape, np.nan)
    for f_idx, source in enumerate(sources):
        q = get_embedding(source)
        if q is None:
            continue
        for t_idx, tid in enumerate(tracker.candidate.current_tracks):
            sims = []
            for cand in feature_dict[tid]:
                g = get_embedding(cand.src_predicted_instance)
                if g is None:
                    continue
                sim = compute_cosine_sim(q, g)
                if not np.isnan(sim):
                    sims.append(sim)
            if sims:
                expected[f_idx][t_idx] = np.nanmean(sims)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
