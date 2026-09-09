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
