"""Tests for embedding (appearance / re-ID) tracking in the core ``Tracker``.

Covers the ``"embeddings"`` feature extractor (:func:`get_embedding`), the hardened
``"cosine_sim"`` scoring (:func:`compute_cosine_sim`), and end-to-end track assignment
via ``Tracker`` with ``features="embeddings"`` / ``scoring_method="cosine_sim"`` on
both candidate makers and on both pose + mask carriers. The headline property: the
track follows the *appearance vector*, not position — so identity survives a swap of
the detections' spatial positions.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.tracking.tracker import Tracker
from sleap_nn.tracking.utils import compute_cosine_sim, get_embedding

# ── compute_cosine_sim — hardening ──────────────────────────────────────────────


def test_cosine_sim_basic_values():
    assert compute_cosine_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert compute_cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert compute_cosine_sim(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0
    # Unnormalized vectors: cosine is scale-invariant.
    assert compute_cosine_sim(np.array([3.0, 0.0]), np.array([7.0, 0.0])) == 1.0


@pytest.mark.parametrize(
    "a,b",
    [
        (None, np.array([1.0, 0.0])),  # missing feature (no identity embedding)
        (np.array([1.0, 0.0]), None),
        (np.zeros(3), np.ones(3)),  # zero-norm
        (np.ones(3), np.ones(4)),  # shape mismatch
        (np.array([np.nan, 1.0]), np.array([1.0, 0.0])),  # non-finite
        (np.array([], dtype=float), np.array([], dtype=float)),  # empty
    ],
)
def test_cosine_sim_degenerate_returns_nan(a, b):
    assert np.isnan(compute_cosine_sim(a, b))


# ── get_embedding — the "embeddings" feature extractor ───────────────────────────


def test_get_embedding_reads_reid_vector():
    skel = sio.Skeleton(["a", "b"])
    inst = sio.PredictedInstance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0]]), skeleton=skel, score=0.9
    )
    inst.identity_embedding = sio.Embedding(np.arange(4, dtype=np.float32))
    vec = get_embedding(inst)
    assert isinstance(vec, np.ndarray)
    np.testing.assert_array_equal(vec, np.arange(4, dtype=np.float32))


def test_get_embedding_missing_returns_none():
    skel = sio.Skeleton(["a", "b"])
    inst = sio.PredictedInstance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0]]), skeleton=skel, score=0.9
    )
    assert get_embedding(inst) is None


def test_get_embedding_passthrough_ndarray():
    arr = np.array([1.0, 2.0, 3.0])
    assert get_embedding(arr) is arr


def test_get_embedding_on_mask():
    yy, xx = np.ogrid[:32, :32]
    disk = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 36
    m = sio.PredictedSegmentationMask.from_numpy(disk, score=0.9)
    m.identity_embedding = sio.Embedding(np.ones(5, dtype=np.float32))
    np.testing.assert_array_equal(get_embedding(m), np.ones(5, dtype=np.float32))


# ── Tracker end-to-end: appearance follows identity across position swaps ─────────


def _pose(x, vec, skel):
    inst = sio.PredictedInstance.from_numpy(
        np.array([[x, 0.0], [x + 1, 0.0]]), skeleton=skel, score=0.9
    )
    inst.identity_embedding = sio.Embedding(np.asarray(vec, np.float32))
    return inst


@pytest.mark.parametrize("candidates_method", ["fixed_window", "local_queues"])
def test_tracker_embeddings_follow_appearance_across_swaps(candidates_method):
    """Two animals with distinct constant embeddings; their x-positions swap every
    frame. The assigned track must ride the embedding (appearance), not position."""
    skel = sio.Skeleton(["a", "b"])
    va, vb = np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])
    tracker = Tracker.from_config(
        features="embeddings",
        scoring_method="cosine_sim",
        candidates_method=candidates_method,
    )
    per_track_vecs: dict = {}
    for fi in range(4):
        ax, bx = (10, 50) if fi % 2 == 0 else (50, 10)
        out = tracker.track([_pose(ax, va, skel), _pose(bx, vb, skel)], fi)
        for inst in out:
            per_track_vecs.setdefault(inst.track.name, set()).add(
                tuple(np.round(inst.identity_embedding.vector, 3))
            )
    # Exactly two tracks, and each rode a single (constant) embedding.
    assert len(per_track_vecs) == 2
    assert all(len(v) == 1 for v in per_track_vecs.values())


def test_tracker_embeddings_mask_carrier():
    """``features="embeddings"`` works on ``PredictedSegmentationMask`` carriers."""
    yy, xx = np.ogrid[:64, :64]
    va, vb = np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])

    def mk(cy, vec):
        disk = ((yy - cy) ** 2 + (xx - 30) ** 2) <= 64
        m = sio.PredictedSegmentationMask.from_numpy(disk, score=0.9)
        m.identity_embedding = sio.Embedding(np.asarray(vec, np.float32))
        return m

    tracker = Tracker.from_config(
        features="embeddings",
        scoring_method="cosine_sim",
        candidates_method="local_queues",
    )
    per_track: dict = {}
    for fi in range(4):
        cy_a, cy_b = (20, 44) if fi % 2 == 0 else (44, 20)
        out = tracker.track([mk(cy_a, va), mk(cy_b, vb)], fi)
        for m in out:
            per_track.setdefault(m.track.name, set()).add(
                tuple(np.round(m.identity_embedding.vector, 3))
            )
    assert len(per_track) == 2
    assert all(len(v) == 1 for v in per_track.values())


def test_tracker_embeddings_invalid_feature_name_raises():
    skel = sio.Skeleton(["a", "b"])
    tracker = Tracker.from_config(features="not_a_feature")
    with pytest.raises(ValueError, match="Invalid `features`"):
        tracker.track([_pose(0, [1, 0], skel)], 0)


# ── degenerate-input robustness (review findings) ────────────────────────────────


def _pose_no_emb(x, skel):
    return sio.PredictedInstance.from_numpy(
        np.array([[x, 0.0], [x + 1, 0.0]]), skeleton=skel, score=0.9
    )


@pytest.mark.parametrize("scoring", ["cosine_sim", "euclidean_dist"])
def test_tracker_missing_embedding_does_not_crash(scoring):
    """A detection missing its embedding must not crash the run (NaN -> no match),
    for BOTH vector metrics (euclidean_dist was unhardened — review finding [1])."""
    skel = sio.Skeleton(["a", "b"])
    va, vb = np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])
    tracker = Tracker.from_config(
        features="embeddings", scoring_method=scoring, candidates_method="local_queues"
    )
    tracker.track([_pose(10, va, skel), _pose(50, vb, skel)], 0)
    # Frame 1: one embedded pose + one with NO embedding.
    out = tracker.track([_pose(10, va, skel), _pose_no_emb(50, skel)], 1)
    assert len(out) == 2  # neither is dropped


def test_tracker_missing_embedding_spawns_fresh_track():
    """The embedding-less detection spawns a NEW track (admission gate) rather than
    stealing an existing identity with a -inf score (review finding [2])."""
    skel = sio.Skeleton(["a", "b"])
    va, vb = np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])
    tracker = Tracker.from_config(
        features="embeddings",
        scoring_method="cosine_sim",
        candidates_method="local_queues",
    )
    tracker.track([_pose(10, va, skel), _pose(50, vb, skel)], 0)  # tracks 0, 1
    emb_pose = _pose(10, va, skel)  # matches track 0
    no_emb = _pose_no_emb(50, skel)  # no embedding -> must NOT steal track 1
    out = tracker.track([emb_pose, no_emb], 1)
    by_name = {id(i): i.track.name for i in out}
    emb_name = by_name[id(emb_pose)]
    no_emb_name = by_name[id(no_emb)]
    assert emb_name == "track_0"  # appearance match preserved
    assert no_emb_name not in ("track_0", "track_1")  # a fresh track, not stolen
    # A spawned track gets a finite score (1.0), never -inf from a forced match.
    assert np.isfinite(no_emb.tracking_score)
