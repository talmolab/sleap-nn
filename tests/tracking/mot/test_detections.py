"""Tests for `sleap_nn.tracking.mot.detections` (FrameDetections / extract_detections)."""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.tracking.mot.detections import FrameDetections, extract_detections

_SKEL = sio.Skeleton(nodes=["a", "b", "c", "d"], name="s")
_SKEL1 = sio.Skeleton(nodes=["c"], name="s1")


def _pred(points, point_scores=None, score=0.5, skeleton=_SKEL):
    points = np.asarray(points, dtype=float)
    if point_scores is None:
        point_scores = np.ones(len(points))
    return sio.PredictedInstance.from_numpy(
        points,
        skeleton=skeleton,
        score=score,
        point_scores=np.asarray(point_scores, dtype=float),
    )


def _user(points, skeleton=_SKEL):
    return sio.Instance.from_numpy(np.asarray(points, dtype=float), skeleton=skeleton)


def _mask(y0, y1, x0, x1, shape=(40, 50), score=0.8, **kwargs):
    data = np.zeros(shape, dtype=bool)
    data[y0:y1, x0:x1] = True
    return sio.PredictedSegmentationMask.from_numpy(data, score=score, **kwargs)


def test_pose_boxes_padding_centroid_and_nan_nodes():
    """Raw box = visible-node extent; padded by p*max(w,h); centroid = box center."""
    pts = [[10, 20], [30, 25], [np.nan, np.nan], [20, 40]]
    dets = extract_detections([_pred(pts)], frame_idx=7, box_padding=0.1)

    assert len(dets) == 1 and dets.frame_idx == 7
    np.testing.assert_allclose(dets.raw_boxes[0], [10, 20, 30, 40])
    # w=20, h=20 -> pad 2 on every side.
    np.testing.assert_allclose(dets.boxes[0], [8, 18, 32, 42])
    # Box center, NOT the node median (which would be (20, 25)).
    np.testing.assert_allclose(dets.centroids[0], [20, 30])
    assert dets.keypoints.shape == (1, 4, 2)
    assert np.isnan(dets.keypoints[0, 2]).all()
    assert dets.n_visible.tolist() == [3]
    assert dets.n_nodes == 4
    assert dets.valid.tolist() == [True]
    assert dets.boxes.dtype == np.float64 and dets.scores.dtype == np.float64


def test_all_nan_pose_is_invalid_with_nan_geometry():
    """A pose with no visible node is invalid and its geometry rows are NaN."""
    good = _pred([[0, 0], [10, 10], [5, 5], [2, 8]])
    bad = _pred(np.full((4, 2), np.nan))
    dets = extract_detections([good, bad], frame_idx=0)

    assert dets.valid.tolist() == [True, False]
    assert dets.n_visible.tolist() == [4, 0]
    for arr in (dets.boxes, dets.raw_boxes, dets.centroids):
        assert np.isfinite(arr[0]).all()
        assert np.isnan(arr[1]).all()
    assert dets.sources == [good, bad]


def test_single_node_uses_pseudo_box():
    """One visible node -> square of side pseudo_box_size on the centroid (raw too)."""
    pts = [[np.nan, np.nan], [np.nan, np.nan], [50, 60], [np.nan, np.nan]]
    dets = extract_detections([_pred(pts)], 0, pseudo_box_size=20.0)

    expected = [40, 50, 60, 70]
    np.testing.assert_allclose(dets.raw_boxes[0], expected)
    np.testing.assert_allclose(dets.boxes[0], expected)
    np.testing.assert_allclose(dets.centroids[0], [50, 60])
    assert dets.valid[0] and dets.n_visible[0] == 1


def test_single_node_skeleton_and_one_px_floor_without_pseudo_box():
    """Standalone callers (pseudo_box_size=None) get a 1-px floor, then padding."""
    dets = extract_detections([_pred([[5, 6]], skeleton=_SKEL1)], 0, box_padding=0.1)

    assert dets.n_nodes == 1
    np.testing.assert_allclose(dets.raw_boxes[0], [4.5, 5.5, 5.5, 6.5])
    np.testing.assert_allclose(dets.boxes[0], [4.4, 5.4, 5.6, 6.6])
    np.testing.assert_allclose(dets.centroids[0], [5, 6])


def test_axis_collinear_nodes_are_degenerate():
    """Nodes on one horizontal line (h < 1 px) also get the pseudo box."""
    pts = [[0, 10], [40, 10], [20, 10.5], [np.nan, np.nan]]
    dets = extract_detections([_pred(pts)], 0, pseudo_box_size=30.0)

    np.testing.assert_allclose(dets.centroids[0], [20, 10.25])
    np.testing.assert_allclose(dets.raw_boxes[0], [5, -4.75, 35, 25.25])
    np.testing.assert_allclose(dets.boxes[0], dets.raw_boxes[0])


def test_user_instance_scores_one():
    """User instances (no `score` attribute) score 1.0 under every score mode."""
    inst = _user([[0, 0], [10, 10], [5, 5], [2, 8]])
    for mode in ("auto", "instance", "none"):
        dets = extract_detections([inst], 0, score_mode=mode)
        assert dets.scores.tolist() == [1.0]


def test_predicted_auto_score_is_mean_of_visible_point_scores():
    """`auto`: nanmean of the point scores of VISIBLE nodes only."""
    pts = [[0, 0], [10, 10], [np.nan, np.nan], [5, 5]]
    inst = _pred(pts, point_scores=[0.2, 0.4, 0.99, 0.9], score=0.1)
    dets = extract_detections([inst], 0)
    np.testing.assert_allclose(dets.scores, [0.5])

    # No finite visible point score -> instance score.
    inst2 = _pred(pts, point_scores=[np.nan, np.nan, 0.9, np.nan], score=0.3)
    np.testing.assert_allclose(extract_detections([inst2], 0).scores, [0.3])

    # Non-finite instance score -> 1.0; out-of-range values are clipped.
    inst3 = _pred(pts, point_scores=[np.nan] * 4, score=np.nan)
    inst4 = _pred(pts, point_scores=[1.5, 1.5, 1.5, 1.5], score=0.3)
    dets = extract_detections([inst3, inst4], 0)
    np.testing.assert_allclose(dets.scores, [1.0, 1.0])


def test_instance_and_none_score_modes():
    """`instance` reads obj.score (clipped); `none` is 1.0 everywhere."""
    pts = [[0, 0], [10, 10], [3, 3], [5, 5]]
    a = _pred(pts, point_scores=[0.2] * 4, score=0.7)
    b = _pred(pts, point_scores=[0.2] * 4, score=-0.5)
    np.testing.assert_allclose(
        extract_detections([a, b], 0, score_mode="instance").scores, [0.7, 0.0]
    )
    np.testing.assert_allclose(
        extract_detections([a, b], 0, score_mode="none").scores, [1.0, 1.0]
    )


def test_bad_carrier_and_score_mode_raise():
    """Unknown options are rejected up front."""
    with pytest.raises(ValueError, match="carrier"):
        extract_detections([], 0, carrier="boxes")
    with pytest.raises(ValueError, match="score_mode"):
        extract_detections([], 0, score_mode="max")


def test_mask_extraction():
    """Masks: raw box from mask.bbox (image space), empty masks invalid, score."""
    m1 = _mask(5, 10, 3, 12, score=0.6)
    m2 = _mask(0, 0, 0, 0)  # empty
    # Stride-2 mask (scale 0.5) offset in the image: bbox is in image space.
    m3 = _mask(2, 6, 4, 8, score=0.9, scale=(0.5, 0.5), offset=(100.0, 50.0))
    dets = extract_detections([m1, m2, m3], 3, carrier="mask", box_padding=0.0)

    assert dets.keypoints is None and dets.n_nodes == 1
    assert dets.valid.tolist() == [True, False, True]
    assert dets.n_visible.tolist() == [1, 0, 1]
    x, y, w, h = m1.bbox
    np.testing.assert_allclose(dets.raw_boxes[0], [x, y, x + w, y + h])
    np.testing.assert_allclose(dets.raw_boxes[0], [3, 5, 12, 10])
    x, y, w, h = m3.bbox
    np.testing.assert_allclose(dets.raw_boxes[2], [x, y, x + w, y + h])
    np.testing.assert_allclose(dets.centroids[2], [x + w / 2, y + h / 2])
    assert np.isnan(dets.raw_boxes[1]).all() and np.isnan(dets.centroids[1]).all()
    np.testing.assert_allclose(dets.scores[[0, 2]], [0.6, 0.9])

    padded = extract_detections([m1], 0, carrier="mask", box_padding=0.1)
    np.testing.assert_allclose(padded.boxes[0], [3 - 0.9, 5 - 0.9, 12.9, 10.9])


def test_user_mask_scores_one():
    """A user mask has no score attribute -> 1.0."""
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 2:5] = True
    m = sio.UserSegmentationMask.from_numpy(data)
    dets = extract_detections([m], 0, carrier="mask")
    assert dets.scores.tolist() == [1.0]


def test_embeddings_missing_mixed_dims_and_renormalization():
    """Vectors are re-normalized; None, zero, non-finite, wrong-dim -> missing."""
    pts = [[0, 0], [10, 10], [3, 3], [5, 5]]
    objs = [_pred(pts) for _ in range(6)]
    objs[0].identity_embedding = sio.Embedding(np.array([3.0, 4.0, 0.0]))
    # objs[1]: no embedding.
    objs[2].identity_embedding = sio.Embedding(np.zeros(3))
    objs[3].identity_embedding = sio.Embedding(np.array([1.0, 2.0]))  # wrong dim
    objs[4].identity_embedding = sio.Embedding(np.array([0.0, 0.0, 2.0]))
    objs[5].identity_embedding = sio.Embedding(np.array([np.nan, 1.0, 0.0]))

    dets = extract_detections(objs, 0)
    assert dets.has_embedding.tolist() == [True, False, False, False, True, False]
    assert dets.embeddings.dtype == np.float32
    assert dets.embeddings.shape == (6, 3)
    np.testing.assert_allclose(dets.embeddings[0], [0.6, 0.8, 0.0], rtol=1e-6)
    np.testing.assert_allclose(dets.embeddings[4], [0.0, 0.0, 1.0])
    assert not dets.embeddings[[1, 2, 3, 5]].any()

    # An explicit dimension rejects the first vector's dimension instead.
    dets2 = extract_detections(objs, 0, embedding_dim=2)
    assert dets2.has_embedding.tolist() == [False, False, False, True, False, False]
    np.testing.assert_allclose(dets2.embeddings[3], np.array([1, 2]) / np.sqrt(5))


def test_embeddings_none_when_no_detection_has_one():
    """No vectors at all -> embeddings is None, has_embedding all False."""
    dets = extract_detections([_pred([[0, 0], [1, 1], [2, 2], [3, 3]])], 0)
    assert dets.embeddings is None
    assert dets.has_embedding.tolist() == [False]


def test_mask_embeddings():
    """Mask detections carry vectors through the same slot."""
    m = _mask(5, 10, 3, 12)
    m.identity_embedding = sio.Embedding(np.array([0.0, 2.0]))
    dets = extract_detections([m], 0, carrier="mask")
    np.testing.assert_allclose(dets.embeddings, [[0.0, 1.0]])


def test_empty_frames():
    """Empty input gives correctly shaped zero-length arrays."""
    dets = extract_detections([], 4)
    assert len(dets) == 0 and dets.frame_idx == 4
    assert dets.boxes.shape == (0, 4) and dets.raw_boxes.shape == (0, 4)
    assert dets.centroids.shape == (0, 2)
    assert dets.keypoints.shape == (0, 1, 2)
    assert dets.scores.shape == (0,) and dets.valid.shape == (0,)
    assert dets.embeddings is None and dets.has_embedding.shape == (0,)

    masks = extract_detections([], 4, carrier="mask")
    assert masks.keypoints is None and masks.n_nodes == 1

    empty = FrameDetections.empty(9, n_nodes=5)
    assert len(empty) == 0 and empty.keypoints.shape == (0, 5, 2)
    assert empty.n_nodes == 5 and empty.n_visible.shape == (0,)


def test_subset_index_and_mask():
    """subset() copies rows for index arrays and bool masks."""
    pts = [[0, 0], [10, 10], [3, 3], [5, 5]]
    objs = [_pred(np.asarray(pts) + 100 * i) for i in range(3)]
    objs[1].identity_embedding = sio.Embedding(np.array([1.0, 0.0]))
    dets = extract_detections(objs, 2)

    sub = dets.subset(np.array([2, 0]))
    assert sub.sources == [objs[2], objs[0]]
    np.testing.assert_allclose(sub.boxes, dets.boxes[[2, 0]])
    np.testing.assert_allclose(sub.keypoints, dets.keypoints[[2, 0]])
    assert sub.frame_idx == 2 and sub.n_nodes == 4
    # Neither selected detection has a vector -> None.
    assert sub.embeddings is None and sub.has_embedding.tolist() == [False, False]
    # Copies, not views.
    sub.boxes[0] = -1
    assert dets.boxes[2, 0] != -1

    sub2 = dets.subset(np.array([False, True, True]))
    assert sub2.sources == objs[1:]
    assert sub2.embeddings.shape == (2, 2) and sub2.has_embedding.tolist() == [
        True,
        False,
    ]

    none = dets.subset(np.array([], dtype=int))
    assert len(none) == 0 and none.boxes.shape == (0, 4)
    assert none.keypoints.shape == (0, 4, 2)

    with pytest.raises(ValueError):
        dets.subset(np.array([True, False]))


def test_mixed_skeleton_sizes_are_nan_padded():
    """Poses with different node counts are padded to the longest skeleton."""
    a = _pred([[0, 0], [10, 10], [3, 3], [5, 5]])
    b = _pred([[20, 20]], skeleton=_SKEL1)
    dets = extract_detections([a, b], 0, pseudo_box_size=10.0)
    assert dets.n_nodes == 4 and dets.keypoints.shape == (2, 4, 2)
    np.testing.assert_allclose(dets.keypoints[1, 0], [20, 20])
    assert np.isnan(dets.keypoints[1, 1:]).all()
    np.testing.assert_allclose(dets.raw_boxes[1], [15, 15, 25, 25])


def test_normalize_scores():
    """Divide by the video's normalizer, clip to [0, 1]; no-op without one."""
    from sleap_nn.tracking.mot.detections import normalize_scores

    objs = [_pred([[0, 0], [5, 5]] * 2, point_scores=np.full(4, s)) for s in (0.3, 0.6)]
    dets = extract_detections(objs, 0)
    np.testing.assert_allclose(dets.scores, [0.3, 0.6])
    normalize_scores(dets, 0.5)
    np.testing.assert_allclose(dets.scores, [0.6, 1.0])
    for normalizer in (None, 0.0, -1.0, float("nan")):
        normalize_scores(dets, normalizer)
        np.testing.assert_allclose(dets.scores, [0.6, 1.0])
    empty = FrameDetections.empty(0)
    normalize_scores(empty, 0.5)
    assert empty.scores.shape == (0,)
