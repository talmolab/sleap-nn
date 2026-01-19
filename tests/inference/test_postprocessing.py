"""Tests for sleap_nn.inference.postprocessing module."""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.inference.postprocessing import (
    _compute_iou_one_to_many,
    _compute_oks,
    _instance_bbox,
    _instance_score,
    _nms_greedy_iou,
    _nms_greedy_oks,
    filter_overlapping_instances,
)


class TestComputeIOU:
    """Tests for IOU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IOU = 1.0."""
        box = np.array([0, 0, 10, 10])
        boxes = np.array([[0, 0, 10, 10]])
        ious = _compute_iou_one_to_many(box, boxes)
        assert np.isclose(ious[0], 1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IOU = 0.0."""
        box = np.array([0, 0, 10, 10])
        boxes = np.array([[20, 20, 30, 30]])
        ious = _compute_iou_one_to_many(box, boxes)
        assert np.isclose(ious[0], 0.0)

    def test_partial_overlap(self):
        """Test partial overlap IOU calculation."""
        box = np.array([0, 0, 10, 10])  # area = 100
        boxes = np.array([[5, 0, 15, 10]])  # area = 100, intersection = 50
        # union = 100 + 100 - 50 = 150, IOU = 50/150 = 0.333...
        ious = _compute_iou_one_to_many(box, boxes)
        assert np.isclose(ious[0], 1 / 3, atol=0.01)

    def test_multiple_boxes(self):
        """Test IOU against multiple boxes."""
        box = np.array([0, 0, 10, 10])
        boxes = np.array(
            [
                [0, 0, 10, 10],  # identical, IOU = 1.0
                [20, 20, 30, 30],  # no overlap, IOU = 0.0
                [5, 0, 15, 10],  # partial, IOU = 0.333
            ]
        )
        ious = _compute_iou_one_to_many(box, boxes)
        assert np.isclose(ious[0], 1.0)
        assert np.isclose(ious[1], 0.0)
        assert np.isclose(ious[2], 1 / 3, atol=0.01)

    def test_zero_area_box(self):
        """Zero-area boxes should return IOU = 0."""
        box = np.array([5, 5, 5, 5])  # zero area
        boxes = np.array([[0, 0, 10, 10]])
        ious = _compute_iou_one_to_many(box, boxes)
        assert np.isclose(ious[0], 0.0)


class TestComputeOKS:
    """Tests for OKS computation."""

    def test_identical_points(self):
        """Identical keypoints should have OKS = 1.0."""
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        oks = _compute_oks(points, points)
        assert np.isclose(oks, 1.0)

    def test_distant_points(self):
        """Very distant keypoints should have OKS close to 0."""
        points_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        points_b = np.array([[1000, 1000], [1010, 1000], [1010, 1010], [1000, 1010]])
        oks = _compute_oks(points_a, points_b)
        assert oks < 0.01

    def test_partial_overlap_oks(self):
        """Slightly offset keypoints should have intermediate OKS."""
        points_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        points_b = np.array([[1, 1], [11, 1], [11, 11], [1, 11]])  # offset by (1, 1)
        oks = _compute_oks(points_a, points_b)
        # OKS depends on scale and kappa; with default kappa=0.1, small offsets
        # relative to bbox size give intermediate OKS
        assert 0.0 < oks < 1.0

    def test_with_nan_keypoints(self):
        """OKS should handle NaN keypoints gracefully."""
        points_a = np.array([[0, 0], [np.nan, np.nan], [10, 10]])
        points_b = np.array([[0, 0], [5, 5], [10, 10]])
        oks = _compute_oks(points_a, points_b)
        # Should only use points 0 and 2
        assert 0.0 <= oks <= 1.0

    def test_all_nan_returns_zero(self):
        """All NaN keypoints should return OKS = 0."""
        points_a = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        points_b = np.array([[0, 0], [10, 10]])
        oks = _compute_oks(points_a, points_b)
        assert oks == 0.0

    def test_single_valid_point_returns_zero(self):
        """Single valid point should return OKS = 0 (need at least 2 for bbox)."""
        points_a = np.array([[5, 5], [np.nan, np.nan]])
        points_b = np.array([[5, 5], [10, 10]])
        oks = _compute_oks(points_a, points_b)
        assert oks == 0.0

    def test_zero_area_bbox_returns_zero(self):
        """Points at same location (zero bbox area) should return OKS = 0."""
        points_a = np.array([[5, 5], [5, 5]])  # same point, zero area
        points_b = np.array([[5, 5], [5, 5]])
        oks = _compute_oks(points_a, points_b)
        assert oks == 0.0


class TestNMSGreedyIOU:
    """Tests for greedy NMS with IOU."""

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _nms_greedy_iou(np.array([]).reshape(0, 4), np.array([]), 0.5)
        assert result == []

    def test_single_box(self):
        """Single box should always be kept."""
        bboxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        result = _nms_greedy_iou(bboxes, scores, 0.5)
        assert result == [0]

    def test_no_overlap_keeps_all(self):
        """Non-overlapping boxes should all be kept."""
        bboxes = np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        result = _nms_greedy_iou(bboxes, scores, 0.5)
        assert set(result) == {0, 1, 2}

    def test_overlapping_removes_lower_score(self):
        """Overlapping boxes should keep only higher-scoring one."""
        bboxes = np.array(
            [
                [0, 0, 10, 10],
                [2, 2, 12, 12],  # high overlap with first
            ]
        )
        scores = np.array([0.9, 0.7])
        result = _nms_greedy_iou(bboxes, scores, 0.3)
        assert result == [0]  # only highest score kept

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        bboxes = np.array(
            [
                [0, 0, 10, 10],
                [5, 0, 15, 10],  # IOU = 0.333 with first
            ]
        )
        scores = np.array([0.9, 0.8])

        # threshold = 0.3, IOU = 0.333 > 0.3, should remove
        result = _nms_greedy_iou(bboxes, scores, 0.3)
        assert result == [0]

        # threshold = 0.4, IOU = 0.333 <= 0.4, should keep both
        result = _nms_greedy_iou(bboxes, scores, 0.4)
        assert set(result) == {0, 1}

    def test_order_by_score(self):
        """Result should be ordered by decreasing score."""
        bboxes = np.array(
            [
                [0, 0, 10, 10],
                [100, 100, 110, 110],
                [200, 200, 210, 210],
            ]
        )
        scores = np.array([0.5, 0.9, 0.7])  # not in order
        result = _nms_greedy_iou(bboxes, scores, 0.5)
        # Should be ordered by score: 1 (0.9), 2 (0.7), 0 (0.5)
        assert result == [1, 2, 0]


class TestNMSGreedyOKS:
    """Tests for greedy NMS with OKS."""

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _nms_greedy_oks([], np.array([]), 0.5)
        assert result == []

    def test_single_instance(self):
        """Single instance should always be kept."""
        points_list = [np.array([[0, 0], [10, 10]])]
        scores = np.array([0.9])
        result = _nms_greedy_oks(points_list, scores, 0.5)
        assert result == [0]

    def test_identical_instances_removes_lower_score(self):
        """Identical instances should keep only highest-scoring."""
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        points_list = [points.copy(), points.copy()]
        scores = np.array([0.9, 0.7])
        result = _nms_greedy_oks(points_list, scores, 0.5)
        assert result == [0]  # only highest score kept

    def test_distant_instances_keeps_all(self):
        """Distant instances should all be kept."""
        points_list = [
            np.array([[0, 0], [10, 10]]),
            np.array([[100, 100], [110, 110]]),
        ]
        scores = np.array([0.9, 0.8])
        result = _nms_greedy_oks(points_list, scores, 0.5)
        assert set(result) == {0, 1}


class TestInstanceBbox:
    """Tests for bounding box computation from instances."""

    def test_valid_keypoints(self):
        """Test bbox from valid keypoints."""
        skeleton = sio.Skeleton(nodes=["a", "b", "c"])
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 5], [5, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        bbox = _instance_bbox(inst)
        np.testing.assert_array_equal(bbox, [0, 0, 10, 10])

    def test_with_nan_keypoints(self):
        """Test bbox ignores NaN keypoints."""
        skeleton = sio.Skeleton(nodes=["a", "b", "c"])
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [np.nan, np.nan], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        bbox = _instance_bbox(inst)
        np.testing.assert_array_equal(bbox, [0, 0, 10, 10])

    def test_all_nan_keypoints(self):
        """All NaN keypoints should return zero bbox."""
        skeleton = sio.Skeleton(nodes=["a", "b"])
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            skeleton=skeleton,
            score=0.9,
        )
        bbox = _instance_bbox(inst)
        np.testing.assert_array_equal(bbox, [0, 0, 0, 0])


class TestFilterOverlappingInstances:
    """Integration tests for filter_overlapping_instances."""

    @pytest.fixture
    def skeleton(self):
        """Create a simple skeleton for testing."""
        return sio.Skeleton(nodes=["head", "tail"])

    @pytest.fixture
    def video(self):
        """Create a dummy video for testing."""
        return sio.Video(filename="test.mp4")

    def test_empty_labels(self, skeleton, video):
        """Empty labels should be unchanged."""
        labels = sio.Labels(videos=[video], skeletons=[skeleton])
        result = filter_overlapping_instances(labels, threshold=0.5)
        assert len(result.labeled_frames) == 0

    def test_single_instance_unchanged(self, skeleton, video):
        """Single instance per frame should be unchanged."""
        inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        result = filter_overlapping_instances(labels, threshold=0.5)
        assert len(result.labeled_frames[0].instances) == 1

    def test_removes_overlapping_iou(self, skeleton, video):
        """Overlapping instances should be filtered with IOU method."""
        inst1 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        inst2 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[2, 2], [12, 12]]),  # overlaps with inst1
            skeleton=skeleton,
            score=0.7,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        result = filter_overlapping_instances(labels, threshold=0.3, method="iou")
        assert len(result.labeled_frames[0].instances) == 1
        assert result.labeled_frames[0].instances[0].score == 0.9

    def test_removes_overlapping_oks(self, skeleton, video):
        """Overlapping instances should be filtered with OKS method."""
        # Use identical instances to ensure OKS = 1.0 (definitely above threshold)
        inst1 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [100, 100]]),
            skeleton=skeleton,
            score=0.9,
        )
        inst2 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [100, 100]]),  # identical to inst1
            skeleton=skeleton,
            score=0.7,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        result = filter_overlapping_instances(labels, threshold=0.5, method="oks")
        assert len(result.labeled_frames[0].instances) == 1
        assert result.labeled_frames[0].instances[0].score == 0.9

    def test_keeps_non_overlapping(self, skeleton, video):
        """Non-overlapping instances should all be kept."""
        inst1 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        inst2 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[50, 50], [60, 60]]),  # no overlap
            skeleton=skeleton,
            score=0.7,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        result = filter_overlapping_instances(labels, threshold=0.3)
        assert len(result.labeled_frames[0].instances) == 2

    def test_preserves_non_predicted_instances(self, skeleton, video):
        """Non-predicted instances (ground truth) should be preserved."""
        pred_inst = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        gt_inst = sio.Instance.from_numpy(
            points_data=np.array([[2, 2], [12, 12]]),  # overlaps but is GT
            skeleton=skeleton,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst, gt_inst])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        result = filter_overlapping_instances(labels, threshold=0.3)
        # Both should be kept: GT is not filtered, and only 1 predicted
        assert len(result.labeled_frames[0].instances) == 2

    def test_multiple_frames(self, skeleton, video):
        """Test filtering across multiple frames."""
        # Frame 0: two overlapping instances
        inst1 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        inst2 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[2, 2], [12, 12]]),
            skeleton=skeleton,
            score=0.7,
        )
        lf0 = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])

        # Frame 1: two non-overlapping instances
        inst3 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.8,
        )
        inst4 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[50, 50], [60, 60]]),
            skeleton=skeleton,
            score=0.6,
        )
        lf1 = sio.LabeledFrame(video=video, frame_idx=1, instances=[inst3, inst4])

        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf0, lf1],
        )

        result = filter_overlapping_instances(labels, threshold=0.3)
        assert len(result.labeled_frames[0].instances) == 1  # filtered
        assert len(result.labeled_frames[1].instances) == 2  # kept both

    def test_invalid_method_raises(self, skeleton, video):
        """Invalid method should raise ValueError."""
        # Need instances to trigger the method check
        inst1 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[0, 0], [10, 10]]),
            skeleton=skeleton,
            score=0.9,
        )
        inst2 = sio.PredictedInstance.from_numpy(
            points_data=np.array([[5, 5], [15, 15]]),
            skeleton=skeleton,
            score=0.7,
        )
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
        labels = sio.Labels(
            videos=[video],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )
        with pytest.raises(ValueError, match="Unknown method"):
            filter_overlapping_instances(labels, method="invalid")
