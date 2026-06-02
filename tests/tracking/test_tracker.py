import pytest
import numpy as np
import sleap_io as sio
from sleap_nn.predict import run_inference
from sleap_nn.tracking.tracker import (
    Tracker,
    FlowShiftTracker,
    KalmanShiftTracker,
    run_tracker,
)
from sleap_nn.tracking.track_instance import (
    TrackedInstanceFeature,
    TrackInstanceLocalQueue,
    TrackInstances,
)
from sleap_nn.tracking.utils import (
    hungarian_matching,
    nms_fast,
    nms_instances,
    cull_instances,
    cull_frame_instances,
)
import math
from loguru import logger
from _pytest.logging import LogCaptureFixture
import torch


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def get_pred_instances(
    minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
):
    """Get `sio.PredictedInstance` objects from Predictor class."""
    result_labels = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.0,
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    pred_instances = []
    imgs = []
    for lf in result_labels:
        pred_instances.extend(lf.instances)
        imgs.append(lf.image)
    return pred_instances, imgs


def centered_pair_predictions(
    minimal_instance_centered_instance_ckpt,
    minmal_instance_centroid_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test centered pair predictions."""
    result_labels = run_inference(
        model_paths=[
            minmal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.0,
        integral_refinement="integral",
        frames=[x for x in range(0, 65)],
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    return result_labels


def test_hungarian_matching_edge_cases():
    """Test hungarian_matching with inf/nan cost matrices."""
    # All-inf matrix (the reported bug in #491)
    cost = np.full((2, 2), np.inf)
    row_ids, col_ids = hungarian_matching(cost)
    assert len(row_ids) == 2
    assert len(col_ids) == 2

    # All-NaN matrix
    cost = np.full((3, 3), np.nan)
    row_ids, col_ids = hungarian_matching(cost)
    assert len(row_ids) == 3

    # Mixed finite and inf
    cost = np.array([[1.0, np.inf], [np.inf, 2.0]])
    row_ids, col_ids = hungarian_matching(cost)
    assert set(zip(row_ids, col_ids)) == {(0, 0), (1, 1)}

    # Normal case still works
    cost = np.array([[1.0, 3.0], [4.0, 2.0]])
    row_ids, col_ids = hungarian_matching(cost)
    assert set(zip(row_ids, col_ids)) == {(0, 0), (1, 1)}


def test_cull_instances(
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test cull instances."""
    preds = centered_pair_predictions(
        minimal_instance_centered_instance_ckpt,
        minimal_instance_centroid_ckpt,
        centered_instance_video,
        tmp_path,
    )
    frames = preds.labeled_frames[52:60]
    cull_instances(frames=frames, instance_count=2)

    for frame in frames:
        assert len(frame.instances) == 2

    frames = preds.labeled_frames[:5]
    cull_instances(frames=frames, instance_count=1)

    for frame in frames:
        assert len(frame.instances) == 1


def test_nms():
    """Test nms."""
    boxes = np.array(
        [[10, 10, 20, 20], [10, 10, 15, 15], [30, 30, 40, 40], [32, 32, 42, 42]]
    )
    scores = np.array([1, 0.3, 1, 0.5])

    picks = nms_fast(boxes, scores, iou_threshold=0.5)
    assert sorted(picks) == [0, 2]


def test_nms_with_target():
    """Test nms with target."""
    boxes = np.array(
        [[10, 10, 20, 20], [10, 10, 15, 15], [30, 30, 40, 40], [32, 32, 42, 42]]
    )
    # Box 1 is suppressed and has lowest score
    scores = np.array([1, 0.3, 1, 0.5])
    picks = nms_fast(boxes, scores, iou_threshold=0.5, target_count=3)
    assert sorted(picks) == [0, 2, 3]

    # Box 3 is suppressed and has lowest score
    scores = np.array([1, 0.5, 1, 0.3])
    picks = nms_fast(boxes, scores, iou_threshold=0.5, target_count=3)
    assert sorted(picks) == [0, 1, 2]


def test_nms_instances_to_remove():
    """Test nms instances to remove."""
    skeleton = sio.Skeleton()
    skeleton.add_nodes(("a", "b"))

    instances = []

    inst = sio.PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skeleton
    )
    inst.score = 1
    instances.append(inst)

    inst = sio.PredictedInstance.from_numpy(
        np.array([[10, 10], [15, 15]]), skeleton=skeleton
    )
    inst.score = 0.3
    instances.append(inst)

    inst = sio.PredictedInstance.from_numpy(
        np.array([[30, 30], [40, 40]]), skeleton=skeleton
    )
    inst.score = 1
    instances.append(inst)

    inst = sio.PredictedInstance.from_numpy(
        np.array([[32, 32], [42, 42]]), skeleton=skeleton
    )
    inst.score = 0.5
    instances.append(inst)

    to_keep, to_remove = nms_instances(instances, iou_threshold=0.5, target_count=3)

    assert len(to_remove) == 1
    assert to_remove[0].same_pose_as(instances[1])


def test_tracker(
    caplog, minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
):
    """Test `Tracker` module."""
    # Test for the first two instances
    # no new tracks should be created
    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(
        min_new_track_points=3
    )  # num visible nodes is less than the threshold
    assert isinstance(tracker, Tracker)
    assert not isinstance(tracker, FlowShiftTracker)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)
    for t in tracked_instances:
        assert t.track is None
    assert len(tracker.candidate.current_tracks) == 0

    # Test Fixed-window method
    # pose as feature, oks scoring method, avg score reduction, hungarian matching
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(candidates_method="fixed_window")
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert (
        tracked_instances[0].track.name == "track_0"
        and tracked_instances[1].track.name == "track_1"
    )
    assert len(tracker.candidate.tracker_queue) == 1
    assert tracker.candidate.current_tracks == [0, 1]
    assert tracker.candidate.tracker_queue[0].track_ids == [0, 1]

    # Test local queue method
    # pose as feature, oks scoring method, max score reduction, hungarian matching
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(candidates_method="local_queues")
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidate.tracker_queue) == 2
    assert tracker.candidate.current_tracks == [0, 1]
    assert (
        tracked_instances[0].track.name == "track_0"
        and tracked_instances[1].track.name == "track_1"
    )

    # Test indv. functions for fixed window
    # with 2 existing tracks in the queue
    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        scoring_reduction="max",
        track_matching_method="greedy",
    )
    _ = tracker.track(pred_instances, 0)

    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    # Test points as feature
    track_instances = tracker.get_features(pred_instances, 0, None)
    assert isinstance(track_instances, TrackInstances)
    for p, t in zip(pred_instances, track_instances.features):
        assert np.all(p.numpy() == t)

    # Test get_scores(), oks as scoring
    candidates_list = tracker.generate_candidates()
    candidate_feature_dict = tracker.update_candidates(candidates_list, None)
    scores = tracker.get_scores(track_instances, candidate_feature_dict)
    assert np.allclose(scores, np.array([[1.0, 0], [0, 1.0]]))

    # Test assign_tracks()
    cost = tracker.scores_to_cost_matrix(scores)
    track_instances = tracker.assign_tracks(track_instances, cost)
    assert track_instances.track_ids[0] == 0 and track_instances.track_ids[1] == 1
    assert np.all(track_instances.features[0] == pred_instances[0].numpy())
    assert len(tracker.candidate.current_tracks) == 2
    assert len(tracker.candidate.tracker_queue) == 2
    assert track_instances.tracking_scores == [1.0, 1.0]

    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidate.tracker_queue) == 3
    assert len(tracker.candidate.current_tracks) == 2
    assert np.all(
        tracker.candidate.tracker_queue[0].features[0]
        == tracker.candidate.tracker_queue[2].features[0]
    )
    assert (
        tracker.candidate.tracker_queue[0].track_ids[1]
        == tracker.candidate.tracker_queue[2].track_ids[1]
    )

    # Test with NaNs
    tracker.candidate.tracker_queue[0].features[0] = np.full(
        tracker.candidate.tracker_queue[0].features[0].shape, np.nan
    )
    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidate.current_tracks) == 2
    assert (
        tracked_instances[0].track.name == "track_0"
        and tracked_instances[1].track.name == "track_1"
    )

    # Test local queue tracker
    # with existing tracks
    pred_instances, _ = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(
        candidates_method="local_queues",
    )
    _ = tracker.track(pred_instances, 0)

    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidate.tracker_queue) == 2
    assert (
        len(tracker.candidate.tracker_queue[0]) == 2
        and len(tracker.candidate.tracker_queue[1]) == 2
    )
    assert np.all(
        tracker.candidate.tracker_queue[0][0].feature
        == tracker.candidate.tracker_queue[0][1].feature
    )

    # test features - centroids + euclidean scoring
    tracker = Tracker.from_config(
        features="centroids", scoring_reduction="max", scoring_method="euclidean_dist"
    )
    tracked_instances = tracker.track(pred_instances, 0)  # add instances to queue
    track_instances = tracker.get_features(pred_instances, 0, None)
    for p, t in zip(pred_instances, track_instances.features):
        pts = p.numpy()
        centroid = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        assert np.all(centroid == t)
    candidates_list = tracker.generate_candidates()
    candidate_feature_dict = tracker.update_candidates(candidates_list, None)
    scores = tracker.get_scores(track_instances, candidate_feature_dict)
    assert scores[0, 0] == 0 and scores[1, 1] == 0
    assert scores[1, 0] == scores[0, 1]

    # test features - bboxes + iou scoring
    tracker = Tracker.from_config(features="bboxes", scoring_method="iou")
    tracked_instances = tracker.track(pred_instances, 0)  # add instances to queue
    track_instances = tracker.get_features(pred_instances, 0, None)
    for p, t in zip(pred_instances, track_instances.features):
        pts = p.numpy()
        bbox = (
            np.nanmin(pts[:, 0]),
            np.nanmin(pts[:, 1]),
            np.nanmax(pts[:, 0]),
            np.nanmax(pts[:, 1]),
        )
        assert np.all(bbox == t)
    candidates_list = tracker.generate_candidates()
    candidate_feature_dict = tracker.update_candidates(candidates_list, None)
    assert isinstance(candidate_feature_dict[0][0], TrackedInstanceFeature)
    assert candidate_feature_dict[0][0].shifted_keypoints is None
    scores = tracker.get_scores(track_instances, candidate_feature_dict)
    assert scores[0, 0] == 1 and scores[1, 1] == 1
    assert scores[1, 0] == scores[0, 1] == 0

    # Test for invalid arguments
    # candidate method
    with pytest.raises(ValueError):
        tracker = Tracker.from_config(
            max_tracks=30, window_size=10, candidates_method="tracking"
        )
    assert "tracking is not a valid method" in caplog.text

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(features="centered")
        track_instances = tracker.track(pred_instances, 0)
    assert "Invalid `features` argument." in caplog.text

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(scoring_method="dist")
        track_instances = tracker.track(pred_instances, 0)

        track_instances = tracker.track(pred_instances, 0)
    assert "Invalid `scoring_method` argument." in caplog.text

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(scoring_reduction="min")
        track_instances = tracker.track(pred_instances, 0)

        track_instances = tracker.track(pred_instances, 0)
    assert "Invalid `scoring_reduction` argument." in caplog.text

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(track_matching_method="min")
        track_instances = tracker.track(pred_instances, 0)

        track_instances = tracker.track(pred_instances, 0)
    assert "Invalid `track_matching_method` argument." in caplog.text


def test_tracker_track_objects_not_shared():
    """Regression test for #574: independent Trackers must not share `_track_objects`.

    `Tracker` is an `attrs.define` class and `_track_objects` was declared with a
    bare mutable default (`{}`), which attrs turns into a single shared default
    object reused across all instances. This caused track-id -> `sio.Track`
    mappings from one tracking run to leak into a subsequently constructed
    `Tracker`, producing cross-contaminated track identities. Each `Tracker`
    (and `FlowShiftTracker`, which inherits the field) must own a distinct dict.
    """
    tracker_a = Tracker.from_config()
    tracker_b = Tracker.from_config()

    # Identity: the two instances must not reference the same dict object.
    assert tracker_a._track_objects is not tracker_b._track_objects

    # Each tracker starts with an empty, isolated mapping.
    assert tracker_a._track_objects == {}
    assert tracker_b._track_objects == {}

    # Isolation: mutating one tracker's mapping must not leak into the other.
    tracker_a._track_objects[0] = sio.Track("track_0")
    assert 0 not in tracker_b._track_objects
    assert tracker_b._track_objects == {}

    # And the reverse direction, to be thorough.
    tracker_b._track_objects[1] = sio.Track("track_1")
    assert 1 not in tracker_a._track_objects
    assert list(tracker_a._track_objects.keys()) == [0]

    # The FlowShiftTracker subclass inherits the field and must also be isolated.
    flow_tracker = Tracker.from_config(use_flow=True)
    assert isinstance(flow_tracker, FlowShiftTracker)
    assert flow_tracker._track_objects is not tracker_a._track_objects
    assert flow_tracker._track_objects == {}


def test_flowshifttracker(
    minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
):
    """Tests for `FlowShiftTracker` class."""
    # Test Fixed-window method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, imgs = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(
        candidates_method="fixed_window",
        use_flow=True,
        track_matching_method="greedy",
    )
    assert isinstance(tracker, FlowShiftTracker)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(
        pred_instances, 0, imgs[0]
    )  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidate.tracker_queue) == 1
    assert len(tracker.candidate.current_tracks) == 2

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0, imgs[0])
    assert len(tracker.candidate.tracker_queue) == 2
    assert len(tracker.candidate.current_tracks) == 2
    assert np.all(
        tracker.candidate.tracker_queue[0].features[0]
        == tracker.candidate.tracker_queue[1].features[0]
    )
    assert (
        tracker.candidate.tracker_queue[0].track_ids[1]
        == tracker.candidate.tracker_queue[1].track_ids[1]
    )

    # Test Local queue method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, imgs = get_pred_instances(
        minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
    )
    tracker = Tracker.from_config(
        candidates_method="local_queues",
        use_flow=True,
        of_img_scale=0.5,
    )
    assert isinstance(tracker, FlowShiftTracker)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(
        pred_instances, 0, imgs[0]
    )  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidate.tracker_queue[0]) == 1
    assert len(tracker.candidate.current_tracks) == 2

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0, imgs[0].astype("float32"))
    assert len(tracker.candidate.tracker_queue[0]) == 2
    assert len(tracker.candidate.current_tracks) == 2
    assert np.all(
        tracker.candidate.tracker_queue[0][0].feature
        == tracker.candidate.tracker_queue[0][1].feature
    )
    assert np.any(
        tracker.candidate.tracker_queue[0][0].feature
        != tracker.candidate.tracker_queue[1][1].feature
    )

    # Test update_candidates()
    candidates_list = tracker.generate_candidates()
    candidate_feature_dict = tracker.update_candidates(candidates_list, imgs[0])
    assert isinstance(candidate_feature_dict[0][0], TrackedInstanceFeature)
    assert candidate_feature_dict[0][0].shifted_keypoints is not None
    assert np.any(
        candidate_feature_dict[0][0].src_predicted_instance.numpy()
        == candidate_feature_dict[0][0].shifted_keypoints
    )

    # Test `_preprocess_imgs`
    ref, new = tracker._preprocess_imgs(
        imgs[0].astype("float32"), imgs[0].astype("float32")
    )
    assert np.issubdtype(ref.dtype, np.integer)
    assert np.issubdtype(new.dtype, np.integer)
    assert imgs[0].shape == (384, 384, 1)
    assert ref.shape == (192, 192) and new.shape == (192, 192)


def test_run_tracker(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    minimal_instance,
    tmp_path,
):
    """Tests for run_tracker."""
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        scoring_reduction="robust_quantile",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    tracked_lfs = run_tracker(
        untracked_frames=[x for x in labels],
        max_tracks=2,
        candidates_method="local_queues",
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
    )
    output = sio.Labels(
        labeled_frames=tracked_lfs,
        videos=labels.videos,
        skeletons=labels.skeletons,
    )
    assert len(output.tracks) == 2

    # test run tracker with post connect single breaks and without target instance count
    with pytest.raises(Exception):
        labels = run_inference(
            model_paths=[
                minimal_instance_centroid_ckpt,
                minimal_instance_centered_instance_ckpt,
            ],
            data_path=centered_instance_video.as_posix(),
            make_labels=True,
            output_path=tmp_path / "test.slp",
            max_instances=2,
            peak_threshold=0.1,
            frames=[x for x in range(0, 10)],
            integral_refinement="integral",
            scoring_reduction="robust_quantile",
            device="cpu" if torch.backends.mps.is_available() else "auto",
        )

        tracked_lfs = run_tracker(
            untracked_frames=[x for x in labels],
            max_tracks=None,
            candidates_method="local_queues",
            post_connect_single_breaks=True,
        )

    # test tracking with only user-labeled instances
    user_labeled_labels = sio.load_slp(minimal_instance)
    assert user_labeled_labels[0].has_user_instances
    tracked_lfs = run_tracker(
        untracked_frames=[x for x in user_labeled_labels],
        max_tracks=2,
        candidates_method="local_queues",
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
    )
    output = sio.Labels(
        labeled_frames=tracked_lfs,
        videos=labels.videos,
        skeletons=labels.skeletons,
    )
    assert len(output.tracks) == 2


def test_post_clean_up(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Tests for post clean up."""
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        scoring_reduction="robust_quantile",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    # test post clean up
    tracked_lfs = run_tracker(
        untracked_frames=[x for x in labels],
        max_tracks=2,
        candidates_method="local_queues",
        tracking_clean_instance_count=1,
    )
    assert len(tracked_lfs[0].instances) == 1


# Tests for connect_single_breaks fix (GitHub issue: sleap#2618)
from sleap_nn.tracking.tracker import connect_single_breaks


def _create_test_skeleton():
    """Create a simple skeleton for testing."""
    return sio.Skeleton(nodes=["head", "tail"])


def _create_instance(skeleton, points, track):
    """Create a PredictedInstance for testing."""
    return sio.PredictedInstance.from_numpy(
        points_data=np.array(points, dtype=np.float32),
        skeleton=skeleton,
        track=track,
        score=1.0,
    )


def test_connect_single_breaks_stale_reference():
    """Test that connect_single_breaks updates last_good_frame_tracks correctly.

    This tests the fix for the bug where last_good_frame_tracks would never update
    when max_instances doesn't match the actual instance count, causing incorrect
    track swaps.

    Bug scenario:
    - Frame 0: Only Mouse A detected → last_good_frame_tracks = {track_0}
    - Frame 1: Both mice detected, but last_good_frame_tracks doesn't update
    - Frame 2: Only Mouse B → swap occurs because track_1 is "extra"
    - Result: Mouse B steals Mouse A's track

    The fix ensures last_good_frame_tracks updates when len(frame_tracks) >=
    len(last_good_frame_tracks), not just when len == max_instances.
    """
    skeleton = _create_test_skeleton()
    video = sio.Video(filename="test.mp4")
    track_0 = sio.Track(name="track_0")
    track_1 = sio.Track(name="track_1")

    # Scenario: First frame has only one instance
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                _create_instance(skeleton, [[100.0, 100.0], [110.0, 110.0]], track_0),
            ],
        ),
        sio.LabeledFrame(
            video=video,
            frame_idx=1,
            instances=[
                _create_instance(skeleton, [[101, 101], [111, 111]], track_0),
                _create_instance(skeleton, [[201, 201], [211, 211]], track_1),
            ],
        ),
        sio.LabeledFrame(
            video=video,
            frame_idx=2,
            instances=[
                _create_instance(skeleton, [[202, 202], [212, 212]], track_1),  # Only B
            ],
        ),
        sio.LabeledFrame(
            video=video,
            frame_idx=3,
            instances=[
                _create_instance(skeleton, [[103, 103], [113, 113]], track_0),
                _create_instance(skeleton, [[203, 203], [213, 213]], track_1),
            ],
        ),
    ]

    # With mismatched max_instances (previously buggy)
    import copy

    frames_test = copy.deepcopy(frames)
    connect_single_breaks(frames_test, max_instances=10)

    # Verify all tracks are correct
    for lf in frames_test:
        for inst in lf.instances:
            pos = inst.numpy()[0, 0]
            expected_track = "track_0" if pos < 150 else "track_1"
            assert inst.track.name == expected_track, (
                f"Frame {lf.frame_idx}: Instance at ({pos},...) has {inst.track.name}, "
                f"expected {expected_track}"
            )


def test_connect_single_breaks_empty_frames():
    """Test that connect_single_breaks handles empty frame list."""
    result = connect_single_breaks([], max_instances=2)
    assert result == []


def test_kalmanshifttracker_from_config():
    """Test KalmanShiftTracker dispatch and validation in `from_config` (#572)."""
    # Dispatch: use_kalman -> KalmanShiftTracker (and NOT FlowShiftTracker).
    tracker = Tracker.from_config(use_kalman=True, tracking_target_instance_count=2)
    assert isinstance(tracker, KalmanShiftTracker)
    assert not isinstance(tracker, FlowShiftTracker)
    assert tracker.kf_init_frame_count == 10
    assert tracker.kf_reset_gap_size == 5
    assert tracker.kf_node_indices is None

    # Local queues + explicit node indices + custom warm-up.
    tracker_lq = Tracker.from_config(
        use_kalman=True,
        candidates_method="local_queues",
        max_tracks=2,
        kf_node_indices=[0, 1],
        kf_init_frame_count=3,
    )
    assert isinstance(tracker_lq, KalmanShiftTracker)
    assert tracker_lq.kf_node_indices == [0, 1]
    assert tracker_lq.is_local_queue
    assert tracker_lq.kf_init_frame_count == 3

    # use_kalman and use_flow are mutually exclusive.
    with pytest.raises(ValueError):
        Tracker.from_config(
            use_kalman=True, use_flow=True, tracking_target_instance_count=2
        )

    # Kalman requires a known target identity count.
    with pytest.raises(ValueError):
        Tracker.from_config(use_kalman=True)

    # max_tracks satisfies the target-count requirement.
    tracker_mt = Tracker.from_config(
        use_kalman=True, candidates_method="local_queues", max_tracks=2
    )
    assert isinstance(tracker_mt, KalmanShiftTracker)


def _run_synthetic_kalman_tracking(tracker, n_frames=10, n_nodes=2):
    """Track two synthetic instances moving in opposite directions.

    Returns the list of sorted per-frame track names.
    """
    skeleton = sio.Skeleton(nodes=[f"n{i}" for i in range(n_nodes)])

    def make_instance(base_x, base_y, t, direction):
        pts = [
            [base_x + direction * 2 * t + 5 * j, base_y + t + 5 * j]
            for j in range(n_nodes)
        ]
        return sio.PredictedInstance.from_numpy(
            points_data=np.array(pts, dtype=np.float32),
            skeleton=skeleton,
            score=1.0,
        )

    per_frame = []
    for t in range(n_frames):
        instance_a = make_instance(100.0, 100.0, t, direction=1)
        instance_b = make_instance(400.0, 100.0, t, direction=-1)
        tracked = tracker.track([instance_a, instance_b], frame_idx=t)
        per_frame.append(
            sorted(inst.track.name for inst in tracked if inst.track is not None)
        )
    return per_frame


def test_kalmanshifttracker_tracking():
    """End-to-end synthetic test: KalmanShiftTracker maintains stable tracks (#572)."""
    tracker = Tracker.from_config(
        use_kalman=True,
        candidates_method="local_queues",
        max_tracks=2,
        kf_init_frame_count=3,
        tracking_target_instance_count=2,
    )
    per_frame = _run_synthetic_kalman_tracking(tracker, n_frames=10, n_nodes=2)

    # Filters are fit after the warm-up window; two filters are maintained.
    assert tracker._initialized
    assert len(tracker._kalman_filters) == 2

    # Two distinct, stable tracks are held across every frame.
    all_tracks = {name for frame in per_frame for name in frame}
    assert all_tracks == {"track_0", "track_1"}
    for frame in per_frame:
        assert frame == ["track_0", "track_1"]


def test_kalmanshifttracker_init_filters_shape():
    """KalmanShiftTracker fits one constant-velocity *centroid* filter per track (#572).

    The motion model tracks the per-track centroid (state ``[cx, vcx, cy, vcy]``),
    not every keypoint independently, so the filter matrices are a fixed 4x4 / 2x4
    regardless of the number of (selected) nodes.
    """
    tracker = Tracker.from_config(
        use_kalman=True,
        candidates_method="local_queues",
        max_tracks=2,
        kf_init_frame_count=3,
        kf_node_indices=[0, 1, 2],
        tracking_target_instance_count=2,
    )
    _run_synthetic_kalman_tracking(tracker, n_frames=8, n_nodes=4)

    assert tracker._initialized
    assert len(tracker._kalman_filters) == 2

    for kf in tracker._kalman_filters.values():
        assert np.asarray(kf.transition_matrices).shape == (4, 4)
        assert np.asarray(kf.observation_matrices).shape == (2, 4)
    for result in tracker._last_results.values():
        assert len(result["means"]) == 4  # [cx, vcx, cy, vcy]


def test_run_tracker_with_kalman(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    minimal_instance,
    tmp_path,
):
    """`run_tracker` wires use_kalman through to KalmanShiftTracker end-to-end (#572)."""
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    tracked_lfs = run_tracker(
        untracked_frames=[x for x in labels],
        candidates_method="local_queues",
        max_tracks=2,
        use_kalman=True,
        kf_init_frame_count=3,
        tracking_target_instance_count=2,
    )
    output = sio.Labels(
        labeled_frames=tracked_lfs,
        videos=labels.videos,
        skeletons=labels.skeletons,
    )
    # max_tracks caps identities at 2; the Kalman tracker should hold them.
    assert len(output.tracks) == 2


def test_run_tracker_kalman_requires_target():
    """`run_tracker` with use_kalman but no target count raises before tracking (#572)."""
    skeleton = sio.Skeleton(nodes=["head", "tail"])
    video = sio.Video(filename="test.mp4")
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=t,
            instances=[
                sio.PredictedInstance.from_numpy(
                    points_data=np.array(
                        [[100.0 + t, 100.0], [110.0 + t, 110.0]], dtype=np.float32
                    ),
                    skeleton=skeleton,
                    score=1.0,
                )
            ],
        )
        for t in range(3)
    ]
    # The ValueError is raised by Tracker.from_config before any frame/image is read.
    with pytest.raises(ValueError):
        run_tracker(untracked_frames=frames, use_kalman=True)


# ──────────────────────────────────────────────────────────────────────────
# KalmanShiftTracker correctness integration tests (#572 follow-up).
#
# These are fast, fixture-free, synthetic stress tests guarding the robustness
# fixes (measurement gating, velocity capping, gap coasting, prediction/last-obs
# blending). The pre-fix implementation produced ID-switch *storms* on these
# scenarios (random motion ~20 switches, false positives ~12, occlusion ~2-24);
# the plain base tracker produces 0. They count an ID switch whenever a ground-
# truth identity's assigned track id changes between consecutive tracked frames.
# A rigorous MOT-metrics benchmark lives in the (gitignored) scratch workspace.
# ──────────────────────────────────────────────────────────────────────────

_BODY3 = np.array([[0.0, -8.0], [0.0, 0.0], [0.0, 8.0]])


def _track_synthetic_with_ids(tracker, frames, skeleton):
    """Run a tracker over synthetic gt-tagged frames; return (n_tracks, n_switches).

    ``frames`` is a list over time of lists of ``(gt_id, keypoints (K, 2))``
    detections (gt_id == -1 marks a false positive). Counts an ID switch whenever a
    ground-truth identity's assigned track id differs from the last frame it was
    tracked in.
    """
    gt_last_track = {}
    switches = 0
    all_tracks = set()
    for t, dets in enumerate(frames):
        instances = []
        gt_by_obj = {}
        for gt_id, kpts in dets:
            inst = sio.PredictedInstance.from_numpy(
                points_data=np.asarray(kpts, dtype=np.float32),
                skeleton=skeleton,
                score=1.0,
            )
            instances.append(inst)
            gt_by_obj[id(inst)] = gt_id
        tracked = tracker.track(instances, frame_idx=t)
        for inst in tracked:
            if inst.track is None:
                continue
            track_name = inst.track.name
            all_tracks.add(track_name)
            gt_id = gt_by_obj.get(id(inst))
            if gt_id is not None and gt_id >= 0:
                if gt_id in gt_last_track and gt_last_track[gt_id] != track_name:
                    switches += 1
                gt_last_track[gt_id] = track_name
    return len(all_tracks), switches


def _kalman_tracker(**overrides):
    kwargs = dict(
        candidates_method="local_queues",
        max_tracks=2,
        kf_init_frame_count=5,
        tracking_target_instance_count=2,
    )
    kwargs.update(overrides)
    return Tracker.from_config(use_kalman=True, **kwargs)


def _kalman_filter_centroid(tracker, track_id):
    """Version-robust centroid of a track's filter state mean (NaN-ignoring)."""
    means = np.asarray(tracker._last_results[track_id]["means"])
    return np.nanmean(means[::2].reshape(-1, 2), axis=0)


def test_kalman_no_switch_storm_on_erratic_motion():
    """Erratic motion with per-keypoint noise must not trigger an ID-switch storm (#572).

    Two well-separated identities follow momentum random walks, with independent
    per-keypoint jitter (not just a rigid centroid shift). The pre-fix per-node EM
    warm-up overfit that jitter into a degenerate velocity and stormed ID switches; the
    centroid motion model + capping must stay stable (0 switches vs the base tracker's 0).
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])
    rng = np.random.default_rng(1)
    T = 60
    # Two nearby (60 px apart) momentum random walks with per-keypoint jitter — close
    # enough that the base warm-up assignment oscillates. This exact config storms on
    # the pre-fix per-node tracker (~12 switches); the fix must hold 0.
    centroids = {0: np.array([150.0, 210.0]), 1: np.array([150.0, 270.0])}
    vel = {0: rng.normal(0, 10, 2), 1: rng.normal(0, 10, 2)}
    frames = []
    for _ in range(T):
        dets = []
        for gid in (0, 1):
            vel[gid] = 0.7 * vel[gid] + 0.3 * rng.normal(0, 10, 2)
            centroids[gid] = centroids[gid] + vel[gid]
            body = _BODY3 + centroids[gid] + rng.normal(0, 3, _BODY3.shape)
            dets.append((gid, body))
        frames.append(dets)
    n_tracks, switches = _track_synthetic_with_ids(
        _kalman_tracker(kf_init_frame_count=10), frames, skeleton
    )
    assert n_tracks == 2
    assert switches == 0


def test_kalman_gate_rejects_false_positives():
    """The measurement gate must keep a far false positive from corrupting the filter (#572).

    A single track moves linearly; on a few frames its real detection is replaced by a
    false positive far away (so the FP is the only detection and is matched to the lone
    track). The gate must reject the FP and coast, keeping the filter near the true
    trajectory. The pre-fix tracker had no gating, so the FP correction yanked the
    filter state hundreds of px away.
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])
    tracker = _kalman_tracker(max_tracks=1, tracking_target_instance_count=1)
    fp_frames = {16, 19, 22}
    max_centroid_error = 0.0
    for t in range(30):
        if t in fp_frames:  # real detection replaced by a far false positive
            center = np.array([300.0, 400.0])
        else:
            center = np.array([60.0 + 5.0 * t, 100.0])
        inst = sio.PredictedInstance.from_numpy(
            points_data=(_BODY3 + center).astype(np.float32),
            skeleton=skeleton,
            score=1.0,
        )
        tracker.track([inst], frame_idx=t)
        if tracker._initialized and 0 in tracker._last_results:
            corrected_frame = tracker._last_corrected_frame.get(0, t)
            true_centroid = np.array([60.0 + 5.0 * corrected_frame, 100.0])
            err = float(
                np.linalg.norm(_kalman_filter_centroid(tracker, 0) - true_centroid)
            )
            max_centroid_error = max(max_centroid_error, err)
    # The gate keeps the filter near the true track; an ungated filter would be pulled
    # hundreds of px toward the (300, 400) false positives.
    assert max_centroid_error < 40.0


def test_kalman_entrant_gets_filter():
    """An identity entering after warm-up gets its own filter, not just the base path (#572).

    The pre-fix tracker fit filters once at warm-up, so an animal that appeared later
    was permanently stuck on the base feature path (filter count capped at the warm-up
    count). Lazy (re)fitting must give the entrant its own filter.
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])
    tracker = _kalman_tracker(max_tracks=3, tracking_target_instance_count=3)
    frames = []
    for t in range(40):
        dets = []
        for gid, y in ((0, 100.0), (1, 350.0)):
            dets.append((gid, _BODY3 + np.array([60.0 + 5.0 * t, y])))
        if t >= 20:  # a third identity enters well after warm-up
            dets.append((2, _BODY3 + np.array([60.0 + 5.0 * t, 225.0])))
        frames.append(dets)
    n_tracks, _ = _track_synthetic_with_ids(tracker, frames, skeleton)
    assert n_tracks == 3
    # The entrant must have been fit its own filter (pre-fix: stuck at 2).
    assert len(tracker._kalman_filters) == 3


def test_kalman_from_config_robustness_knobs():
    """The robustness knobs are configurable via from_config and reach the tracker (#572)."""
    tracker = Tracker.from_config(
        use_kalman=True,
        tracking_target_instance_count=2,
        kf_prediction_blend=0.3,
        kf_gate_step_mult=6.0,
        kf_min_gate_px=55.0,
        kf_velocity_cap_mult=2.0,
        kf_min_velocity_cap_px=12.0,
    )
    assert isinstance(tracker, KalmanShiftTracker)
    assert tracker.kf_prediction_blend == 0.3
    assert tracker.kf_gate_step_mult == 6.0
    assert tracker.kf_min_gate_px == 55.0
    assert tracker.kf_velocity_cap_mult == 2.0
    assert tracker.kf_min_velocity_cap_px == 12.0


def test_kalman_keypoints_mode_tracks_per_node():
    """kf_track_features='keypoints' fits one filter per node and tracks (#572)."""
    n_nodes = 3
    tracker = _kalman_tracker(
        kf_track_features="keypoints", kf_init_frame_count=3, kf_node_indices=None
    )
    assert tracker.kf_track_features == "keypoints"
    per_frame = _run_synthetic_kalman_tracking(tracker, n_frames=10, n_nodes=n_nodes)

    assert tracker._initialized
    assert len(tracker._kalman_filters) == 2
    # Per-node state has 4 dims per tracked node (vs 4 total for the centroid model).
    for result in tracker._last_results.values():
        assert len(result["means"]) == 4 * n_nodes
    # Two stable tracks are maintained.
    assert {name for frame in per_frame for name in frame} == {"track_0", "track_1"}


def test_kalman_keypoints_supports_bbox_iou():
    """Keypoints mode works with the bbox/iou featurization/metric alternative (#572)."""
    tracker = _kalman_tracker(
        kf_track_features="keypoints",
        kf_init_frame_count=3,
        features="bboxes",
        scoring_method="iou",
    )
    per_frame = _run_synthetic_kalman_tracking(tracker, n_frames=10, n_nodes=3)
    assert tracker._initialized
    assert {name for frame in per_frame for name in frame} == {"track_0", "track_1"}


def test_kalman_oks_stddev_auto_resolves_by_mode():
    """oks_stddev auto-resolves: 0.1 for keypoints mode, 0.025 otherwise (#572)."""
    kp = Tracker.from_config(
        use_kalman=True, kf_track_features="keypoints", tracking_target_instance_count=2
    )
    cen = Tracker.from_config(
        use_kalman=True, kf_track_features="centroid", tracking_target_instance_count=2
    )
    assert kp.oks_stddev == 0.1
    assert cen.oks_stddev == 0.025
    assert Tracker.from_config().oks_stddev == 0.025  # base tracker
    # An explicit value always wins over the auto-resolution.
    explicit = Tracker.from_config(
        use_kalman=True,
        kf_track_features="keypoints",
        tracking_target_instance_count=2,
        oks_stddev=0.03,
    )
    assert explicit.oks_stddev == 0.03


def test_kalman_invalid_track_features_raises():
    """An unknown kf_track_features value is rejected by from_config (#572)."""
    with pytest.raises(ValueError):
        Tracker.from_config(
            use_kalman=True,
            kf_track_features="garbage",
            tracking_target_instance_count=2,
        )


@pytest.mark.parametrize("method", ["fixed_window", "local_queues"])
@pytest.mark.parametrize(
    "n_per_frame",
    [
        # interior empty: empty frame between non-empty ones (the canonical #611 repro).
        [2, 0, 2, 2],
        # consecutive interior empties: two empty candidates in the window at once --
        # the strongest stressor of the guard.
        [2, 0, 0, 2],
        # leading empty: frame 0 is empty, so tracking inits on frame 1.
        [0, 2, 2],
        # trailing empty: gap at the end of the sequence.
        [2, 2, 0],
        # all empty: nothing ever enters the queue with instances.
        [0, 0, 0],
    ],
    ids=[
        "interior_empty",
        "consecutive_empty",
        "leading_empty",
        "trailing_empty",
        "all_empty",
    ],
)
def test_flowshifttracker_empty_frame(method, n_per_frame):
    """FlowShiftTracker handles frames with zero instances without crashing (#611).

    Note: only the ``fixed_window`` method actually reproduces the pre-fix
    ``need at least one array to concatenate`` crash, since the fixed-window queue
    appends empty frames. For ``local_queues``,
    ``LocalQueueCandidates.get_instances_groupby_frame_idx`` never creates a group key
    for an empty frame, so this acts as a smoke / no-regression check there; the
    local-queue guard is exercised directly in
    ``test_flowshifttracker_local_queue_empty_group_guard``.
    """
    skel = sio.Skeleton(["a", "b", "c"])

    def mk(cx, cy):
        pts = np.array([[cx, cy], [cx + 5, cy], [cx, cy + 5]], dtype="float32")
        return sio.PredictedInstance.from_numpy(
            points_data=pts,
            skeleton=skel,
            point_scores=np.ones(3),
            score=1.0,
        )

    # Build frames with deterministic, well-separated instances per frame so the
    # tracker assigns a stable number of tracks on recovery.
    def make_frame(n):
        centers = [(20, 30), (80, 90)]
        offset = 0  # jitter is unnecessary; flow is computed on shared images below.
        return [mk(cx + offset, cy + offset) for cx, cy in centers[:n]]

    rng = np.random.default_rng(0)
    frames = [make_frame(n) for n in n_per_frame]

    tracker = Tracker.from_config(
        candidates_method=method,
        features="keypoints",
        scoring_method="oks",
        track_matching_method="greedy",
        use_flow=True,
    )
    assert isinstance(tracker, FlowShiftTracker)

    tracked_per_frame = []
    for fidx, instances in enumerate(frames):
        img = rng.integers(0, 256, size=(128, 128, 1), dtype="uint8")
        tracked = tracker.track(instances, fidx, img)
        tracked_per_frame.append(tracked)

    # Every frame returns exactly as many tracked instances as it had detections,
    # and each tracked instance carries a track. Empty frames return [].
    for n, tracked in zip(n_per_frame, tracked_per_frame):
        assert len(tracked) == n
        if n == 0:
            assert tracked == []
        else:
            for t in tracked:
                assert t.track is not None


def test_flowshifttracker_local_queue_empty_group_guard():
    """Local-queue branch skips a frame group with zero instances (#611 guard).

    ``get_instances_groupby_frame_idx`` never produces an empty group through the
    normal pipeline (empty frames contribute no ``TrackInstanceLocalQueue`` objects),
    so the ``if not ref_pts: continue`` guard in the local-queue branch is otherwise
    dead code w.r.t. the live pipeline. This feeds an empty group directly so the
    guard at ``FlowShiftTracker.get_shifted_instances_from_prv_frames`` is exercised:
    pre-fix, the empty group would reach ``np.concatenate([])`` and raise
    ``need at least one array to concatenate``.
    """
    skel = sio.Skeleton(["a", "b", "c"])
    pts = np.array([[20, 30], [25, 30], [20, 35]], dtype="float32")
    inst = sio.PredictedInstance.from_numpy(
        points_data=pts, skeleton=skel, point_scores=np.ones(3), score=1.0
    )
    img = np.zeros((128, 128, 1), dtype="uint8")

    tracker = Tracker.from_config(
        candidates_method="local_queues",
        features="keypoints",
        scoring_method="oks",
        track_matching_method="greedy",
        use_flow=True,
    )
    feature_method = tracker._feature_methods[tracker.features]

    populated = TrackInstanceLocalQueue(
        src_instance=inst,
        src_instance_idx=0,
        feature=feature_method(pts),
        track_id=0,
        tracking_score=1.0,
        frame_idx=0,
        image=img,
    )

    # Force the grouping to yield one populated group (frame 0) and one empty group
    # (frame 1) so the guard's ``continue`` branch is hit.
    grouped = {0: [populated], 1: []}
    tracker.candidate.get_instances_groupby_frame_idx = lambda candidates_list: grouped

    shifted = tracker.get_shifted_instances_from_prv_frames(
        candidates_list=None,
        new_img=img,
        feature_method=feature_method,
    )

    # The empty group contributes nothing; only the populated frame's track appears.
    assert set(shifted.keys()) == {0}
    assert len(shifted[0]) == 1
