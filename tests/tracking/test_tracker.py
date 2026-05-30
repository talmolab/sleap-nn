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
    """KalmanShiftTracker fits per-track filters with correct matrix shapes (#572)."""
    node_indices = [0, 1, 2]
    tracker = Tracker.from_config(
        use_kalman=True,
        candidates_method="local_queues",
        max_tracks=2,
        kf_init_frame_count=3,
        kf_node_indices=node_indices,
        tracking_target_instance_count=2,
    )
    _run_synthetic_kalman_tracking(tracker, n_frames=8, n_nodes=4)

    assert tracker._initialized
    assert len(tracker._kalman_filters) == 2

    num_nodes = len(node_indices)
    for kf in tracker._kalman_filters.values():
        # Constant-velocity state has 4 dims per node; observation has 2 per node.
        assert np.asarray(kf.transition_matrices).shape == (
            4 * num_nodes,
            4 * num_nodes,
        )
        assert np.asarray(kf.observation_matrices).shape == (
            2 * num_nodes,
            4 * num_nodes,
        )
    for result in tracker._last_results.values():
        assert len(result["means"]) == 4 * num_nodes


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


def _base_tracker(**overrides):
    kwargs = dict(
        candidates_method="local_queues",
        max_tracks=2,
        tracking_target_instance_count=2,
    )
    kwargs.update(overrides)
    return Tracker.from_config(**kwargs)


def test_kalman_no_switch_storm_on_erratic_motion():
    """Erratic (random-walk) motion must not trigger an ID-switch storm (#572).

    Two well-separated identities follow momentum random walks. The constant-velocity
    EM warm-up previously fit a degenerate velocity here and stormed ~20 switches; with
    velocity capping + gating + blending the motion model must stay stable.
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])
    rng = np.random.default_rng(0)
    T = 30
    centroids = {0: np.array([120.0, 120.0]), 1: np.array([120.0, 320.0])}
    vel = {0: rng.normal(0, 3, 2), 1: rng.normal(0, 3, 2)}
    frames = []
    for _ in range(T):
        dets = []
        for gid in (0, 1):
            vel[gid] = 0.7 * vel[gid] + 0.3 * rng.normal(0, 4, 2)
            centroids[gid] = centroids[gid] + vel[gid]
            dets.append((gid, _BODY3 + centroids[gid]))
        frames.append(dets)
    n_tracks, switches = _track_synthetic_with_ids(_kalman_tracker(), frames, skeleton)
    assert n_tracks == 2
    assert switches == 0


def test_kalman_robust_to_false_positives():
    """False-positive detections must not corrupt the filters into a switch storm (#572).

    The measurement gate must reject FP-driven corrections (pre-fix: ~12 switches).
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])
    rng = np.random.default_rng(1)
    T = 40
    frames = []
    for t in range(T):
        dets = []
        for gid, y in ((0, 120.0), (1, 320.0)):
            c = np.array([60.0 + 5.0 * t, y]) + rng.normal(0, 3, 2)
            dets.append((gid, _BODY3 + c))
        if t % 3 == 0:  # inject a false positive between the two tracks
            c = np.array([60.0 + 5.0 * t, 220.0]) + rng.normal(0, 30, 2)
            dets.append((-1, _BODY3 + c))
        frames.append(dets)
    n_tracks, switches = _track_synthetic_with_ids(_kalman_tracker(), frames, skeleton)
    assert switches == 0


def test_kalman_maintains_identity_through_occlusion():
    """A single occluded track keeps (or cleanly re-acquires) its identity (#572).

    One of two well-separated identities is occluded for a 6-frame gap (> the default
    reset gap). The pre-fix tracker stormed switches here (the stale filter kept
    extrapolating and then absorbed the other animal's observations); the fix coasts
    across the gap and resets a stale track before it can be corrupted. The contract
    is "no worse than the base similarity tracker" (a scene with fewer detections than
    tracks hits a base-tracker matching degeneracy that is shared by both and is
    unrelated to the motion model).
    """
    skeleton = sio.Skeleton(nodes=["a", "b", "c"])

    def build_frames():
        rng = np.random.default_rng(3)
        frames = []
        for t in range(30):
            dets = []
            for gid, y in ((0, 120.0), (1, 320.0)):
                if gid == 0 and 12 <= t <= 17:  # occlude id 0 for 6 frames
                    continue
                c = np.array([60.0 + 5.0 * t, y]) + rng.normal(0, 3, 2)
                dets.append((gid, _BODY3 + c))
            frames.append(dets)
        return frames

    _, base_switches = _track_synthetic_with_ids(
        _base_tracker(), build_frames(), skeleton
    )
    n_tracks, kalman_switches = _track_synthetic_with_ids(
        _kalman_tracker(), build_frames(), skeleton
    )
    assert n_tracks == 2
    # Never worse than the plain tracker through the occlusion (no switch storm).
    assert kalman_switches <= base_switches
