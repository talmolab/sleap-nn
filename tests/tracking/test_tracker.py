import pytest
import numpy as np
import sleap_io as sio
from sleap_nn.predict import run_inference
from sleap_nn.tracking.tracker import Tracker, FlowShiftTracker, run_tracker
from sleap_nn.tracking.track_instance import (
    TrackedInstanceFeature,
    TrackInstanceLocalQueue,
    TrackInstances,
)
from sleap_nn.tracking.utils import (
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
