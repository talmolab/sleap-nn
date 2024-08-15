import pytest
import numpy as np
from sleap_nn.inference.predictors import main
from sleap_nn.tracking.tracker import Tracker, FlowShiftTracker
import math


def get_pred_instances(minimal_instance_ckpt, n=2):
    """Get `sio.PredictedInstance` objects from Predictor class."""
    result_labels = main(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.0,
        integral_refinement="integral",
    )
    pred_instances = []
    imgs = []
    for idx, lf in enumerate(result_labels):
        pred_instances.extend(lf.instances)
        imgs.append(lf.image)
        if idx == n:
            break
    return pred_instances, imgs


def test_tracker(minimal_instance_ckpt):
    """Test `Tracker` module."""
    # Test for invalid candidates method
    with pytest.raises(ValueError):
        tracker = Tracker.from_config(
            max_tracks=30, window_size=10, candidates_method="tracking"
        )

    # Test for the first two instances (high instance threshold)
    pred_instances, _ = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config(instance_score_threshold=1.0)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)
    for t in tracked_instances:
        assert t.track is None
    assert len(tracker.candidates.current_tracks) == 0

    # Test Fixed-window method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="fixed_window"
    )
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidates.tracker_queue) == 1
    assert len(tracker.candidates.current_tracks) == 2

    # Test _get_features(): points as feature
    track_instances = tracker._get_features(pred_instances, 0, None)
    for p, t in zip(pred_instances, track_instances.features):
        assert np.all(p.numpy() == t)

    # Test _get_scores()
    scores = tracker._get_scores(track_instances)
    assert np.all(scores == np.array([[1.0, 0], [0, 1.0]]))

    # Test _assign_tracks() with existing 2 tracks
    cost = tracker._scores_to_cost_matrix(scores)
    track_instances = tracker._assign_tracks(track_instances, cost)
    assert track_instances.track_ids[0] == 0 and track_instances.track_ids[1] == 1
    assert np.all(track_instances.features[0] == pred_instances[0].numpy())
    assert len(tracker.candidates.current_tracks) == 2
    assert len(tracker.candidates.tracker_queue) == 2

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidates.tracker_queue) == 3
    assert len(tracker.candidates.current_tracks) == 2
    assert np.all(
        tracker.candidates.tracker_queue[0].features[0]
        == tracker.candidates.tracker_queue[2].features[0]
    )
    assert (
        tracker.candidates.tracker_queue[0].track_ids[1]
        == tracker.candidates.tracker_queue[1].track_ids[1]
    )

    # Test with NaNs
    tracker.candidates.tracker_queue[0].features[0] = np.full(
        tracker.candidates.tracker_queue[0].features[0].shape, np.NaN
    )
    track_instances = tracker._get_features(pred_instances, 0, image=None)
    scores = tracker._get_scores(track_instances)
    cost = tracker._scores_to_cost_matrix(scores)
    track_instances = tracker._assign_tracks(track_instances, cost)
    assert len(tracker.candidates.current_tracks) == 2
    assert track_instances.track_ids[0] == 0 and track_instances.track_ids[1] == 1

    # Test Local queues method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, _ = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="local_queues"
    )
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidates.tracker_queue[0]) == 1
    assert len(tracker.candidates.tracker_queue[1]) == 1
    assert len(tracker.candidates.current_tracks) == 2

    # Test _get_features(): points as feature
    # track_instances = tracker._get_features(pred_instances, 0, None)

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidates.tracker_queue) == 2
    assert (
        len(tracker.candidates.tracker_queue[0]) == 2
        and len(tracker.candidates.tracker_queue[1]) == 2
    )
    assert np.all(
        tracker.candidates.tracker_queue[0][0].feature
        == tracker.candidates.tracker_queue[0][1].feature
    )

    # Test with NaNs
    tracker.candidates.tracker_queue[0][0].feature = np.full(
        tracker.candidates.tracker_queue[0][0].feature.shape, np.NaN
    )
    track_instances = tracker._get_features(pred_instances, 0, None)
    scores = tracker._get_scores(track_instances)
    cost = tracker._scores_to_cost_matrix(scores)
    track_instances = tracker._assign_tracks(track_instances, cost)
    assert len(tracker.candidates.current_tracks) == 2
    assert track_instances[0].track_id == 0 and track_instances[1].track_id == 1

    # test features - centroids + euclidean scoring
    # TODO: test max!!
    tracker = Tracker.from_config(
        features="centroids", scoring_reduction="max", scoring_method="euclidean_dist"
    )
    tracked_instances = tracker.track(pred_instances, 0)  # add instances to queue
    track_instances = tracker._get_features(pred_instances, 0, None)
    for p, t in zip(pred_instances, track_instances.features):
        pts = p.numpy()
        centroid = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        assert np.all(centroid == t)
    scores = tracker._get_scores(track_instances)
    assert scores[0, 0] == 0 and scores[1, 1] == 0
    assert scores[1, 0] == scores[0, 1]
    # assert math.isclose(scores[0, 1], -130.86877, rel_tol=1e-4)

    # test featuires - bboxes + iou scoring
    tracker = Tracker.from_config(features="bboxes", scoring_method="iou")
    tracked_instances = tracker.track(pred_instances, 0)  # add instances to queue
    track_instances = tracker._get_features(pred_instances, 0, None)
    for p, t in zip(pred_instances, track_instances.features):
        pts = p.numpy()
        bbox = (
            np.nanmin(pts[:, 0]),
            np.nanmin(pts[:, 1]),
            np.nanmax(pts[:, 0]),
            np.nanmax(pts[:, 1]),
        )
        assert np.all(bbox == t)
    scores = tracker._get_scores(track_instances)
    assert scores[0, 0] == 1 and scores[1, 1] == 1
    assert scores[1, 0] == scores[0, 1] == 0

    # TODO max scoring reduction

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(features="centered")
        track_instances = tracker._get_features(pred_instances, 0)

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(scoring_method="dist")
        track_instances = tracker._get_features(pred_instances, 0)
        scores = tracker._get_scores(track_instances)

    with pytest.raises(ValueError):
        tracker = Tracker.from_config(scoring_reduction="min")
        track_instances = tracker._get_features(pred_instances, 0)
        tracker.candidates.current_tracks.append(1)
        scores = tracker._get_scores(track_instances)


def test_flowshifttracker(minimal_instance_ckpt):
    # Test Fixed-window method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, imgs = get_pred_instances(minimal_instance_ckpt)
    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="fixed_window", use_flow=True
    )
    assert isinstance(tracker, FlowShiftTracker)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(
        pred_instances, 0, imgs[0]
    )  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidates.tracker_queue) == 1
    assert len(tracker.candidates.current_tracks) == 2

    # TODO: add test for get scores function
    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0, imgs[0])
    assert len(tracker.candidates.tracker_queue) == 2
    assert len(tracker.candidates.current_tracks) == 2
    assert np.all(
        tracker.candidates.tracker_queue[0].features[0]
        == tracker.candidates.tracker_queue[1].features[0]
    )
    assert (
        tracker.candidates.tracker_queue[0].track_ids[1]
        == tracker.candidates.tracker_queue[1].track_ids[1]
    )

    # Test Local queue method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances, imgs = get_pred_instances(minimal_instance_ckpt)
    tracker = Tracker.from_config(
        instance_score_threshold=0.0, candidates_method="local_queues", use_flow=True
    )
    assert isinstance(tracker, FlowShiftTracker)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(
        pred_instances, 0, imgs[0]
    )  # 2 tracks are created
    for t in tracked_instances:
        assert t.track is not None
    assert len(tracker.candidates.tracker_queue[0]) == 1
    assert len(tracker.candidates.current_tracks) == 2

    # TODO: add test for get scores function
    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0, imgs[0])
    assert len(tracker.candidates.tracker_queue[0]) == 2
    assert len(tracker.candidates.current_tracks) == 2
    assert np.all(
        tracker.candidates.tracker_queue[0][0].feature
        == tracker.candidates.tracker_queue[0][1].feature
    )

    # more tests for adding/ updating tracks
