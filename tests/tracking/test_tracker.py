import pytest
import numpy as np
from sleap_nn.inference.predictors import main
from sleap_nn.tracking.tracker import Tracker
import math


def get_pred_instances(minimal_instance_ckpt, n=10):
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
    for idx, lf in enumerate(result_labels):
        pred_instances.extend(lf.instances)
        if idx == n:
            break
    return pred_instances


def test_tracker(minimal_instance_ckpt):
    """Test `Tracker` module."""
    # Test for invalid candidates method
    with pytest.raises(ValueError):
        tracker = Tracker.from_config(
            max_tracks=30, window_size=10, candidates_method="tracking"
        )

    # Test for the first two instances (high instance threshold)
    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config(instance_score_threshold=1.0)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances, 0)
    for t in tracked_instances:
        assert t.track is None

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
    assert len(tracker.candidates.tracker_queue) == 2

    # Test _get_features(): points as feature
    track_instances = tracker._get_features(pred_instances, 0)
    for p, t in zip(pred_instances, track_instances):
        assert np.all(p.numpy() == t.feature)

    # Test _get_scores()
    scores = tracker._get_scores(track_instances)
    assert np.all(scores == np.array([[1.0, 0], [0, 1.0]]))

    # Test _assign_tracks() with existing 2 tracks
    cost = tracker._scores_to_cost_matrix(scores)
    track_instances = tracker._assign_tracks(track_instances, cost)
    assert track_instances[0].track_id == 0 and track_instances[1].track_id == 1
    assert np.all(track_instances[0].feature == pred_instances[0].numpy())
    assert len(tracker.candidates.current_tracks) == 2

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances, 0)
    assert len(tracker.candidates.tracker_queue) == 4
    assert np.all(
        tracker.candidates.tracker_queue[0].feature
        == tracker.candidates.tracker_queue[2].feature
    )
    assert (
        tracker.candidates.tracker_queue[0].track_id
        == tracker.candidates.tracker_queue[2].track_id
    )

    # Test with NaNs
    tracker.candidates.tracker_queue[0].feature = np.full(
        tracker.candidates.tracker_queue[0].feature.shape, np.NaN
    )
    track_instances = tracker._get_features(pred_instances, 0)
    scores = tracker._get_scores(track_instances)
    cost = tracker._scores_to_cost_matrix(scores)
    track_instances = tracker._assign_tracks(track_instances, cost)
    assert len(tracker.candidates.current_tracks) == 2
    assert track_instances[0].track_id == 0 and track_instances[1].track_id == 1

    # Test Local queues method: pose as feature, oks scoring method
    # Test for the first two instances (tracks assigned to each of the new instances)
    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
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

    # Test _get_features(): points as feature
    track_instances = tracker._get_features(pred_instances, 0)

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
    track_instances = tracker._get_features(pred_instances, 0)
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
    track_instances = tracker._get_features(pred_instances, 0)
    for p, t in zip(pred_instances, track_instances):
        pts = p.numpy()
        centroid = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        assert np.all(centroid == t.feature)
    scores = tracker._get_scores(track_instances)
    assert scores[0, 0] == 0 and scores[1, 1] == 0
    assert scores[1, 0] == scores[0, 1]
    assert math.isclose(scores[0, 1], -130.86877, rel_tol=1e-4)

    # test featuires - bboxes + iou scoring
    tracker = Tracker.from_config(features="bboxes", scoring_method="iou")
    tracked_instances = tracker.track(pred_instances, 0)  # add instances to queue
    track_instances = tracker._get_features(pred_instances, 0)
    for p, t in zip(pred_instances, track_instances):
        pts = p.numpy()
        bbox = (
            np.nanmin(pts[:, 0]),
            np.nanmin(pts[:, 1]),
            np.nanmax(pts[:, 0]),
            np.nanmax(pts[:, 1]),
        )
        assert np.all(bbox == t.feature)
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
