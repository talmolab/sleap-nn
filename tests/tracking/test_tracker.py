import pytest
from sleap_nn.tracking.tracker import Tracker
from sleap_nn.inference.predictors import main
import numpy as np


def get_pred_instances(minimal_instance_ckpt, n=10):
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
    # Test for invalid candidates method
    with pytest.raises(Exception):
        tracker = Tracker.from_config(
            max_tracks=30, window_size=10, candidates_method="tracking"
        )

    # Test basic tracker: pose as feature, oks scoring method
    # Test _get_features(): points as feature
    tracker = Tracker.from_config()
    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
    track_instances = tracker._get_features(pred_instances)
    for p, t in zip(pred_instances, track_instances):
        assert np.all(p.numpy() == t.feature)

    # Test for the first two instances (tracks assigned to each of the new instances)
    for p in pred_instances:
        assert p.track is None
    tracked_instances = tracker.track(pred_instances)
    for t in tracked_instances:
        assert t.track is not None

    # Test _get_scores()
    scores = tracker._get_scores(track_instances)
    assert np.all(scores == np.array([[1.0, 0], [0, 1.0]]))

    # Test _assign_tracks()
    track_instances = tracker._assign_tracks(track_instances, -scores)
    assert track_instances[0].track_id == 0 and track_instances[1].track_id == 1
    assert np.all(track_instances[0].feature == pred_instances[0].numpy())

    # Test track() with track_queue not empty
    tracked_instances = tracker.track(pred_instances)
    assert len(tracker.candidates.tracker_queue[0]) == 2
    assert np.all(
        tracker.candidates.tracker_queue[0][0].feature
        == tracker.candidates.tracker_queue[0][1].feature
    )

    # test maxlen in fixed window approach??
