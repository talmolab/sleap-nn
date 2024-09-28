from typing import DefaultDict, Deque, List
import numpy as np
from sleap_nn.inference.predictors import main
from sleap_nn.tracking.track_instance import TrackedInstanceFeature
from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates
from sleap_nn.tracking.tracker import Tracker


def get_pred_instances(minimal_instance_ckpt, n=10):
    result_labels = main(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelReader",
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


def test_fixed_window_candidates(minimal_instance_ckpt):

    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config()
    track_instances = tracker.get_features(pred_instances, 0)

    fixed_window_candidates = FixedWindowCandidates(3)
    assert isinstance(fixed_window_candidates.tracker_queue, Deque)
    fixed_window_candidates.update_tracks(track_instances, None, None, None)
    # (tracks are assigned only if row/ col ids exists)
    assert not fixed_window_candidates.tracker_queue

    track_instances = fixed_window_candidates.add_new_tracks(track_instances)
    assert len(fixed_window_candidates.tracker_queue) == 1

    new_track_id = fixed_window_candidates.get_new_track_id()
    assert new_track_id == 2

    track_instances = tracker.get_features(pred_instances, 0)
    tracked_instances = fixed_window_candidates.add_new_tracks(track_instances)
    assert tracked_instances.track_ids == [2, 3]
    assert tracked_instances.tracking_scores == [1.0, 1.0]

    features_track_id = fixed_window_candidates.get_features_from_track_id(0)
    assert isinstance(features_track_id, list)
    assert len(features_track_id) == 1
