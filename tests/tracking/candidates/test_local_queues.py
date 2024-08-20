from typing import DefaultDict, Deque
import numpy as np

from sleap_nn.inference.predictors import main
from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates
from sleap_nn.tracking.tracker import Tracker


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


def test_local_queues_candidates(minimal_instance_ckpt):

    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config(candidates_method="local_queues")
    track_instances = tracker.get_features(pred_instances, 0)

    local_queues_candidates = LocalQueueCandidates(3, 20)
    assert isinstance(local_queues_candidates.tracker_queue, DefaultDict)
    assert isinstance(local_queues_candidates.tracker_queue[0], Deque)
    local_queues_candidates.update_tracks(track_instances, None, None, None)
    # (tracks are assigned only if row/ col ids exists)
    assert not local_queues_candidates.tracker_queue[0]

    track_instances = local_queues_candidates.add_new_tracks(track_instances)
    assert len(local_queues_candidates.tracker_queue) == 2
    assert len(local_queues_candidates.tracker_queue[0]) == 1
    assert len(local_queues_candidates.tracker_queue[1]) == 1

    new_track_id = local_queues_candidates.get_new_track_id()
    assert new_track_id == 2

    track_instances = tracker.get_features(pred_instances, 0)
    tracked_instances = local_queues_candidates.add_new_tracks(track_instances)
    assert tracked_instances[0].track_id == 2
    assert tracked_instances[1].track_id == 3

    features_track_id = local_queues_candidates.get_features_from_track_id(0)
    assert isinstance(features_track_id, list)
    assert len(features_track_id) == 1
