from typing import DefaultDict, Deque
import numpy as np

from sleap_nn.predict import run_inference
from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates
from sleap_nn.tracking.tracker import Tracker
import torch


def get_pred_instances(
    minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path, n=10
):
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
    for idx, lf in enumerate(result_labels):
        pred_instances.extend(lf.instances)
        if idx == n:
            break
    return pred_instances


def test_local_queues_candidates(
    minimal_instance_centered_instance_ckpt, minimal_instance, tmp_path
):

    pred_instances = get_pred_instances(
        minimal_instance_centered_instance_ckpt,
        minimal_instance,
        n=2,
        tmp_path=tmp_path,
    )
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
