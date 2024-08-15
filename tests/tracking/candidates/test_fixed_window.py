from typing import DefaultDict, Deque
import numpy as np
from sleap_nn.inference.predictors import main
from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates
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


def test_fixed_window_candidates(minimal_instance_ckpt):

    pred_instances = get_pred_instances(minimal_instance_ckpt, 2)
    tracker = Tracker.from_config()
    track_instances = tracker._get_features(pred_instances, 0)

    fixed_window_candidates = FixedWindowCandidates(3)
    assert isinstance(fixed_window_candidates.tracker_queue, Deque)
    fixed_window_candidates.update_candidates(track_instances, None, None)
    # (tracks are assigned only if row/ col ids exists)
    assert not fixed_window_candidates.tracker_queue

    track_instances = fixed_window_candidates.add_new_tracks(track_instances)
    assert len(fixed_window_candidates.tracker_queue) == 1

    new_track_id = fixed_window_candidates.get_new_track_id()
    assert new_track_id == 2
