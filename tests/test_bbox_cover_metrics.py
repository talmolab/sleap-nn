"""Box metrics with ignore regions, and the share of objects covered at fixed false boxes per frame."""

import numpy as np
import pytest

from sleap_nn.config.trainer_config import EvalConfig
from sleap_nn.evaluation import (
    BBOX_COVER_LOG_KEYS,
    compute_bbox_cover_metrics,
    compute_bbox_metrics,
)


def _frame(gt, pred, scores, ignore=None):
    f = dict(
        gt_bboxes=np.asarray(gt, float).reshape(-1, 4),
        gt_classes=np.zeros(len(gt), int),
        pred_bboxes=np.asarray(pred, float).reshape(-1, 4),
        pred_scores=np.asarray(scores, float),
        pred_classes=np.zeros(len(pred), int),
    )
    if ignore is not None:
        f["gt_ignore_bboxes"] = np.asarray(ignore, float).reshape(-1, 4)
    return f


def test_ignore_regions_drop_boxes_on_ignored_objects():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [50, 50, 60, 60]]  # a hit, and a box on an unlabeled object
    # the box on the unlabeled object outranks the hit: it costs AP unless ignored
    plain = compute_bbox_metrics([_frame(gt, pred, [0.8, 0.9])], num_classes=1)
    ign = compute_bbox_metrics(
        [_frame(gt, pred, [0.8, 0.9], ignore=[[52, 52, 58, 58]])], num_classes=1
    )
    assert plain["bbox_AP50"] < 1.0 and ign["bbox_AP50"] == pytest.approx(1.0)
    assert plain["n_pred"] == 2 and ign["n_pred"] == 1
    assert ign["n_gt"] == plain["n_gt"] == 1
    # a box that matches a kept GT is never dropped, even if it covers an ignore region
    keep = compute_bbox_metrics(
        [_frame(gt, [[0, 0, 10, 10]], [0.9], ignore=[[2, 2, 5, 5]])], num_classes=1
    )
    assert keep["bbox_n_tp"] == 1


def test_frames_without_ignore_key_are_unchanged():
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(15):
        g = rng.uniform(0, 100, (2, 2))
        gb = np.c_[g, g + 20]
        frames.append(_frame(gb, gb + rng.normal(0, 4, gb.shape), rng.uniform(0, 1, 2)))
    a = compute_bbox_metrics(frames, num_classes=1)
    b = compute_bbox_metrics(
        [dict(f, gt_ignore_bboxes=np.zeros((0, 4))) for f in frames], num_classes=1
    )
    for k in ("bbox_mAP", "bbox_AP50", "bbox_recall", "bbox_n_fp"):
        assert a[k] == b[k]


def test_cover_rule_counts_large_boxes_on_thin_labels():
    # a thin label box (two keypoints) and a whole-mouse prediction: IoU < 0.5, cover = 1
    gt = [[0, 0, 40, 4]]
    pred = [[-5, -10, 45, 14]]
    m = compute_bbox_metrics([_frame(gt, pred, [0.9])], num_classes=1)
    c = compute_bbox_cover_metrics([_frame(gt, pred, [0.9])])
    assert m["bbox_n_tp"] == 0
    assert c["bbox_cover_recall"] == 1.0 and c["bbox_false_per_frame"] == 0.0


def test_cover_recall_at_fixed_false_boxes_per_frame():
    # 4 frames, one object each; scores: hits 0.9, 0.8, 0.3, 0.2; false boxes 0.6, 0.5
    frames = [
        _frame([[0, 0, 10, 10]], [[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.6]),
        _frame([[0, 0, 10, 10]], [[0, 0, 10, 10], [50, 50, 60, 60]], [0.8, 0.5]),
        _frame([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.3]),
        _frame([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.2]),
    ]
    c = compute_bbox_cover_metrics(
        frames, false_per_frame=(0.0, 0.25, 0.5), score_threshold=0.4
    )
    assert c["bbox_cover_recall_at_0.0fppf"] == pytest.approx(
        0.5
    )  # two hits before the first false box
    assert c["bbox_cover_recall_at_0.25fppf"] == pytest.approx(
        0.5
    )  # one false box allowed; next is false too
    assert c["bbox_cover_recall_at_0.5fppf"] == pytest.approx(1.0)
    assert c["bbox_cover_recall"] == pytest.approx(0.5) and c[
        "bbox_false_per_frame"
    ] == pytest.approx(0.5)
    assert 0.0 < c["bbox_cover_AP"] <= 1.0
    # a duplicate on a covered object is a false box
    d = compute_bbox_cover_metrics(
        [_frame([[0, 0, 10, 10]], [[0, 0, 10, 10], [1, 1, 10, 10]], [0.9, 0.8])]
    )
    assert d["bbox_cover_recall"] == 1.0 and d["bbox_false_per_frame"] == 1.0


def test_cover_metrics_config_and_keys():
    assert EvalConfig().bbox_cover_metrics is False
    assert EvalConfig(bbox_cover_metrics=True).bbox_cover_metrics is True
    c = compute_bbox_cover_metrics([_frame([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.9])])
    assert set(BBOX_COVER_LOG_KEYS) <= set(c)


def test_bbox_callback_logs_cover_keys_only_when_enabled():
    from sleap_nn.training.callbacks import BBoxEvaluationCallback

    assert BBoxEvaluationCallback().cover_metrics is False
    assert BBoxEvaluationCallback(cover_metrics=True).cover_metrics is True
