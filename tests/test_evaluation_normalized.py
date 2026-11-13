"""Point error relative to instance size, with unpredicted points counted as failures."""

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import OmegaConf

from sleap_nn.config.trainer_config import EvalConfig
from sleap_nn.evaluation import (
    Evaluator,
    _weighted_inverted_cdf,
    instance_scales,
    normalized_point_errors,
    run_evaluation,
)


def _labels(with_box=True):
    """Two frames, one mouse each; GT 3 nodes; frame 0 has a 40 x 10 user box.

    Frame 0 prediction: node errors 2, 0, 4 px. Frame 1 prediction: node 0 off by 3 px,
    node 1 unpredicted (NaN), node 2 not labeled in GT (not scored).
    """
    skel = sio.Skeleton(["a", "b", "c"])
    video = sio.Video(filename="toy.mp4")
    g0 = sio.Instance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), skeleton=skel
    )
    g1 = sio.Instance.from_numpy(
        np.array([[0.0, 0.0], [0.0, 30.0], [np.nan, np.nan]]), skeleton=skel
    )
    f0 = sio.LabeledFrame(video=video, frame_idx=0, instances=[g0])
    if with_box:
        f0.bboxes = [sio.UserBoundingBox.from_xyxy(-10.0, -5.0, 30.0, 5.0, instance=g0)]
    f1 = sio.LabeledFrame(video=video, frame_idx=1, instances=[g1])
    p0 = sio.PredictedInstance.from_numpy(
        np.array([[2.0, 0.0], [10.0, 0.0], [20.0, 4.0]]),
        skeleton=skel,
        point_scores=np.ones(3),
        score=1.0,
    )
    p1 = sio.PredictedInstance.from_numpy(
        np.array([[3.0, 0.0], [np.nan, np.nan], [5.0, 5.0]]),
        skeleton=skel,
        point_scores=np.ones(3),
        score=1.0,
    )
    gt = sio.Labels([f0, f1], videos=[video], skeletons=[skel])
    pr = sio.Labels(
        [
            sio.LabeledFrame(video=video, frame_idx=0, instances=[p0]),
            sio.LabeledFrame(video=video, frame_idx=1, instances=[p1]),
        ],
        videos=[video],
        skeletons=[skel],
    )
    return gt, pr


def test_weighted_inverted_cdf_matches_numpy():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 10, 101):
        x = rng.normal(size=n)
        for q in (0.5, 0.95):
            assert _weighted_inverted_cdf(x, np.ones(n), q) == np.percentile(
                x, 100 * q, method="inverted_cdf"
            )
    # weights act as repeat counts
    assert (
        _weighted_inverted_cdf(
            np.array([1.0, 2.0, 3.0]), np.array([0.0, 3.0, 1.0]), 0.5
        )
        == 2.0
    )
    assert np.isnan(_weighted_inverted_cdf(np.zeros(0), np.zeros(0), 0.5))


def test_instance_scales_box_then_points():
    gt, pr = _labels()
    ev = Evaluator(gt, pr)
    sc = instance_scales(ev.positive_pairs, gt, "bbox")
    by_frame = {p[0].frame_idx: s for p, s in zip(ev.positive_pairs, sc)}
    assert by_frame[0] == pytest.approx(np.sqrt(40 * 10))  # the linked user box
    assert by_frame[1] == pytest.approx(0.0, abs=1e-9) or np.isnan(
        by_frame[1]
    )  # 2 collinear points: zero area
    sc_pts = instance_scales(ev.positive_pairs, gt, "points", pad_frac=0.1)
    by_frame = {p[0].frame_idx: s for p, s in zip(ev.positive_pairs, sc_pts)}
    assert by_frame[0] == pytest.approx(
        np.sqrt((20 + 4) * (0 + 4))
    )  # 20 x 0 box padded by 2 px per side
    with pytest.raises(ValueError):
        instance_scales(ev.positive_pairs, gt, "area")


def test_missing_points_count_as_failures():
    gt, pr = _labels()
    ev = Evaluator(gt, pr)
    sc = np.array([10.0] * len(ev.positive_pairs))
    pe = normalized_point_errors(ev.positive_pairs, sc, missing_error=1.0)
    errs = sorted(pe["err"].tolist())
    # frame 0: 0.2, 0.0, 0.4; frame 1: 0.3 and the unpredicted node -> 1.0; unlabeled node not scored
    assert errs == pytest.approx([0.0, 0.2, 0.3, 0.4, 1.0])
    assert pe["found"].sum() == 4


def test_normalized_distance_metrics_values_and_bootstrap():
    gt, pr = _labels()
    ev = Evaluator(gt, pr, norm_scale="points", norm_pad_frac=0.5)
    m = ev.normalized_distance_metrics()
    # frame 0 points box 20 x 0 padded by 10 per side -> 40 x 20; frame 1 box 0 x 30 padded 15 -> 30 x 60
    s0, s1 = np.sqrt(40 * 20), np.sqrt(30 * 60)
    errs = np.array([2 / s0, 0.0, 4 / s0, 3 / s1, 1.0])
    assert m["n_points"] == 5 and m["n_instances"] == 2
    assert m["p50"] == pytest.approx(np.percentile(errs, 50, method="inverted_cdf"))
    assert m["found_p50"] == pytest.approx(
        np.percentile(errs[:4], 50, method="inverted_cdf")
    )
    assert m["miss_rate"] == pytest.approx(0.2)
    b1 = ev.normalized_distance_metrics(bootstrap=50, seed=3)
    b2 = ev.normalized_distance_metrics(bootstrap=50, seed=3)
    assert b1["ci"] == b2["ci"]
    lo, hi = b1["ci"]["p50"]
    assert lo <= m["p50"] <= hi
    assert (
        ev.normalized_distance_metrics(bootstrap=20, bootstrap_by="video")[
            "bootstrap_by"
        ]
        == "video"
    )


def test_evaluate_output_unchanged_by_default():
    gt, pr = _labels()
    base = Evaluator(gt, pr).evaluate()
    assert "normalized_distance_metrics" not in base
    on = Evaluator(gt, pr, norm_scale="bbox").evaluate()
    assert set(on) - set(base) == {"normalized_distance_metrics"}
    for group in base:
        for k, v in base[group].items():
            if isinstance(v, (float, int)) and np.isfinite(v):
                assert on[group][k] == v
    with pytest.raises(ValueError):
        Evaluator(gt, pr, norm_scale="area")


def test_run_evaluation_norm_option(tmp_path):
    gt, pr = _labels()
    gp, pp = tmp_path / "gt.slp", tmp_path / "pr.slp"
    gt.save(str(gp))
    pr.save(str(pp))
    assert "normalized_distance_metrics" not in run_evaluation(str(gp), str(pp))
    m = run_evaluation(
        str(gp), str(pp), norm_scale="points", norm_pad_frac=0.5, norm_bootstrap=10
    )
    assert m["normalized_distance_metrics"]["miss_rate"] == pytest.approx(0.2)
    assert "ci" in m["normalized_distance_metrics"]


def test_eval_config_norm_fields():
    c = EvalConfig()
    assert (
        c.norm_scale is None and c.norm_missing_error == 1.0 and c.norm_bootstrap == 0
    )
    assert EvalConfig(norm_scale="bbox", norm_bootstrap_by="video").norm_scale == "bbox"
    with pytest.raises(ValueError):
        EvalConfig(norm_scale="area")
    with pytest.raises(ValueError):
        EvalConfig(norm_bootstrap_by="session")


def test_post_training_overrides_forward_norm_only_when_set():
    from sleap_nn.train import _oks_run_evaluation_overrides

    off = OmegaConf.create({"trainer_config": {"eval": {"oks_stddev": 0.025}}})
    assert not any(k.startswith("norm_") for k in _oks_run_evaluation_overrides(off))
    on = OmegaConf.create(
        {
            "trainer_config": {
                "eval": {
                    "norm_scale": "bbox",
                    "norm_bootstrap": 100,
                    "norm_pad_frac": 0.1,
                }
            }
        }
    )
    ov = _oks_run_evaluation_overrides(on)
    assert (
        ov["norm_scale"] == "bbox"
        and ov["norm_bootstrap"] == 100
        and ov["norm_pad_frac"] == 0.1
    )


def test_epoch_end_callback_logs_norm_only_when_enabled():
    from sleap_nn.training.callbacks import EpochEndEvaluationCallback

    skel = sio.Skeleton(["a", "b"])
    off = EpochEndEvaluationCallback(skeleton=skel, videos=[])
    assert not any(k.startswith("norm/") for k in off.metrics_to_log)
    on = EpochEndEvaluationCallback(
        skeleton=skel, videos=[], norm_scale="points", norm_pad_frac=0.1
    )
    assert {"norm/p50", "norm/p95", "norm/miss_rate"} <= set(on.metrics_to_log)
