import numpy as np
from typing import List, Tuple
import sleap_io as sio
import pytest
from pathlib import Path
import copy
import torch
import warnings
from sleap_nn.legacy_predict import run_inference
from sleap_nn.evaluation import (
    compute_instance_area,
    compute_oks,
)
from sleap_nn.evaluation import Evaluator, load_metrics
from loguru import logger
import sys
from sleap_nn.train import train
from _pytest.logging import LogCaptureFixture


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


def test_compute_oks():
    # Test compute_oks function with the cocoutils implementation
    inst_gt = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")

    # full-match, oks should be 1
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    # with one nan predicted instance
    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 2 / 3)

    # one additional predicted instance not in ground truth instance
    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    # both gt and pred instances having nan values
    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    # Test compute_oks function with the implementation from the paper
    inst_gt = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr, False)
    np.testing.assert_allclose(oks, 1)

    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr, False)
    np.testing.assert_allclose(oks, 2 / 3)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr, use_cocoeval=False)
    np.testing.assert_allclose(oks, 1)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr, use_cocoeval=False)
    np.testing.assert_allclose(oks, 1)


def create_labels_two_match_one_missed_inst(minimal_instance):
    # two match instances and one missed user instance

    # Create skeleton.
    skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )

    # Get video.
    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # Create user labelled instance.
    user_inst_1 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [11.4, 13.4],
                [13.6, 15.1],
                [0.3, 9.3],
            ]
        ),
        skeleton=skeleton,
    )

    # Create Predicted Instance.
    pred_inst_1 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [11.2, 17.4],
                [12.8, 15.1],
                [0.3, 10.6],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.8]),
        score=0.7,
    )

    # create second user instance
    user_inst_2 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [1.4, 2.9],
                [30.6, 9.5],
                [40.6, 60.7],
            ]
        ),
        skeleton=skeleton,
    )

    pred_inst_2 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [2.3, 2.2],
                [25.6, 10.0],
                [37.6, np.nan],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        score=0.6,
    )

    # create a user instance which shouldn't be matched with other predicted instances
    user_inst_3 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [55.6, 30.2],
                [10.1, 18.5],
                [35.8, 12.0],
            ]
        ),
        skeleton=skeleton,
    )

    # Create labeled frame.
    user_lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_inst_1, user_inst_2, user_inst_3, pred_inst_1],
    )
    # Create ground-truth labels.
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf]
    )

    # Create predicted labels.
    pred_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[pred_inst_1, pred_inst_2]
    )
    pred_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf]
    )

    return user_labels, pred_labels


def test_evaluator_two_match_one_missed_inst(minimal_instance):
    # two match instances and one missed user instance

    user_labels, pred_labels = create_labels_two_match_one_missed_inst(minimal_instance)

    eval = Evaluator(user_labels, pred_labels)

    # test _process_frames function. One user instance should be missed.
    assert len(eval.frame_pairs) == 1
    assert len(eval.positive_pairs) == 2
    assert len(eval.false_negatives) == 1

    gt_1, pred_1, _ = eval.positive_pairs[0]
    gt_3 = eval.false_negatives[0]

    points_gt = np.array(
        [
            [11.4, 13.4],
            [13.6, 15.1],
            [0.3, 9.3],
        ]
    )

    points_pred = np.array(
        [
            [11.2, 17.4],
            [12.8, 15.1],
            [0.3, 10.6],
        ]
    )

    # test if the first user labeled instance is matched with the first predicted instance
    assert (gt_1.instance.numpy() == points_gt).all()
    assert (pred_1.instance.numpy() == points_pred).all()

    # test if the false negative instance is the last predicted instance
    points = np.array(
        [
            [55.6, 30.2],
            [10.1, 18.5],
            [35.8, 12.0],
        ]
    )
    assert (gt_3.instance.numpy() == points).all()


def create_labels_no_match_frame_pairs(minimal_instance):
    """Create labels with no matching frame pairs.

    The ground truth has frame_idx=0, but predictions have frame_idx=999,
    so even though videos match (via sleap-io's robust matching), no frames
    will overlap.
    """
    # Create skeleton.
    skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )

    # Get video.
    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # Create user labelled instance.
    user_inst_1 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [11.4, 13.4],
                [13.6, 15.1],
                [0.3, 9.3],
            ]
        ),
        skeleton=skeleton,
    )

    # Create Predicted Instance.
    pred_inst_1 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [11.2, 17.4],
                [12.8, 15.1],
                [0.3, 10.6],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.8]),
        score=0.7,
    )

    # Ground truth at frame_idx=0
    user_lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_inst_1],
    )
    # create labels object
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf]
    )

    # Predictions at frame_idx=999 (no overlap with GT frames)
    # This ensures no frame pairs can be matched even if videos match
    pred_lf = sio.LabeledFrame(video=video, frame_idx=999, instances=[pred_inst_1])

    # create labels object for predicted labeled frames
    pred_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf]
    )

    return user_labels, pred_labels


def test_evaluator_no_match_frame_pairs(caplog, minimal_instance):
    # with no match frame pairs
    user_labels, pred_labels = create_labels_no_match_frame_pairs(minimal_instance)
    with caplog.at_level("ERROR"):  # Set the log level to capture ERROR messages
        with pytest.raises(Exception):
            eval = Evaluator(user_labels, pred_labels)
    assert "Empty Frame Pairs. No match found for the video frames" in caplog.text


def create_labels_more_predicted_instances(minimal_instance):
    # with more predicted instances than user labeled instances
    # one user lf with no match frame pair in predicted lf

    # Create skeleton.
    skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )

    # Get video.
    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # create a copy of the video
    video1 = copy.deepcopy(video)
    video1.filename = "test.mp4"

    # Create user labelled instance.
    user_inst_1 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [11.4, 13.4],
                [13.6, 15.1],
                [0.3, 9.3],
            ]
        ),
        skeleton=skeleton,
    )

    # create predicted instance
    pred_inst_1 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [11.2, 17.4],
                [12.8, 13.1],
                [0.7, 10.0],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.8]),
        score=0.8,
    )

    # create second user instance
    user_inst_2 = sio.Instance.from_numpy(
        points_data=np.array(
            [
                [1.4, 2.9],
                [30.6, 9.5],
                [40.6, 60.7],
            ]
        ),
        skeleton=skeleton,
    )

    # create second predicted instance
    pred_inst_2 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [1.3, 2.9],
                [29.6, 9.2],
                [39.6, 59.3],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        score=0.7,
    )

    # create a predicted instance with nan values
    pred_inst_3 = sio.PredictedInstance.from_numpy(
        points_data=np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        score=0.7,
    )

    # create labeled frame with the instances
    user_lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_inst_2, user_inst_1],
    )

    # create labeled frame object with different frame index
    user_lf_1 = sio.LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[user_inst_2, user_inst_1],
    )

    # create ground-truth labels object
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf, user_lf_1]
    )

    pred_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[pred_inst_2, pred_inst_1, pred_inst_3]
    )

    # create a single pred labeled frame
    pred_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf]
    )

    return user_labels, pred_labels


def test_evaluator_more_predicted_instances(minimal_instance):
    # with more predicted instances than user labeled instances
    # one user lf with no match frame pair in predicted lf

    user_labels, pred_labels = create_labels_more_predicted_instances(minimal_instance)

    eval = Evaluator(user_labels, pred_labels)
    # there should be exactly 2 matching instances for the first userlf and pred lf.
    # The second user lf should be ignored as the frame index is different.
    # third predicted instance with all nans should be ignored
    assert len(eval.frame_pairs) == 1
    assert len(eval.positive_pairs) == 2
    assert len(eval.false_negatives) == 0

    # test voc with no false negative instances and to test the strictly decreasing sorting of precisions
    eval = Evaluator(user_labels, pred_labels)
    voc = eval.voc_metrics(match_score_by="oks")
    assert np.abs(voc["oks_voc.recalls"][0] - 0.5) <= 1e-5

    # test match_instances function for all oks values lower than the threshold. There shouldn't be any match instances
    eval = Evaluator(user_labels, pred_labels, match_threshold=1)
    assert len(eval.frame_pairs) == 1
    assert len(eval.positive_pairs) == 0
    assert len(eval.false_negatives) == 2


def test_evaluator_zero_matched_instances_no_warnings(caplog, minimal_instance):
    """Evaluator.evaluate() with 0 positive pairs must not warn (#719).

    Regression test: a collapsed model (or here, a match_threshold strict
    enough that nothing clears it) produces 0 positive pairs but non-empty
    frame_pairs/false_negatives. mOKS()/pck_metrics()/voc_metrics(pck) used to
    call .mean() on empty arrays unguarded, spamming "Mean of empty slice"
    RuntimeWarnings. This checks the fix: no warnings, one clear log line, and
    well-defined NaN/0 metrics.
    """
    user_labels, pred_labels = create_labels_more_predicted_instances(minimal_instance)
    eval = Evaluator(user_labels, pred_labels, match_threshold=1)
    assert len(eval.positive_pairs) == 0
    assert len(eval.false_negatives) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level("INFO"):
            metrics = eval.evaluate()

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, [str(w.message) for w in runtime_warnings]
    assert "0 matched instances" in caplog.text

    assert np.isnan(metrics["mOKS"]["mOKS"])
    assert np.isnan(metrics["distance_metrics"]["avg"])
    assert np.isnan(metrics["pck_metrics"]["mPCK"])
    assert np.isnan(metrics["pck_metrics"]["PCK@5"])
    assert metrics["voc_metrics"]["oks_voc.mAP"] == 0
    assert metrics["voc_metrics"]["pck_voc.mAP"] == 0


def test_find_frame_pairs_does_not_mutate_gt_labels():
    """``user_labels_only=True`` must not mutate the caller's GT ``Labels``.

    Regression test: ``find_frame_pairs`` used to do ``lf.instances =
    lf.user_instances`` directly on the ``LabeledFrame`` objects returned by
    ``labels_gt.find(...)``, which are references into the caller's actual
    ``Labels`` object (not copies) -- permanently discarding any
    ``PredictedInstance``s from the real ground-truth object. A second
    ``Evaluator`` built from the same ``labels_gt`` afterward (e.g. with
    ``user_labels_only=False``) would then silently see fewer instances than
    it should.
    """
    skel = sio.Skeleton(nodes=["a", "b"])
    video = sio.Video(filename="dummy.mp4")

    user_inst = sio.Instance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), skeleton=skel
    )
    pred_inst_gt = sio.PredictedInstance.from_numpy(
        points_data=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        skeleton=skel,
        score=0.9,
    )
    lf_gt = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst, pred_inst_gt]
    )
    labels_gt = sio.Labels(videos=[video], skeletons=[skel], labeled_frames=[lf_gt])

    pred_inst_pr = sio.PredictedInstance.from_numpy(
        points_data=np.array([[0.1, 0.1], [1.1, 1.1]], dtype=np.float32),
        skeleton=skel,
        score=0.95,
    )
    lf_pr = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst_pr])
    labels_pr = sio.Labels(videos=[video], skeletons=[skel], labeled_frames=[lf_pr])

    assert len(labels_gt.labeled_frames[0].instances) == 2

    Evaluator(labels_gt, labels_pr, user_labels_only=True, match_threshold=100.0)

    # The real GT Labels object must be untouched -- both the user and the
    # predicted instance are still there.
    assert len(labels_gt.labeled_frames[0].instances) == 2


def test_evaluator_metrics(minimal_instance):
    user_labels, pred_labels = create_labels_two_match_one_missed_inst(minimal_instance)
    eval = Evaluator(user_labels, pred_labels)

    # test the compute_instance_area function by computing the area of the bounding box from the instance points.
    user_lf = user_labels[0]
    points_gt = user_lf.numpy()[0]
    area = compute_instance_area(points_gt)
    area[0] == 77.14

    # test compute_dists function which computes the norm of the distance between the two instances.
    # nan values in the instance points should be retained as nan
    dist_dict = eval.dists_dict
    dists = dist_dict["dists"][0]
    calc_dist = np.array([[4.0049968, 0.8, 1.3], [1.140175, 5.024937, np.nan]])
    assert (np.abs(np.array(dists) - calc_dist[0]) <= 1e-5).all()
    dists = np.array(dist_dict["dists"][1])
    assert (np.abs(dists[:-1] - calc_dist[1][:-1]) <= 1e-5).all()
    assert np.isnan(dists[-1])

    # test visibility_metrics function.
    viz_metrics = eval.visibility_metrics()
    assert viz_metrics["precision"] == float(1)
    assert abs(viz_metrics["recall"] - float(0.833333)) <= 1e-5

    # test distance_metrics. The nan values should be ignored while computing the percentiles
    dist_metrics = eval.distance_metrics()
    assert np.abs(dist_metrics["avg"] - 2.4540217) <= 1e-5
    non_nans = np.array([4.0049968, 0.8, 1.3, 1.140175, 5.024937])
    assert dist_metrics["p90"] - np.percentile(non_nans, 90) <= 1e-5

    # test pck metrics
    pck = eval.pck_metrics()
    assert np.abs(pck["mPCK"] - 0.65) <= 1e-5

    # test voc_metrics
    # test the metrics computation with pck
    voc = eval.voc_metrics(match_score_by="pck")
    assert np.abs(voc["pck_voc.recalls"][0] - 0.3333333) <= 1e-5
    prec = np.zeros((101,))
    prec[:34] = float(1) - np.spacing(1)
    assert (voc["pck_voc.precisions"][0] == prec).all()

    # test the metrics computation with oks
    voc = eval.voc_metrics(match_score_by="oks")
    assert np.abs(voc["oks_voc.recalls"][0] - 0.0) <= 1e-5

    # test the input to match_score_by parameter. voc_metrics only accepts oks or pck
    with pytest.raises(
        Exception,
    ):
        eval.voc_metrics(match_score_by="moks")

    # test mOKS which should be the average of the oks values for each positive pairs
    meanOKS_calc = (0.33308048 + 0.067590989) // 2
    assert int(eval.mOKS()["mOKS"]) == meanOKS_calc


def test_evaluator_main(
    minimal_instance,
    tmp_path,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
):
    output = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        max_instances=6,
        output_path=f"{tmp_path}/test.slp",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    import subprocess

    # Build the command to run sleap-nn eval with the required arguments
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--no-group",
        "gpu",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "eval",
        "--ground_truth_path",
        minimal_instance.as_posix(),
        "--predicted_path",
        f"{tmp_path}/test.slp",
        "--save_metrics",
        f"{tmp_path}/metrics_test.npz",
    ]
    # Run the command and check for errors
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/metrics_test.npz").exists()

    # Load metrics in SLEAP 1.4 format (single "metrics" key)
    metrics_npz = np.load(f"{tmp_path}/metrics_test.npz", allow_pickle=True)
    assert "metrics" in metrics_npz
    metrics = metrics_npz["metrics"].item()
    assert "voc_metrics" in metrics
    assert "mOKS" in metrics
    assert "distance_metrics" in metrics
    assert "pck_metrics" in metrics
    assert "visibility_metrics" in metrics
    voc_metrics = metrics["voc_metrics"]
    assert "pck_voc.mAP" in voc_metrics
    assert "pck_voc.mAR" in voc_metrics
    assert "oks_voc.mAP" in voc_metrics
    assert "oks_voc.mAR" in voc_metrics


# def test_evaluator_logging_empty_frame_pairs(capsys, minimal_instance):
#     """Test that the Evaluator logs an error when there are no matching frame pairs."""

#     # logger.remove()
#     # logger.add(sys.stderr, level="ERROR")
#     # Create user_labels and pred_labels that will lead to empty frame pairs
#     user_labels, pred_labels = create_labels_no_match_frame_pairs(minimal_instance)

#     # Use capsys to capture output
#     with capsys.disabled():  # Disable capturing to see print statements if needed
#         with pytest.raises(Exception):
#             eval = Evaluator(user_labels, pred_labels)
#             eval.voc_metrics(match_score_by="invalid_option")  # This should trigger the error

#     # Capture the output
#     out, err = capsys.readouterr()


#     # Check that the expected log message was captured in standard error
#     assert "Empty Frame Pairs. No match found for the video frames" in err
def test_evaluator_logging_empty_frame_pairs(caplog, minimal_instance):
    """Test that the Evaluator logs an error when there are no matching frame pairs."""
    # Create user_labels and pred_labels that will lead to empty frame pairs
    user_labels, pred_labels = create_labels_no_match_frame_pairs(minimal_instance)

    # Use caplog to capture output
    with caplog.at_level("ERROR"):  # Set the log level to capture ERROR messages
        with pytest.raises(Exception):
            eval = Evaluator(user_labels, pred_labels)
            eval.voc_metrics(
                match_score_by="invalid_option"
            )  # This should trigger the error

    # Check that the expected log message was captured
    assert "Empty Frame Pairs. No match found for the video frames" in caplog.text


def test_load_metrics(single_instance_with_metrics_ckpt, tmp_path):
    """Test load_metrics function."""
    # Test top-level import
    from sleap_nn import load_metrics as load_metrics_top

    assert load_metrics_top is load_metrics

    # Test with model folder (old naming format: {split}_{idx}_pred_metrics.npz)
    metrics = load_metrics(single_instance_with_metrics_ckpt, split="train")
    assert "voc_metrics" in metrics
    assert "mOKS" in metrics
    assert "distance_metrics" in metrics
    assert "pck_metrics" in metrics
    assert "visibility_metrics" in metrics

    # Test with direct .npz file path
    metrics = load_metrics(
        single_instance_with_metrics_ckpt / "train_0_pred_metrics.npz"
    )
    assert "voc_metrics" in metrics
    assert "mOKS" in metrics

    # Test with invalid path
    with pytest.raises(FileNotFoundError):
        load_metrics(Path(tmp_path) / "test_load_metrics" / "invalid.npz")

    # Test new format (single "metrics" key)
    new_format_dir = tmp_path / "new_format_model"
    new_format_dir.mkdir()
    test_metrics = {
        "voc_metrics": {"oks_voc.mAP": 0.5},
        "mOKS": {"mOKS": 0.8},
        "distance_metrics": {"avg": 2.5},
        "pck_metrics": {"mPCK": 0.9},
        "visibility_metrics": {"precision": 0.95, "recall": 0.92},
    }
    np.savez_compressed(
        new_format_dir / "metrics.val.0.npz", **{"metrics": test_metrics}
    )
    loaded = load_metrics(new_format_dir, split="val")
    assert loaded["mOKS"]["mOKS"] == 0.8
    assert loaded["voc_metrics"]["oks_voc.mAP"] == 0.5

    # Test test->val fallback (no test metrics, should fall back to val)
    loaded_fallback = load_metrics(new_format_dir, split="test")
    assert loaded_fallback["mOKS"]["mOKS"] == 0.8

    # Test dataset_idx parameter
    np.savez_compressed(
        new_format_dir / "metrics.val.1.npz",
        **{
            "metrics": {
                "mOKS": {"mOKS": 0.7},
                **{k: {} for k in test_metrics if k != "mOKS"},
            }
        },
    )
    loaded_idx1 = load_metrics(new_format_dir, split="val", dataset_idx=1)
    assert loaded_idx1["mOKS"]["mOKS"] == 0.7

    # Test old format (individual keys at top level)
    old_format_dir = tmp_path / "old_format_model"
    old_format_dir.mkdir()
    np.savez_compressed(
        old_format_dir / "val_0_pred_metrics.npz",
        voc_metrics={"oks_voc.mAP": 0.6},
        mOKS={"mOKS": 0.75},
        distance_metrics={"avg": 3.0},
        pck_metrics={"mPCK": 0.85},
        visibility_metrics={"precision": 0.9},
    )
    loaded_old = load_metrics(old_format_dir, split="val")
    assert loaded_old["mOKS"]["mOKS"] == 0.75
    assert loaded_old["voc_metrics"]["oks_voc.mAP"] == 0.6


def _representative_metrics():
    """A metrics dict mirroring ``Evaluator.evaluate()`` (OKS mode) output.

    Uses numpy scalars/arrays and embedded NaNs to exercise the JSON-safe
    conversion the same way a real ``run_evaluation`` result would.
    """
    return {
        "voc_metrics": {
            "oks_voc.match_score_thresholds": np.linspace(0.5, 0.95, 10),
            "oks_voc.recall_thresholds": np.linspace(0, 1, 101),
            "oks_voc.match_scores": np.array([0.9, 0.7, 0.3]),
            "oks_voc.precisions": np.ones((10, 101)),
            "oks_voc.recalls": np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0, 0, 0, 0]),
            "oks_voc.AP": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0]),
            "oks_voc.AR": np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0, 0, 0, 0]),
            "oks_voc.mAP": np.float64(0.235),
            "oks_voc.mAR": np.float64(0.32),
            "pck_voc.match_score_thresholds": np.linspace(0.5, 0.95, 10),
            "pck_voc.recall_thresholds": np.linspace(0, 1, 101),
            "pck_voc.match_scores": np.array([0.95, 0.75, 0.35]),
            "pck_voc.precisions": np.ones((10, 101)),
            "pck_voc.recalls": np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0, 0, 0, 0]),
            "pck_voc.AP": np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0]),
            "pck_voc.AR": np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0, 0, 0, 0]),
            "pck_voc.mAP": np.float64(0.28),
            "pck_voc.mAR": np.float64(0.38),
        },
        "mOKS": {"mOKS": np.float64(0.6543)},
        "distance_metrics": {
            "frame_idxs": [0, 1],
            "video_paths": ["/data/vid.mp4", "/data/vid.mp4"],
            # (n_pairs, n_nodes) with a missing node (NaN) -> must become null.
            "dists": np.array([[1.5, np.nan], [2.0, 3.0]]),
            "avg": np.float64(2.1667),
            "p50": np.float64(2.0),
            "p75": np.float64(2.5),
            "p90": np.float64(2.8),
            "p95": np.float64(2.9),
            "p99": np.float64(2.99),
        },
        "pck_metrics": {
            "thresholds": np.linspace(1, 10, 10),
            "pcks": np.ones((2, 2, 10), dtype=bool),
            "mPCK_parts": np.array([0.5, 0.5]),
            "mPCK": np.float64(0.5),
            "PCK@5": np.float64(0.75),
            "PCK@10": np.float64(0.9),
        },
        "visibility_metrics": {
            "tp": np.int64(3),
            "fp": np.int64(1),
            "tn": np.int64(0),
            "fn": np.int64(1),
            "precision": np.float64(0.75),
            "recall": np.float64(0.75),
        },
    }


def test_metrics_to_json_safe_conversions():
    """``_metrics_to_json_safe`` maps numpy -> python and NaN/Inf -> None."""
    from sleap_nn.evaluation import _metrics_to_json_safe

    # numpy scalars -> python scalars
    assert _metrics_to_json_safe(np.float64(1.5)) == 1.5
    assert isinstance(_metrics_to_json_safe(np.float64(1.5)), float)
    assert _metrics_to_json_safe(np.int64(3)) == 3
    assert isinstance(_metrics_to_json_safe(np.int64(3)), int)
    assert _metrics_to_json_safe(np.bool_(True)) is True

    # non-finite floats -> None (JSON null), never the string "NaN"
    assert _metrics_to_json_safe(np.float64("nan")) is None
    assert _metrics_to_json_safe(float("nan")) is None
    assert _metrics_to_json_safe(float("inf")) is None
    assert _metrics_to_json_safe(float("-inf")) is None

    # ndarray -> nested lists, NaN inside -> None
    out = _metrics_to_json_safe(np.array([[1.0, np.nan], [2.0, 3.0]]))
    assert out == [[1.0, None], [2.0, 3.0]]

    # passthrough for native types
    assert _metrics_to_json_safe("train") == "train"
    assert _metrics_to_json_safe({"a": [np.int64(1), np.float64(2.0)]}) == {
        "a": [1, 2.0]
    }


def test_write_metrics_emits_json_sibling(tmp_path):
    """``_write_metrics`` writes the .npz AND a JSON sibling matching it.

    Mirrors the existing metrics-write test but exercises the writer directly
    on a representative metrics dict (no heavy inference/subprocess needed):
    the JSON sibling exists, ``json.load`` parses it, NaN is serialized as
    ``null`` (JSON ``None``), and scalar values match the pickled ``.npz``.
    """
    import json

    from sleap_nn.evaluation import _write_metrics

    metrics = _representative_metrics()
    save_path = tmp_path / "metrics.val.0.npz"
    _write_metrics(save_path, metrics)

    json_path = tmp_path / "metrics.val.0.json"
    assert save_path.exists()
    assert json_path.exists()

    # JSON must be valid and parseable (strict=True rejects bare NaN/Infinity).
    with open(json_path) as f:
        loaded = json.load(f, parse_constant=_reject_non_json_constant)

    # Same nested structure the app loader expects.
    assert set(loaded.keys()) == {
        "voc_metrics",
        "mOKS",
        "distance_metrics",
        "pck_metrics",
        "visibility_metrics",
    }

    # NaN in the dists matrix serialized as null (None), not "NaN".
    assert loaded["distance_metrics"]["dists"] == [[1.5, None], [2.0, 3.0]]

    # Scalar values match the npz round-trip.
    npz = np.load(save_path, allow_pickle=True)
    npz_metrics = npz["metrics"].item()
    assert loaded["mOKS"]["mOKS"] == pytest.approx(float(npz_metrics["mOKS"]["mOKS"]))
    assert loaded["voc_metrics"]["oks_voc.mAP"] == pytest.approx(
        float(npz_metrics["voc_metrics"]["oks_voc.mAP"])
    )
    assert loaded["pck_metrics"]["PCK@5"] == pytest.approx(
        float(npz_metrics["pck_metrics"]["PCK@5"])
    )
    assert loaded["visibility_metrics"]["tp"] == int(
        npz_metrics["visibility_metrics"]["tp"]
    )

    # `pcks` (a large n_pairs x n_nodes x n_thresholds boolean array) is pruned
    # from the JSON view to avoid bloat, but retained in the pickled .npz.
    assert "pcks" not in loaded["pck_metrics"]
    assert "pcks" in npz_metrics["pck_metrics"]
    # The small, useful PCK scalars survive the prune.
    assert loaded["pck_metrics"]["mPCK"] == pytest.approx(0.5)

    # App-loader key shapes: precisions is number[][], AP/recalls are number[].
    assert isinstance(loaded["voc_metrics"]["oks_voc.precisions"], list)
    assert isinstance(loaded["voc_metrics"]["oks_voc.precisions"][0], list)
    assert isinstance(loaded["voc_metrics"]["oks_voc.AP"], list)
    assert isinstance(loaded["distance_metrics"]["frame_idxs"], list)
    assert loaded["distance_metrics"]["video_paths"] == [
        "/data/vid.mp4",
        "/data/vid.mp4",
    ]


def _reject_non_json_constant(name):  # pragma: no cover - only fires on bad JSON
    raise AssertionError(f"non-JSON constant {name!r} present in output")


# ---------------------------------------------------------------------------
# Centroid-only / single-node distance evaluation
# ---------------------------------------------------------------------------

from sleap_nn.evaluation import compute_gt_centroids, run_evaluation, match_centroids
from sleap_nn.data.instance_centroids import generate_centroids


@pytest.mark.parametrize("anchor_ind", [None, 0, 1])
def test_compute_gt_centroids_parity_with_generate_centroids(anchor_ind):
    """compute_gt_centroids must EXACTLY mirror generate_centroids (#586)."""
    # Multi-node poses: anchor-visible, anchor-NaN->mean, all-NaN.
    poses = np.array(
        [
            # Anchor visible (all nodes visible).
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            # Anchor (node 0 and node 1) NaN -> mean of visible nodes.
            [[np.nan, np.nan], [np.nan, np.nan], [5.0, 7.0]],
            # All nodes NaN -> NaN centroid.
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            # Partially visible, anchor present for ind 0/1.
            [[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )

    expected = generate_centroids(
        torch.from_numpy(poses), anchor_ind=anchor_ind
    ).numpy()
    got = compute_gt_centroids(poses, anchor_ind=anchor_ind)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("anchor_ind", [None, 0])
def test_compute_gt_centroids_single_node(anchor_ind):
    """Parity on 1-node poses (the collapsed centroid skeleton case)."""
    poses = np.array(
        [
            [[12.0, 34.0]],
            [[np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    expected = generate_centroids(
        torch.from_numpy(poses), anchor_ind=anchor_ind
    ).numpy()
    got = compute_gt_centroids(poses, anchor_ind=anchor_ind)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_compute_gt_centroids_anchor_nan_falls_back_to_mean():
    """When the anchor node is NaN, fall back to mean-of-visible (not bbox)."""
    # Single instance, anchor (node 0) is NaN.
    pose = np.array([[np.nan, np.nan], [0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
    got = compute_gt_centroids(pose, anchor_ind=0)
    # Mean of the two visible nodes = (5, 10). Bbox midpoint would also be
    # (5, 10) here, so use an asymmetric extra node to disambiguate.
    pose2 = np.array(
        [[np.nan, np.nan], [0.0, 0.0], [0.0, 0.0], [9.0, 30.0]], dtype=np.float32
    )
    got2 = compute_gt_centroids(pose2, anchor_ind=0)
    # Mean of visible = (3, 10); bbox midpoint would be (4.5, 15).
    np.testing.assert_allclose(got, [5.0, 10.0])
    np.testing.assert_allclose(got2, [3.0, 10.0])


def _make_centroid_labels(minimal_instance):
    """Build (gt_multi_node, pred_single_node) labels for centroid eval.

    GT is a 3-node skeleton; predictions use a single-node ('centroid')
    skeleton. Two GT instances should match two predicted centroids; one GT is
    a false negative (no nearby prediction); one prediction is a false positive
    (just beyond the match threshold).
    """
    gt_skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    centroid_skeleton = sio.get_centroid_skeleton()

    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # GT instance A: mean of visible nodes = (10, 10).
    gt_a = sio.Instance.from_numpy(
        points_data=np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
        skeleton=gt_skeleton,
    )
    # GT instance B: mean of visible nodes = (100, 100).
    gt_b = sio.Instance.from_numpy(
        points_data=np.array([[90.0, 90.0], [100.0, 100.0], [110.0, 110.0]]),
        skeleton=gt_skeleton,
    )
    # GT instance C: mean = (500, 500) -> false negative (no nearby pred).
    gt_c = sio.Instance.from_numpy(
        points_data=np.array([[490.0, 490.0], [500.0, 500.0], [510.0, 510.0]]),
        skeleton=gt_skeleton,
    )

    gt_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[gt_a, gt_b, gt_c])
    gt_labels = sio.Labels(
        videos=[video], skeletons=[gt_skeleton], labeled_frames=[gt_lf]
    )

    # Pred near A (dist ~3 px) and near B (dist ~4 px) -> true positives.
    pred_a = sio.PredictedInstance.from_numpy(
        points_data=np.array([[13.0, 10.0]]),
        skeleton=centroid_skeleton,
        point_scores=np.array([0.9]),
        score=0.9,
    )
    pred_b = sio.PredictedInstance.from_numpy(
        points_data=np.array([[100.0, 96.0]]),
        skeleton=centroid_skeleton,
        point_scores=np.array([0.8]),
        score=0.8,
    )
    # Pred just beyond threshold from C: C is at (500, 500); place pred at
    # (560, 500) -> dist 60 > threshold 50 -> false positive (+ C is FN).
    pred_fp = sio.PredictedInstance.from_numpy(
        points_data=np.array([[560.0, 500.0]]),
        skeleton=centroid_skeleton,
        point_scores=np.array([0.7]),
        score=0.7,
    )

    pred_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[pred_a, pred_b, pred_fp]
    )
    pred_labels = sio.Labels(
        videos=[video], skeletons=[centroid_skeleton], labeled_frames=[pred_lf]
    )

    return gt_labels, pred_labels


def test_evaluator_centroid_match(minimal_instance):
    """Evaluator(match_method='centroid') TP/FP/FN + detection + distance."""
    gt_labels, pred_labels = _make_centroid_labels(minimal_instance)

    evaluator = Evaluator(
        gt_labels,
        pred_labels,
        match_threshold=50.0,
        match_method="centroid",
        anchor_ind=None,  # mean-of-visible-nodes centroid
    )

    # 2 true positives, 1 false positive (pred_fp), 1 false negative (gt_c).
    assert len(evaluator.positive_pairs) == 2
    assert len(evaluator.false_positives) == 1
    assert len(evaluator.false_negatives) == 1

    det = evaluator.detection_metrics()
    assert det["n_tp"] == 2
    assert det["n_fp"] == 1
    assert det["n_fn"] == 1
    # precision = 2/3, recall = 2/3, f1 = 2/3.
    np.testing.assert_allclose(det["precision"], 2 / 3)
    np.testing.assert_allclose(det["recall"], 2 / 3)
    np.testing.assert_allclose(det["f1"], 2 / 3)

    # Localization distances: A=3 px, B=4 px.
    np.testing.assert_allclose(sorted(evaluator.dists_dict["dists"]), [3.0, 4.0])
    np.testing.assert_allclose(det["avg"], 3.5)
    for key in ("p50", "p75", "p90", "p95", "p99"):
        assert not np.isnan(det[key])

    # evaluate() returns only detection + distance metrics (no OKS/PCK/etc.).
    metrics = evaluator.evaluate()
    assert set(metrics.keys()) == {"detection_metrics", "distance_metrics"}
    assert "voc_metrics" not in metrics
    assert "mOKS" not in metrics


def test_evaluator_centroid_handles_fully_occluded_gt(minimal_instance):
    """Regression: a fully-occluded (all-NaN) GT instance must NOT crash
    centroid matching (scipy cdist/linear_sum_assignment reject NaN). It is
    counted as a false negative.
    """
    gt_skeleton = sio.Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    centroid_skeleton = sio.get_centroid_skeleton()
    video = sio.load_slp(minimal_instance).videos[0]

    gt_match = sio.Instance.from_numpy(
        points_data=np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),  # mean (10,10)
        skeleton=gt_skeleton,
    )
    gt_occluded = sio.Instance.from_numpy(
        points_data=np.full((3, 2), np.nan),  # fully occluded -> centroid NaN
        skeleton=gt_skeleton,
    )
    gt_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[gt_match, gt_occluded]
    )
    gt_labels = sio.Labels(
        videos=[video], skeletons=[gt_skeleton], labeled_frames=[gt_lf]
    )
    pred = sio.PredictedInstance.from_numpy(
        points_data=np.array([[11.0, 10.0]]),  # ~1px from gt_match
        skeleton=centroid_skeleton,
        point_scores=np.array([0.9]),
        score=0.9,
    )
    pred_labels = sio.Labels(
        videos=[video],
        skeletons=[centroid_skeleton],
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0, instances=[pred])],
    )

    # Must not raise.
    evaluator = Evaluator(
        gt_labels,
        pred_labels,
        match_threshold=50.0,
        match_method="centroid",
        anchor_ind=None,
    )
    det = evaluator.detection_metrics()
    assert (det["n_tp"], det["n_fp"], det["n_fn"]) == (1, 0, 1)


def test_evaluator_centroid_middle_occluded_fn_attribution(minimal_instance):
    """An occluded GT between two matched GTs: the NaN-filter index map must
    keep TP/FN attribution and matched distances correct.
    """
    gt_skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
    centroid_skeleton = sio.get_centroid_skeleton()
    video = sio.load_slp(minimal_instance).videos[0]

    gt_a = sio.Instance.from_numpy(
        points_data=np.array([[10.0, 10.0], [10.0, 10.0]]), skeleton=gt_skeleton
    )
    gt_mid = sio.Instance.from_numpy(
        points_data=np.full((2, 2), np.nan), skeleton=gt_skeleton
    )
    gt_c = sio.Instance.from_numpy(
        points_data=np.array([[200.0, 200.0], [200.0, 200.0]]), skeleton=gt_skeleton
    )
    gt_labels = sio.Labels(
        videos=[video],
        skeletons=[gt_skeleton],
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=[gt_a, gt_mid, gt_c])
        ],
    )
    pred_a = sio.PredictedInstance.from_numpy(
        points_data=np.array([[12.0, 10.0]]),  # ~2px from gt_a
        skeleton=centroid_skeleton,
        point_scores=np.array([0.9]),
        score=0.9,
    )
    pred_c = sio.PredictedInstance.from_numpy(
        points_data=np.array([[200.0, 205.0]]),  # 5px from gt_c
        skeleton=centroid_skeleton,
        point_scores=np.array([0.8]),
        score=0.8,
    )
    pred_labels = sio.Labels(
        videos=[video],
        skeletons=[centroid_skeleton],
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_a, pred_c])
        ],
    )

    evaluator = Evaluator(
        gt_labels,
        pred_labels,
        match_threshold=50.0,
        match_method="centroid",
        anchor_ind=None,
    )
    det = evaluator.detection_metrics()
    assert (det["n_tp"], det["n_fp"], det["n_fn"]) == (2, 0, 1)
    np.testing.assert_allclose(sorted(evaluator.dists_dict["dists"]), [2.0, 5.0])


def test_evaluator_centroid_pred_just_beyond_threshold(minimal_instance):
    """A prediction just beyond threshold becomes FP + its GT becomes FN."""
    gt_skeleton = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
    centroid_skeleton = sio.get_centroid_skeleton()
    min_labels = sio.load_slp(minimal_instance)
    video = min_labels.videos[0]

    # GT centroid (mean of visible) at (50, 50).
    gt = sio.Instance.from_numpy(
        points_data=np.array([[40.0, 50.0], [60.0, 50.0]]),
        skeleton=gt_skeleton,
    )
    gt_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[gt])
    gt_labels = sio.Labels(
        videos=[video], skeletons=[gt_skeleton], labeled_frames=[gt_lf]
    )

    # Pred at (61, 50) -> dist 11 from GT centroid.
    pred = sio.PredictedInstance.from_numpy(
        points_data=np.array([[61.0, 50.0]]),
        skeleton=centroid_skeleton,
        point_scores=np.array([0.9]),
        score=0.9,
    )
    pred_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred])
    pred_labels = sio.Labels(
        videos=[video], skeletons=[centroid_skeleton], labeled_frames=[pred_lf]
    )

    # threshold 10 < dist 11 -> FP + FN, no match.
    evaluator = Evaluator(
        gt_labels, pred_labels, match_threshold=10.0, match_method="centroid"
    )
    assert len(evaluator.positive_pairs) == 0
    assert len(evaluator.false_positives) == 1
    assert len(evaluator.false_negatives) == 1
    det = evaluator.detection_metrics()
    assert det["precision"] == 0.0
    assert det["recall"] == 0.0
    assert det["f1"] == 0.0
    assert np.isnan(det["avg"])

    # threshold 12 > dist 11 -> matched.
    evaluator2 = Evaluator(
        gt_labels, pred_labels, match_threshold=12.0, match_method="centroid"
    )
    assert len(evaluator2.positive_pairs) == 1
    assert len(evaluator2.false_positives) == 0
    assert len(evaluator2.false_negatives) == 0
    np.testing.assert_allclose(evaluator2.detection_metrics()["avg"], 11.0)


def test_run_evaluation_auto_detects_centroid(minimal_instance, tmp_path):
    """run_evaluation auto-detects centroid mode for a single-node prediction."""
    gt_labels, pred_labels = _make_centroid_labels(minimal_instance)

    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred.slp"
    sio.save_slp(gt_labels, gt_path.as_posix())
    sio.save_slp(pred_labels, pred_path.as_posix())

    metrics = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pred_path.as_posix(),
        match_method="auto",
        user_labels_only=False,
    )

    # Centroid mode -> detection + distance metrics, no OKS VOC keys.
    assert set(metrics.keys()) == {"detection_metrics", "distance_metrics"}
    assert "voc_metrics" not in metrics
    assert "mOKS" not in metrics
    det = metrics["detection_metrics"]
    assert det["n_tp"] == 2
    assert det["n_fp"] == 1
    assert det["n_fn"] == 1


def test_run_evaluation_skips_on_zero_predicted_instances(minimal_instance, tmp_path):
    """run_evaluation() returns None and skips metric computation entirely
    when predictions have frames but zero usable instances anywhere (#719) --
    not just when the predicted file has zero frames. No metrics file is
    written either.
    """
    user_labels, pred_labels = create_labels_more_predicted_instances(minimal_instance)

    # Strip all instances from the predicted frame but keep it -- both
    # predictor pipelines retain empty-detection frames by default.
    pred_lf = pred_labels[0]
    empty_pred_labels = sio.Labels(
        videos=pred_labels.videos,
        skeletons=pred_labels.skeletons,
        labeled_frames=[
            sio.LabeledFrame(
                video=pred_lf.video, frame_idx=pred_lf.frame_idx, instances=[]
            )
        ],
    )

    gt_path = tmp_path / "gt.slp"
    pred_path = tmp_path / "pred.slp"
    metrics_path = tmp_path / "metrics.npz"
    sio.save_slp(user_labels, gt_path.as_posix())
    sio.save_slp(empty_pred_labels, pred_path.as_posix())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = run_evaluation(
            ground_truth_path=gt_path.as_posix(),
            predicted_path=pred_path.as_posix(),
            save_metrics=metrics_path.as_posix(),
        )

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, [str(w.message) for w in runtime_warnings]
    assert metrics is None
    assert not metrics_path.exists()
