import numpy as np
from typing import List, Tuple
import sleap_io as sio
import pytest
from pathlib import Path
import copy
import torch
from sleap_nn.predict import run_inference
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
