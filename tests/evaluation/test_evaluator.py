import numpy as np
from typing import List, Tuple
import sleap_io as sio
import pytest
import copy
from sleap_nn.evaluation.evaluator import (
    compute_instance_area,
    compute_oks,
)
from sleap_nn.evaluation.evaluator import Evaluator


def test_compute_oks():
    # Test compute_oks function with the cocoutils implementation
    inst_gt = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 2 / 3)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

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


def create_labels(minimal_instance):
    # Create sample Labels object.

    labels = {}

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
        points=np.array(
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
        points=np.array(
            [
                [11.2, 17.4],
                [12.8, 15.1],
                [0.3, 10.6],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.8]),
        instance_score=0.7,
    )

    user_inst_2 = sio.Instance.from_numpy(
        points=np.array(
            [
                [1.4, 2.9],
                [30.6, 9.5],
                [40.6, 60.7],
            ]
        ),
        skeleton=skeleton,
    )

    pred_inst_2 = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [2.3, 2.2],
                [25.6, 10.0],
                [37.6, np.nan],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        instance_score=0.6,
    )

    user_inst_3 = sio.Instance.from_numpy(
        points=np.array(
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

    # two match instances and one missed user instance
    labels["labels_1"] = (user_labels, pred_labels)

    video1 = copy.deepcopy(video)
    video1.filename = "test.mp4"
    pred_inst_2 = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [np.nan, 2.2],
                [14.6, 10.0],
                [3.6, np.nan],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        instance_score=0.6,
    )

    user_lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_inst_1],
    )
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf]
    )

    pred_lf_1 = sio.LabeledFrame(
        video=video1, frame_idx=0, instances=[pred_inst_1, pred_inst_2]
    )

    pred_labels = sio.Labels(
        videos=[video1], skeletons=[skeleton], labeled_frames=[pred_lf_1]
    )

    # with no match frame pairs
    labels["labels_2"] = (user_labels, pred_labels)

    pred_inst_1 = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [11.2, 17.4],
                [12.8, 13.1],
                [0.7, 10.0],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.8]),
        instance_score=0.8,
    )

    user_inst_2 = sio.Instance.from_numpy(
        points=np.array(
            [
                [1.4, 2.9],
                [30.6, 9.5],
                [40.6, 60.7],
            ]
        ),
        skeleton=skeleton,
    )

    pred_inst_2 = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [1.3, 2.9],
                [29.6, 9.2],
                [39.6, 59.3],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        instance_score=0.7,
    )

    pred_inst_3 = sio.PredictedInstance.from_numpy(
        points=np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.6, 0.6]),
        instance_score=0.5,
    )

    user_lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[user_inst_2, user_inst_1],
    )
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf]
    )

    pred_lf_1 = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[pred_inst_2, pred_inst_1, pred_inst_3]
    )

    pred_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf_1]
    )

    # with more predicted instances than user labeled instances
    labels["labels_3"] = (user_labels, pred_labels)
    return labels


def test_evaluator_1(user_labels, pred_labels):
    eval = Evaluator(user_labels, pred_labels)

    # test _process_frames function
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

    assert (gt_1.instance.numpy() == points_gt).all()
    assert (pred_1.instance.numpy() == points_pred).all()

    points = np.array(
        [
            [55.6, 30.2],
            [10.1, 18.5],
            [35.8, 12.0],
        ]
    )
    assert (gt_3.instance.numpy() == points).all()


def test_evaluator_2(user_labels, pred_labels):
    with pytest.raises(Exception) as exc:
        eval = Evaluator(user_labels, pred_labels)
    assert str(exc.value) == "Empty Frame Pairs. No match found for the video frames"


def test_evaluator_3(user_labels, pred_labels):
    eval = Evaluator(user_labels, pred_labels)
    assert len(eval.frame_pairs) == 1
    assert len(eval.positive_pairs) == 2
    assert len(eval.false_negatives) == 0


def test_metrics(labels):
    user_labels, pred_labels = labels["labels_1"]
    eval = Evaluator(user_labels, pred_labels)

    # test compute_instance_area
    user_lf = user_labels[0]
    points_gt = user_lf.numpy()[0]
    area = compute_instance_area(points_gt)
    area[0] == 77.14

    # test compute distance function
    dist_dict = eval.dists_dict
    dists = dist_dict["dists"][0]
    calc_dist = np.array([[4.0049968, 0.8, 1.3], [1.140175, 5.024937, np.nan]])
    assert (np.abs(np.array(dists) - calc_dist[0]) <= 1e-5).all()

    dists = np.array(dist_dict["dists"][1])
    assert (np.abs(dists[:-1] - calc_dist[1][:-1]) <= 1e-5).all()
    assert np.isnan(dists[-1])

    # test visibility metrics
    viz_metrics = eval.visibility_metrics()
    assert viz_metrics["precision"] == float(1)
    assert abs(viz_metrics["recall"] - float(0.833333)) <= 1e-5

    # test distance_metrics
    dist_metrics = eval.distance_metrics()
    assert np.abs(dist_metrics["avg"] - 2.4540217) <= 1e-5
    non_nans = np.array([4.0049968, 0.8, 1.3, 1.140175, 5.024937])
    assert dist_metrics["p90"] - np.percentile(non_nans, 90) <= 1e-5

    # test pck metrics
    pck = eval.pck_metrics()
    assert np.abs(pck["mPCK"] - 0.65) <= 1e-5

    # test voc metrics
    voc = eval.voc_metrics(match_score_by="pck")
    assert np.abs(voc["pck_voc.recalls"][0] - 0.3333333) <= 1e-5
    prec = np.zeros((101,))
    prec[:34] = float(1) - np.spacing(1)
    assert (voc["pck_voc.precisions"][0] == prec).all()

    voc = eval.voc_metrics(match_score_by="oks")
    assert np.abs(voc["oks_voc.recalls"][0] - 0.0) <= 1e-5

    with pytest.raises(Exception) as exc:
        eval.voc_metrics(match_score_by="moks")
    assert (
        str(exc.value)
        == "Invalid Option for match_score_by. Choose either `oks` or `pck`"
    )

    # test mOKS
    meanOKS_calc = (0.33308048 + 0.067590989) // 2
    assert int(eval.mOKS()["mOKS"]) == meanOKS_calc

    # test voc with no false negative
    user_labels, pred_labels = labels["labels_3"]
    eval = Evaluator(user_labels, pred_labels)
    voc = eval.voc_metrics(match_score_by="oks")
    assert np.abs(voc["oks_voc.recalls"][0] - 0.5) <= 1e-5


def test_evaluator(minimal_instance):
    labels = create_labels(minimal_instance)
    user_labels, pred_labels = labels["labels_1"]

    # test two match instances and one missed user instance
    test_evaluator_1(user_labels, pred_labels)

    # test with no match frame pairs
    user_labels, pred_labels = labels["labels_2"]
    test_evaluator_2(user_labels, pred_labels)

    # test with more predicted instances than user labeled instances
    user_labels, pred_labels = labels["labels_3"]
    test_evaluator_3(user_labels, pred_labels)

    # test metrics
    test_metrics(labels)
