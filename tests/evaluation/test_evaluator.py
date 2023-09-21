import numpy as np
from typing import List, Tuple
import sleap_io as sio
from sleap_nn.evaluation import (
    compute_instance_area,
    compute_oks,
)
from sleap_nn.evaluation import Evaluator

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
    oks = compute_oks(inst_gt, inst_pr, False)
    np.testing.assert_allclose(oks, 1)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr, False)
    np.testing.assert_allclose(oks, 1)
    
def create_labels(minimal_instance):
    # Create sample Labels object.

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
                [10.2, 20.4],
                [5.8, 15.1],
                [0.3, 10.6],
            ]
        ),
        skeleton=skeleton,
        point_scores=np.array([0.5, 0.6, 0.8]),
        instance_score=0.6,
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

    # Create labeled frame.
    user_lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=[user_inst_1, user_inst_2]
    )
    # Create ground-truth labels.
    user_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[user_lf]
    )

    # Create predicted labels.
    pred_lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst_1])
    pred_labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[pred_lf]
    )
    
    return user_labels, pred_labels

    

def test_evaluator():
    
    user_labels, pred_labels = create_labels()
    
    eval = Evaluator(user_labels, pred_labels)
    
    # test _process_frames function
    assert len(eval.frame_pairs)==1
    assert len(eval.positive_pairs) == 1
    assert len(eval.false_negatives) == 1
    
    # test compute_instance_area
    user_lf = user_labels[0]
    points_gt = user_lf.numpy()[0]
    area = compute_instance_area(points_gt)
    area[0] == 77.14

    # test compute distance function
    dist_dict = eval.dists_dict
    dists = dist_dict["dists"][0]
    calc_dist = np.array([7.10211236, 7.8, 1.3])
    assert (np.array(dists) - calc_dist <= 1e-5).all()
    
    # test visibility metrics
    viz_metrics = eval.visibility_metrics()
    assert viz_metrics["precision"]== float(1) and viz_metrics["recall"]==float(1)
    
    # test distance_metrics
    dist_metrics = eval.distance_metrics()
    assert dist_metrics["avg"]-5.400704 <= 1e-5
    assert dist_metrics["p90"]-np.percentile(calc_dist, 90) <= 1e-5
    
    # test pck metrics
    pck = eval.pck_metrics()
    assert pck["mPCK"]==0.5
    
    # test voc metrics
    voc = eval.voc_metrics(match_score_by="pck")

    # TODO: add test cases for VOC metrics
    # TODO: add two instances on single frame and add test cases

