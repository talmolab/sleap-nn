"""Centroid evaluation against ground truth that carries no poses.

A centroid model's ground truth need not be poses: it can be ``Centroid``
annotations, made by hand (#702) or derived from segmentation masks by
``data_config.centroids_from_masks`` (#586). Such a file has no instances at all,
so the instance-only frame-pair filter dropped every frame and evaluation died
with "Empty Frame Pairs" — the metrics could only ever be read off the epoch-end
callback, never off ``sleap-nn eval``.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.instance_centroids import add_centroids_from_masks
from sleap_nn.evaluation import Evaluator, find_frame_pairs, run_evaluation

FNAME = "centroid_gt.mp4"
# Two squares per frame, spanning [4, 12) and [20, 28) on both axes.
SQUARE_CENTERS = [(7.5, 7.5), (23.5, 23.5)]


def _mask_only_gt(n_frames=3):
    """Mask-only, skeleton-less ground truth."""
    video = sio.Video.from_filename(FNAME)
    frames = []
    for frame_idx in range(n_frames):
        masks = []
        for k in range(2):
            arr = np.zeros((40, 40), dtype=bool)
            arr[4 + 16 * k : 12 + 16 * k, 4 + 16 * k : 12 + 16 * k] = True
            masks.append(sio.UserSegmentationMask.from_numpy(arr))
        frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=[], masks=masks
            )
        )
    return sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])


def _centroid_only_gt(n_frames=3):
    """Hand-annotated ``UserCentroid`` ground truth: no poses, no masks."""
    video = sio.Video.from_filename(FNAME)
    frames = []
    for frame_idx in range(n_frames):
        frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=[],
                centroids=[sio.UserCentroid(x=x, y=y) for x, y in SQUARE_CENTERS],
            )
        )
    return sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])


def _centroid_predictions(n_frames=3, offset=1.0):
    """Centroid-model output: single-node instances, `offset` px off in x."""
    video = sio.Video.from_filename(FNAME)
    skeleton = sio.get_centroid_skeleton()
    frames = []
    for frame_idx in range(n_frames):
        frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        np.array([[x + offset, y]]),
                        skeleton=skeleton,
                        score=0.9,
                        point_scores=np.ones(1),
                    )
                    for x, y in SQUARE_CENTERS
                ],
            )
        )
    return sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])


def _save(labels, path):
    sio.save_slp(labels, path.as_posix())
    return path.as_posix()


def test_centroid_eval_on_mask_derived_ground_truth(tmp_path):
    """Mask-only GT + `centroids_from_masks` is evaluable end to end."""
    gt = _mask_only_gt()
    assert add_centroids_from_masks(gt, method="center_of_mass") == 6
    pred = _centroid_predictions(offset=1.0)

    metrics = run_evaluation(
        _save(gt, tmp_path / "gt.slp"),
        _save(pred, tmp_path / "pred.slp"),
        match_method="centroid",
    )

    assert metrics is not None
    # Every prediction matches its own centroid, 1 px away by construction.
    assert metrics["distance_metrics"]["p50"] == pytest.approx(1.0)
    assert metrics["detection_metrics"]["precision"] == pytest.approx(1.0)
    assert metrics["detection_metrics"]["recall"] == pytest.approx(1.0)


def test_centroid_eval_on_hand_annotated_centroid_ground_truth(tmp_path):
    """The same holds for `UserCentroid` annotations with no masks behind them."""
    metrics = run_evaluation(
        _save(_centroid_only_gt(), tmp_path / "gt.slp"),
        _save(_centroid_predictions(offset=2.0), tmp_path / "pred.slp"),
        match_method="centroid",
    )

    assert metrics is not None
    assert metrics["distance_metrics"]["p50"] == pytest.approx(2.0)
    assert metrics["detection_metrics"]["recall"] == pytest.approx(1.0)


def test_centroid_annotations_are_exact_under_every_method(tmp_path):
    """A one-node wrapper makes the reduce method irrelevant, as it must.

    `anchor_ind` and `centroid_method` index into a POSE skeleton; a centroid
    annotation is already the answer, so no method may move it.
    """
    gt_path = _save(_centroid_only_gt(), tmp_path / "gt.slp")
    pred_path = _save(_centroid_predictions(offset=1.0), tmp_path / "pred.slp")

    for method in ["center_of_mass", "bbox_center", "geometric_median"]:
        gt, pred = sio.load_slp(gt_path), sio.load_slp(pred_path)
        evaluator = Evaluator(
            gt,
            pred,
            match_method="centroid",
            match_threshold=50.0,
            centroid_method=method,
            # A pose anchor index that does not exist on the one-node wrapper:
            # it must be ignored, not raise.
            anchor_ind=3,
        )
        assert len(evaluator.positive_pairs) == 6
        assert evaluator.distance_metrics()["p50"] == pytest.approx(1.0)


def test_user_instances_still_win_when_a_frame_has_both(tmp_path):
    """A frame with poses uses them; the centroid path is only a fallback."""
    video = sio.Video.from_filename(FNAME)
    skeleton = sio.Skeleton(["A", "B"])
    # Pose spans x=10..30, so its center of mass is 20 -- far from the
    # deliberately wrong centroid annotation at x=0.
    gt = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    sio.Instance.from_numpy(
                        np.array([[10.0, 20.0], [30.0, 20.0]]), skeleton=skeleton
                    )
                ],
                centroids=[sio.UserCentroid(x=0.0, y=0.0)],
            )
        ],
        videos=[video],
        skeletons=[skeleton],
    )
    pred_skeleton = sio.get_centroid_skeleton()
    pred = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        np.array([[20.0, 20.0]]),
                        skeleton=pred_skeleton,
                        score=0.9,
                        point_scores=np.ones(1),
                    )
                ],
            )
        ],
        videos=[video],
        skeletons=[pred_skeleton],
    )

    metrics = run_evaluation(
        _save(gt, tmp_path / "gt.slp"),
        _save(pred, tmp_path / "pred.slp"),
        match_method="centroid",
    )

    # Distance 0 from the pose's own center of mass; had the annotation been
    # used instead, this would be ~28 px.
    assert metrics["distance_metrics"]["p50"] == pytest.approx(0.0)


def test_frames_with_neither_still_raise(tmp_path):
    """No poses and no centroids is still an error, not a silent empty pass."""
    video = sio.Video.from_filename(FNAME)
    gt = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=[], masks=[])
        ],
        videos=[video],
        skeletons=[],
    )

    with pytest.raises(Exception, match="Empty Frame Pairs"):
        Evaluator(gt, _centroid_predictions(n_frames=1), match_method="centroid")


def test_find_frame_pairs_keeps_centroid_frames_only_when_asked():
    """The frame-pair change is opt-in: OKS mode behaves exactly as before."""
    gt = _centroid_only_gt(n_frames=2)
    pred = _centroid_predictions(n_frames=2)

    assert find_frame_pairs(gt, pred) == []
    assert len(find_frame_pairs(gt, pred, keep_user_centroid_frames=True)) == 2
