"""Tests for mask-IoU evaluation of bottom-up instance-segmentation models.

Covers the ``match_method="mask"`` evaluation path added to
``sleap_nn.evaluation`` (IoU matching helpers + ``run_evaluation``) and the
post-training segmentation eval routing in ``sleap_nn.train``.
"""

import numpy as np

import sleap_io as sio

from sleap_nn.evaluation import _mask_iou, match_masks, run_evaluation


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


# ---------------------------------------------------------------------------
# _mask_iou / match_masks units
# ---------------------------------------------------------------------------


def test_mask_iou_identical_disjoint_partial():
    """IoU is 1.0 for identical, 0.0 for disjoint, and exact for partial overlap."""
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True  # 16 px
    assert _mask_iou(a, a) == 1.0

    b = np.zeros((10, 10), dtype=bool)
    b[6:9, 6:9] = True  # disjoint from a
    assert _mask_iou(a, b) == 0.0

    c = np.zeros((10, 10), dtype=bool)
    c[4:8, 2:6] = True  # overlaps a in rows 4-5 (2 of 4 rows): inter=8
    # union = 16 + 16 - 8 = 24 -> 8/24 = 1/3
    assert abs(_mask_iou(a, c) - (8 / 24)) < 1e-9


def test_mask_iou_mismatched_shapes_aligned_top_left():
    """Masks of differing shapes are compared on a common top-left canvas."""
    a = np.zeros((10, 10), dtype=bool)
    a[1:4, 1:4] = True
    big = np.zeros((20, 20), dtype=bool)
    big[1:4, 1:4] = True
    assert _mask_iou(a, big) == 1.0


def test_match_masks_perfect_and_empty():
    """Perfect 1:1 IoU matching; empty inputs produce only FN/FP."""
    g = [_disk(40, 40, 10, 10, 6), _disk(40, 40, 30, 30, 6)]
    p = [_disk(40, 40, 30, 30, 6), _disk(40, 40, 10, 10, 6)]  # swapped order
    mp, mg, up, ug, ious = match_masks(p, g, min_iou=0.5)
    assert len(mp) == 2 and len(up) == 0 and len(ug) == 0
    assert len(ious) == 2 and np.allclose(ious, 1.0)

    # No predictions -> both GT are false negatives.
    mp, mg, up, ug, ious = match_masks([], g, min_iou=0.5)
    assert len(mp) == 0 and len(ug) == 2 and ious.size == 0


def test_match_masks_below_threshold_is_unmatched():
    """A pair whose IoU is below the threshold counts as FP + FN, not a match."""
    g = [_disk(40, 40, 20, 20, 10)]
    p = [_disk(40, 40, 20, 20, 4)]  # much smaller -> low IoU
    mp, mg, up, ug, ious = match_masks(p, g, min_iou=0.5)
    assert len(mp) == 0
    assert len(up) == 1 and len(ug) == 1


def test_match_masks_hungarian_is_globally_optimal():
    """Matching is globally optimal (Hungarian), not greedy per-row argmax.

    pred0 overlaps gt0 best of its own row, but pred1 is an EXACT copy of gt0.
    A greedy matcher that assigns pred0->gt0 first would then strand pred1 on
    the far gt1 (IoU 0) and leave only one match. The optimal assignment pairs
    pred1<->gt0 (IoU 1.0) and pred0<->gt1, yielding two matches.
    """
    gt = [_disk(60, 60, 30, 20, 10), _disk(60, 60, 30, 40, 10)]
    pred = [_disk(60, 60, 30, 28, 10), _disk(60, 60, 30, 20, 10)]  # pred1 == gt0
    mp, mg, up, ug, ious = match_masks(pred, gt, min_iou=0.1)
    # Greedy would yield 1 match; the optimal assignment yields 2.
    assert len(mp) == 2 and len(up) == 0 and len(ug) == 0
    # The exact pair must be pred index 1 <-> gt index 0 with IoU 1.0.
    pairing = dict(zip(mp.tolist(), mg.tolist()))
    assert pairing[1] == 0
    assert any(abs(i - 1.0) < 1e-9 for i in ious)


# ---------------------------------------------------------------------------
# run_evaluation(match_method="mask")
# ---------------------------------------------------------------------------

FNAME = "synthetic_seg_video.mp4"
H, W = 48, 48


def _make_gt(masks, with_masks=True):
    """Build GT labels: a 2-node pose per mask (+ UserSeg masks unless disabled).

    The poses are what ``find_frame_pairs`` keys off (it filters GT frames to
    those with user instances); ``with_masks=False`` yields a poses-only GT
    frame (still paired) but no GT masks.
    """
    skel = sio.Skeleton(nodes=["a", "b"], edges=[("a", "b")])
    video = sio.Video.from_filename(FNAME)
    instances, seg_masks = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        cy, cx = float(ys.mean()), float(xs.mean())
        instances.append(
            sio.Instance.from_numpy(
                np.array([[cx, cy], [cx + 1, cy + 1]], dtype="float64"), skeleton=skel
            )
        )
        if with_masks:
            seg_masks.append(sio.UserSegmentationMask.from_numpy(m))
    lf = sio.LabeledFrame(
        video=video, frame_idx=0, instances=instances, masks=seg_masks
    )
    return sio.Labels(videos=[video], skeletons=[skel], labeled_frames=[lf])


def _make_pred(masks, score=0.9):
    """Build predicted labels: one frame of PredictedSegmentationMasks (no poses)."""
    video = sio.Video.from_filename(FNAME)
    seg_masks = [
        sio.PredictedSegmentationMask.from_numpy(m, score=score) for m in masks
    ]
    lf = sio.LabeledFrame(video=video, frame_idx=0, masks=seg_masks)
    return sio.Labels(videos=[video], labeled_frames=[lf])


def test_run_evaluation_mask_perfect(tmp_path):
    """Identical GT/pred masks -> precision=recall=F1=1.0 and mean IoU=1.0."""
    masks = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
        save_metrics=(tmp_path / "metrics.npz").as_posix(),
    )
    det, mm = m["detection_metrics"], m["mask_metrics"]
    assert det["precision"] == 1.0 and det["recall"] == 1.0 and det["f1"] == 1.0
    assert det["n_tp"] == 2 and det["n_fp"] == 0 and det["n_fn"] == 0
    assert abs(mm["mean_iou"] - 1.0) < 1e-9 and mm["n_matched"] == 2
    # All IoU stat fields are 1.0 for a perfect match.
    for k in ("min", "max", "p25", "p50", "p75"):
        assert abs(mm[k] - 1.0) < 1e-9
    # Mask mode reports detection + mask metrics only (no keypoint keys).
    assert "voc_metrics" not in m and "mOKS" not in m
    # Metrics file round-trips.
    assert (tmp_path / "metrics.npz").exists()


def test_run_evaluation_mask_disjoint(tmp_path):
    """Non-overlapping predictions -> zero TP, all FP + FN."""
    gt = [_disk(H, W, 12, 12, 6)]
    pr = [_disk(H, W, 40, 40, 6)]  # far away, IoU 0
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(gt).save(gt_path.as_posix())
    _make_pred(pr).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )
    det = m["detection_metrics"]
    assert det["n_tp"] == 0 and det["n_fp"] == 1 and det["n_fn"] == 1
    assert det["precision"] == 0.0 and det["recall"] == 0.0
    assert np.isnan(m["mask_metrics"]["mean_iou"])


def test_run_evaluation_mask_partial_threshold(tmp_path):
    """A moderately shrunk prediction matches at IoU 0.5 but not at 0.95."""
    gt = [_disk(H, W, 24, 24, 12)]
    pr = [_disk(H, W, 24, 24, 10)]  # concentric, slightly smaller
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(gt).save(gt_path.as_posix())
    _make_pred(pr).save(pr_path.as_posix())

    lenient = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
        match_threshold=0.5,
    )
    assert lenient["detection_metrics"]["n_tp"] == 1
    iou = lenient["mask_metrics"]["mean_iou"]
    assert 0.5 <= iou < 1.0

    strict = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
        match_threshold=0.95,
    )
    assert strict["detection_metrics"]["n_tp"] == 0


def test_run_evaluation_mask_fp_only_frame(tmp_path):
    """GT frame with poses but no masks -> every prediction is a false positive."""
    shapes = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(shapes, with_masks=False).save(gt_path.as_posix())  # poses, no masks
    _make_pred(shapes).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )
    det = m["detection_metrics"]
    assert det["n_tp"] == 0 and det["n_fp"] == 2 and det["n_fn"] == 0
    assert det["precision"] == 0.0
    assert np.isnan(m["mask_metrics"]["mean_iou"])


def test_run_evaluation_mask_partial_recall(tmp_path):
    """Pred recovers one of two GT masks -> TP=1, FP=0, FN=1 (recall 0.5)."""
    gt = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    pr = [_disk(H, W, 14, 14, 8)]  # only the first GT mask
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(gt).save(gt_path.as_posix())
    _make_pred(pr).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )
    det = m["detection_metrics"]
    assert det["n_tp"] == 1 and det["n_fp"] == 0 and det["n_fn"] == 1
    assert det["recall"] == 0.5 and det["precision"] == 1.0
    assert abs(m["mask_metrics"]["mean_iou"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Post-training segmentation eval routing (train.py)
# ---------------------------------------------------------------------------


def test_train_routes_segmentation_eval(minimal_instance_seg, tmp_path):
    """``train()`` on a seg model runs post-training mask eval without error.

    Exercises ``run_training`` -> ``model_type == 'bottomup_segmentation'`` ->
    ``_run_segmentation_split_eval`` -> new predict flow + ``run_evaluation``.
    The tiny 1-epoch model is not expected to predict masks, so eval skips
    gracefully; the point is that the segmentation branch is wired and robust
    (it neither crashes nor falls through to keypoint OKS evaluation).
    """
    from sleap_nn.train import train

    train(
        train_labels_path=[minimal_instance_seg.as_posix()],
        use_same_data_for_val=True,
        head_configs="bottomup_segmentation",
        backbone_config={
            "unet": {
                "in_channels": 1,
                "filters": 8,
                "filters_rate": 1.5,
                "max_stride": 16,
                "output_stride": 2,
            }
        },
        ensure_grayscale=True,
        scale=1.0,
        batch_size=1,
        max_epochs=1,
        min_train_steps_per_epoch=1,
        num_workers=0,
        save_ckpt=True,
        ckpt_dir=tmp_path.as_posix(),
        run_name="seg_eval_route",
        trainer_accelerator="cpu",
        visualize_preds_during_training=False,
        seed=42,
    )
    # The seg eval path predicts via the NEW flow and writes labels_pr.*.slp
    # for each split. The legacy OKS path uses run_inference, which has no
    # segmentation support and would raise before writing anything — so the
    # presence of these predicted files (and no exception) confirms the
    # segmentation branch ran rather than falling through to keypoint OKS eval.
    run_dir = tmp_path / "seg_eval_route"
    assert run_dir.exists()
    pred_files = list(run_dir.glob("labels_pr.*.slp"))
    assert pred_files, "segmentation eval did not write predicted split files"
    # Any emitted masks are PredictedSegmentationMasks (not keypoint instances).
    for pf in pred_files:
        for mask in sio.load_slp(pf.as_posix()).masks:
            assert isinstance(mask, sio.PredictedSegmentationMask)
