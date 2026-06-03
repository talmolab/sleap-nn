"""Tests for mask-IoU evaluation of bottom-up instance-segmentation models.

Covers the ``match_method="mask"`` evaluation path added to
``sleap_nn.evaluation`` (IoU matching helpers + ``run_evaluation``) and the
post-training segmentation eval routing in ``sleap_nn.train``.
"""

import numpy as np

import sleap_io as sio

from sleap_nn.evaluation import (
    COCO_SIZE_EDGES,
    MASK_IOU_THRESHOLDS,
    _ap_from_pr,
    _boundary_iou,
    _mask_iou,
    _mask_pair_stats,
    _percentile_size_edges,
    _size_mask,
    match_masks,
    run_evaluation,
)


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
    """Build predicted labels: one frame of PredictedSegmentationMasks (no poses).

    ``score`` may be a scalar (broadcast to every mask) or a per-mask sequence.
    """
    video = sio.Video.from_filename(FNAME)
    if np.ndim(score) == 0:
        scores = [float(score)] * len(masks)
    else:
        scores = [float(s) for s in score]
    seg_masks = [
        sio.PredictedSegmentationMask.from_numpy(m, score=s)
        for m, s in zip(masks, scores)
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


def test_run_evaluation_mask_panoptic_quality_perfect(tmp_path):
    """Perfect match -> PQ = SQ = RQ = 1.0 and miss-penalized IoU = 1.0."""
    masks = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    mm = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_metrics"]
    for k in ("pq", "sq", "rq", "mean_iou_all_gt"):
        assert abs(mm[k] - 1.0) < 1e-9, k


def test_run_evaluation_mask_miss_penalized_iou_and_pq(tmp_path):
    """A miss drags down PQ and the all-GT IoU while TP-only mean IoU stays high.

    One of two GT masks is recovered exactly: TP=1, FN=1, FP=0. TP-only mean IoU
    is 1.0, but the miss-penalized mean (over both GT) is 0.5, and
    PQ = iou_sum / (TP + 0.5*FP + 0.5*FN) = 1.0 / 1.5 = 2/3.
    """
    gt = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    pr = [_disk(H, W, 14, 14, 8)]  # only the first GT mask
    gt_path = tmp_path / "gt.slp"
    pr_path = tmp_path / "pr.slp"
    _make_gt(gt).save(gt_path.as_posix())
    _make_pred(pr).save(pr_path.as_posix())

    mm = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_metrics"]
    assert abs(mm["mean_iou"] - 1.0) < 1e-9  # TP-only, blind to the miss
    assert abs(mm["mean_iou_all_gt"] - 0.5) < 1e-9  # penalized over all GT
    assert abs(mm["rq"] - (1.0 / 1.5)) < 1e-9
    assert abs(mm["pq"] - (1.0 / 1.5)) < 1e-9
    assert mm["n_fn"] == 1 and mm["n_fp"] == 0


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


# ---------------------------------------------------------------------------
# COCO mask-AP helper units (_size_mask / _ap_from_pr / _boundary_iou /
# _mask_pair_stats)
# ---------------------------------------------------------------------------


def test_size_mask_edge_buckets_and_nan():
    """Half-open edge buckets partition finite areas; NaN areas fall in none."""
    # COCO edges: small < 32^2 (1024) <= medium < 96^2 (9216) <= large.
    areas = np.array([500.0, 1024.0, 5000.0, 9216.0, 20000.0, np.nan])
    small = _size_mask(areas, 0, COCO_SIZE_EDGES)
    medium = _size_mask(areas, 1, COCO_SIZE_EDGES)
    large = _size_mask(areas, 2, COCO_SIZE_EDGES)
    assert small.tolist() == [True, False, False, False, False, False]
    assert medium.tolist() == [False, True, True, False, False, False]
    assert large.tolist() == [False, False, False, True, True, False]
    # Every finite area lands in exactly one bucket; NaN in none.
    assert np.array_equal(small | medium | large, ~np.isnan(areas))


def test_percentile_size_edges_are_dataset_relative():
    """Percentile edges adapt to the area distribution (terciles by default)."""
    areas = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    e0, e1 = _percentile_size_edges(areas)
    assert e0 < e1
    # Terciles of a uniform spread fall in the interior, unlike COCO's fixed
    # 1024/9216 which would lump these mouse-scale masks all into "small".
    assert 200.0 <= e0 <= 400.0 and 400.0 <= e1 <= 600.0
    assert np.all(np.isnan(_percentile_size_edges(np.array([]))))


def test_ap_from_pr_edge_cases_and_ranking():
    """AP is NaN with no GT, 0.0 with no detections, and rewards ranking TPs first."""
    rt = np.linspace(0, 1, 101)
    assert np.isnan(_ap_from_pr(np.array([0.9]), np.array([True]), 0, rt)[0])
    ap0, rec0 = _ap_from_pr(np.array([]), np.array([], dtype=bool), 3, rt)
    assert ap0 == 0.0 and rec0 == 0.0
    # 1 GT, one TP + one FP. TP scored higher (ranked first) -> AP 1.0;
    # FP scored higher (ranked first) -> AP 0.5.
    ap_good, _ = _ap_from_pr(np.array([0.9, 0.1]), np.array([True, False]), 1, rt)
    ap_bad, _ = _ap_from_pr(np.array([0.9, 0.1]), np.array([False, True]), 1, rt)
    assert abs(ap_good - 1.0) < 1e-9
    assert abs(ap_bad - 0.5) < 1e-2
    assert ap_good > ap_bad


def test_boundary_iou_identical_and_shifted():
    """Boundary IoU is 1.0 for identical masks and < 1.0 once a contour shifts."""
    a = _disk(80, 80, 40, 40, 20)
    assert _boundary_iou(a, a) == 1.0
    shifted = _disk(80, 80, 40, 47, 20)  # translated -> contours diverge
    biou = _boundary_iou(a, shifted)
    assert 0.0 <= biou < 1.0
    # Boundary IoU is stricter than mask IoU for the same contour shift.
    assert biou < _mask_iou(a, shifted)


def test_mask_pair_stats_matches_mask_iou_and_reports_intersection():
    """_mask_pair_stats IoU matches _mask_iou and intersection counts are exact."""
    p = [_disk(40, 40, 20, 20, 8)]
    g = [_disk(40, 40, 20, 20, 8), _disk(40, 40, 35, 35, 5)]
    iou, inter = _mask_pair_stats(p, g)
    assert iou.shape == (1, 2) and inter.shape == (1, 2)
    assert abs(iou[0, 0] - 1.0) < 1e-9 and inter[0, 0] == int(g[0].sum())
    assert abs(iou[0, 1] - _mask_iou(p[0], g[1])) < 1e-9
    assert inter[0, 1] == int(np.logical_and(p[0], g[1]).sum())


# ---------------------------------------------------------------------------
# mask_voc_metrics (COCO-style score-ranked mask AP)
# ---------------------------------------------------------------------------


def test_mask_voc_ap_perfect(tmp_path):
    """Identical GT/pred masks -> mAP = AP50 = AP75 = AR = 1.0."""
    masks = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )
    mvoc = m["mask_voc_metrics"]
    for k in ("mask_voc.mAP", "mask_voc.AP50", "mask_voc.AP75", "mask_voc.AR"):
        assert abs(mvoc[k] - 1.0) < 1e-9, k
    # Per-threshold AP array spans the canonical 10 COCO thresholds, all 1.0.
    assert np.allclose(mvoc["mask_voc.AP"], 1.0)
    assert mvoc["mask_voc.iou_thresholds"].size == MASK_IOU_THRESHOLDS.size


def test_mask_voc_ap50_ge_ap75_on_eroded(tmp_path):
    """A shrunk prediction (0.5 <= IoU < 0.75) -> AP50 = 1.0 > AP75 = 0.0."""
    gt_mask = _disk(H, W, 24, 24, 12)
    pr_mask = _disk(H, W, 24, 24, 9)  # concentric, smaller
    # Precondition: the shrink lands the IoU strictly between the two thresholds.
    iou = _mask_iou(pr_mask, gt_mask)
    assert 0.5 <= iou < 0.75, iou
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt([gt_mask]).save(gt_path.as_posix())
    _make_pred([pr_mask]).save(pr_path.as_posix())

    mvoc = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
        match_threshold=0.5,
    )["mask_voc_metrics"]
    assert abs(mvoc["mask_voc.AP50"] - 1.0) < 1e-9
    assert mvoc["mask_voc.AP75"] == 0.0
    assert mvoc["mask_voc.AP50"] >= mvoc["mask_voc.AP75"]


def test_mask_voc_ap_score_ranking(tmp_path):
    """A high-scored correct mask + low-scored spurious mask -> AP 1.0.

    Reversing the scores (spurious ranked first) drops AP to 0.5, confirming
    detections are ranked by ``PredictedSegmentationMask.score``.
    """
    gt = [_disk(H, W, 14, 14, 8)]
    correct = _disk(H, W, 14, 14, 8)  # exact copy -> IoU 1.0
    spurious = _disk(H, W, 38, 38, 6)  # disjoint -> IoU 0, a false positive
    gt_path = tmp_path / "gt.slp"
    _make_gt(gt).save(gt_path.as_posix())

    pr_good = tmp_path / "good.slp"
    _make_pred([correct, spurious], score=[0.9, 0.1]).save(pr_good.as_posix())
    ap_good = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_good.as_posix(),
        match_method="mask",
    )["mask_voc_metrics"]["mask_voc.AP50"]

    pr_bad = tmp_path / "bad.slp"
    _make_pred([correct, spurious], score=[0.1, 0.9]).save(pr_bad.as_posix())
    ap_bad = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_bad.as_posix(),
        match_method="mask",
    )["mask_voc_metrics"]["mask_voc.AP50"]

    assert abs(ap_good - 1.0) < 1e-9
    assert abs(ap_bad - 0.5) < 1e-2
    assert ap_good > ap_bad


def test_mask_per_size_buckets_sum_to_total(tmp_path):
    """Per-size GT counts sum to total under both schemes; perfect -> AP 1.0.

    Three widely-separated areas land 1/1/1 under both percentile terciles
    (primary) and COCO fixed cutoffs.
    """
    big = 280
    small = _disk(big, big, 30, 30, 10)  # area ~317 -> small (<1024 COCO)
    medium = _disk(big, big, 60, 210, 40)  # area ~5026 -> medium (COCO)
    large = _disk(big, big, 190, 100, 70)  # area ~15394 -> large (>9216 COCO)
    masks = [small, medium, large]
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    m = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )
    mvoc = m["mask_voc_metrics"]
    # Default (primary) scheme is the dataset-relative percentile binning.
    assert mvoc["mask_voc.size_scheme"] == "percentile"
    assert len(mvoc["mask_voc.size_edges"]) == 2
    for prefix in ("mask_voc.", "mask_voc.coco."):
        counts = tuple(mvoc[f"{prefix}n_gt_{b}"] for b in ("small", "medium", "large"))
        assert counts == (1, 1, 1), prefix
        assert sum(counts) == mvoc["mask_voc.n_gt"] == 3
    # Per-size mask_metrics breakdown: primary buckets + a coco sub-dict, both
    # summing to the GT total.
    per_size = m["mask_metrics"]["per_size"]
    assert per_size["scheme"] == "percentile"
    assert sum(per_size[k]["n_gt"] for k in ("small", "medium", "large")) == 3
    assert sum(per_size["coco"][k]["n_gt"] for k in ("small", "medium", "large")) == 3
    # Perfect masks -> every populated per-size AP is 1.0 (both schemes).
    for prefix in ("mask_voc.", "mask_voc.coco."):
        for bucket in ("small", "medium", "large"):
            assert abs(mvoc[f"{prefix}AP_{bucket}"] - 1.0) < 1e-9, prefix + bucket


def test_percentile_buckets_spread_what_coco_lumps(tmp_path):
    """Percentile bins spread mouse-scale masks that COCO lumps into 'small'.

    All three masks are < 1024 px so COCO buckets them identically, but the
    percentile (primary) scheme spreads them across small/medium/large. This is
    the motivation for defaulting to dataset-relative bins: COCO's fixed
    1024/9216 px cutoffs are blind to size structure at animal-mask scale.
    """
    big = 120
    masks = [
        _disk(big, big, 20, 20, 8),  # area ~197
        _disk(big, big, 20, 90, 12),  # area ~452
        _disk(big, big, 90, 55, 16),  # area ~804  (all < 1024 -> COCO 'small')
    ]
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    mvoc = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_voc_metrics"]
    # COCO lumps all three into 'small'.
    assert mvoc["mask_voc.coco.n_gt_small"] == 3
    assert mvoc["mask_voc.coco.n_gt_medium"] == 0
    assert mvoc["mask_voc.coco.n_gt_large"] == 0
    # Percentile terciles spread them one per bucket.
    pct = tuple(mvoc[f"mask_voc.n_gt_{b}"] for b in ("small", "medium", "large"))
    assert pct == (1, 1, 1)


# ---------------------------------------------------------------------------
# Fragmentation (over-/under-segmentation) counts
# ---------------------------------------------------------------------------


def test_mask_fragmentation_oversegmentation(tmp_path):
    """One GT mask split into two half-mask predictions -> oversegmentation = 1."""
    disk = _disk(H, W, 24, 24, 14)
    cols = np.arange(W)[None, :]
    left = disk & (cols < 24)
    right = disk & (cols >= 24)
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt([disk]).save(gt_path.as_posix())
    _make_pred([left, right]).save(pr_path.as_posix())

    mm = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_metrics"]
    assert mm["oversegmentation"] == 1
    assert mm["undersegmentation"] == 0


def test_mask_fragmentation_undersegmentation(tmp_path):
    """One prediction spanning two GT masks -> undersegmentation = 1."""
    d1 = _disk(H, W, 14, 14, 8)
    d2 = _disk(H, W, 34, 34, 8)
    merged = d1 | d2  # single predicted mask covering both GT instances
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt([d1, d2]).save(gt_path.as_posix())
    _make_pred([merged]).save(pr_path.as_posix())

    mm = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_metrics"]
    assert mm["undersegmentation"] == 1
    assert mm["oversegmentation"] == 0


def test_mask_metrics_boundary_iou_perfect(tmp_path):
    """Perfect match -> mean boundary IoU = 1.0; a miss leaves TP-only boundary."""
    masks = [_disk(H, W, 14, 14, 8), _disk(H, W, 34, 34, 8)]
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt(masks).save(gt_path.as_posix())
    _make_pred(masks).save(pr_path.as_posix())

    mm = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_metrics"]
    assert abs(mm["mean_boundary_iou"] - 1.0) < 1e-9


def test_mask_voc_metrics_empty_when_no_gt_masks(tmp_path):
    """A poses-only GT frame (no masks) -> AP NaN (no GT), AR NaN, no crash."""
    shapes = [_disk(H, W, 14, 14, 8)]
    gt_path, pr_path = tmp_path / "gt.slp", tmp_path / "pr.slp"
    _make_gt(shapes, with_masks=False).save(gt_path.as_posix())  # poses, no masks
    _make_pred(shapes).save(pr_path.as_posix())

    mvoc = run_evaluation(
        ground_truth_path=gt_path.as_posix(),
        predicted_path=pr_path.as_posix(),
        match_method="mask",
    )["mask_voc_metrics"]
    assert np.isnan(mvoc["mask_voc.mAP"])
    assert mvoc["mask_voc.n_gt"] == 0
