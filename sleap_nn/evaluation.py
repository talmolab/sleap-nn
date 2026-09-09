"""This module is to compute evaluation metrics for trained models."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import attrs
import sleap_io as sio
import torch
from loguru import logger
import click
from pathlib import Path

from sleap_nn.data.instance_centroids import generate_centroids


def compute_gt_centroids(
    instance_gt_points: np.ndarray,
    anchor_ind: Optional[int] = None,
    method: Optional[str] = None,
    fallback: Optional[str] = None,
) -> np.ndarray:
    """Compute ground-truth centroids for a numpy array of instance keypoints.

    A thin numpy-in/numpy-out wrapper around
    :func:`sleap_nn.data.instance_centroids.generate_centroids`, which is the
    single definition of what a centroid MEANS (see also #586). It used to be a
    hand-written numpy mirror; delegating removes the drift that
    ``sleap_nn.inference.centroid_convert`` warns about — evaluation now cannot
    disagree with the trained target about the centroid, including for the
    ``bbox_center`` / ``geometric_median`` methods.

    Args:
        instance_gt_points: Ground-truth keypoints of shape ``(n_instances,
            n_nodes, 2)`` or ``(n_nodes, 2)``. Missing/occluded nodes are NaN.
        anchor_ind: Index of the node to use as the anchor. Required by (and only
            used by) ``method="anchor"``; when that node is NaN for an instance,
            that instance falls back to ``fallback``.
        method: One of ``sleap_nn.data.instance_centroids.CENTROID_METHODS``.
            ``None`` (default) infers it from ``anchor_ind`` — the historical
            behavior: the anchor node when given, else the NaN-ignoring mean of
            visible nodes.
        fallback: Reduce method for a missing anchor. ``None`` means
            ``"center_of_mass"``.

    Returns:
        Centroids of shape ``(n_instances, 2)`` (or ``(2,)`` for a single
        instance input), reducing the node axis.
    """
    points = np.asarray(instance_gt_points, dtype=np.float64)
    centroids = generate_centroids(
        torch.from_numpy(points),
        anchor_ind=anchor_ind,
        method=method,
        fallback=fallback,
    )
    return centroids.numpy()


def match_centroids(
    pred_centroids: "np.ndarray",
    gt_centroids: "np.ndarray",
    max_distance: float = 50.0,
) -> tuple:
    """Match predicted centroids to ground truth using Hungarian algorithm.

    Args:
        pred_centroids: Predicted centroid locations, shape (n_pred, 2).
        gt_centroids: Ground truth centroid locations, shape (n_gt, 2).
        max_distance: Maximum distance threshold for valid matches (in pixels).

    Returns:
        Tuple of:
            - matched_pred_indices: Indices of matched predictions
            - matched_gt_indices: Indices of matched ground truth
            - unmatched_pred_indices: Indices of unmatched predictions (false positives)
            - unmatched_gt_indices: Indices of unmatched ground truth (false negatives)
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    n_pred = len(pred_centroids)
    n_gt = len(gt_centroids)

    # Handle edge cases
    if n_pred == 0 and n_gt == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if n_pred == 0:
        return np.array([]), np.array([]), np.array([]), np.arange(n_gt)
    if n_gt == 0:
        return np.array([]), np.array([]), np.arange(n_pred), np.array([])

    # Compute pairwise distances
    cost_matrix = cdist(pred_centroids, gt_centroids)

    # Run Hungarian algorithm for optimal matching
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter matches that exceed max_distance
    matched_pred = []
    matched_gt = []
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if cost_matrix[p_idx, g_idx] <= max_distance:
            matched_pred.append(p_idx)
            matched_gt.append(g_idx)

    matched_pred = np.array(matched_pred)
    matched_gt = np.array(matched_gt)

    # Find unmatched indices
    all_pred = set(range(n_pred))
    all_gt = set(range(n_gt))
    unmatched_pred = np.array(list(all_pred - set(matched_pred)))
    unmatched_gt = np.array(list(all_gt - set(matched_gt)))

    return matched_pred, matched_gt, unmatched_pred, unmatched_gt


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks.

    Masks may have differing shapes (e.g. padding differences between GT and
    prediction). Both segmentation masks are top-left aligned (offset (0, 0)),
    so they are compared on a common canvas sized to the max H/W of the pair.
    """
    if a.shape != b.shape:
        h = max(a.shape[0], b.shape[0])
        w = max(a.shape[1], b.shape[1])
        aa = np.zeros((h, w), dtype=bool)
        bb = np.zeros((h, w), dtype=bool)
        aa[: a.shape[0], : a.shape[1]] = a
        bb[: b.shape[0], : b.shape[1]] = b
        a, b = aa, bb
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    # Two empty masks are identical -> IoU 1.0 (consistent with the
    # "identical masks -> 1.0" contract). In practice neither GT (burned-in,
    # non-empty) nor predicted (empty masks are dropped at postprocess) masks
    # are empty, so this only guards the degenerate case.
    if union == 0:
        return 1.0
    return float(inter / union)


def _mask_iou_matrix(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> np.ndarray:
    """Compute the IoU between every ``(pred, gt)`` mask pair.

    Returns:
        ``(n_pred, n_gt)`` float array of IoU values in ``[0, 1]``.
    """
    iou = np.zeros((len(pred_masks), len(gt_masks)), dtype=float)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou[i, j] = _mask_iou(pm, gm)
    return iou


def match_masks(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    min_iou: float = 0.5,
) -> tuple:
    """Match predicted masks to ground-truth masks by IoU (Hungarian).

    Args:
        pred_masks: List of boolean arrays, one per predicted instance.
        gt_masks: List of boolean arrays, one per ground-truth instance.
        min_iou: Minimum IoU for a matched pair to count as a true positive.

    Returns:
        Tuple of:
            - matched_pred_indices: Indices of matched predictions.
            - matched_gt_indices: Indices of matched ground truth.
            - unmatched_pred_indices: Unmatched predictions (false positives).
            - unmatched_gt_indices: Unmatched ground truth (false negatives).
            - matched_ious: IoU of each matched pair, aligned to
              ``matched_pred_indices``.
    """
    from scipy.optimize import linear_sum_assignment

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    empty = np.array([], dtype=int)
    if n_pred == 0 and n_gt == 0:
        return empty, empty, empty, empty, np.array([])
    if n_pred == 0:
        return empty, empty, empty, np.arange(n_gt), np.array([])
    if n_gt == 0:
        return empty, empty, np.arange(n_pred), empty, np.array([])

    iou = _mask_iou_matrix(pred_masks, gt_masks)  # (n_pred, n_gt)
    # Maximize total IoU -> minimize negative IoU.
    pred_indices, gt_indices = linear_sum_assignment(-iou)

    matched_pred, matched_gt, matched_ious = [], [], []
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if iou[p_idx, g_idx] >= min_iou:
            matched_pred.append(int(p_idx))
            matched_gt.append(int(g_idx))
            matched_ious.append(float(iou[p_idx, g_idx]))

    matched_pred = np.array(matched_pred, dtype=int)
    matched_gt = np.array(matched_gt, dtype=int)
    unmatched_pred = np.array(
        sorted(set(range(n_pred)) - set(matched_pred.tolist())), dtype=int
    )
    unmatched_gt = np.array(
        sorted(set(range(n_gt)) - set(matched_gt.tolist())), dtype=int
    )
    return (
        matched_pred,
        matched_gt,
        unmatched_pred,
        unmatched_gt,
        np.array(matched_ious),
    )


def _frame_masks(
    frame: sio.LabeledFrame, drop_predicted_instances: bool = False
) -> List[np.ndarray]:
    """Decode a frame's segmentation masks into boolean arrays on the image grid.

    Scale-aware: masks encoded at output-stride (non-identity ``scale``, the
    default for predicted segmentation masks) are nearest-neighbor resampled up
    to their image extent, so a stride-res prediction and an original-res
    ground-truth mask are compared on a common image-pixel grid. Scale-1 masks
    (legacy full-res GT/preds) take a zero-copy fast path, so existing eval
    numbers are unchanged.

    ``drop_predicted_instances`` (set for the ground-truth side of
    ``match_method="mask"`` when ``user_labels_only=True``) discards masks whose
    linked instance is a ``PredictedInstance``. Segmentation-mask files built from
    poses (``Instance.to_mask`` / ``Labels.convert``) attach a mask to *every*
    instance, so any ``PredictedInstance`` carried in a labels file also gets a
    mask; those are model output, not ground truth, and would otherwise be scored
    as extra ground-truth instances (spurious false negatives that cap recall).
    Masks linked to a user ``Instance``, or with no linked instance (e.g.
    whole-frame semantic union masks), are retained -- so this is scoped to
    per-instance mask matching and never touches the ``match_method="semantic"``
    union path.
    """
    from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

    masks = getattr(frame, "masks", None) or []
    if drop_predicted_instances:
        masks = [
            m
            for m in masks
            if not isinstance(getattr(m, "instance", None), sio.PredictedInstance)
        ]
    return [decode_mask_to_image_res(m) for m in masks]


def _union_frame_fg(frame: sio.LabeledFrame) -> np.ndarray:
    """Union all of a frame's segmentation masks into ONE foreground mask.

    Decodes each mask to the image grid (scale + top-left offset baked in, via
    :func:`_frame_masks`) and ORs them onto a common canvas sized to the max H/W
    across the frame's masks. Returns a ``(1, 1)`` all-False array for a frame
    with no masks, so an empty frame is a valid (empty) foreground rather than an
    error. Used only by ``match_method="semantic"`` (whole-frame binary
    foreground segmentation), where there is a single foreground per frame and no
    instances to separate.
    """
    decoded = _frame_masks(frame)
    if not decoded:
        return np.zeros((1, 1), dtype=bool)
    h = max(d.shape[0] for d in decoded)
    w = max(d.shape[1] for d in decoded)
    canvas = np.zeros((h, w), dtype=bool)
    for d in decoded:
        canvas[: d.shape[0], : d.shape[1]] |= d
    return canvas


def _frame_pred_scores(frame: sio.LabeledFrame) -> np.ndarray:
    """Per-mask detection scores for a frame (``PredictedSegmentationMask.score``).

    Ground-truth (``UserSegmentationMask``) masks carry no score; any mask
    without a score defaults to ``1.0`` so score-ranking degrades gracefully to
    insertion order. Aligned to :func:`_frame_masks`.
    """
    masks = getattr(frame, "masks", None) or []
    scores = []
    for m in masks:
        s = getattr(m, "score", None)
        scores.append(1.0 if s is None else float(s))
    return np.array(scores, dtype=float)


# COCO mask-AP IoU thresholds: 0.50:0.05:0.95.
MASK_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
# Three size buckets (small/medium/large), defined by two inner area edges.
_SIZE_KEYS = ("small", "medium", "large")
# COCO object-size area edges (pixels^2): small < 32^2 <= medium < 96^2 <= large.
COCO_SIZE_EDGES = np.array([32**2, 96**2], dtype=float)  # [1024, 9216]
# Default percentile cut points for the dataset-relative (primary) size buckets:
# terciles, so each bucket holds ~1/3 of GT masks.
DEFAULT_SIZE_PERCENTILES = (100.0 / 3.0, 200.0 / 3.0)


def _percentile_size_edges(
    gt_areas: np.ndarray, percentiles: Tuple[float, float] = DEFAULT_SIZE_PERCENTILES
) -> np.ndarray:
    """Two area edges (px^2) at the given percentiles of the GT area distribution.

    Dataset-relative size bins adapt small/medium/large to the actual mask
    scale (mice vs. flies vs. ...) instead of COCO's fixed pixel cutoffs, which
    bucket every animal mask the same way. Returns ``[nan, nan]`` when there is
    no GT (all buckets then empty -> NaN AP).
    """
    g = np.asarray(gt_areas, dtype=float)
    g = g[~np.isnan(g)]
    if g.size == 0:
        return np.array([np.nan, np.nan])
    return np.percentile(g, list(percentiles))


def _size_mask(areas: np.ndarray, bucket_idx: int, edges: np.ndarray) -> np.ndarray:
    """Boolean mask selecting ``areas`` (px^2) in size bucket ``bucket_idx``.

    Buckets are half-open ``[lo, hi)`` intervals delimited by ``edges`` (the two
    inner boundaries): bucket 0 is ``(-inf, edges[0])``, bucket 1 is
    ``[edges[0], edges[1])``, bucket 2 is ``[edges[1], inf)``. NaN areas (e.g. an
    unmatched detection's missing matched-GT area) and NaN edges compare False
    against every bound and so are excluded from all buckets.
    """
    areas = np.asarray(areas, dtype=float)
    lo = -np.inf if bucket_idx == 0 else edges[bucket_idx - 1]
    hi = np.inf if bucket_idx >= len(edges) else edges[bucket_idx]
    with np.errstate(invalid="ignore"):
        return (areas >= lo) & (areas < hi)


def _align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-left-align two boolean masks onto a common max-H/W canvas."""
    if a.shape == b.shape:
        return a, b
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    aa = np.zeros((h, w), dtype=bool)
    bb = np.zeros((h, w), dtype=bool)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    return aa, bb


def _mask_pair_stats(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute IoU and raw-intersection matrices over all ``(pred, gt)`` pairs.

    Returns ``(iou, inter)`` each ``(n_pred, n_gt)``. ``iou`` matches
    :func:`_mask_iou_matrix`; ``inter`` (intersection pixel counts) additionally
    supports fragmentation (over-/under-segmentation) scoring without a second
    decode pass.
    """
    n_p, n_g = len(pred_masks), len(gt_masks)
    iou = np.zeros((n_p, n_g), dtype=float)
    inter = np.zeros((n_p, n_g), dtype=float)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            a, b = _align_pair(pm, gm)
            inter_ij = int(np.logical_and(a, b).sum())
            union_ij = int(np.logical_or(a, b).sum())
            inter[i, j] = inter_ij
            iou[i, j] = 1.0 if union_ij == 0 else inter_ij / union_ij
    return iou, inter


def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Extract the boundary region of a binary mask (Cheng et al., 2021).

    The boundary region is ``mask`` minus its erosion by a disk of radius
    ``d = round(dilation_ratio * image_diagonal)`` (>= 1 px). A 1-px constant
    border pad makes pixels on the image edge count as boundary, matching the
    reference Boundary-IoU implementation (arXiv:2103.16562).
    """
    import cv2

    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    h, w = mask.shape
    d = int(round(dilation_ratio * float(np.sqrt(h * h + w * w))))
    if d < 1:
        d = 1
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)
    eroded = eroded[1 : h + 1, 1 : w + 1]
    return (mask - eroded).astype(bool)


def _boundary_iou(
    pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02
) -> float:
    """Boundary IoU between two masks (Cheng et al., 2021, arXiv:2103.16562).

    IoU restricted to the masks' boundary regions; more sensitive to contour
    error than mask IoU. Two empty boundary regions (identical masks) -> 1.0.
    """
    a, b = _align_pair(pred, gt)
    ba = _mask_to_boundary(a, dilation_ratio)
    bb = _mask_to_boundary(b, dilation_ratio)
    inter = int(np.logical_and(ba, bb).sum())
    union = int(np.logical_or(ba, bb).sum())
    return 1.0 if union == 0 else float(inter / union)


def _skeletonize(mask: np.ndarray) -> Optional[np.ndarray]:
    """1-px morphological skeleton of a binary mask (via scikit-image).

    Returns an empty skeleton for an empty mask, and ``None`` if scikit-image is
    not installed (clDice is then skipped rather than crashing eval).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None
    return np.asarray(skeletonize(np.ascontiguousarray(mask, dtype=bool)), dtype=bool)


def mask_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice (clDice) between two binary masks.

    Shit et al., "clDice — A Novel Topology-Preserving Loss Function for Tubular
    Structure Segmentation," CVPR 2021 (arXiv:2003.07311). The connectivity-aware
    F-score of two skeleton-overlap terms:

    * ``Tprec`` = fraction of the *predicted* skeleton lying inside the *GT* mask
      (is my centerline drawn on a real object?),
    * ``Tsens`` = fraction of the *GT* skeleton lying inside the *predicted* mask
      (did I cover every real object along its length?),

    with ``clDice = 2·Tprec·Tsens / (Tprec + Tsens)``. Nearly width-insensitive
    and connectivity-sensitive, so it is a fairer quality measure than area IoU
    for thin/tubular structures (roots, vessels, neurites). Uses a hard
    morphological skeleton (exact, no ``k`` to tune).

    Two empty masks return ``1.0`` (matching the ``_mask_iou`` "identical -> 1.0"
    contract). Returns ``nan`` when scikit-image is unavailable so callers can
    drop clDice from the summary without failing.
    """
    a, b = _align_pair(pred, gt)
    if not a.any() and not b.any():
        return 1.0
    sk_p = _skeletonize(a)
    sk_g = _skeletonize(b)
    if sk_p is None or sk_g is None:
        return float("nan")
    sp, sg = int(sk_p.sum()), int(sk_g.sum())
    if sp == 0 or sg == 0:
        return 0.0
    tprec = int(np.logical_and(sk_p, b).sum()) / sp
    tsens = int(np.logical_and(sk_g, a).sum()) / sg
    if (tprec + tsens) == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def _ap_from_pr(
    scores: np.ndarray,
    is_tp: np.ndarray,
    npig: int,
    recall_thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Average precision + max recall from score-ranked TP/FP flags (COCO-style).

    Args:
        scores: Detection scores, one per detection.
        is_tp: Boolean true-positive flag per detection (FPs are ``False``).
            Ignored detections must be filtered out by the caller.
        npig: Number of (non-ignored) ground-truth objects, the recall
            denominator.
        recall_thresholds: Recall grid for 101-point interpolation.

    Returns:
        ``(AP, max_recall)``. ``AP`` is NaN when ``npig == 0`` (undefined), and
        ``0.0`` when there are no detections.
    """
    if npig <= 0:
        return np.nan, np.nan
    scores = np.asarray(scores, dtype=float)
    is_tp = np.asarray(is_tp, dtype=bool)
    if scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    is_tp = is_tp[order]
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)
    rc = tp / npig
    pr = tp / np.maximum(tp + fp, np.spacing(1))
    recall = float(rc[-1])
    # Make precision monotonically non-increasing as recall grows.
    for i in range(pr.size - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, recall_thresholds, side="left")
    precision = np.zeros(recall_thresholds.shape)
    valid = inds < pr.size
    precision[valid] = pr[inds[valid]]
    return float(precision.mean()), recall


@attrs.define(auto_attribs=True, slots=True)
class MatchInstance:
    """Class to have a new structure for sio.Instance object."""

    instance: sio.Instance
    frame_idx: int
    video_path: str


def _video_key(video: Optional[sio.Video]) -> str:
    """Identify a video by path, with fallbacks for embedded and image-sequence videos.

    Args:
        video: The video to identify, or ``None``.

    Returns:
        ``source_filename`` (embedded videos, which carry their original path)
        else ``filename`` (first entry for image sequences), else a per-object
        identifier so two distinct videos never collide.
    """
    if video is None:
        return "unknown"
    video_path = None
    backend = getattr(video, "backend", None)
    if backend is not None:
        # Try source_filename first (for embedded videos with provenance)
        video_path = getattr(backend, "source_filename", None)
        if video_path is None:
            video_path = getattr(backend, "filename", None)
    # Fallback to video.filename if backend doesn't have it
    if video_path is None:
        video_path = getattr(video, "filename", None)
        # Handle list filenames (image sequences)
        if isinstance(video_path, list) and video_path:
            video_path = video_path[0]
    # Final fallback: use a unique identifier
    if video_path is None:
        return f"video_{id(video)}"
    return str(video_path)


def get_instances(labeled_frame: sio.LabeledFrame) -> List[MatchInstance]:
    """Get a list of instances of type MatchInstance from the Labeled Frame.

    Args:
        labeled_frame: Input Labeled frame of type sio.LabeledFrame.

    Returns:
        List of MatchInstance objects for the given labeled frame.
    """
    instance_list = []
    frame_idx = labeled_frame.frame_idx
    video_path = _video_key(labeled_frame.video)

    for instance in labeled_frame.instances:
        match_instance = MatchInstance(
            instance=instance, frame_idx=frame_idx, video_path=video_path
        )
        instance_list.append(match_instance)
    return instance_list


def _user_centroids(frame: sio.LabeledFrame) -> List[Any]:
    """Return a frame's user (non-predicted) ``Centroid`` annotations."""
    return [c for c in getattr(frame, "centroids", []) or [] if not c.is_predicted]


def _instances_from_user_centroids(frame: sio.LabeledFrame) -> List[sio.Instance]:
    """Represent a frame's user centroid annotations as single-node instances.

    The centroid evaluator's matching, distance and detection metrics all speak
    ``sio.Instance``; a ``Centroid`` annotation carries the same information with
    no skeleton. Wrapping each one in a one-node instance on sleap-io's canonical
    centroid skeleton lets the whole pipeline run unchanged on files that have no
    poses at all -- and the wrapped point is exact, since every reduce method over
    a single point returns that point.
    """
    skeleton = sio.get_centroid_skeleton()
    return [
        sio.Instance.from_numpy(
            np.array([[float(c.x), float(c.y)]], dtype="float64"),
            skeleton=skeleton,
            track=getattr(c, "track", None),
        )
        for c in _user_centroids(frame)
    ]


def find_frame_pairs(
    labels_gt: sio.Labels,
    labels_pr: sio.Labels,
    user_labels_only: bool = True,
    keep_user_centroid_frames: bool = False,
) -> List[Tuple[sio.LabeledFrame, sio.LabeledFrame]]:
    """Find corresponding frames across two sets of labels.

    This function uses sleap-io's robust video matching API to handle various
    scenarios including embedded videos, cross-platform paths, and videos with
    different metadata.

    Args:
        labels_gt: A `sio.Labels` instance with ground truth instances.
        labels_pr: A `sio.Labels` instance with predicted instances.
        keep_user_centroid_frames: If True, a ground-truth frame also survives the
            ``user_labels_only`` filter when it carries user ``Centroid``
            annotations but no user instances. Set by ``match_method="centroid"``:
            centroid annotations (hand-made, or derived from segmentation masks by
            ``data_config.centroids_from_masks``) are the ground truth for a
            centroid model, and a mask-only file has no instances at all -- so the
            instance-only filter dropped every frame and evaluation died with
            "Empty Frame Pairs".
        user_labels_only: If False, frames with predicted instances in `labels_gt` will
            also be considered for matching.

    Returns:
        A list of pairs of `sio.LabeledFrame`s in the form `(frame_gt, frame_pr)`.
    """
    # Use sleap-io's robust video matching API (added in 0.6.2)
    # The match() method returns a MatchResult with video_map: {pred_video: gt_video}
    #
    # NOTE: sleap-io's AUTO matcher previously shape-rejected candidates before its
    # definitive is_same_file check, so it failed to pair an embedded-subset GT video
    # with its restored-original prediction counterpart (same file, different frame
    # count) -- e.g. post-training eval on an embedded .pkg.slp logged "Empty Frame
    # Pairs". This is resolved by the pinned sleap-io (talmolab/sleap-io#473/#476),
    # whose AUTO matcher resolves effective shape through the source_video chain, so
    # the match here works with no workaround.
    match_result = labels_gt.match(labels_pr)

    frame_pairs = []
    # Iterate over matched video pairs (pred_video -> gt_video mapping)
    for video_pr, video_gt in match_result.video_map.items():
        if video_gt is None:
            # No match found for this prediction video
            continue

        # Find labeled frames in this video.
        labeled_frames_gt = labels_gt.find(video_gt)
        if user_labels_only:
            # Build fresh LabeledFrame copies restricted to user instances,
            # rather than mutating `lf.instances` in place -- `labels_gt.find`
            # returns references into the caller's actual Labels object, so
            # mutating it here permanently discards PredictedInstances from
            # ground truth the caller may reuse afterward (e.g. a second
            # Evaluator call with user_labels_only=False on the same labels_gt).
            labeled_frames_gt = [
                attrs.evolve(lf, instances=lf.user_instances)
                for lf in labeled_frames_gt
                if len(lf.user_instances) > 0
                or (keep_user_centroid_frames and _user_centroids(lf))
            ]

        # Attempt to match each labeled frame in the ground truth.
        for labeled_frame_gt in labeled_frames_gt:
            labeled_frames_pr = labels_pr.find(
                video_pr, frame_idx=labeled_frame_gt.frame_idx
            )

            if not labeled_frames_pr:
                # No match
                continue
            elif len(labeled_frames_pr) == 1:
                # Match!
                frame_pairs.append((labeled_frame_gt, labeled_frames_pr[0]))

    return frame_pairs


def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: Optional[float] = None,
    stddev: float = 0.025,
    use_cocoeval: bool = True,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        use_cocoeval: Indicates whether the OKS score is calculated like cocoeval
            method or not. True indicating the score is calculated using the
            cocoeval method (widely used and the code can be found here at
            https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192C5-L233C20)
            and False indicating the score is calculated using the method exactly
            as given in the paper referenced in the Notes below.
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement**2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    if use_cocoeval:
        # If use_cocoeval is True, then compute normalization factor according to cocoeval.
        spread_factor = (2 * stddev) ** 2
        scale_factor = 2 * (scale + np.spacing(1))
    else:
        # If use_cocoeval is False, then compute normalization factor according to the paper.
        spread_factor = stddev**2
        scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    # BROADCAST, don't boolean-index. `ks` is (n_gt, n_pr, n_nodes) while the mask
    # is (n_gt, 1, n_nodes); numpy requires a boolean index to match the indexed
    # array's shape exactly, so `ks[mask] = 0` raised an IndexError for every
    # n_pr > 1 -- i.e. for the (n_gt, n_pr) matrix this function documents and
    # returns. Latent because every in-repo caller passes one prediction at a time.
    ks = np.where(missing_gt[:, None, :], 0.0, ks)

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float32"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


# OKS's normalization scale (the bounding-box area of a GT instance's visible
# keypoints) collapses to exactly 0 when that bbox has zero width or height -- most
# commonly with a single visible keypoint, but also with 2+ keypoints that happen to
# be collinear on an axis. That drives the normalization factor to ~1e-18, which turns
# OKS into a strict bit-for-bit equality test (see scratch/2026-08-21-oks-single-
# keypoint-fn). `match_instances` routes those GT instances through
# `compute_distance_match_score` instead.
_DEGENERATE_AREA_EPS = 1e-9


def compute_distance_match_score(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    pixel_threshold: float = 50.0,
) -> np.ndarray:
    """Compute a pixel-distance-based match score for degenerate-scale GT instances.

    Used as a fallback for GT instances whose visible-keypoint bounding box has zero
    area (see `_DEGENERATE_AREA_EPS`), where `compute_oks` degenerates into a strict
    equality test. Mirrors the pixel-distance matching already used for centroid-only
    models (`match_method="centroid"`), but restricted to the nodes that are visible
    in both the ground truth and predicted instance.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed).
        points_pr: Predicted instances of shape (n_pr, n_nodes, n_ed).
        pixel_threshold: Distance (in pixels) at which the score reaches 0.

    Returns:
        Match scores of shape (n_gt, n_pr) in the range [0, 1], with 1.0 denoting a
        perfect match and 0.0 denoting no jointly-visible nodes or a mean distance at
        or beyond `pixel_threshold`. Comparable in scale to `compute_oks`'s output, so
        the two can be combined and thresholded uniformly.
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    n_gt = points_gt.shape[0]
    n_pr = points_pr.shape[0]
    scores = np.zeros((n_gt, n_pr))
    for i in range(n_gt):
        for j in range(n_pr):
            jointly_visible = ~np.isnan(points_gt[i]).any(axis=-1) & ~np.isnan(
                points_pr[j]
            ).any(axis=-1)
            if not jointly_visible.any():
                continue
            dists = np.linalg.norm(
                points_gt[i, jointly_visible] - points_pr[j, jointly_visible], axis=-1
            )
            mean_dist = float(np.mean(dists))
            scores[i, j] = max(0.0, 1.0 - mean_dist / pixel_threshold)
    return scores


def match_instances(
    frame_gt: sio.LabeledFrame,
    frame_pr: sio.LabeledFrame,
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
    degenerate_pixel_threshold: float = 50.0,
) -> Tuple[List[Tuple[sio.Instance, sio.PredictedInstance, float]], List[sio.Instance]]:
    """Match pairs of instances between ground truth and predictions in a frame.

    Args:
        frame_gt: A `sio.LabeledFrame` with ground truth instances.
        frame_pr: A `sio.LabeledFrame` with predicted instances.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.
        degenerate_pixel_threshold: Pixel distance threshold used to score GT
            instances whose visible-keypoint bounding box has zero area (see
            `compute_distance_match_score`), in place of OKS.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.

    Notes:
        This function uses the approach from the PASCAL VOC scoring procedure. Briefly,
        predictions are sorted descending by their instance-level prediction scores and
        greedily matched to ground truth instances which are then removed from the pool
        of available instances.

        Ground truth instances that remain unmatched are considered false negatives.
    """
    # Sort predicted instances by score.
    frame_pr_match_instances = get_instances(frame_pr)

    scores_pr = np.array(
        [
            m.instance.score
            for m in frame_pr_match_instances
            if hasattr(m.instance, "score")
        ]
    )
    idxs_pr = np.argsort(-scores_pr, kind="mergesort")  # descending
    scores_pr = scores_pr[idxs_pr]

    available_instances_gt = get_instances(frame_gt)
    available_instances_gt_idxs = list(range(len(available_instances_gt)))

    positive_pairs = []
    for idx_pr in idxs_pr:
        # Pull out predicted instance.
        instance_pr = frame_pr_match_instances[idx_pr]

        # Convert instances to point arrays.
        points_pr = np.expand_dims(instance_pr.instance.numpy(), axis=0)
        points_gt = np.stack(
            [
                available_instances_gt[idx].instance.numpy()
                for idx in available_instances_gt_idxs
            ],
            axis=0,
        )

        # Find the best match by computing OKS.
        oks = compute_oks(points_gt, points_pr, stddev=stddev, scale=scale)
        oks = np.squeeze(oks, axis=1)
        assert oks.shape == (len(points_gt),)

        # GT instances with a zero-area visible-keypoint bbox make OKS collapse into a
        # strict equality test (see `_DEGENERATE_AREA_EPS`). Score those against this
        # prediction by pixel distance instead.
        degenerate = compute_instance_area(points_gt) < _DEGENERATE_AREA_EPS
        if degenerate.any():
            distance_scores = compute_distance_match_score(
                points_gt[degenerate],
                points_pr,
                pixel_threshold=degenerate_pixel_threshold,
            )
            oks[degenerate] = np.squeeze(distance_scores, axis=1)

        oks[oks <= threshold] = np.nan
        best_match_gt_idx = np.argsort(-oks, kind="mergesort")[0]
        best_match_oks = oks[best_match_gt_idx]
        if np.isnan(best_match_oks):
            continue

        # Remove matched ground truth instance and add as a positive pair.
        instance_gt_idx = available_instances_gt_idxs.pop(best_match_gt_idx)
        instance_gt = available_instances_gt[instance_gt_idx]
        positive_pairs.append((instance_gt, instance_pr, best_match_oks))

        # Stop matching lower scoring instances if we run out of candidates in the
        # ground truth.
        if not available_instances_gt_idxs:
            break

    # Any remaining ground truth instances are considered false negatives.
    false_negatives = [
        available_instances_gt[idx] for idx in available_instances_gt_idxs
    ]

    return positive_pairs, false_negatives


def match_frame_pairs(
    frame_pairs: List[Tuple[sio.LabeledFrame, sio.LabeledFrame]],
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
) -> Tuple[List[Tuple[sio.Instance, sio.PredictedInstance, float]], List[sio.Instance]]:
    """Match all ground truth and predicted instances within each pair of frames.

    This is a wrapper for `match_instances()` but operates on lists of frames.

    Args:
        frame_pairs: A list of pairs of `sleap.LabeledFrame`s in the form
            `(frame_gt, frame_pr)`. These can be obtained with `find_frame_pairs()`.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sio.Instance`s that could not be
        matched.
    """
    positive_pairs = []
    false_negatives = []
    for frame_gt, frame_pr in frame_pairs:
        positive_pairs_frame, false_negatives_frame = match_instances(
            frame_gt,
            frame_pr,
            stddev=stddev,
            scale=scale,
            threshold=threshold,
        )
        positive_pairs.extend(positive_pairs_frame)
        false_negatives.extend(false_negatives_frame)

    return positive_pairs, false_negatives


def compute_dists(
    positive_pairs: List[Tuple[sio.Instance, sio.PredictedInstance, Any]],
) -> Dict[str, Union[np.ndarray, List[int], List[str]]]:
    """Compute Euclidean distances between matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _)`
            containing the matched pair of instances.

    Returns:
        A dictionary with the following keys:
            dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`
            frame_idxs: A list of frame indices corresponding to the `dists`
            video_paths: A list of video paths corresponding to the `dists`
    """
    dists = []
    frame_idxs = []
    video_paths = []
    for instance_gt, instance_pr, _ in positive_pairs:
        points_gt = instance_gt.instance.numpy()
        points_pr = instance_pr.instance.numpy()

        dists.append(np.linalg.norm(points_pr - points_gt, axis=-1))
        frame_idxs.append(instance_gt.frame_idx)
        video_paths.append(instance_gt.video_path)

    dists = np.array(dists)

    # Bundle everything into a dictionary
    dists_dict = {
        "dists": dists,
        "frame_idxs": frame_idxs,
        "video_paths": video_paths,
    }

    return dists_dict


class Evaluator:
    """Compute the standard evaluation metrics with the predicted and the ground-truth Labels.

    This class is used to calculate the common metrics for pose estimation models which
    includes voc metrics (with oks and pck), mOKS, distance metrics, pck metrics and
    visibility metrics.

    Args:
        ground_truth_instances: The `sio.Labels` dataset object with ground truth labels.
        predicted_instances: The `sio.Labels` dataset object with predicted labels.
        oks_stddev: The standard deviation to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        oks_scale: The scale to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        match_threshold: The threshold to use when determining which instances
            match between ground truth and predicted frames. For
            ``match_method="oks"`` this is an OKS threshold; for
            ``match_method="centroid"`` this is a PIXEL distance threshold.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.
        match_method: Either ``"oks"`` (default, full-skeleton OKS matching) or
            ``"centroid"`` (single-point distance matching for centroid-only /
            single-node predictions).
        anchor_ind: For ``match_method="centroid"``, the index of the GT
            skeleton node used to compute each ground-truth centroid (see
            :func:`compute_gt_centroids` and #586). ``None`` falls back to the
            NaN-ignoring mean of visible nodes.
        centroid_method: For ``match_method="centroid"``, how the GT centroid is
            derived -- ``"center_of_mass"``, ``"bbox_center"``,
            ``"geometric_median"`` or ``"anchor"``. ``None`` (default) infers it
            from ``anchor_ind``. Must match what the model was trained on, or the
            distance metric compares two different definitions of "centroid";
            :func:`run_evaluation` reads it off the training config.
        centroid_fallback: Reduce method used when the anchor node is not visible.

    """

    def __init__(
        self,
        ground_truth_instances: sio.Labels,
        predicted_instances: sio.Labels,
        oks_stddev: float = 0.025,
        oks_scale: Optional[float] = None,
        match_threshold: float = 0,
        user_labels_only: bool = True,
        match_method: str = "oks",
        anchor_ind: Optional[int] = None,
        centroid_method: Optional[str] = None,
        centroid_fallback: Optional[str] = None,
        exclude_predicted_instance_masks: bool = False,
    ):
        """Initialize the Evaluator class with ground-truth and predicted labels.

        ``exclude_predicted_instance_masks`` (``match_method="mask"`` only) drops
        masks linked to a ``PredictedInstance`` from the ground-truth labels, so a
        labels file that carries stray predicted instances (each of which gets a
        mask when masks are built from poses) does not treat them as ground truth.
        It is kept separate from ``user_labels_only`` (which controls the frame-pair
        filter) because mask mode disables that frame filter -- see
        :func:`run_evaluation`.
        """
        self.ground_truth_instances = ground_truth_instances
        self.predicted_instances = predicted_instances
        self.match_threshold = match_threshold
        self.oks_stddev = oks_stddev
        self.oks_scale = oks_scale
        self.user_labels_only = user_labels_only
        self.match_method = match_method
        self.anchor_ind = anchor_ind
        self.centroid_method = centroid_method
        self.centroid_fallback = centroid_fallback
        self.exclude_predicted_instance_masks = exclude_predicted_instance_masks
        # Populated only in centroid / mask mode.
        self.false_positives = []
        # Matched-pair IoUs, populated only in mask mode.
        self.mask_ious = np.array([])
        # Per-frame mask records + matched TP mask pairs, populated only in mask
        # mode (feed mask_voc_metrics / boundary-IoU / fragmentation / per-size).
        self._mask_frames = []
        self._matched_mask_pairs = []
        # Per-frame (iou, cldice, boundary_iou) triples, populated only in
        # match_method="semantic" (whole-frame foreground, no matching).
        self._semantic_rows = []

        self._process_frames()

    def _process_frames(self):
        self.frame_pairs = find_frame_pairs(
            self.ground_truth_instances,
            self.predicted_instances,
            self.user_labels_only,
            keep_user_centroid_frames=self.match_method == "centroid",
        )
        if not self.frame_pairs:
            message = "Empty Frame Pairs. No match found for the video frames"
            logger.error(message)
            raise Exception(message)

        if self.match_method == "centroid":
            self._process_frames_centroid()
            return

        if self.match_method == "mask":
            self._process_frames_mask()
            return

        if self.match_method == "semantic":
            self._process_frames_semantic()
            return

        self.positive_pairs, self.false_negatives = match_frame_pairs(
            self.frame_pairs,
            stddev=self.oks_stddev,
            scale=self.oks_scale,
            threshold=self.match_threshold,
        )

        self.dists_dict = compute_dists(self.positive_pairs)

    def _process_frames_centroid(self):
        """Match predicted vs GT centroids by pixel distance (per frame).

        Each predicted instance is collapsed to its single centroid point (its
        sole visible point / node-0 for a 1-node prediction). Ground-truth
        centroids are computed via :func:`compute_gt_centroids` to exactly
        mirror the centroid target used during training (#586). Matching uses
        :func:`match_centroids` with ``self.match_threshold`` as a PIXEL
        distance. Populates ``positive_pairs`` as ``(gt_inst, pr_inst, dist)``
        3-tuples, ``false_negatives`` (unmatched GT), and ``false_positives``
        (unmatched predictions).
        """
        self.positive_pairs = []
        self.false_negatives = []
        self.false_positives = []

        for frame_gt, frame_pr in self.frame_pairs:
            # A mask-only or centroid-annotation-only ground-truth frame has no
            # instances; its centroids ARE the ground truth (#586).
            if not get_instances(frame_gt) and _user_centroids(frame_gt):
                frame_gt = attrs.evolve(
                    frame_gt, instances=_instances_from_user_centroids(frame_gt)
                )
                gt_from_centroid_annotations = True
            else:
                gt_from_centroid_annotations = False
            gt_match_instances = get_instances(frame_gt)
            pr_match_instances = get_instances(frame_pr)

            # Collapse each predicted instance to its single centroid point.
            pred_centroids = np.array(
                [
                    self._collapse_pred_centroid(m.instance.numpy())
                    for m in pr_match_instances
                ]
            ).reshape(-1, 2)

            # GT centroids come from generate_centroids itself (#586) -- except
            # when they came from `Centroid` annotations, which are already the
            # centroid: the wrapper is one node, so `anchor_ind` (an index into
            # the POSE skeleton) does not apply to it.
            gt_centroids = np.array(
                [
                    compute_gt_centroids(
                        m.instance.numpy(),
                        None if gt_from_centroid_annotations else self.anchor_ind,
                        method=(
                            None
                            if gt_from_centroid_annotations
                            else self.centroid_method
                        ),
                        fallback=(
                            None
                            if gt_from_centroid_annotations
                            else self.centroid_fallback
                        ),
                    )
                    for m in gt_match_instances
                ]
            ).reshape(-1, 2)

            # Drop NaN centroids before Hungarian matching: scipy's cdist /
            # linear_sum_assignment reject NaN, and a fully-occluded (all-NaN)
            # GT instance is common in real labels. Index maps translate the
            # filtered match indices back to the original instance lists so
            # FN/FP/positive-pair attribution stays correct. (A NaN-row GT is
            # counted as an automatic false negative — matching the legacy
            # CentroidEvaluationCallback; a NaN-row prediction is not a real
            # detection and is simply excluded.)
            gt_valid = ~np.isnan(gt_centroids).any(axis=1)
            pred_valid = ~np.isnan(pred_centroids).any(axis=1)
            gt_map = np.flatnonzero(gt_valid)
            pred_map = np.flatnonzero(pred_valid)

            matched_pred, matched_gt, unmatched_pred, unmatched_gt = match_centroids(
                pred_centroids[pred_valid],
                gt_centroids[gt_valid],
                max_distance=self.match_threshold,
            )

            for p_local, g_local in zip(matched_pred, matched_gt):
                p_idx = int(pred_map[int(p_local)])
                g_idx = int(gt_map[int(g_local)])
                dist = float(
                    np.linalg.norm(pred_centroids[p_idx] - gt_centroids[g_idx])
                )
                self.positive_pairs.append(
                    (gt_match_instances[g_idx], pr_match_instances[p_idx], dist)
                )

            for g_local in unmatched_gt:
                self.false_negatives.append(
                    gt_match_instances[int(gt_map[int(g_local)])]
                )
            # Fully-occluded (all-NaN) GT instances -> automatic false negatives.
            for g_idx in np.flatnonzero(~gt_valid):
                self.false_negatives.append(gt_match_instances[int(g_idx)])

            for p_local in unmatched_pred:
                self.false_positives.append(
                    pr_match_instances[int(pred_map[int(p_local)])]
                )

        # Build the dists dict directly from matched-pair centroid distances so
        # distance_metrics() works uniformly across match methods.
        dists = np.array([dist for _, _, dist in self.positive_pairs])
        self.dists_dict = {
            "dists": dists,
            "frame_idxs": [gt.frame_idx for gt, _, _ in self.positive_pairs],
            "video_paths": [gt.video_path for gt, _, _ in self.positive_pairs],
        }

    def _process_frames_mask(self):
        """Match predicted vs GT segmentation masks by IoU (per frame).

        Pulls per-instance boolean masks from ``LabeledFrame.masks`` on each
        paired frame and matches them with :func:`match_masks` using
        ``self.match_threshold`` as the IoU threshold. Populates
        ``positive_pairs`` as ``(frame_gt, frame_pr, iou)`` 3-tuples (the frame
        objects are stored only as tokens; detection counting uses the list
        lengths, and per-pair IoUs feed :meth:`mask_metrics`), plus
        ``false_negatives`` (unmatched GT masks) and ``false_positives``
        (unmatched predicted masks). No keypoint distances exist for masks, so
        ``dists_dict`` is left empty (``distance_metrics`` reports NaN; IoU is
        reported via :meth:`mask_metrics`).
        """
        self.positive_pairs = []
        self.false_negatives = []
        self.false_positives = []
        ious: List[float] = []
        # Per-frame decoded masks + scores + IoU/intersection matrices, reused by
        # mask_voc_metrics (score-ranked COCO AP) and the fragmentation/per-size
        # breakdowns without re-decoding RLE masks.
        self._mask_frames = []
        # Matched (pred_mask, gt_mask) TP pairs (aligned to ``self.mask_ious``),
        # used for boundary-IoU scoring.
        self._matched_mask_pairs = []

        for frame_gt, frame_pr in self.frame_pairs:
            # Ground-truth masks drop any PredictedInstance-linked masks when the
            # caller asked for user-only labels; predicted-side masks are the
            # model's output and are always kept in full.
            gt_masks = _frame_masks(
                frame_gt,
                drop_predicted_instances=self.exclude_predicted_instance_masks,
            )
            pr_masks = _frame_masks(frame_pr)
            pr_scores = _frame_pred_scores(frame_pr)
            iou_mat, inter_mat = _mask_pair_stats(pr_masks, gt_masks)
            self._mask_frames.append(
                {
                    "pred_masks": pr_masks,
                    "pred_scores": pr_scores,
                    "gt_masks": gt_masks,
                    "iou": iou_mat,
                    "inter": inter_mat,
                    "gt_areas": np.array([int(m.sum()) for m in gt_masks], dtype=float),
                    "pred_areas": np.array(
                        [int(m.sum()) for m in pr_masks], dtype=float
                    ),
                }
            )

            matched_pred, matched_gt, unmatched_pred, unmatched_gt, pair_ious = (
                match_masks(pr_masks, gt_masks, min_iou=self.match_threshold)
            )

            for iou in pair_ious:
                self.positive_pairs.append((frame_gt, frame_pr, float(iou)))
                ious.append(float(iou))
            for p_idx, g_idx in zip(matched_pred, matched_gt):
                self._matched_mask_pairs.append(
                    (pr_masks[int(p_idx)], gt_masks[int(g_idx)])
                )
            for _ in unmatched_gt:
                self.false_negatives.append(frame_gt)
            for _ in unmatched_pred:
                self.false_positives.append(frame_pr)

        self.mask_ious = np.asarray(ious, dtype=float)
        self.dists_dict = {"dists": np.array([]), "frame_idxs": [], "video_paths": []}

    def _process_frames_semantic(self):
        """Whole-frame foreground evaluation (no instance matching).

        For semantic (binary foreground/background) segmentation there is a single
        foreground mask per frame and no instance grouping, so there is nothing to
        match. Each paired frame's predicted and ground-truth masks are unioned
        into one foreground mask (:func:`_union_frame_fg`) and scored directly with
        :func:`_mask_iou`, :func:`mask_cldice`, and :func:`_boundary_iou`. Frames
        whose GROUND-TRUTH foreground is empty are skipped (there is no foreground
        to score).

        Populates ``self._semantic_rows`` as ``(iou, cldice, boundary_iou)``
        triples (consumed by :meth:`semantic_metrics`). The matching-based
        attributes (``positive_pairs`` / ``false_negatives`` / ``false_positives``
        / ``dists_dict``) are left empty so the shared plumbing degrades gracefully
        (semantic mode reports only ``semantic_metrics``).
        """
        self.positive_pairs = []
        self.false_negatives = []
        self.false_positives = []
        self._semantic_rows = []

        for frame_gt, frame_pr in self.frame_pairs:
            gt_fg = _union_frame_fg(frame_gt)
            if not gt_fg.any():
                # No ground-truth foreground: nothing to score on this frame.
                continue
            pr_fg = _union_frame_fg(frame_pr)
            iou = _mask_iou(pr_fg, gt_fg)
            cldice = mask_cldice(pr_fg, gt_fg)
            biou = _boundary_iou(pr_fg, gt_fg)
            self._semantic_rows.append((iou, cldice, biou))

        self.dists_dict = {"dists": np.array([]), "frame_idxs": [], "video_paths": []}

    @staticmethod
    def _collapse_pred_centroid(points: np.ndarray) -> np.ndarray:
        """Collapse a predicted instance to its single centroid point.

        For a 1-node ('centroid') prediction this is node-0. For predictions
        with multiple nodes (e.g. a single-instance model used as a detector)
        we take the single visible point, falling back to node-0.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        visible = ~np.isnan(points).any(axis=-1)
        if visible.any():
            return points[np.argmax(visible)]
        return points[0]

    def voc_metrics(
        self,
        match_score_by="oks",
        match_score_thresholds: np.ndarray = np.linspace(
            0.5, 0.95, 10
        ),  # 0.5:0.05:0.95
        recall_thresholds: np.ndarray = np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    ):
        """Compute VOC metrics for a matched pairs of instances positive pairs and false negatives.

        Args:
            match_score_by: The score to be used for computing the metrics. "ock" or "pck"
            match_score_thresholds: Score thresholds at which to consider matches as a true
                positive match.
            recall_thresholds: Recall thresholds at which to evaluate Average Precision.

        Returns:
            A dictionary of VOC metrics.
        """
        if match_score_by == "oks":
            match_scores = np.array([oks for _, _, oks in self.positive_pairs])
            name = "oks_voc"
        elif match_score_by == "pck":
            name = "pck_voc"
            if not self.positive_pairs:
                # Guard the empty-match case: the (n_pairs, n_nodes, n_thresholds)
                # ``pcks`` array is empty along the pairs axis, so reducing it with
                # nested .mean() calls would hit "Mean of empty slice".
                match_scores = np.array([])
            else:
                pck_metrics = self.pck_metrics()
                match_scores = pck_metrics["pcks"].mean(axis=-1).mean(axis=-1)
        else:
            message = "Invalid Option for match_score_by. Choose either `oks` or `pck`"
            logger.error(message)
            raise Exception(message)

        detection_scores = np.array(
            [pp[1].instance.score for pp in self.positive_pairs]
        )

        inds = np.argsort(-detection_scores, kind="mergesort")
        detection_scores = detection_scores[inds]
        match_scores = match_scores[inds]

        precisions = []
        recalls = []

        npig = len(self.positive_pairs) + len(
            self.false_negatives
        )  # total number of GT instances

        for match_score_threshold in match_score_thresholds:
            tp = np.cumsum(match_scores >= match_score_threshold)
            fp = np.cumsum(match_scores < match_score_threshold)

            if tp.size == 0:
                return {
                    name + ".match_score_thresholds": 0,
                    name + ".recall_thresholds": 0,
                    name + ".match_scores": 0,
                    name + ".precisions": 0,
                    name + ".recalls": 0,
                    name + ".AP": 0,
                    name + ".AR": 0,
                    name + ".mAP": 0,
                    name + ".mAR": 0,
                }

            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))

            recall = rc[-1]  # best recall at this OKS threshold

            # Ensure strictly decreasing precisions.
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            # Find best precision at each recall threshold.
            rc_inds = np.searchsorted(rc, recall_thresholds, side="left")
            precision = np.zeros(rc_inds.shape)
            is_valid_rc_ind = rc_inds < len(pr)
            precision[is_valid_rc_ind] = pr[rc_inds[is_valid_rc_ind]]

            precisions.append(precision)
            recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        AP = precisions.mean(
            axis=1
        )  # AP = average precision over fixed set of recall thresholds
        AR = recalls  # AR = max recall given a fixed number of detections per image

        mAP = precisions.mean()  # mAP = mean over all OKS thresholds
        mAR = recalls.mean()  # mAR = mean over all OKS thresholds

        return {
            name + ".match_score_thresholds": match_score_thresholds,
            name + ".recall_thresholds": recall_thresholds,
            name + ".match_scores": match_scores,
            name + ".precisions": precisions,
            name + ".recalls": recalls,
            name + ".AP": AP,
            name + ".AR": AR,
            name + ".mAP": mAP,
            name + ".mAR": mAR,
        }

    def mOKS(self):
        """Return the meanOKS value."""
        pair_oks = np.array([oks for _, _, oks in self.positive_pairs])
        return {"mOKS": float(pair_oks.mean()) if pair_oks.size else np.nan}

    def distance_metrics(self):
        """Compute the Euclidean distance error at different percentiles using the pairwise distances.

        Returns:
            A dictionary of distance metrics.
        """
        dists = self.dists_dict["dists"]
        results = {
            "frame_idxs": self.dists_dict["frame_idxs"],
            "video_paths": self.dists_dict["video_paths"],
            "dists": dists,
            # Guard the empty / all-NaN matched set (zero true positives in a
            # split) so np.nanmean doesn't emit a "Mean of empty slice" warning.
            "avg": (
                float(np.nanmean(dists))
                if np.asarray(dists).size and not np.all(np.isnan(dists))
                else np.nan
            ),
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

        is_non_nan = ~np.isnan(dists)
        if np.any(is_non_nan):
            non_nans = dists[is_non_nan]
            for ptile in (50, 75, 90, 95, 99):
                results[f"p{ptile}"] = np.percentile(non_nans, ptile)

        return results

    def detection_metrics(self) -> dict:
        """Compute detection metrics (precision/recall/F1) over TP/FP/FN counts.

        Used by both ``match_method="centroid"`` and ``match_method="mask"``
        (it only reads the matched/unmatched list lengths and ``dists_dict``).
        Mirrors ``CentroidEvaluationCallback._compute_metrics``. For centroid
        mode the localization-error percentiles are computed over the Euclidean
        distances of matched centroid pairs; for mask mode ``dists_dict`` is
        empty so those percentiles are NaN (per-pair IoU is reported separately
        via :meth:`mask_metrics`). Not used for ``match_method="oks"`` (which
        reports OKS-based VOC metrics instead).

        Returns:
            A dict with ``precision``, ``recall``, ``f1``, ``n_tp``, ``n_fp``,
            ``n_fn`` and localization-error percentiles ``avg``/``p50``/``p75``/
            ``p90``/``p95``/``p99`` (NaN when there are no matched pairs).
        """
        n_tp = len(self.positive_pairs)
        n_fp = len(self.false_positives)
        n_fn = len(self.false_negatives)

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        dists = self.dists_dict["dists"]
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "avg": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

        is_non_nan = ~np.isnan(dists) if len(dists) else np.array([], dtype=bool)
        if np.any(is_non_nan):
            non_nans = dists[is_non_nan]
            results["avg"] = float(np.mean(non_nans))
            for ptile in (50, 75, 90, 95, 99):
                results[f"p{ptile}"] = float(np.percentile(non_nans, ptile))

        return results

    def mask_metrics(self) -> dict:
        """Compute mask-IoU summary statistics for ``match_method="mask"``.

        Reports complementary IoU summaries, panoptic-quality, boundary-IoU,
        fragmentation, and per-object-size breakdowns:

        * ``mean_iou`` (and ``min``/``max``/percentiles) over the matched (TP)
          pairs only — COCO-style segmentation quality, blind to misses.
        * ``mean_iou_all_gt`` — IoU averaged over *all* ground-truth masks,
          where an unmatched GT (a miss) contributes ``0``. This penalizes
          recall and complements the TP-only mean.
        * Panoptic Quality ``pq = sq * rq`` with ``sq = mean_iou`` (segmentation
          quality) and ``rq = TP / (TP + 0.5*FP + 0.5*FN)`` (recognition
          quality, == detection F1). See Kirillov et al., "Panoptic
          Segmentation" (2019).
        * ``mean_boundary_iou`` — boundary IoU over the matched pairs (Cheng et
          al., 2021), more sensitive to contour error than mask IoU.
        * ``mean_cldice`` — centerline Dice over the matched pairs (Shit et al.,
          CVPR 2021), connectivity-aware and nearly width-insensitive; a fairer
          score than IoU for thin/tubular structures. NaN if scikit-image is
          unavailable.
        * ``oversegmentation`` / ``undersegmentation`` — fragmentation counts:
          GT masks split across >=2 predictions, and predictions spanning >=2
          GT masks (each with >=10% area overlap). The headline over-/under-
          segmentation failure mode is invisible to the 1-to-1 match.
        * ``per_size`` — COCO small/medium/large breakdown of GT count, TP
          count, and TP-only mean IoU (buckets sum to the GT total).

        Returns:
            A dict with ``mean_iou``, ``min``, ``max``, percentiles ``p25``/
            ``p50``/``p75``, ``mean_iou_all_gt``, ``pq``/``sq``/``rq``,
            ``mean_boundary_iou``, ``oversegmentation``/``undersegmentation``,
            ``per_size``, the TP count ``n_matched`` (plus ``n_fp``/``n_fn``),
            and the raw ``ious`` array. Quantities are NaN when undefined.
        """
        ious = np.asarray(self.mask_ious, dtype=float)
        n_tp = len(self.positive_pairs)
        n_fp = len(self.false_positives)
        n_fn = len(self.false_negatives)
        over, under = self._fragmentation_counts()
        results = {
            "mean_iou": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "mean_iou_all_gt": np.nan,
            "pq": np.nan,
            "sq": np.nan,
            "rq": np.nan,
            "mean_boundary_iou": np.nan,
            "mean_cldice": np.nan,
            "oversegmentation": over,
            "undersegmentation": under,
            "per_size": self._mask_per_size_stats(),
            "n_matched": int(ious.size),
            "n_fp": n_fp,
            "n_fn": n_fn,
            "ious": ious,
        }
        if ious.size:
            results["mean_iou"] = float(np.mean(ious))
            results["min"] = float(np.min(ious))
            results["max"] = float(np.max(ious))
            for ptile in (25, 50, 75):
                results[f"p{ptile}"] = float(np.percentile(ious, ptile))

        if self._matched_mask_pairs:
            boundary_ious = np.array(
                [_boundary_iou(p, g) for p, g in self._matched_mask_pairs],
                dtype=float,
            )
            results["mean_boundary_iou"] = float(np.mean(boundary_ious))
            # Centerline Dice (clDice): connectivity/width-tolerant, fairer than
            # IoU for thin structures. NaN entries (scikit-image missing) drop out.
            cldices = np.array(
                [mask_cldice(p, g) for p, g in self._matched_mask_pairs],
                dtype=float,
            )
            cldices = cldices[~np.isnan(cldices)]
            if cldices.size:
                results["mean_cldice"] = float(np.mean(cldices))

        iou_sum = float(np.sum(ious)) if ious.size else 0.0
        # Miss-penalizing mean: averaged over every GT mask (TP + FN).
        n_gt = n_tp + n_fn
        if n_gt > 0:
            results["mean_iou_all_gt"] = iou_sum / n_gt
        # Panoptic quality: SQ = TP-only mean IoU, RQ = detection F1, PQ = SQ*RQ
        # = iou_sum / (TP + 0.5*FP + 0.5*FN).
        pq_denom = n_tp + 0.5 * n_fp + 0.5 * n_fn
        if pq_denom > 0:
            results["sq"] = results["mean_iou"]
            results["rq"] = n_tp / pq_denom
            results["pq"] = iou_sum / pq_denom
        return results

    def semantic_metrics(self) -> dict:
        """Aggregate whole-frame foreground metrics for ``match_method="semantic"``.

        Averages the per-frame foreground IoU, centerline Dice (clDice), and
        boundary IoU computed by :meth:`_process_frames_semantic` over all frames
        with non-empty ground-truth foreground. clDice entries that are NaN
        (scikit-image unavailable) are dropped from the clDice mean; if every entry
        is NaN the reported ``mean_cldice`` is NaN.

        Returns:
            A dict with ``mean_iou``, ``mean_cldice``, ``mean_boundary_iou``, the
            per-frame ``ious`` / ``cldices`` / ``boundary_ious`` arrays, and
            ``n_frames`` (frames scored). Means are NaN when no frame was scored.
        """
        rows = np.asarray(self._semantic_rows, dtype=float).reshape(-1, 3)
        ious = rows[:, 0]
        cldices = rows[:, 1]
        bious = rows[:, 2]
        cld_valid = cldices[~np.isnan(cldices)]
        return {
            "mean_iou": float(np.mean(ious)) if ious.size else float("nan"),
            "mean_cldice": (
                float(np.mean(cld_valid)) if cld_valid.size else float("nan")
            ),
            "mean_boundary_iou": (
                float(np.mean(bious)) if bious.size else float("nan")
            ),
            "ious": ious,
            "cldices": cldices,
            "boundary_ious": bious,
            "n_frames": int(ious.size),
        }

    def _fragmentation_counts(self, overlap_frac: float = 0.1) -> Tuple[int, int]:
        """Count over-/under-segmented instances across all mask frames.

        A prediction "covers" a GT mask when their intersection is at least
        ``overlap_frac`` of the GT area. Over-segmentation counts GT masks
        covered by >=2 predictions (one animal split into fragments);
        under-segmentation counts predictions covering >=2 GT masks (one mask
        merging neighbors). Both directly surface the failure mode the 1-to-1
        Hungarian match hides (extra fragments otherwise just become FPs).
        """
        over = under = 0
        for f in self._mask_frames:
            inter = f["inter"]
            gt_areas = f["gt_areas"]
            n_pred, n_gt = inter.shape
            if n_pred == 0 or n_gt == 0:
                continue
            # Fraction of each GT (cols) covered by each prediction (rows).
            cov_gt = inter / np.maximum(gt_areas[None, :], 1.0)
            covers = cov_gt >= overlap_frac
            over += int(np.count_nonzero(covers.sum(axis=0) >= 2))  # GT split
            under += int(np.count_nonzero(covers.sum(axis=1) >= 2))  # pred merged
        return over, under

    def _per_size_breakdown(
        self,
        gt_areas_all: np.ndarray,
        tp_iou: np.ndarray,
        tp_gt_area: np.ndarray,
        edges: np.ndarray,
    ) -> dict:
        """small/medium/large GT count, TP count and TP mean IoU under ``edges``.

        ``n_gt`` over the three buckets sums to the total GT count (every GT area
        falls in exactly one half-open bucket).
        """
        out = {"edges": [float(e) for e in edges]}
        for idx, bucket in enumerate(_SIZE_KEYS):
            in_gt = _size_mask(gt_areas_all, idx, edges)
            in_tp = (
                _size_mask(tp_gt_area, idx, edges)
                if tp_gt_area.size
                else np.array([], dtype=bool)
            )
            out[bucket] = {
                "n_gt": int(np.count_nonzero(in_gt)),
                "n_tp": int(np.count_nonzero(in_tp)),
                "mean_iou": (
                    float(np.mean(tp_iou[in_tp])) if np.any(in_tp) else np.nan
                ),
            }
        return out

    def _mask_per_size_stats(self) -> dict:
        """Per-object-size GT/TP/IoU breakdown under both bucketing schemes.

        GT objects are bucketed by mask area (``mask.sum()``). The primary
        scheme (top-level ``small``/``medium``/``large`` keys) uses
        dataset-relative percentile edges (terciles by default) so the buckets
        adapt to the actual mask scale; the COCO fixed-cutoff scheme (small <
        32^2 <= medium < 96^2 <= large) is reported additionally under
        ``"coco"`` for cross-dataset comparability.
        """
        gt_areas_all = np.array(
            [a for f in self._mask_frames for a in f["gt_areas"]], dtype=float
        )
        tp_iou = np.asarray(self.mask_ious, dtype=float)
        tp_gt_area = np.array(
            [int(g.sum()) for _, g in self._matched_mask_pairs], dtype=float
        )
        pct_edges = _percentile_size_edges(gt_areas_all)
        out = self._per_size_breakdown(gt_areas_all, tp_iou, tp_gt_area, pct_edges)
        out["scheme"] = "percentile"
        out["coco"] = self._per_size_breakdown(
            gt_areas_all, tp_iou, tp_gt_area, COCO_SIZE_EDGES
        )
        return out

    def _match_masks_coco(
        self, iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Greedy score-ranked pred->GT matching at one IoU threshold (COCO).

        Per frame, predictions are considered in descending score order; each
        claims the highest-IoU not-yet-claimed GT whose IoU >= ``iou_threshold``
        (a TP), else it is a FP. Mirrors ``pycocotools`` matching.

        Returns:
            ``(scores, matched, matched_gt_area, pred_area)`` flat arrays over
            every prediction across all frames (aligned). ``matched`` is the
            TP flag; ``matched_gt_area`` is the area of the claimed GT (NaN for
            a FP); ``pred_area`` is the prediction's own area.
        """
        scores, matched, matched_gt_area, pred_area = [], [], [], []
        for f in self._mask_frames:
            iou = f["iou"]
            pred_scores = f["pred_scores"]
            gt_areas = f["gt_areas"]
            pred_areas = f["pred_areas"]
            n_pred, n_gt = iou.shape
            order = (
                np.argsort(-pred_scores, kind="mergesort")
                if n_pred
                else np.array([], dtype=int)
            )
            gt_taken = np.zeros(n_gt, dtype=bool)
            for p in order:
                scores.append(float(pred_scores[p]))
                pred_area.append(float(pred_areas[p]))
                if n_gt == 0:
                    matched.append(False)
                    matched_gt_area.append(np.nan)
                    continue
                row = iou[p].copy()
                row[gt_taken] = -1.0
                g = int(np.argmax(row))
                if row[g] >= iou_threshold:
                    gt_taken[g] = True
                    matched.append(True)
                    matched_gt_area.append(float(gt_areas[g]))
                else:
                    matched.append(False)
                    matched_gt_area.append(np.nan)
        return (
            np.array(scores, dtype=float),
            np.array(matched, dtype=bool),
            np.array(matched_gt_area, dtype=float),
            np.array(pred_area, dtype=float),
        )

    def mask_voc_metrics(
        self,
        iou_thresholds: np.ndarray = MASK_IOU_THRESHOLDS,
        recall_thresholds: np.ndarray = np.linspace(0, 1, 101),
        size_percentiles: Tuple[float, float] = DEFAULT_SIZE_PERCENTILES,
    ) -> dict:
        """COCO-style score-ranked mask Average Precision / Recall.

        Re-matches predictions to GT independently at each IoU threshold
        (:meth:`_match_masks_coco`), score-ranks the resulting TP/FP flags, and
        integrates the precision-recall curve (101-point interpolation, mirrors
        :meth:`voc_metrics`). Reports overall AP@[.5:.95]/AP50/AP75/AR plus a
        per-object-size AP breakdown under two bucketing schemes (GT outside a
        bucket is ignored, as in ``pycocotools`` ``areaRng``): the primary
        (default) buckets use dataset-relative percentile edges (terciles), and
        the COCO fixed-cutoff buckets are reported additionally under the
        ``mask_voc.coco.`` prefix — analogous to the dual OKS/PCK VOC.

        Args:
            iou_thresholds: IoU thresholds to average AP over.
            recall_thresholds: Recall grid for 101-point interpolation.
            size_percentiles: Two percentiles of the GT area distribution
                delimiting the primary small/medium/large buckets.

        Returns:
            A dict keyed under ``"mask_voc."``: ``AP`` (per-threshold array),
            ``mAP``, ``AP50``, ``AP75``, ``AR``, ``recalls``, ``iou_thresholds``,
            ``n_gt``; primary per-size ``AP_small``/``AP_medium``/``AP_large``,
            ``n_gt_small``/``..._medium``/``..._large``, ``size_scheme`` and
            ``size_edges``; and COCO per-size ``coco.AP_*``/``coco.n_gt_*``/
            ``coco.size_edges``. AP values are NaN when the relevant GT set is
            empty.
        """
        iou_thresholds = np.asarray(iou_thresholds, dtype=float)
        recall_thresholds = np.asarray(recall_thresholds, dtype=float)
        gt_areas_all = np.array(
            [a for f in self._mask_frames for a in f["gt_areas"]], dtype=float
        )
        npig = int(gt_areas_all.size)

        # Primary (percentile, dataset-relative) + additional (COCO) edges.
        schemes = {
            "percentile": _percentile_size_edges(gt_areas_all, size_percentiles),
            "coco": COCO_SIZE_EDGES,
        }
        n_gt_size = {
            name: [
                int(np.count_nonzero(_size_mask(gt_areas_all, i, edges)))
                for i in range(len(_SIZE_KEYS))
            ]
            for name, edges in schemes.items()
        }

        ap_overall = np.full(iou_thresholds.size, np.nan)
        recall_overall = np.full(iou_thresholds.size, np.nan)
        ap_size = {
            name: [np.full(iou_thresholds.size, np.nan) for _ in _SIZE_KEYS]
            for name in schemes
        }

        for ti, thr in enumerate(iou_thresholds):
            scores, matched, matched_gt_area, pred_area = self._match_masks_coco(
                float(thr)
            )
            ap_overall[ti], recall_overall[ti] = _ap_from_pr(
                scores, matched, npig, recall_thresholds
            )
            for name, edges in schemes.items():
                for i in range(len(_SIZE_KEYS)):
                    # COCO areaRng: keep TPs whose matched GT is in-bucket and
                    # FPs whose own area is in-bucket; ignore everything else.
                    keep_tp = matched & _size_mask(matched_gt_area, i, edges)
                    keep_fp = (~matched) & _size_mask(pred_area, i, edges)
                    keep = keep_tp | keep_fp
                    ap_size[name][i][ti], _ = _ap_from_pr(
                        scores[keep],
                        keep_tp[keep],
                        n_gt_size[name][i],
                        recall_thresholds,
                    )

        def _nanmean(arr: np.ndarray) -> float:
            return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan

        def _at(target: float) -> float:
            return float(ap_overall[int(np.argmin(np.abs(iou_thresholds - target)))])

        results = {
            "mask_voc.iou_thresholds": iou_thresholds,
            "mask_voc.AP": ap_overall,
            "mask_voc.recalls": recall_overall,
            "mask_voc.mAP": _nanmean(ap_overall),
            "mask_voc.AR": _nanmean(recall_overall),
            "mask_voc.AP50": _at(0.5),
            "mask_voc.AP75": _at(0.75),
            "mask_voc.n_gt": npig,
            "mask_voc.size_scheme": "percentile",
            "mask_voc.size_edges": [float(e) for e in schemes["percentile"]],
            "mask_voc.coco.size_edges": [float(e) for e in schemes["coco"]],
        }
        # Primary (percentile) per-size keys are unprefixed; COCO is additional.
        for name, prefix in (("percentile", "mask_voc."), ("coco", "mask_voc.coco.")):
            for i, bucket in enumerate(_SIZE_KEYS):
                results[f"{prefix}AP_{bucket}"] = _nanmean(ap_size[name][i])
                results[f"{prefix}n_gt_{bucket}"] = n_gt_size[name][i]
        return results

    def pck_metrics(self, thresholds: np.ndarray = np.linspace(1, 10, 10)):
        """Compute PCK across a range of thresholds using the pair-wise distances.

        Args:
            thresholds: A list of distance thresholds in pixels.

        Returns:
            A dictionary of PCK metrics evaluated at each threshold.
        """
        dists = self.dists_dict["dists"]
        dists = np.copy(dists)
        dists[np.isnan(dists)] = np.inf
        pcks = np.expand_dims(dists, -1) < np.reshape(thresholds, (1, 1, -1))

        # Guard the empty-match case (0 positive pairs for the whole split) so
        # the nested .mean() reductions below don't hit "Mean of empty slice".
        if dists.size == 0:
            mPCK_parts = np.array([])
            mPCK = np.nan
            pck5 = np.nan
            pck10 = np.nan
        else:
            mPCK_parts = pcks.mean(axis=0).mean(axis=-1)
            mPCK = float(mPCK_parts.mean())

            # Precompute PCK at common thresholds
            idx_5 = np.argmin(np.abs(thresholds - 5))
            idx_10 = np.argmin(np.abs(thresholds - 10))
            pck5 = float(pcks[:, :, idx_5].mean())
            pck10 = float(pcks[:, :, idx_10].mean())

        return {
            "thresholds": thresholds,
            "pcks": pcks,
            "mPCK_parts": mPCK_parts,
            "mPCK": mPCK,
            "PCK@5": pck5,
            "PCK@10": pck10,
        }

    def visibility_metrics(self):
        """Compute node visibility metrics for the matched pair of instances.

        Returns:
            A dictionary of visibility metrics, including the confusion matrix.
        """
        vis_tp = 0
        vis_fn = 0
        vis_fp = 0
        vis_tn = 0

        for instance_gt, instance_pr, _ in self.positive_pairs:
            missing_nodes_gt = np.isnan(instance_gt.instance.numpy()).any(axis=-1)
            missing_nodes_pr = np.isnan(instance_pr.instance.numpy()).any(axis=-1)

            vis_tn += ((missing_nodes_gt) & (missing_nodes_pr)).sum()
            vis_fn += ((~missing_nodes_gt) & (missing_nodes_pr)).sum()
            vis_fp += ((missing_nodes_gt) & (~missing_nodes_pr)).sum()
            vis_tp += ((~missing_nodes_gt) & (~missing_nodes_pr)).sum()

        return {
            "tp": vis_tp,
            "fp": vis_fp,
            "tn": vis_tn,
            "fn": vis_fn,
            "precision": vis_tp / (vis_tp + vis_fp) if (vis_tp + vis_fp) else np.nan,
            "recall": vis_tp / (vis_tp + vis_fn) if (vis_tp + vis_fn) else np.nan,
        }

    def evaluate(self):
        """Return the evaluation metrics."""
        if self.match_method == "centroid":
            # Single-node / centroid-only: OKS/PCK/mOKS/visibility are
            # degenerate for one node, so we only report detection +
            # distance metrics. We intentionally do NOT compute OKS for a
            # single node (no magic OKS-scale constant) — the OKS path stays
            # only for match_method="oks".
            return {
                "detection_metrics": self.detection_metrics(),
                "distance_metrics": self.distance_metrics(),
            }

        if self.match_method == "mask":
            # Instance segmentation: detection (precision/recall/F1 over
            # IoU-matched masks) + mask-IoU quality + COCO-style score-ranked
            # mask AP/AR. OKS/PCK/visibility are keypoint-only and not computed.
            return {
                "detection_metrics": self.detection_metrics(),
                "mask_metrics": self.mask_metrics(),
                "mask_voc_metrics": self.mask_voc_metrics(),
            }

        if self.match_method == "semantic":
            # Whole-frame binary foreground segmentation: no instances to match, so
            # report only matching-free foreground IoU / clDice / boundary-IoU.
            return {"semantic_metrics": self.semantic_metrics()}

        if not self.positive_pairs:
            # 0 matched instances for the whole split (e.g. a collapsed model
            # predicting nothing, or predictions that never clear the OKS
            # threshold) -- every metric below is undefined by construction.
            # The individual methods already guard their own NaN/empty-array
            # math, so this is just one clear line instead of relying on the
            # reader to infer "collapsed model" from a wall of NaNs.
            logger.info(
                "0 matched instances: metrics undefined (model predicted "
                "nothing usable, or training likely collapsed)."
            )

        metrics = {}
        metrics["voc_metrics"] = self.voc_metrics(match_score_by="oks")
        metrics["voc_metrics"].update(self.voc_metrics(match_score_by="pck"))
        metrics["mOKS"] = self.mOKS()
        metrics["distance_metrics"] = self.distance_metrics()
        metrics["pck_metrics"] = self.pck_metrics()
        metrics["visibility_metrics"] = self.visibility_metrics()

        return metrics


def _find_metrics_file(model_dir: Path, split: str, dataset_idx: int) -> Path:
    """Find the metrics file in a model directory.

    Tries new naming format first, then falls back to old format.
    If split is "test" and not found, falls back to "val".
    """
    # Try new naming format first: metrics.{split}.{idx}.npz
    metrics_path = model_dir / f"metrics.{split}.{dataset_idx}.npz"
    if metrics_path.exists():
        return metrics_path

    # Fall back to old naming format: {split}_{idx}_pred_metrics.npz
    metrics_path = model_dir / f"{split}_{dataset_idx}_pred_metrics.npz"
    if metrics_path.exists():
        return metrics_path

    # If split is "test" and not found, try "val" fallback
    if split == "test":
        return _find_metrics_file(model_dir, "val", dataset_idx)

    # Return the new format path (will raise FileNotFoundError later)
    return model_dir / f"metrics.{split}.{dataset_idx}.npz"


def _load_npz_metrics(metrics_path: Path) -> dict:
    """Load metrics from an npz file, supporting both old and new formats.

    New format: single "metrics" key containing a dict with all metrics.
    Old format: individual metric keys at top level (voc_metrics, mOKS, etc.).
    """
    with np.load(metrics_path, allow_pickle=True) as data:
        keys = list(data.keys())

        # New format: single "metrics" key containing dict
        if "metrics" in keys:
            return data["metrics"].item()

        # Old format: individual metric keys at top level
        expected_keys = {
            "voc_metrics",
            "mOKS",
            "distance_metrics",
            "pck_metrics",
            "visibility_metrics",
        }
        if expected_keys.issubset(set(keys)):
            return {
                k: data[k].item() if data[k].ndim == 0 else data[k]
                for k in expected_keys
            }

        # Unknown format - return all keys as dict
        return {k: data[k].item() if data[k].ndim == 0 else data[k] for k in keys}


def load_metrics(
    path: str,
    split: str = "test",
    dataset_idx: int = 0,
) -> dict:
    """Load metrics from a model folder or metrics file.

    This function supports both the new format (single "metrics" key) and the old
    format (individual metric keys at top level). It also handles both old and new
    file naming conventions in model folders.

    Args:
        path: Path to a model folder or metrics file (.npz).
        split: Name of the split to load. Must be "train", "val", or "test".
            Default: "test". If "test" is not found, falls back to "val".
            Ignored if path points directly to a .npz file.
        dataset_idx: Index of the dataset (for multi-dataset training).
            Default: 0. Ignored if path points directly to a .npz file.

    Returns:
        Dictionary containing metrics with keys: voc_metrics, mOKS,
        distance_metrics, pck_metrics, visibility_metrics.

    Raises:
        FileNotFoundError: If no metrics file is found.

    Examples:
        >>> # Load from model folder (tries test, falls back to val)
        >>> metrics = load_metrics("/path/to/model")
        >>> print(metrics["mOKS"]["mOKS"])

        >>> # Load specific split and dataset
        >>> metrics = load_metrics("/path/to/model", split="val", dataset_idx=1)

        >>> # Load directly from npz file
        >>> metrics = load_metrics("/path/to/metrics.val.0.npz")
    """
    path = Path(path)

    if path.suffix == ".npz":
        metrics_path = path
    else:
        metrics_path = _find_metrics_file(path, split, dataset_idx)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    return _load_npz_metrics(metrics_path)


def _resolve_anchor_ind(
    skeleton: "sio.Skeleton", anchor_part: Optional[str]
) -> Optional[int]:
    """Resolve an anchor node name to its index in the GT skeleton.

    Mirrors the anchor resolution in
    ``sleap_nn.inference.predictor`` (#582): returns the node index when
    ``anchor_part`` is present in the skeleton node names, else ``None`` (which
    drives the mean-of-visible-nodes fallback in :func:`compute_gt_centroids`).
    """
    if anchor_part is None or skeleton is None:
        return None
    node_names = list(getattr(skeleton, "node_names", []) or [])
    if anchor_part in node_names:
        return node_names.index(anchor_part)
    logger.warning(
        f"Anchor part {anchor_part!r} not found in GT skeleton node_names: "
        f"{node_names}. Falling back to mean-of-visible-nodes centroid."
    )
    return None


def _is_single_node_skeleton(skeleton: "sio.Skeleton") -> bool:
    """Return True if the skeleton is a single-node (centroid-like) skeleton.

    Detects ``sio.get_centroid_skeleton()`` (node_names == ['centroid']) as well
    as any other single-node skeleton.
    """
    if skeleton is None:
        return False
    node_names = list(getattr(skeleton, "node_names", []) or [])
    return len(node_names) == 1


def _metrics_to_json_safe(obj: Any) -> Any:
    """Recursively convert a metrics object into a JSON-serializable form.

    Used to write the ``.json`` sibling of the pickled ``.npz`` metrics file so
    non-Python consumers (e.g. the sleap-app metrics UI) can read the metrics
    without unpickling a numpy object array. Conversions:

    - numpy scalar (``np.generic``) -> python ``int`` / ``float`` / ``bool``
    - numpy ``ndarray`` -> nested python lists
    - non-finite float (``NaN`` / ``+-Inf``) -> ``None`` (JSON ``null``)
    - ``dict`` -> element-wise converted dict (keys coerced to ``str``)
    - ``list`` / ``tuple`` -> element-wise converted list
    - ``str`` and native JSON scalars pass through unchanged

    Emitting ``null`` (not the string ``"NaN"``) for non-finite values keeps the
    output valid JSON and lets the app treat missing-node distances as gaps.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        # ``.tolist()`` yields nested python lists with python floats/ints;
        # recurse so NaN/Inf inside the array become ``None``.
        return _metrics_to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        # numpy scalar -> python scalar, then fall through to the checks below.
        obj = obj.item()
    if isinstance(obj, bool):  # must precede int (bool is a subclass of int)
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _metrics_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_metrics_to_json_safe(v) for v in obj]
    return obj


# Large per-pair arrays that are useful only for offline analysis, not for the
# sleap-app metrics UI, and are dropped from the JSON sibling to keep it lean
# (they remain in the pickled ``.npz``). ``pck_metrics.pcks`` is an
# ``n_pairs x n_nodes x n_thresholds`` boolean array that otherwise dominates the
# JSON file size.
_JSON_PRUNE_KEYS: dict = {"pck_metrics": ("pcks",)}


def _prune_json_bloat(json_safe: Any) -> None:
    """Drop large, UI-unused arrays from a JSON-safe metrics dict, in place.

    Args:
        json_safe: A JSON-safe metrics dict (from :func:`_metrics_to_json_safe`).
    """
    if not isinstance(json_safe, dict):
        return
    for section, keys in _JSON_PRUNE_KEYS.items():
        sub = json_safe.get(section)
        if isinstance(sub, dict):
            for key in keys:
                sub.pop(key, None)


def _write_metrics(save_path: Path, metrics: dict) -> None:
    """Write ``metrics`` to ``save_path`` (``.npz``) plus a ``.json`` sibling.

    The ``.npz`` is the SLEAP 1.4 format (a single pickled 0-d ``metrics``
    object array, read back by :func:`load_metrics`) and is kept for
    back-compat. The ``.json`` sibling shares the same stem
    (``metrics.{split}.{idx}.json``) and holds the same metrics dict serialized
    JSON-safely via :func:`_metrics_to_json_safe` (minus a few large, UI-unused
    arrays, see :func:`_prune_json_bloat`) so it can be read directly by
    JavaScript (the sleap-app metrics UI).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **{"metrics": metrics})
    json_path = save_path.with_suffix(".json")
    json_safe = _metrics_to_json_safe(metrics)
    _prune_json_bloat(json_safe)
    with open(json_path, "w") as f:
        json.dump(json_safe, f)


def run_evaluation(
    ground_truth_path: str,
    predicted_path: str,
    oks_stddev: float = 0.025,
    oks_scale: Optional[float] = None,
    match_threshold: float = 0,
    user_labels_only: bool = True,
    save_metrics: Optional[str] = None,
    match_method: str = "oks",
    anchor_part: Optional[str] = None,
    centroid_method: Optional[str] = None,
    centroid_fallback: Optional[str] = None,
):
    """Evaluate SLEAP-NN model predictions against ground truth labels.

    Args:
        ground_truth_path: Path to the ground-truth ``.slp`` file.
        predicted_path: Path to the predicted ``.slp`` file.
        oks_stddev: OKS standard deviation (OKS mode only).
        oks_scale: OKS scale override (OKS mode only).
        match_threshold: Matching threshold. OKS threshold for OKS mode; PIXEL
            distance for centroid mode. In centroid mode, if the caller leaves
            the OKS default of ``0.0`` it is bumped to ``50.0`` px.
        user_labels_only: If False, predicted instances in the GT frame may be
            matched. For ``match_method="mask"`` (default True), this additionally
            drops masks linked to a ``PredictedInstance`` from the ground-truth
            labels, so stray predicted instances (each of which gets a mask when
            masks are built per-instance from poses) are not treated as ground
            truth. Pass ``False`` when the GT is intentionally built from predicted
            poses (pseudo-mask GT). The whole-frame ``match_method="semantic"`` union is
            unaffected either way.
        save_metrics: Optional ``.npz`` path to save metrics to.
        match_method: ``"oks"``, ``"centroid"``, ``"mask"``, ``"semantic"``, or
            ``"auto"``. ``"mask"`` matches predicted vs GT segmentation masks by
            IoU (for ``bottomup_segmentation`` models). ``"semantic"`` unions each
            frame's masks into one foreground and scores IoU/clDice/boundary-IoU
            with NO matching (for whole-frame ``semantic_segmentation`` models).
            ``"auto"`` switches to centroid mode when the PREDICTION skeleton is a
            single-node skeleton (e.g. ``sio.get_centroid_skeleton()``); it never
            auto-selects ``"mask"`` or ``"semantic"`` (pass those explicitly).
        anchor_part: Name of the GT skeleton node used to compute GT centroids
            (centroid mode). Resolved against the GT skeleton; ``None`` (or an
            absent name) falls back to the mean of visible nodes (#586).
        centroid_method: How GT centroids are derived (centroid mode) --
            ``"center_of_mass"``, ``"bbox_center"``, ``"geometric_median"`` or
            ``"anchor"``. ``None`` (default) infers it from ``anchor_part``. Pass
            the value the model was TRAINED with (its
            ``head_configs.centroid.confmaps.centroid_method``), or the distance
            metric scores predictions against a different centroid than the one
            they were trained to predict.
        centroid_fallback: Reduce method used when the anchor node is not visible.

    Returns:
        The metrics dict, or ``None`` if the predicted labels have zero
        frames or contain nothing usable (no instances for ``"oks"``/
        ``"centroid"``/``"auto"``, no masks for ``"mask"``/``"semantic"``) --
        metric computation is skipped entirely in that case, and no
        ``save_metrics`` file is written.
    """
    logger.info("Loading ground truth labels...")
    ground_truth_instances = sio.load_slp(ground_truth_path)
    logger.info(
        f"  Ground truth: {len(ground_truth_instances.videos)} videos, "
        f"{len(ground_truth_instances.labeled_frames)} frames"
    )

    logger.info("Loading predicted labels...")
    predicted_instances = sio.load_slp(predicted_path)
    logger.info(
        f"  Predictions: {len(predicted_instances.videos)} videos, "
        f"{len(predicted_instances.labeled_frames)} frames"
    )

    # Detect a fully collapsed prediction set up front and skip the metric
    # math entirely (#719) -- frames may still be present (both predictor
    # pipelines retain empty-detection frames by default), but nothing usable
    # was predicted in any of them, so matching would only produce an
    # all-NaN/all-zero result. ``mask``/``semantic`` predictions live on
    # ``LabeledFrame.masks``, not ``.instances``.
    if match_method in ("mask", "semantic"):
        has_predictions = any(len(lf.masks) for lf in predicted_instances)
    else:
        has_predictions = any(len(lf.instances) for lf in predicted_instances)
    if not len(predicted_instances) or not has_predictions:
        logger.info(
            "0 predicted instances: skipping metric computation (model "
            "likely predicted nothing usable, or training collapsed)."
        )
        return None

    # Auto-detect centroid mode from the PREDICTION skeleton.
    pred_skeleton = (
        predicted_instances.skeletons[0] if predicted_instances.skeletons else None
    )
    if match_method == "auto":
        if _is_single_node_skeleton(pred_skeleton):
            match_method = "centroid"
            logger.info(
                "Auto-detected centroid mode (single-node prediction skeleton)."
            )
        else:
            match_method = "oks"

    # Resolve the anchor node against the GT skeleton (mirror predictor.py).
    gt_skeleton = (
        ground_truth_instances.skeletons[0]
        if ground_truth_instances.skeletons
        else None
    )
    anchor_ind = _resolve_anchor_ind(gt_skeleton, anchor_part)

    # In centroid mode, default the (pixel) match threshold to 50.0 if the
    # caller left the OKS default of 0.0.
    if match_method == "centroid" and match_threshold == 0:
        match_threshold = 50.0

    # In mask mode, default the IoU match threshold to 0.5 if the caller left
    # the OKS default of 0.0.
    if match_method == "mask" and match_threshold == 0:
        match_threshold = 0.5

    # Mask eval matches GT vs predicted MASKS (on ``frame.masks``), independent of
    # whether the frame's keypoint instances are user- or predicted-labeled. The
    # ``user_labels_only`` frame filter (find_frame_pairs) keeps only frames with
    # USER keypoint instances, which silently drops EVERY frame when the GT was
    # built from predicted poses (e.g. pseudo-mask GT from predicted skeletons),
    # raising "Empty Frame Pairs". Mask mode therefore never applies that FRAME
    # filter. The caller's ``user_labels_only`` intent is preserved separately to
    # govern the ground-truth MASK filter: a labels file with stray
    # PredictedInstances gives each a mask (masks are built per-instance), and under
    # user-only labels those must not be treated as ground truth (they would be
    # spurious false negatives that cap recall). Callers evaluating pseudo-mask GT
    # pass ``user_labels_only=False``.
    exclude_predicted_instance_masks = user_labels_only
    if match_method in ("mask", "semantic"):
        user_labels_only = False

    logger.info("Matching videos and frames...")
    # Get match stats before creating evaluator
    match_result = ground_truth_instances.match(predicted_instances)
    logger.info(
        f"  Videos matched: {match_result.n_videos_matched}/{len(match_result.video_map)}"
    )

    logger.info("Matching instances...")
    evaluator = Evaluator(
        ground_truth_instances=ground_truth_instances,
        predicted_instances=predicted_instances,
        oks_stddev=oks_stddev,
        oks_scale=oks_scale,
        match_threshold=match_threshold,
        user_labels_only=user_labels_only,
        match_method=match_method,
        anchor_ind=anchor_ind,
        centroid_method=centroid_method,
        centroid_fallback=centroid_fallback,
        exclude_predicted_instance_masks=exclude_predicted_instance_masks,
    )
    logger.info(
        f"  Frame pairs: {len(evaluator.frame_pairs)}, "
        f"Matched instances: {len(evaluator.positive_pairs)}, "
        f"Unmatched GT: {len(evaluator.false_negatives)}"
    )

    logger.info("Computing evaluation metrics...")
    metrics = evaluator.evaluate()

    if match_method == "centroid":
        # Centroid mode: report detection + distance metrics only (no
        # oks_voc.*/mOKS/PCK/visibility keys exist).
        det = metrics["detection_metrics"]
        dist = metrics["distance_metrics"]
        logger.info("Evaluation Results (centroid mode):")
        logger.info(f"  Precision: {det['precision']:.4f}")
        logger.info(f"  Recall: {det['recall']:.4f}")
        logger.info(f"  F1: {det['f1']:.4f}")
        logger.info(f"  Counts: TP={det['n_tp']}, FP={det['n_fp']}, FN={det['n_fn']}")
        logger.info(f"  Average Distance: {dist['avg']:.2f} px")
        logger.info(f"  dist.p50: {dist['p50']:.2f} px")
        logger.info(f"  dist.p90: {dist['p90']:.2f} px")
        logger.info(f"  dist.p95: {dist['p95']:.2f} px")
        logger.info(f"  dist.p99: {dist['p99']:.2f} px")

        if save_metrics:
            logger.info(f"Saving metrics to {save_metrics}...")
            save_path = Path(save_metrics)
            # Writes the pickled ``.npz`` (back-compat) plus a JSON sibling
            # with the same stem so the app can read metrics without unpickling.
            _write_metrics(save_path, metrics)
            logger.info(f"Metrics saved successfully to {save_path}")

        return metrics

    if match_method == "mask":
        # Mask mode: report detection (IoU-matched) + mask-IoU quality + COCO
        # mask AP/AR (no oks_voc.*/mOKS/PCK/visibility keys exist).
        det = metrics["detection_metrics"]
        mm = metrics["mask_metrics"]
        mvoc = metrics["mask_voc_metrics"]
        logger.info("Evaluation Results (mask mode):")
        logger.info(f"  Precision: {det['precision']:.4f}")
        logger.info(f"  Recall: {det['recall']:.4f}")
        logger.info(f"  F1: {det['f1']:.4f}")
        logger.info(f"  Counts: TP={det['n_tp']}, FP={det['n_fp']}, FN={det['n_fn']}")
        logger.info(f"  Mean mask IoU: {mm['mean_iou']:.4f}")
        logger.info(f"  mask IoU p50: {mm['p50']:.4f}")
        logger.info(f"  mask IoU p25: {mm['p25']:.4f}")
        logger.info(f"  Mean boundary IoU: {mm['mean_boundary_iou']:.4f}")
        logger.info(f"  Mean clDice (centerline): {mm['mean_cldice']:.4f}")
        logger.info(f"  mAP @[.5:.95]: {mvoc['mask_voc.mAP']:.4f}")
        logger.info(
            f"  AP50: {mvoc['mask_voc.AP50']:.4f}  AP75: {mvoc['mask_voc.AP75']:.4f}"
        )
        logger.info(f"  AR @[.5:.95]: {mvoc['mask_voc.AR']:.4f}")
        e0, e1 = mvoc["mask_voc.size_edges"]
        logger.info(
            f"  AP by size [percentile, edges={e0:.0f}/{e1:.0f} px^2]: "
            f"S={mvoc['mask_voc.AP_small']:.4f} "
            f"M={mvoc['mask_voc.AP_medium']:.4f} L={mvoc['mask_voc.AP_large']:.4f} "
            f"(GT S/M/L={mvoc['mask_voc.n_gt_small']}/"
            f"{mvoc['mask_voc.n_gt_medium']}/{mvoc['mask_voc.n_gt_large']})"
        )
        logger.info(
            f"  AP by size [COCO 1024/9216 px^2]: "
            f"S={mvoc['mask_voc.coco.AP_small']:.4f} "
            f"M={mvoc['mask_voc.coco.AP_medium']:.4f} "
            f"L={mvoc['mask_voc.coco.AP_large']:.4f} "
            f"(GT S/M/L={mvoc['mask_voc.coco.n_gt_small']}/"
            f"{mvoc['mask_voc.coco.n_gt_medium']}/{mvoc['mask_voc.coco.n_gt_large']})"
        )
        logger.info(
            f"  Fragmentation: oversegmentation={mm['oversegmentation']}, "
            f"undersegmentation={mm['undersegmentation']}"
        )

        if save_metrics:
            logger.info(f"Saving metrics to {save_metrics}...")
            save_path = Path(save_metrics)
            # Writes the pickled ``.npz`` (back-compat) plus a JSON sibling
            # with the same stem so the app can read metrics without unpickling.
            _write_metrics(save_path, metrics)
            logger.info(f"Metrics saved successfully to {save_path}")

        return metrics

    if match_method == "semantic":
        # Semantic (whole-frame foreground) mode: matching-free IoU / clDice /
        # boundary-IoU only (no detection / mask-AP keys exist).
        sm = metrics["semantic_metrics"]
        logger.info("Evaluation Results (semantic / whole-frame foreground mode):")
        logger.info(f"  Frames scored (non-empty GT fg): {sm['n_frames']}")
        logger.info(f"  Mean foreground IoU: {sm['mean_iou']:.4f}")
        logger.info(f"  Mean clDice (centerline): {sm['mean_cldice']:.4f}")
        logger.info(f"  Mean boundary IoU: {sm['mean_boundary_iou']:.4f}")

        if save_metrics:
            logger.info(f"Saving metrics to {save_metrics}...")
            save_path = Path(save_metrics)
            # Writes the pickled ``.npz`` (back-compat) plus a JSON sibling
            # with the same stem so the app can read metrics without unpickling.
            _write_metrics(save_path, metrics)
            logger.info(f"Metrics saved successfully to {save_path}")

        return metrics

    # Compute PCK at specific thresholds (5 and 10 pixels)
    dists = metrics["distance_metrics"]["dists"]
    dists_clean = np.copy(dists)
    dists_clean[np.isnan(dists_clean)] = np.inf
    # Guard the empty-match case (0 matched instances for the whole split) so
    # this doesn't hit "Mean of empty slice" on top of the evaluate()-level
    # log line already emitted for it.
    pck_5 = float((dists_clean < 5).mean()) if dists_clean.size else np.nan
    pck_10 = float((dists_clean < 10).mean()) if dists_clean.size else np.nan

    # Print key metrics
    logger.info("Evaluation Results:")
    logger.info(f"  mOKS: {metrics['mOKS']['mOKS']:.4f}")
    logger.info(f"  mAP (OKS VOC): {metrics['voc_metrics']['oks_voc.mAP']:.4f}")
    logger.info(f"  mAR (OKS VOC): {metrics['voc_metrics']['oks_voc.mAR']:.4f}")
    logger.info(f"  Average Distance: {metrics['distance_metrics']['avg']:.2f} px")
    logger.info(f"  dist.p50: {metrics['distance_metrics']['p50']:.2f} px")
    logger.info(f"  dist.p95: {metrics['distance_metrics']['p95']:.2f} px")
    logger.info(f"  dist.p99: {metrics['distance_metrics']['p99']:.2f} px")
    logger.info(f"  mPCK: {metrics['pck_metrics']['mPCK']:.4f}")
    logger.info(f"  PCK@5px: {pck_5:.4f}")
    logger.info(f"  PCK@10px: {pck_10:.4f}")
    logger.info(
        f"  Visibility Precision: {metrics['visibility_metrics']['precision']:.4f}"
    )
    logger.info(f"  Visibility Recall: {metrics['visibility_metrics']['recall']:.4f}")

    # Save metrics if path provided
    if save_metrics:
        logger.info(f"Saving metrics to {save_metrics}...")
        save_path = Path(save_metrics)

        # Save metrics in SLEAP 1.4 format (single "metrics" key) plus a JSON
        # sibling (same stem) that the app metrics UI can read without
        # unpickling the numpy object array.
        _write_metrics(save_path, metrics)
        logger.info(f"Metrics saved successfully to {save_path}")

    return metrics


# ---------------------------------------------------------------------------
# Identity-persistence metrics (MOT-style)
# ---------------------------------------------------------------------------
# These score a TRACKED prediction against TRACKED ground truth: not "was the
# animal found" (the detection metrics above) but "did it keep the same identity
# across frames". Any tracker can use them -- fixed-window/local-queues, optical
# flow, Kalman, mask-IoU -- since they read only `track` off the detections.
#
# Deliberately NOT MOTA: MOTA folds detection FP/FN into the identity score, so
# when two trackers are compared over a FROZEN detection stage (the usual A/B) a
# MOTA delta mostly reports detector noise. Detection counts are reported
# separately in `IdentityMetrics` and never folded into IDF1 or the switch count.
#
# Definitions follow the standard multi-object-tracking literature:
#   - ID switches (IDSW): CLEAR-MOT. Per ground-truth trajectory, a switch each
#     time the predicted track matched to it differs from the last one it was
#     matched to. Gaps do not count; a change *across* a gap does.
#   - IDF1 / IDP / IDR: Ristani et al. Global max-weight assignment between
#     ground-truth and predicted identities over co-matched detection counts,
#     then F1 over IDTP / IDFP / IDFN.
#   - MT/PT/ML + fragmentation: coverage of each ground-truth trajectory, so a
#     tracker cannot win on switches by emitting fewer, shorter tracks.
#   - Purity: per predicted track, the share of its matched detections belonging
#     to its dominant ground-truth identity (length-weighted mean).

IDENTITY_CARRIERS = ("pose", "mask")


def _validate_carrier(carrier: str) -> str:
    """Normalize and validate an identity-metric carrier.

    Args:
        carrier: ``"pose"`` (instances, matched by OKS) or ``"mask"``
            (segmentation masks, matched by mask IoU).

    Returns:
        The validated carrier string.

    Raises:
        ValueError: If ``carrier`` is not one of ``IDENTITY_CARRIERS``.
    """
    if carrier not in IDENTITY_CARRIERS:
        raise ValueError(
            f"carrier must be one of {IDENTITY_CARRIERS}, got {carrier!r}. "
            "Use 'pose' for instances (OKS) or 'mask' for segmentation masks (IoU)."
        )
    return carrier


def _identity_frame_key(frame: sio.LabeledFrame) -> Tuple[str, int]:
    """Key a frame by ``(video path, frame index)``.

    Keyed on the video's path rather than its position in ``labels.videos`` so a
    single-video prediction aligns with the right video of a multi-video
    ground-truth project -- position would pair it with whatever happens to be
    first. Matches the video identity `find_frame_pairs` uses for the detection
    metrics.
    """
    return (_video_key(frame.video), frame.frame_idx)


def _identity_dets(frame: sio.LabeledFrame, carrier: str) -> List[Any]:
    """Return a frame's detections for the given carrier, in file order."""
    if carrier == "mask":
        return list(getattr(frame, "masks", None) or [])
    return list(frame.instances)


def _is_predicted_detection(det: Any) -> bool:
    """Return True for model output, on either carrier."""
    return isinstance(det, (sio.PredictedInstance, sio.PredictedSegmentationMask))


def _keeps_identity(det: Any, drop_predicted: bool) -> bool:
    """Return True if a detection should take part in identity scoring.

    Args:
        det: An instance or segmentation mask.
        drop_predicted: Drop model output (the ground-truth side under
            ``user_labels_only``).

    Returns:
        True when the detection carries a track and survives the predicted
        filter. Untracked detections are counted in the totals but never
        matched -- identity is what is being scored, so a detection without one
        has no identity to get right.
    """
    if getattr(det, "track", None) is None:
        return False
    return not (drop_predicted and _is_predicted_detection(det))


def _pose_similarity(gt: List[Any], pr: List[Any]) -> np.ndarray:
    """Compute the ``(n_gt, n_pr)`` OKS matrix between two instance lists.

    Built one predicted instance at a time. OKS is defined per ``(gt, pr)`` pair
    -- each column depends only on its own prediction and the ground-truth
    scales -- so the loop is exactly equivalent to the matrix form while also
    working on repo versions whose ``compute_oks`` only accepted a single
    prediction (see #739).
    """
    if not gt or not pr:
        return np.zeros((len(gt), len(pr)))
    pts_gt = np.stack([np.asarray(inst.numpy(), dtype=float)[:, :2] for inst in gt])
    sim = np.zeros((len(gt), len(pr)))
    for j, inst in enumerate(pr):
        pts_pr = np.asarray(inst.numpy(), dtype=float)[None, :, :2]
        sim[:, j] = np.asarray(compute_oks(pts_gt, pts_pr), dtype=float).reshape(
            len(gt)
        )
    return sim


def _mask_similarity(gt: List[np.ndarray], pr: List[np.ndarray]) -> np.ndarray:
    """Compute the ``(n_gt, n_pr)`` mask-IoU matrix from decoded boolean arrays."""
    if not len(gt) or not len(pr):
        return np.zeros((len(gt), len(pr)))
    # _mask_iou_matrix is (n_pred, n_gt); transpose to (n_gt, n_pr).
    return np.asarray(_mask_iou_matrix(pr, gt), dtype=float).T


def _match_identity_frame(
    gt: List[Any], pr: List[Any], carrier: str, threshold: float
) -> List[Tuple[int, int, float]]:
    """Hungarian-match ground-truth to predicted detections within one frame.

    Args:
        gt: Ground-truth detections -- instances for ``"pose"``, decoded boolean
            mask arrays for ``"mask"``.
        pr: Predicted detections, same convention.
        carrier: ``"pose"`` or ``"mask"``.
        threshold: Minimum similarity (OKS or IoU) for a pair to count as matched.

    Returns:
        List of ``(gt_index, pred_index, similarity)`` for pairs at or above
        ``threshold``.
    """
    from scipy.optimize import linear_sum_assignment

    sim = _pose_similarity(gt, pr) if carrier == "pose" else _mask_similarity(gt, pr)
    if sim.size == 0:
        return []
    rows, cols = linear_sum_assignment(-sim)
    return [
        (int(r), int(c), float(sim[r, c]))
        for r, c in zip(rows, cols)
        if sim[r, c] >= threshold
    ]


def motion_diagnostic(labels: sio.Labels, carrier: str = "pose") -> Dict[str, Any]:
    """Judge whether a labels file is continuous video or temporally sparse samples.

    **Identity metrics are meaningless on a sparse set, and nothing in the file
    says so.** Embedded ``.pkg.slp`` training splits renumber their frames
    ``0..N-1`` and record ``frame_numbers`` as contiguous, so every index-based
    contiguity check passes -- while the animal has actually moved across the
    arena between two "consecutive" frames. Run this before quoting a tracking
    number on an unfamiliar file.

    The decisive quantity is how far the same animal moves between consecutive
    frames relative to its own size. Measured on real files, ``step_over_size``
    lands near ``0.01-0.06`` for genuine video and ``3-9`` for sparse training
    splits -- and at the high end same-animal consecutive mask IoU is ``0.000``
    for most pairs, so geometric association has no signal to work with and any
    IoU tracker must fail.

    Args:
        labels: Tracked labels to inspect.
        carrier: ``"pose"`` (instance keypoints) or ``"mask"`` (segmentation
            masks); decides how a detection's center and size are measured.

    Returns:
        Dict with ``median_step_px``, ``median_size_px``, ``step_over_size`` and
        ``is_continuous`` (``step_over_size < 0.5``). When too few tracked
        detections are present to judge, ``step_over_size`` is NaN,
        ``is_continuous`` is False and a ``note`` explains why.
    """
    carrier = _validate_carrier(carrier)
    prev: Dict[str, np.ndarray] = {}
    steps: List[float] = []
    sizes: List[float] = []

    for frame in sorted(labels, key=lambda lf: lf.frame_idx):
        items: List[Tuple[str, np.ndarray, float]] = []
        if carrier == "mask":
            for arr, det in zip(_frame_masks(frame), _identity_dets(frame, carrier)):
                if getattr(det, "track", None) is None:
                    continue
                ys, xs = np.nonzero(arr)
                if not len(xs):
                    continue
                items.append(
                    (
                        det.track.name,
                        np.array([xs.mean(), ys.mean()]),
                        # Equivalent-circle diameter of the mask.
                        2.0 * float(np.sqrt(arr.sum() / np.pi)),
                    )
                )
        else:
            for det in _identity_dets(frame, carrier):
                if getattr(det, "track", None) is None:
                    continue
                pts = np.asarray(det.numpy(), dtype=float)[:, :2]
                pts = pts[~np.isnan(pts).any(axis=1)]
                if not len(pts):
                    continue
                items.append(
                    (det.track.name, pts.mean(axis=0), float(np.ptp(pts, axis=0).max()))
                )

        for name, center, size in items:
            sizes.append(size)
            if name in prev:
                steps.append(float(np.linalg.norm(center - prev[name])))
            prev[name] = center

    if not steps or not sizes:
        return {
            "median_step_px": float("nan"),
            "median_size_px": float("nan"),
            "step_over_size": float("nan"),
            "is_continuous": False,
            "note": "not enough tracked detections to judge",
        }

    median_step = float(np.median(steps))
    median_size = float(np.median(sizes))
    if median_size <= 0.0:
        # A single-node skeleton (a centroid model) or coincident nodes have no
        # measurable extent, so there is nothing to normalize the step against.
        # Report "cannot judge" rather than dividing by ~0 and calling every
        # centroid prediction sparse.
        return {
            "median_step_px": round(median_step, 2),
            "median_size_px": median_size,
            "step_over_size": float("nan"),
            "is_continuous": False,
            "note": (
                "detections have no measurable extent (single-node skeleton?) -- "
                "cannot judge continuity"
            ),
        }
    ratio = median_step / median_size
    return {
        "median_step_px": round(median_step, 2),
        "median_size_px": round(median_size, 2),
        "step_over_size": round(ratio, 3),
        "is_continuous": bool(ratio < 0.5),
    }


@attrs.define(auto_attribs=True, slots=True)
class IdentityMetrics:
    """Identity-persistence metrics for one tracked prediction.

    Attributes:
        id_switches: CLEAR-MOT ID switches, summed over ground-truth trajectories.
        idf1: Identity F1 (Ristani et al.).
        idp: Identity precision.
        idr: Identity recall.
        mostly_tracked: Ground-truth trajectories covered at or above
            ``mt_threshold``.
        partly_tracked: Ground-truth trajectories between the two coverage cuts.
        mostly_lost: Ground-truth trajectories covered below ``ml_threshold``.
        fragmentations: Matched -> unmatched -> matched interruptions of a
            ground-truth trajectory.
        mean_gt_coverage: Mean share of each trajectory's frames that matched.
        mean_track_purity: Length-weighted mean dominant-identity share per
            predicted track.
        n_gt_dets: Tracked ground-truth detections compared.
        n_pred_dets: Predicted detections in the compared frames.
        n_matched: Ground-truth/predicted pairs matched above threshold.
        n_frames_compared: Frames present on both sides.
        n_gt_tracks: Distinct ground-truth track names seen.
        n_pred_tracks: Distinct predicted track names seen.
        n_pred_untracked: Predicted detections with no ``track`` set.
        notes: Human-readable caveats raised while comparing.
    """

    # Headline.
    id_switches: int = 0
    idf1: float = float("nan")
    idp: float = float("nan")
    idr: float = float("nan")
    # Coverage -- guards against winning on switches by tracking less.
    mostly_tracked: int = 0
    partly_tracked: int = 0
    mostly_lost: int = 0
    fragmentations: int = 0
    mean_gt_coverage: float = float("nan")
    # Purity.
    mean_track_purity: float = float("nan")
    # Detection accounting, reported separately and never folded into the above.
    n_gt_dets: int = 0
    n_pred_dets: int = 0
    n_matched: int = 0
    n_frames_compared: int = 0
    n_gt_tracks: int = 0
    n_pred_tracks: int = 0
    n_pred_untracked: int = 0
    notes: List[str] = attrs.field(factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return the metrics as a plain, JSON-serializable dict."""
        return attrs.asdict(self)

    def summary(self) -> str:
        """Return a one-line summary of the headline metrics."""
        return (
            f"IDSW={self.id_switches}  IDF1={self.idf1:.4f} "
            f"(P={self.idp:.4f} R={self.idr:.4f})  "
            f"MT/PT/ML={self.mostly_tracked}/{self.partly_tracked}/{self.mostly_lost}  "
            f"Frag={self.fragmentations}  purity={self.mean_track_purity:.4f}  "
            f"cov={self.mean_gt_coverage:.4f}  "
            f"[{self.n_matched}/{self.n_gt_dets} GT dets matched, "
            f"{self.n_pred_dets} pred, {self.n_frames_compared} frames]"
        )


def identity_metrics(
    gt_labels: sio.Labels,
    pred_labels: sio.Labels,
    carrier: str = "pose",
    *,
    match_threshold: float = 0.5,
    mt_threshold: float = 0.8,
    ml_threshold: float = 0.2,
    user_labels_only: bool = False,
) -> IdentityMetrics:
    """Score a tracked prediction against tracked ground truth.

    Detections are Hungarian-matched to ground truth within each frame (OKS for
    ``"pose"``, mask IoU for ``"mask"``), then identity is scored over those
    matches. Detections with no ``track`` set are counted but never matched, on
    either side.

    Args:
        gt_labels: Ground truth with ``track`` set on the detections to score.
        pred_labels: Prediction with ``track`` set by the tracker under test.
        carrier: ``"pose"`` (instances, OKS) or ``"mask"`` (segmentation masks,
            IoU) -- which similarity matches detections.
        match_threshold: Minimum similarity for a ground-truth/predicted pair to
            count as matched (OKS or IoU, per carrier).
        mt_threshold: Coverage at or above which a ground-truth trajectory counts
            as mostly-tracked.
        ml_threshold: Coverage below which a ground-truth trajectory counts as
            mostly-lost.
        user_labels_only: Drop model output (``PredictedInstance`` /
            ``PredictedSegmentationMask``) from the GROUND-TRUTH side.
            **Defaults to False, unlike the detection metrics**, because tracked
            ground truth usually *is* predicted: the standard workflow predicts
            poses and then assigns or corrects tracks over them, so filtering by
            type would silently discard the whole ground truth (measured on the
            re-ID benchmark's GT sessions: 2465 detections to 0). Pass True when
            the ground truth is user-labeled and the file also carries stale
            predictions from an earlier run, which would otherwise be scored as
            extra trajectories. Either way the count is reported in ``notes``,
            and the prediction side is never filtered.

    Returns:
        An :class:`IdentityMetrics`. When the two files share no frames, the
        result is empty and ``notes`` says so rather than raising -- check
        ``n_frames_compared`` before quoting a number.
    """
    from collections import Counter, defaultdict
    from scipy.optimize import linear_sum_assignment

    carrier = _validate_carrier(carrier)
    metrics = IdentityMetrics()

    gt_by_key = {_identity_frame_key(lf): lf for lf in gt_labels}
    pred_by_key = {_identity_frame_key(lf): lf for lf in pred_labels}
    shared = sorted(set(gt_by_key) & set(pred_by_key))
    if not shared:
        # One video on each side but under different paths (a clip re-saved or
        # copied elsewhere, an embedded package vs its source) is common enough
        # to be worth rescuing: there is only one possible pairing, so fall back
        # to frame_idx alone rather than reporting nothing. With several videos
        # on either side the pairing is ambiguous, so it is left unaligned.
        if (
            len({k[0] for k in gt_by_key}) == 1
            and len({k[0] for k in pred_by_key}) == 1
        ):
            gt_by_key = {("", lf.frame_idx): lf for lf in gt_labels}
            pred_by_key = {("", lf.frame_idx): lf for lf in pred_labels}
            shared = sorted(set(gt_by_key) & set(pred_by_key))
            metrics.notes.append(
                "aligned on frame_idx only (single video on both sides)"
            )
        if not shared:
            metrics.notes.append("no frames in common -- nothing compared")
            return metrics
    if len(shared) < len(gt_by_key):
        metrics.notes.append(
            f"{len(gt_by_key) - len(shared)} GT frames had no predicted counterpart"
        )

    # gt track name -> ordered list of (frame position, matched pred name or None)
    timeline: Dict[str, List[Tuple[int, Optional[str]]]] = defaultdict(list)
    co_matched: "Counter[Tuple[str, str]]" = Counter()
    pred_track_dets: "Counter[str]" = Counter()
    gt_tracks: set = set()
    pred_tracks: set = set()
    n_gt_predicted = 0

    for position, key in enumerate(shared):
        gt_frame, pred_frame = gt_by_key[key], pred_by_key[key]
        gt_all = _identity_dets(gt_frame, carrier)
        gt_dets = [d for d in gt_all if _keeps_identity(d, user_labels_only)]
        pred_all = _identity_dets(pred_frame, carrier)
        pred_dets = [d for d in pred_all if _keeps_identity(d, False)]
        n_gt_predicted += sum(1 for d in gt_all if _is_predicted_detection(d))

        metrics.n_gt_dets += len(gt_dets)
        metrics.n_pred_dets += len(pred_all)
        metrics.n_pred_untracked += len(pred_all) - len(pred_dets)
        gt_tracks.update(d.track.name for d in gt_dets)
        pred_tracks.update(d.track.name for d in pred_dets)

        # For the mask carrier, match on decoded image-grid arrays rather than on
        # the mask objects: `_frame_masks` is scale-aware, so a stride-res
        # prediction and a full-res ground-truth mask are compared on a common
        # pixel grid (the class of bug #693/#694 fixed). It decodes `lf.masks` in
        # order, so filtering the decoded list by the same tracked-ness
        # predicate keeps it index-aligned with `gt_dets` / `pred_dets`.
        if carrier == "mask":
            match_gt = [
                arr
                for arr, det in zip(_frame_masks(gt_frame), gt_all)
                if _keeps_identity(det, user_labels_only)
            ]
            match_pred = [
                arr
                for arr, det in zip(_frame_masks(pred_frame), pred_all)
                if _keeps_identity(det, False)
            ]
        else:
            match_gt, match_pred = gt_dets, pred_dets

        matched_gt: Dict[int, str] = {}
        for gt_idx, pred_idx, _sim in _match_identity_frame(
            match_gt, match_pred, carrier, match_threshold
        ):
            gt_name = gt_dets[gt_idx].track.name
            pred_name = pred_dets[pred_idx].track.name
            matched_gt[gt_idx] = pred_name
            co_matched[(gt_name, pred_name)] += 1
            pred_track_dets[pred_name] += 1
            metrics.n_matched += 1

        for gt_idx, det in enumerate(gt_dets):
            timeline[det.track.name].append((position, matched_gt.get(gt_idx)))

    metrics.n_frames_compared = len(shared)
    metrics.n_gt_tracks = len(gt_tracks)
    metrics.n_pred_tracks = len(pred_tracks)
    if n_gt_predicted:
        if user_labels_only:
            metrics.notes.append(
                f"{n_gt_predicted} predicted detections were dropped from the "
                "ground truth (user_labels_only=True)"
            )
            if not metrics.n_gt_dets:
                metrics.notes.append(
                    "user_labels_only=True left NO ground truth: every tracked "
                    "detection is model output. Tracked GT is usually predicted "
                    "poses with tracks assigned afterwards -- pass "
                    "user_labels_only=False to score it."
                )
        else:
            metrics.notes.append(
                f"{n_gt_predicted} of the ground-truth detections are model "
                "output (tracks assigned over predictions); scored as ground "
                "truth. Pass user_labels_only=True to exclude them."
            )

    # --- ID switches, coverage and fragmentation, per ground-truth trajectory ---
    coverages: List[float] = []
    for entries in timeline.values():
        entries.sort(key=lambda entry: entry[0])
        matched = [name for _position, name in entries if name is not None]
        coverage = len(matched) / len(entries) if entries else 0.0
        coverages.append(coverage)

        last: Optional[str] = None
        for _position, name in entries:
            if name is None:
                continue
            if last is not None and name != last:
                metrics.id_switches += 1
            last = name

        # Fragmentation: matched -> unmatched -> matched interruptions. Only gaps
        # that are re-matched later count, so a trajectory that simply ends is
        # not a fragmentation.
        matched_seq = [name is not None for _position, name in entries]
        started = False
        for i, is_matched in enumerate(matched_seq):
            if is_matched:
                started = True
            elif started and i > 0 and matched_seq[i - 1] and any(matched_seq[i + 1 :]):
                metrics.fragmentations += 1

        if coverage >= mt_threshold:
            metrics.mostly_tracked += 1
        elif coverage < ml_threshold:
            metrics.mostly_lost += 1
        else:
            metrics.partly_tracked += 1

    metrics.mean_gt_coverage = float(np.mean(coverages)) if coverages else float("nan")

    # --- IDF1: global max-weight ground-truth <-> predicted identity assignment ---
    gt_names = sorted(gt_tracks)
    pred_names = sorted(pred_tracks)
    if gt_names and pred_names:
        weights = np.zeros((len(gt_names), len(pred_names)))
        gt_index = {name: i for i, name in enumerate(gt_names)}
        pred_index = {name: i for i, name in enumerate(pred_names)}
        for (gt_name, pred_name), count in co_matched.items():
            weights[gt_index[gt_name], pred_index[pred_name]] = count
        rows, cols = linear_sum_assignment(-weights)
        idtp = float(weights[rows, cols].sum())
        idfn = metrics.n_gt_dets - idtp
        idfp = (metrics.n_pred_dets - metrics.n_pred_untracked) - idtp
        metrics.idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else float("nan")
        metrics.idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else float("nan")
        denominator = 2 * idtp + idfp + idfn
        metrics.idf1 = 2 * idtp / denominator if denominator > 0 else float("nan")

    # --- Track purity, length-weighted over predicted tracks ---
    by_pred: Dict[str, "Counter[str]"] = defaultdict(Counter)
    for (gt_name, pred_name), count in co_matched.items():
        by_pred[pred_name][gt_name] += count
    if by_pred:
        total = sum(pred_track_dets[name] for name in by_pred)
        metrics.mean_track_purity = (
            float(sum(max(counts.values()) for counts in by_pred.values()) / total)
            if total
            else float("nan")
        )

    return metrics


def compare_identity_metrics(arms: Dict[str, IdentityMetrics]) -> str:
    """Render a Markdown comparison table across tracker arms.

    Args:
        arms: Mapping of arm name (e.g. ``"geometry"``, ``"fused"``) to its
            :class:`IdentityMetrics`.

    Returns:
        A Markdown table, one row per arm, with a footer naming the direction of
        improvement for each column.
    """
    columns = [
        ("IDSW", "id_switches", "{:d}"),
        ("IDF1", "idf1", "{:.4f}"),
        ("purity", "mean_track_purity", "{:.4f}"),
        ("cov", "mean_gt_coverage", "{:.4f}"),
        ("MT", "mostly_tracked", "{:d}"),
        ("ML", "mostly_lost", "{:d}"),
        ("Frag", "fragmentations", "{:d}"),
        ("matched", "n_matched", "{:d}"),
    ]
    lines = [
        "| arm | " + " | ".join(label for label, _attr, _fmt in columns) + " |",
        "|---|" + "|".join("---" for _ in columns) + "|",
    ]
    for name, metrics in arms.items():
        cells = []
        for _label, attr, fmt in columns:
            value = getattr(metrics, attr)
            cells.append(
                "n/a"
                if value is None or (isinstance(value, float) and np.isnan(value))
                else fmt.format(value)
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Lower is better: IDSW, ML, Frag. Higher is better: IDF1, purity, cov, MT."
    )
    return "\n".join(lines)


def run_identity_evaluation(
    ground_truth_path: str,
    predicted_path: str,
    carrier: str = "auto",
    match_threshold: float = 0.5,
    mt_threshold: float = 0.8,
    ml_threshold: float = 0.2,
    user_labels_only: bool = False,
    save_metrics: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate identity persistence of a tracked prediction against tracked GT.

    Args:
        ground_truth_path: Path to the ground-truth ``.slp`` file, tracked.
        predicted_path: Path to the predicted ``.slp`` file, tracked.
        carrier: ``"pose"``, ``"mask"``, or ``"auto"`` -- ``"auto"`` picks
            ``"mask"`` when the prediction carries segmentation masks but no
            instances, else ``"pose"``.
        match_threshold: Minimum OKS (pose) or IoU (mask) for a detection pair to
            count as matched.
        mt_threshold: Mostly-tracked coverage cut.
        ml_threshold: Mostly-lost coverage cut.
        user_labels_only: Drop model output from the ground-truth side; off by
            default (see :func:`identity_metrics` for why).
        save_metrics: Optional ``.json`` path to write the metrics to.

    Returns:
        Dict with the :class:`IdentityMetrics` fields plus ``carrier``,
        ``match_threshold`` and ``motion_diagnostic``, or ``None`` if the
        prediction carries no tracked detections at all (nothing to score).
    """
    logger.info("Loading ground truth labels...")
    gt_labels = sio.load_slp(ground_truth_path)
    logger.info(
        f"  Ground truth: {len(gt_labels.videos)} videos, "
        f"{len(gt_labels.labeled_frames)} frames"
    )

    logger.info("Loading predicted labels...")
    pred_labels = sio.load_slp(predicted_path)
    logger.info(
        f"  Predictions: {len(pred_labels.videos)} videos, "
        f"{len(pred_labels.labeled_frames)} frames"
    )

    if carrier == "auto":
        has_masks = any(len(getattr(lf, "masks", None) or []) for lf in pred_labels)
        has_instances = any(len(lf.instances) for lf in pred_labels)
        carrier = "mask" if (has_masks and not has_instances) else "pose"
        logger.info(f"Auto-detected carrier: {carrier}.")
    carrier = _validate_carrier(carrier)

    n_tracked = sum(
        1
        for lf in pred_labels
        for det in _identity_dets(lf, carrier)
        if getattr(det, "track", None) is not None
    )
    if not n_tracked:
        logger.info(
            "0 tracked predicted detections: skipping identity metrics. Run "
            "`sleap-nn track` (or predict with `-t`) first -- these metrics score "
            "identity, so an untracked prediction has nothing to score."
        )
        return None

    # The sparse-split trap: a `.pkg.slp` training split renumbers its frames
    # contiguously, so it *looks* like video while the animal teleports between
    # "consecutive" frames. Say so loudly rather than reporting a meaningless
    # switch count.
    motion = motion_diagnostic(gt_labels, carrier)
    if np.isnan(motion["step_over_size"]):
        logger.info(f"Continuity check inconclusive: {motion.get('note', '')}")
    elif not motion["is_continuous"]:
        logger.warning(
            "Ground truth does not look like continuous video "
            f"(step/size = {motion['step_over_size']}). Identity metrics on a "
            "temporally sparse set (e.g. a `.pkg.slp` training split) are not "
            "meaningful -- score a real video clip instead."
        )

    metrics = identity_metrics(
        gt_labels,
        pred_labels,
        carrier,
        match_threshold=match_threshold,
        mt_threshold=mt_threshold,
        ml_threshold=ml_threshold,
        user_labels_only=user_labels_only,
    )

    logger.info("Identity Evaluation Results:")
    logger.info(f"  {metrics.summary()}")
    for note in metrics.notes:
        logger.warning(f"  note: {note}")
    if not metrics.n_gt_dets:
        logger.warning(
            "No tracked ground-truth detections were scored -- every metric "
            "above is empty. Check that the ground truth carries tracks."
        )

    result = metrics.as_dict()
    result["carrier"] = carrier
    result["match_threshold"] = match_threshold
    result["motion_diagnostic"] = motion

    if save_metrics:
        save_path = Path(save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(_metrics_to_json_safe(result), f, indent=2)
        logger.info(f"Metrics saved successfully to {save_path}")

    return result
