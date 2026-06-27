"""This module is to compute evaluation metrics for trained models."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import attrs
import sleap_io as sio
from loguru import logger
import click
from pathlib import Path


def compute_gt_centroids(
    instance_gt_points: np.ndarray, anchor_ind: Optional[int]
) -> np.ndarray:
    """Compute ground-truth centroids mirroring ``generate_centroids`` in numpy.

    This is the numpy mirror of
    :func:`sleap_nn.data.instance_centroids.generate_centroids`. The centroid's
    MEANING is defined by that function (see also #586): when the configured
    anchor node is present (non-NaN) it is that node; otherwise the centroid
    falls back to the NaN-ignoring MEAN of visible nodes (NOT the bounding-box
    midpoint). The fallback is computed via ``np.nanmean`` over the node axis,
    and is NaN only when every node of an instance is NaN.

    Args:
        instance_gt_points: Ground-truth keypoints of shape ``(n_instances,
            n_nodes, 2)`` or ``(n_nodes, 2)``. Missing/occluded nodes are NaN.
        anchor_ind: Index of the node to use as the anchor. If ``None``, or if
            the anchor node is NaN for a given instance, the centroid falls back
            to the NaN-ignoring mean of visible nodes for that instance.

    Returns:
        Centroids of shape ``(n_instances, 2)`` (or ``(2,)`` for a single
        instance input), reducing the node axis.
    """
    points = np.asarray(instance_gt_points, dtype=np.float64)

    if anchor_ind is not None:
        centroids = points[..., anchor_ind, :].copy()
    else:
        centroids = np.full(points.shape[:-2] + (2,), np.nan, dtype=points.dtype)

    missing_anchors = np.isnan(centroids).any(axis=-1)
    if np.any(missing_anchors):
        # NaN-ignoring mean of visible nodes. np.nanmean over the node axis
        # yields NaN only when all nodes for that instance are NaN (matching
        # find_points_mean). Suppress the all-NaN-slice RuntimeWarning.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_fallback = np.nanmean(points, axis=-2)
        centroids[missing_anchors] = mean_fallback[missing_anchors]

    return centroids


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


def _frame_masks(frame: sio.LabeledFrame) -> List[np.ndarray]:
    """Decode a frame's segmentation masks into boolean arrays on the image grid.

    Scale-aware: masks encoded at output-stride (non-identity ``scale``, the
    default for predicted segmentation masks) are nearest-neighbor resampled up
    to their image extent, so a stride-res prediction and an original-res
    ground-truth mask are compared on a common image-pixel grid. Scale-1 masks
    (legacy full-res GT/preds) take a zero-copy fast path, so existing eval
    numbers are unchanged.
    """
    from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

    masks = getattr(frame, "masks", None) or []
    return [decode_mask_to_image_res(m) for m in masks]


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


def get_instances(labeled_frame: sio.LabeledFrame) -> List[MatchInstance]:
    """Get a list of instances of type MatchInstance from the Labeled Frame.

    Args:
        labeled_frame: Input Labeled frame of type sio.LabeledFrame.

    Returns:
        List of MatchInstance objects for the given labeled frame.
    """
    instance_list = []
    frame_idx = labeled_frame.frame_idx

    # Extract video path with fallbacks for embedded videos
    video = labeled_frame.video
    video_path = None
    if video is not None:
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
        video_path = f"video_{id(video)}" if video is not None else "unknown"

    for instance in labeled_frame.instances:
        match_instance = MatchInstance(
            instance=instance, frame_idx=frame_idx, video_path=video_path
        )
        instance_list.append(match_instance)
    return instance_list


def find_frame_pairs(
    labels_gt: sio.Labels, labels_pr: sio.Labels, user_labels_only: bool = True
) -> List[Tuple[sio.LabeledFrame, sio.LabeledFrame]]:
    """Find corresponding frames across two sets of labels.

    This function uses sleap-io's robust video matching API to handle various
    scenarios including embedded videos, cross-platform paths, and videos with
    different metadata.

    Args:
        labels_gt: A `sio.Labels` instance with ground truth instances.
        labels_pr: A `sio.Labels` instance with predicted instances.
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
            for lf in labeled_frames_gt:
                lf.instances = lf.user_instances
            labeled_frames_gt = [
                lf for lf in labeled_frames_gt if len(lf.user_instances) > 0
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
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float32"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def match_instances(
    frame_gt: sio.LabeledFrame,
    frame_pr: sio.LabeledFrame,
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
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
    ):
        """Initialize the Evaluator class with ground-truth and predicted labels."""
        self.ground_truth_instances = ground_truth_instances
        self.predicted_instances = predicted_instances
        self.match_threshold = match_threshold
        self.oks_stddev = oks_stddev
        self.oks_scale = oks_scale
        self.user_labels_only = user_labels_only
        self.match_method = match_method
        self.anchor_ind = anchor_ind
        # Populated only in centroid / mask mode.
        self.false_positives = []
        # Matched-pair IoUs, populated only in mask mode.
        self.mask_ious = np.array([])
        # Per-frame mask records + matched TP mask pairs, populated only in mask
        # mode (feed mask_voc_metrics / boundary-IoU / fragmentation / per-size).
        self._mask_frames = []
        self._matched_mask_pairs = []

        self._process_frames()

    def _process_frames(self):
        self.frame_pairs = find_frame_pairs(
            self.ground_truth_instances, self.predicted_instances, self.user_labels_only
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
            gt_match_instances = get_instances(frame_gt)
            pr_match_instances = get_instances(frame_pr)

            # Collapse each predicted instance to its single centroid point.
            pred_centroids = np.array(
                [
                    self._collapse_pred_centroid(m.instance.numpy())
                    for m in pr_match_instances
                ]
            ).reshape(-1, 2)

            # GT centroids mirror generate_centroids exactly (#586).
            gt_centroids = np.array(
                [
                    compute_gt_centroids(m.instance.numpy(), self.anchor_ind)
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
            gt_masks = _frame_masks(frame_gt)
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
            pck_metrics = self.pck_metrics()
            match_scores = pck_metrics["pcks"].mean(axis=-1).mean(axis=-1)
            name = "pck_voc"
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
        return {"mOKS": pair_oks.mean()}

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
        mPCK_parts = pcks.mean(axis=0).mean(axis=-1)
        mPCK = mPCK_parts.mean()

        # Precompute PCK at common thresholds
        idx_5 = np.argmin(np.abs(thresholds - 5))
        idx_10 = np.argmin(np.abs(thresholds - 10))
        pck5 = pcks[:, :, idx_5].mean()
        pck10 = pcks[:, :, idx_10].mean()

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
            matched.
        save_metrics: Optional ``.npz`` path to save metrics to.
        match_method: ``"oks"``, ``"centroid"``, ``"mask"``, or ``"auto"``.
            ``"mask"`` matches predicted vs GT segmentation masks by IoU (for
            ``bottomup_segmentation`` models). ``"auto"`` switches to centroid
            mode when the PREDICTION skeleton is a single-node skeleton (e.g.
            ``sio.get_centroid_skeleton()``).
        anchor_part: Name of the GT skeleton node used to compute GT centroids
            (centroid mode). Resolved against the GT skeleton; ``None`` (or an
            absent name) falls back to the mean of visible nodes (#586).
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
    # raising "Empty Frame Pairs". Mask mode therefore never applies that filter.
    if match_method == "mask":
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **{"metrics": metrics})
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **{"metrics": metrics})
            logger.info(f"Metrics saved successfully to {save_path}")

        return metrics

    # Compute PCK at specific thresholds (5 and 10 pixels)
    dists = metrics["distance_metrics"]["dists"]
    dists_clean = np.copy(dists)
    dists_clean[np.isnan(dists_clean)] = np.inf
    pck_5 = (dists_clean < 5).mean()
    pck_10 = (dists_clean < 10).mean()

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
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics in SLEAP 1.4 format (single "metrics" key)
        np.savez_compressed(save_path, **{"metrics": metrics})
        logger.info(f"Metrics saved successfully to {save_path}")

    return metrics


# ---------------------------------------------------------------------------
# Retrieval / verification metrics for the `embedding` model type (SPEC §8).
#
# Pure numpy (+ sklearn ROC-AUC). Consume embedding matrices + integer group labels;
# they do NOT touch `.slp` instances (retrieval is appearance-only).
# ---------------------------------------------------------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def retrieval_metrics(gallery_emb, gallery_y, query_emb, query_y):
    """Rank-1 (CMC@1) + mAP of queries against a gallery (cosine similarity)."""
    g, q = _l2_normalize(np.asarray(gallery_emb)), _l2_normalize(np.asarray(query_emb))
    gy, qy = np.asarray(gallery_y), np.asarray(query_y)
    sim = q @ g.T
    order = np.argsort(-sim, axis=1)
    ranked = gy[order]
    rank1 = float(np.mean(ranked[:, 0] == qy))
    aps = []
    for i in range(len(qy)):
        rel = (ranked[i] == qy[i]).astype(float)
        if rel.sum() == 0:
            continue
        csum = np.cumsum(rel)
        prec = csum / np.arange(1, len(rel) + 1)
        aps.append((prec * rel).sum() / rel.sum())
    mAP = float(np.mean(aps)) if aps else 0.0
    return {"rank1": round(rank1, 4), "mAP": round(mAP, 4)}


def verification_metrics(
    gallery_emb, gallery_y, query_emb, query_y, exclude_diagonal: bool = False
):
    """ROC-AUC + EER over all query x gallery pairs (same vs different identity).

    When ``exclude_diagonal`` (gallery == query in the same order), the self-pairs on
    the similarity diagonal are dropped before scoring so a leave-self-out evaluation is
    not optimistically biased by ``N`` perfect same-identity matches at sim=1.0.
    """
    from sklearn.metrics import roc_auc_score

    g, q = _l2_normalize(np.asarray(gallery_emb)), _l2_normalize(np.asarray(query_emb))
    gy, qy = np.asarray(gallery_y), np.asarray(query_y)
    sim2d = q @ g.T
    same2d = (qy[:, None] == gy[None, :]).astype(int)
    if exclude_diagonal:
        keep = ~np.eye(sim2d.shape[0], sim2d.shape[1], dtype=bool)
        sim = sim2d[keep]
        same = same2d[keep]
    else:
        sim = sim2d.ravel()
        same = same2d.ravel()
    if same.min() == same.max():
        return {"auc": float("nan"), "eer": float("nan")}
    auc = float(roc_auc_score(same, sim))
    order = np.argsort(-sim)
    lab = same[order]
    P, N = lab.sum(), len(lab) - lab.sum()
    fnr = 1 - np.cumsum(lab) / max(P, 1)
    fpr = np.cumsum(1 - lab) / max(N, 1)
    j = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[j] + fpr[j]) / 2)
    return {"auc": round(auc, 4), "eer": round(eer, 4)}


def knn_classify(gallery_emb, gallery_y, query_emb, k: int = 7):
    """Cosine k-NN classification (weighted vote). Returns (pred, conf)."""
    g, q = _l2_normalize(np.asarray(gallery_emb)), _l2_normalize(np.asarray(query_emb))
    gy = np.asarray(gallery_y)
    sim = q @ g.T
    idx = np.argsort(-sim, 1)[:, :k]
    nn_y, nn_s = gy[idx], np.take_along_axis(sim, idx, 1)
    nclass = int(gy.max()) + 1
    votes = np.zeros((len(q), nclass))
    for c in range(nclass):
        votes[:, c] = (nn_s * (nn_y == c)).sum(1)
    pred = votes.argmax(1)
    conf = votes.max(1) / (np.abs(votes).sum(1) + 1e-8)
    return pred, conf


def embedding_full_eval(gallery_emb, gallery_y, query_emb, query_y, k: int = 7):
    """Combined retrieval + verification + kNN-accuracy metrics dict."""
    out = {}
    out.update(retrieval_metrics(gallery_emb, gallery_y, query_emb, query_y))
    out.update(verification_metrics(gallery_emb, gallery_y, query_emb, query_y))
    pred, _ = knn_classify(gallery_emb, gallery_y, query_emb, k=k)
    out["knn_acc"] = round(float(np.mean(pred == np.asarray(query_y))), 4)
    return out


def embedding_leave_self_out_eval(emb, y, k: int = 7):
    """Leave-self-out retrieval/verification/kNN over one labeled embedding set.

    Gallery == query == the same set, with each item's self-match excluded (the
    similarity diagonal is masked to -inf so an item is never retrieved by itself).
    This is exactly the protocol the per-epoch
    :class:`~sleap_nn.training.callbacks.EmbeddingEvaluationCallback` uses for
    checkpoint selection, so the post-training headline matches the selected metric.

    Args:
        emb: ``(N, D)`` embeddings.
        y: ``(N,)`` integer identity labels.
        k: ``k`` for the cosine-kNN accuracy (clamped to ``N - 1``).

    Returns:
        dict with ``rank1``, ``mAP``, ``auc``, ``eer``, ``knn_acc``.
    """
    emb = np.asarray(emb, dtype=np.float64)
    y = np.asarray(y)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    n = len(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)  # leave-self-out (self sorts to the very end)
    order = np.argsort(-sim, axis=1)[:, : n - 1]  # drop the self slot
    ranked = y[order]

    rank1 = float(np.mean(ranked[:, 0] == y))
    aps = []
    for i in range(n):
        rel = (ranked[i] == y[i]).astype(float)
        if rel.sum() == 0:
            continue
        csum = np.cumsum(rel)
        prec = csum / np.arange(1, len(rel) + 1)
        aps.append((prec * rel).sum() / rel.sum())
    mAP = float(np.mean(aps)) if aps else 0.0

    # kNN accuracy (leave-self-out): top-k excluding self.
    kk = min(k, n - 1)
    idx = order[:, :kk]
    nn_y = y[idx]
    nn_s = np.take_along_axis(sim, idx, 1)
    nclass = int(y.max()) + 1
    votes = np.zeros((n, nclass))
    for c in range(nclass):
        votes[:, c] = (nn_s * (nn_y == c)).sum(1)
    knn_acc = float(np.mean(votes.argmax(1) == y))

    ver = verification_metrics(emb, y, emb, y, exclude_diagonal=True)
    return {
        "rank1": round(rank1, 4),
        "mAP": round(mAP, 4),
        "auc": ver["auc"],
        "eer": ver["eer"],
        "knn_acc": round(knn_acc, 4),
    }
