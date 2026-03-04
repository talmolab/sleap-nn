"""Inference-level postprocessing filters for pose predictions.

This module provides filters that run after model inference but before tracking.
These filters are independent of tracking configuration and can be used standalone.
"""

from typing import List, Literal, Optional

import numpy as np
import sleap_io as sio


def filter_by_node_count(
    labels: sio.Labels,
    min_visible_nodes: int = 0,
    min_visible_node_fraction: float = 0.0,
) -> sio.Labels:
    """Filter instances with insufficient visible keypoints.

    Removes predicted instances that have too few detected/visible keypoints.
    This is useful for cleaning up spurious detections that only have 1-2 nodes
    or for requiring a minimum skeleton completeness.

    This filter runs independently of tracking and can be used to clean up
    model outputs before saving or further processing.

    Args:
        labels: Labels object with predicted instances to filter.
        min_visible_nodes: Minimum number of visible (non-NaN) keypoints required.
            Instances with fewer visible nodes are removed.
            Default: 0 (no filtering by absolute count).
        min_visible_node_fraction: Minimum fraction of skeleton nodes that must
            be visible. Value should be in [0, 1]. For example, 0.5 requires at
            least half of the skeleton's nodes to be detected.
            Default: 0.0 (no filtering by fraction).

    Returns:
        The input Labels object with low-node-count instances removed.
        Modification is done in place, but the object is also returned
        for convenience.

    Example:
        >>> # Require at least 3 visible nodes
        >>> labels = filter_by_node_count(labels, min_visible_nodes=3)
        >>> # Require at least 50% of skeleton nodes
        >>> labels = filter_by_node_count(labels, min_visible_node_fraction=0.5)
        >>> # Combine both criteria (must pass both)
        >>> labels = filter_by_node_count(
        ...     labels, min_visible_nodes=2, min_visible_node_fraction=0.3
        ... )

    Note:
        - Only affects predicted instances (preserves ground truth instances)
        - An instance must pass ALL specified criteria to be kept
        - A keypoint is "visible" if its coordinates are not NaN
    """
    # Early exit if no filtering requested
    if min_visible_nodes <= 0 and min_visible_node_fraction <= 0.0:
        return labels

    for lf in labels.labeled_frames:
        if len(lf.instances) == 0:
            continue

        kept_instances = []
        for inst in lf.instances:
            # Only filter predicted instances
            if not isinstance(inst, sio.PredictedInstance):
                kept_instances.append(inst)
                continue

            # Count visible nodes
            n_visible = _count_visible_nodes(inst)
            n_total = len(inst.skeleton.nodes)

            # Check absolute count criterion
            if min_visible_nodes > 0 and n_visible < min_visible_nodes:
                continue

            # Check fraction criterion
            if min_visible_node_fraction > 0.0:
                fraction = n_visible / n_total if n_total > 0 else 0.0
                if fraction < min_visible_node_fraction:
                    continue

            # Instance passed all criteria
            kept_instances.append(inst)

        lf.instances = kept_instances

    return labels


def filter_by_node_confidence(
    labels: sio.Labels,
    min_mean_node_score: float = 0.0,
    min_instance_score: float = 0.0,
) -> sio.Labels:
    """Filter instances with low confidence scores.

    Removes predicted instances based on their per-node confidence scores
    and/or overall instance score. This is useful for removing uncertain
    predictions that may have passed the peak threshold but are still
    low quality.

    This filter runs independently of tracking and can be used to clean up
    model outputs before saving or further processing.

    Args:
        labels: Labels object with predicted instances to filter.
        min_mean_node_score: Minimum mean confidence score across visible nodes.
            The mean is computed only over non-NaN keypoints.
            Default: 0.0 (no filtering by mean node score).
        min_instance_score: Minimum overall instance confidence score.
            Default: 0.0 (no filtering by instance score).

    Returns:
        The input Labels object with low-confidence instances removed.
        Modification is done in place, but the object is also returned
        for convenience.

    Example:
        >>> # Require mean node confidence >= 0.5
        >>> labels = filter_by_node_confidence(labels, min_mean_node_score=0.5)
        >>> # Require instance score >= 0.3
        >>> labels = filter_by_node_confidence(labels, min_instance_score=0.3)
        >>> # Combine both criteria
        >>> labels = filter_by_node_confidence(
        ...     labels, min_mean_node_score=0.4, min_instance_score=0.2
        ... )

    Note:
        - Only affects predicted instances (preserves ground truth instances)
        - An instance must pass ALL specified criteria to be kept
        - If point_scores is not available, mean node score check is skipped
        - If instance score is not available, instance score check is skipped
    """
    # Early exit if no filtering requested
    if min_mean_node_score <= 0.0 and min_instance_score <= 0.0:
        return labels

    for lf in labels.labeled_frames:
        if len(lf.instances) == 0:
            continue

        kept_instances = []
        for inst in lf.instances:
            # Only filter predicted instances
            if not isinstance(inst, sio.PredictedInstance):
                kept_instances.append(inst)
                continue

            # Check instance score criterion
            if min_instance_score > 0.0:
                inst_score = _instance_score(inst)
                if inst_score < min_instance_score:
                    continue

            # Check mean node score criterion
            if min_mean_node_score > 0.0:
                mean_score = _mean_node_score(inst)
                if mean_score is not None and mean_score < min_mean_node_score:
                    continue

            # Instance passed all criteria
            kept_instances.append(inst)

        lf.instances = kept_instances

    return labels


def _count_visible_nodes(instance: sio.PredictedInstance) -> int:
    """Count the number of visible (non-NaN) keypoints in an instance.

    Args:
        instance: Predicted instance.

    Returns:
        Number of keypoints with valid (non-NaN) coordinates.
    """
    pts = instance.numpy()  # (n_nodes, 2)
    valid = ~np.isnan(pts).any(axis=1)
    return int(valid.sum())


def _mean_node_score(instance: sio.PredictedInstance) -> Optional[float]:
    """Compute mean confidence score across visible nodes.

    Args:
        instance: Predicted instance.

    Returns:
        Mean confidence score, or None if point scores are not available.
    """
    # Point scores are stored in the structured points array as 'score' field
    try:
        point_scores = instance.points["score"]
    except (KeyError, TypeError, IndexError):
        return None

    if point_scores is None or len(point_scores) == 0:
        return None

    point_scores = np.asarray(point_scores)

    # Only consider scores for visible nodes
    pts = instance.numpy()
    valid = ~np.isnan(pts).any(axis=1)

    if not valid.any():
        return 0.0

    valid_scores = point_scores[valid]
    # Handle NaN scores
    valid_scores = valid_scores[~np.isnan(valid_scores)]

    if len(valid_scores) == 0:
        return 0.0

    return float(np.mean(valid_scores))


def filter_overlapping_instances(
    labels: sio.Labels,
    threshold: float = 0.8,
    method: Literal["iou", "oks"] = "iou",
) -> sio.Labels:
    """Filter overlapping instances using greedy non-maximum suppression.

    Removes duplicate/overlapping instances by applying greedy NMS based on
    either bounding box IOU or Object Keypoint Similarity (OKS). When two
    instances overlap above the threshold, the lower-scoring one is removed.

    This filter runs independently of tracking and can be used to clean up
    model outputs before saving or further processing.

    Args:
        labels: Labels object with predicted instances to filter.
        threshold: Similarity threshold for considering instances as overlapping.
            Instances with similarity > threshold are candidates for removal.
            Lower values are more aggressive (remove more).
            Typical values: 0.3 (aggressive) to 0.8 (permissive).
        method: Similarity metric to use for comparing instances.
            "iou": Bounding box intersection-over-union.
            "oks": Object Keypoint Similarity (pose-based).

    Returns:
        The input Labels object with overlapping instances removed.
        Modification is done in place, but the object is also returned
        for convenience.

    Example:
        >>> # Filter instances with >80% bounding box overlap
        >>> labels = filter_overlapping_instances(labels, threshold=0.8, method="iou")
        >>> # Filter using OKS similarity
        >>> labels = filter_overlapping_instances(labels, threshold=0.5, method="oks")

    Note:
        - Only affects frames with 2+ predicted instances
        - Uses instance.score for ranking; higher scores are preferred
        - For IOU: bounding boxes computed from non-NaN keypoints
        - For OKS: uses standard COCO OKS formula with bbox-derived scale
    """
    for lf in labels.labeled_frames:
        if len(lf.instances) <= 1:
            continue

        # Separate predicted instances (have scores) from other instances
        predicted = []
        other = []
        for inst in lf.instances:
            if isinstance(inst, sio.PredictedInstance):
                predicted.append(inst)
            else:
                other.append(inst)

        # Only filter predicted instances
        if len(predicted) <= 1:
            continue

        # Get scores
        scores = np.array([_instance_score(inst) for inst in predicted])

        # Apply greedy NMS with selected method
        if method == "iou":
            bboxes = np.array([_instance_bbox(inst) for inst in predicted])
            keep_indices = _nms_greedy_iou(bboxes, scores, threshold)
        elif method == "oks":
            points = [inst.numpy() for inst in predicted]
            keep_indices = _nms_greedy_oks(points, scores, threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iou' or 'oks'.")

        # Reconstruct instance list: kept predicted + other instances
        kept_predicted = [predicted[i] for i in keep_indices]
        lf.instances = kept_predicted + other

    return labels


def _instance_bbox(instance: sio.PredictedInstance) -> np.ndarray:
    """Compute axis-aligned bounding box from instance keypoints.

    Args:
        instance: Instance with keypoints.

    Returns:
        Bounding box as [xmin, ymin, xmax, ymax].
        Returns [0, 0, 0, 0] if no valid keypoints.
    """
    pts = instance.numpy()  # (n_nodes, 2)
    valid = ~np.isnan(pts).any(axis=1)

    if not valid.any():
        return np.array([0.0, 0.0, 0.0, 0.0])

    pts = pts[valid]
    return np.array(
        [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
    )


def _instance_score(instance: sio.PredictedInstance) -> float:
    """Get instance confidence score.

    Args:
        instance: Predicted instance.

    Returns:
        Instance score, or 1.0 if not available.
    """
    return getattr(instance, "score", 1.0)


def _nms_greedy_iou(
    bboxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> List[int]:
    """Apply greedy NMS using bounding box IOU.

    Args:
        bboxes: Bounding boxes of shape (N, 4) as [xmin, ymin, xmax, ymax].
        scores: Confidence scores of shape (N,).
        threshold: IOU threshold for suppression.

    Returns:
        List of indices to keep, in order of decreasing score.
    """
    if len(bboxes) == 0:
        return []

    # Sort by score descending
    order = scores.argsort()[::-1].tolist()

    keep = []
    while order:
        # Take highest scoring remaining instance
        i = order.pop(0)
        keep.append(i)

        if not order:
            break

        # Compute IOU with all remaining instances
        remaining_indices = np.array(order)
        similarities = _compute_iou_one_to_many(bboxes[i], bboxes[remaining_indices])

        # Keep only instances with similarity <= threshold
        mask = similarities <= threshold
        order = [order[j] for j in range(len(order)) if mask[j]]

    return keep


def _nms_greedy_oks(
    points_list: List[np.ndarray],
    scores: np.ndarray,
    threshold: float,
) -> List[int]:
    """Apply greedy NMS using Object Keypoint Similarity (OKS).

    Args:
        points_list: List of keypoint arrays, each of shape (n_nodes, 2).
        scores: Confidence scores of shape (N,).
        threshold: OKS threshold for suppression.

    Returns:
        List of indices to keep, in order of decreasing score.
    """
    if len(points_list) == 0:
        return []

    # Sort by score descending
    order = scores.argsort()[::-1].tolist()

    keep = []
    while order:
        # Take highest scoring remaining instance
        i = order.pop(0)
        keep.append(i)

        if not order:
            break

        # Compute OKS with all remaining instances
        similarities = np.array(
            [_compute_oks(points_list[i], points_list[j]) for j in order]
        )

        # Keep only instances with similarity <= threshold
        mask = similarities <= threshold
        order = [order[j] for j in range(len(order)) if mask[j]]

    return keep


def _compute_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IOU between one box and multiple boxes.

    Args:
        box: Single box of shape (4,) as [xmin, ymin, xmax, ymax].
        boxes: Multiple boxes of shape (N, 4).

    Returns:
        IOU values of shape (N,).
    """
    # Intersection coordinates
    inter_xmin = np.maximum(box[0], boxes[:, 0])
    inter_ymin = np.maximum(box[1], boxes[:, 1])
    inter_xmax = np.minimum(box[2], boxes[:, 2])
    inter_ymax = np.minimum(box[3], boxes[:, 3])

    # Intersection area (0 if no overlap)
    inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Individual areas
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Union area
    union_area = area_a + area_b - inter_area

    # IOU (avoid division by zero)
    return np.where(union_area > 0, inter_area / union_area, 0.0)


def _compute_oks(
    points_a: np.ndarray,
    points_b: np.ndarray,
    kappa: float = 0.1,
) -> float:
    """Compute Object Keypoint Similarity (OKS) between two instances.

    Uses a simplified OKS formula where all keypoints have equal weight
    and scale is derived from the bounding box of the reference instance.

    Args:
        points_a: Keypoints of first instance, shape (n_nodes, 2).
        points_b: Keypoints of second instance, shape (n_nodes, 2).
        kappa: Per-keypoint constant controlling falloff. Default 0.1.

    Returns:
        OKS value in [0, 1]. Higher means more similar.
    """
    # Find valid keypoints (present in both instances)
    valid_a = ~np.isnan(points_a).any(axis=1)
    valid_b = ~np.isnan(points_b).any(axis=1)
    valid = valid_a & valid_b

    if not valid.any():
        return 0.0

    # Compute scale from bounding box area of instance A
    pts_a_valid = points_a[valid_a]
    if len(pts_a_valid) < 2:
        return 0.0

    bbox_w = pts_a_valid[:, 0].max() - pts_a_valid[:, 0].min()
    bbox_h = pts_a_valid[:, 1].max() - pts_a_valid[:, 1].min()
    scale_sq = bbox_w * bbox_h

    if scale_sq <= 0:
        return 0.0

    # Compute squared distances for valid keypoints
    d_sq = np.sum((points_a[valid] - points_b[valid]) ** 2, axis=1)

    # OKS formula: mean of exp(-d^2 / (2 * s^2 * k^2))
    oks_per_kpt = np.exp(-d_sq / (2 * scale_sq * kappa**2))

    return float(np.mean(oks_per_kpt))
