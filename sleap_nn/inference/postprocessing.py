"""Inference-level postprocessing filters for pose predictions.

This module provides filters that run after model inference but before tracking.
These filters are independent of tracking configuration and can be used standalone.
"""

from typing import List, Literal

import numpy as np
import sleap_io as sio


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
