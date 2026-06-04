"""Helper functions for Tracker module."""

from typing import List, Tuple, Union, Optional
from scipy.optimize import linear_sum_assignment
import operator
import numpy as np
import sleap_io as sio


def hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Match new instances to existing tracks using Hungarian matching."""
    # Replace inf/nan with a large finite value so linear_sum_assignment doesn't
    # raise "cost matrix is infeasible".
    invalid = ~np.isfinite(cost_matrix)
    if invalid.any():
        cost_matrix = np.copy(cost_matrix)
        finite_vals = cost_matrix[~invalid]
        fill = (np.abs(finite_vals).max() * 10 + 1) if finite_vals.size > 0 else 1e6
        cost_matrix[invalid] = fill

    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return row_ids, col_ids


def greedy_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Match new instances to existing tracks using greedy bipartite matching."""
    # Sort edges by ascending cost.
    rows, cols = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    unassigned_edges = list(zip(rows, cols))

    # Greedily assign edges.
    row_inds, col_inds = [], []
    while len(unassigned_edges) > 0:
        # Assign the lowest cost edge.
        row_ind, col_ind = unassigned_edges.pop(0)
        row_inds.append(row_ind)
        col_inds.append(col_ind)

        # Remove all other edges that contain either node (in reverse order).
        for i in range(len(unassigned_edges) - 1, -1, -1):
            if unassigned_edges[i][0] == row_ind or unassigned_edges[i][1] == col_ind:
                del unassigned_edges[i]

    return row_inds, col_inds


def get_keypoints(pred_instance: Union[sio.PredictedInstance, np.ndarray]):
    """Return keypoints as np.array from the `PredictedInstance` object."""
    if isinstance(pred_instance, np.ndarray):
        return pred_instance
    return pred_instance.numpy()


def get_centroid(pred_instance: Union[sio.PredictedInstance, np.ndarray]):
    """Return the centroid of the `PredictedInstance` object."""
    pts = pred_instance
    if not isinstance(pred_instance, np.ndarray):
        pts = pred_instance.numpy()
    centroid = np.nanmedian(pts, axis=0)
    return centroid


def get_bbox(pred_instance: Union[sio.PredictedInstance, np.ndarray]):
    """Return the bounding box coordinates for the `PredictedInstance` object."""
    points = (
        pred_instance.numpy()
        if not isinstance(pred_instance, np.ndarray)
        else pred_instance
    )
    bbox = np.concatenate(
        [
            np.nanmin(points, axis=0),
            np.nanmax(points, axis=0),
        ]  # [xmin, ymin, xmax, ymax]
    )
    return bbox


def is_segmentation_mask(obj) -> bool:
    """True if ``obj`` is a segmentation mask rather than a keypoint instance.

    Mask tracking flows ``sio.PredictedSegmentationMask`` objects through the
    same code paths as ``sio.PredictedInstance``; this is the single predicate
    used to dispatch the pose-vs-mask differences (no ``.numpy()`` keypoints).
    """
    return isinstance(obj, (sio.PredictedSegmentationMask, sio.SegmentationMask))


class MaskFeature:
    """Compact mask feature for fast IoU: a tight bbox crop + absolute offset.

    A full-resolution segmentation mask (e.g. 1280x1024) is almost all
    background, so the per-pair mask-IoU only needs the foreground bbox: store
    the cropped boolean array (``crop``), its absolute top-left ``(y0, x0)`` in
    the original image frame, and the foreground ``area`` (px, precomputed). IoU
    between two features then AND-s only the *overlap* of their bboxes (usually
    tiny or empty) and reads ``area`` in O(1) — vs. allocating + AND/OR-ing a
    full-resolution canvas per pair. Coordinates are absolute, so this matches
    the full-canvas :func:`sleap_nn.evaluation._mask_iou` exactly.
    """

    __slots__ = ("crop", "y0", "x0", "area")

    def __init__(self, crop: np.ndarray, y0: int, x0: int, area: int):
        """Store the bbox crop, its absolute top-left ``(y0, x0)`` and area."""
        self.crop = crop
        self.y0 = int(y0)
        self.x0 = int(x0)
        self.area = int(area)


def _mask_feature_from_dense(data: np.ndarray) -> MaskFeature:
    """Build a :class:`MaskFeature` from a dense bool mask (scans for the bbox)."""
    data = np.ascontiguousarray(data, dtype=bool)
    rows = np.any(data, axis=1)
    if not rows.any():
        return MaskFeature(np.zeros((0, 0), dtype=bool), 0, 0, 0)
    cols = np.any(data, axis=0)
    y0 = int(np.argmax(rows))
    y1 = len(rows) - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = len(cols) - int(np.argmax(cols[::-1]))
    crop = data[y0:y1, x0:x1]
    return MaskFeature(crop, y0, x0, int(np.count_nonzero(crop)))


def get_mask(
    pred_mask: Union["sio.PredictedSegmentationMask", np.ndarray, MaskFeature],
) -> MaskFeature:
    """Return a compact :class:`MaskFeature` for a `PredictedSegmentationMask`.

    Mirrors :func:`get_keypoints`/:func:`get_bbox` as the ``"masks"`` feature
    extractor. The mask is decoded onto the **image-pixel grid** first, then
    cropped to its foreground bbox (using the mask's own ``.bbox`` when available,
    avoiding a scan); the crop is cached as the candidate feature so scoring
    re-uses it without re-decoding and only touches the foreground region.

    The image-grid decode is essential: sio ``.data`` decodes at the mask's
    *stored* resolution, which for the default inference path
    (``full_res_masks=False``, masks encoded at output-stride, ``scale~=0.5``) is
    NOT the image grid, while ``.bbox`` is always in IMAGE space. Cropping the
    stride-res ``.data`` with image-space bbox indices would slice the wrong
    region (often entirely out of bounds -> empty -> ``compute_mask_iou`` 1.0 for
    every pair, scrambling identity). :func:`decode_mask_to_image_res` is a
    zero-copy passthrough for ``scale==1`` masks (legacy full-res), so that path
    is unchanged.
    """
    if isinstance(pred_mask, MaskFeature):
        return pred_mask
    if isinstance(pred_mask, np.ndarray):
        return _mask_feature_from_dense(pred_mask)
    from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

    data = decode_mask_to_image_res(pred_mask)
    bbox = getattr(pred_mask, "bbox", None)
    if bbox is not None:
        # PredictedSegmentationMask.bbox is (x, y, width, height) (XYWH), in IMAGE
        # space -- consistent with the image-grid `data` decoded above.
        x, y, w, h = (int(round(float(v))) for v in bbox)
        height, width = data.shape
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(height, y + h), min(width, x + w)
        if y1 > y0 and x1 > x0:
            crop = data[y0:y1, x0:x1]
            return MaskFeature(crop, y0, x0, int(np.count_nonzero(crop)))
    return _mask_feature_from_dense(data)


def count_valid_points(obj) -> int:
    """Number of valid points used to gate track spawning/matching.

    For a keypoint instance this is the count of non-NaN nodes; for a
    segmentation mask there are no keypoints, so the mask area (foreground px)
    is the analogous "support" measure (``min_new_track_points`` /
    ``min_match_points`` then read as a pixel-area threshold; default 0 keeps
    every non-empty mask and drops empty ones).
    """
    if is_segmentation_mask(obj):
        return int(obj.area)
    points = obj if isinstance(obj, np.ndarray) else obj.numpy()
    return int((~np.isnan(points).any(axis=1)).sum())


def compute_euclidean_distance(a, b):
    """Return the negative euclidean distance between a and b points."""
    return -np.linalg.norm(a - b)


def compute_iou(a, b):
    """Return the intersection over union for given a and b bounding boxes [xmin, ymin, xmax, ymax]."""
    (xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2) = a, b

    xmin_intersection = max(xmin1, xmin2)
    ymin_intersection = max(ymin1, ymin2)
    xmax_intersection = min(xmax1, xmax2)
    ymax_intersection = min(ymax1, ymax2)

    intersection_area = max(0, xmax_intersection - xmin_intersection + 1) * max(
        0, ymax_intersection - ymin_intersection + 1
    )
    bbox1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    bbox2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def compute_mask_iou(a, b) -> float:
    """Return the IoU of two segmentation masks (higher = more similar).

    The ``"mask_iou"`` scoring method. Operates on :class:`MaskFeature` (the
    ``"masks"`` feature); raw dense bool arrays are accepted too (coerced via
    :func:`get_mask`). The intersection is computed only over the *overlap* of
    the two foreground bboxes — see :class:`MaskFeature` — which is numerically
    identical to the full-canvas :func:`sleap_nn.evaluation._mask_iou` (top-left
    aligned, shape-mismatch safe, empty/empty -> 1.0) but avoids touching the
    background. This is a similarity, like ``oks``/``iou``; the cost negation
    (``cost = -score``) happens in :meth:`Tracker.scores_to_cost_matrix`, so it
    must NOT be negated here. Pixel IoU (not bbox-IoU on ``mask.bbox``) sidesteps
    the XYWH-vs-XYXY bbox-format issue.
    """
    fa = a if isinstance(a, MaskFeature) else get_mask(a)
    fb = b if isinstance(b, MaskFeature) else get_mask(b)
    inter = _mask_feature_intersection(fa, fb)
    union = fa.area + fb.area - inter
    # union == 0 only when both masks are empty -> identical -> 1.0 (matches the
    # _mask_iou degenerate contract).
    return 1.0 if union == 0 else float(inter / union)


def _mask_feature_intersection(fa: MaskFeature, fb: MaskFeature) -> int:
    """Foreground-overlap pixel count between two :class:`MaskFeature` (px)."""
    if fa.area == 0 or fb.area == 0:
        return 0
    ay1, ax1 = fa.y0 + fa.crop.shape[0], fa.x0 + fa.crop.shape[1]
    by1, bx1 = fb.y0 + fb.crop.shape[0], fb.x0 + fb.crop.shape[1]
    oy0, oy1 = max(fa.y0, fb.y0), min(ay1, by1)
    ox0, ox1 = max(fa.x0, fb.x0), min(ax1, bx1)
    if oy1 <= oy0 or ox1 <= ox0:
        return 0  # bboxes disjoint -> no intersection (no array op)
    a_sub = fa.crop[oy0 - fa.y0 : oy1 - fa.y0, ox0 - fa.x0 : ox1 - fa.x0]
    b_sub = fb.crop[oy0 - fb.y0 : oy1 - fb.y0, ox0 - fb.x0 : ox1 - fb.x0]
    return int(np.count_nonzero(a_sub & b_sub))


def compute_cosine_sim(a, b):
    """Return cosine simalirity between a and b vectors."""
    number = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine_sim = number / denom
    return cosine_sim


def nms_fast(boxes, scores, iou_threshold, target_count=None) -> List[int]:
    """From: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/."""
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if we already have fewer boxes than the target count, return all boxes
    if target_count and len(boxes) < target_count:
        return list(range(len(boxes)))

    # if the bounding boxes coordinates are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    picked_idxs = []

    # init list of boxes removed by nms
    nms_idxs = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by their scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # we want to add the best box which is the last box in sorted list
        picked_box_idx = idxs[-1]

        # last = len(idxs) - 1
        # i = idxs[last]
        picked_idxs.append(picked_box_idx)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[picked_box_idx], x1[idxs[:-1]])
        yy1 = np.maximum(y1[picked_box_idx], y1[idxs[:-1]])
        xx2 = np.minimum(x2[picked_box_idx], x2[idxs[:-1]])
        yy2 = np.minimum(y2[picked_box_idx], y2[idxs[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]

        # find boxes with iou over threshold
        nms_for_new_box = np.where(overlap > iou_threshold)[0]
        nms_idxs.extend(list(idxs[nms_for_new_box]))

        # delete new box (last in list) plus nms boxes
        idxs = np.delete(idxs, nms_for_new_box)[:-1]

    # if we're below the target number of boxes, add some back
    if target_count and nms_idxs and len(picked_idxs) < target_count:
        # sort by descending score
        nms_idxs.sort(key=lambda idx: -scores[idx])

        add_back_count = min(len(nms_idxs), len(picked_idxs) - target_count)
        picked_idxs.extend(nms_idxs[:add_back_count])

    # return the list of picked boxes
    return picked_idxs


def nms_instances(
    instances, iou_threshold, target_count=None
) -> Tuple[List[sio.PredictedInstance], List[sio.PredictedInstance]]:
    """NMS for instances."""
    # get_bbox: # [xmin, ymin, xmax, ymax]
    boxes = np.array([get_bbox(inst) for inst in instances])
    scores = np.array([inst.score for inst in instances])
    picks = nms_fast(boxes, scores, iou_threshold, target_count)

    to_keep = [inst for i, inst in enumerate(instances) if i in picks]
    to_remove = [inst for i, inst in enumerate(instances) if i not in picks]

    return to_keep, to_remove


def cull_instances(
    frames: List[sio.LabeledFrame],
    instance_count: int,
    iou_threshold: Optional[float] = None,
):
    """Removes instances from frames over instance per frame threshold.

    Args:
        frames: The list of `LabeledFrame` objects with predictions.
        instance_count: The maximum number of instances we want per frame.
        iou_threshold: Intersection over Union (IOU) threshold to use when
            removing overlapping instances over target count; if None, then
            only use score to determine which instances to remove.

    Returns:
        None; modifies frames in place.
    """
    if not frames:
        return

    frames.sort(key=lambda lf: lf.frame_idx)

    lf_inst_list = []
    # Find all frames with more instances than the desired threshold
    for lf in frames:
        if len(lf.predicted_instances) > instance_count:
            # List of instances which we'll pare down
            keep_instances = lf.predicted_instances

            # Use NMS to remove overlapping instances over target count
            if iou_threshold:
                keep_instances, extra_instances = nms_instances(
                    keep_instances,
                    iou_threshold=iou_threshold,
                    target_count=instance_count,
                )
                # Mark for removal
                lf_inst_list.extend([(lf, inst) for inst in extra_instances])

            # Use lower score to remove instances over target count
            if len(keep_instances) > instance_count:
                # Sort by ascending score, get target number of instances
                # from the end of list (i.e., with highest score)
                extra_instances = sorted(
                    keep_instances, key=operator.attrgetter("score")
                )[:-instance_count]

                # Mark for removal
                lf_inst_list.extend([(lf, inst) for inst in extra_instances])

    # Remove instances over per frame threshold
    for lf, inst in lf_inst_list:
        filtered_instances = []
        for instance in lf.instances:
            if not instance.same_pose_as(inst):
                filtered_instances.append(instance)
        lf.instances = filtered_instances

    return frames


def cull_frame_instances(
    instances_list: List[sio.PredictedInstance],
    instance_count: int,
    iou_threshold: Optional[float] = None,
) -> List[sio.PredictedInstance]:
    """Removes instances (for single frame) over instance per frame threshold.

    Args:
        instances_list: The list of instances for a single frame.
        instance_count: The maximum number of instances we want per frame.
        iou_threshold: Intersection over Union (IOU) threshold to use when
            removing overlapping instances over target count; if None, then
            only use score to determine which instances to remove.

    Returns:
        Updated list of frames, also modifies frames in place.
    """
    if not instances_list:
        return

    if len(instances_list) > instance_count:
        # List of instances which we'll pare down
        keep_instances = instances_list

        # Use NMS to remove overlapping instances over target count
        if iou_threshold:
            keep_instances, extra_instances = nms_instances(
                keep_instances,
                iou_threshold=iou_threshold,
                target_count=instance_count,
            )
            updated_instances_list = []
            # Remove the extra instances
            for inst in extra_instances:
                for instance in instances_list:
                    if not instance.same_pose_as(inst):
                        updated_instances_list.append(instance)
            instances_list = updated_instances_list

        # Use lower score to remove instances over target count
        if len(keep_instances) > instance_count:
            # Sort by ascending score, get target number of instances
            # from the end of list (i.e., with highest score)
            extra_instances = sorted(keep_instances, key=operator.attrgetter("score"))[
                :-instance_count
            ]

            # Remove the extra instances
            updated_instances_list = []
            for inst in extra_instances:
                for instance in instances_list:
                    if instance.same_pose_as(inst):
                        updated_instances_list.append(instance)
            instances_list = updated_instances_list

    return instances_list
