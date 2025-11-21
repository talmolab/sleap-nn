"""Helper functions for Tracker module."""

from typing import List, Tuple, Union, Optional
from scipy.optimize import linear_sum_assignment
import operator
import numpy as np
import sleap_io as sio


def hungarian_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Match new instances to existing tracks using Hungarian matching."""
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
