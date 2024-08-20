"""Helper functions for Tracker module."""

from typing import List, Tuple, Union
from scipy.optimize import linear_sum_assignment

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
    numer = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine_sim = numer / denom
    return cosine_sim
