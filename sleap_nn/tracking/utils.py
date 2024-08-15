"""Helper functions for Tracker module."""

from typing import Union
import numpy as np
import sleap_io as sio


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
    # TODO cached?
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
