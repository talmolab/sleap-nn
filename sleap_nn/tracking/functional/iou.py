"""Intersection-over-Union functions."""

import numpy as np

from sleap_nn.tracking.core.instance import Instance


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Computes the intersection over union for a pair of bounding boxes.

    Args:
        bbox1: Bounding box specified by corner coordinates [y1, x1, y2, x2].
        bbox2: Bounding box specified by corner coordinates [y1, x1, y2, x2].

    Returns:
        A float scalar calculated as the ratio between the areas of the intersection
        and the union of the two bounding boxes.
    """

    bbox1_y1, bbox1_x1, bbox1_y2, bbox1_x2 = bbox1
    bbox2_y1, bbox2_x1, bbox2_y2, bbox2_x2 = bbox2

    intersection_y1 = max(bbox1_y1, bbox2_y1)
    intersection_x1 = max(bbox1_x1, bbox2_x1)
    intersection_y2 = min(bbox1_y2, bbox2_y2)
    intersection_x2 = min(bbox1_x2, bbox2_x2)

    intersection_area = max(intersection_x2 - intersection_x1 + 1, 0) * max(
        intersection_y2 - intersection_y1 + 1, 0
    )

    bbox1_area = (bbox1_x2 - bbox1_x1 + 1) * (bbox1_y2 - bbox1_y1 + 1)
    bbox2_area = (bbox2_x2 - bbox2_x1 + 1) * (bbox2_y2 - bbox2_y1 + 1)

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def instance_iou(
    ref_instance: Instance, query_instance: Instance, cache: dict = dict()
) -> float:
    """Computes IOU between bounding boxes of instances."""

    if ref_instance not in cache:
        cache[ref_instance] = ref_instance.bounding_box

    if query_instance not in cache:
        cache[query_instance] = query_instance.bounding_box

    a = cache[ref_instance]
    b = cache[query_instance]

    return compute_iou(a, b)