"""Backward-compatibility re-export shim for post-inference filters.

The implementations live in :mod:`sleap_nn.inference.ops.filters` after PR 1
of #508. This module preserves the old import path for existing callers;
it is scheduled for deletion in #519.
"""

from sleap_nn.inference.ops.filters import (
    filter_by_node_confidence,
    filter_by_node_count,
    filter_overlapping_instances,
)

# These leading-underscore names are imported by the predictor's per-frame
# filtering loop, so they have to be re-exported even though they are
# nominally private.
from sleap_nn.inference.ops.filters import (  # noqa: F401
    _compute_iou_one_to_many,
    _compute_oks,
    _count_visible_nodes,
    _instance_bbox,
    _instance_score,
    _mean_node_score,
    _nms_greedy_iou,
    _nms_greedy_oks,
)

__all__ = [
    "filter_by_node_confidence",
    "filter_by_node_count",
    "filter_overlapping_instances",
]
