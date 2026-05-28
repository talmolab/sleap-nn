"""Backward-compatibility re-export shim for identity ops.

The implementations live in :mod:`sleap_nn.inference.ops.identity` after
PR 1 of #508. This module preserves the old import path for existing callers;
it is scheduled for deletion in #519.
"""

from sleap_nn.inference.ops.identity import (
    classify_peaks_from_maps,
    get_class_inds_from_vectors,
    group_class_peaks,
)

__all__ = [
    "classify_peaks_from_maps",
    "get_class_inds_from_vectors",
    "group_class_peaks",
]
