"""Backward-compatibility re-export shim for peak finding.

The implementations live in :mod:`sleap_nn.inference.ops.peaks` and
:mod:`sleap_nn.inference.ops.crops` after PR 1 of #508. This module
preserves the old import path for existing callers; it is scheduled for
deletion in #519 alongside the rest of the legacy inference layout.
"""

from sleap_nn.inference.ops.crops import crop_bboxes
from sleap_nn.inference.ops.peaks import (
    find_global_peaks,
    find_global_peaks_rough,
    find_local_peaks,
    find_local_peaks_rough,
    integral_regression,
    morphological_dilation,
)

__all__ = [
    "crop_bboxes",
    "find_global_peaks",
    "find_global_peaks_rough",
    "find_local_peaks",
    "find_local_peaks_rough",
    "integral_regression",
    "morphological_dilation",
]
