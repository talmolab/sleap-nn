"""sleap-nn → sleap-io conversion helpers for instance segmentation masks.

Single owner of the mapping from a predicted boolean mask array to a
``sio.PredictedSegmentationMask`` (mirrors ``centroid_convert.py`` for
centroids). Keeping this in one place means the ``Outputs`` packaging path and
any future export path emit masks identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sleap_io as sio


def build_predicted_segmentation_mask(
    mask: np.ndarray,
    score: float,
) -> "sio.PredictedSegmentationMask":
    """Build a ``sio.PredictedSegmentationMask`` from a boolean mask array.

    Args:
        mask: 2-D boolean (or 0/1) array at the original image resolution.
        score: Detection confidence (the instance-center peak value).

    Returns:
        A ``sio.PredictedSegmentationMask`` (RLE-backed) carrying ``score``.
    """
    import sleap_io as sio

    mask = np.ascontiguousarray(mask, dtype=bool)
    return sio.PredictedSegmentationMask.from_numpy(mask, score=float(score))
