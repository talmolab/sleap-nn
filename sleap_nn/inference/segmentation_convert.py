"""sleap-nn <-> sleap-io conversion helpers for instance segmentation masks.

Single owner of the mapping between a predicted boolean mask array and a
``sio.PredictedSegmentationMask`` (mirrors ``centroid_convert.py`` for
centroids), and of the inverse decode used by every mask consumer. Keeping
this in one place means the ``Outputs`` packaging path, the eval IoU, the
training data loader, and any future export path emit and read masks
identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import sleap_io as sio


def build_predicted_segmentation_mask(
    mask: np.ndarray,
    score: float,
    scale: Tuple[float, float] = (1.0, 1.0),
    offset: Tuple[float, float] = (0.0, 0.0),
) -> "sio.PredictedSegmentationMask":
    """Build a ``sio.PredictedSegmentationMask`` from a boolean mask array.

    Args:
        mask: 2-D boolean (or 0/1) array. At the model output-stride resolution
            when ``scale`` is non-identity, else at the original image resolution.
        score: Detection confidence (the instance-center peak value).
        scale: sio resolution ratio ``(sx, sy)`` with ``sx = mask_width /
            image_width`` (``image_coord = mask_coord / scale + offset``). The
            default ``(1.0, 1.0)`` encodes at the array's own resolution.
        offset: Origin ``(x, y)`` of the mask in image pixels. The segmentation
            layer keeps this ``(0.0, 0.0)`` because every preprocessing pad is
            bottom-right (valid content top-left aligned).

    Returns:
        A ``sio.PredictedSegmentationMask`` (RLE-backed) carrying ``score`` and
        the ``scale``/``offset`` that map it back to image pixels.
    """
    import sleap_io as sio

    mask = np.ascontiguousarray(mask, dtype=bool)
    return sio.PredictedSegmentationMask.from_numpy(
        mask,
        score=float(score),
        scale=(float(scale[0]), float(scale[1])),
        offset=(float(offset[0]), float(offset[1])),
    )


def decode_mask_to_image_res(m: "sio.SegmentationMask") -> np.ndarray:
    """Decode a sio segmentation mask to a boolean array on the IMAGE-pixel grid.

    sio ``.data`` decodes at the mask's stored (possibly output-stride)
    resolution. When the mask carries a non-identity ``scale`` (masks encoded
    at output-stride by :class:`SegmentationLayer`), this nearest-neighbor
    resamples it up to its image extent so every consumer — eval IoU
    (:func:`sleap_nn.evaluation._frame_masks`), the segmentation training data
    loader, and mask-IoU tracking — compares masks on a common original-image
    grid. Scale-1 masks (legacy full-res GT/preds) take a zero-copy fast path,
    so old ``.slp`` files behave exactly as before.

    Note:
        ``image_extent`` can differ from the true frame size by +/-1 px because
        the mask resolution is ``round(orig * scale)`` at encode time (e.g. a
        1024-px dimension can recover as 1025). It is therefore NOT authoritative
        for the frame dimensions; callers that index a real image must clamp to
        the actual frame size. For IoU on a shared max-canvas the +/-1 lands on a
        background row/col and does not change the result.
    """
    scale = getattr(m, "scale", (1.0, 1.0))
    if tuple(scale) == (1.0, 1.0):
        return np.asarray(m.data, dtype=bool)
    h, w = m.image_extent
    return np.asarray(m.resampled(int(h), int(w)).data, dtype=bool)


def build_predicted_roi(
    mask_obj: "sio.SegmentationMask",
    score: float,
    epsilon: float = 0.01,
) -> "Optional[sio.PredictedROI]":
    """Build a Douglas-Peucker-simplified ``sio.PredictedROI`` from a sio mask.

    ``mask_obj`` must carry its ``scale``/``offset`` so ``to_polygon()`` returns
    IMAGE-space geometry. The exterior silhouette is simplified with Shapely's
    Douglas-Peucker (``geometry.simplify``) at a tolerance of ``epsilon`` times
    the silhouette perimeter.

    Args:
        mask_obj: A sio (Predicted)SegmentationMask.
        score: Detection confidence carried onto the ROI.
        epsilon: Simplification tolerance as a fraction of the perimeter. ``0``
            keeps the raw polygon.

    Returns:
        A ``sio.PredictedROI``, or ``None`` when the mask has no polygonal
        silhouette (empty/degenerate) so the caller can skip it.
    """
    import sleap_io as sio

    roi = mask_obj.to_polygon()
    geom = getattr(roi, "geometry", None)
    if geom is None or geom.is_empty:
        return None
    if epsilon and epsilon > 0:
        simplified = geom.simplify(float(epsilon) * geom.length, preserve_topology=True)
        if not simplified.is_empty:
            geom = simplified
    return sio.PredictedROI(geometry=geom, score=float(score))
