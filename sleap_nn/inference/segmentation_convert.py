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
    (:func:`sleap_nn.evaluation._frame_masks`) and the segmentation training data
    loader — compares masks on a common original-image grid. Scale-1,
    offset-0 masks (legacy full-res GT/preds, all bottom-up predictions) take a
    zero-copy fast path, so old ``.slp`` files behave exactly as before.

    A non-identity ``offset`` (the crop origin ``(x, y)`` of a top-down
    crop-centered mask) is baked in by top-left zero-padding so the mask lands
    at its full-frame location for IoU/placement; without this, two crops at
    different offsets would both decode to the origin and collide (sio
    ``resampled``/``image_extent`` drop the offset). Offset-0 masks are
    unaffected.

    Note:
        ``image_extent`` can differ from the true frame size by +/-1 px because
        the mask resolution is ``round(orig * scale)`` at encode time (e.g. a
        1024-px dimension can recover as 1025). It is therefore NOT authoritative
        for the frame dimensions; callers that index a real image must clamp to
        the actual frame size. For IoU on a shared max-canvas the +/-1 lands on a
        background row/col and does not change the result.
    """
    scale = tuple(getattr(m, "scale", (1.0, 1.0)))
    offset = tuple(getattr(m, "offset", (0.0, 0.0)))
    # Fast path: full-res, origin-anchored masks (legacy full-res GT/preds and
    # every bottom-up prediction, which always emits offset (0, 0)) are returned
    # zero-copy, byte-identical to the pre-offset-aware behavior.
    if scale == (1.0, 1.0) and offset == (0.0, 0.0):
        return np.asarray(m.data, dtype=bool)

    # Honor scale: decode up to the mask's image extent.
    if scale == (1.0, 1.0):
        arr = np.asarray(m.data, dtype=bool)
    else:
        h, w = m.image_extent
        arr = np.asarray(m.resampled(int(h), int(w)).data, dtype=bool)

    # Honor offset: place the decoded array at its image-space origin by
    # top-left zero-padding. sio ``offset`` is ``(x, y)`` image pixels
    # (``image_coord = mask_coord / scale + offset``); ``image_extent``/
    # ``resampled`` drop it (mask.py), and the eval consumers (``_align_pair`` /
    # ``_mask_iou``) top-left-align masks on a shared max-canvas. Baking the
    # offset here is what makes a crop-centered (top-down) mask at offset
    # ``(x0, y0)`` score at the correct full-frame location instead of colliding
    # with every other crop at the origin. Negative offsets (a crop spilling off
    # the top/left edge) are clipped to the in-frame region.
    ox, oy = int(round(offset[0])), int(round(offset[1]))
    if ox or oy:
        sy, sx = max(0, -oy), max(0, -ox)  # drop rows/cols mapped off-frame
        dy, dx = max(0, oy), max(0, ox)  # pad to the in-frame origin
        src = arr[sy:, sx:]
        out = np.zeros((dy + src.shape[0], dx + src.shape[1]), dtype=bool)
        out[dy:, dx:] = src
        arr = out
    return arr


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

    Cost note: ``to_polygon()`` builds and unions one box per RLE run, so this is
    CPU-heavy for fragmented/speckled masks (thousands of runs). Callers exposing
    polygon output should pair it with ``mask_cleanup`` (keep-largest-CC) so each
    instance is a single clean component before polygonization.

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
        from shapely.geometry import MultiPolygon

        # Simplify per-component for a MultiPolygon (a fragmented/speckled mask):
        # ``geom.length`` on a MultiPolygon is the SUM of all component perimeters,
        # so a single ``epsilon * geom.length`` tolerance would over-simplify each
        # small part. Scale each component's tolerance to its OWN perimeter; the
        # single-Polygon (dominant) case is unchanged.
        if isinstance(geom, MultiPolygon):
            parts = [
                g.simplify(float(epsilon) * g.length, preserve_topology=True)
                for g in geom.geoms
            ]
            simplified = MultiPolygon([p for p in parts if not p.is_empty])
        else:
            simplified = geom.simplify(
                float(epsilon) * geom.length, preserve_topology=True
            )
        if not simplified.is_empty:
            geom = simplified
    return sio.PredictedROI(geometry=geom, score=float(score))
