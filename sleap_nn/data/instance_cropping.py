"""Handle cropping of instances."""

from typing import Iterator, Tuple, Dict, Optional
import math
import numpy as np
import sleap_io as sio
import torch
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.resizing import compute_eff_scale
from sleap_nn.data.skia_augmentation import crop_and_resize_skia as crop_and_resize
from sleap_nn.data.utils import embedding_detections


def compute_augmentation_padding(
    bbox_size: float,
    rotation_max: float = 0.0,
    scale_max: float = 1.0,
) -> int:
    """Compute padding needed to accommodate augmentation transforms.

    When rotation and scaling augmentations are applied, the bounding box of an
    instance can expand beyond its original size. This function calculates the
    padding needed to ensure the full instance remains visible after augmentation.

    Args:
        bbox_size: The size of the instance bounding box (max of width/height).
        rotation_max: Maximum absolute rotation angle in degrees. For symmetric
            rotation ranges like [-180, 180], pass 180.
        scale_max: Maximum scaling factor. For scale range [0.9, 1.1], pass 1.1.

    Returns:
        Padding in pixels to add around the bounding box (total, not per side).
    """
    if rotation_max == 0.0 and scale_max <= 1.0:
        return 0

    # For a square bbox rotated by angle θ, the new bbox has side length:
    # L' = L * (|cos(θ)| + |sin(θ)|)
    # Maximum expansion occurs at 45°: L' = L * sqrt(2)
    # For arbitrary angle: we use the worst case within the rotation range
    rotation_rad = math.radians(min(abs(rotation_max), 90))
    rotation_factor = abs(math.cos(rotation_rad)) + abs(math.sin(rotation_rad))

    # For angles > 45°, the factor increases, max at 45° = sqrt(2)
    # But for angles approaching 90°, it goes back to 1
    # Worst case in any range including 45° is sqrt(2)
    if abs(rotation_max) >= 45:
        rotation_factor = math.sqrt(2)

    # Combined expansion factor
    expansion_factor = rotation_factor * max(scale_max, 1.0)

    # Total padding needed (both sides)
    expanded_size = bbox_size * expansion_factor
    padding = expanded_size - bbox_size

    return int(math.ceil(padding))


def _frame_eff_scale(
    lf: sio.LabeledFrame,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]],
) -> float:
    """Return the size-matcher scale that will be applied to a labeled frame.

    Args:
        lf: The labeled frame, whose video supplies the native resolution.
        max_hw: The configured ``(max_height, max_width)``, or ``None`` to skip
            size matching entirely.

    Returns:
        The scale factor, ``1.0`` when size matching does not apply or the
        video's resolution cannot be determined.
    """
    if max_hw is None:
        return 1.0
    try:
        shape = lf.video.shape
    except Exception:
        # Reading `shape` opens the backend, which fails for a missing or
        # unreadable video file. Sizing should degrade, not crash.
        shape = None
    if shape is None or len(shape) < 3:
        # An unopened or headerless video gives us nothing to scale against.
        return 1.0
    return compute_eff_scale((int(shape[1]), int(shape[2])), max_hw)


def _iter_frame_instances(
    labels: sio.Labels, user_instances_only: bool, superseded_only: bool = False
) -> Iterator[Tuple[sio.LabeledFrame, np.ndarray]]:
    """Yield ``(labeled_frame, points)`` for every non-empty instance.

    Args:
        labels: A `sio.Labels` to walk.
        user_instances_only: When ``True``, skip `sio.PredictedInstance`s. This
            must match ``data_config.user_instances_only``, or the crop size is
            derived from instances the model is never trained on.
        superseded_only: With ``user_instances_only``, skip only the predictions a
            user instance on the same frame supersedes -- the ``embedding`` model's
            reading of the flag (:func:`sleap_nn.data.utils.embedding_detections`).

    Yields:
        The frame and that instance's ``(n_nodes, 2)`` point array, in the
        video's native pixel coordinates.
    """
    for lf in labels:
        if user_instances_only and superseded_only:
            instances = embedding_detections(lf, True)[0]
        elif user_instances_only:
            instances = [
                inst
                for inst in lf.instances
                if not isinstance(inst, sio.PredictedInstance)
            ]
        else:
            instances = lf.instances
        for inst in instances:
            if inst.is_empty:  # every point NaN
                continue
            yield lf, inst.numpy()


def find_max_instance_bbox_size(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    user_instances_only: bool = False,
    superseded_only: bool = False,
) -> float:
    """Find the maximum bounding box dimension across all instances in labels.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        max_hw: The configured ``(max_height, max_width)``. When given, each
            instance is measured in the size-matched space it will actually be
            cropped in, rather than in its video's native pixels. ``None``
            (default) measures native pixels, the historical behavior.
        user_instances_only: When ``True``, ignore predicted instances.
        superseded_only: See :func:`_iter_frame_instances`.

    Returns:
        The maximum bounding box dimension (max of width or height) across all instances.
    """
    max_length = 0.0
    for lf, pts in _iter_frame_instances(labels, user_instances_only, superseded_only):
        eff_scale = _frame_eff_scale(lf, max_hw)
        diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
        diff_x = 0 if np.isnan(diff_x) else diff_x * eff_scale
        max_length = np.maximum(max_length, diff_x)
        diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
        diff_y = 0 if np.isnan(diff_y) else diff_y * eff_scale
        max_length = np.maximum(max_length, diff_y)
    return float(max_length)


def iter_required_crop_sizes(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    anchor_ind: Optional[int] = None,
    centroid_method: Optional[str] = None,
    centroid_fallback: Optional[str] = None,
    user_instances_only: bool = False,
    superseded_only: bool = False,
) -> Iterator[float]:
    """Yield the crop size each labeled instance needs to avoid being clipped.

    Crops are **centered on the instance's centroid**, not on its bounding-box
    midpoint (see `generate_crops` / `make_centered_bboxes`), so the size an
    instance requires is twice its greatest node offset from that centroid --
    *not* its bounding-box extent. The two agree only when the centroid happens
    to sit at the middle of the bounding box. For an anchor node near one end of
    the animal (a mouse anchored on the thorax, with a long tail) the required
    size approaches twice the extent, which is why a crop sized to the bounding
    box still clips the far nodes.

    The centroid is derived through `generate_centroids`, the same op that
    positions the crop at training time, so this can't disagree with the actual
    crop center.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        max_hw: The configured ``(max_height, max_width)``. When given, points
            are measured in the size-matched space the crop is taken in.
        anchor_ind: Index of the anchor node, or ``None`` for a centroid reduce
            method. Passed through to `generate_centroids`.
        centroid_method: The resolved centroid method, or ``None`` to infer from
            ``anchor_ind``. See `resolve_centroid_method`.
        centroid_fallback: The reduce method used when the anchor node is not
            visible on an instance.
        user_instances_only: When ``True``, ignore predicted instances.
        superseded_only: See :func:`_iter_frame_instances`.

    Yields:
        The required crop side length, in size-matched pixels, per instance.
        Instances whose centroid is undefined are skipped.
    """
    for lf, pts in _iter_frame_instances(labels, user_instances_only, superseded_only):
        eff_scale = _frame_eff_scale(lf, max_hw)
        scaled = torch.from_numpy(pts.astype("float32")) * eff_scale
        centroid = generate_centroids(
            scaled,
            anchor_ind=anchor_ind,
            method=centroid_method,
            fallback=centroid_fallback,
        )
        if torch.isnan(centroid).any():
            continue
        # Largest per-axis offset from the crop center; the crop spans
        # centroid +/- size/2 on each axis, so it must be twice this to reach it.
        offsets = (scaled - centroid).abs()
        reach = torch.nan_to_num(offsets, nan=0.0).max()
        yield float(2.0 * reach)


def count_clipped_instances(
    labels: sio.Labels,
    crop_size: int,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    anchor_ind: Optional[int] = None,
    centroid_method: Optional[str] = None,
    centroid_fallback: Optional[str] = None,
    user_instances_only: bool = False,
    superseded_only: bool = False,
) -> Tuple[int, int, float]:
    """Count labeled instances that a given crop size would clip.

    Used to warn about an explicitly configured ``crop_size``, which we must not
    silently override -- some users knowingly accept clipping an extremity.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        crop_size: The configured crop size, in size-matched pixels.
        max_hw: The configured ``(max_height, max_width)``.
        anchor_ind: Index of the anchor node, or ``None``.
        centroid_method: The resolved centroid method, or ``None`` to infer.
        centroid_fallback: The reduce method for a non-visible anchor node.
        user_instances_only: When ``True``, ignore predicted instances.
        superseded_only: See :func:`_iter_frame_instances`.

    Returns:
        ``(n_clipped, n_total, max_required)`` -- how many instances have at
        least one node outside the crop, how many were examined, and the crop
        size that would contain all of them (``0.0`` if there are none).
    """
    n_clipped = 0
    n_total = 0
    max_required = 0.0
    for required in iter_required_crop_sizes(
        labels,
        max_hw=max_hw,
        anchor_ind=anchor_ind,
        centroid_method=centroid_method,
        centroid_fallback=centroid_fallback,
        user_instances_only=user_instances_only,
        superseded_only=superseded_only,
    ):
        n_total += 1
        if required > crop_size:
            n_clipped += 1
        max_required = max(max_required, required)
    return n_clipped, n_total, max_required


_NO_RECTS = np.zeros((0, 4), dtype=np.int64)


def mask_foreground_rects(mask: "sio.SegmentationMask") -> np.ndarray:
    """A mask's foreground on the image-pixel grid, as disjoint rectangles.

    The union of the returned rectangles is exactly the foreground of
    :func:`sleap_nn.inference.segmentation_convert.decode_mask_to_image_res`: the
    stored grid, resampled nearest-neighbor to its image extent when ``scale`` is
    not 1, placed at the rounded ``offset``, with pixels an offset puts above or left
    of the image dropped. It is read off the run-length encoding, so the cost grows
    with the number of runs, not with the frame; no full-grid decode is made.

    Args:
        mask: A ``sio.SegmentationMask``.

    Returns:
        An ``(n, 4)`` int64 array of half-open ``[y0, y1, x0, x1)`` rectangles in
        image pixels, empty when the mask has no foreground pixel on the image.
    """
    height, width = int(mask.height), int(mask.width)
    counts = np.asarray(mask.rle_counts, dtype=np.int64)
    total = height * width
    if total == 0 or counts.size < 2:
        return _NO_RECTS
    ends = np.cumsum(counts)
    # Odd runs are foreground (the encoding starts with a background run); the
    # decoder truncates the runs at the grid size.
    run_start = np.minimum((ends - counts)[1::2], total)
    run_end = np.minimum(ends[1::2], total)
    keep = run_end > run_start
    run_start, run_end = run_start[keep], run_end[keep]
    if run_start.size == 0:
        return _NO_RECTS

    # Split each run (row-major) into one segment per row it covers.
    first_row = run_start // width
    n_rows = (run_end - 1) // width - first_row + 1
    run = np.repeat(np.arange(run_start.size), n_rows)
    rows = first_row[run] + (
        np.arange(run.size) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
    )
    col0 = np.maximum(run_start[run], rows * width) - rows * width
    col1 = np.minimum(run_end[run], (rows + 1) * width) - rows * width

    scale = tuple(getattr(mask, "scale", (1.0, 1.0)))
    offset = tuple(getattr(mask, "offset", (0.0, 0.0)))
    if scale == (1.0, 1.0):
        y0, y1, x0, x1 = rows, rows + 1, col0, col1
    else:
        # `SegmentationMask.resampled`'s nearest-neighbor maps (image row i reads
        # stored row row_idx[i]). They never decrease, so the image rows reading a
        # stored row, and the image columns reading a stored column range, are
        # contiguous.
        img_h, img_w = (int(v) for v in mask.image_extent)
        row_idx = (np.arange(img_h) * height / img_h).astype(int).clip(0, height - 1)
        col_idx = (np.arange(img_w) * width / img_w).astype(int).clip(0, width - 1)
        y0 = np.searchsorted(row_idx, rows, side="left")
        y1 = np.searchsorted(row_idx, rows, side="right")
        x0 = np.searchsorted(col_idx, col0, side="left")
        x1 = np.searchsorted(col_idx, col1, side="left")
    # The offset is baked in the way `decode_mask_to_image_res` bakes it.
    ox, oy = int(round(offset[0])), int(round(offset[1]))
    rects = np.stack(
        [
            np.maximum(y0 + oy, 0),
            np.maximum(y1 + oy, 0),
            np.maximum(x0 + ox, 0),
            np.maximum(x1 + ox, 0),
        ],
        axis=1,
    ).astype(np.int64)
    return rects[(rects[:, 1] > rects[:, 0]) & (rects[:, 3] > rects[:, 2])]


def clip_rects(rects: np.ndarray, frame_hw: Tuple[int, int]) -> np.ndarray:
    """Clip ``[y0, y1, x0, x1)`` rectangles to a ``(height, width)`` frame."""
    clipped = rects.copy()
    clipped[:, 0:2] = np.clip(clipped[:, 0:2], 0, int(frame_hw[0]))
    clipped[:, 2:4] = np.clip(clipped[:, 2:4], 0, int(frame_hw[1]))
    return clipped[(clipped[:, 1] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 2])]


def mask_rects_center(
    rects: np.ndarray,
    crop_centering: str = "auto",
    frame_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """The ``(cx, cy)`` a mask crop centers on, from the mask's rectangles.

    ``auto`` / ``mask_com``: the center of mass; ``bbox``: the bounding-box
    midpoint. Both are the values ``_compute_mask_centroids`` /
    ``_mask_bbox_midpoint`` compute on the decoded mask, to the last bit: the
    coordinate sums are exact integers, and the mean is their one rounded quotient,
    as ``np.mean`` of the pixel coordinates is.

    Args:
        rects: :func:`mask_foreground_rects` output (clipped to the frame the crop
            is taken from, see :func:`clip_rects`).
        crop_centering: ``auto`` | ``mask_com`` | ``bbox``.
        frame_hw: The frame ``(height, width)``, for an empty mask, which centers on
            the frame like the decoded-mask helpers do. Required when ``rects`` can
            be empty.
    """
    if rects.shape[0] == 0:
        if frame_hw is None:
            raise ValueError("An empty mask has no center without the frame size.")
        return float(frame_hw[1]) / 2.0, float(frame_hw[0]) / 2.0
    y0, y1, x0, x1 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    if crop_centering == "bbox":
        return (
            (float(x0.min()) + float(x1.max() - 1)) / 2.0,
            (float(y0.min()) + float(y1.max() - 1)) / 2.0,
        )
    n, sum_x, sum_y = mask_rects_moments(rects)
    return float(sum_x) / float(n), float(sum_y) / float(n)


def mask_rects_moments(rects: np.ndarray) -> Tuple[int, int, int]:
    """``(n, sum_x, sum_y)``: a mask's pixel count and coordinate sums, exactly.

    Python integers, from the :func:`mask_foreground_rects` rectangles: no pixel is
    enumerated.
    """
    y0, y1, x0, x1 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    heights, widths = y1 - y0, x1 - x0
    n = int((heights * widths).sum())
    # Sum of the integers a..b-1 is (a + b - 1) * (b - a) / 2, always whole.
    sum_x = int((heights * ((x0 + x1 - 1) * widths // 2)).sum())
    sum_y = int((widths * ((y0 + y1 - 1) * heights // 2)).sum())
    return n, sum_x, sum_y


def rasterize_rects(
    rects: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """The ``[y0:y1, x0:x1]`` window of the mask ``rects`` cover, as a bool array."""
    height, width = max(y1 - y0, 0), max(x1 - x0, 0)
    if height == 0 or width == 0 or rects.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)
    local = rects - np.array([y0, y0, x0, x0], dtype=np.int64)
    local[:, 0:2] = np.clip(local[:, 0:2], 0, height)
    local[:, 2:4] = np.clip(local[:, 2:4], 0, width)
    local = local[(local[:, 1] > local[:, 0]) & (local[:, 3] > local[:, 2])]
    # One segment per row a rectangle covers, then a per-row difference array: +1
    # where a segment starts, -1 where it ends, a prefix sum along the row fills it.
    n_rows = local[:, 1] - local[:, 0]
    rect = np.repeat(np.arange(local.shape[0]), n_rows)
    rows = local[rect, 0] + (
        np.arange(rect.size) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
    )
    diff = np.zeros((height, width + 1), dtype=np.int32)
    np.add.at(diff, (rows, local[rect, 2]), 1)
    np.add.at(diff, (rows, local[rect, 3]), -1)
    return np.cumsum(diff, axis=1, dtype=np.int32)[:, :width] > 0


def iter_mask_extents(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    crop_centering: str = "auto",
    user_instances_only: bool = False,
    min_mask_area: float = 1.0,
) -> Iterator[Tuple[float, float]]:
    """Yield ``(required_crop_size, bbox_size)`` for every mask, in size-matched pixels.

    The mask analog of :func:`iter_required_crop_sizes`, for the ``embedding`` model's
    mask mode, whose crop is centered on the mask's center of mass (``crop_centering``
    ``auto`` / ``mask_com``) or on its bounding-box midpoint (``bbox``), as
    ``EmbeddingDataset._crop_mask`` centers it. ``required_crop_size`` is the side a
    crop around that center needs to hold the whole mask (twice the largest distance
    from the center to a mask pixel's far edge); ``bbox_size`` is the larger side of
    the mask's bounding box.

    Args:
        labels: A `sio.Labels` whose frames carry masks (``lf.masks``).
        max_hw: The configured ``(max_height, max_width)``; masks are measured in the
            space the size matcher puts the frame in.
        crop_centering: ``auto`` | ``mask_com`` | ``bbox``.
        user_instances_only: When ``True``, skip the predicted masks a user mask on
            the same frame supersedes, as the embedding dataset does
            (:func:`sleap_nn.data.utils.embedding_detections`).
        min_mask_area: Skip masks with fewer foreground pixels (image pixels) than
            this, as the embedding dataset does.
    """
    for lf in labels:
        eff_scale = None
        for mask in embedding_detections(lf, user_instances_only)[1]:
            sx, sy = tuple(getattr(mask, "scale", (1.0, 1.0)) or (1.0, 1.0))
            if mask.area / max(sx * sy, 1e-12) < max(min_mask_area, 1.0):
                continue
            # From the run-length encoding: no full-frame decode per mask.
            rects = mask_foreground_rects(mask)
            if rects.shape[0] == 0:
                continue
            y0, y1 = int(rects[:, 0].min()), int(rects[:, 1].max()) - 1
            x0, x1 = int(rects[:, 2].min()), int(rects[:, 3].max()) - 1
            if crop_centering == "bbox":
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            else:
                # Center of mass, as offsets from the box corner (the arithmetic
                # these extents were always computed with, so crop sizing does not
                # move by a rounding step).
                n, sum_x, sum_y = mask_rects_moments(rects)
                cx = x0 + float(sum_x - n * x0) / float(n)
                cy = y0 + float(sum_y - n * y0) / float(n)
            if eff_scale is None:
                eff_scale = _frame_eff_scale(lf, max_hw)
            # Pixel centers sit at integer coordinates; +1 reaches their far edges.
            reach = max(cx - x0, x1 - cx, cy - y0, y1 - cy)
            required = (2.0 * reach + 1.0) * eff_scale
            bbox = (max(x1 - x0, y1 - y0) + 1.0) * eff_scale
            yield float(required), float(bbox)


def find_mask_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    min_crop_size: Optional[int] = None,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    crop_centering: str = "auto",
    user_instances_only: bool = False,
    min_mask_area: float = 1.0,
) -> int:
    """Compute a crop size that contains every mask (the mask analog of `find_instance_crop_size`).

    Args:
        labels: A `sio.Labels` whose frames carry masks.
        padding: Pixels of margin to add.
        maximum_stride: The returned size is divisible by this.
        min_crop_size: A floor for the size, before padding.
        max_hw: The configured ``(max_height, max_width)``.
        crop_centering: Where the crop is centered on each mask (see
            :func:`iter_mask_extents`).
        user_instances_only: When ``True``, ignore predicted masks.
        min_mask_area: Ignore masks smaller than this (image pixels).

    Returns:
        The crop side length, a multiple of ``maximum_stride``.
    """
    min_crop_size = 0 if min_crop_size is None else min_crop_size
    max_length = float(min_crop_size - padding)
    for required, _ in iter_mask_extents(
        labels,
        max_hw=max_hw,
        crop_centering=crop_centering,
        user_instances_only=user_instances_only,
        min_mask_area=min_mask_area,
    ):
        max_length = max(max_length, required)
    max_length = max(max_length, 0.0) + float(padding)
    return int(math.ceil(max_length / float(maximum_stride)) * maximum_stride)


def find_instance_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    min_crop_size: Optional[int] = None,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    anchor_ind: Optional[int] = None,
    centroid_method: Optional[str] = None,
    centroid_fallback: Optional[str] = None,
    user_instances_only: bool = False,
    superseded_only: bool = False,
) -> int:
    """Compute a crop size that contains every labeled instance.

    The size is measured the way the crop is actually taken: centered on each
    instance's centroid (`iter_required_crop_sizes`) and, when ``max_hw`` is
    given, in the size-matched pixel space that cropping happens in. Both
    matter -- a bounding-box measurement in native pixels under-sizes the crop
    whenever the centroid is off-center or the video is rescaled to ``max_hw``.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin padding.
        maximum_stride: Ensure that the returned crop size is divisible by this value.
            Useful for ensuring that the crop size will not be truncated in a given
            architecture.
        min_crop_size: A floor for the returned crop size, before padding.
        max_hw: The configured ``(max_height, max_width)``, so instances are
            measured in the space the size matcher puts them in. ``None``
            (default) measures native pixels.
        anchor_ind: Index of the anchor node the crop is centered on, or
            ``None`` for a centroid reduce method.
        centroid_method: The resolved centroid method, or ``None`` to infer from
            ``anchor_ind``.
        centroid_fallback: The reduce method used when the anchor node is not
            visible on an instance.
        user_instances_only: When ``True``, ignore predicted instances -- pass
            ``data_config.user_instances_only`` so the crop is not sized from
            instances that are excluded from training.
        superseded_only: See :func:`_iter_frame_instances`.

    Returns:
        An integer crop size denoting the length of the side of the boxes that
        will contain the instances when cropped. The returned crop size will be
        larger or equal to the input `min_crop_size`.

        This accounts for stride and padding when ensuring divisibility.
    """
    min_crop_size = 0 if min_crop_size is None else min_crop_size

    # `min_crop_size` is a floor, applied before padding is added.
    max_length = float(min_crop_size - padding)
    for required in iter_required_crop_sizes(
        labels,
        max_hw=max_hw,
        anchor_ind=anchor_ind,
        centroid_method=centroid_method,
        centroid_fallback=centroid_fallback,
        user_instances_only=user_instances_only,
        superseded_only=superseded_only,
    ):
        max_length = max(max_length, required)

    max_length = max(max_length, 0.0) + float(padding)
    crop_size = math.ceil(max_length / float(maximum_stride)) * maximum_stride

    return int(crop_size)


def make_centered_bboxes(
    centroids: torch.Tensor, box_height: int, box_width: int
) -> torch.Tensor:
    """Create centered bounding boxes around centroid.

    To be used with `kornia.geometry.transform.crop_and_resize`in the following
    (clockwise) order: top-left, top-right, bottom-right and bottom-left.

    Args:
        centroids: A tensor of centroids with shape (n_centroids, 2), where n_centroids is the
            number of centroids, and the last dimension represents x and y coordinates.
        box_height: The desired height of the bounding boxes.
        box_width: The desired width of the bounding boxes.

    Returns:
        torch.Tensor: A tensor containing bounding box coordinates for each centroid.
            The output tensor has shape (n_centroids, 4, 2), where n_centroids is the number
            of centroids, and the second dimension represents the four corner points of
            the bounding boxes, each with x and y coordinates. The order of the corners
            follows a clockwise arrangement: top-left, top-right, bottom-right, and
            bottom-left.
    """
    half_h = box_height / 2
    half_w = box_width / 2

    # Get x and y values from the centroids tensor.
    x = centroids[..., 0]
    y = centroids[..., 1]

    # Calculate the corner points.
    top_left = torch.stack([x - half_w, y - half_h], dim=-1)
    top_right = torch.stack([x + half_w, y - half_h], dim=-1)
    bottom_left = torch.stack([x - half_w, y + half_h], dim=-1)
    bottom_right = torch.stack([x + half_w, y + half_h], dim=-1)

    # Get bounding box.
    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=-2)

    offset = torch.tensor([[+0.5, +0.5], [-0.5, +0.5], [-0.5, -0.5], [+0.5, -0.5]]).to(
        corners.device
    )

    return corners + offset


def generate_crops(
    image: torch.Tensor,
    instance: torch.Tensor,
    centroid: torch.Tensor,
    crop_size: Tuple[int],
) -> Dict[str, torch.Tensor]:
    """Generate cropped image for the given centroid.

    Args:
        image: Input source image. (n_samples, C, H, W)
        instance: Keypoints for the instance to be cropped. (n_nodes, 2)
        centroid: Centroid of the instance to be cropped. (2)
        crop_size: (height, width) of the crop to be generated.

    Returns:
        A dictionary with cropped images, bounding box for the cropped instance, keypoints and
        centroids adjusted to the crop.
    """
    box_size = crop_size

    # Generate bounding boxes from centroid.
    instance_bbox = torch.unsqueeze(
        make_centered_bboxes(centroid, box_size[0], box_size[1]), 0
    )  # (n_samples=1, 4, 2)

    # Generate cropped image of shape (n_samples, C, crop_H, crop_W)
    instance_image = crop_and_resize(
        image,
        boxes=instance_bbox,
        size=box_size,
    )

    # Access top left point (x,y) of bounding box and subtract this offset from
    # position of nodes.
    point = instance_bbox[0][0]
    center_instance = (instance - point).unsqueeze(0)  # (n_samples=1, n_nodes, 2)
    centered_centroid = (centroid - point).unsqueeze(0)  # (n_samples=1, 2)

    cropped_sample = {
        "instance_image": instance_image,
        "instance_bbox": instance_bbox,
        "instance": center_instance,
        "centroid": centered_centroid,
    }

    return cropped_sample
