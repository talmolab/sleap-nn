"""Shared tiling primitives for splitting large images into overlapping tiles.

This module holds the low-level building blocks used to tile large frames into
fixed-size square tiles for training and inference. For now it only provides
:func:`generate_tile_grid`, which computes the grid of snapped tile origins in
input-pixel space. It will be extended in a later change with the tile samplers
and extractors that consume this grid.
"""

from typing import List, Tuple


def _axis_tile_origins(
    image_dim: int,
    tile_size: int,
    overlap: int,
    output_stride: int,
    max_stride: int,
    min_overlap_fraction: float,
) -> List[int]:
    """Compute snapped tile origins along a single axis.

    This helper operates on a single axis (either the height or the width) and
    returns the sorted, de-duplicated list of tile origins in input-pixel space.
    See :func:`generate_tile_grid` for the full description of the arguments and
    the snapping conventions.

    Args:
        image_dim: Size of the axis in pixels.
        tile_size: Side length of the (square) tile in pixels.
        overlap: Requested overlap between neighboring tiles in pixels.
        output_stride: Origins are snapped down to multiples of this value.
        max_stride: Coarse stride used to snap the step between tiles when it is
            compatible with ``output_stride``.
        min_overlap_fraction: Minimum overlap expressed as a fraction of
            ``tile_size``. If ``overlap`` is below this floor, the step is
            shrunk so the effective overlap meets the floor.

    Returns:
        A sorted list of unique tile origins (the top edge for the height axis
        or the left edge for the width axis). Each origin is a multiple of
        ``output_stride`` and lies within ``[0, image_dim - tile_size]`` when
        ``image_dim > tile_size``. When ``image_dim <= tile_size`` the single
        origin ``0`` is returned (the tile is zero-padded to size later).
    """
    # Frames no larger than a tile yield a single origin; the extractor pads.
    if image_dim <= tile_size:
        return [0]

    # Enforce a minimum overlap floor by (possibly) shrinking the step.
    eff_overlap = max(overlap, round(min_overlap_fraction * tile_size))
    step = tile_size - eff_overlap

    # Snap the step to a coarse multiple: prefer ``max_stride`` when it is large
    # enough and compatible with ``output_stride``, otherwise fall back to
    # ``output_stride`` so tiles land on network-friendly boundaries.
    if step >= max_stride and max_stride % output_stride == 0:
        snap_unit = max_stride
    else:
        snap_unit = output_stride
    step = (step // snap_unit) * snap_unit
    if step < output_stride:
        step = output_stride

    # Walk interior origins, snapping each down to a multiple of output_stride,
    # while the full tile still fits strictly inside the frame.
    origins: List[int] = []
    origin = 0
    while origin + tile_size < image_dim:
        origins.append((origin // output_stride) * output_stride)
        origin += step

    # Append the inward-snapped final origin so the far edge is fully covered
    # without overrunning the frame, de-duplicating if it repeats the last one.
    last_origin = ((image_dim - tile_size) // output_stride) * output_stride
    if not origins or origins[-1] != last_origin:
        origins.append(last_origin)

    return origins


def generate_tile_grid(
    image_hw: Tuple[int, int],
    tile_size: int,
    overlap: int,
    output_stride: int,
    max_stride: int = 1,
    min_overlap_fraction: float = 0.25,
) -> List[Tuple[int, int]]:
    """Compute snapped square-tile top-left origins covering an image.

    Origins are computed independently for the height and width axes and then
    combined via a Cartesian product in row-major (``y`` then ``x``) order. Each
    origin is snapped to a multiple of ``output_stride`` and constrained so that
    tiles never overrun the frame while still covering the far edges.

    Args:
        image_hw: Image size as a ``(height, width)`` tuple in pixels.
        tile_size: Side length of the (square) tile in pixels.
        overlap: Requested overlap between neighboring tiles in pixels. This is
            raised to ``round(min_overlap_fraction * tile_size)`` if it falls
            below that floor.
        output_stride: Tile origins are snapped down to multiples of this value.
            This is typically the output stride of the model so that tiles align
            to the network's prediction grid.
        max_stride: Coarse stride used to snap the step between neighboring
            tiles when it is at least as large as ``max_stride`` and evenly
            divisible into a multiple of ``output_stride``. Defaults to ``1``.
        min_overlap_fraction: Minimum overlap expressed as a fraction of
            ``tile_size``. Defaults to ``0.25``.

    Returns:
        A list of ``(y0, x0)`` tile origins (top-left corners) in input-pixel
        space, ordered row-major (all tiles for the first row of origins, then
        the next, and so on). At least one origin is always returned.

    Notes:
        The union of the resulting tiles covers the frame up to the last origin
        that can be placed on the ``output_stride`` grid without overrunning.
        When ``(image_dim - tile_size)`` is a multiple of ``output_stride`` (the
        typical case) this is the entire frame including the right/bottom edge.
    """
    y_origins = _axis_tile_origins(
        image_dim=image_hw[0],
        tile_size=tile_size,
        overlap=overlap,
        output_stride=output_stride,
        max_stride=max_stride,
        min_overlap_fraction=min_overlap_fraction,
    )
    x_origins = _axis_tile_origins(
        image_dim=image_hw[1],
        tile_size=tile_size,
        overlap=overlap,
        output_stride=output_stride,
        max_stride=max_stride,
        min_overlap_fraction=min_overlap_fraction,
    )
    return [(y0, x0) for y0 in y_origins for x0 in x_origins]
