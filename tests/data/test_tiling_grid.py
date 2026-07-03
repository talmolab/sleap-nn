"""Tests for the tile-grid primitive in `sleap_nn.data.tiling`."""

import pytest

from sleap_nn.data.tiling import generate_tile_grid

# --- Invariant helpers -------------------------------------------------------


def _axis_values(origins, axis):
    """Return sorted unique origin coordinates for one axis (0=y, 1=x)."""
    return sorted({o[axis] for o in origins})


def _stride_boundary(dim, tile_size, output_stride):
    """Largest pixel index (exclusive) guaranteed covered on the grid.

    When ``dim <= tile_size`` the single tile (padded later) covers the axis.
    Otherwise the last origin is snapped *down* to the ``output_stride`` grid,
    so coverage reaches ``dim`` exactly when ``(dim - tile_size)`` is a multiple
    of ``output_stride``, and is short by the remainder otherwise.
    """
    if dim <= tile_size:
        return dim
    return dim - ((dim - tile_size) % output_stride)


def assert_common_invariants(origins, image_hw, tile_size, output_stride):
    """Assert the structural invariants that must hold for every grid."""
    height, width = image_hw

    # At least one origin is always returned.
    assert len(origins) >= 1

    # Every returned coordinate is a multiple of output_stride.
    for y0, x0 in origins:
        assert y0 % output_stride == 0
        assert x0 % output_stride == 0

    for axis, dim in ((0, height), (1, width)):
        vals = _axis_values(origins, axis)

        if dim <= tile_size:
            # The only origin on a too-small axis is 0.
            assert vals == [0]
            continue

        # Never overrun the frame.
        assert vals[0] >= 0
        assert vals[-1] <= dim - tile_size

        # Strictly increasing and deduped.
        assert all(b > a for a, b in zip(vals, vals[1:]))

        # Consecutive interior origins differ by a constant positive step that
        # is a multiple of output_stride; the final (inward-snapped) gap is no
        # larger than that step.
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        for d in diffs:
            assert d > 0
            assert d % output_stride == 0
        if len(diffs) >= 2:
            step = diffs[0]
            assert all(dd == step for dd in diffs[:-1])
            assert diffs[-1] <= step

    # Row-major (y then x) ordering.
    assert origins == sorted(origins)

    # Exactly the Cartesian product of the per-axis origins.
    ys = _axis_values(origins, 0)
    xs = _axis_values(origins, 1)
    assert origins == [(y, x) for y in ys for x in xs]


def _axis_coverage(vals, tile_size, dim):
    """Return a boolean mask of covered pixels in ``[0, dim)`` for one axis."""
    covered = [False] * dim
    for o in vals:
        for p in range(o, min(o + tile_size, dim)):
            covered[p] = True
    return covered


def assert_full_coverage(origins, image_hw, tile_size):
    """Assert every pixel in the frame is inside at least one tile."""
    height, width = image_hw
    assert all(_axis_coverage(_axis_values(origins, 0), tile_size, height))
    assert all(_axis_coverage(_axis_values(origins, 1), tile_size, width))


# --- Parametrized invariants -------------------------------------------------

# Each case has ``(image_dim - tile_size)`` divisible by ``output_stride`` on
# both axes (or an axis <= tile_size), so full-frame coverage is guaranteed.
_ALIGNED_CASES = [
    ((256, 256), 128, 32, 4, 1, 0.25),  # base square case
    ((200, 360), 128, 32, 4, 1, 0.25),  # non-square
    ((300, 300), 128, 40, 2, 16, 0.25),  # output_stride=2, max_stride=16
    ((224, 224), 128, 32, 4, 1, 0.25),  # exact fit of last origin
    ((300, 300), 128, 32, 4, 1, 0.25),  # last gap smaller than step
    ((512, 512), 128, 0, 8, 1, 0.25),  # overlap below the min-overlap floor
    ((128, 256), 64, 16, 4, 1, 0.25),  # smaller tile
    ((400, 100), 128, 32, 4, 1, 0.25),  # one axis smaller than the tile
    ((256, 256), 128, 120, 4, 16, 0.0),  # else-branch: step < max_stride
    ((256, 256), 128, 32, 4, 6, 0.25),  # else-branch: max_stride % os != 0
]


@pytest.mark.parametrize(
    "image_hw,tile_size,overlap,output_stride,max_stride,min_overlap_fraction",
    _ALIGNED_CASES,
)
def test_grid_invariants(
    image_hw, tile_size, overlap, output_stride, max_stride, min_overlap_fraction
):
    """All structural invariants plus full coverage hold on aligned frames."""
    origins = generate_tile_grid(
        image_hw,
        tile_size=tile_size,
        overlap=overlap,
        output_stride=output_stride,
        max_stride=max_stride,
        min_overlap_fraction=min_overlap_fraction,
    )
    assert_common_invariants(origins, image_hw, tile_size, output_stride)
    assert_full_coverage(origins, image_hw, tile_size)


# --- Concrete cases ----------------------------------------------------------


def test_concrete_square_256():
    """A 256x256 frame with 128px tiles yields the expected 3x3 grid."""
    origins = generate_tile_grid((256, 256), tile_size=128, overlap=32, output_stride=4)
    expected_axis = [0, 96, 128]
    assert _axis_values(origins, 0) == expected_axis
    assert _axis_values(origins, 1) == expected_axis
    assert origins == [(y, x) for y in expected_axis for x in expected_axis]

    # Every origin is a multiple of 4 and never overruns the frame.
    for y0, x0 in origins:
        assert y0 % 4 == 0 and x0 % 4 == 0
        assert 0 <= y0 <= 256 - 128
        assert 0 <= x0 <= 256 - 128

    assert_common_invariants(origins, (256, 256), 128, 4)
    assert_full_coverage(origins, (256, 256), 128)


def test_frame_smaller_than_tile():
    """A frame smaller than the tile yields a single origin at (0, 0)."""
    origins = generate_tile_grid((64, 64), tile_size=128, overlap=32, output_stride=4)
    assert origins == [(0, 0)]


def test_one_axis_smaller_than_tile():
    """Only the too-small axis collapses to a single origin."""
    origins = generate_tile_grid((64, 300), tile_size=128, overlap=32, output_stride=4)
    assert _axis_values(origins, 0) == [0]
    assert _axis_values(origins, 1) != [0]
    assert_common_invariants(origins, (64, 300), 128, 4)
    assert_full_coverage(origins, (64, 300), 128)


def test_non_square_frame():
    """A non-square frame tiles each axis independently."""
    image_hw = (200, 360)
    origins = generate_tile_grid(image_hw, tile_size=128, overlap=32, output_stride=4)
    assert _axis_values(origins, 0) == [0, 72]
    assert _axis_values(origins, 1) == [0, 96, 192, 232]
    assert_common_invariants(origins, image_hw, 128, 4)
    assert_full_coverage(origins, image_hw, 128)


def test_exact_fit_last_origin():
    """An exact-fit frame appends a distinct inward-snapped last origin."""
    # 224 = 128 (tile) + 96 (step); (224 - 128) % 4 == 0, so no dedup needed.
    origins = generate_tile_grid((224, 224), tile_size=128, overlap=32, output_stride=4)
    assert _axis_values(origins, 0) == [0, 96]
    assert _axis_values(origins, 1) == [0, 96]
    assert_common_invariants(origins, (224, 224), 128, 4)
    assert_full_coverage(origins, (224, 224), 128)


def test_last_origin_dedup():
    """An off-by-a-bit frame dedups the inward-snapped last origin.

    With ``image_dim = 225`` the interior walk already lands on ``96`` and the
    inward-snapped last origin ``(225 - 128) // 4 * 4 == 96`` repeats it, so it
    is dropped. Coverage reaches the last ``output_stride``-aligned boundary
    (224); the final pixel (224) cannot be covered without overrunning the frame
    or leaving the output_stride grid.
    """
    image_hw = (225, 225)
    origins = generate_tile_grid(image_hw, tile_size=128, overlap=32, output_stride=4)
    # Dedup: two origins per axis, not three.
    assert _axis_values(origins, 0) == [0, 96]
    assert _axis_values(origins, 1) == [0, 96]
    assert len(origins) == 4

    assert_common_invariants(origins, image_hw, 128, 4)

    # Coverage holds up to the aligned boundary (224), one pixel short of 225.
    for axis, dim in ((0, 225), (1, 225)):
        vals = _axis_values(origins, axis)
        covered = _axis_coverage(vals, 128, dim)
        boundary = _stride_boundary(dim, 128, 4)
        assert boundary == 224
        assert all(covered[:boundary])
        assert not covered[224]


def test_origins_snap_to_output_stride_not_max_stride():
    """Origins snap to output_stride even when max_stride is larger."""
    image_hw = (300, 300)
    output_stride, max_stride = 2, 16
    origins = generate_tile_grid(
        image_hw,
        tile_size=128,
        overlap=40,
        output_stride=output_stride,
        max_stride=max_stride,
    )
    vals = _axis_values(origins, 0)
    assert vals == [0, 80, 160, 172]

    # The inward-snapped last origin is a multiple of output_stride (2) but not
    # of the larger max_stride (16), proving it snaps to output_stride.
    assert 172 in vals
    assert 172 % output_stride == 0
    assert 172 % max_stride != 0
    for y0, x0 in origins:
        assert y0 % output_stride == 0
        assert x0 % output_stride == 0

    assert_common_invariants(origins, image_hw, 128, output_stride)
    assert_full_coverage(origins, image_hw, 128)


def test_overlap_floor_is_enforced():
    """A too-small overlap is raised to the min-overlap fraction floor."""
    # overlap=0 but round(0.25 * 128) == 32, so the effective step is 96, same
    # as passing overlap=32 explicitly.
    origins_zero = generate_tile_grid(
        (512, 512), tile_size=128, overlap=0, output_stride=8
    )
    origins_floor = generate_tile_grid(
        (512, 512), tile_size=128, overlap=32, output_stride=8
    )
    assert origins_zero == origins_floor
    assert _axis_values(origins_zero, 0) == [0, 96, 192, 288, 384]
