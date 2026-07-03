"""Tests for the torch-native tiled-inference importance windows + TileMerger."""

import torch

from sleap_nn.inference.tile_merger import build_importance_window, TileMerger


def test_gaussian_window_basic():
    """Gaussian window: shape, center peak, monotone decrease, floor, symmetry."""
    w = build_importance_window((64, 64), mode="gaussian")

    # Shape.
    assert w.shape == (64, 64)
    assert w.dtype == torch.float32

    # Peak ~1.0 at the center (even dims -> the two center rows/cols are ~1.0).
    center = w[31:33, 31:33]
    assert torch.allclose(center.max(), torch.tensor(1.0), atol=1e-2)
    # The global max sits at the center.
    assert torch.allclose(w.max(), center.max())

    # Floor: every covered weight is >= 1e-3 (minus fp slack).
    assert w.min().item() >= 1e-3 - 1e-6

    # Monotone (non-increasing) decrease from the center outward, both axes.
    # Take a center row/column and walk from the center toward each edge.
    row = w[32]
    col = w[:, 32]
    right = row[32:]
    down = col[32:]
    assert torch.all(right[1:] - right[:-1] <= 1e-6)
    assert torch.all(down[1:] - down[:-1] <= 1e-6)

    # Symmetry: horizontal, vertical, and transpose (square tile).
    assert torch.allclose(w, torch.flip(w, dims=[0]), atol=1e-6)
    assert torch.allclose(w, torch.flip(w, dims=[1]), atol=1e-6)
    assert torch.allclose(w, w.T, atol=1e-6)


def test_gaussian_window_non_square():
    """Gaussian window on a non-square tile: shape + per-axis behavior."""
    th, tw = 48, 64
    w = build_importance_window((th, tw), mode="gaussian")

    assert w.shape == (th, tw)

    # Per-axis monotone decrease from center to edge.
    center_r, center_c = th // 2, tw // 2
    row = w[center_r, center_c:]  # walk right along the center row
    col = w[center_r:, center_c]  # walk down along the center column
    assert torch.all(row[1:] - row[:-1] <= 1e-6)
    assert torch.all(col[1:] - col[:-1] <= 1e-6)

    # Symmetric per axis.
    assert torch.allclose(w, torch.flip(w, dims=[0]), atol=1e-6)
    assert torch.allclose(w, torch.flip(w, dims=[1]), atol=1e-6)

    # Peak ~1.0 near the center.
    assert torch.allclose(
        w[center_r - 1 : center_r + 1, center_c - 1 : center_c + 1].max(),
        torch.tensor(1.0),
        atol=1e-2,
    )


def test_constant_window_is_ones():
    """Constant mode is uniform (all ones) even after the clamp tail."""
    w = build_importance_window((16, 24), mode="constant")
    assert w.shape == (16, 24)
    assert torch.allclose(w, torch.ones_like(w))


def test_merger_uniform_field_reconstruction():
    """Overlapping uniform tiles stitch back to the tile value everywhere covered."""
    H, W = 20, 20
    th, tw = 10, 10
    v = 3.5

    window = build_importance_window((th, tw), mode="gaussian")
    merger = TileMerger((H, W), channels=2, window=window)

    # Two overlapping tiles of the same constant value v.
    #   tile A covers rows/cols [0:10)
    #   tile B covers rows/cols [5:15)  -> overlap is [5:10) x [5:10)
    tile = torch.full((2, th, tw), v)
    merger.integrate(tile, y0=0, x0=0)
    merger.integrate(tile, y0=5, x0=5)

    out = merger.merge(eps=None)  # bare divide
    assert out.shape == (2, H, W)

    covered = merger.cnt[0] > 0
    # No NaNs / infs on covered pixels.
    assert torch.isfinite(out[:, covered]).all()
    # Uniform field of value v reconstructs to ~v everywhere covered
    # (holds in both the overlap and non-overlap regions, independent of window).
    assert torch.allclose(
        out[:, covered], torch.full_like(out[:, covered], v), atol=1e-5
    )

    # Overlap region specifically equals v (weighted average of v and v).
    overlap = out[:, 5:10, 5:10]
    assert torch.allclose(overlap, torch.full_like(overlap, v), atol=1e-5)
    # A non-overlap covered region (e.g. top-left corner) also equals v.
    corner = out[:, 0:5, 0:5]
    assert torch.allclose(corner, torch.full_like(corner, v), atol=1e-5)


def test_merger_weighted_average_two_values():
    """Overlap of two distinct constant tiles equals the exact window-weighted mean."""
    H, W = 20, 20
    th, tw = 10, 10
    va, vb = 2.0, 6.0

    window = build_importance_window((th, tw), mode="gaussian")
    merger = TileMerger((H, W), channels=1, window=window)

    merger.integrate(torch.full((1, th, tw), va), y0=0, x0=0)
    merger.integrate(torch.full((1, th, tw), vb), y0=5, x0=5)

    out = merger.merge(eps=None)

    # In the overlap [5:10) x [5:10): tile A window is its bottom-right corner
    # (rows/cols 5:10), tile B window is its top-left corner (rows/cols 0:5).
    wa = window[5:10, 5:10]
    wb = window[0:5, 0:5]
    expected = (va * wa + vb * wb) / (wa + wb)
    assert torch.allclose(out[0, 5:10, 5:10], expected, atol=1e-5)


def test_merger_partial_tile_edge_guard():
    """A tile whose origin runs off the canvas edge is clipped without shape error."""
    H, W = 12, 12
    th, tw = 8, 8
    v = 4.0

    window = build_importance_window((th, tw), mode="gaussian")
    merger = TileMerger((H, W), channels=1, window=window)

    # Origin at (8, 8): only a 4x4 slice of the tile fits on the 12x12 canvas.
    full_tile = torch.full((1, th, tw), v)
    partial = full_tile[:, :4, :4]  # (1, 4, 4)
    merger.integrate(partial, y0=8, x0=8)

    out = merger.merge(eps=1e-8)

    # Accumulation landed exactly in the clipped window: acc = v * w, cnt = w.
    w_clip = window[:4, :4]
    assert torch.allclose(merger.acc[0, 8:12, 8:12], v * w_clip, atol=1e-5)
    assert torch.allclose(merger.cnt[0, 8:12, 8:12], w_clip, atol=1e-5)
    # Merge over the covered region recovers v.
    assert torch.allclose(out[0, 8:12, 8:12], torch.full((4, 4), v), atol=1e-5)


def test_merger_fp16_accumulates_in_float32():
    """Window on cpu float32; an fp16 tile is accumulated in float32."""
    window = build_importance_window((8, 8), mode="gaussian", device="cpu")
    assert window.device.type == "cpu"

    merger = TileMerger((16, 16), channels=1, window=window)  # default float32
    assert merger.acc.dtype == torch.float32

    tile = torch.full((1, 8, 8), 2.0, dtype=torch.float16)
    merger.integrate(tile, y0=0, x0=0)

    # Accumulation stayed in float32 despite the fp16 input tile.
    assert merger.acc.dtype == torch.float32
    assert merger.cnt.dtype == torch.float32
    out = merger.merge(eps=1e-8)
    assert out.dtype == torch.float32
    covered = merger.cnt[0] > 0
    assert torch.allclose(
        out[0][covered], torch.full_like(out[0][covered], 2.0), atol=1e-3
    )
