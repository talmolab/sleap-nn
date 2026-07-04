"""Tests for the tiling grid/coverage visualization helper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from sleap_nn.data.tiling import generate_tile_grid
from sleap_nn.training.utils import plot_tile_grid


def test_plot_tile_grid_returns_figure_with_all_tiles():
    """One rectangle patch per grid origin, over the image."""
    img = np.zeros((256, 320, 1), dtype=np.uint8)
    fig = plot_tile_grid(
        img, tile_size=128, overlap=32, output_stride=4, show_coverage=False
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    origins = generate_tile_grid((256, 320), 128, 32, 4)
    rects = [p for p in fig.gca().patches if isinstance(p, mpatches.Rectangle)]
    # plot_img may add no rectangles of its own; every origin => one rectangle.
    assert len(rects) == len(origins)
    plt.close(fig)


def test_plot_tile_grid_coverage_has_no_interior_holes():
    """Summed importance-window coverage is strictly positive everywhere.

    With the >=1e-3 window floor and full grid coverage, every output cell is
    touched by at least one tile, so the coverage denominator is never zero (a
    zero would produce a seam hole in the stitched map).
    """
    img = np.zeros((200, 360, 1), dtype=np.uint8)
    fig = plot_tile_grid(
        img,
        tile_size=128,
        overlap=32,
        output_stride=4,
        blend="gaussian",
        show_coverage=True,
    )
    # The coverage heatmap is the last AxesImage added after the base image.
    images = fig.gca().images
    assert len(images) >= 2  # base image + coverage overlay
    coverage = images[-1].get_array()
    assert np.all(np.asarray(coverage) > 0.0)
    plt.close(fig)


def test_plot_tile_grid_accepts_chw_tensor():
    """A (C, H, W) torch tensor is coerced to HWC without error."""
    img = torch.zeros((1, 128, 192), dtype=torch.float32)
    fig = plot_tile_grid(img, tile_size=64, overlap=16, output_stride=2)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_tile_grid_tiny_frame_single_tile():
    """A frame smaller than the tile yields a single tile and still renders."""
    img = np.zeros((48, 48, 1), dtype=np.uint8)
    fig = plot_tile_grid(img, tile_size=128, overlap=32, output_stride=4)
    rects = [p for p in fig.gca().patches if isinstance(p, mpatches.Rectangle)]
    assert len(rects) == 1
    plt.close(fig)
