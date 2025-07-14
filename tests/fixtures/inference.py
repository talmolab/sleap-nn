"""Inference fixtures for unit testing."""

from pathlib import Path

import pytest


@pytest.fixture
def sleap_nn_inference_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets/inference"


@pytest.fixture
def minimal_cms(sleap_nn_inference_dir):
    """Single (13, 80, 80) confidence map in `.pt` format."""
    return Path(sleap_nn_inference_dir) / "minimal_cms.pt"


@pytest.fixture
def minimal_bboxes(sleap_nn_inference_dir):
    """Single (13, 4, 2) bbox in `.pt` format.

    The order is (n_bboxes, 4, 2 ) where n_bboxes is the number of centroids, and the second dimension
    represents the four corner points of the bounding boxes, each with x and y coordinates.
    The order of the corners follows a clockwise arrangement: top-left, top-right,
    bottom-right, and bottom-left.
    """
    return Path(sleap_nn_inference_dir) / "minimal_bboxes.pt"
