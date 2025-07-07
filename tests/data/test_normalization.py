import torch
import numpy as np

from sleap_nn.data.normalization import (
    convert_to_grayscale,
    apply_normalization,
)


def test_convert_to_grayscale():
    """Test convert_to_gray_scale function."""
    img = torch.randint(0, 255, (3, 200, 200))
    res = convert_to_grayscale(img)
    assert res.shape[0] == 1


def test_apply_normalization():
    """Test `apply_normalization` function."""
    img = torch.randint(0, 255, (3, 200, 200))
    res = apply_normalization(img)
    assert torch.max(res) <= 1.0 and torch.min(res) == 0.0
    assert res.dtype == torch.float32
