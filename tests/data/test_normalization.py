import torch
import numpy as np

from sleap_nn.data.normalization import (
    apply_normalization,
    convert_to_grayscale,
    normalize_on_gpu,
)


def test_convert_to_grayscale():
    """Test convert_to_gray_scale function."""
    img = torch.randint(0, 255, (3, 200, 200))
    res = convert_to_grayscale(img)
    assert res.shape[0] == 1


def test_apply_normalization():
    """Test `apply_normalization` function for training pipeline."""
    img = torch.randint(0, 255, (3, 200, 200))
    res = apply_normalization(img)
    assert torch.max(res) <= 1.0 and torch.min(res) >= 0.0
    assert res.dtype == torch.float32


def test_normalize_on_gpu():
    """Test `normalize_on_gpu` function for inference pipeline."""
    # Test uint8 input (typical inference case)
    img = torch.randint(0, 255, (3, 200, 200), dtype=torch.uint8)
    res = normalize_on_gpu(img)
    assert torch.max(res) <= 1.0 and torch.min(res) >= 0.0
    assert res.dtype == torch.float32

    # Test float32 input in [0, 255] range
    img_float = torch.randint(0, 255, (3, 200, 200)).float()
    res_float = normalize_on_gpu(img_float)
    assert torch.max(res_float) <= 1.0 and torch.min(res_float) >= 0.0
    assert res_float.dtype == torch.float32

    # Test already normalized float32 input (should be unchanged)
    img_normalized = torch.rand(3, 200, 200)
    res_normalized = normalize_on_gpu(img_normalized)
    assert torch.allclose(res_normalized, img_normalized)
