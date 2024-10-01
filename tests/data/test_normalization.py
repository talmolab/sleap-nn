import torch
import numpy as np

from sleap_nn.data.normalization import (
    Normalizer,
    convert_to_grayscale,
    apply_normalization,
)
from sleap_nn.data.providers import LabelsReaderDP


def test_normalizer(minimal_instance):
    """Test Normalizer module."""
    p = LabelsReaderDP.from_filename(minimal_instance)
    p = Normalizer(p)

    ex = next(iter(p))
    assert ex["image"].dtype == torch.float32
    assert ex["image"].shape[-3] == 1

    # test is_rgb
    p = LabelsReaderDP.from_filename(minimal_instance)
    p = Normalizer(p, is_rgb=True)

    ex = next(iter(p))
    assert ex["image"].dtype == torch.float32
    assert ex["image"].shape[-3] == 3


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
