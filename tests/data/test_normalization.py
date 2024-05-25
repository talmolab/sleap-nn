import torch
import numpy as np

from sleap_nn.data.normalization import Normalizer, convert_to_grayscale
from sleap_nn.data.providers import LabelsReader


def test_normalizer(minimal_instance):
    """Test Normalizer module."""
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)

    ex = next(iter(p))
    assert ex["image"].dtype == torch.float32
    assert ex["image"].shape[-3] == 1

    # test is_rgb
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p, is_rgb=True)

    ex = next(iter(p))
    assert ex["image"].dtype == torch.float32
    assert ex["image"].shape[-3] == 3


def test_convert_to_grayscale():
    """Test convert_to_gray_scale function."""
    img = torch.randint(0, 255, (3, 200, 200))
    res = convert_to_grayscale(img)
    assert res.shape[0] == 1
