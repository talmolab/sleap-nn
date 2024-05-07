import torch

from sleap_nn.data.normalization import Normalizer
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
