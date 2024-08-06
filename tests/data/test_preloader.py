import pytest
import torch

from sleap_nn.data.dataset_ops import Preloader
from sleap_nn.data.providers import LabelsReader


def test_preloader(minimal_instance):
    """Test Preloader module."""
    p = LabelsReader.from_filename(minimal_instance)
    p = Preloader(p)

    sample = next(iter(p))
    instances, image = sample["instances"], sample["image"]

    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()
