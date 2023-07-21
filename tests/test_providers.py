from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import pytest
import torch


def test_providers(minimal_instance):
    """Test LabelsReader

    Args:
        minimal_instance: minimal_instance testing fixture
    """
    l = LabelsReader.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
