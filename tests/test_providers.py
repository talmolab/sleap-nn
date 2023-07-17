from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import pytest
import torch


def test_providers(minimal_instance):
    """Test sleap dataset

    Args:
        minimal_instance: minimal_instance testing fixture
    """
    l = LabelsReader.from_filename(minimal_instance)
    image, instance = next(iter(l))
    assert image.shape == torch.Size([384, 384, 1])
    assert instance.shape == torch.Size([2, 2])
