import torch

from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest


def test_providers(minimal_instance):
    """Test LabelsReader module."""
    l = LabelsReader.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    labels = sio.load_slp(minimal_instance)
    org_image = labels[0].image
    image = image.squeeze().squeeze().unsqueeze(dim=-1)
    assert np.all(org_image == image.numpy())
