import torch

from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest


def test_providers(minimal_instance):
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

    # check max_width, max_height and is_rgb
    l = LabelsReader.from_filename(
        minimal_instance, max_height=500, max_width=500, is_rgb=True
    )
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 3, 500, 500])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    l = LabelsReader.from_filename(
        minimal_instance, max_height=100, max_width=500, is_rgb=True
    )
    with pytest.raises(
        Exception,
        match=f"Max height {100} should be greater than the current image height: {384}",
    ):
        sample = next(iter(l))

    l = LabelsReader.from_filename(
        minimal_instance, max_height=500, max_width=100, is_rgb=True
    )
    with pytest.raises(
        Exception,
        match=f"Max width {100} should be greater than the current image width: {384}",
    ):
        sample = next(iter(l))
