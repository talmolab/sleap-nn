import torch

from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import numpy as np


def test_providers(minimal_instance):
    l = LabelsReader.from_filename(minimal_instance, max_instances=20)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 20, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    # check max_width, max_height and is_rgb
    l = LabelsReader.from_filename(
        minimal_instance, max_instances=20, max_height=1000, max_width=1000, is_rgb=True
    )
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 3, 1000, 1000])
    assert instances.shape == torch.Size([1, 20, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()
