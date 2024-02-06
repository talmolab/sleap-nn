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
