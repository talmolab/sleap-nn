import torch

from sleap_nn.data.providers import LabelsReader


def test_providers(minimal_instance):
    l = LabelsReader.from_filename(minimal_instance)
    sample = next(iter(l))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
