from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.normalization import Normalizer
import torch


def test_normalizer(minimal_instance):
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)

    ex = next(iter(p))
    assert ex["image"].dtype == torch.float32
