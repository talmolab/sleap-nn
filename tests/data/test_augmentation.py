"""Module for testing augmentations with Kornia"""

from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.normalization import Normalizer
from torch.utils.data import DataLoader
import torch


def test_kornia_augmentation(minimal_instance):
    """Test the Kornia augmentations."""
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)
    p = KorniaAugmenter(p, rotation=90, probability=1.0, scale=0.05)

    sample = next(iter(p))
    img, pts = sample["image"], sample["instances"]

    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)
