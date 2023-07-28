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
    p = KorniaAugmenter(
        p,
        crop_hw=(384, 384),
        crop_p=1.0,
        affine_p=1.0,
        uniform_noise_p=1.0,
        gaussian_noise_p=1.0,
        contrast_p=1.0,
        brightness_p=1.0,
        erase_p=1.0,
        mixup_p=1.0,
        mixup_lambda=(0.0, 1.0),
    )

    sample = next(iter(p))
    img, pts = sample["image"], sample["instances"]

    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)
