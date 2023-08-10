"""Module for testing augmentations with Kornia."""
from sleap_nn.data.augmentation import KorniaAugmenter, RandomUniformNoise
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.normalization import Normalizer
from torch.utils.data import DataLoader
import torch
import pytest


def test_uniform_noise(minimal_instance):
    """Test the RandomUniformNoise class."""
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)

    sample = next(iter(p))
    img = sample["image"]

    # Testing forward pass.
    aug = RandomUniformNoise(noise=(0.0, 0.04), p=1.0)
    aug_img = aug(img)

    assert torch.is_tensor(aug_img)
    assert aug_img.shape == (1, 1, 384, 384)

    # Testing the _params parameter.
    new_aug_img = aug(img, params=aug._params)
    assert torch.is_tensor(new_aug_img)
    assert new_aug_img.shape == (1, 1, 384, 384)
    assert (new_aug_img == aug_img).all()

    # Testing without clipping output.
    aug = RandomUniformNoise(noise=(0.0, 0.04), p=1.0, clip_output=False)
    aug_img = aug(img)
    assert torch.is_tensor(aug_img)
    assert aug_img.shape == (1, 1, 384, 384)


def test_kornia_augmentation(minimal_instance):
    """Test the Kornia augmentations."""
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)
    p = KorniaAugmenter(
        p,
        affine_p=1.0,
        uniform_noise_p=1.0,
        gaussian_noise_p=1.0,
        contrast_p=1.0,
        brightness_p=1.0,
        erase_p=1.0,
        mixup_p=1.0,
        mixup_lambda=(0.0, 1.0),
        crop_hw=(384, 384),
        crop_p=1.0,
    )

    # Test all augmentations.
    sample = next(iter(p))
    img, pts = sample["image"], sample["instances"]

    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)

    # Test RandomCrop value error.
    p = LabelsReader.from_filename(minimal_instance)
    p = Normalizer(p)
    with pytest.raises(
        ValueError, match="crop_hw height and width must be greater than 0."
    ):
        p = KorniaAugmenter(
            p,
            crop_hw=(0, 0),
            crop_p=1.0,
        )
