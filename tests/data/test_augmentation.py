import pytest
import torch
from torch.utils.data import DataLoader
import sleap_io as sio
from sleap_nn.data.augmentation import KorniaAugmenter, RandomUniformNoise
from sleap_nn.data.augmentation import (
    apply_intensity_augmentation,
    apply_geometric_augmentation,
)
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.providers import process_lf
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.providers import LabelsReaderDP


def test_uniform_noise(minimal_instance):
    """Test RandomUniformNoise module."""
    p = LabelsReaderDP.from_filename(minimal_instance)
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


def test_apply_intensity_augmentation(minimal_instance):
    """Test `apply_intensity_augmentation` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)
    ex["image"] = apply_normalization(ex["image"])

    img, pts = apply_intensity_augmentation(
        ex["image"],
        ex["instances"],
        uniform_noise_p=1.0,
        contrast_p=1.0,
        brightness_p=1.0,
        gaussian_noise_p=1.0,
    )
    # Test all augmentations.
    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert not torch.equal(img, ex["image"])
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)


def test_apply_geometric_augmentation(minimal_instance):
    """Test `apply_geometric_augmentation` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)
    ex["image"] = apply_normalization(ex["image"])

    img, pts = apply_geometric_augmentation(
        ex["image"],
        ex["instances"],
        scale=0.5,
        affine_p=1.0,
        erase_p=1.0,
        mixup_p=1.0,
    )
    # Test all augmentations.
    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert not torch.equal(img, ex["image"])
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)


def test_kornia_augmentation(minimal_instance):
    """Test KorniaAugmenter module."""
    p = LabelsReaderDP.from_filename(minimal_instance)

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
    )

    # Test all augmentations.
    sample = next(iter(p))
    img, pts = sample["image"], sample["instances"]

    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert img.shape == (1, 1, 384, 384)
    assert pts.shape == (1, 2, 2, 2)
