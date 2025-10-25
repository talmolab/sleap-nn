import pytest
import torch
import sleap_io as sio
from sleap_nn.data.augmentation import (
    apply_intensity_augmentation,
    apply_geometric_augmentation,
)
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.providers import process_lf


def test_apply_intensity_augmentation(minimal_instance):
    """Test `apply_intensity_augmentation` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )
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
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"])

    img, pts = apply_geometric_augmentation(
        ex["image"],
        ex["instances"],
        scale_min=0.5,
        scale_max=0.5,
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
