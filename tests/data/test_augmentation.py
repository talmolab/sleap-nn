"""Tests for data augmentation functions."""

import pytest
import numpy as np
import torch
import skia
import sleap_io as sio
from sleap_nn.data.augmentation import (
    apply_intensity_augmentation,
    apply_geometric_augmentation,
)
from sleap_nn.data.skia_augmentation import (
    _transform_image_skia,
    _SKIA_IS_BGRA,
    crop_and_resize_skia,
    apply_geometric_augmentation_skia,
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


class TestSkiaChannelOrdering:
    """Test that Skia-based augmentation preserves RGB channel ordering.

    Skia surfaces use kBGRA_8888 on Linux, so toarray() returns BGRA.
    On macOS, surfaces use kRGBA_8888. These tests verify that channel
    ordering is correct on all platforms.
    """

    def test_skia_bgra_detection(self):
        """Test that _SKIA_IS_BGRA correctly detects the surface format."""
        surface = skia.Surface(1, 1)
        expected_bgra = (
            surface.imageInfo().colorType() == skia.ColorType.kBGRA_8888_ColorType
        )
        assert _SKIA_IS_BGRA == expected_bgra

    def test_transform_image_skia_preserves_rgb(self):
        """Test _transform_image_skia returns RGB, not BGR."""
        # Pure red image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # R=255, G=0, B=0

        result = _transform_image_skia(img, skia.Matrix())

        assert result[5, 5, 0] == 255, "Red channel should be preserved"
        assert result[5, 5, 2] == 0, "Blue channel should remain 0"

    def test_transform_image_skia_all_channels(self):
        """Test that distinct R, G, B values are preserved after transform."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # R
        img[:, :, 1] = 100  # G
        img[:, :, 2] = 50  # B

        result = _transform_image_skia(img, skia.Matrix())

        assert result[5, 5, 0] == 200, f"R expected 200, got {result[5, 5, 0]}"
        assert result[5, 5, 1] == 100, f"G expected 100, got {result[5, 5, 1]}"
        assert result[5, 5, 2] == 50, f"B expected 50, got {result[5, 5, 2]}"

    def test_transform_image_skia_grayscale(self):
        """Test that grayscale images are unaffected."""
        img = np.full((10, 10, 1), 128, dtype=np.uint8)

        result = _transform_image_skia(img, skia.Matrix())

        assert result.shape == (10, 10, 1)
        assert np.allclose(result, 128, atol=1)

    def test_transform_image_skia_contiguous(self):
        """Test that the output array is C-contiguous when BGR swap is applied."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255

        result = _transform_image_skia(img, skia.Matrix())

        # On BGRA platforms, result should be contiguous due to ascontiguousarray
        # On RGBA platforms, it may or may not be depending on Skia internals
        if _SKIA_IS_BGRA:
            assert result.flags["C_CONTIGUOUS"]

    def test_crop_and_resize_skia_preserves_rgb(self):
        """Test crop_and_resize_skia returns RGB, not BGR."""
        img = torch.zeros(1, 3, 10, 10, dtype=torch.uint8)
        img[0, 0] = 200  # R
        img[0, 1] = 100  # G
        img[0, 2] = 50  # B

        boxes = torch.tensor(
            [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=torch.float32
        )
        result = crop_and_resize_skia(img, boxes, size=(10, 10))

        assert result[0, 0, 5, 5] == 200, "R channel should be preserved"
        assert result[0, 1, 5, 5] == 100, "G channel should be preserved"
        assert result[0, 2, 5, 5] == 50, "B channel should be preserved"

    def test_geometric_augmentation_preserves_rgb(self):
        """Test full geometric augmentation pipeline preserves RGB."""
        img = torch.zeros(1, 3, 50, 50, dtype=torch.uint8)
        img[0, 0, :, :25] = 255  # Left half red
        img[0, 2, :, 25:] = 255  # Right half blue
        instances = torch.zeros(1, 1, 2)

        np.random.seed(0)
        result, _ = apply_geometric_augmentation_skia(
            img,
            instances,
            rotation_min=0.0,
            rotation_max=0.001,
            rotation_p=1.0,
        )

        # Left half should still be red (R > B)
        assert (
            result[0, 0, 25, 10] > result[0, 2, 25, 10]
        ), "Left half should be red, not blue (channels may be swapped)"
