"""Tests for data augmentation functions."""

import pytest
import numpy as np
import torch
import skia
import sleap_io as sio
from sleap_nn.data.augmentation import (
    apply_intensity_augmentation,
    apply_geometric_augmentation,
    apply_flip_augmentation,
)
from sleap_nn.data.skia_augmentation import (
    _transform_image_skia,
    crop_and_resize_skia,
    apply_geometric_augmentation_skia,
)
from sleap_nn.data.utils import get_symmetric_inds
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.providers import process_lf


def _synthetic_img_and_mask(h=128, w=128):
    """A float image whose foreground IS the mask (so co-transform alignment is exact)."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - 54) / 30.0) ** 2 + ((yy - 64) / 14.0) ** 2) <= 1.0
    img = np.zeros((1, 1, h, w), dtype=np.float32)
    img[0, 0][mask] = 1.0
    masks = torch.from_numpy(mask[None, None].astype(np.float32))  # (1, 1, H, W)
    return torch.from_numpy(img), masks, mask


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def test_geometric_augmentation_cotransforms_mask_under_rotation():
    """A mask passed via `masks=` is warped by the SAME affine as the image."""
    np.random.seed(0)
    img, masks, _ = _synthetic_img_and_mask()
    kp = torch.zeros((1, 1, 2))
    out = apply_geometric_augmentation(
        img, kp, rotation_min=90, rotation_max=90, rotation_p=1.0, masks=masks
    )
    assert len(out) == 3, "passing masks must return a 3-tuple"
    img_r, _, masks_r = out
    # Mask stays crisply binary and aligned with the rotated image foreground.
    assert set(np.unique(masks_r.numpy())).issubset({0.0, 1.0})
    img_fg = img_r[0, 0].numpy() > 0.5
    mask_fg = masks_r[0, 0].numpy() > 0.5
    assert _iou(img_fg, mask_fg) > 0.95


def test_geometric_augmentation_without_mask_is_backcompat():
    """Omitting `masks` returns the original 2-tuple (pose path unchanged)."""
    img, _, _ = _synthetic_img_and_mask()
    kp = torch.zeros((1, 1, 2))
    out = apply_geometric_augmentation(
        img, kp, rotation_p=1.0, rotation_min=10, rotation_max=10
    )
    assert len(out) == 2


def test_flip_augmentation_mirrors_mask_exactly():
    """`apply_flip_augmentation(masks=...)` mirrors the mask losslessly (np.fliplr)."""
    img, masks, mask = _synthetic_img_and_mask()
    kp = torch.zeros((1, 1, 2))
    out = apply_flip_augmentation(img, kp, flip_p=1.0, masks=masks)
    assert len(out) == 3
    _, _, masks_f = out
    assert np.array_equal(masks_f[0, 0].numpy() > 0.5, np.fliplr(mask))


def test_geometric_augmentation_mask_excluded_from_erase():
    """Random erase is image-only; the co-transformed mask is untouched by it."""
    np.random.seed(0)
    img, masks, mask = _synthetic_img_and_mask()
    kp = torch.zeros((1, 1, 2))
    # No affine (all p=0), erase always -> mask must equal the input mask exactly.
    _, _, masks_out = apply_geometric_augmentation(
        img, kp, erase_p=1.0, erase_scale_min=0.05, erase_scale_max=0.1, masks=masks
    )
    assert np.array_equal(masks_out[0, 0].numpy() > 0.5, mask)


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

    The implementation uses array-backed Skia surfaces (following the sleap-io
    pattern) which avoids platform-specific BGR/RGBA pixel format issues. These
    tests verify that channel ordering is correct on all platforms.
    """

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

    def test_transform_image_skia_output_shape(self):
        """Test that the output array has correct shape and dtype."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255

        result = _transform_image_skia(img, skia.Matrix())

        assert result.shape == (10, 10, 3), f"Expected (10, 10, 3), got {result.shape}"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

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


class TestFlipAugmentation:
    """Tests for symmetry-aware horizontal (left/right) flip augmentation."""

    def _img_and_instances(self):
        """Return a small (1,1,2,3) image and (1,1,2,2) two-node instance."""
        # Pixels are distinct so mirroring is detectable.
        img = torch.arange(6, dtype=torch.uint8).reshape(1, 1, 2, 3)
        # node0 at (x=0, y=0); node1 at (x=2, y=1)
        instances = torch.tensor([[[[0.0, 0.0], [2.0, 1.0]]]])
        return img, instances

    def test_horizontal_flip_coords_and_image(self):
        """H-flip mirrors x (x' = W-1-x) and reverses image columns."""
        img, instances = self._img_and_instances()
        out_img, out_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=None, flip_p=1.0
        )
        # W=3 -> columns reversed.
        assert torch.equal(
            out_img[0, 0], torch.tensor([[2, 1, 0], [5, 4, 3]], dtype=torch.uint8)
        )
        # x mirrored: node0 (0,0)->(2,0); node1 (2,1)->(0,1). No swap.
        assert torch.allclose(out_inst[0, 0], torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
        assert out_img.dtype == torch.uint8

    def test_symmetry_swap(self):
        """Symmetric node pairs are swapped after mirroring."""
        img, instances = self._img_and_instances()
        _, out_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=[(0, 1)], flip_p=1.0
        )
        # node1 mirrored (0,1) lands in slot 0; node0 mirrored (2,0) in slot 1.
        assert torch.allclose(out_inst[0, 0], torch.tensor([[0.0, 1.0], [2.0, 0.0]]))

    def test_flip_p_zero_is_noop(self):
        """flip_p=0 returns inputs unchanged."""
        img, instances = self._img_and_instances()
        out_img, out_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=[(0, 1)], flip_p=0.0
        )
        assert torch.equal(out_img, img)
        assert torch.equal(out_inst, instances)

    def test_flip_p_one_always_flips(self):
        """flip_p=1 always applies the flip (image actually changes)."""
        img, instances = self._img_and_instances()
        for _ in range(5):
            out_img, _ = apply_flip_augmentation(img, instances, flip_p=1.0)
            assert not torch.equal(out_img, img)

    def test_nan_keypoints_preserved(self):
        """NaN keypoints stay NaN through mirroring and swap."""
        img, _ = self._img_and_instances()
        instances = torch.tensor([[[[float("nan"), float("nan")], [2.0, 1.0]]]])
        _, out_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=[(0, 1)], flip_p=1.0
        )
        # NaN node swapped into slot 1, still NaN.
        assert torch.isnan(out_inst[0, 0, 1, 0])
        assert torch.isnan(out_inst[0, 0, 1, 1])

    def test_topdown_shape_no_node_swap(self):
        """Works on top-down crop shape (1, n_nodes, 2)."""
        img = torch.arange(6, dtype=torch.uint8).reshape(1, 1, 2, 3)
        instances = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])  # (1, 2, 2)
        _, out_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=[(0, 1)], flip_p=1.0
        )
        assert out_inst.shape == (1, 2, 2)
        assert torch.allclose(out_inst[0], torch.tensor([[0.0, 1.0], [2.0, 0.0]]))

    def test_double_flip_is_identity(self):
        """Flipping twice (with the same swap) restores the original."""
        img, instances = self._img_and_instances()
        once_img, once_inst = apply_flip_augmentation(
            img, instances, symmetric_inds=[(0, 1)], flip_p=1.0
        )
        twice_img, twice_inst = apply_flip_augmentation(
            once_img, once_inst, symmetric_inds=[(0, 1)], flip_p=1.0
        )
        assert torch.equal(twice_img, img)
        assert torch.allclose(twice_inst, instances)

    def test_flip_via_geometric_wrapper(self):
        """flip_p threads through apply_geometric_augmentation."""
        img, instances = self._img_and_instances()
        np.random.seed(0)
        out_img, out_inst = apply_geometric_augmentation(
            img,
            instances,
            rotation_p=0.0,
            scale_p=0.0,
            translate_p=0.0,
            flip_p=1.0,
            symmetric_inds=[(0, 1)],
        )
        assert torch.equal(
            out_img[0, 0], torch.tensor([[2, 1, 0], [5, 4, 3]], dtype=torch.uint8)
        )
        assert torch.allclose(out_inst[0, 0], torch.tensor([[0.0, 1.0], [2.0, 0.0]]))


class TestGetSymmetricInds:
    """Tests for resolving symmetric node-index pairs from a skeleton."""

    def test_resolves_pairs_from_raw_symmetries(self):
        skel = sio.Skeleton(["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
        skel.add_symmetry("left_eye", "right_eye")
        skel.add_symmetry("left_ear", "right_ear")
        pairs = get_symmetric_inds(skel)
        # Within-pair order is set-derived; normalize before comparing.
        assert sorted(tuple(sorted(p)) for p in pairs) == [(1, 2), (3, 4)]

    def test_no_symmetries_returns_empty(self):
        skel = sio.Skeleton(["a", "b", "c"])
        assert get_symmetric_inds(skel) == []

    def test_none_skeleton_returns_empty(self):
        assert get_symmetric_inds(None) == []
