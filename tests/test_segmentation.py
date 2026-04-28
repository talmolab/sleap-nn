"""Comprehensive tests for the instance segmentation feature in sleap-nn."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# 1. GT generation (sleap_nn/data/segmentation_maps.py)
# ---------------------------------------------------------------------------

from sleap_nn.data.segmentation_maps import (
    generate_foreground_mask,
    generate_center_heatmap,
    generate_center_offsets,
    _compute_mask_centroids,
)


def test_generate_foreground_mask_basic():
    """Two non-overlapping rectangular masks produce correct shape and pixel counts."""
    h, w = 64, 64
    output_stride = 1

    # Mask A: 10x10 block at top-left
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[0:10, 0:10] = True

    # Mask B: 10x10 block at bottom-right
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[50:60, 50:60] = True

    fg = generate_foreground_mask([mask_a, mask_b], (h, w), output_stride=output_stride)

    assert fg.shape == (1, 1, h, w)
    assert fg.dtype == torch.float32

    # Count foreground pixels: should be exactly 100 + 100 = 200
    assert fg.sum().item() == 200.0


def test_generate_foreground_mask_empty():
    """Empty mask list returns all zeros."""
    h, w = 32, 32
    fg = generate_foreground_mask([], (h, w), output_stride=1)

    assert fg.shape == (1, 1, h, w)
    assert fg.sum().item() == 0.0


def test_generate_foreground_mask_overlapping():
    """Two overlapping masks produce the correct union."""
    h, w = 64, 64
    output_stride = 1

    # Mask A: rows 0-19, cols 0-19  (20x20 = 400 pixels)
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[0:20, 0:20] = True

    # Mask B: rows 10-29, cols 10-29  (20x20 = 400 pixels)
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[10:30, 10:30] = True

    fg = generate_foreground_mask([mask_a, mask_b], (h, w), output_stride=output_stride)

    # Union = 400 + 400 - overlap(10x10=100) = 700
    assert fg.shape == (1, 1, h, w)
    assert fg.sum().item() == 700.0


def test_generate_center_heatmap():
    """Verify shape, peak locations near mask centroids, and max value close to 1."""
    h, w = 64, 64
    output_stride = 1
    sigma = 5.0

    # Single 20x20 mask centered at (30, 30) -- centroid at (cols, rows) = (30, 30)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:40] = True

    heatmap = generate_center_heatmap(
        [mask], (h, w), output_stride=output_stride, sigma=sigma
    )

    assert heatmap.shape == (1, 1, h, w)
    assert heatmap.dtype == torch.float32

    # Max value should be very close to 1 (Gaussian evaluated at center)
    assert heatmap.max().item() > 0.95

    # Peak location should be near centroid (29.5, 29.5)
    peak_idx = heatmap[0, 0].argmax().item()
    peak_row = peak_idx // w
    peak_col = peak_idx % w
    # Allow 1 pixel tolerance
    assert abs(peak_row - 29.5) <= 1.0
    assert abs(peak_col - 29.5) <= 1.0


def test_generate_center_heatmap_empty():
    """Empty mask list returns zeros."""
    h, w = 32, 32
    heatmap = generate_center_heatmap([], (h, w), output_stride=1, sigma=5.0)
    assert heatmap.shape == (1, 1, h, w)
    assert heatmap.sum().item() == 0.0


def test_generate_center_offsets():
    """Verify offset shape, that offsets point toward mask centroid, and weight mask matches foreground."""
    h, w = 64, 64
    output_stride = 1

    # 20x20 mask at rows 20-39, cols 20-39 -> centroid ~ (29.5, 29.5)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:40] = True

    offsets, weight_mask = generate_center_offsets(
        [mask], (h, w), output_stride=output_stride
    )

    assert offsets.shape == (1, 2, h, w)
    assert weight_mask.shape == (1, 1, h, w)

    # Weight mask should match foreground
    expected_fg = generate_foreground_mask([mask], (h, w), output_stride=output_stride)
    assert (weight_mask == expected_fg).all()

    # Check that offsets on foreground pixels point toward centroid
    cx, cy = 29.5, 29.5  # centroid in pixel coords

    # For a foreground pixel, pixel + offset should equal the centroid
    # offsets are stored as (dx, dy) where pixel_coord + offset = center
    # But the actual computation is: offset = center - pixel_coord_in_orig_space
    # where pixel_coord_in_orig_space = idx * output_stride + output_stride / 2
    # With output_stride=1, pixel_coord = idx * 1 + 0.5

    # Test a specific foreground pixel at row=20, col=20
    # pixel coords = (20*1 + 0.5, 20*1 + 0.5) = (20.5, 20.5)
    # dx = cx - 20.5 = 29.5 - 20.5 = 9.0
    # dy = cy - 20.5 = 29.5 - 20.5 = 9.0
    assert abs(offsets[0, 0, 20, 20].item() - 9.0) < 0.5
    assert abs(offsets[0, 1, 20, 20].item() - 9.0) < 0.5

    # Test a pixel at the centroid itself: row=30, col=30
    # pixel coords = (30.5, 30.5) => dx = 29.5 - 30.5 = -1.0
    assert abs(offsets[0, 0, 30, 30].item() - (-1.0)) < 0.5
    assert abs(offsets[0, 1, 30, 30].item() - (-1.0)) < 0.5


def test_generate_center_offsets_empty():
    """Empty mask list returns zeros for offsets and weight mask."""
    h, w = 32, 32
    offsets, weight_mask = generate_center_offsets([], (h, w), output_stride=1)
    assert offsets.shape == (1, 2, h, w)
    assert weight_mask.shape == (1, 1, h, w)
    assert offsets.sum().item() == 0.0
    assert weight_mask.sum().item() == 0.0


def test_compute_mask_centroids():
    """Known rectangles produce known centroids."""
    # Rectangle at rows 0-9, cols 0-9 -> centroid = (4.5, 4.5)
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_a[0:10, 0:10] = True

    # Rectangle at rows 20-29, cols 40-49 -> centroid = (44.5, 24.5)
    mask_b = np.zeros((64, 64), dtype=bool)
    mask_b[20:30, 40:50] = True

    centers = _compute_mask_centroids([mask_a, mask_b])

    assert len(centers) == 2
    # Center of mask_a: x=mean(0..9)=4.5, y=mean(0..9)=4.5
    assert abs(centers[0][0] - 4.5) < 0.1
    assert abs(centers[0][1] - 4.5) < 0.1
    # Center of mask_b: x=mean(40..49)=44.5, y=mean(20..29)=24.5
    assert abs(centers[1][0] - 44.5) < 0.1
    assert abs(centers[1][1] - 24.5) < 0.1


def test_compute_mask_centroids_empty_mask():
    """Empty mask falls back to image center."""
    mask = np.zeros((64, 80), dtype=bool)
    centers = _compute_mask_centroids([mask])
    # Fallback: (width/2, height/2) = (40.0, 32.0)
    assert abs(centers[0][0] - 40.0) < 0.1
    assert abs(centers[0][1] - 32.0) < 0.1


# ---------------------------------------------------------------------------
# 2. Head classes (sleap_nn/architectures/heads.py)
# ---------------------------------------------------------------------------

from sleap_nn.architectures.heads import (
    Head,
    SegmentationHead,
    InstanceCenterHead,
    CenterOffsetHead,
)


def test_segmentation_head():
    """SegmentationHead has channels=1, activation='sigmoid', loss_function='bce_dice'."""
    head = SegmentationHead(output_stride=2, loss_weight=1.0)

    assert head.channels == 1
    assert head.activation == "identity"
    assert head.loss_function == "bce_dice"
    assert head.output_stride == 2
    assert head.loss_weight == 1.0
    assert isinstance(head, Head)
    assert head.name == "SegmentationHead"


def test_instance_center_head():
    """InstanceCenterHead has channels=1 and stores sigma."""
    head = InstanceCenterHead(sigma=10.0, output_stride=2, loss_weight=1.0)

    assert head.channels == 1
    assert head.sigma == 10.0
    assert head.output_stride == 2
    assert head.loss_weight == 1.0
    # Default activation and loss from base Head
    assert head.activation == "identity"
    assert head.loss_function == "mse"
    assert isinstance(head, Head)
    assert head.name == "InstanceCenterHead"


def test_center_offset_head():
    """CenterOffsetHead has channels=2 and loss_function='smooth_l1'."""
    head = CenterOffsetHead(output_stride=2, loss_weight=0.1)

    assert head.channels == 2
    assert head.loss_function == "smooth_l1"
    assert head.output_stride == 2
    assert head.loss_weight == 0.1
    # Default activation from base Head
    assert head.activation == "identity"
    assert isinstance(head, Head)
    assert head.name == "CenterOffsetHead"


def test_segmentation_head_make_head():
    """Create SegmentationHead, run forward pass with dummy input, check output shape."""
    sample_input = torch.randn(2, 64, 32, 32)  # (B, C_in, H, W)
    head = SegmentationHead(output_stride=2, loss_weight=1.0)

    head_module = head.make_head(x_in=sample_input.size(1))
    output = head_module(sample_input)

    assert output.shape == (2, 1, 32, 32)  # channels=1
    assert output.dtype == torch.float32
    # Identity activation means raw logits; just check they are finite
    assert torch.isfinite(output).all()


def test_instance_center_head_make_head():
    """InstanceCenterHead forward pass produces single-channel output."""
    sample_input = torch.randn(2, 64, 32, 32)
    head = InstanceCenterHead(sigma=10.0, output_stride=2, loss_weight=1.0)

    head_module = head.make_head(x_in=sample_input.size(1))
    output = head_module(sample_input)

    assert output.shape == (2, 1, 32, 32)
    assert output.dtype == torch.float32


def test_center_offset_head_make_head():
    """CenterOffsetHead forward pass produces 2-channel output."""
    sample_input = torch.randn(2, 64, 32, 32)
    head = CenterOffsetHead(output_stride=2, loss_weight=0.1)

    head_module = head.make_head(x_in=sample_input.size(1))
    output = head_module(sample_input)

    assert output.shape == (2, 2, 32, 32)
    assert output.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. Loss functions (sleap_nn/training/losses.py)
# ---------------------------------------------------------------------------

from sleap_nn.training.losses import compute_bce_dice_loss, compute_masked_smooth_l1


def test_bce_dice_loss_perfect():
    """Identical pred and gt should give approximately 0 loss."""
    gt = torch.ones(1, 1, 16, 16)
    pred = torch.full((1, 1, 16, 16), 10.0)

    loss = compute_bce_dice_loss(pred, gt)

    assert loss.item() < 0.01


def test_bce_dice_loss_worst():
    """All-zeros pred with all-ones gt should give high loss."""
    gt = torch.ones(1, 1, 16, 16)
    # Large negative logit -> sigmoid near 0 -> worst case against gt of all ones
    pred = torch.full((1, 1, 16, 16), -10.0)

    loss = compute_bce_dice_loss(pred, gt)

    # Loss should be significantly larger than 0
    assert loss.item() > 1.0


def test_bce_dice_loss_half():
    """Pred of 0.5 everywhere with all-ones gt gives intermediate loss."""
    gt = torch.ones(1, 1, 16, 16)
    pred = torch.full((1, 1, 16, 16), 0.0)

    loss = compute_bce_dice_loss(pred, gt)

    # Should be between perfect and worst
    assert 0.0 < loss.item() < 5.0


def test_bce_dice_loss_gradient():
    """Loss supports backpropagation."""
    pred = torch.full((1, 1, 8, 8), 0.0, requires_grad=True)
    gt = torch.ones(1, 1, 8, 8)

    loss = compute_bce_dice_loss(pred, gt)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_masked_smooth_l1_basic():
    """Basic masked offset loss computation with known values."""
    # 2-channel offsets
    pred = torch.ones(1, 2, 4, 4)
    gt = torch.zeros(1, 2, 4, 4)
    # Mask only top-left 2x2
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, :2, :2] = 1.0

    loss = compute_masked_smooth_l1(pred, gt, mask)

    # smooth_l1_loss(1, 0) = 0.5 (since |1-0|=1.0, smooth_l1 for |x|>=1 is |x|-0.5=0.5)
    # 8 valid elements (2 channels * 2*2 pixels), sum = 8 * 0.5 = 4.0, mean = 4.0 / 8 = 0.5
    assert abs(loss.item() - 0.5) < 0.01


def test_masked_smooth_l1_empty_mask():
    """Empty mask returns 0 loss."""
    pred = torch.ones(1, 2, 4, 4, requires_grad=True)
    gt = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)

    loss = compute_masked_smooth_l1(pred, gt, mask)

    assert loss.item() == 0.0
    # Should still be differentiable
    assert loss.requires_grad


def test_masked_smooth_l1_full_mask():
    """Full mask computes loss over all pixels."""
    pred = torch.full((1, 2, 4, 4), 0.3)
    gt = torch.zeros(1, 2, 4, 4)
    mask = torch.ones(1, 1, 4, 4)

    loss = compute_masked_smooth_l1(pred, gt, mask)

    # smooth_l1 for |x|<1: 0.5 * x^2 = 0.5 * 0.09 = 0.045
    # All 32 elements have this value, so mean = 0.045
    assert abs(loss.item() - 0.045) < 0.01


# ---------------------------------------------------------------------------
# 4. Model forward pass
# ---------------------------------------------------------------------------

from sleap_nn.architectures.model import Model


def test_bottomup_segmentation_model_forward():
    """Create Model with bottomup_segmentation type, run forward, verify output dict keys and shapes."""
    backbone_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 8,
            "filters_rate": 1.0,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "output_stride": 2,
        }
    )

    head_configs = OmegaConf.create(
        {
            "segmentation": {
                "output_stride": 2,
                "loss_weight": 1.0,
            },
            "center": {
                "sigma": 10.0,
                "output_stride": 2,
                "loss_weight": 1.0,
            },
            "offsets": {
                "output_stride": 2,
                "loss_weight": 0.1,
            },
        }
    )

    model = Model(
        backbone_type="unet",
        backbone_config=backbone_config,
        head_configs=head_configs,
        model_type="bottomup_segmentation",
    )

    # Forward pass with a batch of 2 grayscale 64x64 images
    x = torch.randn(2, 1, 64, 64)
    outputs = model(x)

    assert "SegmentationHead" in outputs
    assert "InstanceCenterHead" in outputs
    assert "CenterOffsetHead" in outputs

    B = 2
    # Output spatial dims = 64 / output_stride(2) = 32
    assert outputs["SegmentationHead"].shape == (B, 1, 32, 32)
    assert outputs["InstanceCenterHead"].shape == (B, 1, 32, 32)
    assert outputs["CenterOffsetHead"].shape == (B, 2, 32, 32)

    # All outputs should be float32
    for key in outputs:
        assert outputs[key].dtype == torch.float32


def test_bottomup_segmentation_model_three_heads():
    """Verify that the model creates exactly three heads of the right types."""
    backbone_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 8,
            "filters_rate": 1.0,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "output_stride": 2,
        }
    )

    head_configs = OmegaConf.create(
        {
            "segmentation": {
                "output_stride": 2,
                "loss_weight": 1.0,
            },
            "center": {
                "sigma": 10.0,
                "output_stride": 2,
                "loss_weight": 1.0,
            },
            "offsets": {
                "output_stride": 2,
                "loss_weight": 0.1,
            },
        }
    )

    model = Model(
        backbone_type="unet",
        backbone_config=backbone_config,
        head_configs=head_configs,
        model_type="bottomup_segmentation",
    )

    assert len(model.heads) == 3
    assert isinstance(model.heads[0], SegmentationHead)
    assert isinstance(model.heads[1], InstanceCenterHead)
    assert isinstance(model.heads[2], CenterOffsetHead)


# ---------------------------------------------------------------------------
# 5. Instance grouping (sleap_nn/inference/segmentation.py)
# ---------------------------------------------------------------------------

from sleap_nn.inference.segmentation import group_instances_from_offsets


def test_group_instances_two_blobs():
    """Two well-separated blobs with correct offsets should find 2 instances."""
    h, w = 32, 32
    output_stride = 2

    # Create foreground with two blobs
    foreground = torch.zeros(1, 1, h, w)
    # Blob 1: rows 2-7, cols 2-7
    foreground[0, 0, 2:8, 2:8] = 1.0
    # Blob 2: rows 22-27, cols 22-27
    foreground[0, 0, 22:28, 22:28] = 1.0

    # Center heatmap with peaks at blob centroids (in output stride coords)
    center_heatmap = torch.zeros(1, 1, h, w)
    # Blob 1 center: row=5, col=5 (at output res)
    center_heatmap[0, 0, 5, 5] = 1.0
    # Blob 2 center: row=25, col=25 (at output res)
    center_heatmap[0, 0, 25, 25] = 1.0

    # Create offsets that point each foreground pixel to its blob's center
    offsets = torch.zeros(1, 2, h, w)

    # For blob 1: center in original coords = (5*2 + 1, 5*2 + 1) = (11, 11)
    for r in range(2, 8):
        for c in range(2, 8):
            pixel_x = c * output_stride + output_stride / 2.0
            pixel_y = r * output_stride + output_stride / 2.0
            center_x = 5 * output_stride + output_stride / 2.0
            center_y = 5 * output_stride + output_stride / 2.0
            offsets[0, 0, r, c] = center_x - pixel_x
            offsets[0, 1, r, c] = center_y - pixel_y

    # For blob 2: center in original coords = (25*2 + 1, 25*2 + 1) = (51, 51)
    for r in range(22, 28):
        for c in range(22, 28):
            pixel_x = c * output_stride + output_stride / 2.0
            pixel_y = r * output_stride + output_stride / 2.0
            center_x = 25 * output_stride + output_stride / 2.0
            center_y = 25 * output_stride + output_stride / 2.0
            offsets[0, 0, r, c] = center_x - pixel_x
            offsets[0, 1, r, c] = center_y - pixel_y

    instances = group_instances_from_offsets(
        foreground=foreground,
        center_heatmap=center_heatmap,
        offsets=offsets,
        fg_threshold=0.5,
        peak_threshold=0.5,
        output_stride=output_stride,
    )

    assert len(instances) == 2

    # Each instance should have a mask, center, and score
    for inst in instances:
        assert "mask" in inst
        assert "center" in inst
        assert "score" in inst
        assert inst["mask"].shape == (h, w)
        assert isinstance(inst["center"], tuple)
        assert len(inst["center"]) == 2
        assert inst["score"] >= 0.5

    # Each blob has 6*6 = 36 pixels; verify roughly correct pixel counts
    mask_sizes = sorted([inst["mask"].sum() for inst in instances])
    assert mask_sizes[0] == 36
    assert mask_sizes[1] == 36


def test_group_instances_no_foreground():
    """Empty foreground returns empty list."""
    h, w = 16, 16
    foreground = torch.zeros(1, 1, h, w)
    center_heatmap = torch.zeros(1, 1, h, w)
    offsets = torch.zeros(1, 2, h, w)

    instances = group_instances_from_offsets(
        foreground=foreground,
        center_heatmap=center_heatmap,
        offsets=offsets,
        fg_threshold=0.5,
        peak_threshold=0.2,
        output_stride=2,
    )

    assert instances == []


def test_group_instances_no_centers():
    """Foreground present but no center peaks returns empty list."""
    h, w = 16, 16
    foreground = torch.ones(1, 1, h, w)  # All foreground
    center_heatmap = torch.full((1, 1, h, w), 0.01)  # Very low values, no peaks
    offsets = torch.zeros(1, 2, h, w)

    instances = group_instances_from_offsets(
        foreground=foreground,
        center_heatmap=center_heatmap,
        offsets=offsets,
        fg_threshold=0.5,
        peak_threshold=0.5,  # Higher than any value in center_heatmap
        output_stride=2,
    )

    assert instances == []


def test_group_instances_single_blob():
    """Single blob with one center peak returns exactly one instance."""
    h, w = 16, 16
    output_stride = 1

    foreground = torch.zeros(1, 1, h, w)
    foreground[0, 0, 4:12, 4:12] = 1.0

    center_heatmap = torch.zeros(1, 1, h, w)
    center_heatmap[0, 0, 8, 8] = 1.0  # Peak at center of blob

    offsets = torch.zeros(1, 2, h, w)
    center_x = 8 * output_stride + output_stride / 2.0
    center_y = 8 * output_stride + output_stride / 2.0
    for r in range(4, 12):
        for c in range(4, 12):
            px = c * output_stride + output_stride / 2.0
            py = r * output_stride + output_stride / 2.0
            offsets[0, 0, r, c] = center_x - px
            offsets[0, 1, r, c] = center_y - py

    instances = group_instances_from_offsets(
        foreground=foreground,
        center_heatmap=center_heatmap,
        offsets=offsets,
        fg_threshold=0.5,
        peak_threshold=0.5,
        output_stride=output_stride,
    )

    assert len(instances) == 1
    assert instances[0]["mask"].sum() == 64  # 8x8


# ---------------------------------------------------------------------------
# 6. Config
# ---------------------------------------------------------------------------

from sleap_nn.config.model_config import (
    BottomUpSegmentationConfig,
    SegmentationHeadConfig,
    InstanceCenterConfig,
    CenterOffsetConfig,
    HeadConfig,
)


def test_bottomup_segmentation_config():
    """Create BottomUpSegmentationConfig and verify defaults."""
    cfg = BottomUpSegmentationConfig()

    assert cfg.segmentation is None
    assert cfg.center is None
    assert cfg.offsets is None


def test_bottomup_segmentation_config_with_values():
    """Create BottomUpSegmentationConfig with explicit sub-configs."""
    seg_cfg = SegmentationHeadConfig(output_stride=2, loss_weight=1.0)
    center_cfg = InstanceCenterConfig(sigma=10.0, output_stride=2, loss_weight=1.0)
    offset_cfg = CenterOffsetConfig(output_stride=2, loss_weight=0.1)

    cfg = BottomUpSegmentationConfig(
        segmentation=seg_cfg,
        center=center_cfg,
        offsets=offset_cfg,
    )

    assert cfg.segmentation.output_stride == 2
    assert cfg.segmentation.loss_weight == 1.0
    assert cfg.center.sigma == 10.0
    assert cfg.center.output_stride == 2
    assert cfg.center.loss_weight == 1.0
    assert cfg.offsets.output_stride == 2
    assert cfg.offsets.loss_weight == 0.1


def test_segmentation_head_config_defaults():
    """Verify default values of SegmentationHeadConfig."""
    cfg = SegmentationHeadConfig()
    assert cfg.output_stride == 2
    assert cfg.loss_weight == 1.0


def test_instance_center_config_defaults():
    """Verify default values of InstanceCenterConfig."""
    cfg = InstanceCenterConfig()
    assert cfg.sigma == 10.0
    assert cfg.output_stride == 2
    assert cfg.loss_weight == 1.0


def test_center_offset_config_defaults():
    """Verify default values of CenterOffsetConfig."""
    cfg = CenterOffsetConfig()
    assert cfg.output_stride == 2
    assert cfg.loss_weight == 0.1


def test_head_config_oneof_bottomup_segmentation():
    """HeadConfig oneof works with bottomup_segmentation."""
    head_cfg = HeadConfig(
        bottomup_segmentation=BottomUpSegmentationConfig(
            segmentation=SegmentationHeadConfig(),
            center=InstanceCenterConfig(),
            offsets=CenterOffsetConfig(),
        )
    )

    assert head_cfg.bottomup_segmentation is not None
    assert head_cfg.single_instance is None
    assert head_cfg.centroid is None
    assert head_cfg.centered_instance is None
    assert head_cfg.bottomup is None
    assert head_cfg.multi_class_bottomup is None
    assert head_cfg.multi_class_topdown is None


# ---------------------------------------------------------------------------
# 7. Predictor (sleap_nn/inference/predictors.py)
# ---------------------------------------------------------------------------

from sleap_nn.inference.predictors import BottomUpSegmentationPredictor
from sleap_nn.inference.segmentation import (
    BottomUpSegmentationInferenceModel,
)


def test_inference_model_no_padding_params():
    """BottomUpSegmentationInferenceModel no longer accepts max_stride/input_scale."""

    def dummy_model(x):
        b = x.shape[0]
        h, w = x.shape[-2] // 2, x.shape[-1] // 2
        return {
            "SegmentationHead": torch.zeros(b, 1, h, w),
            "InstanceCenterHead": torch.zeros(b, 1, h, w),
            "CenterOffsetHead": torch.zeros(b, 2, h, w),
        }

    # Should work with just the 3 required params
    model = BottomUpSegmentationInferenceModel(
        torch_model=dummy_model,
        fg_threshold=0.5,
        peak_threshold=0.2,
        output_stride=2,
    )
    assert model.output_stride == 2
    assert not hasattr(model, "max_stride")
    assert not hasattr(model, "input_scale")


def test_run_inference_on_batch_output():
    """_run_inference_on_batch yields per-frame dicts with correct keys."""

    def dummy_forward(x):
        """Simulate model forward pass (receives (B, 1, C, H, W))."""
        x = x.squeeze(1)  # (B, C, H, W)
        b = x.shape[0]
        h, w = x.shape[-2] // 2, x.shape[-1] // 2
        return {
            "SegmentationHead": torch.sigmoid(torch.randn(b, 1, h, w)),
            "InstanceCenterHead": torch.zeros(b, 1, h, w),
            "CenterOffsetHead": torch.zeros(b, 2, h, w),
        }

    predictor = BottomUpSegmentationPredictor(
        output_stride=2,
        batch_size=2,
        max_stride=16,
        preprocess_config={
            "scale": 1.0,
            "ensure_rgb": False,
            "ensure_grayscale": False,
            "crop_size": None,
            "max_height": None,
            "max_width": None,
        },
    )
    predictor.inference_model = BottomUpSegmentationInferenceModel(
        torch_model=dummy_forward,
        fg_threshold=0.5,
        peak_threshold=0.2,
        output_stride=2,
    )

    # Build a fake batch (2 frames, 3-channel 32x32 images)
    imgs = [torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)]
    fidxs = [0, 1]
    vidxs = [0, 0]
    org_szs = [torch.tensor([[32.0, 32.0]]), torch.tensor([[32.0, 32.0]])]
    instances = []
    eff_scales = [1.0, 1.0]

    results = list(
        predictor._run_inference_on_batch(
            imgs, fidxs, vidxs, org_szs, instances, eff_scales
        )
    )

    assert len(results) == 2
    for r in results:
        assert "video_idx" in r
        assert "frame_idx" in r
        assert "orig_size" in r
        assert "instances" in r
        assert "padded_size" in r
        assert isinstance(r["instances"], list)


def test_mask_upscaling():
    """Masks are correctly cropped from padding and upscaled to original resolution."""
    import sleap_io as sio

    # Simulate: original image 30x30, output_stride=2
    # After pad_to_stride(16), image becomes 32x32
    # Model output maps are 16x16 (padded/stride)
    # Original output size should be 15x15 (30/2)
    orig_h, orig_w = 30, 30
    output_stride = 2
    padded_h, padded_w = 32, 32

    # A mask at output stride resolution (16x16 due to padding)
    mask = np.zeros((padded_h // output_stride, padded_w // output_stride), dtype=bool)
    mask[2:8, 2:8] = True  # 6x6 block

    # Crop to original output dims
    crop_h = orig_h // output_stride  # 15
    crop_w = orig_w // output_stride  # 15
    cropped = mask[:crop_h, :crop_w]
    assert cropped.shape == (15, 15)
    # The 6x6 block should still be fully within bounds
    assert cropped[2:8, 2:8].sum() == 36

    # Upscale to original resolution
    mask_tensor = torch.from_numpy(cropped.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_upscaled = torch.nn.functional.interpolate(
        mask_tensor, size=(orig_h, orig_w), mode="nearest"
    )
    mask_full = mask_upscaled.squeeze().numpy() > 0.5

    assert mask_full.shape == (30, 30)
    assert mask_full.sum() > 0

    # Verify it can be stored as a SegmentationMask
    seg = sio.SegmentationMask.from_numpy(mask_full)
    assert seg.height == 30
    assert seg.width == 30
    np.testing.assert_array_equal(seg.data, mask_full)


def test_make_labeled_frames_from_generator():
    """_make_labeled_frames_from_generator produces sio.Labels with masks."""
    import sleap_io as sio

    video = sio.Video(filename="test.mp4")

    predictor = BottomUpSegmentationPredictor(
        output_stride=2,
        batch_size=1,
    )
    predictor.videos = [video]

    # Create a mock generator output
    mask1 = np.zeros((16, 16), dtype=bool)
    mask1[0:8, 0:8] = True
    mask2 = np.zeros((16, 16), dtype=bool)
    mask2[8:16, 8:16] = True

    def mock_generator():
        yield {
            "video_idx": 0,
            "frame_idx": 5,
            "orig_size": np.array([32.0, 32.0]),
            "instances": [
                {"mask": mask1, "center": (8.0, 8.0), "score": 0.9},
                {"mask": mask2, "center": (24.0, 24.0), "score": 0.7},
            ],
            "padded_size": (32, 32),
        }

    labels = predictor._make_labeled_frames_from_generator(mock_generator())

    assert isinstance(labels, sio.Labels)
    assert len(labels.masks) == 2
    assert labels.videos == [video]

    # Check mask properties
    for mask_obj in labels.masks:
        assert isinstance(mask_obj, sio.SegmentationMask)
        assert mask_obj.video == video
        assert mask_obj.frame_idx == 5
        assert mask_obj.height == 32
        assert mask_obj.width == 32
        assert mask_obj.data.shape == (32, 32)

    # Check scores preserved
    scores = sorted([m.score for m in labels.masks], reverse=True)
    assert scores == [0.9, 0.7]


def test_make_labeled_frames_empty_instances():
    """_make_labeled_frames_from_generator handles frames with no instances."""
    import sleap_io as sio

    video = sio.Video(filename="test.mp4")
    predictor = BottomUpSegmentationPredictor(output_stride=2)
    predictor.videos = [video]

    def mock_generator():
        yield {
            "video_idx": 0,
            "frame_idx": 0,
            "orig_size": np.array([32.0, 32.0]),
            "instances": [],
            "padded_size": (32, 32),
        }

    labels = predictor._make_labeled_frames_from_generator(mock_generator())
    assert len(labels.masks) == 0
