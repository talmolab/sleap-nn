"""Tests for the bottom-up instance segmentation model type (training stack).

Covers ground-truth tensor generation, head modules, loss functions, model
forward, instance grouping, config, the dataset, the sleap-io mask I/O
contract, and an end-to-end training smoke test. Inference-pipeline (Predictor)
roundtrip tests live in ``tests/inference/test_segmentation_inference.py``.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import sleap_io as sio

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
    """Two non-overlapping rectangular masks produce correct shape and counts."""
    h, w = 64, 64
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[0:10, 0:10] = True
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[50:60, 50:60] = True

    fg = generate_foreground_mask([mask_a, mask_b], (h, w), output_stride=1)

    assert fg.shape == (1, 1, h, w)
    assert fg.dtype == torch.float32
    assert fg.sum().item() == 200.0


def test_generate_foreground_mask_empty():
    """Empty mask list returns all zeros."""
    fg = generate_foreground_mask([], (32, 32), output_stride=1)
    assert fg.shape == (1, 1, 32, 32)
    assert fg.sum().item() == 0.0


def test_generate_foreground_mask_overlapping():
    """Two overlapping masks produce the correct union count."""
    h, w = 64, 64
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[0:20, 0:20] = True
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[10:30, 10:30] = True

    fg = generate_foreground_mask([mask_a, mask_b], (h, w), output_stride=1)
    # Union = 400 + 400 - overlap(100) = 700
    assert fg.sum().item() == 700.0


def test_generate_foreground_mask_downsampled():
    """Output stride downsamples the foreground map."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[16:48, 16:48] = True
    fg = generate_foreground_mask([mask], (h, w), output_stride=2)
    assert fg.shape == (1, 1, 32, 32)
    assert fg.sum().item() > 0


def test_generate_center_heatmap():
    """Verify shape, peak near centroid, and max value close to 1."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:40] = True

    heatmap = generate_center_heatmap([mask], (h, w), output_stride=1, sigma=5.0)

    assert heatmap.shape == (1, 1, h, w)
    assert heatmap.max().item() > 0.95
    peak_idx = heatmap[0, 0].argmax().item()
    assert abs(peak_idx // w - 29.5) <= 1.0
    assert abs(peak_idx % w - 29.5) <= 1.0


def test_generate_center_heatmap_empty():
    """Empty mask list returns zeros."""
    heatmap = generate_center_heatmap([], (32, 32), output_stride=1, sigma=5.0)
    assert heatmap.shape == (1, 1, 32, 32)
    assert heatmap.sum().item() == 0.0


def test_generate_center_offsets():
    """Offsets point toward centroid; weight mask matches foreground."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:40] = True

    offsets, weight_mask = generate_center_offsets([mask], (h, w), output_stride=1)

    assert offsets.shape == (1, 2, h, w)
    assert weight_mask.shape == (1, 1, h, w)
    expected_fg = generate_foreground_mask([mask], (h, w), output_stride=1)
    assert (weight_mask == expected_fg).all()
    # Pixel (20,20) -> coords (20.5, 20.5); centroid ~ (29.5, 29.5) => dx=dy=9
    assert abs(offsets[0, 0, 20, 20].item() - 9.0) < 0.5
    assert abs(offsets[0, 1, 20, 20].item() - 9.0) < 0.5


def test_generate_center_offsets_empty():
    """Empty mask list returns zeros for offsets and weight mask."""
    offsets, weight_mask = generate_center_offsets([], (32, 32), output_stride=1)
    assert offsets.sum().item() == 0.0
    assert weight_mask.sum().item() == 0.0


def test_compute_mask_centroids():
    """Known rectangles produce known centroids."""
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_a[0:10, 0:10] = True
    mask_b = np.zeros((64, 64), dtype=bool)
    mask_b[20:30, 40:50] = True

    centers = _compute_mask_centroids([mask_a, mask_b])
    assert len(centers) == 2
    assert abs(centers[0][0] - 4.5) < 0.1 and abs(centers[0][1] - 4.5) < 0.1
    assert abs(centers[1][0] - 44.5) < 0.1 and abs(centers[1][1] - 24.5) < 0.1


def test_compute_mask_centroids_empty_mask():
    """Empty mask falls back to image center."""
    centers = _compute_mask_centroids([np.zeros((64, 80), dtype=bool)])
    assert abs(centers[0][0] - 40.0) < 0.1 and abs(centers[0][1] - 32.0) < 0.1


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
    """SegmentationHead: channels=1, identity activation, bce_dice loss."""
    head = SegmentationHead(output_stride=2, loss_weight=1.0)
    assert head.channels == 1
    assert head.activation == "identity"
    assert head.loss_function == "bce_dice"
    assert head.output_stride == 2 and head.loss_weight == 1.0
    assert isinstance(head, Head) and head.name == "SegmentationHead"


def test_instance_center_head():
    """InstanceCenterHead: channels=1, stores sigma, mse loss."""
    head = InstanceCenterHead(sigma=10.0, output_stride=2, loss_weight=1.0)
    assert head.channels == 1 and head.sigma == 10.0
    assert head.activation == "identity" and head.loss_function == "mse"
    assert head.name == "InstanceCenterHead"


def test_center_offset_head():
    """CenterOffsetHead: channels=2, smooth_l1 loss."""
    head = CenterOffsetHead(output_stride=2, loss_weight=0.1)
    assert head.channels == 2 and head.loss_function == "smooth_l1"
    assert head.name == "CenterOffsetHead"


@pytest.mark.parametrize(
    "head_cls,kwargs,out_ch",
    [
        (SegmentationHead, dict(output_stride=2, loss_weight=1.0), 1),
        (InstanceCenterHead, dict(sigma=10.0, output_stride=2), 1),
        (CenterOffsetHead, dict(output_stride=2, loss_weight=0.1), 2),
    ],
)
def test_head_make_head_forward(head_cls, kwargs, out_ch):
    """Each head's make_head produces the expected channel count."""
    sample_input = torch.randn(2, 64, 32, 32)
    head = head_cls(**kwargs)
    module = head.make_head(x_in=sample_input.size(1))
    output = module(sample_input)
    assert output.shape == (2, out_ch, 32, 32)
    assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# 3. Loss functions (sleap_nn/training/losses.py)
# ---------------------------------------------------------------------------

from sleap_nn.training.losses import compute_bce_dice_loss, compute_masked_smooth_l1


def test_bce_dice_loss_perfect():
    """Identical pred (logits) and gt give ~0 loss."""
    gt = torch.ones(1, 1, 16, 16)
    pred = torch.full((1, 1, 16, 16), 10.0)
    assert compute_bce_dice_loss(pred, gt).item() < 0.01


def test_bce_dice_loss_worst():
    """Confidently wrong pred gives high loss."""
    gt = torch.ones(1, 1, 16, 16)
    pred = torch.full((1, 1, 16, 16), -10.0)
    assert compute_bce_dice_loss(pred, gt).item() > 1.0


def test_bce_dice_loss_gradient():
    """Loss supports backpropagation."""
    pred = torch.full((1, 1, 8, 8), 0.0, requires_grad=True)
    gt = torch.ones(1, 1, 8, 8)
    compute_bce_dice_loss(pred, gt).backward()
    assert pred.grad is not None and pred.grad.shape == pred.shape


def test_masked_smooth_l1_basic():
    """Masked offset loss matches the analytic value on the masked region."""
    pred = torch.ones(1, 2, 4, 4)
    gt = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, :2, :2] = 1.0
    # smooth_l1(1,0)=0.5; mean over 8 valid elements = 0.5
    assert abs(compute_masked_smooth_l1(pred, gt, mask).item() - 0.5) < 0.01


def test_masked_smooth_l1_empty_mask():
    """Empty mask returns differentiable 0 loss."""
    pred = torch.ones(1, 2, 4, 4, requires_grad=True)
    gt = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    loss = compute_masked_smooth_l1(pred, gt, mask)
    assert loss.item() == 0.0 and loss.requires_grad


# ---------------------------------------------------------------------------
# 4. Model forward
# ---------------------------------------------------------------------------

from sleap_nn.architectures.model import Model


def _seg_backbone_head_configs():
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
            "segmentation": {"output_stride": 2, "loss_weight": 1.0},
            "center": {"sigma": 10.0, "output_stride": 2, "loss_weight": 1.0},
            "offsets": {"output_stride": 2, "loss_weight": 0.1},
        }
    )
    return backbone_config, head_configs


def test_bottomup_segmentation_model_forward():
    """Model forward returns the three head outputs at the right shapes."""
    backbone_config, head_configs = _seg_backbone_head_configs()
    model = Model(
        backbone_type="unet",
        backbone_config=backbone_config,
        head_configs=head_configs,
        model_type="bottomup_segmentation",
    )
    x = torch.randn(2, 1, 64, 64)
    outputs = model(x)
    assert outputs["SegmentationHead"].shape == (2, 1, 32, 32)
    assert outputs["InstanceCenterHead"].shape == (2, 1, 32, 32)
    assert outputs["CenterOffsetHead"].shape == (2, 2, 32, 32)


def test_bottomup_segmentation_model_three_heads():
    """Model has exactly three heads of the right types."""
    backbone_config, head_configs = _seg_backbone_head_configs()
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


def _make_blob_offsets(foreground, centers_rc, output_stride):
    """Build offsets pointing each fg pixel to the nearest given center."""
    offsets = torch.zeros(1, 2, *foreground.shape[-2:])
    fg = foreground[0, 0] > 0.5
    ys, xs = torch.nonzero(fg, as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        # nearest center (in grid coords)
        cr, cc = min(centers_rc, key=lambda c: (c[0] - y) ** 2 + (c[1] - x) ** 2)
        px = x * output_stride + output_stride / 2.0
        py = y * output_stride + output_stride / 2.0
        cx = cc * output_stride + output_stride / 2.0
        cy = cr * output_stride + output_stride / 2.0
        offsets[0, 0, y, x] = cx - px
        offsets[0, 1, y, x] = cy - py
    return offsets


def test_group_instances_two_blobs():
    """Two well-separated blobs with correct offsets recover 2 instances."""
    h, w = 32, 32
    s = 2
    foreground = torch.zeros(1, 1, h, w)
    foreground[0, 0, 2:8, 2:8] = 1.0
    foreground[0, 0, 22:28, 22:28] = 1.0
    center_heatmap = torch.zeros(1, 1, h, w)
    center_heatmap[0, 0, 5, 5] = 1.0
    center_heatmap[0, 0, 25, 25] = 1.0
    offsets = _make_blob_offsets(foreground, [(5, 5), (25, 25)], s)

    instances = group_instances_from_offsets(
        foreground,
        center_heatmap,
        offsets,
        fg_threshold=0.5,
        peak_threshold=0.5,
        output_stride=s,
    )
    assert len(instances) == 2
    for inst in instances:
        assert set(inst) == {"mask", "center", "score"}
        assert inst["mask"].shape == (h, w)
        assert inst["score"] >= 0.5
    assert sorted(int(i["mask"].sum()) for i in instances) == [36, 36]


def test_group_instances_no_foreground():
    """Empty foreground returns empty list."""
    h, w = 16, 16
    instances = group_instances_from_offsets(
        torch.zeros(1, 1, h, w),
        torch.zeros(1, 1, h, w),
        torch.zeros(1, 2, h, w),
        fg_threshold=0.5,
        peak_threshold=0.2,
        output_stride=2,
    )
    assert instances == []


def test_group_instances_no_centers():
    """Foreground present but no center peaks returns empty list."""
    h, w = 16, 16
    instances = group_instances_from_offsets(
        torch.ones(1, 1, h, w),
        torch.full((1, 1, h, w), 0.01),
        torch.zeros(1, 2, h, w),
        fg_threshold=0.5,
        peak_threshold=0.5,
        output_stride=2,
    )
    assert instances == []


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


def test_segmentation_config_defaults():
    """Sub-config defaults match the head defaults."""
    assert SegmentationHeadConfig().output_stride == 2
    assert InstanceCenterConfig().sigma == 10.0
    assert CenterOffsetConfig().loss_weight == 0.1


def test_head_config_oneof_bottomup_segmentation():
    """HeadConfig oneof selects bottomup_segmentation exclusively."""
    head_cfg = HeadConfig(
        bottomup_segmentation=BottomUpSegmentationConfig(
            segmentation=SegmentationHeadConfig(),
            center=InstanceCenterConfig(),
            offsets=CenterOffsetConfig(),
        )
    )
    assert head_cfg.bottomup_segmentation is not None
    assert head_cfg.single_instance is None
    assert head_cfg.bottomup is None
    assert head_cfg.multi_class_topdown is None


def test_get_head_configs_builder_string():
    """get_head_configs builds a bottomup_segmentation HeadConfig from a string."""
    from sleap_nn.config.get_config import get_head_configs

    hc = get_head_configs("bottomup_segmentation")
    assert hc.bottomup_segmentation is not None
    assert hc.bottomup_segmentation.segmentation.output_stride == 2


def test_get_model_type_from_cfg_segmentation():
    """Model type is auto-detected from the head config key."""
    from sleap_nn.config.utils import get_model_type_from_cfg

    cfg = OmegaConf.create(
        {
            "model_config": {
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "centered_instance": None,
                    "bottomup": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None,
                    "bottomup_segmentation": {
                        "segmentation": {"output_stride": 2, "loss_weight": 1.0},
                        "center": {
                            "sigma": 5.0,
                            "output_stride": 2,
                            "loss_weight": 1.0,
                        },
                        "offsets": {"output_stride": 2, "loss_weight": 0.1},
                    },
                }
            }
        }
    )
    assert get_model_type_from_cfg(cfg) == "bottomup_segmentation"


# ---------------------------------------------------------------------------
# 7. sleap-io mask I/O contract
# ---------------------------------------------------------------------------


def test_seg_mask_slp_roundtrip(minimal_instance_seg):
    """Synthetic seg labels roundtrip through .slp preserving mask data."""
    labels = sio.load_slp(minimal_instance_seg.as_posix())
    assert len(labels.masks) >= 1
    for m in labels.masks:
        assert isinstance(m, sio.SegmentationMask)
        assert m.data.dtype == bool
        assert m.data.any()


# ---------------------------------------------------------------------------
# 8. Dataset (Part A: GT integration, no model)
# ---------------------------------------------------------------------------

from sleap_nn.data.custom_datasets import BottomUpSegmentationDataset


@pytest.mark.parametrize("cache_img", [None, "memory"])
def test_bottomup_segmentation_dataset(minimal_instance_seg, cache_img, tmp_path):
    """Dataset yields GT tensors at the output stride for both cache paths."""
    seg_labels = sio.load_slp(minimal_instance_seg.as_posix())
    seg_cfg = OmegaConf.create({"output_stride": 2, "loss_weight": 1.0})
    center_cfg = OmegaConf.create(
        {"sigma": 5.0, "output_stride": 2, "loss_weight": 1.0}
    )
    offset_cfg = OmegaConf.create({"output_stride": 2, "loss_weight": 0.1})

    ds = BottomUpSegmentationDataset(
        labels=[seg_labels],
        seg_head_config=seg_cfg,
        center_head_config=center_cfg,
        offset_head_config=offset_cfg,
        max_stride=16,
        ensure_grayscale=True,
        cache_img=cache_img,
        cache_img_path=str(tmp_path) if cache_img == "disk" else None,
    )
    assert len(ds) == 1
    s = ds[0]
    h, w = s["image"].shape[-2:]
    oh, ow = h // 2, w // 2
    assert s["foreground_mask"].shape == (1, 1, oh, ow)
    assert s["center_heatmap"].shape == (1, 1, oh, ow)
    assert s["center_offsets"].shape == (1, 2, oh, ow)
    assert s["foreground_weight"].shape == (1, 1, oh, ow)
    assert s["foreground_mask"].sum() > 0
    assert s["center_heatmap"].max() > 0.9
    # offsets only defined on foreground (weight) pixels
    assert (s["foreground_weight"] == (s["foreground_mask"] > 0).float()).all()


# ---------------------------------------------------------------------------
# 9. End-to-end training smoke test
# ---------------------------------------------------------------------------


def _seg_train_config(seg_path, tmp_path):
    return OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": [seg_path],
                "val_labels_path": [seg_path],
                "validation_fraction": 0.1,
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "preprocessing": {
                    "ensure_rgb": False,
                    "ensure_grayscale": True,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "crop_size": None,
                    "min_crop_size": None,
                },
                "use_augmentations_train": False,
                "augmentation_config": None,
            },
            "model_config": {
                "init_weights": "default",
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 8,
                        "filters_rate": 1.5,
                        "max_stride": 16,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": True,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None,
                    "bottomup_segmentation": {
                        "segmentation": {"output_stride": 2, "loss_weight": 1.0},
                        "center": {
                            "sigma": 5.0,
                            "output_stride": 2,
                            "loss_weight": 1.0,
                        },
                        "offsets": {"output_stride": 2, "loss_weight": 0.1},
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {"batch_size": 1, "num_workers": 0},
                "model_ckpt": {"save_top_k": 1, "save_last": True},
                "early_stopping": {
                    "stop_training_on_plateau": False,
                    "min_delta": 1e-08,
                    "patience": 20,
                },
                "trainer_devices": 1,
                "trainer_device_indices": None,
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 1,
                "train_steps_per_epoch": None,
                "max_epochs": 1,
                "seed": 1000,
                "keep_viz": False,
                "use_wandb": False,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path),
                "run_name": "seg_smoke",
                "resume_ckpt_path": None,
                "optimizer_name": "Adam",
                "optimizer": {"lr": 1e-3, "amsgrad": False},
                "lr_scheduler": {
                    "reduce_lr_on_plateau": {
                        "threshold": 1e-07,
                        "threshold_mode": "rel",
                        "cooldown": 3,
                        "patience": 5,
                        "factor": 0.5,
                        "min_lr": 1e-08,
                    }
                },
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 2,
                    "max_hard_keypoints": None,
                    "loss_scale": 5.0,
                },
                "eval": {"enabled": False},
            },
        }
    )


def test_bottomup_segmentation_train_smoke(minimal_instance_seg, tmp_path):
    """ModelTrainer trains a segmentation model for one epoch and writes a ckpt."""
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _seg_train_config(minimal_instance_seg.as_posix(), tmp_path)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    run_dir = tmp_path / "seg_smoke"
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "training_config.yaml").exists()
    # CSV logged the segmentation-specific metrics.
    csv_text = (run_dir / "training_log.csv").read_text()
    assert "val/fg_iou" in csv_text
    assert "train/fg_loss" in csv_text
