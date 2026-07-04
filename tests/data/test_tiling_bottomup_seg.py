"""Real-data tests for tiled bottom-up segmentation training (Phase C).

Exercises :class:`BottomUpSegmentationTiledDataset` on a small subset of the plant
root segmentation dataset (1080x2048 grayscale frames with per-instance masks):

  - a train (foreground) tile sample matches the ``BottomUpSegmentationDataset``
    key/shape/dtype contract (image ``(1, 1, 512, 512)``; GT at output stride 2 =>
    ``(..., 256, 256)``; no NaNs);
  - the ownership filter: a tile that owns a root yields a non-empty foreground mask
    with a center-heatmap peak, an all-background tile yields empty foreground + a
    zero center heatmap, and foreground <=> heatmap (fg only comes from owned masks);
  - ``len(dataset) == n_mask_frames * samples_per_frame`` (train) with per-frame blocks;
  - the geometric (rotation) halo-aug path runs and keeps masks binary;
  - ``check_tiling`` rejects an unsupported model type + enabled, and passes for
    ``bottomup_segmentation``;
  - a 1-step end-to-end training smoke through ``ModelTrainer`` (tiling on).

These are gated on the fixture ``.pkg.slp`` being present; they skip otherwise.
"""

from pathlib import Path

import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.config.utils import check_tiling
from sleap_nn.data.custom_datasets import (
    BottomUpSegmentationDataset,
    BottomUpSegmentationTiledDataset,
)

# Real fixture: plant primary-root segmentation labels (embedded images + masks).
_SEG_SLP = Path("scratch/2026-07-01-plant-seg/masks/primary_train.pkg.slp")
_N_FRAMES = 6
_TILE = 512
_OS = 2  # output stride for all three heads
_OUT = _TILE // _OS  # 256

pytestmark = pytest.mark.skipif(
    not _SEG_SLP.exists(), reason=f"segmentation fixture not found: {_SEG_SLP}"
)


@pytest.fixture(scope="module")
def seg_labels():
    """A small (N-frame) subset of the plant-root seg labels (loaded once)."""
    full = sio.load_slp(_SEG_SLP.as_posix())
    return sio.Labels(
        labeled_frames=full.labeled_frames[:_N_FRAMES],
        videos=full.videos,
        skeletons=full.skeletons,
    )


def _head_configs():
    """Segmentation / center / offset head configs (all at output stride 2)."""
    seg_cfg = OmegaConf.create({"output_stride": _OS, "loss_weight": 1.0})
    center_cfg = OmegaConf.create(
        {"sigma": 8.0, "output_stride": _OS, "loss_weight": 1.0}
    )
    offset_cfg = OmegaConf.create({"output_stride": _OS, "loss_weight": 0.1})
    return seg_cfg, center_cfg, offset_cfg


def _tiling(sampling="foreground", **overrides):
    """A TilingConfig-like OmegaConf for the tiled seg dataset."""
    cfg = dict(
        enabled=True,
        tile_size=_TILE,
        overlap=256,
        sampling=sampling,
        samples_per_frame=2,
        tile_fg_fraction=0.5,
        min_overlap_fraction=0.25,
        center_jitter=0.0,
        min_visible_keypoints=0,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _make_dataset(
    seg_labels, sampling="foreground", apply_aug=False, geometric_aug=None
):
    seg_cfg, center_cfg, offset_cfg = _head_configs()
    return BottomUpSegmentationTiledDataset(
        labels=[seg_labels],
        seg_head_config=seg_cfg,
        center_head_config=center_cfg,
        offset_head_config=offset_cfg,
        max_stride=16,
        ensure_grayscale=True,
        apply_aug=apply_aug,
        geometric_aug=geometric_aug,
        scale=1.0,
        cache_img=None,
        tiling=_tiling(sampling=sampling),
        base_seed=0,
    )


# ---------------------------------------------------------------------------
# 1. Train tile sample: shape / dtype / key contract
# ---------------------------------------------------------------------------
def test_train_tile_sample_contract(seg_labels):
    """A train tile sample matches the BottomUpSegmentationDataset contract."""
    ds = _make_dataset(seg_labels, sampling="foreground", apply_aug=False)

    # Foreground slots are the trailing tile_fg_fraction of each frame block; with
    # samples_per_frame=2 and tile_fg_fraction=0.5, sample_k=1 (odd flat index) is fg.
    s = ds[1]

    assert s["image"].shape == (1, 1, _TILE, _TILE)
    assert s["image"].dtype == torch.uint8  # unnormalized (model normalizes on GPU)
    assert s["foreground_mask"].shape == (1, 1, _OUT, _OUT)
    assert s["center_heatmap"].shape == (1, 1, _OUT, _OUT)
    assert s["center_offsets"].shape == (1, 2, _OUT, _OUT)
    assert s["foreground_weight"].shape == (1, 1, _OUT, _OUT)
    assert s["tile_origin"].shape == (2,)
    assert s["tile_origin"].dtype == torch.int32

    # No NaNs anywhere in the GT tensors.
    for k in ("image", "foreground_mask", "center_heatmap", "center_offsets"):
        assert not torch.isnan(s[k].float()).any()

    # Foreground mask stays binary; offsets only defined on foreground (weight) pixels.
    assert set(torch.unique(s["foreground_mask"]).tolist()) <= {0.0, 1.0}
    assert (s["foreground_weight"] == (s["foreground_mask"] > 0).float()).all()

    # A foreground slot owns a root -> non-empty foreground + a center peak.
    assert s["foreground_mask"].sum() > 0
    assert s["center_heatmap"].max() > 0.9

    # Key parity with the non-tiled dataset (plus the extra tile_origin).
    nt = BottomUpSegmentationDataset(
        labels=[seg_labels],
        seg_head_config=_head_configs()[0],
        center_head_config=_head_configs()[1],
        offset_head_config=_head_configs()[2],
        max_stride=16,
        ensure_grayscale=True,
        cache_img=None,
    )
    nt_keys = set(nt[0].keys())
    tiled_keys = set(s.keys())
    assert nt_keys <= tiled_keys, nt_keys - tiled_keys
    assert tiled_keys - nt_keys == {"tile_origin"}


# ---------------------------------------------------------------------------
# 2. Ownership filter: root-owning vs background tiles
# ---------------------------------------------------------------------------
def test_ownership_foreground_vs_background(seg_labels):
    """Owned mask <=> foreground + heatmap; a background tile is empty."""
    # A deterministic full-coverage grid gives us both root and background tiles.
    ds = _make_dataset(seg_labels, sampling="grid", apply_aug=False)

    saw_root_tile = False
    saw_background_tile = False
    for i in range(len(ds)):
        s = ds[i]
        fg = float(s["foreground_mask"].sum())
        hm = float(s["center_heatmap"].max())

        # Ownership invariant: fg pixels ONLY come from owned masks, and every owned
        # mask contributes a center-heatmap peak -> a heatmap peak implies fg pixels.
        if hm > 0.5:
            assert fg > 0, f"tile {i}: heatmap peak but empty foreground"

        if fg > 0 and hm > 0.5:
            saw_root_tile = True
        if fg == 0:
            # An all-background tile: empty foreground AND a zero center heatmap.
            assert hm == 0.0, f"tile {i}: empty fg but non-zero heatmap"
            assert float(s["center_offsets"].abs().sum()) == 0.0
            saw_background_tile = True

    assert saw_root_tile, "expected at least one root-owning grid tile"
    assert saw_background_tile, "expected at least one all-background grid tile"


# ---------------------------------------------------------------------------
# 3. Length + frame blocks
# ---------------------------------------------------------------------------
def test_len_and_frame_blocks(seg_labels):
    """Length equals n_mask_frames * samples_per_frame; blocks group per frame."""
    ds = _make_dataset(seg_labels, sampling="foreground", apply_aug=False)
    n_mask_frames = sum(1 for lf in seg_labels if getattr(lf, "masks", None))
    assert n_mask_frames == _N_FRAMES
    assert len(ds) == n_mask_frames * 2  # samples_per_frame == 2

    # Blocks: one contiguous run of samples_per_frame indices per frame.
    assert len(ds.frame_blocks) == n_mask_frames
    assert all(len(b) == 2 for b in ds.frame_blocks)
    # Blocks tile the full flat index range contiguously.
    flat = [idx for b in ds.frame_blocks for idx in b]
    assert flat == list(range(len(ds)))


# ---------------------------------------------------------------------------
# 4. Rotation halo-aug path keeps masks binary
# ---------------------------------------------------------------------------
def test_rotation_aug_keeps_masks_binary(seg_labels):
    """The +/- rotation halo-aug path runs and foreground masks stay binary."""
    geo = {"rotation_min": -15.0, "rotation_max": 15.0, "rotation_p": 1.0}
    ds = _make_dataset(
        seg_labels, sampling="foreground", apply_aug=True, geometric_aug=geo
    )
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (1, 1, _TILE, _TILE)
        assert s["foreground_mask"].shape == (1, 1, _OUT, _OUT)
        # Binary after the nearest-neighbor co-transform + re-binarize.
        assert set(torch.unique(s["foreground_mask"]).tolist()) <= {0.0, 1.0}
        assert not torch.isnan(s["center_offsets"]).any()

    # Aug varies the sample across draws (halo rotation is seed-driven per epoch).
    a = ds[1]["image"].clone()
    ds._epoch.fill_(7)  # bump epoch -> new tile-sample seed
    b = ds[1]["image"].clone()
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# 5. check_tiling model-type allowlist guard
# ---------------------------------------------------------------------------
def _finalized_cfg(head_configs):
    return OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "tiling": {
                        "enabled": True,
                        "tile_size": 64,
                        "overlap": 16,
                        "min_overlap_fraction": 0.25,
                    }
                }
            },
            "model_config": {
                "pretrained_backbone_weights": None,
                "backbone_config": {"unet": {"max_stride": 16, "output_stride": 2}},
                "head_configs": head_configs,
            },
        }
    )


def test_check_tiling_rejects_unsupported_model_type():
    """A centroid model with tiling enabled raises the allowlist guard."""
    cfg = _finalized_cfg({"centroid": {"confmaps": {"output_stride": 2, "sigma": 1.5}}})
    with pytest.raises(ValueError, match="not yet implemented for model_type"):
        check_tiling(cfg)


def test_check_tiling_allows_bottomup_segmentation():
    """bottomup_segmentation is on the allowlist -> check_tiling passes."""
    cfg = _finalized_cfg(
        {
            "bottomup_segmentation": {
                "segmentation": {"output_stride": 2},
                "center": {"output_stride": 2, "sigma": 8.0},
                "offsets": {"output_stride": 2},
            }
        }
    )
    # Should not raise; returns the (in-place reconciled) config.
    assert check_tiling(cfg) is cfg


# ---------------------------------------------------------------------------
# 6. End-to-end 1-step training smoke (tiling on)
# ---------------------------------------------------------------------------
def _seg_tiling_train_config(seg_path, tmp_path):
    """One-epoch, 1-step CPU training config for tiled bottom-up segmentation."""
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
                    "tiling": {
                        "enabled": True,
                        "tile_size": _TILE,
                        "overlap": 256,
                        "sampling": "foreground",
                        "samples_per_frame": 2,
                        "tile_fg_fraction": 0.5,
                        "min_overlap_fraction": 0.25,
                        "center_jitter": 0.0,
                        "min_visible_keypoints": 0,
                    },
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": None,
                    "geometric": {
                        "rotation_min": -15.0,
                        "rotation_max": 15.0,
                        "rotation_p": 1.0,
                    },
                },
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
                        "output_stride": _OS,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None,
                    "centered_instance_segmentation": None,
                    "bottomup_segmentation": {
                        "segmentation": {"output_stride": _OS, "loss_weight": 1.0},
                        "center": {
                            "sigma": 8.0,
                            "output_stride": _OS,
                            "loss_weight": 1.0,
                        },
                        "offsets": {"output_stride": _OS, "loss_weight": 0.1},
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 2,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {"batch_size": 2, "num_workers": 0},
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
                "train_steps_per_epoch": 1,
                "max_epochs": 1,
                "seed": 1000,
                "keep_viz": False,
                "use_wandb": False,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path),
                "run_name": "seg_tiling_smoke",
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


def test_bottomup_seg_tiling_train_smoke(seg_labels, tmp_path):
    """A tiled bottom-up seg model trains one step end-to-end and writes a ckpt."""
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _seg_tiling_train_config(_SEG_SLP.as_posix(), tmp_path)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[seg_labels], val_labels=[seg_labels]
    )
    trainer.train()

    run_dir = tmp_path / "seg_tiling_smoke"
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "training_config.yaml").exists()
    # The tiling geometry survived setup + was persisted.
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    assert saved.data_config.preprocessing.tiling.enabled is True
    assert saved.data_config.preprocessing.tiling.tile_size == _TILE
