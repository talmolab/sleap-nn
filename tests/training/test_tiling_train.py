"""Real smoke tests for the Phase-A tiled single-instance training path.

Exercises the full wiring end to end on the ``minimal_instance`` fixture: the
tiled dataset/factory/sampler, per-tile batch shapes, and one training step
via ``ModelTrainer``. Also pins the non-tiled single-instance regression and
the multi-instance warning.
"""

import numpy as np
import sleap_io as sio
import torch
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.data.custom_datasets import (
    SingleInstanceTiledDataset,
    get_train_val_dataloaders,
    get_train_val_datasets,
)
from sleap_nn.training.model_trainer import ModelTrainer


def _single_instance_labels(minimal_instance):
    """Load the fly fixture and strip to one instance per frame."""
    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = [lf.instances[0]]
    return labels


def _single_instance_config(config, tmp_path, run_name):
    """Turn the shared centered-instance ``config`` fixture into single-instance."""
    cfg = config.copy()
    head_config = cfg.model_config.head_configs.centered_instance
    del cfg.model_config.head_configs.centered_instance
    OmegaConf.update(cfg, "model_config.head_configs.single_instance", head_config)
    del cfg.model_config.head_configs.single_instance.confmaps.anchor_part

    if torch.backends.mps.is_available():
        OmegaConf.update(cfg, "trainer_config.trainer_accelerator", "cpu")
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", run_name)
    OmegaConf.update(cfg, "trainer_config.save_ckpt", False)
    OmegaConf.update(cfg, "trainer_config.max_epochs", 1)
    OmegaConf.update(cfg, "trainer_config.visualize_preds_during_training", False)
    OmegaConf.update(cfg, "trainer_config.train_data_loader.num_workers", 0)
    OmegaConf.update(cfg, "trainer_config.val_data_loader.num_workers", 0)
    return cfg


def _tiled_config(config, tmp_path, run_name):
    """Single-instance config with a small explicit tiling geometry."""
    cfg = _single_instance_config(config, tmp_path, run_name)
    # tile_size 128: divisible by max_stride=8 and output_stride=2.
    # overlap 32: divisible by 2 and >= 0.25 * 128 floor.
    tiling = {
        "enabled": True,
        "tile_size": 128,
        "overlap": 32,
        "samples_per_frame": 2,
    }
    OmegaConf.update(cfg, "data_config.preprocessing.tiling", tiling, force_add=True)
    return cfg


def test_tiling_single_instance_batch_shapes(config, tmp_path, minimal_instance):
    """One tiled batch has per-tile image and confmap shapes."""
    cfg = _tiled_config(config, tmp_path, "tiling_batch_shapes")
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    # Geometry was auto-validated + preserved (explicit values).
    tiling = trainer.config.data_config.preprocessing.tiling
    assert tiling.tile_size == 128
    assert tiling.overlap == 32
    assert tiling.samples_per_frame == 2  # explicit value preserved

    train_ds, val_ds = get_train_val_datasets(
        train_labels=trainer.train_labels,
        val_labels=trainer.val_labels,
        config=trainer.config,
    )
    assert isinstance(train_ds, SingleInstanceTiledDataset)
    # samples_per_frame tiles per (single) frame for train.
    assert len(train_ds) == 2
    assert train_ds.tile_sampling == "foreground"
    # Val always uses a full-coverage grid.
    assert val_ds.tile_sampling == "grid"
    assert len(val_ds) == len(val_ds.frame_blocks[0]) > 1

    train_dl, val_dl = get_train_val_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=trainer.config,
        rank=0,
        trainer_devices=1,
    )

    batch = next(iter(train_dl))
    bs = trainer.config.trainer_config.train_data_loader.batch_size
    # image: (B, 1, C, tile, tile); C == 1 for the grayscale fly fixture.
    assert batch["image"].shape == (bs, 1, 1, 128, 128)
    # confmaps: (B, 1, N, tile/output_stride, tile/output_stride).
    assert batch["confidence_maps"].shape == (bs, 1, 2, 64, 64)
    assert batch["tile_origin"].shape == (bs, 2)

    # Val batch is also per-tile.
    vbatch = next(iter(val_dl))
    assert vbatch["image"].shape[-2:] == (128, 128)


def test_tiling_single_instance_trains_one_step(config, tmp_path, minimal_instance):
    """A tiled single-instance model trains for one epoch without error."""
    cfg = _tiled_config(config, tmp_path, "tiling_trains")
    OmegaConf.update(cfg, "trainer_config.min_train_steps_per_epoch", 1)
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()

    from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

    assert isinstance(trainer.lightning_model, SingleInstanceLightningModule)
    # The tiling epoch callback was registered.
    from sleap_nn.training.callbacks import TilingEpochCallback

    assert any(isinstance(cb, TilingEpochCallback) for cb in trainer.trainer.callbacks)


def test_non_tiled_single_instance_still_trains(config, tmp_path, minimal_instance):
    """Regression: the non-tiled single-instance path still trains one step."""
    cfg = _single_instance_config(config, tmp_path, "non_tiled_trains")
    OmegaConf.update(cfg, "trainer_config.min_train_steps_per_epoch", 1)
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )
    # Tiling stays inert.
    tiling = OmegaConf.select(
        trainer.config, "data_config.preprocessing.tiling", default=None
    )
    assert tiling is None or not tiling.enabled

    trainer.train()

    from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

    assert isinstance(trainer.lightning_model, SingleInstanceLightningModule)


def test_multi_instance_warning_fires(minimal_instance):
    """Multi-instance labels trigger the one-time single-instance warning."""
    labels = sio.load_slp(minimal_instance)  # keeps 2 instances per frame
    confmap = OmegaConf.create(
        {"part_names": ["A", "B"], "sigma": 1.5, "output_stride": 2}
    )
    tiling = OmegaConf.create(
        {
            "enabled": True,
            "tile_size": 128,
            "overlap": 32,
            "min_overlap_fraction": 0.25,
            "sampling": "foreground",
            "tile_fg_fraction": 0.5,
            "samples_per_frame": 2,
            "center_jitter": 0.5,
            "min_visible_keypoints": 1,
        }
    )

    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        ds = SingleInstanceTiledDataset(
            labels=[labels],
            confmap_head_config=confmap,
            max_stride=8,
            scale=1.0,
            apply_aug=False,
            tiling=tiling,
            base_seed=0,
        )
    finally:
        logger.remove(sink_id)

    assert ds.max_instances > 1
    assert any("more than one" in m for m in messages)
