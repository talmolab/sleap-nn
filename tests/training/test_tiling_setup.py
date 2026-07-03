"""Tests for ``ModelTrainer._setup_tiling_config`` / ``_get_confmap_sigma``.

Exercises the auto-sizing of tiling geometry at train setup on a tiny
single-instance ``.slp`` fixture: the sparse-labels warning, write-back of
divisible geometry, and preservation of explicit user values.
"""

import sleap_io as sio
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.training.model_trainer import ModelTrainer, _SPARSE_LABEL_THRESHOLD


def _capture_tiling_setup(trainer):
    """Re-run ``_setup_tiling_config`` with tiling reset and capture its logs.

    The trainer build reconfigures loguru's sinks (removing the pytest ``caplog``
    shim), so we attach a fresh sink around a direct call instead. Resetting
    ``tile_size`` / ``overlap`` to ``None`` re-triggers the auto-sizing path.
    """
    trainer.config.data_config.preprocessing.tiling.tile_size = None
    trainer.config.data_config.preprocessing.tiling.overlap = None
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        trainer._setup_tiling_config()
    finally:
        logger.remove(sink_id)
    return "".join(messages)


def _single_instance_labels(minimal_instance):
    """Load the fly fixture and strip to one instance per frame (single-instance)."""
    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = [lf.instances[0]]
    return labels


def _tiled_single_instance_config(config, tmp_path, tiling):
    """Turn the shared centered-instance ``config`` fixture into a tiled single-instance one."""
    cfg = config.copy()
    head_config = cfg.model_config.head_configs.centered_instance
    del cfg.model_config.head_configs.centered_instance
    OmegaConf.update(cfg, "model_config.head_configs.single_instance", head_config)
    del cfg.model_config.head_configs.single_instance.confmaps.anchor_part

    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", "test_tiling_setup")
    OmegaConf.update(cfg, "trainer_config.save_ckpt", False)
    OmegaConf.update(cfg, "trainer_config.train_data_loader.num_workers", 0)
    OmegaConf.update(cfg, "trainer_config.val_data_loader.num_workers", 0)
    # Merge the partial tiling block; verify_training_cfg fills in defaults.
    OmegaConf.update(cfg, "data_config.preprocessing.tiling", tiling, force_add=True)
    return cfg


def test_tiling_auto_sized_and_sparse_warning(config, tmp_path, minimal_instance):
    """Auto-sized geometry is written back and sparse labels warn."""
    cfg = _tiled_single_instance_config(
        config, tmp_path, {"enabled": True, "tile_size": None, "overlap": None}
    )
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    tiling = trainer.config.data_config.preprocessing.tiling
    # tile_size / overlap were auto-sized (no longer None).
    assert tiling.tile_size is not None
    assert tiling.overlap is not None

    # Backbone max_stride=8, head output_stride=2 -> divisor lcm(8, 2) = 8.
    assert tiling.tile_size % 8 == 0
    assert tiling.overlap % 2 == 0
    assert 0 <= tiling.overlap < tiling.tile_size

    # The fly fixture has < _SPARSE_LABEL_THRESHOLD instances -> conservative warn.
    captured = _capture_tiling_setup(trainer)
    assert f"< {_SPARSE_LABEL_THRESHOLD}" in captured
    # Re-run wrote a conservative overlap back (min_overlap_fraction floor).
    assert trainer.config.data_config.preprocessing.tiling.overlap is not None


def test_tiling_explicit_values_preserved(config, tmp_path, minimal_instance):
    """Explicit tile_size / overlap are preserved (not overwritten)."""
    # divisor lcm(8, 2) = 8; 64 % 8 == 0; overlap 16 % 2 == 0 and >= 0.25*64 floor.
    cfg = _tiled_single_instance_config(
        config, tmp_path, {"enabled": True, "tile_size": 64, "overlap": 16}
    )
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    tiling = trainer.config.data_config.preprocessing.tiling
    assert tiling.tile_size == 64
    assert tiling.overlap == 16


def test_get_confmap_sigma_reads_head_sigma(config, tmp_path, minimal_instance):
    """`_get_confmap_sigma` returns the active head's confmap sigma."""
    cfg = _tiled_single_instance_config(
        config, tmp_path, {"enabled": True, "tile_size": 64, "overlap": 16}
    )
    # The shared fixture uses confmaps.sigma == 1.5.
    labels = _single_instance_labels(minimal_instance)

    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    assert trainer._get_confmap_sigma(output_stride=2) == 1.5
