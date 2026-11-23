"""Run a training dry run in a test and check that it leaves nothing behind."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from loguru import logger


def dry_run(config, tmp_path: Path, train_labels=None, val_labels=None):
    """Dry-run ``config`` on the CPU, as ``sleap-nn train --dry-run`` does.

    The run is pointed at ``tmp_path / "runs" / "dry"`` and saves checkpoints, so
    the write check covers the checkpoint too. Asserts that the run got to its
    summary ("Dry run passed" is logged, after one training step) and that
    ``tmp_path`` holds exactly the files it held before: no run folder, no
    ``.dryrun-*`` scratch folder and, for a config whose ``cache_img_path`` is
    inside ``tmp_path``, no image cache.

    Args:
        config: The training config (modified in place).
        tmp_path: The test's scratch folder.
        train_labels: Training labels, to use instead of the config's files.
        val_labels: Validation labels, to use instead of the config's files.

    Returns:
        The run's ``ModelTrainer``.
    """
    from sleap_nn.train import run_training
    from sleap_nn.training.model_trainer import ModelTrainer

    config.trainer_config.trainer_accelerator = "cpu"
    config.trainer_config.ckpt_dir = (tmp_path / "runs").as_posix()
    config.trainer_config.run_name = "dry"
    config.trainer_config.save_ckpt = True
    before = sorted(tmp_path.rglob("*"))
    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        # The real summary still runs; the spy only hands back the trainer.
        with patch.object(
            ModelTrainer,
            "_log_dry_run_summary",
            autospec=True,
            side_effect=ModelTrainer._log_dry_run_summary,
        ) as summary:
            run_training(
                config, train_labels=train_labels, val_labels=val_labels, dry_run=True
            )
    finally:
        logger.remove(sink)
    summary.assert_called_once()
    trainer = summary.call_args.args[0]
    assert "Dry run passed" in "".join(messages)
    assert trainer.trainer.global_step == 1
    assert sorted(tmp_path.rglob("*")) == before
    return trainer
