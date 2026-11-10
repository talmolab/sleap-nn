"""Tests for the distributed rank (``sleap_nn.RANK``) and the rank the logs use."""

import os
import subprocess
import sys

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import sleap_nn
from sleap_nn import _get_rank

RANK_KEYS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every rank variable so each test sets only its own."""
    for key in RANK_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_not_distributed(clean_env):
    assert _get_rank() == -1


def test_lightning_spawned_or_torchrun_local_rank(clean_env):
    clean_env.setenv("LOCAL_RANK", "1")
    assert _get_rank() == 1


def test_srun_task_uses_slurm_procid(clean_env):
    # srun starts one task per GPU and sets only SLURM_* variables.
    clean_env.setenv("SLURM_PROCID", "1")
    clean_env.setenv("SLURM_LOCALID", "1")
    assert _get_rank() == 1


def test_srun_second_node_uses_global_rank(clean_env):
    # Task 0 on the second node: SLURM_LOCALID is 0 but it is not global rank 0.
    clean_env.setenv("SLURM_PROCID", "2")
    clean_env.setenv("SLURM_LOCALID", "0")
    assert _get_rank() == 2


def test_torchrun_multinode_prefers_global_rank(clean_env):
    clean_env.setenv("RANK", "3")
    clean_env.setenv("LOCAL_RANK", "1")
    assert _get_rank() == 3


def test_lightning_child_inside_slurm_job(clean_env):
    # A child Lightning spawns inside an sbatch job inherits SLURM_PROCID=0
    # from its parent; LOCAL_RANK (set by Lightning) must win.
    clean_env.setenv("SLURM_PROCID", "0")
    clean_env.setenv("LOCAL_RANK", "1")
    assert _get_rank() == 1


def test_empty_value_is_ignored(clean_env):
    clean_env.setenv("LOCAL_RANK", "")
    clean_env.setenv("SLURM_PROCID", "1")
    assert _get_rank() == 1


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, -1),
        ({"SLURM_PROCID": "1", "SLURM_LOCALID": "1"}, 1),
        ({"SLURM_PROCID": "0", "SLURM_LOCALID": "0"}, 0),
    ],
)
def test_rank_at_import(env, expected):
    """``sleap_nn.RANK`` is set at import, in a fresh interpreter."""
    child_env = {k: v for k, v in os.environ.items() if k not in RANK_KEYS}
    child_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", "import sleap_nn; print(sleap_nn.RANK)"],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(out.stdout.strip().splitlines()[-1]) == expected


def _run_fresh(code, env):
    """Run ``code`` in a fresh interpreter with only ``env``'s rank variables."""
    child_env = {k: v for k, v in os.environ.items() if k not in RANK_KEYS}
    child_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout + out.stderr


def test_log_rank_at_import_ignores_slurm_procid():
    """Outside training, an srun task keeps its logs: the log rank is LOCAL_RANK's."""
    out = _run_fresh(
        "import sleap_nn; print('ranks', sleap_nn.RANK, sleap_nn._LOG_RANK)",
        {"SLURM_PROCID": "1", "SLURM_LOCALID": "1"},
    )
    assert "ranks 1 -1" in out


LOG_SCRIPT = """
import sleap_nn
from loguru import logger
logger.info("setup-info")
sleap_nn._use_training_rank_for_logs()
logger.info("train-info")
logger.warning("train-warning")
logger.error("train-error")
"""


def test_srun_task_logs_until_training_starts():
    """Task 1 of ``srun -n 2``: logs as usual, then only errors once training starts."""
    out = _run_fresh(LOG_SCRIPT, {"SLURM_PROCID": "1", "SLURM_LOCALID": "1"})
    assert "setup-info" in out
    assert "train-info" not in out
    assert "train-warning" not in out
    assert "train-error" in out


def test_srun_rank_zero_keeps_training_logs():
    out = _run_fresh(LOG_SCRIPT, {"SLURM_PROCID": "0", "SLURM_LOCALID": "0"})
    for message in ("setup-info", "train-info", "train-warning"):
        assert message in out


def test_run_training_switches_logs_to_training_rank(clean_env):
    from sleap_nn.train import ModelTrainer, run_training

    clean_env.setenv("SLURM_PROCID", "1")
    clean_env.setattr(sleap_nn, "_LOG_RANK", -1)

    def stop(*args, **kwargs):
        raise RuntimeError("stop before training")

    clean_env.setattr(ModelTrainer, "get_model_trainer_from_config", stop)
    with pytest.raises(RuntimeError, match="stop before training"):
        run_training(OmegaConf.create({}))
    assert sleap_nn._LOG_RANK == 1


def test_model_trainer_train_switches_logs_to_training_rank(clean_env):
    from sleap_nn.training.model_trainer import ModelTrainer

    clean_env.setenv("SLURM_PROCID", "1")
    clean_env.setattr(sleap_nn, "_LOG_RANK", -1)
    fake = SimpleNamespace(_train=lambda dry_run: "trained")
    assert ModelTrainer.train(fake) == "trained"
    assert sleap_nn._LOG_RANK == 1
