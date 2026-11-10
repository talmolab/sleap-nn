"""Tests for the distributed rank read at import (``sleap_nn.RANK``)."""

import os
import subprocess
import sys

import pytest

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
