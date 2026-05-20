"""Tests for the Intel XPU code path.

All tests are gated by ``requires_xpu`` and silently skip on machines
without a working ``torch.xpu`` install (the default CI matrix).

Covers:
- The Lightning ``XPUAccelerator`` registration side-effect from
  ``sleap_nn.training.xpu_accelerator``.
- The strategy passthrough wiring in
  ``sleap_nn.training.model_trainer.ModelTrainer.train`` (Lightning's
  built-in strategy chooser falls back to CPU for non-CUDA/MPS
  accelerators, so we construct ``SingleDeviceStrategy`` explicitly).
- A 1-epoch centered-instance training run hitting the XPU.
"""

from __future__ import annotations

import pytest
import torch
import lightning as L
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import OmegaConf

# Import the side-effect: register the XPU accelerator with Lightning.
from sleap_nn.training import xpu_accelerator  # noqa: F401
from sleap_nn.training.xpu_accelerator import XPUAccelerator
from sleap_nn.training.model_trainer import ModelTrainer

requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="Intel XPU not available"
)


def test_xpu_accelerator_registered():
    """Importing the module registers the accelerator under name 'xpu'."""
    assert "xpu" in AcceleratorRegistry.keys()


@requires_xpu
def test_xpu_accelerator_reports_available():
    assert XPUAccelerator.is_available() is True
    assert XPUAccelerator.auto_device_count() >= 1


@requires_xpu
def test_lightning_trainer_accepts_xpu_accelerator():
    """`Trainer(accelerator='xpu', ...)` should construct cleanly with an
    explicit single-device strategy (the passthrough sleap-nn uses)."""
    trainer = L.Trainer(
        accelerator="xpu",
        devices=1,
        strategy=SingleDeviceStrategy(device=torch.device("xpu", 0)),
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    assert trainer.strategy.root_device.type == "xpu"


@requires_xpu
def test_model_trainer_trains_one_epoch_on_xpu(config, tmp_path):
    """End-to-end: drive a 1-epoch centered-instance run with
    `trainer_accelerator='xpu'`. Verifies strategy passthrough + XPU
    seeding + that some XPU memory gets allocated."""
    cfg = OmegaConf.merge(config, OmegaConf.create({}))
    cfg.trainer_config.trainer_accelerator = "xpu"
    cfg.trainer_config.trainer_devices = 1
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = False
    cfg.trainer_config.keep_viz = False
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = "xpu_smoke"
    cfg.trainer_config.enable_progress_bar = False

    torch.xpu.reset_peak_memory_stats()

    model_trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    model_trainer.train()

    assert model_trainer.trainer.strategy.root_device.type == "xpu"
    # the model + activations have to land somewhere on the device
    assert torch.xpu.max_memory_allocated() > 0
