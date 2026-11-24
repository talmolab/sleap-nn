"""Two-process CPU (gloo) DDP training run for ``tests/training/test_ddp.py``.

Not collected by pytest. The test launches it either with torchrun (every rank runs
this script from the start) or as a plain process that Lightning's own launcher
re-runs for rank 1 (the ``sleap-nn train`` path). Each rank writes
``<out_dir>/rank<k>.json`` with what it observed:

* ``n_eval``: val entries the epoch-end evaluator scored (rank 0; the whole val set
  only if every rank's shard was gathered),
* ``mOKS``: the ``eval/val/mOKS`` value in this rank's ``callback_metrics`` per
  epoch (the evaluator's mOKS is replaced by ``0.1 * (epoch + 1)`` so the
  checkpoint has an improvement to track),
* ``best_score`` / ``best_path``: this rank's ``best.ckpt`` checkpointer state,
* ``early_stopping``: the early stopper's class and monitor,
* ``probes``: one ``np.random.random()`` draw per train sample, taken in the
  dataloader worker (the augmentation RNG).

Usage: ``python _ddp_smoke_train.py <out_dir> <train.slp> <val.slp>``
"""

import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from sleap_nn.data.custom_datasets import CenteredInstanceDataset
from sleap_nn.training.callbacks import EpochEndEvaluationCallback
from sleap_nn.training.model_trainer import ModelTrainer

RECORD = {"n_eval": [], "mOKS": [], "probes": []}


def _probe_getitem(orig):
    def getitem(self, index):
        sample = orig(self, index)
        sample["rng_probe"] = torch.tensor(np.random.random())
        return sample

    return getitem


def _synthetic_evaluate(orig):
    def evaluate(self, gts, preds, epoch):
        RECORD["n_eval"].append(len(gts))
        metrics = orig(self, gts, preds, epoch) or {
            group: {key: float("nan") for key in keys}
            for group, keys in self._LOGGED_METRICS.items()
        }
        metrics["mOKS"]["mOKS"] = 0.1 * (epoch + 1)
        return metrics

    return evaluate


class Recorder(Callback):
    """Record per-rank observations (runs before the checkpointer each epoch)."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.training and not trainer.sanity_checking:
            RECORD["probes"].extend(batch["rng_probe"].tolist())

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            RECORD["mOKS"].append(
                float(trainer.callback_metrics["eval/val/mOKS"].item())
            )


def _setup_with_recorder(orig):
    def setup(self, *args, **kwargs):
        loggers, callbacks = orig(self, *args, **kwargs)
        return loggers, [*callbacks, Recorder()]

    return setup


def main(out_dir: str, train_path: str, val_path: str) -> None:
    CenteredInstanceDataset.__getitem__ = _probe_getitem(
        CenteredInstanceDataset.__getitem__
    )
    EpochEndEvaluationCallback._evaluate = _synthetic_evaluate(
        EpochEndEvaluationCallback._evaluate
    )
    ModelTrainer._setup_loggers_callbacks = _setup_with_recorder(
        ModelTrainer._setup_loggers_callbacks
    )
    config = OmegaConf.create(
        {
            "data_config": {
                "train_labels_path": [train_path],
                "val_labels_path": [val_path],
                "preprocessing": {"crop_size": 64, "ensure_grayscale": True},
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {"contrast_p": 1.0},
                    "geometric": {"rotation_min": -180.0, "rotation_max": 180.0},
                },
            },
            "model_config": {
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "filters": 8,
                        "max_stride": 8,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "centered_instance": {
                        "confmaps": {
                            "part_names": ["A", "B"],
                            "anchor_part": "A",
                            "sigma": 1.5,
                            "output_stride": 2,
                        }
                    }
                },
            },
            "trainer_config": {
                "train_data_loader": {"batch_size": 2, "num_workers": 1},
                "val_data_loader": {"batch_size": 2, "num_workers": 0},
                "trainer_accelerator": "cpu",
                "trainer_devices": 2,
                "trainer_strategy": "ddp",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 4,
                "max_epochs": 3,
                "seed": 1000,
                "save_ckpt": True,
                "ckpt_dir": out_dir,
                "run_name": "run",
                "model_ckpt": {"monitor": "eval/val/mOKS", "mode": "max"},
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "patience": 1,
                    "min_delta": 1e-8,
                },
                "eval": {"enabled": True, "frequency": 1},
                "optimizer": {"lr": 1e-4},
            },
        }
    )
    model_trainer = ModelTrainer.get_model_trainer_from_config(
        config,
        train_labels=[sio.load_slp(train_path)],
        val_labels=[sio.load_slp(val_path)],
    )
    model_trainer.train()

    trainer = model_trainer.trainer
    best = next(cb for cb in trainer.checkpoint_callbacks if cb.monitor is not None)
    es = trainer.early_stopping_callback
    RECORD.update(
        rank=trainer.global_rank,
        world_size=trainer.world_size,
        best_score=float(best.best_model_score),
        best_path=Path(best.best_model_path).name,
        early_stopping=[type(es).__name__, es.monitor],
        epochs=trainer.current_epoch,
    )
    Path(out_dir, f"rank{trainer.global_rank}.json").write_text(json.dumps(RECORD))


if __name__ == "__main__":
    main(*sys.argv[1:4])
