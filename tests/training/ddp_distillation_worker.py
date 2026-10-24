"""2-process DDP (gloo, CPU) training run with ``trainer_config.distillation``.

Run by ``test_distillation.py::test_distillation_under_ddp`` as a plain process:
Lightning's DDP launcher re-runs this file for rank 1 (the ``sleap-nn train``
path), so it is not a test module itself. Each rank writes
``<out_dir>/rank<k>.json`` with what it observed: whether DDP wrapped or holds
the teacher, whether the teacher stayed frozen and unchanged, and sums of the
trained student and projector weights (equal on both ranks when DDP synced
them).

Usage: ``python ddp_distillation_worker.py OUT_DIR TEACHER_DIR``
"""

import json
import sys
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from sleap_nn.training.model_trainer import ModelTrainer

RECORD = {}


class Recorder(Callback):
    """Snapshot the teacher at fit start, compare at fit end."""

    def on_fit_start(self, trainer, pl_module):
        teacher = pl_module.distiller.teacher
        RECORD["teacher_start"] = [p.detach().clone() for p in teacher.parameters()]
        RECORD["projector_start"] = pl_module.distiller.projector.weight.clone()
        ddp_params = {id(p) for p in trainer.strategy.model.parameters()}
        RECORD["teacher_in_ddp_params"] = any(
            id(p) in ddp_params for p in teacher.parameters()
        )
        RECORD["strategy_wraps_teacher"] = any(
            m is teacher for m in trainer.strategy.model.modules()
        )

    def on_fit_end(self, trainer, pl_module):
        distiller = pl_module.distiller
        teacher = distiller.teacher
        RECORD["teacher_unchanged"] = all(
            torch.equal(a, b.detach())
            for a, b in zip(RECORD.pop("teacher_start"), teacher.parameters())
        )
        RECORD["teacher_training"] = teacher.training
        start = RECORD.pop("projector_start")
        RECORD["projector_changed"] = not torch.equal(start, distiller.projector.weight)
        RECORD["projector_sum"] = float(distiller.projector.weight.double().sum())
        RECORD["student_sum"] = float(
            sum(p.detach().double().sum() for p in pl_module.model.parameters())
        )
        RECORD["steps"] = trainer.global_step


def _setup_with_recorder(orig):
    def setup(self, *args, **kwargs):
        loggers, callbacks = orig(self, *args, **kwargs)
        return loggers, [*callbacks, Recorder()]

    return setup


def main(out_dir: str, teacher_dir: str) -> None:
    ModelTrainer._setup_loggers_callbacks = _setup_with_recorder(
        ModelTrainer._setup_loggers_callbacks
    )
    data = (
        Path(__file__).parents[1] / "assets" / "datasets" / "minimal_instance.pkg.slp"
    )
    teacher_cfg = OmegaConf.load(Path(teacher_dir) / "training_config.yaml")
    head = OmegaConf.to_container(
        teacher_cfg.model_config.head_configs.centered_instance
    )
    config = OmegaConf.create(
        {
            "data_config": {
                "train_labels_path": [data.as_posix()],
                "val_labels_path": [data.as_posix()],
                "preprocessing": {"crop_size": 96, "ensure_grayscale": True},
                "use_augmentations_train": False,
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
                "head_configs": {"centered_instance": head},
            },
            "trainer_config": {
                "train_data_loader": {"batch_size": 1, "num_workers": 0},
                "val_data_loader": {"batch_size": 1, "num_workers": 0},
                "trainer_accelerator": "cpu",
                "trainer_devices": 2,
                "trainer_strategy": "ddp",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 8,
                "train_steps_per_epoch": 8,
                "max_epochs": 1,
                "seed": 1000,
                "save_ckpt": True,
                "ckpt_dir": out_dir,
                "run_name": "run",
                "optimizer": {"lr": 1e-3},
                "distillation": {"teacher": teacher_dir, "alpha": 1.0},
            },
        }
    )
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    model_trainer.train()
    trainer = model_trainer.trainer
    RECORD.update(rank=trainer.global_rank, world_size=trainer.world_size)
    Path(out_dir, f"rank{trainer.global_rank}.json").write_text(json.dumps(RECORD))


if __name__ == "__main__":
    main(*sys.argv[1:3])
