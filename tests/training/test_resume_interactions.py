"""Resuming a run (`sleap-nn train --resume`) with step- and epoch-keyed features.

A resumed run is a new process that starts again from the config and a checkpoint,
so anything keyed on "how far training is" has to come from the checkpoint:

* `trainer_config.freeze_schedule` counts optimizer steps;
* the order of the training samples is a new shuffle every epoch.

The first block runs the pieces in a real `lightning.Trainer` with a toy module
(milliseconds); the second trains a tiny model through `ModelTrainer` on eight
generated frames, on one CPU device.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import lightning as L
import numpy as np
import pytest
import sleap_io as sio
import torch
from click.testing import CliRunner
from omegaconf import OmegaConf
from PIL import Image

from sleap_nn.cli import cli
from sleap_nn.data.custom_datasets import InfiniteDataLoader
from sleap_nn.training.callbacks import EncoderFreezeSchedule, WeightEMA
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.training.weight_transfer import select_params

_WORKER = Path(__file__).parent / "ddp_freeze_schedule_worker.py"


def _toy_module():
    """The toy encoder/decoder/head Lightning module of the DDP freeze test."""
    spec = importlib.util.spec_from_file_location("_freeze_worker", _WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._Module()


class _EncoderRecorder(L.Callback):
    """Per training step: was the encoder trainable, and did its weights move.

    Keyed on the global step BEFORE the optimizer step (step ``k`` is the
    ``k + 1``-th step of the whole run, resumed or not). Add it after
    ``EncoderFreezeSchedule`` so it sees the state that step trains with.
    """

    def __init__(self):
        self.steps = {}
        self.ema_names = None
        self.encoder_names = None

    @staticmethod
    def _encoder(pl_module):
        params = dict(pl_module.named_parameters())
        return {n: params[n] for n in select_params(pl_module, ["encoder"])}

    def on_fit_start(self, trainer, pl_module):
        self.encoder_names = set(self._encoder(pl_module))
        for callback in trainer.callbacks:
            if isinstance(callback, WeightEMA):
                self.ema_names = set(callback.shadow)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        encoder = self._encoder(pl_module)
        self._step = trainer.global_step
        self._trainable = all(p.requires_grad for p in encoder.values())
        self._before = {n: p.detach().clone() for n, p in encoder.items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        moved = any(
            not torch.equal(p, self._before[n])
            for n, p in self._encoder(pl_module).items()
        )
        self.steps[self._step] = (self._trainable, moved)


_FROZEN, _TRAINING = (False, False), (True, True)


def _toy_fit(max_steps, unfreeze_at_step, ckpt_path=None, save_to=None):
    L.seed_everything(0)
    data = torch.utils.data.TensorDataset(torch.randn(16, 1, 8, 8))
    recorder = _EncoderRecorder()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        callbacks=[
            EncoderFreezeSchedule(
                targets=["encoder"], unfreeze_at_step=unfreeze_at_step
            ),
            recorder,
        ],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    module = _toy_module()
    trainer.fit(
        module, torch.utils.data.DataLoader(data, batch_size=2), ckpt_path=ckpt_path
    )
    if save_to is not None:
        trainer.save_checkpoint(save_to)
    return recorder.steps, module


@pytest.mark.parametrize(
    "stopped_at, unfreeze_at_step",
    [(5, 3), (3, 3), (2, 5)],
    ids=["past", "at", "before"],
)
def test_freeze_schedule_counts_steps_from_the_checkpoint(
    tmp_path, stopped_at, unfreeze_at_step
):
    """A resumed run is frozen until the schedule's step of the WHOLE run.

    Stopped after the unfreeze step: the encoder trains from the first resumed
    step (not frozen for another ``unfreeze_at_step`` steps). Stopped before it:
    the encoder stays frozen and is released at ``unfreeze_at_step``, not at
    ``unfreeze_at_step`` + the steps already done.
    """
    total = 8
    expected = {
        step: _FROZEN if step < unfreeze_at_step else _TRAINING for step in range(total)
    }
    ckpt = tmp_path / "stopped.ckpt"
    first, _ = _toy_fit(stopped_at, unfreeze_at_step, save_to=ckpt)
    assert first == {step: expected[step] for step in range(stopped_at)}

    resumed, module = _toy_fit(total, unfreeze_at_step, ckpt_path=ckpt)
    assert resumed == {step: expected[step] for step in range(stopped_at, total)}
    # Released for good: train mode again, every encoder weight trainable.
    backbone = module.model.backbone
    assert not backbone.freeze
    module.train()
    assert backbone.enc.training
    assert all(p.requires_grad for p in backbone.enc.parameters())


class _IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index


class _OrderModule(L.LightningModule):
    """Records the sample indices of every training batch, per epoch."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.epochs = {}

    def training_step(self, batch, batch_idx):
        self.epochs.setdefault(self.current_epoch, []).extend(batch.tolist())
        return self.weight**2

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_epoch_aware_sampler_continues_on_resume(tmp_path):
    """Multi-device and tiled training shuffle by epoch number: a resume continues.

    Their samplers (``DistributedSampler``, the tile sampler) seed each epoch's
    order from the epoch, and ``InfiniteDataLoader`` starts iterating only after
    Lightning has set the resumed epoch. Here: the multi-device sampler with one
    replica, in a real ``Trainer`` on one process.
    """
    dataset = _IndexDataset()

    def fit(max_epochs, ckpt_path=None, save_to=None):
        module = _OrderModule()
        loader = InfiniteDataLoader(
            dataset=dataset,
            sampler=torch.utils.data.DistributedSampler(
                dataset, num_replicas=1, rank=0
            ),
            batch_size=4,
            len_dataloader=4,
        )
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            use_distributed_sampler=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(module, loader, ckpt_path=ckpt_path)
        if save_to is not None:
            trainer.save_checkpoint(save_to)
        return module.epochs

    reference = torch.utils.data.DistributedSampler(dataset, num_replicas=1, rank=0)
    expected = {}
    for epoch in range(4):
        reference.set_epoch(epoch)
        expected[epoch] = list(reference)
    assert len({tuple(order) for order in expected.values()}) == 4

    ckpt = tmp_path / "two_epochs.ckpt"
    assert fit(2, save_to=ckpt) == {e: expected[e] for e in (0, 1)}
    assert fit(4, ckpt_path=ckpt) == {e: expected[e] for e in (2, 3)}


# ------------------------------------------------- through ModelTrainer (1 CPU)

_N_FRAMES = 8


@pytest.fixture
def frames_config(config, tmp_path):
    """The centered-instance test config on eight generated one-animal frames."""
    rng = np.random.default_rng(0)
    files = []
    for i in range(_N_FRAMES):
        files.append((tmp_path / f"frame_{i}.png").as_posix())
        Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(files[-1])
    video = sio.Video.from_filename(files)
    skeleton = sio.Skeleton(["A", "B"])
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[24.0 + i, 30.0], [40.0, 34.0 + i]]), skeleton=skeleton
                )
            ],
        )
        for i in range(_N_FRAMES)
    ]
    labels_path = (tmp_path / "labels.slp").as_posix()
    sio.Labels(frames, videos=[video], skeletons=[skeleton]).save(labels_path)

    cfg = config.copy()
    for key, value in {
        "data_config.train_labels_path": [labels_path],
        "data_config.val_labels_path": [labels_path],
        "data_config.preprocessing.crop_size": 32,
        "trainer_config.trainer_accelerator": "cpu",
        "trainer_config.min_train_steps_per_epoch": 1,
        "trainer_config.save_ckpt": True,
        "trainer_config.keep_viz": False,
        "trainer_config.ckpt_dir": (tmp_path / "runs").as_posix(),
    }.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


class _OrderRecorder(L.Callback):
    """The frame index of every training sample, per epoch, in training order."""

    def __init__(self):
        self.epochs = {}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.epochs.setdefault(trainer.current_epoch, []).extend(
            int(i) for i in batch["frame_idx"].flatten()
        )


def _train(cfg, recorder, monkeypatch, **updates):
    """Train ``cfg`` (with ``updates``) through ``ModelTrainer``, recording."""
    cfg = cfg.copy()
    for key, value in updates.items():
        OmegaConf.update(cfg, f"trainer_config.{key}", value, force_add=True)
    setup = ModelTrainer._setup_loggers_callbacks

    def setup_with_recorder(self, *args, **kwargs):
        loggers, callbacks = setup(self, *args, **kwargs)
        return loggers, [*callbacks, recorder]

    with monkeypatch.context() as m:
        m.setattr(ModelTrainer, "_setup_loggers_callbacks", setup_with_recorder)
        ModelTrainer.get_model_trainer_from_config(cfg).train()
    return recorder


def _last_ckpt(cfg, run_name):
    """The checkpoint `sleap-nn train --resume <run folder>` resumes from."""
    path = Path(cfg.trainer_config.ckpt_dir) / run_name / "last.ckpt"
    if not path.is_file():
        raise RuntimeError(f"{path} was not written.")
    return path.as_posix()


def test_cli_resume_with_freeze_schedule_and_ema(frames_config, monkeypatch):
    """`sleap-nn train --resume` + `freeze_schedule` + weight EMA, end to end.

    A run of 3 steps is resumed twice through the `train` command's own option
    handling: with the schedule's step already passed (the saved config), and
    with it still ahead (an override).
    """
    cfg = frames_config
    schedule = {"targets": ["encoder"], "unfreeze_at_step": 2}
    ema = {"enabled": True, "decay": 0.5, "warmup_steps": 0}
    first = _train(
        cfg,
        _EncoderRecorder(),
        monkeypatch,
        run_name="orig",
        max_epochs=1,
        train_steps_per_epoch=3,
        freeze_schedule=schedule,
        ema=ema,
    )
    assert first.steps == {0: _FROZEN, 1: _FROZEN, 2: _TRAINING}
    run_dir = Path(cfg.trainer_config.ckpt_dir) / "orig"

    def resume(*overrides):
        with patch("sleap_nn.train.run_training") as run_training:
            result = CliRunner().invoke(
                cli,
                ["train", "--resume", run_dir.as_posix(), *overrides],
            )
        assert result.exit_code == 0, result.output
        resumed_cfg = run_training.call_args.kwargs["config"]
        assert resumed_cfg.trainer_config.resume_ckpt_path == _last_ckpt(cfg, "orig")
        recorder = _train(resumed_cfg, _EncoderRecorder(), monkeypatch)
        return resumed_cfg, recorder

    # Stopped at step 3, past the unfreeze step 2: trains from the first step.
    resumed_cfg, past = resume("trainer_config.max_epochs=2")
    assert resumed_cfg.trainer_config.run_name == "orig-1"
    assert resumed_cfg.trainer_config.freeze_schedule.unfreeze_at_step == 2
    assert past.steps == {3: _TRAINING, 4: _TRAINING, 5: _TRAINING}
    # The EMA is set up before the schedule freezes anything, so it averages
    # the encoder too.
    assert past.encoder_names and past.encoder_names <= past.ema_names

    # Unfreeze step 5 still ahead: frozen for steps 3 and 4 only (not until 5 + 3).
    resumed_cfg, before = resume(
        "trainer_config.max_epochs=3",
        "trainer_config.freeze_schedule.unfreeze_at_step=5",
    )
    assert resumed_cfg.trainer_config.run_name == "orig-2"
    assert before.steps == {
        3: _FROZEN,
        4: _FROZEN,
        5: _TRAINING,
        6: _TRAINING,
        7: _TRAINING,
        8: _TRAINING,
    }
    assert before.encoder_names <= before.ema_names


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "On one device without `balanced_sampler`, the train loader shuffles with "
        "the global torch RNG, which a resumed run seeds again with "
        "`trainer_config.seed`: its first epoch replays the original run's first "
        "epoch (order and augmentation draws), its second the second, and so on."
    ),
)
def test_resume_does_not_replay_the_first_epochs_order(frames_config, monkeypatch):
    """The first resumed epoch must not train in the first epoch's order again.

    (Stronger, and what the epoch-aware samplers above already do: it should be
    the order epoch 1 of an uninterrupted run draws.)
    """
    cfg = frames_config
    first = _train(cfg, _OrderRecorder(), monkeypatch, run_name="orig", max_epochs=1)
    resumed = _train(
        cfg,
        _OrderRecorder(),
        monkeypatch,
        run_name="orig-1",
        max_epochs=2,
        resume_ckpt_path=_last_ckpt(cfg, "orig"),
    )
    # One pass over the eight frames per epoch; the resumed run ran epoch 1 only.
    if sorted(first.epochs[0]) != list(range(_N_FRAMES)) or list(resumed.epochs) != [1]:
        raise RuntimeError(f"Unexpected runs: {first.epochs}, {resumed.epochs}")
    assert resumed.epochs[1] != first.epochs[0]


def test_resume_continues_the_balanced_sampler(frames_config, monkeypatch):
    """`balanced_sampler` + resume: the stream goes on where the checkpoint left it.

    The sampler's position is in every checkpoint, but Lightning loads it only
    from a mid-epoch checkpoint; `last.ckpt` is written when an epoch is over.
    """
    cfg = frames_config.copy()
    pre = cfg.data_config.preprocessing
    pre.crop_mode = "bbox"
    pre.crop_pipeline = "warp"
    pre.crop_size = [32, 32]
    head = cfg.model_config.head_configs.centered_instance.confmaps
    head.anchor_part = None
    head.centroid_method = "bbox_center"
    # Six batches of one sample per epoch over eight frames: epochs and passes
    # over the data do not line up.
    common = {"balanced_sampler": {"counts": [1]}, "train_steps_per_epoch": 6}

    full = _train(
        cfg, _OrderRecorder(), monkeypatch, run_name="full", max_epochs=3, **common
    )
    first = _train(
        cfg, _OrderRecorder(), monkeypatch, run_name="orig", max_epochs=1, **common
    )
    ckpt = _last_ckpt(cfg, "orig")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["loops"]["fit_loop"]["state_dict"]["combined_loader"] == [
        {"position": 6}
    ]
    resumed = _train(
        cfg,
        _OrderRecorder(),
        monkeypatch,
        run_name="orig-1",
        max_epochs=3,
        resume_ckpt_path=ckpt,
        **common,
    )
    assert first.epochs == {0: full.epochs[0]}
    assert len({tuple(order) for order in full.epochs.values()}) == 3
    assert resumed.epochs == {1: full.epochs[1], 2: full.epochs[2]}
