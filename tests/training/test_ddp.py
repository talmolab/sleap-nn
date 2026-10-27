"""DDP correctness: epoch-end eval, checkpoint/early-stop selection, worker seeds.

The unit tests fake the collectives; ``test_ddp_smoke`` runs a real 2-process CPU
(gloo) training job, launched by torchrun and by Lightning's own launcher.
"""

import functools
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sleap_io as sio
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from sleap_nn.data.custom_datasets import (
    _train_worker_init_fn,
    freeze_heap_worker_init_fn,
    rank_worker_init_fn,
)
from sleap_nn.training.callbacks import EvalCadenceEarlyStopping, _gather_val_entries
from sleap_nn.training.model_trainer import ModelTrainer

ASSETS = Path(__file__).parents[1] / "assets" / "datasets"
SMOKE_SCRIPT = Path(__file__).with_name("_ddp_smoke_train.py")


def _fake_ddp(monkeypatch, per_rank: list):
    """Make `_gather_val_entries` see `per_rank` as the ranks' (gts, preds)."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def all_gather_object(out, _obj):
        out[:] = per_rank

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)


def _crop_entries(sample_ids):
    """Top-down style entries: one per crop, `crop_id` counted on the rank."""
    gts = [{"sample": s, "crop_id": i} for i, s in enumerate(sample_ids)]
    preds = [{"sample": s, "crop_id": i} for i, s in enumerate(sample_ids)]
    return gts, preds


def test_gather_val_entries_restores_order_and_drops_padding(monkeypatch):
    """5 val samples over 2 ranks: rank 1's last entry is sampler padding."""
    sampler = DistributedSampler(range(5), num_replicas=2, rank=0, shuffle=False)
    shards = [list(DistributedSampler(range(5), 2, r, shuffle=False)) for r in (0, 1)]
    assert shards == [[0, 2, 4], [1, 3, 0]]
    _fake_ddp(monkeypatch, [_crop_entries(s) for s in shards])
    trainer = SimpleNamespace(
        world_size=2, val_dataloaders=SimpleNamespace(sampler=sampler)
    )

    gts, preds = _gather_val_entries(trainer, [], [])

    assert [g["sample"] for g in gts] == [0, 1, 2, 3, 4]
    assert [p["sample"] for p in preds] == [0, 1, 2, 3, 4]
    # Per-rank crop ids collided (0..2 on both ranks); gathered ones are unique.
    assert [g["crop_id"] for g in gts] == [0, 1, 2, 3, 4]
    assert [p["crop_id"] for p in preds] == [0, 1, 2, 3, 4]


def test_gather_val_entries_concatenates_unaligned_lists(monkeypatch):
    """Lists that do not match the sampler one-to-one are kept whole."""
    sampler = DistributedSampler(range(5), num_replicas=2, rank=0, shuffle=False)
    per_rank = [_crop_entries([0, 2]), _crop_entries([1, 3, 0])]
    _fake_ddp(monkeypatch, per_rank)
    trainer = SimpleNamespace(
        world_size=2, val_dataloaders=[SimpleNamespace(sampler=sampler)]
    )

    gts, preds = _gather_val_entries(trainer, [], [])

    assert [g["sample"] for g in gts] == [0, 2, 1, 3, 0]
    assert [g["crop_id"] for g in gts] == [0, 1, 2, 3, 4]


def test_gather_val_entries_single_process_is_identity():
    """Without a process group the rank's own lists come back unchanged."""
    gts, preds = _crop_entries([0, 1])
    trainer = SimpleNamespace(world_size=1, val_dataloaders=None)
    assert _gather_val_entries(trainer, gts, preds) == (gts, preds)


class _RngDraws(Dataset):
    """Each item is one draw from each RNG an augmentation might use."""

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(
            [random.random(), np.random.random(), torch.rand(()).item()]
        )


def _worker_draws(rank, worker_init_fn=rank_worker_init_fn):
    torch.manual_seed(1000)  # every rank's main RNG is seeded the same way
    loader = DataLoader(
        _RngDraws(),
        batch_size=2,
        num_workers=2,
        worker_init_fn=(
            functools.partial(worker_init_fn, rank=rank) if worker_init_fn else None
        ),
    )
    return torch.cat(list(loader))


def test_rank_worker_init_fn_decorrelates_ranks():
    """Same main seed: identical streams without the fix, distinct with it."""
    assert torch.equal(
        _worker_draws(0, worker_init_fn=None), _worker_draws(1, worker_init_fn=None)
    )
    rank0, rank1 = _worker_draws(0), _worker_draws(1)
    # Every RNG (python, numpy, torch) differs per rank, and stays reproducible.
    assert not (rank0 == rank1).any(dim=0).any()
    assert torch.equal(rank0, _worker_draws(0))


def test_train_worker_init_fn_only_under_ddp():
    """One device keeps PyTorch's default worker seeding (behaviour unchanged)."""
    config = OmegaConf.create(
        {"trainer_config": {"train_data_loader": {"num_workers": 2}}}
    )
    assert (
        _train_worker_init_fn(config, rank=0, devices=1) is freeze_heap_worker_init_fn
    )
    fn = _train_worker_init_fn(config, rank=3, devices=4)
    assert fn.func is rank_worker_init_fn and fn.keywords == {"rank": 3}


@pytest.mark.parametrize(
    "monitor, mode, expected_cls",
    [
        ("val/loss", "min", EarlyStopping),
        ("eval/val/mOKS", "max", EvalCadenceEarlyStopping),
    ],
)
def test_early_stopping_follows_checkpoint_monitor(
    config, tmp_path, minimal_instance, monitor, mode, expected_cls
):
    """Early stopping watches the same metric (and mode) as `best.ckpt`."""
    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", "es_monitor")
    OmegaConf.update(cfg, "trainer_config.eval.enabled", True)
    OmegaConf.update(cfg, "trainer_config.eval.frequency", 3)
    OmegaConf.update(cfg, "trainer_config.model_ckpt.monitor", monitor)
    OmegaConf.update(cfg, "trainer_config.model_ckpt.mode", mode)
    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    _, callbacks = trainer._setup_loggers_callbacks(
        viz_train_dataset=None, viz_val_dataset=None
    )

    best = next(c for c in callbacks if isinstance(c, ModelCheckpoint) and c.monitor)
    es = next(c for c in callbacks if isinstance(c, EarlyStopping))
    assert (best.monitor, best.mode) == (monitor, mode)
    assert (es.monitor, es.mode) == (monitor, mode)
    assert type(es) is expected_cls
    if expected_cls is EvalCadenceEarlyStopping:
        assert es.eval_frequency == 3


def _shifted(image, points, dy, dx):
    return np.roll(image, (dy, dx), axis=(0, 1)), [p + [dx, dy] for p in points]


@pytest.fixture(scope="module")
def ddp_labels(tmp_path_factory):
    """Train (6 frames, 12 crops) and val (3 frames, 5 crops) label files.

    5 val crops over 2 ranks is 3 + 3: the sampler pads rank 1 with a repeat.
    """
    root = tmp_path_factory.mktemp("ddp_data")
    src = sio.load_slp(ASSETS / "minimal_instance.pkg.slp")
    lf = src[0]
    image = lf.image[..., 0]
    points = [inst.numpy() for inst in lf.user_instances]
    skeleton = src.skeletons[0]

    def write(name, shifts, n_instances):
        paths, frames = [], []
        for k, ((dy, dx), n) in enumerate(zip(shifts, n_instances)):
            img, pts = _shifted(image, points, dy, dx)
            path = root / f"{name}_{k}.png"
            Image.fromarray(img).save(path)
            paths.append(path.as_posix())
            frames.append((k, pts[:n]))
        video = sio.Video.from_filename(paths)
        labels = sio.Labels(
            labeled_frames=[
                sio.LabeledFrame(
                    video=video,
                    frame_idx=k,
                    instances=[sio.Instance.from_numpy(p, skeleton) for p in pts],
                )
                for k, pts in frames
            ],
            videos=[video],
            skeletons=[skeleton],
        )
        out = root / f"{name}.slp"
        sio.save_slp(labels, out.as_posix())
        return out.as_posix()

    train = write("train", [(4 * k, -3 * k) for k in range(6)], [2] * 6)
    val = write("val", [(5, 7), (-6, 2), (3, -9)], [2, 2, 1])
    return train, val


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="gloo DDP smoke test runs on Linux"
)
@pytest.mark.parametrize("launcher", ["torchrun", "lightning"])
def test_ddp_smoke(tmp_path, ddp_labels, launcher):
    """2 ranks: the whole val set is scored, every rank agrees on the metric, the
    best checkpoint tracks the improving metric, early stopping watches it too, and
    the ranks' augmentation streams differ."""
    train, val = ddp_labels
    out = tmp_path / "out"
    out.mkdir()
    args = [SMOKE_SCRIPT.as_posix(), out.as_posix(), train, val]
    if launcher == "torchrun":
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
        cmd += ["--nproc_per_node=2", *args]
    else:
        cmd = [sys.executable, *args]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_", "SLURM"))
    }
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WANDB_MODE="disabled")
    proc = subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    r0, r1 = (json.loads((out / f"rank{k}.json").read_text()) for k in (0, 1))
    assert (r0["rank"], r1["rank"], r0["world_size"]) == (0, 1, 2)
    assert r0["epochs"] == r1["epochs"] == 3

    # (1) Rank 0 scored every val crop each epoch, not just its own shard of 3.
    assert r0["n_eval"] == [5, 5, 5]
    assert r1["n_eval"] == []

    # (2) Every rank holds rank 0's value, so the all-reduced "is this better?"
    # vote passes and best.ckpt follows the improving metric to the last epoch.
    assert r0["mOKS"] == r1["mOKS"] == pytest.approx([0.1, 0.2, 0.3])
    assert r0["best_score"] == r1["best_score"] == pytest.approx(0.3)
    ckpt = torch.load(out / "run" / "best.ckpt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2

    # (3) Early stopping watches the configured checkpoint monitor.
    assert r0["early_stopping"] == ["EvalCadenceEarlyStopping", "eval/val/mOKS"]

    # (4) Augmentation draws differ between the ranks.
    assert len(r0["probes"]) == len(r1["probes"]) > 0
    assert not set(r0["probes"]) & set(r1["probes"])
