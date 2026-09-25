"""End-to-end `ModelTrainer.train` runs for the ``embedding`` model type.

The epoch length, the eval-gated early stopping and the checkpoint cadence are all
decided inside `ModelTrainer.train` (it computes `train_steps_per_epoch` and hands it
to the dataloader factory, and it builds the callbacks). Tests that call the helpers
directly passed while the real path was wrong, so everything here goes through
`ModelTrainer.train` (or `sleap_nn.train.train`) on a tiny CPU model.
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf

from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.model_trainer import ModelTrainer

N_FRAMES = 48
N_ANIMALS = 2
N_CROPS = N_FRAMES * N_ANIMALS  # 96
P, K = 2, 4  # P x K = 8 crops per step


@pytest.fixture
def tracked_slp(centered_instance_video, tmp_path):
    """Two tracked (and identified) animals in 48 frames of a real video."""
    video = sio.load_video(centered_instance_video.as_posix())
    skeleton = sio.Skeleton(["a", "b"])
    tracks = [sio.Track(name=f"track_{i}") for i in range(N_ANIMALS)]
    identities = [sio.Identity(name=f"animal_{i}") for i in range(N_ANIMALS)]
    rng = np.random.default_rng(0)
    frames = []
    for frame_idx in range(N_FRAMES):
        instances = []
        for track, identity, (cx, cy) in zip(
            tracks, identities, [(120, 180), (260, 200)]
        ):
            points = np.array([[cx - 10, cy], [cx + 10, cy]], float)
            instance = sio.Instance.from_numpy(
                points + rng.normal(0, 2, 2), skeleton=skeleton, track=track
            )
            instance.identity = identity
            instances.append(instance)
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    labels = sio.Labels(
        labeled_frames=frames, videos=[video], skeletons=[skeleton], tracks=tracks
    )
    path = tmp_path / "tracked.slp"
    labels.save(path.as_posix())
    return path


def _raw_config(tracked_slp, tmp_path, run_name, **updates):
    """The training config as a user would write it (not yet schema-merged)."""
    cfg = OmegaConf.create(
        {
            "data_config": {
                "train_labels_path": [tracked_slp.as_posix()],
                "val_labels_path": [tracked_slp.as_posix()],
                "data_pipeline_fw": "torch_dataset",
                "preprocessing": {"crop_size": 64},
                "identity": {"track_names_are_global": True},
            },
            "model_config": {
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "filters": 8,
                        "max_stride": 16,
                        "output_stride": 16,
                    }
                },
                "head_configs": {
                    "embedding": {
                        "embedding": {
                            "embedding_dim": 16,
                            "output_stride": 16,
                            "objective": {
                                "sampler": {
                                    "kind": "pk",
                                    "groups_per_batch": P,
                                    "samples_per_group": K,
                                }
                            },
                        }
                    }
                },
            },
            "trainer_config": {
                # Deliberately NOT P x K: the embedding loader ignores it.
                "train_data_loader": {"batch_size": 4, "num_workers": 0},
                "val_data_loader": {"batch_size": 16, "num_workers": 0},
                "trainer_accelerator": "cpu",
                "trainer_devices": 1,
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 1,
                "max_epochs": 1,
                "save_ckpt": True,
                "ckpt_dir": tmp_path.as_posix(),
                "run_name": run_name,
                "lr_scheduler": None,
                "early_stopping": {"stop_training_on_plateau": False},
                "model_ckpt": {"save_last": False},
            },
        }
    )
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def _config(tracked_slp, tmp_path, run_name, **updates):
    return verify_training_cfg(_raw_config(tracked_slp, tmp_path, run_name, **updates))


def _load_epoch(path):
    return torch.load(path, map_location="cpu", weights_only=False)["epoch"]


@pytest.mark.parametrize(
    "min_steps,expected_steps",
    [
        # 96 crops / (P x K = 8) = 12 steps -- not 96 / batch_size 4 = 24.
        (1, 12),
        # The floor applies to the P x K step count (12 -> 20), not to the
        # batch_size-based count it was compared against before (24 > 20).
        (20, 20),
    ],
)
def test_epoch_length_counts_pk_batches(
    tracked_slp, tmp_path, min_steps, expected_steps
):
    """One epoch is one pass over the crops, in P x K steps (FINDINGS #1)."""
    cfg = _config(
        tracked_slp,
        tmp_path,
        "steps",
        **{"trainer_config.min_train_steps_per_epoch": min_steps},
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    loader = trainer.trainer.train_dataloader
    assert len(loader.dataset) == N_CROPS
    assert trainer.config.trainer_config.train_steps_per_epoch == expected_steps
    assert loader.batch_sampler.batches_per_epoch == expected_steps
    assert trainer.trainer.global_step == expected_steps
    assert loader.batch_sampler.samples_per_batch == P * K


def test_early_stopping_waits_for_evals_and_last_ckpt_is_final(tracked_slp, tmp_path):
    """Patience counts evaluations; `last.ckpt` is the last epoch (FINDINGS #4, T3).

    `eval.frequency=2 > patience=1`. Evaluations run at epochs 1, 3, 5, ... and
    `min_delta` is so large that only the first can count as an improvement, so
    the run must stop right after the SECOND evaluation, at epoch 3. Before, the
    NaN placeholder logged at epoch 0 already used up the patience: the run
    stopped before its first evaluation and never wrote `best.ckpt`.
    """
    cfg = _config(
        tracked_slp,
        tmp_path,
        "cadence",
        **{
            "trainer_config.max_epochs": 10,
            "trainer_config.eval.frequency": 2,
            "trainer_config.early_stopping.stop_training_on_plateau": True,
            "trainer_config.early_stopping.patience": 1,
            "trainer_config.early_stopping.min_delta": 10.0,
            "trainer_config.model_ckpt.save_last": True,
        },
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    early_stopping = next(
        c for c in trainer.trainer.callbacks if isinstance(c, EarlyStopping)
    )
    assert early_stopping.stopped_epoch == 3
    run_dir = tmp_path / "cadence"
    assert (run_dir / "best.ckpt").exists()
    # "best" is only ever chosen on an eval epoch.
    assert _load_epoch(run_dir / "best.ckpt") in (1, 3)
    # `last.ckpt` is the final epoch trained, not the last eval / best epoch.
    assert _load_epoch(run_dir / "last.ckpt") == 3


def test_last_ckpt_is_final_epoch_between_evals(tracked_slp, tmp_path):
    """With the last epoch NOT an eval epoch, `last.ckpt` still holds it (T3)."""
    cfg = _config(
        tracked_slp,
        tmp_path,
        "last",
        **{
            "trainer_config.max_epochs": 4,
            "trainer_config.eval.frequency": 3,
            "trainer_config.model_ckpt.save_last": True,
        },
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    run_dir = tmp_path / "last"
    assert _load_epoch(run_dir / "best.ckpt") == 2  # the only eval epoch
    last = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 3
    assert last["global_step"] == trainer.trainer.global_step


def test_no_empty_eval_warning_on_the_sanity_pass(tracked_slp, tmp_path):
    """The sanity pass collects nothing, so it must not warn about it (T3)."""
    from loguru import logger

    messages = []
    cfg = _config(tracked_slp, tmp_path, "sanity")
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    # Added after the trainer build, which reconfigures loguru's sinks.
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        trainer.train()
    finally:
        logger.remove(sink_id)

    assert not any("No embeddings collected" in m for m in messages)


def test_string_head_config_trains(tracked_slp, tmp_path):
    """`head_configs="embedding"` (the `get_head_configs` factory) trains (FINDINGS #6).

    The factory leaves `negatives.sources` at its schema default, which used to be
    `None` and crashed the module constructor with `TypeError`.
    """
    from sleap_nn.train import train

    train(
        train_labels_path=[tracked_slp.as_posix()],
        val_labels_path=[tracked_slp.as_posix()],
        head_configs="embedding",
        backbone_config={
            "unet": {
                "in_channels": 1,
                "filters": 8,
                "max_stride": 32,
                "output_stride": 32,
            }
        },
        crop_size=64,
        max_epochs=1,
        min_train_steps_per_epoch=1,
        trainer_accelerator="cpu",
        trainer_num_devices=1,
        enable_progress_bar=False,
        save_ckpt=True,
        ckpt_dir=tmp_path.as_posix(),
        run_name="string_head",
    )

    saved = OmegaConf.load(tmp_path / "string_head" / "training_config.yaml")
    objective = saved.model_config.head_configs.embedding.embedding.objective
    assert list(objective.negatives.sources) == ["same_frame", "in_batch"]
    assert (tmp_path / "string_head" / "best.ckpt").exists()


# ─────────────────────────────────────────────────────────────────────────
# The objective is validated when the trainer is built -- before any data is
# loaded or cached -- and read the same way by every consumer.
# ─────────────────────────────────────────────────────────────────────────
OBJECTIVE = "model_config.head_configs.embedding.embedding.objective"


@pytest.mark.parametrize(
    "key,typo",
    [
        # Each of these used to train silently on the wrong pairs.
        ("positives.scope", "tracklets"),  # -> aug-view positives only
        ("negatives.sources", ["in-batch"]),  # -> no negatives at all
        ("loss.name", "supcom"),
        ("sampler.kind", "within-video"),  # raised only on the first batch
    ],
)
def test_misspelled_objective_option_fails_at_setup(tracked_slp, tmp_path, key, typo):
    """An unknown objective option is rejected by name (FINDINGS T3)."""
    cfg = _raw_config(tracked_slp, tmp_path, "typo", **{f"{OBJECTIVE}.{key}": typo})
    with pytest.raises(ValueError, match=f"objective.{key}"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_omitted_negative_sources_mean_both(tracked_slp, tmp_path):
    """A `negatives:` block without `sources` gets the default, for every reader.

    `sources` defaulted to `None`, which the module read as "both" -- and crashed
    on (`list(None)`) -- while the identity validator read it as "none" and
    skipped its tracklet/in-batch check. Tracklet scope + `in_batch` +
    `restrict_same_video=False` is the combination that check forbids.
    """
    cfg = _raw_config(
        tracked_slp,
        tmp_path,
        "sources",
        **{
            f"{OBJECTIVE}.positives": {"scope": "tracklet"},
            f"{OBJECTIVE}.negatives": {"restrict_same_video": False},
            "data_config.identity.tracks_are_proofread": True,
        },
    )
    with pytest.raises(ValueError, match="restrict_same_video=True"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_sampler_rejects_an_unknown_kind_when_built():
    """Not on the first batch, after the (possibly long) dataset caching."""
    from sleap_nn.data.custom_datasets import GroupAwareBatchSampler

    ids = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Unknown sampler kind"):
        GroupAwareBatchSampler(ids, np.zeros(4), np.arange(4), kind="within-video")


def test_objective_vocabularies_match_their_consumers():
    """The config's option lists are the ones the loss / sampler dispatch on."""
    from sleap_nn.config.model_config import (
        EMBEDDING_LOSSES,
        EMBEDDING_SAMPLER_KINDS,
    )
    from sleap_nn.data.custom_datasets import GroupAwareBatchSampler
    from sleap_nn.training.losses import _CONTRASTIVE_LOSSES

    assert set(EMBEDDING_LOSSES) == set(_CONTRASTIVE_LOSSES)
    ids = np.array([0, 0, 1, 1])
    for kind in EMBEDDING_SAMPLER_KINDS:
        sampler = GroupAwareBatchSampler(ids, np.zeros(4), np.arange(4), kind=kind)
        assert len(next(iter(sampler))) > 0
