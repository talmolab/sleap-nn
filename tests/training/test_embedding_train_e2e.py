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


# ─────────────────────────────────────────────────────────────────────────
# The post-training retrieval eval (`metrics.<split>.npz`) must group the
# detections exactly as the per-epoch metric that selected the checkpoint.
# ─────────────────────────────────────────────────────────────────────────
FRAMES_PER_VIDEO = 8


@pytest.fixture
def two_video_slp(centered_instance_video, tmp_path):
    """Build two-video labels whose track names are per-video.

    Each video has its OWN `track_0` / `track_1` tracks, so `track_0` names two
    different tracklets. With `identified_video` set, that video's animals also
    carry a `sio.Identity`. Its frames are stored last, so the other video's
    tracklets get tracklet ids 0 and 1, the same numbers as the identity indices.
    """

    def build(identified_video=None):
        skeleton = sio.Skeleton(["a", "b"])
        identities = [sio.Identity(name=f"animal_{i}") for i in range(N_ANIMALS)]
        rng = np.random.default_rng(0)
        videos, tracks, frames = [], [], []
        for v in range(2):
            video = sio.load_video(centered_instance_video.as_posix())
            video_tracks = [sio.Track(name=f"track_{i}") for i in range(N_ANIMALS)]
            videos.append(video)
            tracks += video_tracks
            for frame_idx in range(FRAMES_PER_VIDEO):
                instances = []
                for i, (cx, cy) in enumerate([(120, 180), (260, 200)]):
                    points = np.array([[cx - 10, cy], [cx + 10, cy]], float)
                    instance = sio.Instance.from_numpy(
                        points + rng.normal(0, 2, 2),
                        skeleton=skeleton,
                        track=video_tracks[i],
                    )
                    if v == identified_video:
                        instance.identity = identities[i]
                    instances.append(instance)
                frames.append(
                    sio.LabeledFrame(
                        # Different frames of the (shared) movie per video, so no two
                        # crops are identical.
                        video=video,
                        frame_idx=100 * v + frame_idx,
                        instances=instances,
                    )
                )
        labels = sio.Labels(
            labeled_frames=frames, videos=videos, skeletons=[skeleton], tracks=tracks
        )
        path = tmp_path / f"two_video_{identified_video}.slp"
        labels.save(path.as_posix())
        return path

    return build


def _record_retrieval_evals(monkeypatch):
    """Record every call of the shared retrieval metric, tagged by its caller.

    The per-epoch callback (`_compute_metrics`) and the post-training eval
    (`_run_embedding_split_eval`, tagged with its split name) both call it, so the
    recorded labels are exactly the groups each one scored. Returns the calls and
    the real metric function.
    """
    import sys

    import sleap_nn.evaluation as evaluation

    real = evaluation.embedding_leave_self_out_eval
    calls = []

    def record(emb, y, **kwargs):
        metrics = real(emb, y, **kwargs)
        caller = sys._getframe(1)
        calls.append(
            {
                "caller": caller.f_code.co_name,
                "split": caller.f_locals.get("d_name"),
                "emb": np.asarray(emb, np.float64),
                "y": np.asarray(y),
                "kwargs": kwargs,
                "metrics": metrics,
            }
        )
        return metrics

    monkeypatch.setattr(evaluation, "embedding_leave_self_out_eval", record)
    return calls, real


def _partition(y):
    """The grouping as a set of row sets, independent of the label values."""
    return {frozenset(np.flatnonzero(y == g).tolist()) for g in np.unique(y)}


def _unit(emb):
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


TWO_VIDEO_CROPS = 2 * FRAMES_PER_VIDEO * N_ANIMALS  # 32
TRACKLET = {
    f"{OBJECTIVE}.positives": {"scope": "tracklet"},
    f"{OBJECTIVE}.negatives": {"restrict_same_video": True},
    f"{OBJECTIVE}.sampler.kind": "within_video",
    "data_config.identity": {
        "track_names_are_global": False,
        "tracks_are_proofread": True,
    },
}
GLOBAL_ID = {
    f"{OBJECTIVE}.positives": {"scope": "global_id"},
    "data_config.identity": {"track_names_are_global": False},
}


@pytest.mark.parametrize(
    "identified_video,objective,n_crops,n_groups",
    [
        # Per-video names, no identities: each (video, track) is its own group.
        # The eval grouped by track NAME, merging `track_0` of both videos (2 groups).
        pytest.param(None, TRACKLET, TWO_VIDEO_CROPS, 4, id="tracklet-per-video"),
        # Video 1 identified, video 0 bare tracklets: 2 identities + 2 tracklets.
        # The per-epoch metric keyed a bare tracklet by its tracklet id, which
        # collided with the identity index: tracklet k == identity k (2 groups).
        pytest.param(1, TRACKLET, TWO_VIDEO_CROPS, 4, id="tracklet-mixed"),
        # `global_id` without `track_names_are_global`: only identified detections
        # are samples. The eval also scored video 0 by track name (32 crops,
        # 4 groups).
        pytest.param(1, GLOBAL_ID, TWO_VIDEO_CROPS // 2, 2, id="global-id"),
    ],
)
def test_post_training_eval_groups_like_the_selection_metric(
    two_video_slp,
    tmp_path,
    monkeypatch,
    identified_video,
    objective,
    n_crops,
    n_groups,
):
    """The reported retrieval metric scores the same groups as the per-epoch one.

    Goes through `sleap_nn.train.run_training` (what `sleap-nn train` calls): one
    epoch, so `best.ckpt` holds the weights the per-epoch metric was computed with,
    then the post-training eval of `train.0` / `val.0` (FINDINGS #7, #8).
    """
    from sleap_nn.train import run_training

    slp = two_video_slp(identified_video)
    cfg = _config(
        slp,
        tmp_path,
        "grouping",
        **{"trainer_config.val_data_loader.batch_size": 8, **objective},
    )
    calls, retrieval_metric = _record_retrieval_evals(monkeypatch)
    run_training(cfg)

    per_epoch = [c for c in calls if c["caller"] == "_compute_metrics"]
    post = [c for c in calls if c["caller"] == "_run_embedding_split_eval"]
    assert len(per_epoch) == 1  # max_epochs=1
    # train.0 and val.0 (both are `slp`)
    assert sorted(c["split"] for c in post) == ["train.0", "val.0"]
    selection = per_epoch[0]
    assert len(selection["y"]) == n_crops
    assert len(np.unique(selection["y"])) == n_groups
    for split in post:
        assert len(split["y"]) == n_crops
        assert len(np.unique(split["y"])) == n_groups
        # Same crops in the same order, embedded by the same weights...
        np.testing.assert_allclose(
            _unit(split["emb"]), _unit(selection["emb"]), atol=1e-4
        )
        # ...grouped the same way.
        assert _partition(split["y"]) == _partition(selection["y"])

    # The metric values are NOT compared across the two forward passes: after one
    # epoch the tiny model has near-tied similarities, and float noise within the
    # tolerance above flips rankings differently per platform. Each check below
    # stays within one pass.
    val = next(c for c in post if c["split"] == "val.0")
    # The saved headline is the metric of the post-training pass's own crops and
    # groups...
    saved = np.load(tmp_path / "grouping" / "metrics.val.0.npz")
    own = retrieval_metric(val["emb"], val["y"], **val["kwargs"])
    for key in ("rank1", "mAP", "auc", "eer", "knn_acc"):
        assert float(saved[key]) == pytest.approx(own[key], abs=1e-6)
    # ...and its groups are interchangeable with the selection metric's: scoring
    # the per-epoch embeddings with the post-training labels reproduces the
    # selection metric exactly.
    swapped = retrieval_metric(selection["emb"], val["y"], **selection["kwargs"])
    assert swapped == selection["metrics"]


# ─────────────────────────────────────────────────────────────────────────
# The per-epoch retrieval eval and the embedding scatter (emb-review F2).
# ─────────────────────────────────────────────────────────────────────────
SMALL_VAL_FRAMES = [(0, [0]), (1, [0]), (2, [0, 1])]  # (frame_idx, animals present)
SMALL_VAL_CROPS = 4


@pytest.fixture
def small_val_slp(centered_instance_video, tmp_path):
    """Four val crops: `animal_0` in three frames, `animal_1` in only one.

    `animal_1` is a singleton identity: its one crop has nothing to retrieve.
    """
    video = sio.load_video(centered_instance_video.as_posix())
    skeleton = sio.Skeleton(["a", "b"])
    tracks = [sio.Track(name=f"track_{i}") for i in range(N_ANIMALS)]
    identities = [sio.Identity(name=f"animal_{i}") for i in range(N_ANIMALS)]
    centers = [(120, 180), (260, 200)]
    frames = []
    for frame_idx, animals in SMALL_VAL_FRAMES:
        instances = []
        for i in animals:
            cx, cy = centers[i]
            instance = sio.Instance.from_numpy(
                np.array([[cx - 10, cy], [cx + 10, cy]], float),
                skeleton=skeleton,
                track=tracks[i],
            )
            instance.identity = identities[i]
            instances.append(instance)
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    labels = sio.Labels(
        labeled_frames=frames, videos=[video], skeletons=[skeleton], tracks=tracks
    )
    path = tmp_path / "small_val.slp"
    labels.save(path.as_posix())
    return path


def test_per_epoch_retrieval_eval_leaves_out_singleton_identities(
    tracked_slp, small_val_slp, tmp_path, monkeypatch
):
    """The selection metric scores only the val crops that have a positive (F8).

    `animal_1`'s one val crop has no other crop of its identity to retrieve. It
    used to count as a rank-1 / kNN miss (while mAP dropped it); it is now left
    out of all three and counted.
    """
    cfg = _config(
        tracked_slp,
        tmp_path,
        "singleton",
        **{"data_config.val_labels_path": [small_val_slp.as_posix()]},
    )
    calls, _ = _record_retrieval_evals(monkeypatch)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    per_epoch = [c for c in calls if c["caller"] == "_compute_metrics"]
    assert len(per_epoch) == 1
    assert len(per_epoch[0]["y"]) == SMALL_VAL_CROPS
    metrics = per_epoch[0]["metrics"]
    # A fraction of the 3 scored queries, not of all 4 crops...
    assert metrics["rank1"] * 3 == pytest.approx(round(metrics["rank1"] * 3), abs=1e-3)
    # ...it is the value the checkpoint is selected on...
    selected = float(trainer.trainer.callback_metrics["eval/val/rank1"])
    assert selected == pytest.approx(metrics["rank1"], abs=1e-6)
    # ...and the left-out crop is counted.
    assert metrics["n_no_positive_queries"] == 1


def test_embedding_scatter_reuses_the_val_pass_and_plots_clean_crops(
    tracked_slp, small_val_slp, tmp_path, monkeypatch
):
    """The scatter plots the val pass's own embeddings and un-augmented crops (F7).

    It used to re-decode and re-embed up to 256 val crops every epoch, and plot the
    TRAIN crops as the augmented contrastive view the training dataset emits, all
    in one forward pass. Now: the val points are the embeddings the validation pass
    just computed, and the train crops are decoded once, un-augmented, and kept.
    Few points (4 val crops) must work too, with wandb off.
    """
    from sleap_nn.data.custom_datasets import EmbeddingDataset
    from sleap_nn.training.callbacks import (
        EmbeddingEvaluationCallback,
        UnifiedVizCallback,
    )

    in_viz = [False]
    reads = []  # `apply_aug` of every crop the scatter decodes
    scatters = []  # per scatter: embeddings, groups, n_train, the val pass's output

    real_getitem = EmbeddingDataset.__getitem__

    def getitem(self, index):
        if in_viz[0]:
            reads.append(bool(self.apply_aug))
        return real_getitem(self, index)

    real_epoch_end = UnifiedVizCallback.on_train_epoch_end

    def on_train_epoch_end(self, trainer, pl_module):
        evaluator = next(
            c for c in trainer.callbacks if isinstance(c, EmbeddingEvaluationCallback)
        )
        scatters.append({"val_pass": getattr(evaluator, "last_val_embeddings", None)})
        in_viz[0] = True
        try:
            real_epoch_end(self, trainer, pl_module)
        finally:
            in_viz[0] = False

    real_reduce = UnifiedVizCallback._reduce_to_2d
    real_render = UnifiedVizCallback._render_embedding_scatter

    def reduce_to_2d(x):
        scatters[-1]["embedding"] = np.array(x)
        return real_reduce(x)

    def render(emb2d, groups, n_train, epoch, method):
        scatters[-1].update(groups=np.array(groups), n_train=n_train)
        return real_render(emb2d, groups, n_train, epoch, method)

    monkeypatch.setattr(EmbeddingDataset, "__getitem__", getitem)
    monkeypatch.setattr(UnifiedVizCallback, "on_train_epoch_end", on_train_epoch_end)
    monkeypatch.setattr(UnifiedVizCallback, "_reduce_to_2d", staticmethod(reduce_to_2d))
    monkeypatch.setattr(
        UnifiedVizCallback, "_render_embedding_scatter", staticmethod(render)
    )

    cfg = _config(
        tracked_slp,
        tmp_path,
        "scatter",
        **{
            "data_config.val_labels_path": [small_val_slp.as_posix()],
            # The training views are really augmented.
            "data_config.augmentation_config": {
                "geometric": {"rotation_min": -180.0, "rotation_max": 180.0}
            },
            "trainer_config.max_epochs": 2,
            "trainer_config.visualize_preds_during_training": True,
            "trainer_config.keep_viz": True,
        },
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    # Over both epochs, the scatter decoded each train crop once, un-augmented, and
    # no val crop at all.
    assert not any(reads)
    assert len(reads) == N_CROPS
    assert len(scatters) == 2  # one per epoch
    for epoch, scatter in enumerate(scatters):
        val_pass = scatter["val_pass"]
        assert val_pass["epoch"] == epoch
        n_train = scatter["n_train"]
        assert n_train == N_CROPS  # < 256: every train crop
        # The val points ARE the val pass's embeddings (no second forward).
        np.testing.assert_array_equal(
            scatter["embedding"][n_train:], val_pass["embedding"]
        )
        np.testing.assert_array_equal(scatter["groups"][n_train:], val_pass["label"])
        assert len(val_pass["label"]) == SMALL_VAL_CROPS
    viz_dir = tmp_path / "scatter" / "viz"
    assert (viz_dir / "embedding_scatter.0000.png").exists()
    assert (viz_dir / "embedding_scatter.0001.png").exists()

    # An epoch with no validation pass to reuse embeds clean val crops instead.
    viz = next(
        c for c in trainer.trainer.callbacks if isinstance(c, UnifiedVizCallback)
    )
    reads.clear()
    in_viz[0] = True
    scatters.append({})
    try:
        viz._embedding_viz_epoch(99, None, trainer=trainer.trainer)
    finally:
        in_viz[0] = False
    assert len(reads) == SMALL_VAL_CROPS and not any(reads)
    assert len(scatters[-1]["embedding"]) == N_CROPS + SMALL_VAL_CROPS
    assert (viz_dir / "embedding_scatter.0099.png").exists()
