"""What an `embedding` model trains on (embedding-stack review, PR D2).

FINDINGS #9, #10, #11, #21, #22, #23 and the Tier-3 data items. Every test goes
through a real entry point -- `ModelTrainer.get_model_trainer_from_config` (setup:
split, crop sizing, objective checks), `ModelTrainer.train` (the datasets and loader
the model actually trains on) or `sleap_nn.train.run_training` (plus the
post-training eval) -- and each one fails on the code before this PR.
"""

import math

import numpy as np
import pytest
import sleap_io as sio
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.training.model_trainer import ModelTrainer
from tests.training.test_embedding_train_e2e import (  # noqa: F401
    FRAMES_PER_VIDEO,
    OBJECTIVE,
    TRACKLET,
    _config,
    _raw_config,
    _record_retrieval_evals,
    two_video_slp,
)

N_FRAMES = 24
SIZE = 384  # centered_pair_small.mp4 is 384 x 384


@pytest.fixture
def warnings_log():
    """Collect loguru WARNING+ messages (whitespace-collapsed)."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(" ".join(str(m).split())))
    yield messages
    logger.remove(sink_id)


def _instance(cx, cy, skeleton, rng, *, predicted=False, track=None, identity=None):
    points = np.array([[cx - 10, cy], [cx + 10, cy]], float) + rng.normal(0, 2, 2)
    if predicted:
        inst = sio.PredictedInstance.from_numpy(
            points, skeleton=skeleton, score=0.9, track=track
        )
    else:
        inst = sio.Instance.from_numpy(points, skeleton=skeleton, track=track)
    inst.identity = identity
    return inst


def _rect_mask(x0, y0, w, h, *, predicted=False, track=None, identity=None):
    arr = np.zeros((SIZE, SIZE), bool)
    arr[y0 : y0 + h, x0 : x0 + w] = True
    cls = sio.PredictedSegmentationMask if predicted else sio.UserSegmentationMask
    mask = cls.from_numpy(arr)
    mask.track = track
    mask.identity = identity
    return mask


def _save(labels, tmp_path, name):
    path = tmp_path / f"{name}.slp"
    labels.save(path.as_posix())
    return path


def _mask_labels(video, *, extra=lambda fi, ids, tracks: [], tracked=True, size=24):
    """Two animals as ``size`` px squares on every frame, carrying a sio.Identity."""
    ids = [sio.Identity(name=f"animal_{i}") for i in range(2)]
    tracks = [sio.Track(name=f"track_{i}") for i in range(2)]
    frames = []
    for fi in range(N_FRAMES):
        masks = [
            _rect_mask(
                x0,
                150,
                size,
                size,
                track=tracks[i] if tracked else None,
                identity=ids[i],
            )
            for i, x0 in enumerate((100, 240))
        ]
        masks += extra(fi, ids, tracks)
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=fi, instances=[], masks=masks)
        )
    return sio.Labels(
        labeled_frames=frames,
        videos=[video],
        skeletons=[],
        tracks=tracks if tracked else [],
    )


def _train(cfg):
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    return trainer, trainer.trainer.train_dataloader.dataset


def _mask_config(slp, tmp_path, run_name, **updates):
    return _config(
        slp,
        tmp_path,
        run_name,
        **{"data_config.identity.track_names_are_global": False, **updates},
    )


# ── D3: `user_instances_only` drops SUPERSEDED predictions (FINDINGS #9) ────────
#
# For the embedding model the flag keeps every user label and every prediction no user
# label on the same frame supersedes (sleap-io's `unused_predictions` /
# `unused_predicted_masks`): re-ID data is often predicted poses with proofread tracks,
# which stay `PredictedInstance`s.


def _pose_labels(video, frames_spec):
    """Two animals per frame; ``frames_spec[fi]`` says which carriers each frame has.

    ``"user"``: user instances; ``"predicted"``: predictions only; ``"both"``: the user
    instances plus a predicted duplicate of each animal carrying the SAME track (the
    merged-predictions case).
    """
    skeleton = sio.Skeleton(["a", "b"])
    tracks = [sio.Track(name=f"track_{i}") for i in range(2)]
    rng = np.random.default_rng(0)
    frames = []
    for fi, kind in enumerate(frames_spec):
        instances = []
        for i, cx in enumerate((120, 260)):
            if kind in ("user", "both"):
                instances.append(_instance(cx, 180, skeleton, rng, track=tracks[i]))
            if kind in ("predicted", "both"):
                instances.append(
                    _instance(cx, 180, skeleton, rng, predicted=True, track=tracks[i])
                )
        frames.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=instances))
    return sio.Labels(
        labeled_frames=frames, videos=[video], skeletons=[skeleton], tracks=tracks
    )


@pytest.mark.parametrize("val_split", ["val_labels_path", "validation_fraction"])
def test_tracked_predictions_train_under_the_default(
    centered_instance_video, tmp_path, val_split
):
    """Predicted poses with proofread tracks, and no user instance at all.

    Honoring `user_instances_only=True` (the default) as "no predictions" made this
    file untrainable ("No labeled frames available"), also through the random
    `validation_fraction` split, which removed every prediction.
    """
    video = sio.load_video(centered_instance_video.as_posix())
    labels = _pose_labels(video, ["predicted"] * N_FRAMES)
    slp = _save(labels, tmp_path, "tracked_predictions")
    updates = {"trainer_config.seed": 0}
    if val_split == "validation_fraction":
        updates["data_config.val_labels_path"] = None
    trainer, dataset = _train(_config(slp, tmp_path, "tracked_preds", **updates))

    n_train = sum(len(lf.instances) for lf in trainer.train_labels[0])
    assert n_train > 0
    kinds = {type(m["mask_obj"]).__name__ for m in dataset.mask_idx_list}
    assert kinds == {"PredictedInstance"}
    assert len(dataset) == n_train  # every tracked prediction is a sample


@pytest.mark.parametrize(
    "rank",
    [
        # A single process. Before this PR rank 0 also rewrote the training labels'
        # frames to their user instances while saving `labels_gt.*.slp`, which hid
        # the bug here -- and made rank 0 train on different data than the others.
        -1,
        # A DDP worker, which never ran that rewrite: the dataset trained on the
        # predicted duplicates.
        1,
    ],
)
def test_prediction_superseded_by_a_user_instance_is_not_trained_on(
    centered_instance_video, tmp_path, monkeypatch, rank
):
    """A merged predictions file: a user `Instance` and a `PredictedInstance` of the
    same animal (same track) on one frame are one sample, the user one. Predictions
    on frames nobody proofread stay samples."""
    monkeypatch.setattr("sleap_nn.training.model_trainer.RANK", rank)
    # Under DDP, rank 0 creates the run directory; this process stands in for a
    # worker, so create it here.
    (tmp_path / "merged").mkdir()
    video = sio.load_video(centered_instance_video.as_posix())
    half = N_FRAMES // 2
    labels = _pose_labels(video, ["both"] * half + ["predicted"] * half)
    slp = _save(labels, tmp_path, "merged")

    _, dataset = _train(_config(slp, tmp_path, "merged"))

    samples = [
        (m["frame_idx"], type(m["mask_obj"]).__name__) for m in dataset.mask_idx_list
    ]
    assert sorted(samples) == sorted(
        [(fi, "Instance") for fi in range(half) for _ in range(2)]
        + [(fi, "PredictedInstance") for fi in range(half, N_FRAMES) for _ in range(2)]
    )


def test_superseded_predicted_masks_are_not_trained_on_or_scored(
    centered_instance_video, tmp_path, monkeypatch
):
    """The mask analogue, through training, the per-epoch eval and the post-training
    eval (`embed_labels_for_eval`).

    Each frame holds the two user masks, a predicted duplicate of animal 0 (the user
    mask adopted it: same place) labelled as animal 1, and a prediction of a third
    animal that no user mask adopted.
    """
    from sleap_nn.train import run_training

    video = sio.load_video(centered_instance_video.as_posix())
    third = sio.Identity(name="animal_2")

    def predictions(fi, ids, tracks):
        return [
            _rect_mask(100, 150, 24, 24, predicted=True, identity=ids[1]),
            _rect_mask(300, 300, 24, 24, predicted=True, identity=third),
        ]

    slp = _save(_mask_labels(video, extra=predictions), tmp_path, "pred_masks")
    cfg = _mask_config(slp, tmp_path, "pred_masks")
    calls, _ = _record_retrieval_evals(monkeypatch)
    run_training(cfg)

    per_epoch = [c for c in calls if c["caller"] == "_compute_metrics"]
    post = [c for c in calls if c["caller"] == "_run_embedding_split_eval"]
    assert per_epoch and post
    for call in per_epoch + post:
        # 2 user masks + the unadopted prediction per frame; the duplicate is out.
        assert len(call["y"]) == 3 * N_FRAMES
        assert len(np.unique(call["y"])) == 3


def test_random_split_of_user_labels_is_unchanged(centered_instance_video, tmp_path):
    """Without predictions, the embedding split is `make_training_splits`' split."""
    video = sio.load_video(centered_instance_video.as_posix())
    labels = _mask_labels(video)
    slp = _save(labels, tmp_path, "user_masks")
    trainer = ModelTrainer.get_model_trainer_from_config(
        _mask_config(
            slp,
            tmp_path,
            "split",
            **{"data_config.val_labels_path": None, "trainer_config.seed": 0},
        )
    )

    train, val = sio.load_slp(slp.as_posix()).make_training_splits(
        n_train=0.9, n_val=0.1, seed=0
    )
    assert [lf.frame_idx for lf in trainer.train_labels[0]] == [
        lf.frame_idx for lf in train
    ]
    assert [lf.frame_idx for lf in trainer.val_labels[0]] == [
        lf.frame_idx for lf in val
    ]


# ── D4: auto crop_size for mask-mode data sizes from the masks (FINDINGS #10) ──


def _big_mask_slp(centered_instance_video, tmp_path):
    video = sio.load_video(centered_instance_video.as_posix())
    frames = []
    ids = [sio.Identity(name=f"animal_{i}") for i in range(2)]
    for fi in range(N_FRAMES):
        masks = [
            # 200 x 120 px and 60 x 60 px animals.
            _rect_mask(20, 40, 200, 120, identity=ids[0]),
            _rect_mask(250, 250, 60, 60, identity=ids[1]),
        ]
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=fi, instances=[], masks=masks)
        )
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])
    return _save(labels, tmp_path, "big_masks")


def test_auto_crop_size_holds_the_masks(centered_instance_video, tmp_path):
    """Mask-only data has no keypoints: the crop fell back to `min_crop_size`."""
    slp = _big_mask_slp(centered_instance_video, tmp_path)
    cfg = _mask_config(slp, tmp_path, "auto_crop")
    OmegaConf.update(cfg, "data_config.preprocessing.crop_size", None)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)

    crop_size = trainer.config.data_config.preprocessing.crop_size
    # A crop centered on the 200 x 120 mask's center must span its 200 px width.
    assert crop_size >= 200
    assert crop_size % 16 == 0


def test_configured_crop_size_that_clips_masks_warns(
    centered_instance_video, tmp_path, warnings_log
):
    """The clip warning counted keypoints only, so clipped masks went unreported."""
    slp = _big_mask_slp(centered_instance_video, tmp_path)
    ModelTrainer.get_model_trainer_from_config(
        _mask_config(slp, tmp_path, "clip")  # crop_size 64
    )

    clip = [m for m in warnings_log if "crop size 64px clips" in m]
    assert len(clip) == 1
    # Every 200 x 120 mask is clipped (train and val are the same file); the
    # 60 x 60 ones fit.
    assert f"clips {2 * N_FRAMES} of {4 * N_FRAMES} masks" in clip[0]


# ── D5: the group-aware split keeps masks with their instances (FINDINGS #11) ──


def test_split_keeps_linked_masks_and_their_links(centered_instance_video, tmp_path):
    """Pose + linked-mask frames lost every mask once `data_config.split` was set,
    which flipped the model from mask to pose mode."""
    video = sio.load_video(centered_instance_video.as_posix())
    skeleton = sio.Skeleton(["a", "b"])
    ids = [sio.Identity(name=f"animal_{i}") for i in range(2)]
    rng = np.random.default_rng(0)
    frames = []
    for fi in range(N_FRAMES):
        instances, masks = [], []
        for i, cx in enumerate((120, 260)):
            inst = _instance(cx, 180, skeleton, rng, identity=ids[i])
            mask = _rect_mask(cx - 12, 168, 24, 24, identity=ids[i])
            mask.instance = inst
            instances.append(inst)
            masks.append(mask)
        frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=fi, instances=instances, masks=masks
            )
        )
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])
    slp = _save(labels, tmp_path, "linked")
    cfg = _mask_config(
        slp,
        tmp_path,
        "split",
        **{
            "data_config.val_labels_path": None,
            "data_config.split": {"split_by": "frame", "n_folds": 2},
        },
    )
    trainer, dataset = _train(cfg)

    train_labels = trainer.train_labels[0]
    n_masks = sum(len(lf.masks) for lf in train_labels)
    n_instances = sum(len(lf.instances) for lf in train_labels)
    assert n_masks == n_instances > 0
    for lf in train_labels:
        for mask in lf.masks:
            assert any(mask.instance is inst for inst in lf.instances)
    assert {i.name for i in train_labels.identities} == {"animal_0", "animal_1"}
    # A tie between carriers goes to masks, as it did before the split existed.
    assert dataset.detection_mode == "mask"
    assert len(dataset) == n_masks


# ── D6: the carrier -- membership counting (#22) and the recorded carrier ──────


def test_identity_only_masks_train_in_mask_mode(centered_instance_video, tmp_path):
    """Masks carrying a sio.Identity but no track (the sleap-io #535 style).

    Mode detection counted only TRACKED detections, so it chose pose mode and
    the dataset came out empty.
    """
    from sleap_nn.train import run_training

    video = sio.load_video(centered_instance_video.as_posix())
    slp = _save(_mask_labels(video, tracked=False), tmp_path, "identity_only")
    cfg = _mask_config(slp, tmp_path, "identity_only")
    run_training(cfg)

    saved = OmegaConf.load(tmp_path / "identity_only" / "training_config.yaml")
    # The carrier is recorded, so inference embeds the same one.
    assert saved.model_config.head_configs.embedding.embedding.detection_mode == "mask"
    assert (tmp_path / "identity_only" / "best.ckpt").exists()


# ── D7: `aug_view` is self-supervised (#21) ──────────────────────────────────


def _unlabeled_pose_slp(centered_instance_video, tmp_path):
    video = sio.load_video(centered_instance_video.as_posix())
    skeleton = sio.Skeleton(["a", "b"])
    rng = np.random.default_rng(0)
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=fi,
            instances=[_instance(cx, 180, skeleton, rng) for cx in (120, 260)],
        )
        for fi in range(N_FRAMES)
    ]
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])
    return _save(labels, tmp_path, "unlabeled")


AUG_VIEW = {
    f"{OBJECTIVE}.positives": {"scope": "aug_view"},
    f"{OBJECTIVE}.sampler": {"kind": "random", "groups_per_batch": 2},
    "data_config.identity.track_names_are_global": False,
}


def test_aug_view_trains_on_unlabeled_data(centered_instance_video, tmp_path):
    """No tracks, no identities: every detection is a sample and its own group."""
    slp = _unlabeled_pose_slp(centered_instance_video, tmp_path)
    _, dataset = _train(_config(slp, tmp_path, "aug_view", **AUG_VIEW))

    assert len(dataset) == 2 * N_FRAMES
    assert len(np.unique(dataset.group_ids)) == 2 * N_FRAMES


def test_aug_view_with_identical_views_is_rejected(centered_instance_video, tmp_path):
    """With no augmentation the two views are one tensor: nothing to learn."""
    slp = _unlabeled_pose_slp(centered_instance_video, tmp_path)
    cfg = _raw_config(
        slp,
        tmp_path,
        "no_views",
        **{**AUG_VIEW, "data_config.use_augmentations_train": False},
    )
    with pytest.raises(ValueError, match="two different views"):
        ModelTrainer.get_model_trainer_from_config(cfg)


# ── D8: empty / tiny masks are not samples ───────────────────────────────────


def test_empty_and_tiny_masks_are_not_trained_on(centered_instance_video, tmp_path):
    """A zero-area mask cropped the image center and labelled it with its
    identity; a few-pixel one standardized over a handful of pixels."""
    video = sio.load_video(centered_instance_video.as_posix())

    def slips(fi, ids, tracks):
        if fi != 0:
            return []
        empty = _rect_mask(0, 0, 0, 0, track=tracks[0], identity=ids[0])
        tiny = _rect_mask(300, 300, 2, 2, track=tracks[1], identity=ids[1])
        return [empty, tiny]

    slp = _save(_mask_labels(video, extra=slips), tmp_path, "slips")
    _, dataset = _train(_mask_config(slp, tmp_path, "slips"))

    assert len(dataset) == 2 * N_FRAMES
    assert all(m["mask_obj"].area >= 16 for m in dataset.mask_idx_list)


# ── D9: an objective that can never see a negative is an error (#23) ─────────


def test_single_identity_objective_is_rejected(centered_instance_video, tmp_path):
    """One animal: every pair is a positive, so the loss has nothing to contrast."""
    video = sio.load_video(centered_instance_video.as_posix())
    identity = sio.Identity(name="animal_0")
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=fi,
            instances=[],
            masks=[_rect_mask(100, 150, 24, 24, identity=identity)],
        )
        for fi in range(N_FRAMES)
    ]
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[])
    slp = _save(labels, tmp_path, "one_animal")
    cfg = _raw_config(
        slp, tmp_path, "one", **{"data_config.identity.track_names_are_global": False}
    )
    with pytest.raises(ValueError, match="never see a negative pair"):
        ModelTrainer.get_model_trainer_from_config(cfg)


def test_one_track_per_video_under_within_video_is_rejected(
    centered_instance_video, tmp_path
):
    """`within_video` draws a batch from one video: one track each -> no negative."""
    skeleton = sio.Skeleton(["a", "b"])
    rng = np.random.default_rng(0)
    videos, tracks, frames = [], [], []
    for v in range(2):
        video = sio.load_video(centered_instance_video.as_posix())
        track = sio.Track(name="track_0")
        videos.append(video)
        tracks.append(track)
        frames += [
            sio.LabeledFrame(
                video=video,
                frame_idx=100 * v + fi,
                instances=[_instance(120, 180, skeleton, rng, track=track)],
            )
            for fi in range(FRAMES_PER_VIDEO)
        ]
    labels = sio.Labels(
        labeled_frames=frames, videos=videos, skeletons=[skeleton], tracks=tracks
    )
    slp = _save(labels, tmp_path, "one_track")
    cfg = _raw_config(slp, tmp_path, "one_track", **TRACKLET)
    with pytest.raises(ValueError, match="restrict_same_video=True only pairs"):
        ModelTrainer.get_model_trainer_from_config(cfg)


# ── D10: a `within_video` epoch is one pass over the crops ────────────────────


def test_within_video_epoch_counts_its_real_batch_size(two_video_slp, tmp_path):
    """2 animals per video, P=4: each batch holds 2 x K crops, not P x K.

    Counting it as P x K made an epoch cover half the data here (a quarter with
    the default P=8).
    """
    P, K = 4, 4
    slp = two_video_slp()
    cfg = _config(
        slp,
        tmp_path,
        "within_video",
        **{
            **TRACKLET,
            f"{OBJECTIVE}.sampler.groups_per_batch": P,
            f"{OBJECTIVE}.sampler.samples_per_group": K,
        },
    )
    trainer, dataset = _train(cfg)

    sampler = trainer.trainer.train_dataloader.batch_sampler
    batch_sizes = {len(batch) for batch in sampler}
    assert batch_sizes == {2 * K}
    assert sampler.samples_per_batch == 2 * K
    expected = math.ceil(len(dataset) / (2 * K))
    assert trainer.config.trainer_config.train_steps_per_epoch == expected
    assert trainer.trainer.global_step == expected


# ── Embedding-specific defaults: warnings only ───────────────────────────────


def test_random_frame_validation_split_warns(
    centered_instance_video, tmp_path, warnings_log
):
    """Adjacent near-identical frames on both sides inflate the selection rank-1."""
    video = sio.load_video(centered_instance_video.as_posix())
    slp = _save(_mask_labels(video), tmp_path, "random_split")
    ModelTrainer.get_model_trainer_from_config(
        _mask_config(
            slp, tmp_path, "random_split", **{"data_config.val_labels_path": None}
        )
    )

    assert any("splits FRAMES at random" in m for m in warnings_log)
