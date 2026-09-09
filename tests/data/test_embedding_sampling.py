"""Cross-file bookkeeping for embedding training.

Two things the group-aware sampler and the contrastive masks depend on:

- **Video identity must be unique across labels files.** `video_idx` indexes ONE
  file's `labels.videos`, so video 0 of file A and video 0 of file B collided:
  a cross-file pair at the same frame looked like a same-video same-frame pair,
  which `restrict_same_video=True` treats as a known negative and the batch
  sampler treats as satisfying its same-video guarantee.
- **An epoch's step count must reflect the sampler's batch size.** The
  group-aware batch sampler yields P*K crops per step, not
  `train_data_loader.batch_size`.
"""

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import OmegaConf

from sleap_nn.data.custom_datasets import EmbeddingDataset

_SKEL = sio.Skeleton(nodes=["a"], name="s")
_HEAD = OmegaConf.create(
    {
        "embedding_dim": 16,
        "output_stride": 16,
        "objective": {
            "positives": {"scope": "tracklet", "aug_views": 2},
            "negatives": {
                "sources": ["in_batch"],
                "restrict_same_video": True,
            },
        },
    }
)


def _one_video_labels(video_name, track_names, n_frames=2):
    """One video, `track_names` tracked instances per frame."""
    video = sio.Video.from_filename(video_name)
    tracks = [sio.Track(name=name) for name in track_names]
    frames = [
        sio.LabeledFrame(
            video=video,
            frame_idx=fi,
            instances=[
                sio.Instance.from_numpy(
                    np.array([[float(i), float(fi)]]), skeleton=_SKEL, track=track
                )
                for i, track in enumerate(tracks)
            ],
        )
        for fi in range(n_frames)
    ]
    return sio.Labels(
        videos=[video], labeled_frames=frames, skeletons=[_SKEL], tracks=tracks
    )


def _dataset(labels_list):
    return EmbeddingDataset(
        labels=labels_list,
        crop_size=16,
        class_names=[],
        embedding_head_config=_HEAD,
        max_stride=16,
        id_scope="tracklet",
        cache_img=None,
    )


def test_video_ids_are_unique_across_labels_files():
    """Video 0 of each file must not share a video id."""
    dataset = _dataset(
        [
            _one_video_labels("a.mp4", ["t0", "t1"]),
            _one_video_labels("b.mp4", ["t0", "t1"]),
        ]
    )

    # Both files' frames index video 0 within their own file...
    per_file = {m["video_idx"] for m in dataset.mask_idx_list}
    assert per_file == {0}
    # ...but the sampler/mask id distinguishes them.
    assert len(np.unique(dataset.video_ids)) == 2

    file_of = np.array([m["labels_idx"] for m in dataset.mask_idx_list])
    for video_id in np.unique(dataset.video_ids):
        files = np.unique(file_of[dataset.video_ids == video_id])
        assert len(files) == 1, "a video id spans two labels files"


def test_video_ids_are_stable_within_a_file():
    """Crops of the same video share one id, whichever file they came from."""
    dataset = _dataset(
        [
            _one_video_labels("a.mp4", ["t0", "t1"], n_frames=3),
            _one_video_labels("b.mp4", ["t0"], n_frames=3),
        ]
    )

    file_of = np.array([m["labels_idx"] for m in dataset.mask_idx_list])
    assert len(np.unique(dataset.video_ids[file_of == 0])) == 1
    assert len(np.unique(dataset.video_ids[file_of == 1])) == 1


def test_single_file_video_ids_match_the_video_index():
    """With one file the ids are unchanged, so existing runs are unaffected."""
    dataset = _dataset([_one_video_labels("a.mp4", ["t0", "t1"])])

    assert np.array_equal(
        dataset.video_ids, np.array([m["video_idx"] for m in dataset.mask_idx_list])
    )


def test_sample_carries_the_global_video_id():
    """The loss keys same-frame / same-video on the global id."""
    dataset = _dataset(
        [
            _one_video_labels("a.mp4", ["t0"]),
            _one_video_labels("b.mp4", ["t0"]),
        ]
    )

    ids = set()
    for i in range(len(dataset.mask_idx_list)):
        meta = dataset.mask_idx_list[i]
        assert "video_id" in meta
        ids.add(meta["video_id"])
    assert len(ids) == 2


# ─────────────────────────────────────────────────────────────────────────
# steps_per_epoch must use the sampler's batch size (P*K), not batch_size
# ─────────────────────────────────────────────────────────────────────────
def test_steps_per_epoch_uses_pk_for_embedding_datasets(config):
    """An epoch must cover the data once, at P*K crops per step.

    Dividing by `train_data_loader.batch_size` overstated the step count by
    P*K/batch_size — with P=8, K=16 and batch_size=4 an "epoch" walked the
    dataset 32 times.
    """
    from sleap_nn.data.custom_datasets import get_train_val_dataloaders

    dataset = _dataset([_one_video_labels("a.mp4", [f"t{i}" for i in range(8)], 8)])
    assert len(dataset.mask_idx_list) == 64

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.trainer_config.train_data_loader.batch_size = 4
    cfg.trainer_config.train_data_loader.num_workers = 0
    cfg.trainer_config.val_data_loader.batch_size = 4
    cfg.trainer_config.val_data_loader.num_workers = 0
    cfg.trainer_config.train_steps_per_epoch = None
    cfg.trainer_config.val_steps_per_epoch = None
    heads = {key: None for key in cfg.model_config.head_configs}
    heads["embedding"] = {
        "embedding": {
            "embedding_dim": 16,
            "output_stride": 16,
            "objective": {
                "positives": {"scope": "tracklet", "aug_views": 2},
                "negatives": {"sources": ["in_batch"], "restrict_same_video": True},
                "sampler": {
                    "kind": "pk",
                    "groups_per_batch": 2,
                    "samples_per_group": 4,
                },
            },
        }
    }
    cfg.model_config.head_configs = OmegaConf.create(heads)

    train_loader, _ = get_train_val_dataloaders(
        train_dataset=dataset, val_dataset=dataset, config=cfg
    )

    # 64 crops at P*K = 8 per step -> 8 steps, not 64/4 = 16.
    assert train_loader.batch_sampler.batches_per_epoch == 8


# ─────────────────────────────────────────────────────────────────────────
# `EmbeddingDataset.__getitem__` — never executed by the suite before
# ─────────────────────────────────────────────────────────────────────────
def _tracked_pose_labels(minimal_instance, n_identities=2):
    """The embedded fixture, with tracks + identities attached in memory."""
    labels = sio.load_slp(minimal_instance)
    tracks = [sio.Track(name=f"t{i}") for i in range(n_identities)]
    identities = [sio.Identity(name=f"id{i}") for i in range(n_identities)]
    for lf in labels:
        for i, inst in enumerate(lf.instances[:n_identities]):
            inst.track = tracks[i % n_identities]
            inst.identity = identities[i % n_identities]
        lf.instances = lf.instances[:n_identities]
    labels.tracks = tracks
    return labels, [identity.name for identity in identities]


def test_getitem_returns_a_usable_crop(minimal_instance):
    """A real sample: crop, mask, and the metadata the loss keys on."""
    import torch

    labels, class_names = _tracked_pose_labels(minimal_instance)
    dataset = EmbeddingDataset(
        labels=[labels],
        crop_size=32,
        class_names=class_names,
        embedding_head_config=OmegaConf.create(
            {
                "embedding_dim": 16,
                "output_stride": 16,
                "objective": {"positives": {"scope": "global_id", "aug_views": 2}},
            }
        ),
        max_stride=16,
        id_scope="global_id",
        track_names_are_global=True,
        cache_img=None,
    )
    assert len(dataset.mask_idx_list) > 0

    sample = dataset[0]

    assert sample["instance_image"].shape[-2:] == (32, 32)
    assert sample["instance_mask"].shape[-2:] == (32, 32)
    assert torch.isfinite(sample["instance_image"]).all()
    # The keys `build_contrastive_masks` reads.
    for key in ("group_id", "global_group_id", "video_id", "frame_idx", "item_id"):
        assert key in sample, key
    assert int(sample["group_id"]) in range(len(class_names))


def test_getitem_two_views_when_augmenting(minimal_instance):
    """`aug_views=2` yields a second view of the same crop."""
    labels, class_names = _tracked_pose_labels(minimal_instance)
    dataset = EmbeddingDataset(
        labels=[labels],
        crop_size=32,
        class_names=class_names,
        embedding_head_config=OmegaConf.create(
            {
                "embedding_dim": 16,
                "output_stride": 16,
                "objective": {"positives": {"scope": "global_id", "aug_views": 2}},
            }
        ),
        max_stride=16,
        id_scope="global_id",
        track_names_are_global=True,
        apply_aug=True,
        intensity_aug=OmegaConf.create(
            {
                "uniform_noise_min": 0.0,
                "uniform_noise_max": 0.04,
                "uniform_noise_p": 1.0,
            }
        ),
        cache_img=None,
    )

    sample = dataset[0]

    assert "instance_image_view2" in sample
    assert sample["instance_image_view2"].shape == sample["instance_image"].shape


def test_getitem_scale_and_max_hw_change_the_crop(minimal_instance):
    """The knobs the inference path was dropping actually move pixels."""
    import torch

    labels, class_names = _tracked_pose_labels(minimal_instance)

    def _sample(**kwargs):
        dataset = EmbeddingDataset(
            labels=[labels],
            crop_size=32,
            class_names=class_names,
            embedding_head_config=OmegaConf.create(
                {
                    "embedding_dim": 16,
                    "output_stride": 16,
                    "objective": {"positives": {"scope": "global_id"}},
                }
            ),
            max_stride=16,
            id_scope="global_id",
            track_names_are_global=True,
            cache_img=None,
            **kwargs,
        )
        return dataset[0]["instance_image"]

    baseline = _sample()
    sizematched = _sample(max_hw=(192, 192))

    assert baseline.shape == sizematched.shape
    assert not torch.allclose(baseline, sizematched), (
        "sizematching the frame before cropping must change the crop -- this is "
        "what made the inference path's dropped max_hw a silent scale mismatch"
    )
