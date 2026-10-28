"""bbox_detr data for M3: categories, multi-file mixing, reduced-size JPEG decoding.

Covers the person-only ground truth of multi-category box files (Objects365 /
Open Images / CrowdHuman store faces, heads, crowds and ignore regions as other
box categories), the per-sample weighted mixing of several training files by
``trainer_config.balanced_sampler`` (sampler, bbox loaders, and an end-to-end CPU
run on a tiny DINOv3 ``pretrained`` backbone), and the DCT-domain reduced decode of
large JPEGs.
"""

from collections import Counter

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from PIL import Image

from sleap_nn.data.bboxes import SetTargetEncoder, frame_bbox_arrays
from sleap_nn.data.custom_datasets import DetrBBoxDataset, get_train_val_dataloaders
from sleap_nn.data.topdown_warp import BalancedBatchSampler, BalancedDataLoader


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #
def _box(x1, y1, x2, y2, category=None, **metadata):
    box = sio.UserBoundingBox.from_xyxy(x1, y1, x2, y2, category=category)
    box.metadata.update(metadata)
    return box


def test_single_category_skips_other_categories():
    """A person detector does not train on faces, crowds or ignore regions."""
    lf = sio.LabeledFrame(video=sio.Video(filename="x.mp4"), frame_idx=0)
    lf.bboxes = [
        _box(0, 0, 10, 10, "person"),
        _box(5, 5, 8, 8, "face"),
        _box(0, 0, 50, 50, "crowd", is_group_of=1),  # Open Images: no iscrowd
        _box(20, 20, 40, 40, "ignore", isfake=1),
        _box(30, 30, 60, 60, "crowd", iscrowd=1),  # dropped as crowd anyway
        _box(1, 1, 4, 4),  # no category: class 0 of a single-class detector
    ]
    skipped = Counter()
    xyxy, classes = frame_bbox_arrays(lf, ["person"], skipped)
    np.testing.assert_allclose(xyxy, [[0, 0, 10, 10], [1, 1, 4, 4]])
    assert classes.tolist() == [0, 0]
    assert skipped == {"face": 1, "crowd": 1, "ignore": 1}
    # Class-agnostic ("object") still takes every non-crowd box.
    assert len(frame_bbox_arrays(lf, ["object"])[0]) == 5
    # Multi-class: by name.
    assert frame_bbox_arrays(lf, ["face", "person"])[1].tolist() == [1, 0]


# --------------------------------------------------------------------------- #
# Per-sample mixing
# --------------------------------------------------------------------------- #
def test_per_sample_mixing_sampler():
    groups = np.array([0] * 3 + [1] * 50 + [2] * 20)
    s = BalancedBatchSampler(
        groups,
        None,
        batches_per_epoch=400,
        seed=1,
        rank=0,
        weights=[2, 1, 1],
        batch_size=4,
    )
    batches = list(s)
    assert len(batches) == 400 and all(len(b) == 4 for b in batches)
    share = np.bincount(groups[np.concatenate(batches)], minlength=3) / 1600
    np.testing.assert_allclose(share, [0.5, 0.25, 0.25], atol=0.04)
    # The 3-sample file is oversampled and fully covered.
    assert set(np.concatenate(batches)[groups[np.concatenate(batches)] == 0]) == {
        0,
        1,
        2,
    }
    # Deterministic per (seed, rank, batch); ranks differ; resumes mid-epoch.
    again = BalancedBatchSampler(
        groups, None, 400, seed=1, rank=0, weights=[2, 1, 1], batch_size=4
    )
    again.load_state_dict({"position": 150})
    assert [list(b) for b in again] == [list(b) for b in batches[150:]]
    other = BalancedBatchSampler(
        groups, None, 400, seed=1, rank=1, weights=[2, 1, 1], batch_size=4
    )
    assert [list(b) for b in other] != [list(b) for b in batches]
    with pytest.raises(ValueError, match="no samples"):
        BalancedBatchSampler(groups, None, 4, weights=[1, 1, 1, 1], batch_size=4)


def _image_labels(tmp_path, name, size_hw, boxes, seed=0):
    """A one-frame Labels on a JPEG of ``size_hw`` with person boxes."""
    h, w = size_hw
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [(xx * 255 // w), (yy * 255 // h), (xx + yy) * 255 // (h + w)], -1
    ).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        img[int(y1) : int(y2), int(x1) : int(x2)] = rng.integers(0, 255, 3)
    path = tmp_path / f"{name}.jpg"
    Image.fromarray(img).save(path, quality=95)
    video = sio.Video.from_filename([str(path)])
    lf = sio.LabeledFrame(video=video, frame_idx=0)
    lf.bboxes = [_box(*b, "person") for b in boxes]
    return sio.Labels(videos=[video], labeled_frames=[lf])


def _detr_dataset(labels, hw=(256, 256), **kwargs):
    return DetrBBoxDataset(
        labels=labels,
        target_encoder=SetTargetEncoder(),
        categories=["person"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=hw,
        resize_mode="letterbox",
        **kwargs,
    )


def test_bbox_balanced_loader_mixes_files(tmp_path):
    files = [
        _image_labels(tmp_path, f"f{i}", (128, 160), [(10, 10, 60, 90)], seed=i)
        for i in range(3)
    ]
    ds = _detr_dataset(files, hw=(128, 160))
    assert ds.num_files == 3 and ds.labels_idx.tolist() == [0, 1, 2]
    config = OmegaConf.create(
        {
            "trainer_config": {
                "seed": 0,
                "train_steps_per_epoch": 30,
                "train_data_loader": {"batch_size": 2, "num_workers": 0},
                "val_data_loader": {"batch_size": 2, "num_workers": 0},
                "balanced_sampler": {"per_sample": True, "weights": [1, 0, 1]},
            }
        }
    )
    train, _ = get_train_val_dataloaders(ds, ds, config, train_steps_per_epoch=30)
    assert isinstance(train, BalancedDataLoader)
    seen = Counter(int(i) for b in train for i in b["labels_idx"])
    assert set(seen) == {0, 2} and sum(seen.values()) == 60


# --------------------------------------------------------------------------- #
# Reduced-size JPEG decoding
# --------------------------------------------------------------------------- #
def test_reduced_decode_matches_full_decode(tmp_path):
    boxes = [(200, 100, 900, 800), (1300, 300, 1900, 1000)]
    labels = _image_labels(tmp_path, "big", (1024, 2048), boxes)
    ds = _detr_dataset([labels])
    assert ds.min_decode_scale(1024, 2048) == pytest.approx(0.125)
    img, scale = ds._read_frame(ds.lf_idx_list[0])
    assert img.shape == (128, 256, 3) and scale == (0.125, 0.125)
    reduced = ds[0]

    ds.reduced_decode = False
    assert ds._read_frame(ds.lf_idx_list[0])[1] is None
    full = ds[0]
    n = full["num_instances"]
    assert reduced["num_instances"] == n == 2
    torch.testing.assert_close(reduced["bboxes"][:n], full["bboxes"][:n])
    torch.testing.assert_close(reduced["eff_scale"], full["eff_scale"])
    torch.testing.assert_close(reduced["orig_size"], full["orig_size"])
    diff = (reduced["image"].float() - full["image"].float()).abs().mean()
    assert diff < 3.0  # 0..255 levels: the same picture


def test_reduced_decode_only_when_nothing_is_lost(tmp_path):
    # 1.5x the training size: no 1/2 decode.
    labels = _image_labels(tmp_path, "mid", (384, 384), [(10, 10, 100, 100)])
    ds = _detr_dataset([labels])
    assert ds._read_frame(ds.lf_idx_list[0])[1] is None
    # RF-DETR's crop branch resizes the shortest side to 600 first.
    big = _image_labels(tmp_path, "big", (1400, 1400), [(10, 10, 100, 100)])
    aug = {
        "crop_p": 0.5,
        "shortest_sides": [400, 500, 600],
        "crop_range": [384, 600],
        "flip_p": 0.5,
    }
    ds = _detr_dataset([big], apply_aug=True, detr_aug=aug)
    assert ds.min_decode_scale(1400, 1400) == pytest.approx(600 / 1400)
    img, scale = ds._read_frame(ds.lf_idx_list[0])
    assert img.shape[0] >= 600 and scale == (0.5, 0.5)


# --------------------------------------------------------------------------- #
# End to end: tiny DINOv3 `pretrained` backbone, two mixed training files
# --------------------------------------------------------------------------- #
def test_train_pretrained_backbone_with_mixing(config, minimal_instance, tmp_path):
    pytest.importorskip("transformers")
    from transformers import DINOv3ViTConfig

    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.test_bbox_detr_training import _detr_train_config

    model_dir = tmp_path / "tiny_dinov3"
    DINOv3ViTConfig(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=32,
        patch_size=16,
        num_register_tokens=4,
    ).save_pretrained(model_dir)
    cfg = _detr_train_config(config, tmp_path, run_name="dinov3_mix")
    cfg.data_config.train_labels_path = [minimal_instance.as_posix()] * 2
    cfg.data_config.preprocessing.resize_mode = "letterbox"
    cfg.model_config.backbone_config = {
        "pretrained": {
            "model_name": model_dir.as_posix(),
            "weights": False,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "drop_path_rate": 0.1,
            "max_stride": 32,
            "output_stride": 16,
        }
    }
    cfg.trainer_config.eval = {"enabled": False}
    cfg.trainer_config.ema = {"enabled": False}
    cfg.trainer_config.visualize_preds_during_training = False
    cfg.trainer_config.balanced_sampler = {"per_sample": True, "weights": [3, 1]}
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    saved = OmegaConf.load(tmp_path / "dinov3_mix" / "training_config.yaml")
    assert saved.model_config.backbone_config.pretrained.mode == "tokens"
    assert isinstance(trainer.trainer.train_dataloader, BalancedDataLoader)
    assert torch.isfinite(trainer.trainer.callback_metrics["train/loss"])
