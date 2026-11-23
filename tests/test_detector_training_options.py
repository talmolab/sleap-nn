"""Experimental box-detector training options.

* ``data_config.ignore_box_category`` / ``ignore_box_min_cover``: ignore boxes
  in the ``bbox_detr`` score loss.
* ``data_config.val_negative_frames``: negative frames in validation.
* ``trainer_config.balanced_sampler.frame_weight_key``: per-frame sampling
  weights from frame metadata.
* None of them is written to the saved configs unless used.
"""

from collections import Counter

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from PIL import Image

from sleap_nn.data.bboxes import (
    IGNORE_BOX_CLASS,
    SetTargetEncoder,
    collate_set_targets,
    frame_bbox_arrays,
    frame_bbox_hulls,
    labels_bbox_categories,
)
from sleap_nn.data.custom_datasets import DetrBBoxDataset, get_train_val_dataloaders

RED, BLUE = (255, 0, 0), (0, 0, 255)


def _box(x1, y1, x2, y2, category):
    return sio.UserBoundingBox.from_xyxy(x1, y1, x2, y2, category=category)


def _frame_labels(tmp_path, name, frames, size=(128, 128), negatives=0):
    """One frame (own PNG) per ``(animal boxes, ignore boxes, metadata)`` item.

    Animal regions are painted red, ignore regions blue, on a dark gradient.
    ``negatives`` extra frames are user-confirmed empty (negative) frames.
    """
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    lfs, videos = [], []
    items = list(frames) + [([], [], {})] * negatives
    for i, (animals, ignores, meta) in enumerate(items):
        img = np.stack([xx * 60 // w, yy * 60 // h, (xx + yy) * 30 // (h + w)], -1)
        img = img.astype(np.uint8)
        for color, rects in ((RED, animals), (BLUE, ignores)):
            for x1, y1, x2, y2 in rects:
                img[int(y1) : int(y2), int(x1) : int(x2)] = color
        path = tmp_path / f"{name}{i}.png"
        Image.fromarray(img).save(path)
        video = sio.Video.from_filename([str(path)])
        videos.append(video)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        if i >= len(frames):
            lf.is_negative = True
        lf.bboxes = [_box(*b, "animal") for b in animals] + [
            _box(*b, "ignore") for b in ignores
        ]
        lf.metadata.update(meta)
        lfs.append(lf)
    return sio.Labels(videos=videos, labeled_frames=lfs)


def _dataset(labels, hw=(128, 128), **kwargs):
    kwargs.setdefault("ignore_box_category", "ignore")
    return DetrBBoxDataset(
        labels=labels if isinstance(labels, list) else [labels],
        target_encoder=SetTargetEncoder(),
        categories=["animal"],
        max_stride=32,
        ensure_rgb=True,
        max_hw=hw,
        resize_mode="letterbox",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Ignore boxes: ground truth
# --------------------------------------------------------------------------- #
def test_frame_bbox_arrays_put_ignore_boxes_last():
    lf = sio.LabeledFrame(video=sio.Video(filename="x.mp4"), frame_idx=0)
    lf.bboxes = [_box(0, 0, 5, 5, "ignore"), _box(1, 1, 9, 9, "animal")]
    xyxy, cls = frame_bbox_arrays(lf, ["animal"], ignore_category="ignore")
    np.testing.assert_allclose(xyxy, [[1, 1, 9, 9], [0, 0, 5, 5]])
    assert cls.tolist() == [0, IGNORE_BOX_CLASS]
    # A class-agnostic detector does not take them as objects either.
    xyxy, cls = frame_bbox_arrays(lf, ["object"], ignore_category="ignore")
    assert cls.tolist() == [0, IGNORE_BOX_CLASS]
    # Hulls stay aligned (ignore boxes have none: never pasted).
    hulls = frame_bbox_hulls(lf, ["object"], ignore_category="ignore")
    assert hulls.shape[0] == 2 and np.isnan(hulls[1]).all()
    # Off: as before.
    assert frame_bbox_arrays(lf, ["animal"])[1].tolist() == [0]
    assert frame_bbox_arrays(lf, ["object"])[1].tolist() == [0, 0]
    labels = sio.Labels(labeled_frames=[lf])
    assert labels_bbox_categories([labels], "ignore") == ["animal"]


def test_ignore_category_checks():
    from sleap_nn.data.bboxes import CenterNetTargetEncoder
    from sleap_nn.data.custom_datasets import _check_ignore_box_category

    assert _check_ignore_box_category(None, object(), ["a"]) is None
    with pytest.raises(ValueError, match="bbox_detr models only"):
        _check_ignore_box_category(
            "ignore", CenterNetTargetEncoder(num_classes=1, output_stride=4), ["a"]
        )
    with pytest.raises(ValueError, match="cannot also be objects"):
        _check_ignore_box_category("a", SetTargetEncoder(), ["a"])


def test_ignore_boxes_are_a_separate_target(tmp_path):
    labels = _frame_labels(
        tmp_path,
        "f",
        [
            ([(10, 10, 40, 50)], [(70, 60, 110, 100)], {}),
            ([], [(20, 20, 60, 60)], {}),  # only ignore boxes: still a sample
        ],
    )
    ds = _dataset(labels)
    assert len(ds) == 2
    s = ds[0]
    assert int(s["set_count"]) == 1 and int(s["num_instances"]) == 1
    assert int(s["num_ignore"]) == 1
    np.testing.assert_allclose(s["ignore_bboxes"][0], [70, 60, 110, 100])
    assert torch.isnan(s["ignore_bboxes"][1:]).all()
    np.testing.assert_allclose(s["bboxes"][0], [10, 10, 40, 50])
    s = ds[1]
    assert int(s["set_count"]) == 0 and int(s["num_ignore"]) == 1

    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    targets = collate_set_targets(batch)
    np.testing.assert_allclose(
        targets[0]["ignore_boxes"], [[90 / 128, 80 / 128, 40 / 128, 40 / 128]]
    )
    assert len(targets[1]["labels"]) == 0 and len(targets[1]["ignore_boxes"]) == 1
    # Off by default: no ignore keys, and the ignore box is not an object.
    plain = _dataset(labels, ignore_box_category=None)
    assert len(plain) == 1  # the ignore-only frame has no boxes of `categories`
    assert "num_ignore" not in plain[0] and int(plain[0]["set_count"]) == 1


def _share_of_color(image, box, channel):
    """Share of the box's pixels painted in ``channel`` (0 red, 2 blue)."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    patch = image[0, :, max(y1, 0) : y2, max(x1, 0) : x2].float()
    other = 2 - channel
    painted = (patch[channel] > 150) & (patch[other] < 100)
    return float(painted.float().mean()) if painted.numel() else 0.0


@pytest.mark.parametrize("seed", range(3))
def test_ignore_boxes_follow_every_augmentation(tmp_path, seed):
    """After crop, flips, 90 degree turns, zoom-out onto another frame and the stock
    rotation, every ignore box still sits on its (blue) region and every object box
    on a red one (at least a third of the box: a canvas box may be partly covered
    by the zoomed-out frame, and rotation enlarges the boxes)."""
    frames = [
        ([(8, 8, 48, 40)], [(70, 70, 120, 120)], {}),
        ([(80, 10, 120, 50)], [(10, 70, 60, 120)], {}),
    ]
    labels = _frame_labels(tmp_path, f"a{seed}", frames)
    aug = {
        "crop_p": 0.5,
        "shortest_sides": [128],
        "crop_range": [96, 128],
        "flip_p": 0.5,
        "vflip_p": 0.5,
        "rot90_p": 0.5,
        "zoom_out_p": 0.5,
        "zoom_out_min": 0.5,
        "zoom_out_max": 0.9,
        "zoom_out_canvas": "frame",
    }
    ds = _dataset(
        labels,
        apply_aug=True,
        detr_aug=aug,
        geometric_aug={"rotation_p": 0.5, "rotation_min": -8, "rotation_max": 8},
    )
    torch.manual_seed(seed)
    seen = Counter()
    for i in range(16):
        s = ds[i % 2]
        img = s["image"]
        for k in range(int(s["num_ignore"])):
            box = s["ignore_bboxes"][k]
            if float((box[2] - box[0]) * (box[3] - box[1])) < 64:
                continue  # a sliver at a crop edge
            assert _share_of_color(img, box, 2) > 0.33, (i, box)
            seen["ignore"] += 1
        for k in range(int(s["num_instances"])):
            box = s["bboxes"][k]
            if float((box[2] - box[0]) * (box[3] - box[1])) < 64:
                continue
            assert _share_of_color(img, box, 0) > 0.33, (i, box)
            seen["object"] += 1
    assert seen["ignore"] >= 8 and seen["object"] >= 8


def test_box_evaluation_treats_ignore_boxes_as_ignore_regions():
    from sleap_nn.evaluation import bbox_frame_pairs, compute_bbox_metrics

    video = sio.Video(filename="x.mp4")
    gt = sio.LabeledFrame(video=video, frame_idx=0)
    gt.bboxes = [_box(0, 0, 10, 10, "animal"), _box(50, 50, 90, 90, "ignore")]
    pr = sio.LabeledFrame(video=video, frame_idx=0)
    pr.bboxes = [
        sio.PredictedBoundingBox.from_xyxy(0, 0, 10, 10, score=0.9),
        sio.PredictedBoundingBox.from_xyxy(52, 52, 88, 88, score=0.8),
    ]
    labels_gt = sio.Labels(videos=[video], labeled_frames=[gt])
    labels_pr = sio.Labels(videos=[video], labeled_frames=[pr])

    frames, cats = bbox_frame_pairs(labels_gt, labels_pr, ignore_box_category="ignore")
    assert cats == ["object"] and len(frames[0]["gt_bboxes"]) == 1
    np.testing.assert_allclose(frames[0]["gt_ignore_bboxes"], [[50, 50, 90, 90]])
    m = compute_bbox_metrics(frames, num_classes=1)
    assert (m["bbox_n_tp"], m["bbox_n_fp"], m["bbox_n_fn"]) == (1, 0, 0)
    # Off: the ignore box is a second category of ground truth.
    frames, cats = bbox_frame_pairs(labels_gt, labels_pr)
    assert "ignore" in cats and "gt_ignore_bboxes" not in frames[0]


# --------------------------------------------------------------------------- #
# Ignore boxes: the score loss
# --------------------------------------------------------------------------- #
def _criterion_case():
    from sleap_nn.training.detr.criterion import SetCriterion

    torch.manual_seed(0)
    crit = SetCriterion(num_classes=1, losses=("labels",)).eval()
    logits = torch.randn(1, 3, 2, requires_grad=True)
    # Query 0 on the object, 1 inside the ignore box, 2 elsewhere.
    boxes = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.2], [0.75, 0.75, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1]]]
    )
    target = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]]),
    }
    return crit, logits, boxes, target


def test_unmatched_queries_in_ignore_boxes_get_no_background_loss():
    crit, logits, boxes, target = _criterion_case()
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    plain = crit(outputs, [target])["loss_ce"]
    (g_plain,) = torch.autograd.grad(plain, logits)

    ignore = {**target, "ignore_boxes": torch.tensor([[0.75, 0.75, 0.3, 0.3]])}
    loss = crit(outputs, [ignore])["loss_ce"]
    (g,) = torch.autograd.grad(loss, logits)
    assert float(loss.detach()) < float(plain.detach())
    assert torch.all(g[0, 1] == 0)  # inside the ignore box: no loss at all
    assert torch.any(g_plain[0, 1] != 0)
    torch.testing.assert_close(g[0, [0, 2]], g_plain[0, [0, 2]])  # others as before

    # The threshold is on the share of the QUERY box inside the ignore box.
    crit.ignore_min_cover = 0.5
    # 55% of the query box (x 0.70..0.80) lies inside x 0.69..0.755.
    part = {**ignore, "ignore_boxes": torch.tensor([[0.7225, 0.75, 0.065, 0.3]])}
    (g_half,) = torch.autograd.grad(crit(outputs, [part])["loss_ce"], logits)
    assert torch.all(g_half[0, 1] == 0)
    crit.ignore_min_cover = 0.6
    (g_less,) = torch.autograd.grad(crit(outputs, [part])["loss_ce"], logits)
    torch.testing.assert_close(g_less, g_plain)

    # An ignore box is never matched nor counted.
    assert float(crit.num_boxes([ignore], torch.device("cpu"))) == 1.0


def test_matched_query_keeps_its_loss_inside_an_ignore_box():
    crit, logits, boxes, target = _criterion_case()
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    plain = crit(outputs, [target])["loss_ce"]
    # An ignore box over the object itself: the matched query is unaffected.
    over = {**target, "ignore_boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]])}
    torch.testing.assert_close(crit(outputs, [over])["loss_ce"], plain)


def test_all_empty_batch_loss_is_finite():
    """A batch of only empty (negative) frames: the divisor clamps at 1."""
    crit, logits, boxes, _ = _criterion_case()
    empty = {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    assert float(crit.num_boxes([empty, empty], torch.device("cpu"))) == 1.0
    losses = crit(
        {k: v.expand(2, *v.shape[1:]) for k, v in outputs.items()}, [empty, empty]
    )
    loss = losses["loss_ce"].detach()
    assert torch.isfinite(loss) and float(loss) > 0


# --------------------------------------------------------------------------- #
# Per-frame sampling weights
# --------------------------------------------------------------------------- #
def _loader(ds, **bs):
    config = OmegaConf.create(
        {
            "trainer_config": {
                "seed": 0,
                "train_steps_per_epoch": 400,
                "train_data_loader": {"batch_size": 4, "num_workers": 0},
                "val_data_loader": {"batch_size": 2, "num_workers": 0},
                "balanced_sampler": bs,
            }
        }
    )
    return get_train_val_dataloaders(ds, ds, config, train_steps_per_epoch=400)[0]


def test_frame_weights_from_metadata(tmp_path):
    f0 = _frame_labels(
        tmp_path,
        "w",
        [([(10, 10, 40, 40)], [], {"w": 3.0}), ([(10, 10, 40, 40)], [], {})],
        negatives=1,
    )
    f1 = _frame_labels(tmp_path, "v", [([(10, 10, 40, 40)], [], {"w": 0.5})])
    f0.negative_frames[0].metadata["w"] = 0
    ds = _dataset([f0, f1], ignore_box_category=None, use_negative_frames=True)
    weights = ds.frame_metadata_weights([f0, f1], "w")
    order = [(int(s["labels_idx"]), bool(s["is_negative"])) for s in ds.lf_idx_list]
    assert order == [(0, False), (0, False), (1, False), (0, True)]
    np.testing.assert_allclose(weights, [3.0, 1.0, 0.5, 0.0])

    ds.frame_weights = weights
    train = _loader(ds, per_sample=True, weights=[1, 1], frame_weight_key="w")
    idx = np.concatenate(list(train.batch_sampler))
    file0 = idx[ds.labels_idx[idx] == 0]
    assert abs(np.mean(file0 == 0) - 0.75) < 0.05  # 3:1 within file 0
    assert not (file0 == 3).any()  # weight 0: never drawn
    assert abs(np.mean(ds.labels_idx[idx] == 0) - 0.5) < 0.05  # file shares kept
    with pytest.raises(ValueError, match="needs per_sample"):
        _loader(ds, weights=[1, 1], frame_weight_key="w")
    del ds.frame_weights
    with pytest.raises(ValueError, match="bbox_\\* models"):
        _loader(ds, per_sample=True, weights=[1, 1], frame_weight_key="w")

    f1.labeled_frames[0].metadata["w"] = "heavy"
    with pytest.raises(ValueError, match="expected a number >= 0"):
        ds.frame_metadata_weights([f0, f1], "w")


# --------------------------------------------------------------------------- #
# Config: validation, and nothing new saved unless used
# --------------------------------------------------------------------------- #
def test_config_fields_and_unused_keys_left_out():
    from sleap_nn.config.data_config import DataConfig
    from sleap_nn.config.trainer_config import BalancedSamplerConfig
    from sleap_nn.config.training_job_config import (
        TrainingJobConfig,
        config_without_unused_keys,
    )

    d = DataConfig()
    assert (d.val_negative_frames, d.ignore_box_category) == (False, None)
    assert d.ignore_box_min_cover == 0.5
    assert BalancedSamplerConfig().frame_weight_key is None
    with pytest.raises(ValueError):
        DataConfig(ignore_box_min_cover=0)

    from sleap_nn.config.trainer_config import BalancedSamplerConfig as BSC

    cfg = OmegaConf.structured(TrainingJobConfig())
    cfg.trainer_config.balanced_sampler = BSC(per_sample=True)
    out = config_without_unused_keys(cfg)
    for key in ("val_negative_frames", "ignore_box_category", "ignore_box_min_cover"):
        assert key not in out.data_config
    assert "frame_weight_key" not in out.trainer_config.balanced_sampler
    assert "per_sample" in out.trainer_config.balanced_sampler
    assert "frame_weight_key" in cfg.trainer_config.balanced_sampler  # not modified

    cfg.data_config.ignore_box_category = "ignore"
    cfg.data_config.val_negative_frames = True
    out = config_without_unused_keys(cfg)
    assert out.data_config.ignore_box_category == "ignore"
    assert out.data_config.val_negative_frames is True
    assert "ignore_box_min_cover" not in out.data_config  # still the default
    # Nothing to drop: the config itself.
    bare = OmegaConf.create({"data_config": {"use_negative_frames": False}})
    assert config_without_unused_keys(bare) is bare


def test_val_negative_frames_is_for_box_models(config):
    from sleap_nn.data.custom_datasets import _val_negative_frames

    cfg = config.copy()
    assert _val_negative_frames(cfg) is False
    cfg.data_config.val_negative_frames = True
    with pytest.raises(ValueError, match="bbox_\\* models only"):
        _val_negative_frames(cfg)


# --------------------------------------------------------------------------- #
# End to end on the CPU
# --------------------------------------------------------------------------- #
def _train(config, tmp_path, run_name, train, val, **data_options):
    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.test_bbox_detr_training import _detr_train_config

    cfg = _detr_train_config(config, tmp_path, run_name=run_name)
    train_path, val_path = (
        tmp_path / f"{run_name}_train.slp",
        tmp_path / f"{run_name}_val.slp",
    )
    sio.save_slp(train, train_path.as_posix())
    sio.save_slp(val, val_path.as_posix())
    cfg.data_config.train_labels_path = [train_path.as_posix()]
    cfg.data_config.val_labels_path = [val_path.as_posix()]
    cfg.data_config.augmentation_config.intensity = None
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.ema = {"enabled": False}
    cfg.trainer_config.visualize_preds_during_training = False
    cfg.trainer_config.eval = {"enabled": True, "frequency": 1}
    cfg.trainer_config.balanced_sampler = data_options.pop("balanced_sampler", None)
    for key, value in data_options.items():
        cfg.data_config[key] = value
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    return trainer, tmp_path / run_name


def _datasets(trainer):
    val = trainer.trainer.val_dataloaders
    val = val[0] if isinstance(val, (list, tuple)) else val
    return trainer.trainer.train_dataloader.dataset, val.dataset


def test_train_with_ignore_boxes_val_negatives_and_frame_weights(config, tmp_path):
    frames = [
        ([(8, 8, 48, 40)], [(70, 70, 120, 120)], {"w": 2.0}),
        ([(80, 10, 120, 50)], [(10, 70, 60, 120)], {}),
        ([(30, 30, 80, 80)], [], {"w": 0.5}),
    ]
    train = _frame_labels(tmp_path, "t", frames)
    val = _frame_labels(tmp_path, "v", frames[:2], negatives=2)

    # Defaults: nothing new in the saved configs, negatives left out of validation.
    base, base_dir = _train(config, tmp_path, "plain", train, val)
    for name in ("training_config.yaml", "initial_config.yaml"):
        text = (base_dir / name).read_text()
        for key in (
            "val_negative_frames",
            "ignore_box_category",
            "ignore_box_min_cover",
            "frame_weight_key",
        ):
            assert key not in text, (name, key)
    assert len(_datasets(base)[1]) == 2
    saved = OmegaConf.load(base_dir / "training_config.yaml")
    # The ignore category is an object category when the option is off.
    assert list(saved.model_config.head_configs.bbox_detr.detr.categories) == [
        "animal",
        "ignore",
    ]

    trainer, run_dir = _train(
        config,
        tmp_path,
        "opts",
        train,
        val,
        ignore_box_category="ignore",
        ignore_box_min_cover=0.6,
        val_negative_frames=True,
        balanced_sampler={
            "per_sample": True,
            "weights": [1.0],
            "frame_weight_key": "w",
        },
    )
    assert trainer.lightning_model.criterion.ignore_min_cover == 0.6
    train_ds, val_ds = _datasets(trainer)
    assert len(val_ds) == 4  # 2 frames + 2 negative frames
    np.testing.assert_allclose(train_ds.frame_weights, [2.0, 1.0, 0.5])
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    assert list(saved.model_config.head_configs.bbox_detr.detr.categories) == ["animal"]
    assert saved.data_config.ignore_box_category == "ignore"
    assert saved.data_config.ignore_box_min_cover == 0.6
    assert saved.data_config.val_negative_frames is True
    assert saved.trainer_config.balanced_sampler.frame_weight_key == "w"
    metrics = trainer.trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"]) and torch.isfinite(metrics["val/loss"])
    assert "eval/val/bbox_mAP" in metrics


# --------------------------------------------------------------------------- #
# Review follow-ups
# --------------------------------------------------------------------------- #
def _pose_labels_with_ignore(with_masks=False):
    """Two animals labelled with poses (and masks), one region boxed as ignore."""
    skel = sio.Skeleton(["a", "b", "c"])
    video = sio.Video(filename="x.mp4")
    lf = sio.LabeledFrame(video=video, frame_idx=0)
    for x0 in (10, 60):
        pts = np.array([[x0, 10], [x0 + 20, 20], [x0 + 10, 40]], float)
        lf.instances.append(sio.Instance.from_numpy(pts, skeleton=skel))
    if with_masks:
        masks = []
        for x0 in (10, 60):
            m = np.zeros((64, 128), bool)
            m[10:41, x0 : x0 + 21] = True
            masks.append(sio.UserSegmentationMask.from_numpy(m))
        lf.masks = masks
    lf.bboxes = [_box(100, 40, 120, 60, "ignore")]
    return sio.Labels(videos=[video], labeled_frames=[lf], skeletons=[skel])


@pytest.mark.parametrize("source", ["auto", "instances", "masks"])
def test_ignore_boxes_with_pose_or_mask_labels(source):
    from sleap_nn.data.bboxes import add_bboxes_to_labels, resolve_bbox_source

    labels = _pose_labels_with_ignore(with_masks=source == "masks")
    # The ignore box alone does not make `auto` read the file's user boxes.
    assert resolve_bbox_source(labels, "auto", ignore_category="ignore") == "instances"
    assert resolve_bbox_source(labels, "auto") == "bboxes"  # off: as before
    resolved = add_bboxes_to_labels(labels, source=source, ignore_category="ignore")
    assert resolved == ("instances" if source == "auto" else source)
    xyxy, cls = frame_bbox_arrays(labels[0], ["object"], ignore_category="ignore")
    assert cls.tolist() == [0, 0, IGNORE_BOX_CLASS]
    np.testing.assert_allclose(xyxy[2], [100, 40, 120, 60])
    assert labels_bbox_categories([labels], "ignore") == ["object"]
    # Off: deriving drops the user boxes, as before.
    labels = _pose_labels_with_ignore(with_masks=source == "masks")
    add_bboxes_to_labels(labels, source="instances" if source == "auto" else source)
    assert frame_bbox_arrays(labels[0], ["object"])[1].tolist() == [0, 0]


def test_post_training_evaluation_keeps_ignore_boxes_of_pose_labels():
    from sleap_nn.evaluation import _run_bbox_evaluation

    gt = _pose_labels_with_ignore()
    pr_lf = sio.LabeledFrame(video=gt.videos[0], frame_idx=0)
    pr_lf.bboxes = [
        sio.PredictedBoundingBox.from_xyxy(10, 10, 30, 40, score=0.9),
        sio.PredictedBoundingBox.from_xyxy(60, 10, 80, 40, score=0.9),
        sio.PredictedBoundingBox.from_xyxy(102, 42, 110, 50, score=0.8),
    ]
    pr = sio.Labels(videos=gt.videos, labeled_frames=[pr_lf])
    m = _run_bbox_evaluation(gt, pr, ignore_box_category="ignore")
    assert (m["bbox_n_tp"], m["bbox_n_fp"], m["bbox_n_fn"]) == (2, 0, 0)


def test_metrics_ignore_rule_matches_the_loss():
    """A 20x20 detection inside a 100x100 ignore box: ignored by the loss's rule
    (share of the detection), counted as a false positive by the default rule
    (share of the ignore box)."""
    from sleap_nn.evaluation import compute_bbox_metrics

    frame = {
        "gt_bboxes": np.zeros((0, 4)),
        "gt_classes": np.zeros(0, np.int64),
        "gt_ignore_bboxes": np.array([[0, 0, 100, 100]], float),
        "pred_bboxes": np.array([[40, 40, 60, 60]], float),
        "pred_scores": np.array([0.9]),
        "pred_classes": np.array([0]),
    }
    default = compute_bbox_metrics([frame], num_classes=1)
    assert default["bbox_n_fp"] == 1
    rule = compute_bbox_metrics(
        [frame], num_classes=1, ignore_cover=0.5, ignore_cover_of="detection"
    )
    assert rule["n_pred"] == 0


def test_ignore_box_config_checks(config):
    from sleap_nn.data.custom_datasets import _check_ignore_box_config

    cfg = config.copy()
    cfg.data_config.ignore_box_category = "ignore"
    with pytest.raises(ValueError, match="bbox_detr models only"):
        _check_ignore_box_config(cfg)  # a single-instance pose config
    from sleap_nn.config.get_config import get_head_configs

    heads = OmegaConf.structured(get_head_configs("bbox_detr"))
    cfg.model_config.head_configs = heads
    _check_ignore_box_config(cfg)
    cfg.data_config.ignore_box_category = "object"
    with pytest.raises(ValueError, match="without a category"):
        _check_ignore_box_config(cfg)


def test_ignore_rule_on_every_matched_layer_and_group():
    from sleap_nn.training.detr.criterion import SetCriterion

    torch.manual_seed(0)
    crit = SetCriterion(num_classes=1, losses=("labels",), group_detr=2).train()
    boxes = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.2], [0.75, 0.75, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1]] * 2]
    )

    def layer():
        return {
            "pred_logits": torch.randn(1, 6, 2, requires_grad=True),
            "pred_boxes": boxes,
        }

    out = layer()
    out["aux_outputs"] = [layer()]
    out["enc_outputs"] = layer()
    t = {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]])}
    ti = {**t, "ignore_boxes": torch.tensor([[0.75, 0.75, 0.3, 0.3]])}
    losses = crit(out, [ti], num_boxes=torch.tensor(1.0))
    plain = crit(out, [t], num_boxes=torch.tensor(1.0))
    for key, lay in (
        ("loss_ce", out),
        ("loss_ce_0", out["aux_outputs"][0]),
        ("loss_ce_enc", out["enc_outputs"]),
    ):
        (g,) = torch.autograd.grad(losses[key], lay["pred_logits"], retain_graph=True)
        (gp,) = torch.autograd.grad(plain[key], lay["pred_logits"], retain_graph=True)
        assert torch.all(g[0, [1, 4]] == 0), key  # both groups' covered queries
        assert torch.any(gp[0, [1, 4]] != 0), key
        torch.testing.assert_close(g[0, [2, 5]], gp[0, [2, 5]])


def test_default_path_unchanged_with_the_code_on_but_unused(tmp_path):
    """Seeded draws through every augmentation: with the ignore option on but no
    ignore box in the labels, every sample tensor equals the option-off one (the
    new code consumes no random draw), and the loss is identical."""
    frames = [
        ([(8, 8, 48, 40), (70, 70, 120, 120)], [], {}),
        ([(80, 10, 120, 50)], [], {}),
    ]
    labels = _frame_labels(tmp_path, "d", frames, negatives=1)
    aug = {
        "crop_p": 0.5,
        "shortest_sides": [128],
        "crop_range": [96, 128],
        "flip_p": 0.5,
        "vflip_p": 0.5,
        "rot90_p": 0.5,
        "zoom_out_p": 0.5,
        "zoom_out_canvas": "frame",
    }

    def draws(**kw):
        ds = _dataset(
            labels,
            apply_aug=True,
            detr_aug=aug,
            use_negative_frames=True,
            geometric_aug={"rotation_p": 0.5, "rotation_min": -8, "rotation_max": 8},
            intensity_aug={"contrast_p": 0.5},
            **kw,
        )
        torch.manual_seed(0)
        np.random.seed(0)
        return [ds[k % len(ds)] for k in range(12)]

    off, on = draws(ignore_box_category=None), draws(ignore_box_category="ignore")
    for a, b in zip(off, on):
        assert int(b["num_ignore"]) == 0
        for key, value in a.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value, b[key], rtol=0, atol=0, equal_nan=True
                )
            else:
                assert value == b[key]

    crit, logits, boxes, target = _criterion_case()
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    empty = {**target, "ignore_boxes": torch.zeros(0, 4)}
    torch.testing.assert_close(
        crit(outputs, [target])["loss_ce"], crit(outputs, [empty])["loss_ce"]
    )


def test_frame_weights_multiply_with_overlap_weight(tmp_path):
    f0 = _frame_labels(
        tmp_path,
        "o",
        [
            ([(10, 10, 60, 90), (35, 10, 85, 90)], [], {"w": 1.0}),  # overlapping
            ([(10, 10, 40, 40)], [], {"w": 2.0}),
        ],
    )
    ds = _dataset([f0], ignore_box_category=None)
    ds.frame_weights = ds.frame_metadata_weights([f0], "w")
    train = _loader(
        ds, per_sample=True, weights=[1], overlap_weight=4.0, frame_weight_key="w"
    )
    idx = np.concatenate(list(train.batch_sampler))
    # 4 x 1 against 1 x 2: the overlapping frame is drawn 2/3 of the time.
    assert abs(np.mean(idx == 0) - 2 / 3) < 0.05


def _pose_frames(tmp_path, name, n=3):
    """Pose-labelled frames (one animal) with one ignore box each, no user boxes."""
    skel = sio.Skeleton(["a", "b", "c"])
    lfs, videos = [], []
    for i in range(n):
        img = np.zeros((128, 128, 3), np.uint8)
        img[10:50, 10:40] = RED
        img[70:120, 70:120] = BLUE
        path = tmp_path / f"{name}{i}.png"
        Image.fromarray(img).save(path)
        video = sio.Video.from_filename([str(path)])
        videos.append(video)
        lf = sio.LabeledFrame(video=video, frame_idx=0)
        pts = np.array([[12, 12], [38, 30], [25, 48]], float)
        lf.instances.append(sio.Instance.from_numpy(pts, skeleton=skel))
        lf.bboxes = [_box(70, 70, 120, 120, "ignore")]
        lfs.append(lf)
    return sio.Labels(videos=videos, labeled_frames=lfs, skeletons=[skel])


def test_train_pose_labels_with_ignore_boxes(config, tmp_path):
    """Through the trainer: `bbox_source: auto` derives the animal boxes from the
    poses and keeps the ignore boxes, and the box-metric callback scores with the
    loss's rule (`ignore_cover_of="detection"`)."""
    from unittest.mock import patch

    from sleap_nn.evaluation import compute_bbox_metrics as real

    seen = []

    def spy(frames, *a, **kw):
        n = sum(len(f.get("gt_ignore_bboxes", ())) for f in frames)
        seen.append((kw, n))
        return real(frames, *a, **kw)

    train, val = _pose_frames(tmp_path, "t"), _pose_frames(tmp_path, "v", 2)
    with patch("sleap_nn.evaluation.compute_bbox_metrics", spy):
        trainer, _ = _train(
            config,
            tmp_path,
            "pose",
            train,
            val,
            ignore_box_category="ignore",
            bbox_source="auto",
        )
    train_ds, _ = _datasets(trainer)
    s = train_ds[0]
    assert int(s["num_ignore"]) == 1 and int(s["num_instances"]) == 1
    assert list(trainer.lightning_model.categories) == ["object"]
    assert seen and all(kw.get("ignore_cover_of") == "detection" for kw, _ in seen)
    assert all(kw.get("ignore_cover") == 0.5 for kw, _ in seen)
    assert all(n >= 1 for _, n in seen)
