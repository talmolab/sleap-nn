"""Training ``bbox_detr`` models (RF-DETR recipe): data, module, optimizer, init.

Covers the set targets and their collate step, RF-DETR's resize/crop/flip box
bookkeeping and per-batch multi-scale, the Lightning module's losses and
parameter groups, the class-subset ``init_from`` of an imported RF-DETR, EMA
wiring, config round-trips and an end-to-end ``ModelTrainer`` run on CPU.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.data.bboxes import (
    SetTargetEncoder,
    clip_bboxes,
    collate_set_targets,
    frame_bbox_arrays,
    frame_user_bboxes,
    get_bbox_target_encoder,
    is_crowd_bbox,
    labels_bbox_categories,
)
from sleap_nn.data.detr_augmentation import (
    batch_scale,
    detr_train_transform,
    hflip,
    multi_scale_sizes,
    random_resize_crop,
    resize_batch,
    resize_frame,
    shortest_side_size,
)

TINY_DIR = Path(__file__).resolve().parent / "assets" / "detr" / "rfdetr_tiny"

# A tiny RF-DETR-shaped model (windowed DINOv2 + decoder) that trains on CPU.
TINY_BACKBONE = dict(
    in_channels=3,
    hidden_size=16,
    num_layers=4,
    num_heads=2,
    patch_size=16,
    pos_embed_grid=4,
    out_indices=[2, 4],
    num_windows=2,
    max_stride=32,
    output_stride=16,
)
TINY_DETR = dict(
    hidden_dim=16,
    num_queries=12,
    group_detr=3,
    dec_layers=2,
    sa_nheads=2,
    ca_nheads=4,
    dec_n_points=2,
    dim_feedforward=32,
    num_select=12,
)


# --------------------------------------------------------------------------- #
# Ground truth, set targets and collate
# --------------------------------------------------------------------------- #
def _frame_with_boxes():
    video = sio.Video(filename="fake.mp4")
    boxes = [
        sio.UserBoundingBox.from_xyxy(10, 20, 50, 80, category="person"),
        sio.UserBoundingBox.from_xyxy(0, 0, 100, 100, category="crowd"),
        sio.UserBoundingBox.from_xyxy(60, 60, 90, 70, category="person"),
    ]
    boxes[1].metadata["iscrowd"] = 1
    boxes[0].metadata["iscrowd"] = 0
    return sio.LabeledFrame(video=video, frame_idx=0, bboxes=boxes)


def test_crowd_boxes_are_not_ground_truth():
    lf = _frame_with_boxes()
    assert [is_crowd_bbox(b) for b in lf.bboxes] == [False, True, False]
    assert len(frame_user_bboxes(lf)) == 2
    labels = sio.Labels(videos=[lf.video], labeled_frames=[lf])
    assert labels_bbox_categories([labels]) == ["person"]
    xyxy, classes = frame_bbox_arrays(lf, ["person"])
    np.testing.assert_allclose(xyxy, [[10, 20, 50, 80], [60, 60, 90, 70]])
    assert classes.tolist() == [0, 0]


def test_set_target_encoder_normalizes_and_pads():
    assert isinstance(get_bbox_target_encoder("bbox_detr", None), SetTargetEncoder)
    enc = SetTargetEncoder(max_targets=4)
    boxes = torch.tensor([[0.0, 0.0, 50.0, 100.0], [100.0, 50.0, 200.0, 100.0]])
    out = enc(boxes, torch.tensor([1, 0]), img_hw=(100, 200))
    assert out["set_boxes"].shape == (4, 4) and int(out["set_count"]) == 2
    torch.testing.assert_close(
        out["set_boxes"][:2],
        torch.tensor([[0.125, 0.5, 0.25, 1.0], [0.75, 0.75, 0.5, 0.5]]),
    )
    assert out["set_labels"].tolist() == [1, 0, -1, -1]
    assert (out["set_boxes"][2:] == 0).all()
    with pytest.raises(ValueError, match="exceed max_targets"):
        SetTargetEncoder(max_targets=1)(boxes, torch.tensor([1, 0]), (100, 200))
    empty = enc(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), (100, 200))
    assert int(empty["set_count"]) == 0


def test_collate_set_targets():
    enc = SetTargetEncoder(max_targets=3)
    samples = [
        enc(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([2]), (20, 20)),
        enc(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), (20, 20)),
        enc(
            torch.tensor([[0, 0, 4.0, 4], [4, 4, 20, 20]]),
            torch.tensor([0, 1]),
            (20, 20),
        ),
    ]
    batch = torch.utils.data.default_collate(samples)
    targets = collate_set_targets(batch)
    assert [len(t["labels"]) for t in targets] == [1, 0, 2]
    assert targets[0]["labels"].dtype == torch.int64
    torch.testing.assert_close(
        targets[0]["boxes"], torch.tensor([[0.25, 0.25, 0.5, 0.5]])
    )
    assert targets[2]["labels"].tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# Augmentation box bookkeeping
# --------------------------------------------------------------------------- #
def _box_image(hw, boxes, value=255):
    """A (1, 1, H, W) image with ``value`` inside each xyxy box (edge coords)."""
    img = torch.zeros(1, 1, *hw, dtype=torch.uint8)
    for x1, y1, x2, y2 in boxes:
        img[..., int(y1) : int(y2), int(x1) : int(x2)] = value
    return img


def _box_of_mask(img):
    """Edge-coordinate xyxy extent of the nonzero pixels of a (1, 1, H, W) image."""
    ys, xs = torch.nonzero(img[0, 0] > 127, as_tuple=True)
    return [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


@pytest.mark.parametrize("mode", ["squash", "letterbox"])
def test_resize_frame_moves_boxes_with_pixels(mode):
    box = [40.0, 20.0, 120.0, 60.0]
    img = _box_image((100, 200), [box])
    out, boxes, (sx, sy) = resize_frame(img, torch.tensor([box]), (64, 64), mode)
    assert out.shape[-2:] == (64, 64)
    if mode == "squash":
        assert (sx, sy) == (64 / 200, 64 / 100)
    else:
        assert sx == sy == pytest.approx(64 / 200)
        assert (out[..., 33:, :] == 0).all()  # padded bottom
    np.testing.assert_allclose(_box_of_mask(out), boxes[0].tolist(), atol=1.0)


def test_hflip_boxes():
    box = [10.0, 5.0, 30.0, 15.0]
    img = _box_image((20, 50), [box])
    out, boxes = hflip(img, torch.tensor([box]))
    assert boxes[0].tolist() == [20.0, 5.0, 40.0, 15.0]
    assert _box_of_mask(out) == boxes[0].tolist()


def test_shortest_side_size_matches_rfdetr():
    assert shortest_side_size(480, 640, 400) == (400, 533)
    assert shortest_side_size(640, 480, 500) == (667, 500)
    assert shortest_side_size(400, 600, 400) == (400, 600)


def test_random_resize_crop_boxes_follow_pixels():
    torch.manual_seed(0)
    box = [100.0, 150.0, 300.0, 350.0]
    img = _box_image((480, 640), [box])
    for _ in range(20):
        out, boxes = random_resize_crop(img, torch.tensor([box]))
        h, w = out.shape[-2:]
        assert h == w and 384 <= h <= 600  # a square crop of 384..600 px
        clipped, keep = clip_bboxes(boxes, (h, w), 0.0, 0.0, edge_coords=True)
        if keep[0]:
            np.testing.assert_allclose(_box_of_mask(out), clipped[0].tolist(), atol=1.5)
        else:
            assert (out > 127).sum() == 0 or min(out.shape[-2:]) < 1


def test_crop_drops_degenerate_boxes():
    # A box left of the crop window has no area once clipped: dropped.
    boxes = torch.tensor([[-30.0, 10.0, -5.0, 20.0], [5.0, 5.0, 15.0, 15.0]])
    clipped, keep = clip_bboxes(
        boxes, (32, 32), min_size=0.0, min_visible_fraction=0.0, edge_coords=True
    )
    assert keep.tolist() == [False, True]
    # Edge coordinates clip to the frame size, not size - 1.
    clipped, _ = clip_bboxes(
        torch.tensor([[0.0, 0.0, 40.0, 40.0]]), (32, 32), 0.0, 0.0, edge_coords=True
    )
    assert clipped[0].tolist() == [0.0, 0.0, 32.0, 32.0]


def test_detr_train_transform_output_and_boxes():
    torch.manual_seed(1)
    box = [200.0, 100.0, 400.0, 300.0]
    img = _box_image((480, 640), [box])
    for _ in range(10):
        out, boxes = detr_train_transform(img, torch.tensor([box]), (96, 128), "squash")
        assert out.shape[-2:] == (96, 128)
        clipped, keep = clip_bboxes(boxes, (96, 128), 0.0, 0.0, edge_coords=True)
        if keep[0] and (clipped[0, 2:] - clipped[0, :2]).min() > 4:
            np.testing.assert_allclose(_box_of_mask(out), clipped[0].tolist(), atol=2.0)


# --------------------------------------------------------------------------- #
# Multi-scale
# --------------------------------------------------------------------------- #
def test_multi_scale_sizes_match_rfdetr():
    # RF-DETR-M: resolution 576, patch 16 x 2 windows -> 416..736 in steps of 32.
    sizes = multi_scale_sizes((576, 576), 32, 5)
    assert [h for h, _ in sizes] == list(range(416, 737, 32))
    assert all(h == w for h, w in sizes)
    # Sizes under 2 blocks are left out; non-square inputs step both sides.
    assert multi_scale_sizes((96, 160), 32, 2) == [
        (64, 128),
        (96, 160),
        (128, 192),
        (160, 224),
    ]


def test_multi_scale_drops_sizes_with_too_few_tokens():
    """A training size never has fewer tokens than num_queries (top-k proposals)."""
    from sleap_nn.data.custom_datasets import detr_multi_scale_sizes

    cfg = OmegaConf.create(
        {
            "data_config": {
                "use_augmentations_train": True,
                "preprocessing": {"max_height": 256, "max_width": 256},
                "augmentation_config": {
                    "detr": {"multi_scale": True, "multi_scale_steps": 5}
                },
            },
            "model_config": {
                "backbone_config": {"pretrained": {"max_stride": 32}},
                "head_configs": {
                    "bbox_detr": {"detr": {"output_stride": 16, "num_queries": 100}}
                },
            },
        }
    )
    sizes = detr_multi_scale_sizes(cfg)
    # 96 x 96 (36 tokens) and 128 x 128 (64) are dropped; 160 x 160 has 100.
    assert sizes[0] == (160, 160) and sizes[-1] == (416, 416)
    cfg.model_config.head_configs.bbox_detr.detr.num_queries = 30
    assert detr_multi_scale_sizes(cfg)[0] == (96, 96)


def test_batch_scale_is_step_deterministic():
    import random

    sizes = multi_scale_sizes((576, 576), 32, 5)
    picks = [batch_scale(step, sizes) for step in range(50)]
    assert picks == [batch_scale(step, sizes) for step in range(50)]
    assert picks == [tuple(random.Random(s).choice(sizes)) for s in range(50)]
    assert len(set(picks)) > 3


def test_resize_batch_keeps_normalized_boxes():
    x = torch.zeros(2, 3, 64, 96)
    x[..., 16:32, 24:48] = 1.0  # box (0.25, 0.25)-(0.5, 0.5) normalized
    out = resize_batch(x, (128, 192))
    assert out.shape == (2, 3, 128, 192)
    ys, xs = torch.nonzero(out[0, 0] > 0.5, as_tuple=True)
    assert (xs.min() / 192, ys.min() / 128) == (0.25, 0.25)
    assert resize_batch(x, (64, 96)) is x


# --------------------------------------------------------------------------- #
# End to end: ModelTrainer on CPU
# --------------------------------------------------------------------------- #
def _detr_train_config(config, tmp_path, run_name="bbox_detr", **detr_aug):
    from sleap_nn.config.get_config import get_head_configs

    cfg = config.copy()
    pre = cfg.data_config.preprocessing
    pre.crop_size = None
    pre.ensure_rgb = True
    pre.max_height = 128
    pre.max_width = 128
    pre.resize_mode = "squash"
    cfg.data_config.augmentation_config = {
        "intensity": {"contrast_p": 0.5},
        "geometric": None,
        "detr": {
            "shortest_sides": [96, 128],
            "crop_range": [64, 128],
            "multi_scale_steps": 1,
            **detr_aug,
        },
    }
    cfg.model_config.backbone_config = {"dinov2_windowed": dict(TINY_BACKBONE)}
    heads = OmegaConf.structured(get_head_configs("bbox_detr"))
    heads.bbox_detr.detr.update(TINY_DETR)
    cfg.model_config.head_configs = heads
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.max_epochs = 2
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = run_name
    cfg.trainer_config.visualize_preds_during_training = True
    cfg.trainer_config.eval = {"enabled": True, "frequency": 1}
    cfg.trainer_config.optimizer_name = "AdamW"
    cfg.trainer_config.optimizer = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "encoder_lr_scale": 1.5,
        "layer_decay": 0.8,
    }
    cfg.trainer_config.lr_scheduler = {"reduce_lr_on_plateau": None}
    cfg.trainer_config.ema = {"enabled": True, "decay": 0.993, "warmup_steps": 100}
    cfg.trainer_config.gradient_clip_val = 0.1
    return cfg


def test_bbox_detr_train_predict_eval(config, minimal_instance, tmp_path):
    """Train a tiny bbox_detr on derived boxes, reload it, predict and evaluate."""
    from sleap_nn.evaluation import run_evaluation
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.callbacks import WeightEMA
    from sleap_nn.training.lightning_modules import BBoxDetrLightningModule
    from sleap_nn.training.model_trainer import ModelTrainer

    cfg = _detr_train_config(config, tmp_path)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    module = trainer.lightning_model
    assert isinstance(module, BBoxDetrLightningModule)
    # Multi-scale: 128 +/- 1 step of 32.
    assert module.multi_scale_sizes == [(96, 96), (128, 128), (160, 160)]
    # EMA from trainer_config.ema with RF-DETR's decay and tau.
    ema = [cb for cb in trainer.trainer.callbacks if isinstance(cb, WeightEMA)]
    assert len(ema) == 1 and ema[0].decay == 0.993 and ema[0].warmup_steps == 100
    assert ema[0].num_updates > 0
    assert trainer.trainer.gradient_clip_val == 0.1
    assert trainer.trainer.checkpoint_callback.monitor == "eval/val/bbox_mAP"

    run_dir = tmp_path / "bbox_detr"
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    assert list(saved.model_config.head_configs.bbox_detr.detr.categories) == ["object"]
    assert "eval/val/bbox_mAP" in (run_dir / "training_log.csv").read_text()
    # Every loss term is logged: final, per earlier decoder layer and encoder.
    logged = trainer.trainer.callback_metrics
    for stage in ("train", "val"):
        for term in ("loss_ce", "loss_bbox", "loss_giou", "cardinality_error"):
            for suffix in ("", "_0", "_enc"):
                assert f"{stage}/{term}{suffix}" in logged

    pred = Predictor.from_model_paths(
        [run_dir.as_posix()], device="cpu", peak_threshold=0.0
    )
    assert isinstance(pred.layer, BBoxLayer)
    labels = pred.predict(minimal_instance.as_posix())
    boxes = labels[0].bboxes
    assert boxes and all(isinstance(b, sio.PredictedBoundingBox) for b in boxes)
    assert all(-1 <= b.x1 < b.x2 <= 385 and -1 <= b.y1 < b.y2 <= 385 for b in boxes)
    out = tmp_path / "pred.slp"
    labels.save(out.as_posix())
    m = run_evaluation(minimal_instance.as_posix(), out.as_posix(), match_method="bbox")
    assert m is not None and 0.0 <= m["bbox_mAP"] <= 1.0


# --------------------------------------------------------------------------- #
# Lightning module: optimizer groups, class-subset init, config round-trip
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_import(tmp_path_factory):
    """The tiny HF RF-DETR fixture imported as a sleap-nn model directory.

    Reading the checkpoint needs ``safetensors`` (the ``backbones`` extra); CI runs
    these tests in the pretrained-backbones job.
    """
    pytest.importorskip("safetensors")
    from sleap_nn.convert.rfdetr import import_rfdetr

    out = tmp_path_factory.mktemp("rfdetr") / "tiny"
    import_rfdetr(TINY_DIR, out)
    return out


def _tiny_training_config(model_dir, **trainer_overrides):
    """The imported model's config, made trainable (optimizer knobs as RF-DETR)."""
    cfg = OmegaConf.load(Path(model_dir) / "training_config.yaml")
    cfg.trainer_config.optimizer_name = "AdamW"
    cfg.trainer_config.optimizer.lr = 1e-4
    cfg.trainer_config.optimizer.weight_decay = 1e-4
    cfg.trainer_config.optimizer.encoder_lr_scale = 1.5
    cfg.trainer_config.optimizer.layer_decay = 0.8
    for key, value in trainer_overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def _module(cfg):
    """The training module of ``cfg``, schema-checked as the trainer does."""
    from sleap_nn.config.training_job_config import verify_training_cfg
    from sleap_nn.training.lightning_modules import LightningModel

    return LightningModel.get_lightning_model_from_config(verify_training_cfg(cfg))


def test_param_groups_follow_rfdetr(tiny_import):
    """lr / weight decay of every parameter, as RF-DETR's get_param_dict."""
    module = _module(_tiny_training_config(tiny_import))
    optimizer = module._build_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    group_of = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in group_of, "a parameter is in two groups"
            group_of[id(p)] = group
    trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
    assert set(group_of) == {id(p) for p in trainable.values()}
    # Never used by the forward pass: frozen, so DDP needs no unused-param search.
    assert not module.model.backbone.embeddings.mask_token.requires_grad

    lr, wd = 1e-4, 1e-4
    # RF-DETR: num_layers = last tap + 1 (= 5 here), rate 0.8 ** (num_layers + 1 -
    # depth), embeddings depth 0, block k depth k + 1, final norm depth 6.
    num_layers = max(module.model.backbone.out_indices) + 1
    for name, p in trainable.items():
        group = group_of[id(p)]
        if name.startswith("model.backbone."):
            rel = name[len("model.backbone.") :]
            if rel.startswith("embeddings."):
                depth = 0
            elif rel.startswith("encoder.layer."):
                depth = int(rel.split(".")[2]) + 1
            else:
                depth = num_layers + 1
            expected_lr = lr * 1.5 * 0.7**2 * 0.8 ** (num_layers + 1 - depth)
            no_decay = any(
                k in name
                for k in ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")
            )
            expected_wd = 0.0 if no_decay else wd
        elif name.startswith("model.detr.decoder."):
            expected_lr, expected_wd = lr * 0.7, wd
        else:
            expected_lr, expected_wd = lr, wd
        assert group["lr"] == pytest.approx(expected_lr), name
        assert group["weight_decay"] == expected_wd, name
    # Spot checks of RF-DETR's weight-decay rule.
    names = {id(p): n for n, p in trainable.items()}
    by_name = {names[k]: g for k, g in group_of.items()}
    assert (
        by_name["model.backbone.encoder.layer.0.layer_scale1.lambda1"]["weight_decay"]
        == wd
    )  # RF-DETR decays the layer scale ("gamma" is not in its name)
    assert (
        by_name["model.backbone.embeddings.patch_embeddings.projection.weight"][
            "weight_decay"
        ]
        == 0.0
    )
    assert by_name["model.detr.class_embed.bias"]["weight_decay"] == wd
    assert (
        by_name["model.detr.decoder.layers.0.self_attn_layer_norm.weight"][
            "weight_decay"
        ]
        == wd
    )


def test_param_groups_without_layer_decay(tiny_import):
    cfg = _tiny_training_config(
        tiny_import,
        **{
            "trainer_config.optimizer.layer_decay": None,
            "trainer_config.optimizer.encoder_lr_scale": None,
            "trainer_config.optimizer.weight_decay": None,
        },
    )
    module = _module(cfg)
    groups = {g["name"]: g for g in module._trainable_param_groups()}
    assert groups["base"]["lr"] == pytest.approx(1e-4)
    assert groups["decoder"]["lr"] == pytest.approx(0.7e-4)
    assert groups["backbone"]["lr"] == pytest.approx(0.49e-4)
    # weight_decay unset: torch's AdamW default on every group.
    optimizer = module._build_optimizer()
    assert {g["weight_decay"] for g in optimizer.param_groups} == {0.01}


def test_class_subset_init_from_imported_rfdetr(tiny_import):
    """A class subset of an imported RF-DETR keeps the matching class rows."""
    source = torch.load(tiny_import / "best.ckpt", weights_only=False)["state_dict"]
    cfg = _tiny_training_config(tiny_import)
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert list(detr.categories)[:3] == ["LABEL_0", "LABEL_1", "LABEL_2"]
    detr.categories = ["LABEL_3", "LABEL_1"]
    cfg.model_config.init_from = {
        "path": tiny_import.as_posix(),
        "node_map": {"__extra__": "LABEL_3"},
    }
    module = _module(cfg)
    state = module.state_dict()
    keys = ["model.detr.class_embed"] + [
        f"model.detr.enc_out_class_embed.{g}" for g in range(detr.group_detr)
    ]
    rows = [3, 1, 3]  # LABEL_3, LABEL_1, then the extra logit from LABEL_3
    for key in keys:
        for suffix in ("weight", "bias"):
            ours = state[f"{key}.{suffix}"]
            assert ours.shape[0] == 3
            torch.testing.assert_close(ours, source[f"{key}.{suffix}"][rows])
    # Everything else is copied as is.
    for key in (
        "model.backbone.encoder.layer.1.mlp.fc1.weight",
        "model.detr.query_feat.weight",
    ):
        torch.testing.assert_close(state[key], source[key])
    report = module.transfer_reports[-1]
    assert any("class rows copied by name" in v for v in report.remapped.values())
    assert not report.skipped

    # Default: the extra logit comes from the source's own extra logit.
    cfg.model_config.init_from = {"path": tiny_import.as_posix()}
    state = _module(cfg).state_dict()
    torch.testing.assert_close(
        state["model.detr.class_embed.bias"],
        source["model.detr.class_embed.bias"][[3, 1, 5]],
    )


def test_checkpoint_records_detr_class_rows(tiny_import, tmp_path):
    from sleap_nn.training.weight_transfer import (
        CHECKPOINT_HEADS_KEY,
        DETR_EXTRA_LOGIT,
    )

    cfg = _tiny_training_config(tiny_import)
    module = _module(cfg)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    names = list(cfg.model_config.head_configs.bbox_detr.detr.categories)
    assert checkpoint[CHECKPOINT_HEADS_KEY] == [
        {"name": "BBoxDetrHead", "channel_names": names + [DETR_EXTRA_LOGIT]}
    ]


def test_bbox_detr_training_config_round_trip(tmp_path):
    """The bbox_detr training knobs survive a TrainingJobConfig YAML round-trip."""
    from sleap_nn.config.training_job_config import verify_training_cfg

    sample = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "sample_configs"
        / "config_bbox_detr_rfdetr_m_person.yaml"
    )
    cfg = verify_training_cfg(OmegaConf.load(sample))
    detr_aug = cfg.data_config.augmentation_config.detr
    assert (detr_aug.crop_p, detr_aug.flip_p, detr_aug.multi_scale_steps) == (
        0.5,
        0.5,
        5,
    )
    assert list(detr_aug.shortest_sides) == [400, 500, 600]
    assert list(detr_aug.crop_range) == [384, 600]
    detr = cfg.model_config.head_configs.bbox_detr.detr
    assert detr.lr_component_decay == 0.7 and list(detr.categories) == ["person"]
    assert cfg.model_config.init_from.node_map == {"__extra__": "person"}
    assert cfg.trainer_config.ema.decay == 0.993
    assert cfg.trainer_config.ema.warmup_steps == 100
    assert cfg.trainer_config.gradient_clip_val == 0.1
    assert cfg.trainer_config.precision == "bf16-mixed"
    assert cfg.data_config.preprocessing.resize_mode == "squash"

    out = tmp_path / "round_trip.yaml"
    OmegaConf.save(cfg, out)
    again = verify_training_cfg(OmegaConf.load(out))
    assert OmegaConf.to_container(again) == OmegaConf.to_container(cfg)


# --------------------------------------------------------------------------- #
# Loss parity with HF transformers' RfDetrForObjectDetection(labels=...)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("index", [0, 1], ids=["2-images", "96px"])
def test_training_losses_match_hf(tiny_import, index):
    """Same weights, batch and targets: every loss term matches HF in float64.

    ``hf_losses.pt`` (``tests/convert/make_rfdetr_fixture.py``) holds HF's training
    losses of the tiny RF-DETR (train mode: 3 query groups) on fixed images and
    boxes. The imported model, run through the module's training-loss path
    (normalization, dataset targets, collate, criterion), must match each term:
    final layer, every earlier decoder layer and the encoder proposals.
    """
    from sleap_nn.data.normalization import normalize_on_gpu

    case = torch.load(TINY_DIR / "hf_losses.pt", weights_only=False)["cases"][index]
    module = _module(_tiny_training_config(tiny_import))
    state = torch.load(tiny_import / "best.ckpt", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module = module.double().train()
    side = case["case"]["side"]
    encoder = SetTargetEncoder(max_targets=4)
    samples = []
    for t in case["labels"]:
        cx, cy, w, h = t["boxes"].unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        samples.append(encoder(xyxy * side, t["class_labels"], (side, side)))
    batch = torch.utils.data.default_collate(samples)
    batch["image"] = case["image"][:, None]  # (B, 1, C, H, W) uint8

    x = normalize_on_gpu(torch.squeeze(batch["image"], dim=1)).double()
    with torch.no_grad():
        loss, terms = module._compute_losses(module.model(x), batch)

    expected = case["loss_dict"]
    assert set(terms) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(
            terms[key].double(), value.double(), rtol=0, atol=1e-4, msg=key
        )
    torch.testing.assert_close(loss.double(), case["loss"], rtol=0, atol=1e-4)


@pytest.mark.parametrize("schedule", ["constant", "step", "cosine"])
def test_lr_schedules(tiny_import, schedule):
    """constant / step / cosine from the stock lr_scheduler options, per step.

    Every param group follows the same curve, keeping its RF-DETR ratio to ``lr``.
    """
    from types import SimpleNamespace

    sched = {
        "constant": {"reduce_lr_on_plateau": None},
        "step": {
            "reduce_lr_on_plateau": None,
            "multi_step_lr": {"milestones": [15], "gamma": 0.1},
        },
        "cosine": {
            "reduce_lr_on_plateau": None,
            "cosine_annealing_warmup": {"warmup_steps": 0, "eta_min": 0.0},
        },
    }[schedule]
    cfg = _tiny_training_config(tiny_import)
    cfg.trainer_config.lr_scheduler = sched
    module = _module(cfg)
    steps_per_epoch, epochs = 10, 20
    module._trainer = SimpleNamespace(
        num_training_batches=steps_per_epoch,
        accumulate_grad_batches=1,
        max_epochs=epochs,
        max_steps=-1,
        train_dataloader=object(),
        check_val_every_n_epoch=1,
    )
    out = module.configure_optimizers()
    optimizer = out["optimizer"]
    base = [g["lr"] for g in optimizer.param_groups]
    if schedule == "constant":
        assert "lr_scheduler" not in out
        return
    scheduler = out["lr_scheduler"]["scheduler"]
    assert out["lr_scheduler"]["interval"] == "step"
    factors = []
    for _ in range(steps_per_epoch * epochs):
        factors.append(optimizer.param_groups[0]["lr"] / base[0])
        for g, b in zip(optimizer.param_groups, base):
            assert g["lr"] / b == pytest.approx(factors[-1], abs=1e-9)
        optimizer.step()
        scheduler.step()
    if schedule == "step":
        assert factors[15 * steps_per_epoch - 1] == pytest.approx(1.0)
        assert factors[15 * steps_per_epoch] == pytest.approx(0.1)
    else:
        assert factors[0] == pytest.approx(1.0)
        assert factors[len(factors) // 2] == pytest.approx(0.5, abs=0.02)
        assert factors[-1] < 0.01


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def _derived_box_labels(minimal_instance):
    from sleap_nn.data.bboxes import add_bboxes_to_labels

    labels = sio.load_slp(minimal_instance.as_posix())
    add_bboxes_to_labels(labels, source="instances", padding=2.0)
    return labels


def test_detr_dataset_paths_and_validation(minimal_instance):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _derived_box_labels(minimal_instance)
    common = dict(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
    )
    # train_hw defaults to max_hw; grayscale frames, stock geometric aug on top.
    ds = DetrBBoxDataset(
        **common,
        ensure_grayscale=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        detr_aug={
            "crop_p": 1.0,
            "shortest_sides": [96],
            "crop_range": [64, 96],
            "flip_p": 1.0,
        },
        geometric_aug={"scale_min": 0.9, "scale_max": 1.1, "affine_p": 1.0},
    )
    assert ds.train_hw == (96, 128)
    sample = ds[0]
    assert sample["image"].shape[-3:] == (1, 96, 128)
    n = int(sample["set_count"])
    assert n == sample["num_instances"] and n <= 2
    if n:
        assert (sample["set_boxes"][:n, 2:] > 0).all()
    with pytest.raises(ValueError, match="scale must be 1.0"):
        DetrBBoxDataset(**common, scale=0.5, max_hw=(96, 96))
    with pytest.raises(ValueError, match="max_height"):
        DetrBBoxDataset(**common)
    with pytest.raises(ValueError, match="squash or letterbox"):
        resize_frame(torch.zeros(1, 1, 8, 8), torch.zeros(0, 4), (4, 4), "stretch")


def test_detr_dataset_kwargs_need_max_hw(config):
    from sleap_nn.data.custom_datasets import detr_dataset_kwargs

    cfg = config.copy()
    cfg.data_config.preprocessing.max_height = None
    with pytest.raises(ValueError, match="max_height and"):
        detr_dataset_kwargs(cfg)


def test_module_ignores_xavier_and_validates_optimizer(tiny_import, caplog):
    cfg = _tiny_training_config(tiny_import)
    cfg.model_config.init_weights = "xavier"
    module = _module(cfg)
    assert module.init_weights == "default"
    # Without multi-scale the training batch passes through unchanged.
    x = torch.zeros(1, 3, 64, 64)
    assert module._train_input(x) is x
    module.optimizer = "SGD"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        module._build_optimizer()


def test_float_outputs_casts_half_only():
    from sleap_nn.training.lightning_modules import _float_outputs

    out = _float_outputs(
        {
            "pred_logits": torch.zeros(1, dtype=torch.bfloat16),
            "aux_outputs": [{"pred_boxes": torch.zeros(1, dtype=torch.float16)}],
            "enc_outputs": None,
            "kept": torch.zeros(1, dtype=torch.float64),
        }
    )
    assert out["pred_logits"].dtype == torch.float32
    assert out["aux_outputs"][0]["pred_boxes"].dtype == torch.float32
    assert out["enc_outputs"] is None and out["kept"].dtype == torch.float64


def test_dinov2_unused_blocks_frozen_and_lr_groups():
    from sleap_nn.architectures.detr import WindowedDinov2Backbone

    bb = WindowedDinov2Backbone(
        hidden_size=16, num_layers=4, num_heads=2, pos_embed_grid=4, out_indices=[2]
    )
    assert not any(p.requires_grad for p in bb.encoder.layer[2:].parameters())
    assert all(p.requires_grad for p in bb.encoder.layer[:2].parameters())
    assert bb.encoder_param_groups(1.0) is None
    (group,) = bb.encoder_param_groups(1.0, encoder_lr_scale=2.0)
    assert group["lr"] == 2.0
    assert {id(p) for p in group["params"]} == {
        id(p) for p in bb.parameters() if p.requires_grad
    }


def test_class_rows_fresh_and_mismatched(tiny_import):
    from sleap_nn.training.weight_transfer import (
        DETR_HEAD_NAME,
        transfer_state_dict,
    )

    source = torch.load(tiny_import / "best.ckpt", weights_only=False)["state_dict"]
    cfg = _tiny_training_config(tiny_import)
    cfg.model_config.head_configs.bbox_detr.detr.categories = ["LABEL_2", "new"]
    cfg.model_config.init_from = {"path": tiny_import.as_posix()}
    module = _module(cfg)
    report = module.transfer_reports[-1]
    assert any("new" in v for v in report.reinitialized.values())
    torch.testing.assert_close(
        module.state_dict()["model.detr.class_embed.bias"][0],
        source["model.detr.class_embed.bias"][2],
    )
    # Source class names that do not match the rows: skipped, not misloaded.
    names = [f"LABEL_{i}" for i in range(4)] + ["__extra__"]  # one row short
    report = transfer_state_dict(
        module,
        source,
        source_heads=[{"name": DETR_HEAD_NAME, "channel_names": names}],
    )
    assert "model.detr.class_embed.weight" in report.skipped
