"""Tests for shape-tolerant / skeleton-aware weight transfer, reinit and freezing."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.training.callbacks import EncoderFreezeSchedule
from sleap_nn.training.lightning_modules import LightningModel
from sleap_nn.training.weight_transfer import (
    CHECKPOINT_HEADS_KEY,
    _resize_pos_embed,
    init_from_checkpoint,
    load_source,
    model_head_channels,
    reinit,
    resolve_selector,
    select_keys,
    set_requires_grad,
    transfer_state_dict,
    vit_blocks,
)

HEAD_W = "model.head_layers.0.CenteredInstanceConfmapsHead.0.weight"
HEAD_B = "model.head_layers.0.CenteredInstanceConfmapsHead.0.bias"

requires_timm = pytest.mark.skipif(
    importlib.util.find_spec("timm") is None, reason="needs the `timm` package"
)


def _ci_config(ckpt_dir, part_names=None, filters=None):
    cfg = OmegaConf.load(Path(ckpt_dir) / "training_config.yaml")
    if part_names is not None:
        cfg.model_config.head_configs.centered_instance.confmaps.part_names = list(
            part_names
        )
    if filters is not None:
        cfg.model_config.backbone_config.unet.filters = filters
    return cfg


def _build(cfg, **init_from):
    if init_from:
        cfg = cfg.copy()
        cfg.model_config.init_from = init_from
    return LightningModel.get_lightning_model_from_config(cfg)


def _source_sd(ckpt_dir):
    return torch.load(
        Path(ckpt_dir) / "best.ckpt", map_location="cpu", weights_only=False
    )["state_dict"]


# ----------------------------------------------------------------- transfer


def test_same_architecture_loads_everything(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt), path=str(ckpt))
    report = model.transfer_reports[-1]
    src = _source_sd(ckpt)
    assert not report.skipped and not report.remapped and not report.missing
    assert len(report.loaded) == len(src)
    for k, v in model.state_dict().items():
        assert torch.equal(v, src[k]), k


def test_head_channels_remapped_by_node_name(minimal_instance_centered_instance_ckpt):
    """Source A,B -> target B,C,A: permuted copy + one fresh channel."""
    ckpt = minimal_instance_centered_instance_ckpt
    fresh = _build(_ci_config(ckpt, part_names=["B", "C", "A"]))
    fresh_w = fresh.state_dict()[HEAD_W].clone()
    torch.manual_seed(0)
    model = _build(_ci_config(ckpt, part_names=["B", "C", "A"]), path=str(ckpt))
    report = model.transfer_reports[-1]
    src = _source_sd(ckpt)
    w, b = model.state_dict()[HEAD_W], model.state_dict()[HEAD_B]
    assert w.shape[0] == 3
    assert torch.equal(w[0], src[HEAD_W][1]) and torch.equal(w[2], src[HEAD_W][0])
    assert torch.equal(b[0], src[HEAD_B][1]) and torch.equal(b[2], src[HEAD_B][0])
    # The new node is not a copy of any source channel.
    assert not any(torch.equal(w[1], src[HEAD_W][i]) for i in range(2))
    assert set(report.remapped) == {HEAD_W, HEAD_B}
    assert "2/3 channels" in report.remapped[HEAD_W]
    assert any("C" in v for v in report.reinitialized.values())
    assert not report.skipped
    assert fresh_w.shape == w.shape


def test_same_count_reordered_skeleton_is_permuted(
    minimal_instance_centered_instance_ckpt,
):
    """Same shape but a different node order must NOT be copied blindly."""
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt, part_names=["B", "A"]), path=str(ckpt))
    src = _source_sd(ckpt)
    w = model.state_dict()[HEAD_W]
    assert torch.equal(w[0], src[HEAD_W][1]) and torch.equal(w[1], src[HEAD_W][0])


def test_node_map_seeds_new_node(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(
        _ci_config(ckpt, part_names=["A", "B", "C"]),
        path=str(ckpt),
        node_map={"C": "B"},
    )
    src = _source_sd(ckpt)
    w = model.state_dict()[HEAD_W]
    assert torch.equal(w[2], src[HEAD_W][1])
    assert not model.transfer_reports[-1].reinitialized


def test_shape_mismatch_is_skipped_and_reported(
    minimal_instance_centered_instance_ckpt,
):
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt, filters=8), path=str(ckpt))
    report = model.transfer_reports[-1]
    assert report.skipped  # conv weights with 8 vs 16 filters
    assert all("!= target" in r for r in report.skipped.values())
    assert "skipped" in report.summary()
    with pytest.raises(ValueError, match="strict=True"):
        _build(_ci_config(ckpt, filters=8), path=str(ckpt), strict=True)


def test_include_exclude(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt), path=str(ckpt), include=["head"])
    report = model.transfer_reports[-1]
    assert report.loaded and all("head_layers" in k for k in report.loaded)
    model = _build(_ci_config(ckpt), path=str(ckpt), exclude=["head"])
    report = model.transfer_reports[-1]
    assert not any("head_layers" in k for k in report.loaded)
    keys = list(model.state_dict())
    assert select_keys(model, keys, ["backbone"], ["decoder"]) == [
        k for k in keys if "backbone" in k and ".dec." not in k
    ]


def test_legacy_pretrained_head_weights_remap(
    minimal_instance_centered_instance_ckpt,
):
    """`pretrained_head_weights` no longer crashes on a grown skeleton."""
    ckpt = minimal_instance_centered_instance_ckpt
    cfg = _ci_config(ckpt, part_names=["B", "A", "C"])
    cfg.model_config.pretrained_head_weights = str(Path(ckpt) / "best.ckpt")
    cfg.model_config.pretrained_backbone_weights = str(Path(ckpt) / "best.ckpt")
    model = LightningModel.get_lightning_model_from_config(cfg)
    src = _source_sd(ckpt)
    assert torch.equal(model.state_dict()[HEAD_W][1], src[HEAD_W][0])
    assert len(model.transfer_reports) == 2


def test_checkpoint_records_head_channels(
    tmp_path, minimal_instance_centered_instance_ckpt
):
    """Checkpoints carry their channel names; no training_config.yaml needed."""
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt))
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint[CHECKPOINT_HEADS_KEY] == [
        {"name": "CenteredInstanceConfmapsHead", "channel_names": ["A", "B"]}
    ]
    torch.save(checkpoint, tmp_path / "x.ckpt")  # no yaml next to it
    _, heads = load_source(str(tmp_path / "x.ckpt"))
    assert heads[0]["channel_names"] == ["A", "B"]
    target = _build(_ci_config(ckpt, part_names=["B", "A"]))
    init_from_checkpoint(target, str(tmp_path / "x.ckpt"))
    assert torch.equal(target.state_dict()[HEAD_W][0], model.state_dict()[HEAD_W][1])


def test_source_part_names_override(tmp_path, minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    sd = _source_sd(ckpt)
    target = _build(_ci_config(ckpt, part_names=["B", "A"]))
    report = transfer_state_dict(target, sd, source_part_names=["A", "B"])
    assert HEAD_W in report.remapped
    assert torch.equal(target.state_dict()[HEAD_W][0], sd[HEAD_W][1])


def test_resize_pos_embed():
    src = torch.randn(1, 1 + 4 * 4, 8)
    out = _resize_pos_embed(src, torch.Size([1, 1 + 6 * 6, 8]))
    assert out.shape == (1, 37, 8) and torch.equal(out[:, 0], src[:, 0])
    assert _resize_pos_embed(torch.randn(1, 15, 8), torch.Size([1, 37, 8])) is None
    out = _resize_pos_embed(torch.randn(16, 8), torch.Size([36, 8]))
    assert out.shape == (36, 8)


# -------------------------------------------------------------------- reinit


def test_reinit_head_only(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    model = _build(_ci_config(ckpt), path=str(ckpt), reinit=["head"])
    src = _source_sd(ckpt)
    sd = model.state_dict()
    assert not torch.equal(sd[HEAD_W], src[HEAD_W])
    assert not torch.equal(sd[HEAD_B], src[HEAD_B])
    for k in src:
        if "head_layers" not in k:
            assert torch.equal(sd[k], src[k]), k


# ------------------------------------------------------------ plain-ViT bits

VIT = "facebook/dinov2-with-registers-small"
TIMM_VIT = "timm/vit_small_patch16_dinov3.lvd1689m"


def _vit_lightning(model_name, part_names=("a", "b"), **bb):
    pretrained = dict(
        source="hf",
        model_name=model_name,
        weights=False,
        mode="auto",
        in_channels=3,
        output_stride=4,
        max_stride=32,
        normalize=True,
    )
    pretrained.update(bb)
    heads = OmegaConf.create(
        {
            "centered_instance": {
                "confmaps": {
                    "part_names": list(part_names),
                    "anchor_part": None,
                    "sigma": 2.5,
                    "output_stride": 4,
                    "loss_weight": 1.0,
                }
            }
        }
    )
    from sleap_nn.training.lightning_modules import (
        TopDownCenteredInstanceLightningModule,
    )

    return TopDownCenteredInstanceLightningModule(
        model_type="centered_instance",
        backbone_type="pretrained",
        backbone_config=OmegaConf.create({"pretrained": pretrained}),
        head_configs=heads,
        init_weights="xavier",
        learning_rate=1e-3,
        optimizer="AdamW",
    )


@pytest.fixture(scope="module")
def dinov2_model():
    torch.manual_seed(0)
    return _vit_lightning(VIT)


def test_vit_block_selectors(dinov2_model):
    m = dinov2_model
    name, blocks = vit_blocks(m.model.backbone)
    assert len(blocks) == 12
    globs = resolve_selector(m, "blocks[-2:]")
    assert globs == [f"model.backbone.{name}.10.*", f"model.backbone.{name}.11.*"]
    emb = resolve_selector(m, "embeddings")
    assert emb and all("embeddings" in g for g in emb)
    enc = resolve_selector(m, "encoder")
    assert enc == ["model.backbone.enc.*"]


def test_reinit_last_blocks(dinov2_model):
    m = _vit_lightning(VIT)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    done = reinit(m, ["blocks[-2:]"], layer_scale=1e-5)
    assert len(done) == 2
    name, _ = vit_blocks(m.model.backbone)
    after = m.state_dict()
    changed = {k for k in after if not torch.equal(after[k], before[k])}
    assert changed and all(
        k.startswith(f"model.backbone.{name}.10.")
        or k.startswith(f"model.backbone.{name}.11.")
        for k in changed
    )
    ls = [v for k, v in after.items() if f"{name}.11." in k and "lambda1" in k]
    assert ls and torch.allclose(ls[0], torch.full_like(ls[0], 1e-5))


@pytest.mark.parametrize(
    "model_name",
    [VIT, pytest.param(TIMM_VIT, marks=requires_timm)],
    ids=["hf-dinov2", "timm-dinov3"],
)
def test_resolution_finetune_roundtrip(model_name):
    """A model trained at one crop size loads and runs at another.

    Plain-ViT pos-embeds are interpolated per call (HF) or RoPE (timm DINOv3), so
    the stored state dict does not depend on the crop size: 256 -> 384 loads with
    nothing skipped and the output grid follows the input.
    """
    torch.manual_seed(0)
    a = _vit_lightning(model_name).eval()
    b = _vit_lightning(model_name).eval()
    report = transfer_state_dict(b, a.state_dict(), source="a")
    assert not report.skipped and not report.missing and not report.resized
    pos = [
        k for k in a.state_dict() if k.endswith(("position_embeddings", "pos_embed"))
    ]
    for k in pos:  # stored grid stays at pretraining size
        assert a.state_dict()[k].shape == b.state_dict()[k].shape
    with torch.no_grad():
        for size in (256, 384):
            x = torch.rand(1, 3, size, size)
            ya = a.model(x)["CenteredInstanceConfmapsHead"]
            yb = b.model(x)["CenteredInstanceConfmapsHead"]
            assert ya.shape == (1, 2, size // 4, size // 4)
            torch.testing.assert_close(ya, yb)


def test_freeze_schedule_keeps_layer_decay_groups():
    m = _vit_lightning(VIT, layer_decay=0.75)
    opt = m.configure_optimizers()["optimizer"]
    lrs_before = [g["lr"] for g in opt.param_groups]
    assert len(lrs_before) > 2
    dead = {n for n, p in m.named_parameters() if not p.requires_grad}
    trainer = SimpleNamespace(optimizers=[opt], global_step=0)
    cb = EncoderFreezeSchedule(targets=["encoder"], unfreeze_at_step=3)
    cb.on_fit_start(trainer, m)
    enc = [n for n, _ in m.named_parameters() if n.startswith("model.backbone.enc.")]
    assert cb.frozen and all(
        not p.requires_grad for n, p in m.named_parameters() if n in enc
    )
    assert not m.model.backbone.enc.training or m.model.backbone.freeze
    m.train()
    assert not m.model.backbone.enc.training  # mixin keeps frozen encoder in eval
    trainer.global_step = 2
    cb.on_train_batch_start(trainer, m, None, 0)
    assert cb.frozen
    trainer.global_step = 3
    cb.on_train_batch_start(trainer, m, None, 0)
    assert not cb.frozen and m.model.backbone.enc.training
    now_trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert set(enc) - dead <= now_trainable
    assert not (dead & now_trainable)  # parameters excluded on purpose stay off
    assert [g["lr"] for g in opt.param_groups] == lrs_before


def test_set_requires_grad_first_blocks(dinov2_model):
    m = _vit_lightning(VIT)
    changed = set_requires_grad(m, ["embeddings", "blocks[:6]"], False)
    name, _ = vit_blocks(m.model.backbone)
    assert any(f"{name}.5." in n for n in changed)
    assert not any(f"{name}.6." in n for n in changed)
    assert any("embeddings" in n for n in changed)


# --------------------------------------------- real checkpoint (opt-in, local)


@pytest.mark.skipif(
    not os.environ.get("SLEAP_NN_TEST_COCO17_CKPT"),
    reason="set SLEAP_NN_TEST_COCO17_CKPT to a COCO-17 model dir",
)
def test_real_coco17_to_union46():
    run = Path(os.environ["SLEAP_NN_TEST_COCO17_CKPT"])
    cfg = OmegaConf.load(run / "training_config.yaml")
    coco = list(cfg.model_config.head_configs.centered_instance.confmaps.part_names)
    union = coco[:5] + ["aic_neck"] + coco[5:] + ["left_big_toe", "jrdb_head"]
    cfg.model_config.head_configs.centered_instance.confmaps.part_names = union
    model = _build(cfg, path=str(run))
    report = model.transfer_reports[-1]
    assert not report.skipped and not report.missing
    assert "17/20 channels" in report.remapped[HEAD_W]
    src = _source_sd(run)
    w = model.state_dict()[HEAD_W]
    for ti, name in enumerate(union):
        if name in coco:
            assert torch.equal(w[ti], src[HEAD_W][coco.index(name)])
    assert model_head_channels(model)[0]["channel_names"] == union


# ---------------------------------------------------- freeze schedule details

_WORKER = Path(__file__).parent / "ddp_freeze_schedule_worker.py"


def _worker_module():
    spec = importlib.util.spec_from_file_location("_ddp_freeze_worker", _WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "strategy", ["ddp", "ddp_find_unused_parameters_false"], ids=["auto", "refused"]
)
def test_freeze_schedule_under_ddp(tmp_path, strategy):
    """2-process gloo run: DDP needs find_unused_parameters for a late freeze.

    The schedule freezes after DDP has wrapped the model. With plain ``ddp`` the
    callback turns ``find_unused_parameters`` on (without it the second step
    raises "parameters that were not used in producing the loss"); an explicit
    ``find_unused_parameters=False`` is refused with a clear error.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PL_DISABLE_FORK": "1"}
    proc = subprocess.run(
        [sys.executable, str(_WORKER), str(tmp_path), strategy],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if strategy == "ddp_find_unused_parameters_false":
        assert proc.returncode != 0
        assert "freeze_schedule freezes parameters after DDP" in (
            proc.stderr + proc.stdout
        )
        return
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = torch.load(tmp_path / "result.pt", weights_only=False)
    assert result["find_unused"] is True
    snaps = result["snapshots"]
    # Frozen for steps 0-1 (the snapshot at step 2 is taken before its update),
    # trained from step 2 on, and identical on both ranks.
    assert torch.equal(snaps[0], snaps[1]) and torch.equal(snaps[0], snaps[2])
    assert not torch.equal(snaps[2], snaps[3])
    final = result["final_per_rank"]
    assert torch.equal(final[0], final[1])


def test_freeze_schedule_errors():
    mod = _worker_module()
    module = mod._Module()
    opt = module.configure_optimizers()
    trainer = SimpleNamespace(optimizers=[opt], global_step=0)
    with pytest.raises(ValueError, match="match no parameter"):
        EncoderFreezeSchedule(targets=["nothing.*"]).on_fit_start(trainer, module)
    module.model.backbone.freeze_encoder()  # what `freeze: true` does
    with pytest.raises(ValueError, match="freeze: false"):
        EncoderFreezeSchedule(unfreeze_at_step=2).on_fit_start(trainer, module)
    # Static freezing on top of `freeze: true` is merely redundant.
    EncoderFreezeSchedule().on_fit_start(trainer, module)
    # Resumed past the unfreeze point: nothing to do.
    module = mod._Module()
    cb = EncoderFreezeSchedule(unfreeze_at_step=2)
    cb.on_fit_start(SimpleNamespace(optimizers=[opt], global_step=5), module)
    assert not cb.frozen
    assert all(p.requires_grad for p in module.parameters())


def test_unfreeze_encoder_restores_only_what_it_froze():
    mod = _worker_module()
    bb = mod._Backbone()
    bb.enc[1].weight.requires_grad_(False)  # excluded on purpose
    bb.freeze_encoder()
    assert not any(p.requires_grad for p in bb.enc.parameters())
    bb.train()
    assert not bb.enc.training
    bb.unfreeze_encoder()
    assert bb.enc.training and not bb.freeze
    assert bb.enc[0].weight.requires_grad and bb.enc[1].bias.requires_grad
    assert not bb.enc[1].weight.requires_grad


# --------------------------------------------------- transfer edge cases


def test_unknown_source_names_copy_by_position_with_warning(
    tmp_path, minimal_instance_centered_instance_ckpt
):
    """A raw state dict without node names is copied by position, with a warning."""
    ckpt = minimal_instance_centered_instance_ckpt
    torch.save(_source_sd(ckpt), tmp_path / "raw.pt")  # no names, no yaml
    target = _build(_ci_config(ckpt, part_names=["B", "A"]))
    report = init_from_checkpoint(target, str(tmp_path / "raw.pt"))
    assert report.warnings and "by position" in report.warnings[0]
    assert "WARNING" in report.summary()
    assert torch.equal(target.state_dict()[HEAD_W], _source_sd(ckpt)[HEAD_W])


def test_only_head_output_layer_is_remapped():
    """Hidden layers of a deeper head are shape-matched, never permuted."""
    from torch import nn

    from sleap_nn.training.weight_transfer import _head_output_keys

    class _Deep(nn.Module):
        def __init__(self, names):
            super().__init__()
            self.backbone = nn.Conv2d(1, 2, 1)
            self.head_layers = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(2, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1))]
            )
            self.heads = [SimpleNamespace(name="H", part_names=names, channels=2)]

    src, tgt = _Deep(["a", "b"]), _Deep(["b", "a"])
    assert _head_output_keys(tgt) == {"head_layers.0.2.weight", "head_layers.0.2.bias"}
    report = transfer_state_dict(
        tgt,
        src.state_dict(),
        source_heads=[{"name": "H", "channel_names": ["a", "b"]}],
    )
    s, t = src.state_dict(), tgt.state_dict()
    assert torch.equal(t["head_layers.0.0.weight"], s["head_layers.0.0.weight"])
    assert torch.equal(t["head_layers.0.2.weight"][0], s["head_layers.0.2.weight"][1])
    assert set(report.remapped) == {"head_layers.0.2.weight", "head_layers.0.2.bias"}


def test_resume_skips_init_from(minimal_instance_centered_instance_ckpt):
    ckpt = minimal_instance_centered_instance_ckpt
    cfg = _ci_config(ckpt)
    cfg.trainer_config.resume_ckpt_path = str(Path(ckpt) / "best.ckpt")
    model = _build(cfg, path="/does/not/exist.ckpt", reinit=["head"])
    assert model.transfer_reports == []


def test_config_schema():
    from sleap_nn.config.model_config import InitFromConfig
    from sleap_nn.config.trainer_config import FreezeScheduleConfig, TrainerConfig

    cfg = OmegaConf.structured(TrainerConfig())
    assert cfg.freeze_schedule is None
    cfg = OmegaConf.merge(
        cfg, {"freeze_schedule": {"targets": ["blocks[:6]"], "unfreeze_at_step": 10}}
    )
    assert OmegaConf.to_object(cfg).freeze_schedule.unfreeze_at_step == 10
    with pytest.raises(ValueError):
        FreezeScheduleConfig(unfreeze_at_step=-1)
    init = OmegaConf.structured(
        InitFromConfig(path="x.ckpt", node_map={"neck": "nose"}, reinit=["head"])
    )
    assert init.node_map == {"neck": "nose"} and init.strict is False


def test_train_with_init_from_and_freeze_schedule(
    config, tmp_path, minimal_instance_centered_instance_ckpt
):
    """End to end: init_from + reinit + freeze schedule through ModelTrainer."""
    from sleap_nn.training.model_trainer import ModelTrainer

    ckpt = minimal_instance_centered_instance_ckpt
    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", str(tmp_path))
    OmegaConf.update(cfg, "trainer_config.run_name", "init_from")
    OmegaConf.update(cfg, "trainer_config.max_epochs", 1)
    OmegaConf.update(cfg, "trainer_config.trainer_accelerator", "cpu")
    OmegaConf.update(
        cfg,
        "model_config.init_from",
        {"path": str(ckpt), "reinit": ["head"]},
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "trainer_config.freeze_schedule",
        {"targets": ["encoder"], "unfreeze_at_step": 1},
        force_add=True,
    )
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    report = trainer.lightning_model.transfer_reports[-1]
    assert report.loaded and not report.skipped
    run = tmp_path / "init_from"
    saved = torch.load(run / "best.ckpt", map_location="cpu", weights_only=False)
    assert saved[CHECKPOINT_HEADS_KEY][0]["channel_names"] == ["A", "B"]
    saved_cfg = OmegaConf.load(run / "training_config.yaml")
    assert saved_cfg.model_config.init_from.path == str(ckpt)
    assert saved_cfg.trainer_config.freeze_schedule.unfreeze_at_step == 1
