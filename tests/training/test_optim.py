"""Tests for the optimizer param-group helpers (decay split, encoder LR keys)."""

import pytest
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import nn

from sleap_nn.training.optim import (
    is_no_decay,
    migrate_encoder_lr_options,
    no_weight_decay_names,
    resolve_encoder_lr_options,
    split_weight_decay,
)


class _TinyViT(nn.Module):
    """Names and shapes of the parameters the split has to recognise."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 4, stride=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, 8))  # 3-D
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 8))  # 3-D
        self.register_tokens = nn.Parameter(torch.zeros(1, 2, 8))
        self.mask_token = nn.Parameter(torch.zeros(1, 8))  # 2-D
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8)) for _ in range(2)]
        )
        self.gamma = nn.Parameter(torch.ones(8))  # layer scale (1-D)
        self.special = nn.Parameter(torch.zeros(2, 8))  # declared below
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 2)

    def no_weight_decay(self):
        return {"special"}


class _Wrapper(nn.Module):
    """A timm model nested under another module (names must be prefixed)."""

    def __init__(self):
        super().__init__()
        self.enc = _TinyViT()
        self.position_embedding = nn.Embedding(4, 8)  # CLIP-style, 2-D weight


def _names(module, params):
    ids = {id(p) for p in params}
    return sorted(n for n, p in module.named_parameters() if id(p) in ids)


def test_no_weight_decay_names_are_prefixed():
    assert no_weight_decay_names(_Wrapper()) == {"enc.special"}


@pytest.mark.parametrize(
    "name, shape, expected",
    [
        ("enc.blocks.0.1.weight", (8, 8), False),
        ("enc.patch_embed.weight", (8, 3, 4, 4), False),
        ("enc.blocks.0.1.bias", (8,), True),
        ("enc.blocks.0.0.weight", (8,), True),  # LayerNorm
        ("embeddings.position_embeddings", (1, 4, 8), True),
        ("embeddings.position_embedding.weight", (4, 8), True),
        ("embeddings.cls_token", (1, 1, 8), True),
        ("embeddings.register_tokens", (1, 4, 8), True),
        ("embeddings.mask_token", (1, 8), True),
        ("timm_model.reg_token", (1, 4, 8), True),
        ("features.0.attn.relative_position_bias_table", (49, 3), True),
        ("my_pos_embedder.weight", (8, 8), False),  # whole-component match only
    ],
)
def test_is_no_decay(name, shape, expected):
    assert is_no_decay(name, torch.zeros(shape)) is expected


def test_flat_parameters_split_in_two():
    model = _Wrapper()
    groups = split_weight_decay(list(model.parameters()), model, 0.1)
    assert [g["name"] for g in groups] == ["base/decay", "base/no_decay"]
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    assert _names(model, groups[0]["params"]) == [
        "enc.blocks.0.1.weight",
        "enc.blocks.1.1.weight",
        "enc.head.weight",
        "enc.patch_embed.weight",
    ]
    no_decay = _names(model, groups[1]["params"])
    for n in (
        "enc.pos_embed",
        "enc.cls_token",
        "enc.register_tokens",
        "enc.mask_token",
        "enc.gamma",
        "enc.special",
        "enc.bn.weight",
        "enc.bn.bias",
        "enc.blocks.0.0.weight",
        "enc.head.bias",
        "position_embedding.weight",
    ):
        assert n in no_decay
    covered = [id(p) for g in groups for p in g["params"]]
    assert sorted(covered) == sorted(id(p) for p in model.parameters())


def test_lr_groups_keep_their_lr_and_extra_keys():
    """Each LR group (e.g. a layer_decay depth) splits and keeps its lr."""
    model = _TinyViT()
    groups_in = [
        {"params": list(model.head.parameters()), "lr": 1e-3},
        {"params": list(model.blocks[1].parameters()), "lr": 7.5e-4, "foo": 1},
        {"params": [model.pos_embed, model.cls_token], "lr": 1e-4},  # no-decay only
    ]
    out = split_weight_decay(groups_in, model, 0.05)
    assert [(g["name"], g["lr"], g["weight_decay"]) for g in out] == [
        ("group0/decay", 1e-3, 0.05),
        ("group0/no_decay", 1e-3, 0.0),
        ("group1/decay", 7.5e-4, 0.05),
        ("group1/no_decay", 7.5e-4, 0.0),
        ("group2/no_decay", 1e-4, 0.0),  # empty decay half dropped
    ]
    assert out[2]["foo"] == out[3]["foo"] == 1


def test_optimizer_accepts_the_groups_and_applies_decay_only_to_weights():
    """AdamW with lr=0 isolates the decoupled decay: only `decay` params shrink."""
    model = _TinyViT()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    groups = split_weight_decay(list(model.parameters()), model, 0.5)
    opt = torch.optim.AdamW(groups, lr=1.0, weight_decay=0.5)
    for g in opt.param_groups:
        g["lr"] = 1.0
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    # AdamW: p -= lr * wd * p, and the Adam update is 0 for a zero grad.
    assert torch.allclose(model.head.weight, torch.full_like(model.head.weight, 0.5))
    assert torch.all(model.head.bias == 1.0)
    assert torch.all(model.pos_embed == 1.0)
    assert torch.all(model.cls_token == 1.0)


# ── encoder_lr_scale / layer_decay: trainer_config.optimizer + deprecated alias ──


def _lr_cfg(optimizer=None, pretrained=None):
    """A config with just the keys the encoder-LR resolution reads."""
    return OmegaConf.create(
        {
            "trainer_config": {"optimizer": {"lr": 1e-3, **(optimizer or {})}},
            "model_config": {
                "backbone_config": {
                    "pretrained": (
                        None
                        if pretrained is None
                        else {
                            "encoder_lr_scale": 1.0,
                            "layer_decay": None,
                            **pretrained,
                        }
                    )
                }
            },
        }
    )


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(handler)


def test_encoder_lr_options_unset_everywhere(warnings_log):
    assert resolve_encoder_lr_options(_lr_cfg()) == (None, None)
    assert resolve_encoder_lr_options(_lr_cfg(pretrained={})) == (None, None)
    assert not warnings_log


def test_encoder_lr_options_from_the_optimizer(warnings_log):
    cfg = _lr_cfg(optimizer={"encoder_lr_scale": 0.1, "layer_decay": 0.75})
    assert resolve_encoder_lr_options(cfg) == (0.1, 0.75)
    assert not warnings_log


def test_encoder_lr_options_fall_back_to_the_deprecated_backbone_keys(warnings_log):
    cfg = _lr_cfg(pretrained={"encoder_lr_scale": 0.5, "layer_decay": 0.8})
    assert resolve_encoder_lr_options(cfg) == (0.5, 0.8)
    assert sum("deprecated" in m for m in warnings_log) == 2


def test_encoder_lr_options_same_value_in_both_places_is_fine():
    cfg = _lr_cfg(optimizer={"layer_decay": 0.8}, pretrained={"layer_decay": 0.8})
    assert resolve_encoder_lr_options(cfg) == (None, 0.8)


def test_encoder_lr_options_conflict_raises():
    cfg = _lr_cfg(
        optimizer={"encoder_lr_scale": 0.1}, pretrained={"encoder_lr_scale": 0.5}
    )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_encoder_lr_options(cfg)


def test_migrate_moves_the_backbone_keys_to_the_optimizer():
    """The saved config names each knob once, in its new place."""
    from sleap_nn.config.model_config import PretrainedConfig
    from sleap_nn.config.training_job_config import TrainingJobConfig

    cfg = OmegaConf.structured(TrainingJobConfig())
    cfg.model_config.backbone_config.pretrained = PretrainedConfig(
        model_name="x", encoder_lr_scale=0.25, layer_decay=0.7
    )
    migrate_encoder_lr_options(cfg)
    assert cfg.trainer_config.optimizer.encoder_lr_scale == 0.25
    assert cfg.trainer_config.optimizer.layer_decay == 0.7
    assert cfg.model_config.backbone_config.pretrained.encoder_lr_scale == 1.0
    assert cfg.model_config.backbone_config.pretrained.layer_decay is None
    # Idempotent, and nothing left to warn about.
    assert resolve_encoder_lr_options(cfg, warn=False) == (0.25, 0.7)
    migrate_encoder_lr_options(cfg)
    assert cfg.trainer_config.optimizer.encoder_lr_scale == 0.25


def test_migrate_is_a_no_op_without_backbone_keys():
    cfg = _lr_cfg()
    before = OmegaConf.to_container(cfg)
    migrate_encoder_lr_options(cfg)
    assert OmegaConf.to_container(cfg) == before
