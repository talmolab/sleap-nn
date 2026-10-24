"""Tests for external pretrained (HuggingFace) backbones.

Most tests build backbones with ``weights=False`` (random init from the model
config) so they never touch the network — the HF architecture code is exercised
without downloading checkpoints. They are skipped when ``transformers`` (the
``backbones`` optional extra) is not installed, mirroring the ``requires_onnxruntime``
convention in ``tests/export``.

**That convention hid a bug.** ``weights=True`` and ``weights=False`` used to
configure the backbone by *different mechanisms* — kwargs to ``from_pretrained``
vs. attributes on the config — and only the weights=False one tolerated an option
the model class does not declare. So ``mode="encoder"`` was broken for every CNN
backbone with real weights while every test here passed. The two paths now share
one mechanism, and ``TestWeightsTrueParity`` below pins the parity directly. Those
tests need a cached model (or the network), so they are marked ``hf_weights`` and
skipped unless ``SLEAP_NN_TEST_HF_WEIGHTS=1``.
"""

import copy
import importlib.util
import os
import re

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

transformers = pytest.importorskip("transformers")

from sleap_nn.architectures.model import Model
from sleap_nn.architectures.pretrained import (
    PretrainedBackbone,
    _default_layers,
    _default_token_stride,
    _resolve_mode,
)
from tests.training.test_embedding_model_semantics import (  # noqa: F401 (fixture)
    _config as _emb_train_config,
    _train as _emb_train,
    tracked_slp,
)


def _caseA_cfg(model_name, output_stride=2, max_stride=32, **kw):
    base = dict(
        source="hf",
        model_name=model_name,
        weights=False,
        mode="auto",
        in_channels=3,
        output_stride=output_stride,
        max_stride=max_stride,
        normalize=True,
    )
    base.update(kw)
    return OmegaConf.create(base)


# --------------------------------------------------------------- Case A wrappers


@pytest.mark.parametrize(
    "model_name",
    [
        "facebook/convnextv2-nano-22k-224",
        "microsoft/resnet-50",
        "microsoft/swinv2-tiny-patch4-window8-256",
    ],
)
def test_caseA_wrapper_contract(model_name):
    """Hierarchical backbones build a decoder and emit the full contract."""
    bb = PretrainedBackbone.from_config(_caseA_cfg(model_name, output_stride=2))
    bb.eval()

    assert bb.mode == "decoder"
    assert bb.max_stride == 32
    # decoder_stride_to_filters must cover the head's output stride.
    assert 2 in bb.decoder_stride_to_filters
    assert 32 in bb.decoder_stride_to_filters
    # middle_blocks[-1].filters exposes the bottleneck channels (class-vectors path).
    assert isinstance(bb.middle_blocks[-1].filters, int)

    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out = bb(x)
    assert set(out) == {"outputs", "strides", "middle_output", "intermediate_feat"}
    assert len(out["outputs"]) == len(out["strides"])
    # Finest output at stride 2 => half the input resolution.
    assert out["outputs"][-1].shape[-2:] == (128, 128)
    assert out["strides"][-1] == 2
    # middle_output/intermediate_feat are the bottleneck (deepest stride).
    assert out["middle_output"].shape[-2:] == (8, 8)


@pytest.mark.parametrize("output_stride", [2, 4])
def test_caseA_output_stride(output_stride):
    """The decoder reaches the requested output stride."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg("facebook/convnextv2-nano-22k-224", output_stride=output_stride)
    )
    bb.eval()
    with torch.no_grad():
        out = bb(torch.rand(1, 3, 256, 256))
    assert out["strides"][-1] == output_stride
    assert out["outputs"][-1].shape[-2] == 256 // output_stride
    assert output_stride in bb.decoder_stride_to_filters


def test_caseA_two_feature_maps_no_skip_block():
    """A 2-map pyramid (n_skips=1) with more decoder blocks than skips must not
    crash: the extra block(s) take the no-concat path. Regression for the
    down_blocks clamp (max(0, ...) not max(1, ...))."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg(
            "facebook/convnextv2-nano-22k-224",
            out_indices=[3, 4],  # strides 16, 32 -> n_skips=1
            output_stride=8,  # up_blocks=2 => one skip block + one no-skip block
        )
    )
    bb.eval()
    assert bb.mode == "decoder"
    with torch.no_grad():
        out = bb(torch.rand(1, 3, 256, 256))
    assert out["strides"][-1] == 8
    assert out["outputs"][-1].shape[-2] == 256 // 8


def test_caseA_grayscale_and_rgb_equivalent_shapes():
    """Grayscale (1ch) input is handled by Model.forward's 1->3 replicate."""
    m = Model(
        backbone_type="pretrained",
        backbone_config=_caseA_cfg("facebook/convnextv2-nano-22k-224"),
        head_configs=OmegaConf.create(
            {"confmaps": {"part_names": ["a", "b"], "sigma": 2.5, "output_stride": 2}}
        ),
        model_type="single_instance",
    )
    m.eval()
    with torch.no_grad():
        y_gray = m(torch.rand(1, 1, 256, 256))
        y_rgb = m(torch.rand(1, 3, 256, 256))
    assert y_gray["SingleInstanceConfmapsHead"].shape == (1, 2, 128, 128)
    assert y_rgb["SingleInstanceConfmapsHead"].shape == (1, 2, 128, 128)


def test_caseA_in_model_segmentation():
    """A bottomup_segmentation model builds and runs on a pretrained backbone."""
    m = Model(
        backbone_type="pretrained",
        backbone_config=_caseA_cfg("facebook/convnextv2-nano-22k-224"),
        head_configs=OmegaConf.create(
            {
                "segmentation": {"output_stride": 2, "loss_weight": 1.0},
                "center": {"sigma": 8.0, "output_stride": 2, "loss_weight": 1.0},
                "offsets": {"output_stride": 2, "loss_weight": 1.0},
            }
        ),
        model_type="bottomup_segmentation",
    )
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 256, 256))
    assert y["SegmentationHead"].shape == (1, 1, 128, 128)
    assert y["InstanceCenterHead"].shape == (1, 1, 128, 128)
    assert y["CenterOffsetHead"].shape == (1, 2, 128, 128)


# --------------------------------------------------------------- Case B wrappers


def test_caseB_encoder_only_contract():
    """Isotropic ViTs (DINOv2) build an encoder-only pooled bottleneck."""
    bb = PretrainedBackbone.from_config(
        OmegaConf.create(
            {
                "source": "hf",
                "model_name": "facebook/dinov2-with-registers-base",
                "weights": False,
                "mode": "auto",
                "in_channels": 3,
            }
        )
    )
    bb.eval()
    assert bb.mode == "encoder"
    # No decoder => outputs empty => all heads consume middle_output.
    assert bb.decoder_stride_to_filters == {}
    assert isinstance(bb.middle_blocks[-1].filters, int)

    with torch.no_grad():
        out = bb(torch.rand(1, 3, 224, 224))
    assert out["outputs"] == []
    assert out["strides"] == []
    # Reshaped spatial bottleneck (tokens stripped): [B, hidden, H/patch, W/patch].
    assert out["middle_output"].ndim == 4
    assert out["middle_output"].shape[1] == bb.middle_blocks[-1].filters
    assert out["intermediate_feat"] is out["middle_output"]


def test_mode_override_encoder_on_hierarchical():
    """mode='encoder' forces the pooled path even on a hierarchical backbone."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg("facebook/convnextv2-nano-22k-224", mode="encoder")
    )
    bb.eval()
    assert bb.mode == "encoder"
    assert bb.decoder_stride_to_filters == {}


# --------------------------------------------------------------- normalization


def test_normalization_applied_and_toggle():
    """normalize=True subtracts mean/divides std; normalize=False is identity."""
    bb_norm = PretrainedBackbone.from_config(
        _caseA_cfg(
            "facebook/convnextv2-nano-22k-224",
            normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.25, 0.25, 0.25],
        )
    )
    x = torch.zeros(1, 3, 8, 8)
    normed = bb_norm._normalize(x)
    # (0 - 0.5) / 0.25 = -2.0 for all channels.
    assert torch.allclose(normed, torch.full_like(x, -2.0))

    bb_off = PretrainedBackbone.from_config(
        _caseA_cfg("facebook/convnextv2-nano-22k-224", normalize=False)
    )
    assert torch.allclose(bb_off._normalize(x), x)


def test_explicit_norm_stats_registered_as_buffers():
    """Explicit image_mean/std are stored as (1,3,1,1) buffers."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg(
            "microsoft/resnet-50",
            image_mean=[0.1, 0.2, 0.3],
            image_std=[0.4, 0.5, 0.6],
        )
    )
    assert bb.image_mean.shape == (1, 3, 1, 1)
    assert bb.image_std.shape == (1, 3, 1, 1)
    assert bb.image_mean.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3])


# --------------------------------------------------------------- freeze / reload


def test_freeze_encoder():
    """freeze_encoder() disables grads on the encoder only."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg("facebook/convnextv2-nano-22k-224", freeze=True)
    )
    bb.freeze_encoder()
    assert all(not p.requires_grad for p in bb.enc.parameters())
    # Decoder stays trainable.
    assert any(p.requires_grad for p in bb.dec.parameters())


def test_reload_is_noop_without_weights():
    """With weights=False there is no snapshot; reload is a safe no-op."""
    bb = PretrainedBackbone.from_config(_caseA_cfg("facebook/convnextv2-nano-22k-224"))
    assert bb._pretrained_sd is None
    bb.reload_pretrained_weights()  # must not raise


def test_encoder_trainable_by_default():
    """The encoder is left in train mode after construction (freeze=False).

    `from_pretrained` returns models in eval mode; the wrapper must reset to train
    so fine-tuning updates BatchNorm running stats (ResNet). Lightning respects
    module eval flags and won't flip it back.
    """
    bb = PretrainedBackbone.from_config(_caseA_cfg("microsoft/resnet-50"))
    assert bb.enc.training is True
    assert all(m.training for m in bb.enc.modules())
    assert all(p.requires_grad for p in bb.enc.parameters())
    # After freezing, the encoder switches to eval so norm layers use fixed stats.
    bb.freeze_encoder()
    assert bb.enc.training is False


# --------------------------------------------------------------- errors / misc


def test_unsupported_source_raises():
    with pytest.raises(ValueError, match="Unsupported pretrained backbone source"):
        PretrainedBackbone(source="torchvision", model_name="resnet50", weights=False)


def test_encoder_only_backbone_with_spatial_head_raises():
    """A spatial head on an encoder-only backbone gives an actionable error,
    not a bare KeyError."""
    with pytest.raises(ValueError, match="encoder-only backbone supports only"):
        Model(
            backbone_type="pretrained",
            backbone_config=OmegaConf.create(
                {
                    "source": "hf",
                    "model_name": "facebook/dinov2-with-registers-base",
                    "weights": False,
                    "mode": "encoder",
                    "in_channels": 3,
                    "output_stride": 2,
                    "max_stride": 14,
                }
            ),
            head_configs=OmegaConf.create(
                {"confmaps": {"part_names": ["a"], "sigma": 2.5, "output_stride": 2}}
            ),
            model_type="single_instance",
        )


def test_resolve_mode():
    class _Cfg:
        def __init__(self, model_type, stage_names):
            self.model_type = model_type
            self.stage_names = stage_names

    # explicit modes pass through
    assert _resolve_mode(_Cfg("convnextv2", ["stem", "s1"]), "encoder") == "encoder"
    assert _resolve_mode(_Cfg("dinov2", None), "decoder") == "decoder"
    # auto: hierarchical -> decoder, isotropic ViT -> encoder
    assert _resolve_mode(_Cfg("convnextv2", ["stem", "s1", "s2"]), "auto") == "decoder"
    assert _resolve_mode(_Cfg("dinov2", None), "auto") == "encoder"
    assert _resolve_mode(_Cfg("resnet", ["stem", "s1", "s2"]), "auto") == "decoder"


# ─────────────────────────────────────────────────────────────────────────
# weights=True — the path CI never runs, where the config-vs-kwargs
# divergence lived. Opt in with SLEAP_NN_TEST_HF_WEIGHTS=1 (needs a cached
# model or the network).
# ─────────────────────────────────────────────────────────────────────────

requires_hf_weights = pytest.mark.skipif(
    os.environ.get("SLEAP_NN_TEST_HF_WEIGHTS") != "1",
    reason="needs HF weights; set SLEAP_NN_TEST_HF_WEIGHTS=1 to run",
)


class TestWeightsTrueParity:
    """`weights=True` must accept exactly what `weights=False` accepts."""

    @requires_hf_weights
    @pytest.mark.parametrize(
        "model_name",
        [
            "facebook/convnextv2-nano-22k-224",  # CNN: no reshape_hidden_states
            "facebook/dinov2-with-registers-base",  # ViT: has it
        ],
    )
    @pytest.mark.parametrize("mode", ["auto", "encoder"])
    def test_builds_with_and_without_weights(self, model_name, mode):
        """Both weight settings build for both families, in both modes.

        `mode="encoder"` + a CNN + `weights=True` raised
        `TypeError: ConvNextV2Backbone.__init__() got an unexpected keyword
        argument 'reshape_hidden_states'` — the Case-B options were forwarded to
        the model constructor, and only ViT classes declare them.
        """
        for weights in (False, True):
            bb = PretrainedBackbone.from_config(
                _caseA_cfg(model_name, mode=mode, weights=weights)
            )
            out = bb(torch.rand(1, 3, 128, 128))
            assert "middle_output" in out and "intermediate_feat" in out

    @requires_hf_weights
    def test_weights_are_actually_loaded_and_snapshotted(self):
        """weights=True must snapshot real weights for the post-xavier reload."""
        bb = PretrainedBackbone.from_config(
            _caseA_cfg("facebook/convnextv2-nano-22k-224", weights=True)
        )
        assert bb._pretrained_sd is not None and len(bb._pretrained_sd) > 0
        live = bb.enc.state_dict()
        assert all(torch.equal(v, live[k].cpu()) for k, v in bb._pretrained_sd.items())


def test_backbone_class_resolves_from_config_not_automodel():
    """The concrete `*Backbone` class is resolved from the (offline-capable) config.

    `AutoBackbone.from_pretrained` probes `hf_api().repo_exists()` before resolving
    anything, so it raises `OfflineModeIsEnabled` under `HF_HUB_OFFLINE=1` even for
    a fully-cached model — breaking the air-gapped workflow the guide documents.
    Resolving the class off `AutoConfig` (which has no such probe) avoids it.
    """
    from transformers import AutoBackbone, AutoConfig

    from sleap_nn.architectures.pretrained import _resolve_backbone_class

    cfg = AutoConfig.from_pretrained("facebook/convnextv2-nano-22k-224")
    cls = _resolve_backbone_class(cfg)
    assert cls is not AutoBackbone
    assert cls.__name__.endswith("Backbone")

    # An unmapped config type falls back to AutoBackbone rather than raising.
    class _Unmapped:
        pass

    assert _resolve_backbone_class(_Unmapped()) is AutoBackbone


def test_frozen_encoder_stays_in_eval_through_train():
    """A frozen encoder must not be flipped back to train mode.

    `freeze_encoder()` calls `enc.eval()` once, but Lightning calls `model.train()`
    at the start of every epoch and that recurses into submodules. Weights stayed
    frozen; BatchNorm running statistics did not — measured on `microsoft/resnet-18`,
    `running_mean` moved 0.63 in a single forward, so a "frozen" run was not
    reproducible.
    """
    bb = PretrainedBackbone.from_config(_caseA_cfg("microsoft/resnet-18", freeze=True))
    bb.freeze_encoder()
    assert not bb.enc.training

    bb.train()  # what Lightning does at every epoch start
    assert not bb.enc.training, "frozen encoder was flipped back into train mode"

    bns = [
        m
        for m in bb.enc.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert bns, "resnet-18 should have BatchNorm layers"
    before = bns[0].running_mean.clone()
    with torch.no_grad():
        bb(torch.rand(2, 3, 128, 128))
    assert torch.equal(before, bns[0].running_mean), "frozen BN stats drifted"


def test_unfrozen_encoder_still_follows_train_and_eval():
    """The override must not pin an UNfrozen encoder to eval."""
    bb = PretrainedBackbone.from_config(_caseA_cfg("microsoft/resnet-18", freeze=False))
    bb.train()
    assert bb.enc.training
    bb.eval()
    assert not bb.enc.training


# ------------------------------------------------------- embedding (re-ID) x pretrained


def _embedding_cfg(model_name, mode="auto", max_stride=32):
    """A full model config for a lone pooled `embedding` head on `model_name`."""
    return OmegaConf.create(
        {
            "model_config": {
                "backbone_config": {
                    "pretrained": _caseA_cfg(
                        model_name, output_stride=2, max_stride=max_stride, mode=mode
                    )
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "centered_instance": None,
                    "bottomup": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None,
                    "embedding": {
                        "embedding": {
                            "embedding_dim": 128,
                            "num_fc_layers": 1,
                            "num_fc_units": 256,
                            "pool": "gem",
                            "normalize": True,
                            "output_stride": max_stride,
                            "loss_weight": 1.0,
                            "freeze_backbone": False,
                            "anchor_part": None,
                            "objective": None,
                        }
                    },
                },
            }
        }
    )


def _build_embedding_model(model_name, mode="auto"):
    """Config -> model, through the same `check_output_strides` the trainer runs."""
    from sleap_nn.config.utils import check_output_strides

    cfg = check_output_strides(_embedding_cfg(model_name, mode=mode))
    return Model.from_config(
        backbone_type="pretrained",
        backbone_config=cfg.model_config.backbone_config.pretrained,
        head_configs=cfg.model_config.head_configs.embedding,
        model_type="embedding",
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "facebook/convnextv2-nano-22k-224",
        "microsoft/resnet-50",
        "facebook/dinov2-with-registers-small",
    ],
)
def test_embedding_builds_on_the_default_mode(model_name):
    """`embedding` + `pretrained` must build with `mode: auto` (the default).

    This combination used to be unconfigurable: `check_output_strides` pins the
    embedding head's stride to `max_stride`, and the pretrained wrapper reads
    `output_stride == max_stride` as "nothing to decode" and raises. Only an
    isotropic ViT with an explicit `mode: encoder` got through, so the hierarchical
    CNN backbones -- the useful ones -- were closed off entirely.
    """
    model = _build_embedding_model(model_name)
    out = model(torch.rand(2, 3, 64, 64))
    assert out["EmbeddingHead"].shape == (2, 128)


def test_embedding_has_no_dead_decoder_parameters():
    """Every parameter must receive gradient: a decoder here would be dead weight.

    A pooled head reads the encoder bottleneck (`Model.forward` routes it to
    `intermediate_feat`), so a decoder built underneath it is never touched --
    measured at 4.9 M of 20.1 M parameters on convnextv2-nano. Unused parameters
    are also a DDP hazard, which is why the fix is `mode="encoder"` rather than
    relaxing the stride check.
    """
    model = _build_embedding_model("facebook/convnextv2-nano-22k-224")
    model(torch.rand(2, 3, 64, 64))["EmbeddingHead"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"{len(no_grad)} parameter tensor(s) received no gradient"


def test_embedding_rejects_an_explicit_decoder_mode():
    """`mode: decoder` under a pooled head fails with a reason, not a stride error."""
    from sleap_nn.config.utils import check_output_strides

    with pytest.raises(ValueError, match="no gradient"):
        check_output_strides(
            _embedding_cfg("facebook/convnextv2-nano-22k-224", mode="decoder")
        )


def _bn_buffers(module):
    """Every BatchNorm buffer (running mean/var, batch count) of `module`."""
    return {
        f"{name}.{key}": buf.detach().clone()
        for name, m in module.named_modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for key, buf in m.named_buffers(recurse=False)
    }


def test_embedding_freeze_backbone_keeps_batchnorm_frozen_through_training(
    tracked_slp, tmp_path, monkeypatch
):
    """`freeze_backbone` freezes a BatchNorm encoder for real, through `train()`.

    It used to set `requires_grad=False` only: the encoder stayed in train mode, so
    every training forward normalized with batch statistics and overwrote the
    running statistics (measured on resnet-18: 20 forwards, zero optimizer steps,
    eval embeddings of a fixed batch at cos 0.35 to the originals). Now it is the
    `pretrained.freeze` mechanism: no gradient, eval mode for the whole run -- and
    only the encoder is frozen.
    """
    from sleap_nn.training.lightning_modules import EmbeddingLightningModule

    steps = []
    original = EmbeddingLightningModule.training_step

    def spy(self, batch, batch_idx):
        enc = self.model.backbone.enc
        steps.append(
            {
                "bn": _bn_buffers(enc),
                "modes": {m.training for m in enc.modules()},
                "enc": {n: p.detach().clone() for n, p in enc.named_parameters()},
            }
        )
        return original(self, batch, batch_idx)

    monkeypatch.setattr(EmbeddingLightningModule, "training_step", spy)

    backbone = {
        "pretrained": {
            "model_name": "microsoft/resnet-18",
            "weights": False,
            "in_channels": 3,
            "normalize": False,
        }
    }
    cfg = _emb_train_config(
        tracked_slp,
        tmp_path,
        "freeze_resnet",
        backbone=backbone,
        **{"model_config.head_configs.embedding.embedding.freeze_backbone": True},
    )
    module = _emb_train(cfg).lightning_model
    enc = module.model.backbone.enc

    assert len(steps) >= 2
    first = steps[0]
    assert first["bn"], "resnet-18 should have BatchNorm layers"
    final_bn = _bn_buffers(enc)
    for step, record in enumerate(steps):
        assert record["modes"] == {False}, f"step {step}: encoder in train mode"
    for key, value in first["bn"].items():
        assert torch.equal(final_bn[key], value), f"BatchNorm {key} changed"
    final_enc = dict(enc.named_parameters())
    for name, value in first["enc"].items():
        assert torch.equal(final_enc[name].detach(), value), f"{name} changed"

    enc_ids = {id(p) for p in enc.parameters()}
    for name, p in module.named_parameters():
        assert p.requires_grad is (id(p) not in enc_ids), name


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "gem"),
        ({"model_config.backbone_config.pretrained.freeze": True}, "avg"),
        (
            {"model_config.head_configs.embedding.embedding.freeze_backbone": True},
            "avg",
        ),
    ],
    ids=["fine-tuned", "pretrained.freeze", "freeze_backbone"],
)
def test_embedding_default_pool_is_avg_for_a_frozen_pretrained_encoder(
    tracked_slp, tmp_path, overrides, expected
):
    """An unset `pool` trains with avg on a frozen pretrained encoder, else GeM.

    GeM floors its input at eps, so it pools only the positive part of each
    channel; a pretrained encoder's LayerNorm output is about half negative and a
    frozen one cannot adapt. Measured (8 seeds): avg beats GeM by ~+0.04 val rank-1
    on frozen DINOv2 / ConvNeXtV2 and ties when fine-tuned. The resolved pooling is
    written into the saved config so inference rebuilds the same head.
    """
    from sleap_nn.architectures.heads import GeM

    backbone = {
        "pretrained": {
            "model_name": "microsoft/resnet-18",
            "weights": False,
            "in_channels": 3,
            "normalize": False,
        }
    }
    cfg = _emb_train_config(
        tracked_slp, tmp_path, "default_pool", backbone=backbone, **overrides
    )
    module = _emb_train(cfg).lightning_model

    saved = OmegaConf.load(tmp_path / "default_pool" / "training_config.yaml")
    assert saved.model_config.head_configs.embedding.embedding.pool == expected
    pool = module.model.head_layers[0].pre_embedding_pool
    if expected == "avg":
        assert isinstance(pool, torch.nn.AdaptiveAvgPool2d)
    else:
        assert isinstance(pool, GeM)


def test_embedding_explicit_pool_is_kept_on_a_frozen_pretrained_encoder():
    """The default applies only to an UNSET pool; an explicit choice is kept."""
    from sleap_nn.architectures.heads import GeM
    from sleap_nn.config.utils import check_output_strides

    cfg = _embedding_cfg("microsoft/resnet-18")
    cfg.model_config.backbone_config.pretrained.freeze = True
    assert cfg.model_config.head_configs.embedding.embedding.pool == "gem"
    cfg = check_output_strides(cfg)
    assert cfg.model_config.head_configs.embedding.embedding.pool == "gem"

    # And a model built without the trainer's config setup resolves an unset pool
    # the same way.
    cfg.model_config.head_configs.embedding.embedding.pool = None
    model = Model.from_config(
        backbone_type="pretrained",
        backbone_config=cfg.model_config.backbone_config.pretrained,
        head_configs=cfg.model_config.head_configs.embedding,
        model_type="embedding",
    )
    assert isinstance(
        model.head_layers[0].pre_embedding_pool, torch.nn.AdaptiveAvgPool2d
    )
    cfg.model_config.backbone_config.pretrained.freeze = False
    model = Model.from_config(
        backbone_type="pretrained",
        backbone_config=cfg.model_config.backbone_config.pretrained,
        head_configs=cfg.model_config.head_configs.embedding,
        model_type="embedding",
    )
    assert isinstance(model.head_layers[0].pre_embedding_pool, GeM)


# ─────────────────────────────────────────────────────────────────────────
# Case C: a plain (isotropic) ViT + feature-pyramid neck + the sleap decoder,
# so DINOv2/CLIP/SigLIP/MAE/... can feed spatial heads.
# ─────────────────────────────────────────────────────────────────────────

# Patch 14 (-> token stride 16), 12 layers, hidden 384, CLS + 4 register tokens.
VIT = "facebook/dinov2-with-registers-small"
# Two-tower checkpoints with no `*Backbone` class. CLIP: patch 16, CLS token;
# SigLIP: patch 16, NO prefix token (so the prefix count to strip is 0).
CLIP = "openai/clip-vit-base-patch16"
SIGLIP = "google/siglip-base-patch16-224"
# Pretraining-time random masking + token shuffling, both of which must be off.
MAE = "facebook/vit-mae-base"
# A `timm/...` Hub id (TimmWrapperModel): patch 16, RoPE, CLS + 4 register tokens.
TIMM_VIT = "timm/vit_small_patch16_dinov3.lvd1689m"

requires_timm = pytest.mark.skipif(
    importlib.util.find_spec("timm") is None, reason="needs the `timm` package"
)


def _confmaps_heads(output_stride=2):
    return OmegaConf.create(
        {
            "confmaps": {
                "part_names": ["a", "b"],
                "sigma": 2.5,
                "output_stride": output_stride,
            }
        }
    )


def _vit_model(
    model_name=VIT, output_stride=2, model_type="single_instance", heads=None, **kw
):
    """A `Model` on a plain ViT under the default `mode: auto`."""
    return Model(
        backbone_type="pretrained",
        backbone_config=_caseA_cfg(model_name, output_stride=output_stride, **kw),
        head_configs=heads if heads is not None else _confmaps_heads(output_stride),
        model_type=model_type,
    )


def _vit_backbone(model_name=VIT, **kw):
    """A plain-ViT backbone built the way `Model` builds it for a spatial head."""
    return PretrainedBackbone.from_config(
        _caseA_cfg(model_name, **kw), spatial_heads=True
    )


@pytest.fixture(scope="module")
def caseC_model():
    """Build each single-instance plain-ViT model once per module (read-only use)."""
    cache = {}

    def build(model_name, output_stride=2):
        key = (model_name, output_stride)
        if key not in cache:
            cache[key] = _vit_model(model_name, output_stride=output_stride).eval()
        return cache[key]

    return build


@pytest.mark.parametrize(
    "model_name, output_stride, patch_size",
    [
        (VIT, 2, 14),
        (VIT, 4, 14),
        (CLIP, 2, 16),
        (SIGLIP, 2, 16),
        pytest.param(TIMM_VIT, 2, 16, marks=requires_timm),
    ],
    ids=["dinov2-os2", "dinov2-os4", "clip-os2", "siglip-os2", "timm-dinov3-os2"],
)
def test_caseC_plain_vit_feeds_a_spatial_head(
    caseC_model, model_name, output_stride, patch_size
):
    """`mode: auto` + a spatial head on a plain ViT builds the neck and decoder.

    The token grid lands on stride 16 whatever the patch size (a patch-14 DINOv2
    input is resampled), so a non-square grayscale frame gets confidence maps at
    exactly ``H / output_stride`` x ``W / output_stride``.
    """
    model = caseC_model(model_name, output_stride)
    bb = model.backbone
    assert bb.mode == "decoder"
    assert bb.token_path
    assert bb.is_timm == model_name.startswith("timm/")
    assert bb.patch_size == patch_size
    assert bb.token_stride == 16
    assert bb.max_stride == 32
    assert sorted(bb.decoder_stride_to_filters) == [
        s for s in (2, 4, 8, 16, 32) if s >= output_stride
    ]

    with torch.no_grad():
        y = model(torch.rand(1, 1, 160, 224))["SingleInstanceConfmapsHead"]
    assert y.shape == (1, 2, 160 // output_stride, 224 // output_stride)


def test_caseC_backbone_contract(caseC_model):
    """The backbone emits the same dict contract as Case A, from a 4-level pyramid."""
    bb = caseC_model(VIT, 2).backbone
    # Neck levels at token_stride/4 .. 2*token_stride; default channels h/8..h.
    assert bb._pyramid_strides == [4, 8, 16, 32]
    assert bb._pyramid_channels == [48, 96, 192, 384]
    assert bb.middle_blocks[-1].filters == 384

    with torch.no_grad():
        out = bb(torch.rand(1, 3, 160, 224))
    assert set(out) == {"outputs", "strides", "middle_output", "intermediate_feat"}
    assert out["strides"] == [16, 8, 4, 2]
    assert [tuple(o.shape[-2:]) for o in out["outputs"]] == [
        (10, 14),
        (20, 28),
        (40, 56),
        (80, 112),
    ]
    # The bottleneck is the neck's stride-32 level.
    assert out["middle_output"].shape == (1, 384, 5, 7)
    assert out["intermediate_feat"] is out["middle_output"]


def test_caseC_neck_channels_override():
    """`neck_channels` sets every pyramid level, finest first; it needs four."""
    bb = _vit_backbone(neck_channels=[16, 32, 64, 128])
    assert bb._pyramid_channels == [16, 32, 64, 128]
    assert bb.middle_blocks[-1].filters == 128

    with pytest.raises(ValueError, match="4 output channel counts"):
        _vit_backbone(neck_channels=[16, 32, 64])


@pytest.mark.parametrize("model_name", [CLIP, SIGLIP])
def test_caseC_two_tower_checkpoint_builds_the_vision_tower_only(
    caseC_model, model_name
):
    """CLIP/SigLIP ship a text tower too; it must never become backbone params."""
    from transformers import AutoConfig

    bb = caseC_model(model_name, 2).backbone
    names = [n for n, _ in bb.enc.named_parameters()]
    assert names
    assert not any("text" in n for n in names), "text-tower parameters were built"
    assert not any(
        n.startswith(("logit_scale", "logit_bias", "visual_projection")) for n in names
    )
    assert type(bb.enc).__name__.endswith("VisionModel")
    vision = AutoConfig.from_pretrained(model_name).vision_config
    assert bb.hidden_size == vision.hidden_size
    assert bb.num_layers == vision.num_hidden_layers


class _HFCfg:
    """A minimal stand-in for an HF config, enough for `_resolve_mode`."""

    def __init__(self, model_type, stage_names):
        self.model_type = model_type
        self.stage_names = stage_names


def test_resolve_mode_follows_the_heads():
    """Under `auto`, a plain ViT is decoder iff a spatial head needs one."""
    vit = _HFCfg("dinov2", None)
    cnn = _HFCfg("convnextv2", ["stem", "stage1", "stage2"])

    assert _resolve_mode(vit, "auto", spatial_heads=True) == "decoder"
    assert _resolve_mode(vit, "auto", spatial_heads=False) == "encoder"
    assert _resolve_mode(vit, "auto") == "encoder"  # legacy default
    # A config with no stages and an unlisted model_type (CLIP) is isotropic too.
    assert _resolve_mode(_HFCfg("clip", None), "auto", spatial_heads=True) == "decoder"
    # Hierarchical backbones always decode under `auto`.
    assert _resolve_mode(cnn, "auto", spatial_heads=False) == "decoder"
    assert _resolve_mode(cnn, "auto", spatial_heads=True) == "decoder"
    # An explicit mode wins over the heads.
    assert _resolve_mode(vit, "encoder", spatial_heads=True) == "encoder"
    assert _resolve_mode(cnn, "decoder", spatial_heads=False) == "decoder"


def test_caseC_auto_keeps_a_pooled_only_model_on_case_b():
    """A pooled-only model on DINOv2 stays encoder-only, on the AutoBackbone path.

    Only spatial heads switch `auto` to the neck; a DINOv2 model with a lone
    pooled head keeps its `*Backbone` module, so existing checkpoints load.
    """
    cfg = _embedding_cfg(VIT, mode="auto")
    model = Model(
        backbone_type="pretrained",
        backbone_config=cfg.model_config.backbone_config.pretrained,
        head_configs=cfg.model_config.head_configs.embedding,
        model_type="embedding",
    )
    bb = model.backbone
    assert bb.mode == "encoder"
    assert not bb.token_path
    assert type(bb.enc).__name__.endswith("Backbone")
    assert not hasattr(bb, "neck") and not hasattr(bb, "tap_norms")
    assert bb.decoder_stride_to_filters == {}


def test_caseC_multi_class_topdown_on_a_plain_vit():
    """Confmaps + class vectors on a plain ViT build and run.

    The spatial head pulls in the neck; the pooled head reads the bottleneck.
    """
    heads = OmegaConf.create(
        {
            "confmaps": {
                "part_names": ["a", "b"],
                "anchor_part": None,
                "sigma": 2.5,
                "output_stride": 4,
            },
            "class_vectors": {
                "classes": ["x", "y", "z"],
                "num_fc_layers": 1,
                "num_fc_units": 32,
                "global_pool": True,
                "output_stride": 32,
            },
        }
    )
    model = _vit_model(
        VIT, output_stride=4, model_type="multi_class_topdown", heads=heads
    ).eval()
    assert model.backbone.mode == "decoder"
    assert model.backbone.token_path
    with torch.no_grad():
        out = model(torch.rand(2, 1, 96, 128))
    assert out["CenteredInstanceConfmapsHead"].shape == (2, 2, 24, 32)
    assert out["ClassVectorsHead"].shape == (2, 3)


@pytest.mark.parametrize(
    "model_name, hidden, unreachable",
    [
        (CLIP, 768, "post_layernorm.weight"),
        pytest.param(TIMM_VIT, 384, "timm_model.norm.weight", marks=requires_timm),
    ],
    ids=["clip", "timm-dinov3"],
)
def test_caseB_on_a_vit_without_a_backbone_class_uses_the_token_path(
    model_name, hidden, unreachable
):
    """A ViT with no `*Backbone` class (CLIP, timm) reads hidden states when pooled.

    The last layer is tapped, LayerNorm'd and reshaped to a stride-16 grid; no
    neck or decoder is built. The construction probe excludes the encoder's
    unreachable parameters (the norm after the last block) from training while
    keeping them in the state dict, so every trainable parameter gets gradient.
    """
    bb = PretrainedBackbone.from_config(_caseA_cfg(model_name))  # auto, pooled-only
    bb.eval()
    assert bb.mode == "encoder"
    assert bb.token_path
    assert bb.layers == [12]
    assert len(bb.tap_norms) == 1
    assert not hasattr(bb, "neck") and not hasattr(bb, "dec")
    assert bb.decoder_stride_to_filters == {}
    assert bb.max_stride == 16
    assert bb.middle_blocks[-1].filters == hidden

    out = bb(torch.rand(1, 3, 96, 128))
    assert out["outputs"] == [] and out["strides"] == []
    assert out["middle_output"].shape == (1, hidden, 6, 8)
    assert out["intermediate_feat"] is out["middle_output"]

    out["middle_output"].sum().backward()
    dead = [n for n, p in bb.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"{len(dead)} trainable tensor(s) received no gradient: {dead}"
    name = next(n for n, _ in bb.enc.named_parameters() if n.endswith(unreachable))
    assert not dict(bb.enc.named_parameters())[name].requires_grad
    assert name in bb.enc.state_dict()


def test_caseB_token_path_takes_exactly_one_tap():
    """Encoder mode pools one tapped layer; more would be unused tap norms."""
    with pytest.raises(ValueError, match="exactly one tapped layer"):
        PretrainedBackbone.from_config(_caseA_cfg(CLIP, mode="encoder", layers=[6, 12]))


# ------------------------------------------------------------ token stride / taps


def test_default_token_stride_is_the_nearest_power_of_two():
    assert _default_token_stride(14) == 16
    assert _default_token_stride(16) == 16
    assert _default_token_stride(8) == 8
    assert _default_token_stride(32) == 32


def test_caseC_token_stride_8_halves_every_pyramid_stride():
    """A finer token grid: pyramid at strides 2..16, so max_stride 16 is enough."""
    bb = _vit_backbone(token_stride=8, max_stride=16).eval()
    assert bb.token_stride == 8
    assert bb._pyramid_strides == [2, 4, 8, 16]
    assert bb.max_stride == 16
    assert sorted(bb.decoder_stride_to_filters) == [2, 4, 8, 16]

    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        taps = bb._tap_tokens(bb._normalize(x))
        out = bb(x)
    assert all(tuple(t.shape[-2:]) == (8, 12) for t in taps)  # grid at stride 8
    assert out["strides"] == [8, 4, 2]
    assert out["outputs"][-1].shape[-2:] == (32, 48)
    assert out["middle_output"].shape[-2:] == (4, 6)


@pytest.mark.parametrize("token_stride, max_stride", [(None, 16), (None, 48), (8, 8)])
def test_caseC_max_stride_must_be_a_multiple_of_twice_the_token_stride(
    token_stride, max_stride
):
    with pytest.raises(ValueError, match="must be a multiple of"):
        _vit_backbone(token_stride=token_stride, max_stride=max_stride)


@pytest.mark.parametrize("token_stride", [12, 2])
def test_caseC_token_stride_must_be_a_power_of_two(token_stride):
    with pytest.raises(ValueError, match="power of two"):
        _vit_backbone(token_stride=token_stride)


def test_default_layer_taps():
    """Evenly spaced, 1-indexed, ending at the last layer."""
    assert _default_layers(12, 4) == [3, 6, 9, 12]
    assert _default_layers(24, 4) == [6, 12, 18, 24]
    assert _default_layers(12, 1) == [12]
    # Fewer layers than taps: layers repeat, and layer 0 (the embeddings) is never
    # a default tap (round(0.5) == 0 used to yield one).
    assert _default_layers(2, 4) == [1, 1, 2, 2]


def test_caseC_default_taps(caseC_model):
    bb = caseC_model(VIT, 2).backbone
    assert bb.num_layers == 12
    assert bb.layers == [3, 6, 9, 12]
    assert len(bb.tap_norms) == 4


def test_caseC_negative_layer_taps_count_from_the_last():
    bb = _vit_backbone(layers=[-1, -4, 2, 6])
    assert bb.layers == [2, 6, 9, 12]


@pytest.mark.parametrize("layers", [[13], [0], [-13]])
def test_caseC_out_of_range_layer_tap_raises(layers):
    with pytest.raises(ValueError, match="out of range"):
        _vit_backbone(layers=layers)


def test_caseC_sfp_neck_builds_every_level_from_one_tap():
    bb = _vit_backbone(vit_neck="sfp", layers=[-1]).eval()
    assert bb.layers == [12]
    assert len(bb.tap_norms) == 1
    with torch.no_grad():
        out = bb(torch.rand(1, 3, 64, 96))
    assert out["strides"] == [16, 8, 4, 2]
    assert out["outputs"][-1].shape[-2:] == (32, 48)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"layers": [6, 12]}, "needs 4 layers"),
        ({"vit_neck": "sfp", "layers": [6, 12]}, "exactly one tapped layer"),
        ({"vit_neck": "fpn"}, "Unknown ViT neck mode"),
    ],
    ids=["dpt-with-2-taps", "sfp-with-2-taps", "unknown-neck"],
)
def test_caseC_neck_misconfiguration_raises(kw, match):
    with pytest.raises(ValueError, match=match):
        _vit_backbone(**kw)


# ------------------------------------------------------------------- MAE


@pytest.fixture(scope="module")
def mae_backbone():
    """One random-init MAE backbone, tapping its first four layers."""
    return _vit_backbone(MAE, layers=[1, 2, 3, 4]).eval()


def test_caseC_mae_is_deterministic(mae_backbone):
    """MAE's random masking/shuffling must be off: same input, same output."""
    assert mae_backbone.enc.config.mask_ratio == 0.0
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        a = mae_backbone(x)["outputs"][-1]
        b = mae_backbone(x)["outputs"][-1]
    assert torch.equal(a, b)


def test_caseC_mae_keeps_every_token_in_place(mae_backbone):
    """A change in one input patch moves only that patch's token-map cell.

    Zeroing every Linear in the encoder makes each pre-norm residual block the
    identity, so the layer-1 tap is the per-patch embedding at whatever grid
    cell its token was put in. MAE shuffles tokens by ``argsort(noise)``; with a
    random shuffle the change would land in a random cell (and differ per call).
    """
    bb = copy.deepcopy(mae_backbone)
    for m in bb.enc.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    x = torch.rand(1, 3, 64, 96)  # a 4 x 6 grid of 16 px patches
    x2 = x.clone()
    x2[..., 16:32, 64:80] += 0.5  # the patch at grid cell (1, 4)
    with torch.no_grad():
        a = bb._tap_tokens(bb._normalize(x))[0]
        b = bb._tap_tokens(bb._normalize(x2))[0]
    changed = ((a - b).abs().sum(dim=1)[0] > 1e-6).nonzero().tolist()
    assert changed == [[1, 4]]


def test_caseC_probe_excludes_blocks_after_the_last_tap(mae_backbone):
    """Blocks after the deepest tap stop requiring grad but stay in the state dict.

    Tapping layers 1-4 of 12: blocks 5-12 never reach a tap; blocks 1-4 train.
    """
    block = re.compile(r"(?:^|\.)layers?\.(\d+)\.")
    by_block = {}
    for name, p in mae_backbone.enc.named_parameters():
        match = block.search(name)
        if match:
            by_block.setdefault(int(match.group(1)), []).append(p.requires_grad)
    assert sorted(by_block) == list(range(12))
    for i, flags in by_block.items():
        assert all(flags) if i < 4 else not any(flags), f"block {i}"
    state = mae_backbone.enc.state_dict()
    assert all(name in state for name, _ in mae_backbone.enc.named_parameters())


# --------------------------------------------------------------- freeze / grads


def test_caseC_freeze_leaves_the_tap_norms_neck_and_decoder_trainable():
    bb = _vit_backbone(freeze=True)
    bb.freeze_encoder()
    assert all(not p.requires_grad for p in bb.enc.parameters())
    for part in (bb.tap_norms, bb.neck, bb.dec):
        assert all(p.requires_grad for p in part.parameters())

    bb.train()  # what Lightning does at every epoch start
    assert not bb.enc.training
    assert bb.tap_norms.training and bb.neck.training and bb.dec.training

    bb(torch.rand(1, 3, 64, 96))["outputs"][-1].sum().backward()
    assert all(p.grad is None for p in bb.enc.parameters())
    for part in (bb.tap_norms, bb.neck):
        assert all(p.grad is not None for p in part.parameters())


def test_caseC_gradient_checkpointing_is_enabled_on_the_encoder():
    bb = _vit_backbone(gradient_checkpointing=True)
    assert bb.enc.is_gradient_checkpointing
    bb.train()
    bb(torch.rand(1, 3, 64, 96))["outputs"][-1].sum().backward()
    assert any(p.grad is not None for p in bb.enc.parameters())


@pytest.mark.parametrize("output_stride", [2, 4, 8, 16])
def test_caseC_every_trainable_parameter_receives_gradient(output_stride):
    """Same standard as `test_embedding_has_no_dead_decoder_parameters`.

    After one forward/backward through `Model`, no parameter that requires grad
    -- encoder, tap norms, neck, decoder or head -- is left without a gradient.
    At output_stride >= 8 the neck must not build the levels (and taps) finer
    than the output stride, which the decoder would never read.
    """
    model = _vit_model(VIT, output_stride=output_stride)
    out = model(torch.rand(1, 1, 64, 96))["SingleInstanceConfmapsHead"]
    assert out.shape == (1, 2, 64 // output_stride, 96 // output_stride)
    out.sum().backward()
    dead = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not dead, f"{len(dead)} parameter tensor(s) received no gradient: {dead}"


@pytest.mark.parametrize(
    "model_name, unreachable",
    [
        (VIT, ["mask_token"]),
        (CLIP, ["post_layernorm.weight"]),
        (SIGLIP, ["post_layernorm.weight", "head.probe"]),
        pytest.param(TIMM_VIT, ["timm_model.norm.weight"], marks=requires_timm),
    ],
    ids=["dinov2", "clip", "siglip", "timm-dinov3"],
)
def test_caseC_unreachable_encoder_parameters_are_excluded_but_kept(
    caseC_model, model_name, unreachable
):
    """The construction probe stops params the taps never reach from training.

    Final/pooler norms, pooling heads and unused tokens would never get a
    gradient (a DDP unused-parameter error); they stop requiring grad but stay in
    the state dict so pretrained and trained weights load unchanged. Every
    transformer block feeds a tap at the default taps, so none is excluded; and
    every parameter that still requires grad gets one through `Model`.
    """
    model = caseC_model(model_name, 2)
    bb = model.backbone
    named = dict(bb.enc.named_parameters())
    state = bb.enc.state_dict()
    for suffix in unreachable:
        name = next(n for n in named if n.endswith(suffix))
        assert not named[name].requires_grad, name
        assert name in state, name
    block = re.compile(r"(?:^|\.)(?:layers?|blocks?)\.\d+\.")
    assert all(p.requires_grad for n, p in named.items() if block.search(n))

    model.zero_grad(set_to_none=True)
    try:
        model(torch.rand(1, 1, 64, 96))["SingleInstanceConfmapsHead"].sum().backward()
        dead = [
            n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
    finally:
        model.zero_grad(set_to_none=True)
    assert not dead, f"{len(dead)} parameter tensor(s) received no gradient: {dead}"


# ------------------------------------------------------------ encoder LR groups


def _ids(params):
    return sorted(id(p) for p in params)


def _trainable(module):
    return [p for p in module.parameters() if p.requires_grad]


def test_encoder_param_groups_default_to_none(caseC_model):
    """No scaling asked for: the optimizer keeps its single group."""
    assert caseC_model(VIT, 2).backbone.encoder_param_groups(1e-3) is None


def test_encoder_lr_scale_makes_one_scaled_encoder_group():
    """`encoder_lr_scale` alone: one group, every trainable encoder param."""
    bb = _vit_backbone(encoder_lr_scale=0.1)
    groups = bb.encoder_param_groups(1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1e-4)
    assert _ids(groups[0]["params"]) == _ids(_trainable(bb.enc))


def test_layer_decay_lr_grows_with_depth():
    """Block i of L trains at scale * decay**(L + 1 - i); embeddings decay most."""
    base, scale, decay = 1e-3, 0.5, 0.75
    bb = _vit_backbone(encoder_lr_scale=scale, layer_decay=decay)
    groups = bb.encoder_param_groups(base)
    num_layers = bb.num_layers

    lrs = [g["lr"] for g in groups]
    assert all(a < b for a, b in zip(lrs, lrs[1:])), lrs
    # Every trainable encoder parameter, exactly once (excluded ones not at all).
    covered = [id(p) for g in groups for p in g["params"]]
    assert len(covered) == len(set(covered))
    assert sorted(covered) == _ids(_trainable(bb.enc))

    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    named = dict(bb.enc.named_parameters())
    patch = next(p for n, p in named.items() if "patch_embed" in n)
    assert lr_of[id(patch)] == min(lrs)
    assert lr_of[id(patch)] == pytest.approx(base * scale * decay ** (num_layers + 1))
    for i in (0, num_layers // 2, num_layers - 1):
        block = [p for n, p in named.items() if re.search(rf"layers?\.{i}\.", n)]
        assert block
        for p in block:  # block i (0-indexed) -> depth i + 1
            assert lr_of[id(p)] == pytest.approx(
                base * scale * decay ** (num_layers - i)
            )


def test_layer_decay_puts_clip_pre_norm_with_the_embeddings(caseC_model, monkeypatch):
    """CLIP's `pre_layrnorm` decays like the embeddings.

    Depth follows registration order, not names: `pre_layrnorm` runs before
    block 1. `post_layernorm` is excluded by the probe, so it is in no group.
    """
    bb = caseC_model(CLIP, 2).backbone
    monkeypatch.setattr(bb, "encoder_lr_scale", 0.5)
    monkeypatch.setattr(bb, "layer_decay", 0.75)
    groups = bb.encoder_param_groups(1.0)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    named = dict(bb.enc.named_parameters())

    pre = next(p for n, p in named.items() if "pre_layrnorm" in n)
    patch = next(p for n, p in named.items() if "patch_embedding" in n)
    assert lr_of[id(pre)] == lr_of[id(patch)] == min(lr_of.values())
    post = next(p for n, p in named.items() if "post_layernorm" in n)
    assert id(post) not in lr_of
    assert sorted(lr_of) == _ids(_trainable(bb.enc))


def test_layer_decay_on_a_hierarchical_backbone_falls_back_to_lr_scale():
    """`layer_decay` is for plain ViTs; a CNN warns and uses one scaled group."""
    from loguru import logger

    bb = PretrainedBackbone.from_config(
        _caseA_cfg(
            "facebook/convnextv2-nano-22k-224", encoder_lr_scale=0.5, layer_decay=0.75
        )
    )
    assert not bb.token_path
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        groups = bb.encoder_param_groups(1e-3)
    finally:
        logger.remove(sink)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(5e-4)
    assert _ids(groups[0]["params"]) == _ids(_trainable(bb.enc))
    assert any("layer_decay" in m for m in messages)


def test_frozen_encoder_has_no_encoder_param_groups():
    """A frozen encoder has nothing to group; the optimizer stays flat."""
    bb = _vit_backbone(encoder_lr_scale=0.1, layer_decay=0.75)
    bb.freeze_encoder()
    assert not bb.encoder_param_groups(1e-3)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"encoder_lr_scale": 0.1},
        {"encoder_lr_scale": 0.5, "layer_decay": 0.75},
        {"encoder_lr_scale": 0.1, "layer_decay": 0.75, "freeze": True},
    ],
    ids=["default", "lr-scale", "layer-decay", "frozen"],
)
def test_optimizer_covers_every_trainable_parameter_exactly_once(overrides):
    """Through the LightningModule: config -> backbone -> optimizer param groups.

    The non-encoder parameters (tap norms, neck, decoder, heads) always train at
    the base LR; the encoder trains at its scaled/decayed LRs, or not at all.
    """
    from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

    lr = 1e-3
    module = SingleInstanceLightningModule(
        model_type="single_instance",
        backbone_type="pretrained",
        backbone_config={
            "pretrained": {
                "model_name": VIT,
                "weights": False,
                "in_channels": 3,
                "output_stride": 2,
                "max_stride": 32,
                **overrides,
            }
        },
        head_configs=OmegaConf.create(
            {
                "single_instance": {
                    "confmaps": {
                        "part_names": ["a", "b"],
                        "sigma": 2.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                    }
                }
            }
        ),
        lr_scheduler=None,
        learning_rate=lr,
    )
    assert module.model.backbone.token_path
    optimizer = module.configure_optimizers()["optimizer"]
    groups = optimizer.param_groups
    in_optimizer = [id(p) for g in groups for p in g["params"]]
    assert len(in_optimizer) == len(set(in_optimizer)), "a parameter is in 2 groups"
    trainable = _trainable(module)
    assert sorted(in_optimizer) == _ids(trainable)

    enc = {id(p) for p in _trainable(module.model.backbone.enc)}
    rest = [id(p) for p in trainable if id(p) not in enc]
    if overrides.get("freeze") or "encoder_lr_scale" not in overrides:
        assert len(groups) == 1
        assert groups[0]["lr"] == lr
        return
    assert groups[0]["lr"] == lr
    assert sorted(id(p) for p in groups[0]["params"]) == sorted(rest)
    scale = overrides["encoder_lr_scale"]
    assert all(g["lr"] <= lr * scale * (1 + 1e-9) for g in groups[1:])
    assert sorted(id(p) for g in groups[1:] for p in g["params"]) == sorted(enc)


@pytest.mark.parametrize(
    "scheduler, floors",
    [
        (
            {
                "cosine_annealing_warmup": {
                    "warmup_epochs": 2,
                    "max_epochs": 10,
                    "warmup_start_lr": 1e-6,
                    "eta_min": 1e-6,
                }
            },
            ("warmup_start_lr", "eta_min"),
        ),
        (
            {
                "linear_warmup_linear_decay": {
                    "warmup_epochs": 2,
                    "max_epochs": 10,
                    "warmup_start_lr": 1e-6,
                    "end_lr": 1e-6,
                }
            },
            ("warmup_start_lr", "end_lr"),
        ),
        (
            {
                "reduce_lr_on_plateau": {
                    "threshold": 1e-4,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "patience": 1,
                    "factor": 0.5,
                    "min_lr": 1e-6,
                }
            },
            ("min_lrs",),
        ),
    ],
    ids=["cosine", "linear", "plateau"],
)
def test_schedule_floors_scale_with_the_encoder_lr_groups(scheduler, floors):
    """An absolute warmup start / floor is set against the base LR; each layer-
    decayed encoder group gets its own share of it, so the decay ratios hold for
    the whole run (one 1e-6 floor would lift a 1e-8 group 100x)."""
    from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

    lr = 1e-3
    module = SingleInstanceLightningModule(
        model_type="single_instance",
        backbone_type="pretrained",
        backbone_config={
            "pretrained": {
                "model_name": VIT,
                "weights": False,
                "in_channels": 3,
                "output_stride": 2,
                "max_stride": 32,
                "layer_decay": 0.5,
            }
        },
        head_configs=OmegaConf.create(
            {
                "single_instance": {
                    "confmaps": {
                        "part_names": ["a", "b"],
                        "sigma": 2.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                    }
                }
            }
        ),
        lr_scheduler=scheduler,
        learning_rate=lr,
    )
    out = module.configure_optimizers()
    optimizer, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
    base = [g.get("initial_lr", g["lr"]) for g in optimizer.param_groups]
    ratios = [b / lr for b in base]
    assert min(ratios) * lr < 1e-6  # the deepest groups sit below the 1e-6 floor
    if floors == ("min_lrs",):
        assert sched.min_lrs == pytest.approx([1e-6 * r for r in ratios])
        return
    for name in floors:
        assert getattr(sched, f"_{name}s") == pytest.approx([1e-6 * r for r in ratios])
    for _ in range(10):
        current = [g["lr"] for g in optimizer.param_groups]
        assert [c / current[0] for c in current] == pytest.approx(ratios)
        optimizer.step()
        sched.step()


@pytest.mark.parametrize(
    "min_lr, ok", [([1e-6], True), ([1e-6, 1e-7], False)], ids=["one", "two"]
)
def test_list_min_lr_with_encoder_lr_groups(min_lr, ok):
    """A one-entry `min_lr` list means one floor, scaled per group; a list that
    cannot match the (unknowable) group count is refused with a reason."""
    from sleap_nn.training.lightning_modules import SingleInstanceLightningModule

    module = SingleInstanceLightningModule(
        model_type="single_instance",
        backbone_type="pretrained",
        backbone_config={
            "pretrained": {
                "model_name": VIT,
                "weights": False,
                "in_channels": 3,
                "output_stride": 2,
                "max_stride": 32,
                "layer_decay": 0.5,
            }
        },
        head_configs=OmegaConf.create(
            {
                "single_instance": {
                    "confmaps": {
                        "part_names": ["a", "b"],
                        "sigma": 2.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                    }
                }
            }
        ),
        lr_scheduler={
            "reduce_lr_on_plateau": {
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "cooldown": 0,
                "patience": 1,
                "factor": 0.5,
                "min_lr": min_lr,
            }
        },
        learning_rate=1e-3,
    )
    if not ok:
        with pytest.raises(ValueError, match="param groups"):
            module.configure_optimizers()
        return
    out = module.configure_optimizers()
    groups = out["optimizer"].param_groups
    assert out["lr_scheduler"]["scheduler"].min_lrs == pytest.approx(
        [1e-6 * g["lr"] / 1e-3 for g in groups]
    )


def test_embedding_module_optimizer_uses_the_encoder_lr_groups():
    """The embedding module's own `configure_optimizers` honors the LR groups."""
    from sleap_nn.training.lightning_modules import EmbeddingLightningModule

    lr = 1e-3
    module = EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="pretrained",
        backbone_config={
            "pretrained": {
                "model_name": VIT,
                "weights": False,
                "in_channels": 3,
                "mode": "encoder",
                "encoder_lr_scale": 0.5,
            }
        },
        head_configs=OmegaConf.create(
            {
                "embedding": {
                    "embedding": {
                        "embedding_dim": 16,
                        "num_fc_layers": 1,
                        "num_fc_units": 32,
                        "pool": "avg",
                        "normalize": True,
                        "output_stride": 32,
                        "loss_weight": 1.0,
                        "freeze_backbone": False,
                        "objective": {
                            "positives": {"scope": "global_id", "aug_views": 2},
                            "negatives": {"sources": ["in_batch"]},
                            "loss": {"name": "supcon", "temperature": 0.1},
                            "use_projection": True,
                            "projection_dim": 16,
                        },
                    }
                }
            }
        ),
        lr_scheduler=None,
        learning_rate=lr,
    )
    bb = module.model.backbone
    groups = module.configure_optimizers()["optimizer"].param_groups
    in_optimizer = [id(p) for g in groups for p in g["params"]]
    assert len(in_optimizer) == len(set(in_optimizer))
    assert sorted(in_optimizer) == _ids(_trainable(module))
    assert len(groups) == 2
    assert groups[0]["lr"] == lr
    assert groups[1]["lr"] == pytest.approx(lr * 0.5)
    assert _ids(groups[1]["params"]) == _ids(_trainable(bb.enc))


# ----------------------------------------------------- weights=True (opt-in)


@requires_hf_weights
@pytest.mark.parametrize(
    "model_name", [VIT, CLIP, pytest.param(TIMM_VIT, marks=requires_timm)]
)
def test_caseC_weights_true_loads_the_vision_weights(model_name):
    """`weights=True` loads real (vision-tower) weights and snapshots them."""
    a = _vit_backbone(model_name, weights=True)
    b = _vit_backbone(model_name, weights=True)
    assert a._pretrained_sd
    assert not any("text" in k for k in a._pretrained_sd)
    live_a, live_b = a.enc.state_dict(), b.enc.state_dict()
    # Two independent builds agree exactly: loaded, not randomly initialized.
    assert all(torch.equal(v, live_b[k]) for k, v in live_a.items())
    assert all(torch.equal(v, live_a[k].cpu()) for k, v in a._pretrained_sd.items())

    # And they survive the LightningModule's xavier init via the reload.
    from sleap_nn.training.utils import xavier_init_weights

    a.apply(xavier_init_weights)
    a.reload_pretrained_weights()
    assert all(torch.equal(v, a.enc.state_dict()[k]) for k, v in live_b.items())


@pytest.mark.parametrize("output_stride", [2, 4, 8])
def test_caseC_detail_stem_joins_the_pyramid(output_stride):
    """The detail stem's stride-2/4/8 features become decoder skips, all trained."""
    model = _vit_model(VIT, output_stride=output_stride, detail_stem=True)
    bb = model.backbone
    expected = [st for st in (2, 4, 8, 16, 32) if st >= output_stride]
    assert bb._pyramid_strides == expected
    out = model(torch.rand(1, 1, 64, 96))["SingleInstanceConfmapsHead"]
    assert out.shape == (1, 2, 64 // output_stride, 96 // output_stride)
    out.sum().backward()
    assert all(p.grad is not None for p in bb.stem.parameters())
    dead = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not dead, dead


def test_caseC_detail_stem_is_dropped_when_coarser_than_its_levels():
    model = _vit_model(VIT, output_stride=16, detail_stem=True)
    assert not model.backbone.detail_stem
    assert not hasattr(model.backbone, "stem")


def test_resolve_mode_isotropic_override_beats_the_config():
    """timm wrapper configs say nothing about strides; the built model decides."""

    class _Cfg:
        model_type = "timm_wrapper"
        stage_names = None

    assert _resolve_mode(_Cfg(), "auto", spatial_heads=False, isotropic=False) == (
        "decoder"
    )
    assert _resolve_mode(_Cfg(), "auto", spatial_heads=False, isotropic=True) == (
        "encoder"
    )
    assert _resolve_mode(_Cfg(), "auto", spatial_heads=True, isotropic=True) == (
        "decoder"
    )
    assert _resolve_mode(_Cfg(), "encoder", spatial_heads=True, isotropic=False) == (
        "encoder"
    )


# ─────────────────────────────────────────────────────────────────────────
# Remaining Case C / timm paths: hierarchical timm models, misconfiguration
# errors, fixed-resolution models, and knobs a backbone ignores.
# ─────────────────────────────────────────────────────────────────────────

# A hierarchical `timm/...` checkpoint (stride 4-32), so it takes Case A.
TIMM_CNN = "timm/convnext_tiny.dinov3_lvd1689m"


@pytest.fixture
def loguru_caplog(caplog):
    """Route loguru records into pytest's caplog."""
    from loguru import logger

    handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
    yield caplog
    logger.remove(handler_id)


@requires_timm
def test_timm_hierarchical_checkpoint_takes_the_decoder_path():
    """A timm ConvNeXt is not a plain ViT: Case A on its stage outputs.

    timm's ConvNeXt rejects `dynamic_img_size`, so the wrapper must retry the
    build without it.
    """
    model = _vit_model(TIMM_CNN, output_stride=2)
    bb = model.backbone
    assert bb.is_timm and not bb.token_path and bb.mode == "decoder"
    assert bb._pyramid_strides == [4, 8, 16, 32]
    out = model(torch.rand(1, 1, 64, 96))["SingleInstanceConfmapsHead"]
    assert out.shape == (1, 2, 32, 48)


@requires_timm
def test_timm_hierarchical_checkpoint_honours_out_indices():
    bb = _vit_backbone(TIMM_CNN, out_indices=[1, 2, 3])
    assert bb._timm_stage_indices == [1, 2, 3]
    assert bb._pyramid_strides == [8, 16, 32]


@requires_timm
def test_timm_gradient_checkpointing_is_set_per_forward():
    """timm's switch is not gated on training (HF's is), and Lightning restores
    module modes by assigning `.training` (no `train()` call), so the backbone
    sets the switch from the encoder's mode and grad state on every forward."""
    from lightning.pytorch.utilities.model_helpers import _ModuleMode

    bb = _vit_backbone(TIMM_VIT, gradient_checkpointing=True)
    x = torch.rand(1, 3, 64, 96)

    def on():
        return bb.enc.timm_model.grad_checkpointing

    # `from_pretrained` hands back an eval-mode model; construction must not
    # leave that baked into the switch.
    bb.enc.eval()
    bb.enc.train()
    bb(x)
    assert on()

    # A validation pass the way Lightning runs it: capture, eval, restore.
    modes = _ModuleMode()
    modes.capture(bb)
    bb.eval()
    with torch.no_grad():
        bb(x)
    assert not on()
    modes.restore(bb)
    bb(x)
    assert on()

    with torch.no_grad():  # inference / export
        bb(x)
    assert not on()
    bb.freeze_encoder()  # a frozen encoder has nothing to recompute
    bb(x)
    assert not on()


@requires_timm
def test_timm_repeated_layer_taps_are_mapped_back():
    """timm returns one map per distinct block; a repeated tap reuses it."""
    bb = _vit_backbone(TIMM_VIT, layers=[6, 6, 9, 12]).eval()
    assert bb._timm_kwargs == {"stop_early": True}
    with torch.no_grad():
        maps = bb._tap_tokens(torch.rand(1, 3, 64, 96))
    assert len(maps) == 4
    # Fresh tap LayerNorms are identical, so the two layer-6 maps are too.
    assert torch.equal(maps[0], maps[1])
    with torch.no_grad():
        out = bb(torch.rand(1, 3, 64, 96))
    assert out["outputs"][-1].shape[-2:] == (32, 48)


def test_caseC_hf_encoder_stops_after_the_last_tap():
    """Blocks after the deepest tap are skipped, not just frozen, and stay in the
    state dict (checkpoints load unchanged)."""
    from sleap_nn.architectures.pretrained import _LeadingBlocks

    bb = _vit_backbone(VIT, layers=[3, 6, 9], output_stride=8).eval()
    (blocks,) = [m for m in bb.enc.modules() if isinstance(m, _LeadingBlocks)]
    assert (len(blocks), blocks.num_active) == (12, 9)
    assert any(".layer.11." in k for k in bb.enc.state_dict())

    ran = []
    hooks = [
        blk.register_forward_hook(lambda *_, i=i: ran.append(i))
        for i, blk in enumerate(blocks._modules.values())
    ]
    x = torch.rand(1, 3, 64, 96)
    try:
        with torch.no_grad():
            early = bb(x)["outputs"][-1]
        assert ran == list(range(9))
        blocks.num_active = None  # run every block
        with torch.no_grad():
            full = bb(x)["outputs"][-1]
    finally:
        for h in hooks:
            h.remove()
    torch.testing.assert_close(early, full)


def test_caseC_block_list_swap_is_kept_only_when_confirmed(monkeypatch):
    """The block list is found by length; a test forward must confirm that the
    encoder runs exactly its leading blocks, or the full list is put back."""
    from sleap_nn.architectures.pretrained import _LeadingBlocks

    bb = _vit_backbone(VIT, layers=[3, 6, 9], output_stride=8)
    # A same-length list the encoder never runs is not confirmed.
    stray = _LeadingBlocks([torch.nn.Identity() for _ in range(12)], num_active=9)
    assert not bb._runs_leading_blocks(stray, 9)

    monkeypatch.setattr(
        PretrainedBackbone, "_runs_leading_blocks", lambda self, blocks, last: False
    )
    bb = _vit_backbone(VIT, layers=[3, 6, 9], output_stride=8)
    assert not any(isinstance(m, _LeadingBlocks) for m in bb.enc.modules())
    with torch.no_grad():
        bb.eval()(torch.rand(1, 3, 64, 96))


def test_detail_channels_needs_one_count_per_stride():
    """A 4th stage would run without reaching the decoder (no gradient: DDP)."""
    with pytest.raises(ValueError, match="detail_channels must give 3"):
        _vit_backbone(VIT, detail_stem=True, detail_channels=[16, 32, 64, 128])
    # Unused (no detail stem): not validated, like any other ignored knob.
    _vit_backbone(VIT, detail_channels=[16, 32])


def test_token_path_warns_about_knobs_it_ignores(loguru_caplog):
    """CLIP pooled-only takes the token path in encoder mode: no neck, no stem,
    no stages -- say so instead of ignoring them."""
    bb = PretrainedBackbone.from_config(
        _caseA_cfg(
            CLIP, out_indices=[1], vit_neck="sfp", detail_stem=True, neck_channels=[8]
        ),
        spatial_heads=False,
    )
    assert bb.token_path and bb.mode == "encoder"
    text = loguru_caplog.text
    assert "do not apply to the plain-ViT encoder-only path" in text
    for knob in ("out_indices", "vit_neck", "detail_stem", "neck_channels"):
        assert knob in text


def test_vit_knobs_on_a_hierarchical_backbone_warn(loguru_caplog):
    """`token_stride` etc. do nothing on Case A, so say so instead of ignoring them."""
    _vit_backbone("facebook/convnextv2-nano-22k-224", token_stride=8, vit_neck="sfp")
    assert "only apply to plain-ViT backbones" in loguru_caplog.text
    assert "token_stride" in loguru_caplog.text and "vit_neck" in loguru_caplog.text


@pytest.mark.parametrize("output_stride", [32, 64])
def test_output_stride_coarser_than_the_token_stride_raises(output_stride):
    """One pyramid level (output_stride 32) is as unusable as none; the message
    names the fix instead of Case A's "use a hierarchical backbone"."""
    with pytest.raises(ValueError, match="Use output_stride <= 16"):
        _vit_backbone(VIT, output_stride=output_stride, max_stride=64)


def test_patch_size_reading():
    from sleap_nn.architectures.pretrained import _patch_size

    class _Cfg:
        def __init__(self, patch_size):
            self.patch_size = patch_size

    assert _patch_size(_Cfg(14)) == 14
    assert _patch_size(_Cfg([16, 16])) == 16
    with pytest.raises(ValueError, match="Non-square patch size"):
        _patch_size(_Cfg((16, 8)))
    with pytest.raises(ValueError, match="does not look like a ViT"):
        _patch_size(_Cfg(None))


def test_fixed_resolution_vit_fails_at_construction():
    """SAM's ViT only runs at 1024 px: refused when the model is built, with a
    message that says why, not at the first training step."""
    with pytest.raises(ValueError, match="Could not extract a dense token grid"):
        _vit_backbone("facebook/sam-vit-base")


def test_neck_loads_checkpoints_from_the_wrapper_layernorm():
    """`LayerNorm2d` used to wrap an inner `nn.LayerNorm` (`.norm.weight`); models
    saved then (the benchmark runs) still load into the torchvision-based class."""
    from sleap_nn.architectures.vit_neck import LayerNorm2d, ViTPyramidNeck

    torch.manual_seed(0)
    neck = ViTPyramidNeck(in_channels=16, out_channels=[4, 8, 16, 16])
    norms = {n for n, m in neck.named_modules() if isinstance(m, LayerNorm2d)}
    assert norms
    old_style = {}
    for key, value in neck.state_dict().items():
        module, _, param = key.rpartition(".")
        if module in norms:
            key = f"{module}.norm.{param}"
        old_style[key] = value.clone() + 1  # distinguishable from the init

    fresh = ViTPyramidNeck(in_channels=16, out_channels=[4, 8, 16, 16])
    fresh.load_state_dict(old_style, strict=True)
    some_norm = sorted(norms)[0]
    torch.testing.assert_close(
        fresh.get_submodule(some_norm).weight, old_style[f"{some_norm}.norm.weight"]
    )


# ─────────────────────────────────────────────────────────────────────────
# Case C with a ViTPose-style direct decoder (`vit_decoder: classic|simple`):
# last token map -> (pretrained final norm) -> deconvs / bilinear -> head.
# ─────────────────────────────────────────────────────────────────────────


def _dead_params(model):
    return [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]


@pytest.mark.parametrize("vit_decoder", ["classic", "simple"])
@pytest.mark.parametrize(
    "model_name, final_norm",
    [
        (VIT, "layernorm"),
        (CLIP, "post_layernorm"),
        pytest.param(TIMM_VIT, "timm_model.norm", marks=requires_timm),
    ],
    ids=["dinov2", "clip", "timm-dinov3"],
)
def test_direct_decoder_centered_instance(model_name, final_norm, vit_decoder):
    """No neck, no UNet decoder: the last tap goes through the encoder's final norm
    into the ViTPose decoder, and every trainable parameter gets a gradient (so DDP
    needs no `find_unused_parameters`)."""
    heads = OmegaConf.create(
        {"confmaps": {"part_names": ["a", "b"], "sigma": 2.5, "output_stride": 4}}
    )
    model = _vit_model(
        model_name,
        output_stride=4,
        model_type="centered_instance",
        heads=heads,
        mode="decoder",
        vit_decoder=vit_decoder,
    )
    bb = model.backbone
    assert bb.direct_decoder and bb.vit_final_norm
    assert not hasattr(bb, "neck") and not hasattr(bb, "dec")
    assert bb.layers == [bb.num_layers]
    assert bb._final_norm_name == final_norm
    assert isinstance(bb.tap_norms[0], torch.nn.Identity)
    # The final norm feeds the decoder now, so the probe keeps it trainable.
    norm = bb.enc.get_submodule(final_norm)
    assert all(p.requires_grad for p in norm.parameters())
    assert bb.max_stride == bb.token_stride == 16
    channels = 256 if vit_decoder == "classic" else bb.hidden_size
    assert bb.decoder_stride_to_filters == {4: channels}

    # ViTPose's heatmap conv: 1x1 after the deconvs, 3x3 after `simple`.
    kernel = 1 if vit_decoder == "classic" else 3
    head_conv = model.head_layers[0][0][0]
    assert head_conv.weight.shape == (2, channels, kernel, kernel)

    out = model(torch.rand(2, 1, 64, 96))["CenteredInstanceConfmapsHead"]
    assert out.shape == (2, 2, 16, 24)
    out.sum().backward()
    dead = _dead_params(model)
    assert not dead, f"{len(dead)} parameter tensor(s) received no gradient: {dead}"


def test_direct_decoder_uses_the_pretrained_final_norm():
    """With the HF DINOv2 final norm, the decoder input is `last_hidden_state`."""
    bb = _vit_backbone(VIT, output_stride=4, vit_decoder="simple").eval()
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        tokens = bb(x)["middle_output"]
        b, c, h, w = tokens.shape
        # Patch 14 runs at token stride 16: the input is resampled to the grid.
        resized = torch.nn.functional.interpolate(
            bb._normalize(x), size=(h * 14, w * 14), mode="bilinear"
        )
        hf = bb.enc(pixel_values=resized).last_hidden_state
    ref = hf[:, -h * w :].transpose(1, 2).reshape(b, c, h, w)
    torch.testing.assert_close(tokens, ref)


def test_direct_decoder_fresh_norm_when_final_norm_off():
    """`vit_final_norm: false` keeps today's fresh tap LayerNorm; the unreachable
    final norm is excluded from training again."""
    bb = _vit_backbone(
        VIT, output_stride=4, vit_decoder="classic", vit_final_norm=False
    )
    assert isinstance(bb.tap_norms[0], torch.nn.LayerNorm)
    assert not any(p.requires_grad for p in bb.enc.layernorm.parameters())


def test_unet_decoder_can_opt_into_the_final_norm():
    """`vit_final_norm` also applies to the neck path (sfp from the last layer)."""
    model = _vit_model(VIT, output_stride=4, vit_neck="sfp", layers=[-1])
    assert isinstance(model.backbone.tap_norms[0], torch.nn.LayerNorm)
    model = _vit_model(
        VIT, output_stride=4, vit_neck="sfp", layers=[-1], vit_final_norm=True
    )
    bb = model.backbone
    assert isinstance(bb.tap_norms[0], torch.nn.Identity)
    assert hasattr(bb, "neck")
    model(torch.rand(1, 1, 64, 96))["SingleInstanceConfmapsHead"].sum().backward()
    assert not _dead_params(model)


@pytest.mark.parametrize(
    "model_type, output_stride, vit_decoder",
    [
        ("single_instance", 4, "classic"),
        ("single_instance", 2, "classic"),
        ("centroid", 2, "simple"),
        ("centroid", 8, "classic"),
        ("single_instance", 16, "simple"),
    ],
)
def test_direct_decoder_other_heads_and_strides(model_type, output_stride, vit_decoder):
    heads = OmegaConf.create(
        {
            "confmaps": {
                "part_names": ["a", "b"],
                "anchor_part": None,
                "sigma": 2.5,
                "output_stride": output_stride,
            }
        }
    )
    if model_type == "centroid":
        heads.confmaps.pop("part_names")
    else:
        heads.confmaps.pop("anchor_part")
    model = _vit_model(
        VIT,
        output_stride=output_stride,
        model_type=model_type,
        heads=heads,
        vit_decoder=vit_decoder,
    )
    if vit_decoder == "classic":
        deconvs = [
            m
            for m in model.backbone.vit_dec.modules()
            if isinstance(m, torch.nn.ConvTranspose2d)
        ]
        assert len(deconvs) == int(np.log2(16 // output_stride))
    out = next(iter(model(torch.rand(1, 1, 64, 96)).values()))
    assert out.shape[-2:] == (64 // output_stride, 96 // output_stride)
    out.sum().backward()
    assert not _dead_params(model)


def test_direct_decoder_param_counts():
    from sleap_nn.architectures.vit_neck import ViTPoseDecoder

    classic = ViTPoseDecoder(768, "classic", upsample=4, channels=256)
    expected = 768 * 256 * 16 + 256 * 256 * 16 + 2 * (2 * 256)  # deconvs + BN
    assert sum(p.numel() for p in classic.parameters()) == expected
    assert classic.head_kernel_size == 1
    simple = ViTPoseDecoder(768, "simple", upsample=4)
    assert sum(p.numel() for p in simple.parameters()) == 0
    assert simple.head_kernel_size == 3


def test_simple_decoder_is_relu_then_upsample():
    """ViTPose's order (ReLU on the token map, then bilinear), which differs from
    upsampling first: bilinear mixes negative and positive neighbors."""
    from sleap_nn.architectures.vit_neck import ViTPoseDecoder

    x = torch.randn(1, 8, 4, 3)
    out = ViTPoseDecoder(8, "simple", upsample=4)(x)
    up = lambda t: torch.nn.functional.interpolate(
        t, scale_factor=4, mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(out, up(torch.relu(x)))
    assert not torch.allclose(out, torch.relu(up(x)))
    assert out.shape == (1, 8, 16, 12) and (out >= 0).all()


def test_simple_decoder_3x3_conv_on_every_spatial_head():
    """A bottom-up model's confmap and PAF heads both get the 3x3 conv; the input
    is padded so the output keeps the feature map's size."""
    heads = OmegaConf.create(
        {
            "confmaps": {
                "part_names": ["a", "b"],
                "sigma": 2.5,
                "output_stride": 4,
                "loss_weight": 1.0,
            },
            "pafs": {
                "edges": [["a", "b"]],
                "sigma": 15.0,
                "output_stride": 4,
                "loss_weight": 1.0,
            },
        }
    )
    model = _vit_model(
        VIT, output_stride=4, model_type="bottomup", heads=heads, vit_decoder="simple"
    )
    convs = [layer[0][0] for layer in model.head_layers]
    assert [c.kernel_size for c in convs] == [(3, 3), (3, 3)]
    outs = model(torch.rand(1, 1, 64, 96))
    assert [o.shape[-2:] for o in outs.values()] == [(16, 24), (16, 24)]


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"vit_decoder": "fpn"}, "Unknown vit_decoder"),
        ({"vit_decoder": "classic", "layers": [6, 12]}, "exactly one tapped layer"),
        ({"vit_decoder": "classic", "output_stride": 32}, "coarser than the token"),
        ({"vit_decoder": "classic", "output_stride": 16}, "upsampling factor >= 2"),
        ({"vit_decoder": "simple", "output_stride": 4, "max_stride": 24}, "multiple"),
    ],
    ids=["unknown", "two-taps", "os-too-coarse", "classic-no-upsample", "max-stride"],
)
def test_direct_decoder_misconfiguration_raises(kw, match):
    kw = {"output_stride": 4, **kw}
    with pytest.raises(ValueError, match=match):
        _vit_backbone(VIT, **kw)


def test_direct_decoder_warns_about_neck_knobs(loguru_caplog):
    bb = _vit_backbone(
        VIT,
        output_stride=4,
        vit_decoder="classic",
        vit_neck="sfp",
        neck_channels=[8, 8, 8, 8],
        detail_stem=True,
    )
    assert bb.direct_decoder
    for knob in ("vit_neck", "neck_channels", "detail_stem"):
        assert knob in loguru_caplog.text


def test_vit_decoder_ignored_by_hierarchical_backbones(loguru_caplog):
    bb = PretrainedBackbone.from_config(
        _caseA_cfg("facebook/convnextv2-nano-22k-224", vit_decoder="classic")
    )
    assert not bb.direct_decoder
    assert "vit_decoder" in loguru_caplog.text


def test_direct_decoder_from_training_config():
    """The new keys round-trip through `PretrainedConfig` into the backbone."""
    from sleap_nn.config.model_config import PretrainedConfig

    cfg = OmegaConf.structured(
        PretrainedConfig(
            model_name=VIT,
            weights=False,
            output_stride=4,
            vit_decoder="classic",
            vit_decoder_channels=64,
        )
    )
    assert cfg.vit_final_norm is None
    bb = PretrainedBackbone.from_config(cfg, spatial_heads=True)
    assert bb.direct_decoder and bb.vit_final_norm
    assert bb.decoder_stride_to_filters == {4: 64}


def test_direct_decoder_layer_decay_groups():
    """With `layer_decay`, the now-trainable final norm trains at the post-block
    rate (`encoder_lr_scale` x base LR) and the ViTPose decoder at the base LR."""
    bb = _vit_backbone(
        VIT,
        output_stride=4,
        vit_decoder="classic",
        encoder_lr_scale=0.5,
        layer_decay=0.75,
    )
    groups = bb.encoder_param_groups(base_lr=1.0)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    for p in bb.enc.layernorm.parameters():
        assert lr_of[id(p)] == pytest.approx(0.5)
    assert not any(id(p) in lr_of for p in bb.vit_dec.parameters())
    trainable = {id(p) for p in bb.enc.parameters() if p.requires_grad}
    assert set(lr_of) == trainable


def test_final_norm_falls_back_to_a_fresh_layernorm(loguru_caplog, monkeypatch):
    """No recognizable final norm: warn and use a fresh tap LayerNorm."""
    import sleap_nn.architectures.pretrained as pretrained_mod

    monkeypatch.setattr(pretrained_mod, "_FINAL_NORM_NAMES", ())
    bb = _vit_backbone(VIT, output_stride=4, vit_decoder="classic")
    assert bb._final_norm_name is None
    assert isinstance(bb.tap_norms[0], torch.nn.LayerNorm)
    assert "no final norm found" in loguru_caplog.text


def test_final_norm_needs_a_last_layer_tap(loguru_caplog):
    """`vit_final_norm` on a tap that is not the last layer: warn, fresh norm."""
    bb = _vit_backbone(
        VIT, output_stride=4, vit_neck="sfp", layers=[6], vit_final_norm=True
    )
    assert bb._final_norm_name == "layernorm"
    assert isinstance(bb.tap_norms[0], torch.nn.LayerNorm)
    assert "applies only to a tap of the last layer" in loguru_caplog.text


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"kind": "fpn"}, "Unknown ViT decoder"),
        ({"kind": "simple", "upsample": 3}, "power of two"),
        ({"kind": "classic", "upsample": 0}, "power of two"),
    ],
)
def test_vitpose_decoder_rejects_bad_arguments(kw, match):
    from sleap_nn.architectures.vit_neck import ViTPoseDecoder

    with pytest.raises(ValueError, match=match):
        ViTPoseDecoder(8, **kw)
