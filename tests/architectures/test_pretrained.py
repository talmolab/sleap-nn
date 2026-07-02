"""Tests for external pretrained (HuggingFace) backbones.

These tests build backbones with ``weights=False`` (random init from the model
config) so they never touch the network — the HF architecture code is exercised
without downloading checkpoints. They are skipped when ``transformers`` (the
``backbones`` optional extra) is not installed, mirroring the ``requires_onnxruntime``
convention in ``tests/export``.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

transformers = pytest.importorskip("transformers")

from sleap_nn.architectures.model import Model
from sleap_nn.architectures.pretrained import PretrainedBackbone, _resolve_mode


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
