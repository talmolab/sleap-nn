"""Encoder-only autocast at prediction (``precision="bf16-encoder"``).

``TorchBackend`` applies autocast on CUDA only; these tests drive the same
``autocast_forward`` on the CPU, where bf16 autocast also works.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from click.testing import CliRunner
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_head_configs
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.backends.torch_backend import (
    autocast_forward,
    encoder_modules,
)


@pytest.fixture
def tiny_detr():
    """A small random ``Model(bbox_detr)`` on a convolutional test backbone."""
    from sleap_nn.architectures.detr import DETR_BACKBONES, TinyDetrBackbone
    from sleap_nn.architectures.model import Model

    DETR_BACKBONES["unet"] = lambda cfg: TinyDetrBackbone(
        in_channels=cfg.in_channels, out_channels=16, stride=16
    )
    try:
        hc = OmegaConf.structured(get_head_configs("bbox_detr"))
        hc.bbox_detr.detr.update(
            hidden_dim=32,
            num_queries=10,
            group_detr=1,
            dec_layers=2,
            sa_nheads=2,
            ca_nheads=4,
            dim_feedforward=64,
            num_select=10,
        )
        torch.manual_seed(0)
        model = Model(
            backbone_type="unet",
            backbone_config=OmegaConf.create({"in_channels": 1, "output_stride": 16}),
            head_configs=hc.bbox_detr,
            model_type="bbox_detr",
        ).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.1)
        yield model
    finally:
        del DETR_BACKBONES["unet"]


def _max_err(out, ref):
    return max(
        (out[k].float() - ref[k]).abs().max().item()
        for k in ("pred_logits", "pred_boxes")
    )


def test_encoder_modules_of_a_detr_detector_is_its_backbone(tiny_detr):
    assert encoder_modules(tiny_detr) == [tiny_detr.backbone]
    # Through a wrapper (a Lightning module holds the model as `.model`).
    wrapper = torch.nn.Module()
    wrapper.model = tiny_detr
    assert encoder_modules(wrapper) == [tiny_detr.backbone]
    assert encoder_modules(torch.nn.Conv2d(1, 1, 1)) == []


def test_encoder_modules_of_a_unet_model():
    from sleap_nn.architectures.model import Model
    from sleap_nn.config.get_config import get_backbone_config

    bb = OmegaConf.structured(get_backbone_config("unet"))
    bb.unet.update(filters=8, max_stride=8, output_stride=2)
    hc = OmegaConf.structured(get_head_configs("single_instance"))
    hc.single_instance.confmaps.update(part_names=["a", "b"], output_stride=2)
    model = Model(
        backbone_type="unet",
        backbone_config=bb.unet,
        head_configs=hc.single_instance,
        model_type="single_instance",
    )
    found = encoder_modules(model)
    assert found and all(any(m is e for e in model.backbone.encoders) for m in found)


@torch.inference_mode()
def test_encoder_scope_keeps_the_detr_head_in_fp32(tiny_detr):
    """Head outputs are fp32 and closer to the fp32 forward than whole-forward bf16."""
    torch.manual_seed(1)
    x = torch.rand(2, 1, 64, 64)
    ref = tiny_detr(x)
    whole = autocast_forward(tiny_detr, x, torch.bfloat16, "all", device_type="cpu")
    enc = autocast_forward(tiny_detr, x, torch.bfloat16, "encoder", device_type="cpu")

    # Whole-forward autocast runs the head's linear layers in bf16.
    assert whole["pred_logits"].dtype == torch.bfloat16
    for key in ("pred_logits", "pred_boxes"):
        assert enc[key].dtype == torch.float32
    assert enc["enc_outputs"]["pred_logits"].dtype == torch.float32
    # The encoder did run in bf16: the outputs are not the fp32 ones...
    assert _max_err(enc, ref) > 0
    # ...but they are closer to them than with everything in bf16.
    assert _max_err(enc, ref) < _max_err(whole, ref)
    # No hooks are left behind, and autocast is off again.
    assert not tiny_detr.backbone._forward_hooks
    assert not tiny_detr.backbone._forward_pre_hooks
    assert not torch.is_autocast_enabled("cpu")


def test_encoder_scope_needs_an_encoder():
    with pytest.raises(ValueError, match="encoder"):
        autocast_forward(
            torch.nn.Conv2d(1, 1, 1),
            torch.zeros(1, 1, 4, 4),
            torch.bfloat16,
            "encoder",
            device_type="cpu",
        )


def test_hooks_removed_and_autocast_left_after_an_error(tiny_detr):
    """An exception inside the encoder still leaves autocast and removes the hooks."""
    with pytest.raises(RuntimeError):
        # 2 channels into a 1-channel patch embedding.
        autocast_forward(
            tiny_detr,
            torch.zeros(1, 2, 64, 64),
            torch.bfloat16,
            "encoder",
            device_type="cpu",
        )
    assert not torch.is_autocast_enabled("cpu")
    assert not tiny_detr.backbone._forward_hooks
    assert not tiny_detr.backbone._forward_pre_hooks


def test_set_autocast_precision_sets_the_scope(tiny_detr):
    from sleap_nn.inference.predictor import (
        _autocast_precision_used,
        _set_autocast_precision,
    )

    backend = TorchBackend(model=tiny_detr, device="cpu")
    assert backend.autocast_scope == "all"  # default unchanged
    _set_autocast_precision(backend, "bf16")
    assert (backend.autocast_dtype, backend.autocast_scope) == (torch.bfloat16, "all")

    backend = TorchBackend(model=tiny_detr, device="cpu")
    _set_autocast_precision(backend, "fp16-encoder")
    assert (backend.autocast_dtype, backend.autocast_scope) == (
        torch.float16,
        "encoder",
    )
    # Provenance names the scope (CUDA only: here the device says cpu).
    backend.device = "cuda:0"
    assert _autocast_precision_used(backend) == "fp16-encoder"

    with pytest.raises(ValueError, match="no\\s+encoder"):
        _set_autocast_precision(
            TorchBackend(model=torch.nn.Conv2d(1, 1, 1), device="cpu"), "bf16-encoder"
        )
    with pytest.raises(ValueError, match="precision must be"):
        _set_autocast_precision(backend, "bf16-decoder")


def test_backend_on_cpu_is_unchanged_by_the_scope(tiny_detr):
    """Off CUDA no autocast applies, encoder scope or not (as for ``bf16``)."""
    x = torch.rand(1, 1, 64, 64)
    plain = TorchBackend(model=tiny_detr, device="cpu")(x)
    scoped = TorchBackend(
        model=tiny_detr,
        device="cpu",
        autocast_dtype=torch.bfloat16,
        autocast_scope="encoder",
    )(x)
    assert torch.equal(plain["pred_logits"], scoped["pred_logits"])


def test_precision_flag_accepts_encoder_values():
    from sleap_nn.cli import cli

    for args, expected in (
        ([], "32"),
        (["--precision", "bf16-encoder"], "bf16-encoder"),
        (["--precision", "FP16-Encoder"], "fp16-encoder"),
    ):
        with patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict:
            result = CliRunner().invoke(
                cli,
                [
                    "predict",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                ]
                + args,
            )
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.kwargs["precision"] == expected


def test_encoder_precision_refused_for_exported_models(tmp_path):
    from sleap_nn.cli import cli

    d = tmp_path / "exp"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"")
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = CliRunner().invoke(
            cli,
            ["predict", "-i", "/fake/path.mp4", "-m", str(d)]
            + ["--precision", "bf16-encoder"],
        )
    assert result.exit_code != 0
    assert "trained checkpoints only" in " ".join(result.output.split())
    assert not mock_predict.called
