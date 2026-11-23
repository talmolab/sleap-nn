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
from sleap_nn.architectures.common import encoder_autocast
from sleap_nn.inference.layers.backends.torch_backend import (
    autocast_forward,
    supports_encoder_autocast,
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


def _first_layer(module):
    return next(
        m for m in module.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    )


def _probe(model, x, encoder, after):
    """Output dtype of the first layer inside ``encoder`` and of ``after``'s first
    layer, under the encoder scope (CPU bf16), and the encoder calls counted."""
    seen = {}
    hooks = [
        _first_layer(encoder).register_forward_hook(
            lambda m, a, o: seen.setdefault("encoder", o.dtype) and None
        ),
        _first_layer(after).register_forward_hook(
            lambda m, a, o: seen.setdefault("after", (a[0].dtype, o.dtype)) and None
        ),
    ]
    try:
        with torch.inference_mode(), encoder_autocast("cpu", torch.bfloat16) as st:
            out = model(x)
    finally:
        for h in hooks:
            h.remove()
    return seen, st["calls"], out


def _check(seen, calls):
    assert calls >= 1
    assert seen["encoder"] == torch.bfloat16  # the encoder ran under autocast
    assert seen["after"] == (torch.float32, torch.float32)  # the rest in fp32


def test_encoder_scope_unet_with_stem():
    from sleap_nn.architectures.unet import UNet

    cfg = OmegaConf.create(
        dict(
            in_channels=1,
            kernel_size=3,
            filters=8,
            filters_rate=2,
            max_stride=16,
            convs_per_block=2,
            stacks=1,
            stem_stride=2,
            middle_block=True,
            up_interpolate=True,
            block_contraction=False,
            output_stride=2,
        )
    )
    unet = UNet.from_config(cfg).eval()
    seen, calls, _ = _probe(unet, torch.rand(1, 1, 64, 64), unet.stem, unet.decoders[0])
    _check(seen, calls)
    assert calls == 2  # the stem and the encoder blocks


@pytest.mark.parametrize("kind", ["convnext", "swint"])
def test_encoder_scope_convnext_and_swin(kind):
    from sleap_nn.architectures.convnext import ConvNextWrapper
    from sleap_nn.architectures.swint import SwinTWrapper

    cfg = dict(
        in_channels=1,
        model_type="tiny",
        arch=None,
        kernel_size=3,
        filters_rate=2,
        convs_per_block=2,
        up_interpolate=True,
        stem_patch_stride=2,
        output_stride=2,
        max_stride=32,
    )
    if kind == "convnext":
        cfg.update(stem_patch_kernel=4)
        net = ConvNextWrapper.from_config(OmegaConf.create(cfg))
    else:
        cfg.update(patch_size=4, window_size=7)
        net = SwinTWrapper.from_config(OmegaConf.create(cfg))
    net.eval()
    seen, calls, _ = _probe(net, torch.rand(1, 1, 64, 64), net.enc, net.dec)
    _check(seen, calls)


def _pretrained_model(name):
    from tests.architectures.test_pretrained import _vit_model

    return _vit_model(name, output_stride=2).eval()


@pytest.mark.parametrize("family", ["timm_cnn", "timm_vit", "hf_vit"])
def test_encoder_scope_pretrained_backbones(family):
    """timm calls ``forward_intermediates`` (no module call on ``enc``): the
    backbone's own entry point still runs the encoder under autocast."""
    from tests.architectures import test_pretrained as tp

    name = {"timm_cnn": tp.TIMM_CNN, "timm_vit": tp.TIMM_VIT, "hf_vit": tp.VIT}[family]
    if family != "hf_vit":
        pytest.importorskip("timm")
    model = _pretrained_model(name)
    x = torch.rand(1, 1, 64, 64)
    seen, calls, _ = _probe(model, x, model.backbone.enc, model.head_layers[0])
    _check(seen, calls)
    with torch.inference_mode():
        ref = model(x)
        enc = autocast_forward(model, x, torch.bfloat16, "encoder", device_type="cpu")
    key = next(iter(ref))
    assert enc[key].dtype == torch.float32
    assert (enc[key] - ref[key]).abs().max() > 0  # not silently fp32


def test_encoder_scope_detr_backbones(tiny_detr):
    from sleap_nn.architectures.detr import DETR_BACKBONES, TinyDetrBackbone

    assert isinstance(tiny_detr.backbone, TinyDetrBackbone)
    assert "unet" in DETR_BACKBONES
    seen, calls, out = _probe(
        tiny_detr, torch.rand(1, 1, 64, 64), tiny_detr.backbone, tiny_detr.detr
    )
    _check(seen, calls)
    assert out["pred_logits"].dtype == torch.float32


def test_supports_encoder_autocast(tiny_detr):
    assert supports_encoder_autocast(tiny_detr)
    wrapper = torch.nn.Module()  # a Lightning module holds the model as `.model`
    wrapper.model = tiny_detr
    assert supports_encoder_autocast(wrapper)
    assert not supports_encoder_autocast(torch.nn.Conv2d(1, 1, 1))


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
    with pytest.raises(ValueError, match="Encoder-only"):
        autocast_forward(
            torch.nn.Conv2d(1, 1, 1),
            torch.zeros(1, 1, 4, 4),
            torch.bfloat16,
            "encoder",
            device_type="cpu",
        )


def test_request_cleared_after_an_error(tiny_detr):
    """An exception inside the model still clears the request and leaves autocast."""
    from sleap_nn.architectures import common

    with pytest.raises(RuntimeError):
        # 2 channels into a 1-channel patch embedding.
        autocast_forward(
            tiny_detr,
            torch.zeros(1, 2, 64, 64),
            torch.bfloat16,
            "encoder",
            device_type="cpu",
        )
    assert common._ENCODER_AUTOCAST is None
    assert not torch.is_autocast_enabled("cpu")


def test_scope_reported_only_if_the_encoder_ran():
    """A model whose backbone claims support but never runs its encoder raises."""

    class Silent(torch.nn.Module):
        def forward(self, x):
            return x

    from sleap_nn.inference.layers.backends import torch_backend

    m = Silent()
    orig = torch_backend.supports_encoder_autocast
    torch_backend.supports_encoder_autocast = lambda model: True
    try:
        with pytest.raises(RuntimeError, match="no encoder ran"):
            autocast_forward(m, torch.zeros(1), torch.bfloat16, "encoder", "cpu")
    finally:
        torch_backend.supports_encoder_autocast = orig


def test_encoder_scope_refused_with_compile(tiny_detr):
    with pytest.raises(ValueError, match="use_compile"):
        TorchBackend(
            model=tiny_detr, device="cpu", use_compile=True, autocast_scope="encoder"
        )


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

    with pytest.raises(ValueError, match="supports it"):
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


def test_exported_backends_keep_the_whole_forward_scope(tiny_detr, caplog):
    """Mixed checkpoint + exported models: the exported one keeps scope "all" (it
    runs as for plain bf16), the checkpoint one gets the encoder scope; only
    exported models warn as for bf16 instead of failing."""
    from loguru import logger

    from sleap_nn.inference.exported_network import ExportedNetwork
    from sleap_nn.inference.predictor import (
        _set_autocast_precision,
        _warn_precision_scope,
    )

    class FakeExported(ExportedNetwork):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, x):
            return x * 2

    exported = TorchBackend(model=FakeExported(), device="cpu")
    checkpoint = TorchBackend(model=tiny_detr, device="cpu")
    found = _set_autocast_precision([exported, checkpoint], "bf16-encoder")
    assert len(found) == 2
    assert (exported.autocast_dtype, exported.autocast_scope) == (
        torch.bfloat16,
        "all",
    )
    assert checkpoint.autocast_scope == "encoder"
    # The exported backend's forward runs as for plain bf16.
    autocast_forward(
        exported.model,
        torch.zeros(1, 1, 4, 4),
        exported.autocast_dtype,
        scope=exported.autocast_scope,
        device_type="cpu",
    )

    only = TorchBackend(model=FakeExported(), device="cpu")
    _set_autocast_precision(only, "bf16-encoder")
    assert only.autocast_scope == "all"
    messages = []
    handle = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        _warn_precision_scope("bf16-encoder", "cuda", [only])
    finally:
        logger.remove(handle)
    assert any("exported" in m for m in messages)


def _unet():
    from sleap_nn.architectures.unet import UNet

    cfg = OmegaConf.create(
        dict(
            in_channels=1,
            kernel_size=3,
            filters=8,
            filters_rate=2,
            max_stride=16,
            convs_per_block=2,
            stacks=1,
            stem_stride=None,
            middle_block=True,
            up_interpolate=True,
            block_contraction=False,
            output_stride=2,
        )
    )
    return UNet.from_config(cfg).eval()


def _encoder_dtype(net, x):
    seen = []
    h = _first_layer(net.encoders[0]).register_forward_hook(
        lambda m, a, o: seen.append(o.dtype) and None
    )
    try:
        with torch.inference_mode():
            net(x)
    finally:
        h.remove()
    return seen[0]


def test_requests_belong_to_their_thread():
    """A enters, B tries to enter (refused while A holds a request) and runs its
    own forward in fp32, A exits, B enters and exits: nothing is left set, and a
    later plain forward stays fp32."""
    import threading

    from sleap_nn.architectures import common

    net, x = _unet(), torch.rand(1, 1, 32, 32)
    a_in, b_done, a_out = threading.Event(), threading.Event(), threading.Event()
    result = {}

    def thread_a():
        with encoder_autocast("cpu", torch.bfloat16) as st:
            result["a_dtype"] = _encoder_dtype(net, x)
            a_in.set()
            b_done.wait(30)
        result["a_calls"] = st["calls"]
        a_out.set()

    def thread_b():
        a_in.wait(30)
        try:
            with encoder_autocast("cpu", torch.bfloat16):
                pass
        except RuntimeError as e:
            result["b_refused"] = "another thread" in str(e)
        result["b_dtype_during_a"] = _encoder_dtype(net, x)  # A's request: not B's
        b_done.set()
        a_out.wait(30)
        with encoder_autocast("cpu", torch.bfloat16) as st:
            result["b_dtype"] = _encoder_dtype(net, x)
        result["b_calls"] = st["calls"]

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(60)
    tb.join(60)
    assert result["a_dtype"] == torch.bfloat16 and result["a_calls"] == 1
    assert result["b_refused"] is True
    assert result["b_dtype_during_a"] == torch.float32
    assert result["b_dtype"] == torch.bfloat16 and result["b_calls"] == 1
    assert common._ENCODER_AUTOCAST is None
    assert _encoder_dtype(net, x) == torch.float32


def test_nested_requests_in_one_thread_restore():
    from sleap_nn.architectures import common

    with encoder_autocast("cpu", torch.bfloat16) as outer:
        with encoder_autocast("cpu", torch.float16) as inner:
            assert common._ENCODER_AUTOCAST is inner
        assert common._ENCODER_AUTOCAST is outer
    assert common._ENCODER_AUTOCAST is None


def test_support_is_opt_in():
    from sleap_nn.architectures.common import FreezableEncoderMixin
    from sleap_nn.architectures.convnext import ConvNextWrapper
    from sleap_nn.architectures.pretrained import PretrainedBackbone
    from sleap_nn.architectures.swint import SwinTWrapper
    from sleap_nn.architectures.unet import UNet

    assert FreezableEncoderMixin.supports_encoder_autocast is False
    for cls in (UNet, ConvNextWrapper, SwinTWrapper, PretrainedBackbone):
        assert cls.supports_encoder_autocast is True


def test_scope_error_leaves_the_backend_unchanged(tiny_detr):
    from sleap_nn.inference.predictor import _set_autocast_precision

    backend = TorchBackend(model=tiny_detr, device="cpu", use_compile=False)
    backend.use_compile = True  # as if compiled: the encoder scope is refused
    with pytest.raises(ValueError, match="use_compile"):
        _set_autocast_precision(backend, "bf16-encoder")
    assert backend.autocast_dtype is None and backend.autocast_scope == "all"
