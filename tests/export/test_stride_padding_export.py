"""Exported UNet / ConvNeXt / SwinT / pretrained graphs pad the network input to the
backbone's max stride at run time.

The pooling layers' "same" padding and the decoder's skip resizes are decided from
the frame size in Python, so a trace freezes them. Before the fix a graph traced at
one size failed (decoder Concat) or padded wrongly at a size needing other padding.
The model now pads its input to a multiple of ``max_stride`` with shape arithmetic
while tracing, as checkpoint inference does on the host.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.model import Model
from sleap_nn.data.resizing import apply_pad_to_stride
from sleap_nn.export.wrappers.network import NetworkONNXWrapper

HEAD = OmegaConf.create(
    {
        "confmaps": {
            "part_names": ["a", "b"],
            "sigma": 2.5,
            "output_stride": 2,
            "loss_weight": 1.0,
        }
    }
)


def _unet(max_stride=8):
    cfg = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 4,
            "filters_rate": 1.5,
            "max_stride": max_stride,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "stacks": 1,
            "convs_per_block": 2,
            "output_stride": 2,
        }
    )
    torch.manual_seed(0)
    return Model("unet", cfg, HEAD, "single_instance").eval()


def _convnext():
    cfg = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": None,
            "arch": {"depths": [1, 1, 1, 1], "channels": [8, 16, 32, 64]},
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 2,
            "max_stride": 32,
        }
    )
    torch.manual_seed(0)
    return Model("convnext", cfg, HEAD, "single_instance").eval()


def _pretrained(tmp_path):
    """A tiny random-init HF ConvNeXt V2 encoder, saved so it loads offline."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ConvNextV2Config(
        num_channels=3, hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1]
    )
    path = tmp_path / "convnextv2"
    cfg.save_pretrained(path)
    backbone = OmegaConf.create(
        dict(
            source="hf",
            model_name=path.as_posix(),
            weights=False,
            mode="auto",
            in_channels=3,
            output_stride=4,
            max_stride=32,
            normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
    )
    head = OmegaConf.create(
        {"confmaps": {**HEAD.confmaps, "output_stride": 4}}  # the encoder's finest
    )
    torch.manual_seed(0)
    return Model("pretrained", backbone, head, "single_instance").eval()


def _export(model, path, hw):
    from sleap_nn.export.exporters import export_to_onnx

    wrapper = NetworkONNXWrapper(model, ["SingleInstanceConfmapsHead"]).eval()
    export_to_onnx(
        wrapper,
        path,
        input_shape=(1, 1, *hw),
        input_names=["image"],
        output_names=["SingleInstanceConfmapsHead"],
        verify=False,
    )


def _checkpoint_forward(model, x_uint8):
    """The checkpoint path: [0, 1] frame, padded to the max stride on the host."""
    x = x_uint8.float() / 255.0
    x = apply_pad_to_stride(x, int(model.backbone_config.max_stride))
    with torch.no_grad():
        return model(x)["SingleInstanceConfmapsHead"].numpy()


@pytest.mark.parametrize(
    "build, traced, sizes",
    [
        # traced at a multiple of the stride; run at multiples and at sizes needing
        # other padding in one or both axes
        (_unet, (64, 64), [(64, 64), (72, 80), (70, 66), (61, 64), (64, 77)]),
        # traced at a size needing padding
        (_unet, (61, 70), [(61, 70), (64, 64), (75, 59)]),
        (_convnext, (64, 64), [(64, 64), (96, 64), (70, 90)]),
        (_convnext, (70, 90), [(70, 90), (64, 64), (100, 61)]),
        (_pretrained, (64, 64), [(64, 64), (96, 64), (70, 90)]),
    ],
)
def test_exported_graph_matches_checkpoint_at_every_size(
    build, traced, sizes, tmp_path
):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = build(tmp_path) if build is _pretrained else build()
    path = tmp_path / "model.onnx"
    _export(model, path, traced)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for h, w in sizes:
        x = torch.from_numpy(rng.integers(0, 256, (1, 1, h, w), dtype=np.uint8))
        (got,) = sess.run(None, {"image": x.numpy()})
        want = _checkpoint_forward(model, x)
        assert got.shape == want.shape, (h, w, got.shape, want.shape)
        np.testing.assert_allclose(got, want, atol=1e-5, err_msg=f"{h}x{w}")


def test_unchanged_at_sizes_that_need_no_padding(tmp_path, monkeypatch):
    """At multiples of the max stride the graph's outputs are bit-identical to a
    graph exported without the pad (what exports computed before)."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import sleap_nn.architectures.model as model_module

    model = _unet()
    _export(model, tmp_path / "new.onnx", (64, 64))
    monkeypatch.setattr(model_module, "_STRIDE_PADDED_BACKBONES", ())
    _export(model, tmp_path / "old.onnx", (64, 64))
    new = ort.InferenceSession(
        str(tmp_path / "new.onnx"), providers=["CPUExecutionProvider"]
    )
    old = ort.InferenceSession(
        str(tmp_path / "old.onnx"), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    for h, w in [(64, 64), (72, 80), (40, 96)]:
        x = rng.integers(0, 256, (2, 1, h, w), dtype=np.uint8)
        (a,) = new.run(None, {"image": x})
        (b,) = old.run(None, {"image": x})
        assert np.array_equal(a, b), (h, w)


def test_full_wrapper_matches_checkpoint_at_odd_sizes(tmp_path):
    """A single-instance graph (peaks baked in) at sizes needing other padding."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from sleap_nn.export.exporters import export_to_onnx
    from sleap_nn.export.wrappers.single_instance import SingleInstanceONNXWrapper

    model = _unet()
    wrapper = SingleInstanceONNXWrapper(model, output_stride=2, peak_threshold=0.0)
    wrapper.eval()
    export_to_onnx(
        wrapper,
        tmp_path / "m.onnx",
        input_shape=(1, 1, 64, 64),
        input_names=["image"],
        output_names=["peaks", "peak_vals"],
        verify=False,
    )
    sess = ort.InferenceSession(
        str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"]
    )

    class _HostPadded(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(
                apply_pad_to_stride(x, int(self.m.backbone_config.max_stride))
            )

    reference = SingleInstanceONNXWrapper(
        _HostPadded(model), output_stride=2, peak_threshold=0.0
    ).eval()
    rng = np.random.default_rng(1)
    for h, w in [(64, 64), (70, 66), (61, 77)]:
        x = torch.from_numpy(rng.integers(0, 256, (1, 1, h, w), dtype=np.uint8))
        peaks, vals = sess.run(None, {"image": x.numpy()})
        with torch.no_grad():
            want = reference(x)
        np.testing.assert_allclose(peaks, want["peaks"].numpy(), atol=1e-4)
        np.testing.assert_allclose(vals, want["peak_vals"].numpy(), atol=1e-5)


@pytest.mark.parametrize("hw", [(64, 64), (61, 70)])
def test_numerical_check_compares_like_for_like(hw, tmp_path):
    """``numerical_check`` runs its eager reference with the graph's padding."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from sleap_nn.export.exporters import export_to_onnx

    wrapper = NetworkONNXWrapper(_unet(), ["SingleInstanceConfmapsHead"]).eval()
    export_to_onnx(
        wrapper,
        tmp_path / "m.onnx",
        input_shape=(1, 1, *hw),
        input_names=["image"],
        output_names=["SingleInstanceConfmapsHead"],
        verify=False,
        numerical_check=True,
    )


def test_eager_forward_is_not_padded():
    """Only traced graphs pad: the eager forward keeps its sizes."""
    model = _unet()
    with torch.no_grad():
        out = model(torch.rand(1, 1, 64, 64))["SingleInstanceConfmapsHead"]
    assert out.shape[-2:] == (32, 32)
    x = torch.rand(1, 1, 61, 70)
    with torch.no_grad():
        out = model(x)["SingleInstanceConfmapsHead"]
    # what it computed before the fix: the eager "same" pooling on the raw frame
    assert out.shape[-2:] == (31, 35)


@pytest.mark.parametrize("hw", [(64, 64), (61, 70), (1, 1), (33, 8)])
@pytest.mark.parametrize("stride", [1, 8, 16])
def test_pad_to_stride_traced_matches_the_host_pad(hw, stride):
    from sleap_nn.architectures.model import _pad_to_stride_traced

    x = torch.rand(2, 3, *hw)
    torch.testing.assert_close(
        _pad_to_stride_traced(x, stride), apply_pad_to_stride(x, stride)
    )
