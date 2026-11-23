"""Exported UNet / ConvNeXt graphs pad frames to the backbone's max stride at run time.

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
    ],
)
def test_exported_graph_matches_checkpoint_at_every_size(
    build, traced, sizes, tmp_path
):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = build()
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


def test_eager_forward_is_not_padded():
    """Only traced graphs pad: eager and compiled forwards keep their sizes."""
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
