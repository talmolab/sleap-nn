"""The exported single-instance resize: antialiased, built from gathers, dynamic size.

TensorRT's ONNX parser rejects an antialiased ``Resize`` node, so single-instance
models with an input scale built no engine. The wrapper now resizes with gathers and
sums (``antialiased_resize``), sized from the input's shape at run time.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sleap_nn.export.wrappers.base import (
    MAX_RESIZE_INPUT,
    antialiased_resize,
    resize_taps,
    resized_size_rule,
)
from sleap_nn.export.wrappers.single_instance import SingleInstanceONNXWrapper


@pytest.mark.parametrize("scale", [0.5, 0.25, 0.37, 0.3, 0.125, 1.5, 2.0])
def test_size_rule_reproduces_python(scale):
    p, q, exact = resized_size_rule(scale)
    assert exact
    n = np.arange(MAX_RESIZE_INPUT + 1, dtype=np.int64)
    assert np.array_equal((n * p) // q, [int(int(k) * scale) for k in n])


def test_size_rule_without_an_exact_fraction():
    # int(10 * 0.7) == 7 but int(20 * 0.7) == 13: no fraction floors like that.
    p, q, exact = resized_size_rule(0.7)
    assert not exact and (p, q) == (7, 10)


@pytest.mark.parametrize("scale", [0.5, 0.25, 0.37, 0.8, 1.5])
@pytest.mark.parametrize("hw", [(64, 64), (97, 131), (32, 37), (240, 320)])
def test_matches_antialiased_interpolate(scale, hw):
    torch.manual_seed(0)
    x = torch.rand(2, 3, *hw)
    p, q, _ = resized_size_rule(scale)
    got = antialiased_resize(x, p, q, resize_taps(p, q))
    want = F.interpolate(
        x,
        size=(int(hw[0] * scale), int(hw[1] * scale)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    assert got.shape == want.shape
    # (float rounding only: torch computes its centers in float)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=0)


class _Confmaps(torch.nn.Module):
    """Two fixed 1x1-conv confidence maps at stride 1."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor([[1.0, -0.5, 0.2], [-0.3, 0.8, 0.4]])[..., None, None]
            )
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


def _reference_peaks(image_u8, scale):
    """The checkpoint path's resize (antialiased, at the frame's own size), then argmax."""
    x = image_u8.float() / 255.0
    h, w = x.shape[-2:]
    x = F.interpolate(
        x,
        size=(int(h * scale), int(w * scale)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    cm = _Confmaps()(x)
    b, c, hh, ww = cm.shape
    idx = cm.reshape(b, c, -1).argmax(-1)
    return torch.stack([idx % ww, idx // ww], -1).float() / scale


def test_single_instance_graph_tensorrt_can_parse_and_follows_the_frame_size(tmp_path):
    """No antialiased Resize in the graph; ONNX Runtime matches the checkpoint resize
    at the traced size and at other sizes."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from sleap_nn.export.exporters.onnx_exporter import export_to_onnx

    wrapper = SingleInstanceONNXWrapper(
        _Confmaps(), output_stride=1, input_scale=0.5, peak_threshold=-1e3
    ).eval()
    path = export_to_onnx(
        wrapper,
        tmp_path / "model.onnx",
        input_shape=(1, 3, 64, 96),
        input_names=["image"],
        output_names=["peaks", "peak_vals"],
    )
    graph = onnx.load(str(path)).graph
    for node in graph.node:
        if node.op_type == "Resize":
            assert not any(
                a.name == "antialias" and a.i for a in node.attribute
            ), "TensorRT parses no antialiased Resize"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for hw in [(64, 96), (80, 120), (50, 70)]:
        image = torch.from_numpy(rng.integers(0, 256, (2, 3, *hw), dtype=np.uint8))
        peaks = session.run(["peaks"], {"image": image.numpy()})[0]
        np.testing.assert_array_equal(peaks, _reference_peaks(image, 0.5).numpy())


def test_a_frame_that_scales_to_zero_pixels_gives_one():
    """int(1 * 0.5) is 0: the graph returns one pixel instead of dividing by zero."""
    p, q, _ = resized_size_rule(0.5)
    out = antialiased_resize(torch.rand(1, 1, 1, 9), p, q, resize_taps(p, q))
    assert out.shape == (1, 1, 1, 4)


def test_inexact_scale_warns(caplog):
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        SingleInstanceONNXWrapper(torch.nn.Identity(), input_scale=0.7)
        SingleInstanceONNXWrapper(torch.nn.Identity(), input_scale=0.5)
    finally:
        logger.remove(sink)
    assert len(seen) == 1 and "one pixel" in seen[0]
