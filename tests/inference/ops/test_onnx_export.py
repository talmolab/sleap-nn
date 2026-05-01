"""ONNX-export smoke tests for the inference ops library.

Per design doc §4C, ``ops/peaks.py``, ``ops/coord.py``, and ``ops/crops.py``
must remain ONNX-exportable so the ONNX/TensorRT backends in PR 7 (#515) can
share one implementation with the Torch path. ``ops/filters.py`` and
``ops/identity.py`` are intentionally Python-only (NMS / Hungarian
matching) and exempt from this gate.

Each test wraps the function under test in a tiny ``nn.Module``, exports
to ONNX at opset 16, runs the resulting session via ``onnxruntime``, and
compares the output to the eager Torch reference within ``1e-5``.

If a future PR adds a non-exportable op (``tensor.item()``, variable-length
indexing, Python scalar comparisons, …), one of these tests fails fast.
"""

from __future__ import annotations

import io
from typing import Any, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from sleap_nn.inference.ops.coord import (
    add_crop_offset,
    apply_input_scale,
    undo_eff_scale,
    undo_input_scale,
    undo_stride,
)
from sleap_nn.inference.ops.peaks import (
    find_global_peaks_rough,
    find_local_peaks_rough,
    integral_regression,
    morphological_dilation,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None


pytestmark = pytest.mark.skipif(ort is None, reason="onnxruntime not installed")


OPSET = 16
ATOL = 1e-5
RTOL = 1e-5


def _export_and_run(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
) -> list[np.ndarray]:
    """Export ``module`` to ONNX and run it via onnxruntime."""
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        inputs,
        buf,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        do_constant_folding=True,
        # Use the legacy TorchScript exporter to dodge the swint torch.fx.wrap
        # bug (#527) that breaks dynamo-mode export.
        dynamo=False,
    )
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    feeds = {name: t.detach().cpu().numpy() for name, t in zip(input_names, inputs)}
    return sess.run(output_names, feeds)


def _assert_close(out_onnx: Any, out_torch: torch.Tensor) -> None:
    np.testing.assert_allclose(
        out_onnx, out_torch.detach().cpu().numpy(), atol=ATOL, rtol=RTOL
    )


# ──────────────────────────────────────────────────────────────────────────
# coord.py — every function should export trivially
# ──────────────────────────────────────────────────────────────────────────


class _UndoStrideModule(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return undo_stride(x, self.stride)


def test_undo_stride_exports():
    coords = torch.randn(2, 3, 2)
    module = _UndoStrideModule(stride=4)
    (out_onnx,) = _export_and_run(module, (coords,), ["coords"], ["out"])
    _assert_close(out_onnx, module(coords))


class _UndoInputScaleModule(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return undo_input_scale(x, self.scale)


def test_undo_input_scale_exports():
    coords = torch.randn(2, 3, 2)
    module = _UndoInputScaleModule(scale=0.5)
    (out_onnx,) = _export_and_run(module, (coords,), ["coords"], ["out"])
    _assert_close(out_onnx, module(coords))


class _UndoEffScaleModule(nn.Module):
    def forward(self, x: torch.Tensor, eff: torch.Tensor) -> torch.Tensor:
        return undo_eff_scale(x, eff)


def test_undo_eff_scale_exports():
    coords = torch.randn(2, 3, 2)
    eff = torch.tensor([2.0, 0.5])
    module = _UndoEffScaleModule()
    (out_onnx,) = _export_and_run(module, (coords, eff), ["coords", "eff"], ["out"])
    _assert_close(out_onnx, module(coords, eff))


class _AddCropOffsetModule(nn.Module):
    def forward(self, peaks: torch.Tensor, topleft: torch.Tensor) -> torch.Tensor:
        return add_crop_offset(peaks, topleft)


def test_add_crop_offset_exports():
    peaks = torch.randn(4, 5, 2)
    topleft = torch.randn(4, 2)
    module = _AddCropOffsetModule()
    (out_onnx,) = _export_and_run(
        module, (peaks, topleft), ["peaks", "topleft"], ["out"]
    )
    _assert_close(out_onnx, module(peaks, topleft))


class _ApplyInputScaleModule(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_input_scale(x, self.scale)


def test_apply_input_scale_exports_identity():
    img = torch.randn(2, 1, 8, 8)
    module = _ApplyInputScaleModule(scale=1.0)
    (out_onnx,) = _export_and_run(module, (img,), ["img"], ["out"])
    _assert_close(out_onnx, module(img))


def test_apply_input_scale_exports_resize():
    img = torch.randn(2, 1, 8, 8)
    module = _ApplyInputScaleModule(scale=0.5)
    (out_onnx,) = _export_and_run(module, (img,), ["img"], ["out"])
    _assert_close(out_onnx, module(img))


# ──────────────────────────────────────────────────────────────────────────
# peaks.py — exportability of the parts that already ship to ONNX
# ──────────────────────────────────────────────────────────────────────────


class _MorphologicalDilationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kernel",
            torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return morphological_dilation(x, self.kernel)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "morphological_dilation uses Tensor.unfold which the legacy "
        "TorchScript ONNX exporter cannot handle (Unfold input size not "
        "accessible). PR 5 (#513) replaces this with F.max_pool2d-based "
        "NMS, after which this test should xpass — at which point drop "
        "this marker."
    ),
)
def test_morphological_dilation_exports():
    img = torch.randn(2, 1, 16, 16)
    module = _MorphologicalDilationModule()
    (out_onnx,) = _export_and_run(module, (img,), ["img"], ["out"])
    _assert_close(out_onnx, module(img))


class _IntegralRegressionModule(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        gv = torch.arange(size, dtype=torch.float32) - ((size - 1) / 2)
        self.register_buffer("xv", gv)
        self.register_buffer("yv", gv.clone())

    def forward(self, cms: torch.Tensor) -> torch.Tensor:
        x_hat, y_hat = integral_regression(cms, self.xv, self.yv)
        return torch.cat([x_hat, y_hat], dim=1)


def test_integral_regression_exports():
    cms = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)
    module = _IntegralRegressionModule(size=5)
    (out_onnx,) = _export_and_run(module, (cms,), ["cms"], ["out"])
    _assert_close(out_onnx, module(cms))


class _FindGlobalPeaksRoughModule(nn.Module):
    def forward(self, cms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return find_global_peaks_rough(cms, threshold=0.1)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "find_global_peaks_rough uses boolean-mask index_put_ for the "
        "below-threshold NaN-padding, which the legacy TorchScript ONNX "
        "exporter rejects. PR 5 (#513) rewrites the threshold logic with "
        "torch.where for both fixed-shape masking and ONNX exportability; "
        "this test will xpass then — drop the marker at that point."
    ),
)
def test_find_global_peaks_rough_exports():
    # Construct confmaps with known peaks so threshold=0.1 leaves valid points.
    cms = torch.zeros(2, 3, 8, 8)
    cms[..., 4, 4] = 1.0
    module = _FindGlobalPeaksRoughModule()
    out_pts_onnx, out_vals_onnx = _export_and_run(
        module, (cms,), ["cms"], ["pts", "vals"]
    )
    pts_torch, vals_torch = module(cms)
    _assert_close(out_pts_onnx, pts_torch)
    _assert_close(out_vals_onnx, vals_torch)


# ──────────────────────────────────────────────────────────────────────────
# Documented non-exportability — ``find_local_peaks_rough`` (variable peaks)
# ──────────────────────────────────────────────────────────────────────────
# ``find_local_peaks_rough`` returns a variable number of peaks (depends on
# the data), which the legacy TorchScript exporter doesn't support. This is
# documented as a known limitation; PR 7 (#515) addresses it by baking
# ``find_top_k_peaks`` (fixed-shape) into the ONNX wrappers.


def test_find_local_peaks_rough_known_export_gap():
    """Document — and pin — the current export limitation."""
    cms = torch.zeros(1, 1, 8, 8)
    cms[0, 0, 4, 4] = 1.0
    module = nn.Module()
    module.forward = lambda x: find_local_peaks_rough(x, threshold=0.1)  # type: ignore[assignment]
    with pytest.raises(Exception):  # noqa: B017 — torch raises a varied set
        _export_and_run(module, (cms,), ["cms"], ["pts", "vals", "s", "c"])
