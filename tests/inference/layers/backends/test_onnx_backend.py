"""Tests for ``ONNXBackend``.

The whole module is gated on ``onnxruntime`` being importable. On CI
this lives behind the ``[export]`` extra and is installed there; locally
it lands when ``uv sync --extra export`` runs. If neither, all tests
skip cleanly.

Coverage:

1. Protocol satisfaction: ``isinstance(backend, ModelBackend)``.
2. ``does_baked_postproc`` is ``True``.
3. Loading a tiny synthetic ONNX model and running its forward via the
   backend's ``__call__`` produces output dict matching the model's
   declared output names.
4. ``warmup`` runs without raising and primes the session.
5. ``from_export_dir`` finds and loads ``model.onnx`` from a directory.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

try:
    import onnxruntime  # noqa: F401
except ImportError:
    onnxruntime = None

pytestmark = pytest.mark.skipif(onnxruntime is None, reason="onnxruntime not installed")


def _export_tiny_onnx_to(path: Path) -> str:
    """Export a 1-conv model to ONNX with named ``peaks`` + ``peak_vals`` outputs.

    Names are chosen to match what ``InferenceLayer`` subclasses expect on
    the ``does_baked_postproc=True`` path so a layer can drive this
    backend in tests if needed.
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())
            # ``peaks`` / ``peak_vals`` derived from the conv output so the
            # ONNX optimizer doesn't constant-fold the input out of the graph.
            peaks = torch.stack(
                [cms.amax(dim=(-2, -1)), cms.amin(dim=(-2, -1))], dim=-1
            )  # (B, n_nodes, 2)
            peak_vals = cms.amax(dim=(-2, -1))  # (B, n_nodes)
            return peaks, peak_vals

    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 8, 8),),
        str(path),
        input_names=["image"],
        output_names=["peaks", "peak_vals"],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return str(path)


# ─────────────────────────────────────────────────────────────────────────
# 1 + 2. Protocol invariants
# ─────────────────────────────────────────────────────────────────────────


def test_onnx_backend_satisfies_model_backend(tmp_path):
    """Static structural typing — ``isinstance(backend, ModelBackend)``."""
    from sleap_nn.inference.layers.backends import ModelBackend, ONNXBackend

    onnx_path = _export_tiny_onnx_to(tmp_path / "tiny.onnx")
    backend = ONNXBackend(model_path=onnx_path, device="cpu")
    assert isinstance(backend, ModelBackend)


def test_onnx_backend_does_baked_postproc_is_true(tmp_path):
    """ONNX wrappers bake peak finding; this property reflects that."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    onnx_path = _export_tiny_onnx_to(tmp_path / "tiny.onnx")
    backend = ONNXBackend(model_path=onnx_path, device="cpu")
    assert backend.does_baked_postproc is True


# ─────────────────────────────────────────────────────────────────────────
# 3. Forward returns dict of torch tensors
# ─────────────────────────────────────────────────────────────────────────


def test_onnx_backend_forward_returns_named_dict(tmp_path):
    """Session output names become the dict keys; values are torch tensors."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    onnx_path = _export_tiny_onnx_to(tmp_path / "tiny.onnx")
    backend = ONNXBackend(model_path=onnx_path, device="cpu")

    x = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    out = backend(x)
    assert isinstance(out, dict)
    assert set(out) == {"peaks", "peak_vals"}
    assert isinstance(out["peaks"], torch.Tensor)
    assert isinstance(out["peak_vals"], torch.Tensor)
    assert out["peaks"].shape == (1, 4, 2)
    assert out["peak_vals"].shape == (1, 4)


def test_onnx_backend_casts_to_session_dtype(tmp_path):
    """Inputs in the wrong dtype get auto-cast to the session's expected dtype."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    onnx_path = _export_tiny_onnx_to(tmp_path / "tiny.onnx")
    backend = ONNXBackend(model_path=onnx_path, device="cpu")

    # Pass uint8 input — the synthetic model expects float32; conversion
    # happens inside the backend's __call__.
    x = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    out = backend(x)  # should not raise
    assert "peaks" in out


# ─────────────────────────────────────────────────────────────────────────
# 4. warmup
# ─────────────────────────────────────────────────────────────────────────


def test_onnx_backend_warmup_runs(tmp_path):
    """``warmup`` does not raise on a CPU-provider session."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    onnx_path = _export_tiny_onnx_to(tmp_path / "tiny.onnx")
    backend = ONNXBackend(model_path=onnx_path, device="cpu")
    backend.warmup((1, 1, 8, 8))


# ─────────────────────────────────────────────────────────────────────────
# 5. from_export_dir helper
# ─────────────────────────────────────────────────────────────────────────


def test_from_export_dir_finds_onnx_file(tmp_path):
    """``from_export_dir(dir)`` locates ``*.onnx`` and constructs the backend."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    onnx_path = _export_tiny_onnx_to(export_dir / "model.onnx")
    backend = ONNXBackend.from_export_dir(export_dir, device="cpu")
    assert isinstance(backend, ONNXBackend)
    assert backend.model_path == onnx_path


def test_from_export_dir_raises_when_no_onnx(tmp_path):
    """Missing ``.onnx`` raises a clear ``FileNotFoundError``."""
    from sleap_nn.inference.layers.backends import ONNXBackend

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=".onnx"):
        ONNXBackend.from_export_dir(empty_dir, device="cpu")
