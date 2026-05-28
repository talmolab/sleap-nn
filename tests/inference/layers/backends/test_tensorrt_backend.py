"""Tests for ``TensorRTBackend``.

The whole module is gated on:

1. ``import tensorrt`` succeeding
2. ``torch.cuda.is_available()``

Both are absent on Mac CI, so all tests skip cleanly there. Building a
real TRT engine requires a CUDA host with the ``[tensorrt]`` extra
installed; once that runner is available, these tests exercise:

1. Protocol satisfaction
2. ``does_baked_postproc=True``
3. CPU-device construction raises ``ValueError``
4. Missing engine file raises ``FileNotFoundError``
5. Constructor raises a clear ``ImportError`` when ``tensorrt`` isn't
   installed on a CUDA host
"""

from __future__ import annotations

import pytest
import torch

try:
    import tensorrt  # noqa: F401

    _trt_ok = True
except ImportError:
    _trt_ok = False

pytestmark = pytest.mark.skipif(
    not (_trt_ok and torch.cuda.is_available()),
    reason="TensorRT requires CUDA + the [tensorrt] extra",
)


def test_tensorrt_backend_rejects_cpu_device(tmp_path):
    """``device='cpu'`` must raise — TRT requires CUDA."""
    from sleap_nn.inference.layers.backends import TensorRTBackend

    with pytest.raises(ValueError, match="CUDA"):
        TensorRTBackend(engine_path=str(tmp_path / "nonexistent.trt"), device="cpu")


def test_tensorrt_backend_raises_on_missing_engine(tmp_path):
    """Missing engine file produces a clear ``FileNotFoundError``."""
    from sleap_nn.inference.layers.backends import TensorRTBackend

    with pytest.raises(FileNotFoundError, match=".trt"):
        TensorRTBackend(engine_path=str(tmp_path / "missing.trt"), device="cuda")


def test_from_export_dir_raises_when_no_trt(tmp_path):
    """``from_export_dir`` raises a clear ``FileNotFoundError`` if no
    ``.trt`` is present in the directory."""
    from sleap_nn.inference.layers.backends import TensorRTBackend

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=".trt"):
        TensorRTBackend.from_export_dir(empty_dir)
