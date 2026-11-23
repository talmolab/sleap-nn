"""``TensorRTBackend`` device names (runs without TensorRT or a GPU)."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest
import torch

from sleap_nn.inference.layers.backends.tensorrt_backend import (
    TensorRTBackend,
    _cuda_index,
)


def test_cuda_index():
    assert _cuda_index("cuda") is None
    assert _cuda_index("auto") is None
    assert _cuda_index("cuda:0") == 0
    assert _cuda_index("CUDA:3") == 3
    for bad in ("cpu", "mps", "cuda:", "cuda:x", "cuda0"):
        with pytest.raises(ValueError, match="only supports CUDA"):
            _cuda_index(bad)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_cuda_devices_pass_the_device_check(monkeypatch, tmp_path, device):
    """``cuda:N`` reaches the CUDA check, as ``cuda`` does (it was refused before)."""
    trt = types.ModuleType("tensorrt")
    trt.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        TensorRTBackend(engine_path=str(tmp_path / "model.trt"), device=device)


def test_cuda_index_beyond_the_visible_devices(monkeypatch, tmp_path):
    trt = types.ModuleType("tensorrt")
    trt.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="only 2 CUDA device"):
        TensorRTBackend(engine_path=str(tmp_path / "model.trt"), device="cuda:2")


def test_cpu_is_refused(tmp_path):
    with pytest.raises(ValueError, match="only supports CUDA"):
        TensorRTBackend(engine_path=str(tmp_path / "model.trt"), device="cpu")
