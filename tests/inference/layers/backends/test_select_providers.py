"""Tests for ``onnx_backend._select_providers`` (#584 — restores coverage lost
when tests/export/test_onnx_predictor.py was deleted).

``_select_providers`` is a pure function (device + available-providers list ->
ordered provider list), so these tests need NO onnxruntime and run everywhere.
"""

from __future__ import annotations

from sleap_nn.inference.layers.backends.onnx_backend import _select_providers

ALL = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
DML = ["DmlExecutionProvider", "CPUExecutionProvider"]
CPU_ONLY = ["CPUExecutionProvider"]


def test_cpu_and_host_select_cpu_only():
    assert _select_providers("cpu", ALL) == ["CPUExecutionProvider"]
    assert _select_providers("host", ALL) == ["CPUExecutionProvider"]


def test_cuda_prefers_cuda_then_cpu_skipping_trt():
    # CUDA/auto skip the ORT TensorRT EP (we ship a native TensorRTBackend).
    assert _select_providers("cuda", ALL) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _select_providers("cuda:0", ALL) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_auto_matches_cuda_selection():
    assert _select_providers("auto", ALL) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_cuda_without_cuda_available_falls_back_to_cpu():
    assert _select_providers("cuda", CPU_ONLY) == ["CPUExecutionProvider"]


def test_directml_and_dml_alias():
    assert _select_providers("directml", DML) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _select_providers("dml", DML) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_directml_falls_back_when_dml_absent():
    # No DmlExecutionProvider available -> fall back to whatever is available.
    assert _select_providers("dml", CPU_ONLY) == ["CPUExecutionProvider"]


def test_unknown_device_returns_available_unchanged():
    assert _select_providers("tpu", ALL) == ALL
    assert _select_providers("weird-device", DML) == DML
