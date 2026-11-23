"""``ExportedNetwork`` device names with a TensorRT engine (stubbed: no GPU needed)."""

import json
import types

import pytest
import torch

import sleap_nn.inference.predictor as predictor_mod
from sleap_nn.inference.exported_network import ExportedNetwork


def _export_dir(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    (d / "export_metadata.json").write_text(
        json.dumps({"model_type": "bbox_centernet", "max_batch_size": 4})
    )
    (d / "model.trt").write_bytes(b"")
    return d


@pytest.fixture
def built(monkeypatch):
    seen = {}

    def fake_backend(runtime, path, device):
        seen["device"] = device
        return types.SimpleNamespace(_cuda_device=torch.device("cuda", 0))

    monkeypatch.setattr(predictor_mod, "_build_export_backend", fake_backend)
    return seen


def test_auto_picks_cuda_for_an_engine(monkeypatch, tmp_path, built):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    net = ExportedNetwork(_export_dir(tmp_path), runtime="tensorrt", device="auto")
    assert built["device"] == "cuda" and net.out_device == torch.device("cuda", 0)


def test_auto_without_cuda_is_refused_like_cpu(monkeypatch, tmp_path, built):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = _export_dir(tmp_path)
    for device in ("auto", "cpu"):
        with pytest.raises(ValueError, match="runs on CUDA only"):
            ExportedNetwork(d, runtime="tensorrt", device=device)
    assert "device" not in built


def test_cuda_index_is_passed_on(monkeypatch, tmp_path, built):
    ExportedNetwork(_export_dir(tmp_path), runtime="tensorrt", device="cuda:1")
    assert built["device"] == "cuda:1"


def test_auto_with_onnx_runtime_keeps_outputs_on_the_cpu(tmp_path, built):
    d = _export_dir(tmp_path)
    (d / "model.onnx").write_bytes(b"")
    net = ExportedNetwork(d, runtime="onnx", device="auto")
    assert built["device"] == "auto" and net.out_device == torch.device("cpu")
