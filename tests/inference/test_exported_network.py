"""Export folders on the export-directory route and their runtime choice.

The folders are built by hand (metadata, a copied training config, empty graph
files); no ONNX runtime or GPU is needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CENTERED = ASSETS / "model_ckpts" / "minimal_instance_centered_instance"
SLP = ASSETS / "datasets" / "minimal_instance.pkg.slp"


def _folder(tmp_path, name, metadata, *, ckpt=False, graphs=("model.onnx",)):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(CENTERED / "training_config.yaml", d / "training_config.yaml")
    if ckpt:
        shutil.copy(CENTERED / "best.ckpt", d / "best.ckpt")
    (d / "export_metadata.json").write_text(json.dumps(metadata))
    for g in graphs:
        (d / g).write_bytes(b"")
    return d


NETWORK_ONLY = {"model_type": "centered_instance", "network_only": True}


def test_export_dir_route_refuses_a_network_only_export(tmp_path):
    """A network-only export is a model path, not an export directory."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.run import predict

    alone = _folder(tmp_path, "net", NETWORK_ONLY)
    both = _folder(tmp_path, "both", NETWORK_ONLY, ckpt=True)
    for d in (alone, both):
        with pytest.raises(ValueError, match="model path"):
            Predictor.from_export_dir(d, device="cpu")
        with pytest.raises(ValueError, match="network-only export"):
            predict(str(SLP), export_dir=str(d), device="cpu")


def test_box_detector_exports_stay_on_the_export_dir_route(tmp_path):
    """Detector exports are network-only by type and keep their route."""
    from sleap_nn.inference.predictor import _refuse_network_only_export_dir

    for meta in ({"model_type": "bbox_centernet"}, {"model_type": "centroid"}):
        d = _folder(tmp_path, meta["model_type"], meta)
        _refuse_network_only_export_dir(d, d / "export_metadata.json")


def test_auto_runtime_off_cuda_takes_the_onnx_graph(tmp_path):
    """With both graph files, ``auto`` runs ONNX where an engine cannot run."""
    from sleap_nn.inference.predictor import _resolve_export_runtime

    d = _folder(
        tmp_path, "both_graphs", NETWORK_ONLY, graphs=("model.onnx", "model.trt")
    )
    for device in ("cpu", "mps"):
        assert _resolve_export_runtime(d, "auto", device) == ("onnx", d / "model.onnx")
    # On CUDA, and where no device is given, the engine is preferred as before.
    for device in ("cuda", "cuda:1", None):
        assert _resolve_export_runtime(d, "auto", device) == (
            "tensorrt",
            d / "model.trt",
        )
    # An explicit runtime is taken as given.
    assert _resolve_export_runtime(d, "tensorrt", "cpu")[0] == "tensorrt"


def test_auto_runtime_off_cuda_picks_onnx_on_both_routes(tmp_path, monkeypatch):
    """The model-path and the export-directory routes build the ONNX graph on CPU."""
    import sleap_nn.inference.predictor as predictor_mod
    from sleap_nn.inference.exported_network import ExportedNetwork

    built = []

    def backend(runtime, path, device):
        built.append((runtime, Path(path).name))
        raise RuntimeError("stop")

    monkeypatch.setattr(predictor_mod, "_build_export_backend", backend)
    net = _folder(tmp_path, "net", NETWORK_ONLY, graphs=("model.onnx", "model.trt"))
    with pytest.raises(RuntimeError, match="stop"):
        ExportedNetwork(net, runtime="auto", device="cpu")
    fused = _folder(
        tmp_path,
        "fused",
        {"model_type": "centered_instance"},
        graphs=("model.onnx", "model.trt"),
    )
    with pytest.raises(RuntimeError, match="stop"):
        predictor_mod.Predictor.from_export_dir(fused, device="cpu")
    assert built == [("onnx", "model.onnx"), ("onnx", "model.onnx")]


def test_an_engine_only_folder_off_cuda_gets_an_accurate_message(tmp_path):
    """No model.onnx to fall back to: the message does not point at ONNX."""
    from sleap_nn.inference.exported_network import ExportedNetwork
    from sleap_nn.inference.predictor import _resolve_export_runtime

    d = _folder(tmp_path, "engine_only", NETWORK_ONLY, graphs=("model.trt",))
    with pytest.raises(ValueError, match="holds only a TensorRT engine"):
        _resolve_export_runtime(d, "auto", "cpu")
    with pytest.raises(ValueError, match="has no model.onnx"):
        ExportedNetwork(d, runtime="tensorrt", device="cpu")
    both = _folder(tmp_path, "both", NETWORK_ONLY, graphs=("model.onnx", "model.trt"))
    with pytest.raises(ValueError, match="--runtime onnx"):
        ExportedNetwork(both, runtime="tensorrt", device="cpu")
