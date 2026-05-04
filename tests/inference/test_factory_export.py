"""Tests for ``Predictor.from_export_dir`` (PR 18 — single-instance only).

The full export-dir factory dispatches on ``ExportMetadata.model_type``.
This file exercises the ``"single_instance"`` path (the case where the
existing :class:`SingleInstanceLayer` consumes ONNX baked output natively
via ``does_baked_postproc=True``). Other model types currently raise
``NotImplementedError`` — those tests live here too, asserting the
clear-error behavior so the contract doesn't drift before adapters land.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

try:
    import onnxruntime  # noqa: F401
except ImportError:  # pragma: no cover — gated below
    onnxruntime = None

pytestmark = pytest.mark.skipif(onnxruntime is None, reason="onnxruntime not installed")


# ──────────────────────────────────────────────────────────────────────
# Synthetic export-dir fixture: tiny single-instance ONNX + metadata
# ──────────────────────────────────────────────────────────────────────


def _export_tiny_single_instance(export_dir: Path, n_nodes: int = 2) -> Path:
    """Export a 1-conv ONNX model that emits ``peaks`` + ``peak_vals``.

    Mirrors the schema produced by ``sleap_nn.export.wrappers`` for a
    single-instance model: the wrapper bakes peak finding into the
    graph, so the ONNX session output has ``peaks`` (B, n_nodes, 2)
    and ``peak_vals`` (B, n_nodes).
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, n_nodes, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())
            B, C, H, W = cms.shape
            # Argmax over (H, W) → (row, col) per channel, packed as (x, y).
            flat = cms.view(B, C, -1)
            idx = flat.argmax(dim=-1)  # (B, C)
            row = (idx // W).float()
            col = (idx % W).float()
            peaks = torch.stack([col, row], dim=-1)  # (B, C, 2) → (x, y)
            peak_vals = flat.amax(dim=-1)  # (B, C)
            return peaks, peak_vals

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=["peaks", "peak_vals"],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _write_metadata(
    export_dir: Path,
    model_type: str = "single_instance",
    n_nodes: int = 2,
    output_stride: int = 1,
    input_scale: float = 1.0,
    peak_threshold: float = 0.0,
) -> Path:
    """Write an ``export_metadata.json`` with the minimum fields the factory reads."""
    meta = {
        "sleap_nn_version": "0.0.0",
        "export_timestamp": "2026-01-01T00:00:00",
        "export_format": "onnx",
        "model_type": model_type,
        "model_name": "test_single_instance",
        "checkpoint_path": "/tmp/fake.ckpt",
        "backbone": "unet",
        "n_nodes": n_nodes,
        "n_edges": 0,
        "node_names": [f"n{i}" for i in range(n_nodes)],
        "edge_inds": [],
        "input_scale": input_scale,
        "input_channels": 1,
        "output_stride": output_stride,
        "peak_threshold": peak_threshold,
    }
    path = export_dir / "export_metadata.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def single_instance_export(tmp_path):
    """A complete export dir: model.onnx + export_metadata.json."""
    export_dir = tmp_path / "single_instance_export"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)
    _write_metadata(export_dir, model_type="single_instance", n_nodes=2)
    return export_dir


# ──────────────────────────────────────────────────────────────────────
# Single-instance happy path
# ──────────────────────────────────────────────────────────────────────


def test_from_export_dir_single_instance_builds_predictor(single_instance_export):
    """``from_export_dir`` produces a :class:`Predictor` whose layer is a
    :class:`SingleInstanceLayer` wired to an :class:`ONNXBackend`."""
    from sleap_nn.inference.factory import from_export_dir
    from sleap_nn.inference.layers.backends import ONNXBackend
    from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
    from sleap_nn.inference.predictor import Predictor

    predictor = from_export_dir(single_instance_export, device="cpu")
    assert isinstance(predictor, Predictor)
    assert isinstance(predictor.layer, SingleInstanceLayer)
    assert isinstance(predictor.layer.backend, ONNXBackend)
    assert predictor.layer.backend.does_baked_postproc is True


def test_from_export_dir_single_instance_predict_smoke(single_instance_export):
    """End-to-end: build via ``from_export_dir`` and call ``predict()``
    on a synthetic ``NumpyProvider`` batch. Output should be one
    ``Outputs`` per batch with populated ``pred_keypoints``."""
    from sleap_nn.inference.factory import from_export_dir
    from sleap_nn.inference.providers import NumpyProvider

    predictor = from_export_dir(single_instance_export, device="cpu")
    images = np.zeros((1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider)
    assert len(outputs_list) == 1
    out = outputs_list[0]
    assert out.pred_keypoints is not None
    # One frame, n_nodes=2, (x, y) → shape (1, 1, 2, 2).
    assert out.pred_keypoints.shape[-1] == 2
    assert out.pred_keypoints.shape[-2] == 2


def test_predictor_classmethod_alias_matches_function(single_instance_export):
    """``Predictor.from_export_dir`` and ``factory.from_export_dir`` are equivalent."""
    from sleap_nn.inference.factory import from_export_dir as fn
    from sleap_nn.inference.predictor import Predictor

    direct = fn(single_instance_export, device="cpu")
    via_classmethod = Predictor.from_export_dir(single_instance_export, device="cpu")

    assert type(direct.layer) is type(via_classmethod.layer)
    assert direct.layer.output_stride == via_classmethod.layer.output_stride


def test_from_export_dir_respects_input_scale_and_output_stride(tmp_path):
    """``input_scale`` + ``output_stride`` from metadata flow into the layer's configs."""
    from sleap_nn.inference.factory import from_export_dir

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)
    _write_metadata(
        export_dir,
        model_type="single_instance",
        n_nodes=2,
        output_stride=4,
        input_scale=0.5,
    )

    predictor = from_export_dir(export_dir, device="cpu")
    assert predictor.layer.output_stride == 4
    assert predictor.layer.preprocess_config.scale == 0.5


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


def test_from_export_dir_missing_metadata_raises(tmp_path):
    """No ``export_metadata.json`` ⇒ ``FileNotFoundError`` with a clear message."""
    from sleap_nn.inference.factory import from_export_dir

    export_dir = tmp_path / "empty"
    export_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="export_metadata.json"):
        from_export_dir(export_dir, device="cpu")


def test_from_export_dir_missing_model_file_raises(tmp_path):
    """Metadata present but no ``model.onnx`` / ``model.trt`` ⇒ ``FileNotFoundError``."""
    from sleap_nn.inference.factory import from_export_dir

    export_dir = tmp_path / "metadata_only"
    export_dir.mkdir()
    _write_metadata(export_dir, model_type="single_instance")
    with pytest.raises(FileNotFoundError, match="model.onnx or model.trt"):
        from_export_dir(export_dir, device="cpu")


@pytest.mark.parametrize(
    "model_type",
    [
        "centroid",
        "centered_instance",
        "topdown",
        "bottomup",
        "multi_class_bottomup",
        "multi_class_topdown",
    ],
)
def test_from_export_dir_unsupported_model_type_raises(tmp_path, model_type):
    """Non-single-instance model types raise ``NotImplementedError`` (PR 18 scope)."""
    from sleap_nn.inference.factory import from_export_dir

    export_dir = tmp_path / f"export_{model_type}"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)  # any ONNX file works
    _write_metadata(export_dir, model_type=model_type)

    with pytest.raises(NotImplementedError, match=model_type):
        from_export_dir(export_dir, device="cpu")


def test_from_export_dir_unknown_runtime_raises(single_instance_export):
    """``runtime`` other than auto/onnx/tensorrt ⇒ ``ValueError``."""
    from sleap_nn.inference.factory import from_export_dir

    with pytest.raises(ValueError, match="Unknown runtime"):
        from_export_dir(single_instance_export, runtime="foo", device="cpu")


def test_from_export_dir_explicit_onnx_runtime(single_instance_export):
    """``runtime='onnx'`` works when the .onnx file exists."""
    from sleap_nn.inference.factory import from_export_dir

    predictor = from_export_dir(single_instance_export, runtime="onnx", device="cpu")
    assert predictor.layer is not None


def test_from_export_dir_tensorrt_missing_engine_raises(single_instance_export):
    """``runtime='tensorrt'`` on an ONNX-only dir ⇒ ``FileNotFoundError``."""
    from sleap_nn.inference.factory import from_export_dir

    with pytest.raises(FileNotFoundError, match="model.trt"):
        from_export_dir(single_instance_export, runtime="tensorrt", device="cpu")
