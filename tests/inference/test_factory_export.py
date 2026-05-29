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
    """``from_export_dir`` produces a :class:`Predictor` whose layer is an
    :class:`ExportedSingleInstanceLayer` wired to an :class:`ONNXBackend`."""
    from sleap_nn.inference.layers.backends import ONNXBackend
    from sleap_nn.inference.layers.exported import ExportedSingleInstanceLayer
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_export_dir(single_instance_export, device="cpu")
    assert isinstance(predictor, Predictor)
    assert isinstance(predictor.layer, ExportedSingleInstanceLayer)
    assert isinstance(predictor.layer.backend, ONNXBackend)
    assert predictor.layer.backend.does_baked_postproc is True


def test_from_export_dir_single_instance_predict_smoke(single_instance_export):
    """End-to-end: build via ``from_export_dir`` and call ``predict()``
    on a synthetic ``NumpyProvider`` batch. Output should be one
    ``Outputs`` per batch with populated ``pred_keypoints``."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(single_instance_export, device="cpu")
    images = np.zeros((1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    assert len(outputs_list) == 1
    out = outputs_list[0]
    assert out.pred_keypoints is not None
    # One frame, n_nodes=2, (x, y) → shape (1, 1, 2, 2).
    assert out.pred_keypoints.shape[-1] == 2
    assert out.pred_keypoints.shape[-2] == 2


def test_from_export_dir_single_instance_no_double_coord_ladder(tmp_path):
    """Export adapter must not re-apply ``output_stride`` / ``input_scale``.

    The wrapper already produces peaks in original-image space; the
    adapter therefore bypasses the standard layer's coord ladder.
    With ``output_stride=4`` and ``input_scale=0.5`` in metadata, the
    wrapper output should pass through unchanged (no peaks * 16 bug).
    """
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    export_dir = tmp_path / "scaled_export"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)
    _write_metadata(
        export_dir,
        model_type="single_instance",
        n_nodes=2,
        output_stride=4,
        input_scale=0.5,
    )

    predictor = Predictor.from_export_dir(export_dir, device="cpu")
    images = np.zeros((1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    out = outputs_list[0]
    # Whatever the wrapper produced (argmax over 16x16 → values in [0, 15])
    # must come through unchanged. Specifically NOT re-multiplied by
    # output_stride/input_scale = 4/0.5 = 8.
    peaks = out.pred_keypoints
    assert peaks.max() < 16, (
        "Export adapter applied coord ladder twice — peaks should stay "
        f"in original [0, 16) space; got max={peaks.max()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


def test_from_export_dir_missing_metadata_raises(tmp_path):
    """No ``export_metadata.json`` ⇒ ``FileNotFoundError`` with a clear message."""
    from sleap_nn.inference.predictor import Predictor

    export_dir = tmp_path / "empty"
    export_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="export_metadata.json"):
        Predictor.from_export_dir(export_dir, device="cpu")


def test_from_export_dir_missing_model_file_raises(tmp_path):
    """Metadata present but no ``model.onnx`` / ``model.trt`` ⇒ ``FileNotFoundError``."""
    from sleap_nn.inference.predictor import Predictor

    export_dir = tmp_path / "metadata_only"
    export_dir.mkdir()
    _write_metadata(export_dir, model_type="single_instance")
    with pytest.raises(FileNotFoundError, match="model.onnx or model.trt"):
        Predictor.from_export_dir(export_dir, device="cpu")


def test_from_export_dir_unrecognized_model_type_raises(tmp_path):
    """An unknown ``model_type`` value raises ``ValueError``."""
    from sleap_nn.inference.predictor import Predictor

    export_dir = tmp_path / "export_unknown"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)
    _write_metadata(export_dir, model_type="some_future_model")

    with pytest.raises(ValueError, match="Unrecognized model_type"):
        Predictor.from_export_dir(export_dir, device="cpu")


def test_from_export_dir_unknown_runtime_raises(single_instance_export):
    """``runtime`` other than auto/onnx/tensorrt ⇒ ``ValueError``."""
    from sleap_nn.inference.predictor import Predictor

    with pytest.raises(ValueError, match="Unknown runtime"):
        Predictor.from_export_dir(single_instance_export, runtime="foo", device="cpu")


def test_from_export_dir_explicit_onnx_runtime(single_instance_export):
    """``runtime='onnx'`` works when the .onnx file exists."""
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_export_dir(
        single_instance_export, runtime="onnx", device="cpu"
    )
    assert predictor.layer is not None


def test_from_export_dir_tensorrt_missing_engine_raises(single_instance_export):
    """``runtime='tensorrt'`` on an ONNX-only dir ⇒ ``FileNotFoundError``."""
    from sleap_nn.inference.predictor import Predictor

    with pytest.raises(FileNotFoundError, match="model.trt"):
        Predictor.from_export_dir(
            single_instance_export, runtime="tensorrt", device="cpu"
        )


# ──────────────────────────────────────────────────────────────────────
# Centroid + centered-instance + topdown synthetic exports (PR 19)
# ──────────────────────────────────────────────────────────────────────


def _export_tiny_centroid(
    export_dir: Path, max_instances: int = 4, n_nodes: int = 2
) -> Path:
    """Export a 1-conv ONNX that emits centroid wrapper output schema.

    Output: ``centroids`` (B, I, 2), ``centroid_vals`` (B, I), ``instance_valid`` (B, I).
    The model picks the top-k peaks per sample over the confmap.
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.max_instances = max_instances

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())  # (B, 1, H, W)
            B, _C, H, W = cms.shape
            flat = cms.view(B, -1)
            vals, idx = flat.topk(self.max_instances, dim=-1)
            row = (idx // W).float()
            col = (idx % W).float()
            centroids = torch.stack([col, row], dim=-1)  # (B, I, 2)
            centroid_vals = vals  # (B, I)
            instance_valid = vals > 0.0  # (B, I) bool
            return centroids, centroid_vals, instance_valid

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=["centroids", "centroid_vals", "instance_valid"],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _export_tiny_topdown(
    export_dir: Path, max_instances: int = 3, n_nodes: int = 2
) -> Path:
    """Export an ONNX that emits the combined top-down wrapper output schema.

    Outputs: ``centroids`` (B, I, 2), ``centroid_vals`` (B, I),
    ``peaks`` (B, I, N, 2), ``peak_vals`` (B, I, N), ``instance_valid`` (B, I).
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.max_instances = max_instances
            self.n_nodes = n_nodes

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())
            B, _C, H, W = cms.shape
            flat = cms.view(B, -1)
            vals, idx = flat.topk(self.max_instances, dim=-1)
            row = (idx // W).float()
            col = (idx % W).float()
            centroids = torch.stack([col, row], dim=-1)  # (B, I, 2)
            instance_valid = vals > 0.0
            # Synthesize per-instance peaks: each node placed near the centroid.
            peaks = centroids.unsqueeze(2).expand(-1, -1, self.n_nodes, -1).clone()
            peak_vals = vals.unsqueeze(-1).expand(-1, -1, self.n_nodes).clone()
            return centroids, vals, peaks, peak_vals, instance_valid

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=[
            "centroids",
            "centroid_vals",
            "peaks",
            "peak_vals",
            "instance_valid",
        ],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


@pytest.fixture
def centroid_export(tmp_path):
    """Export dir for a centroid model: model.onnx + metadata."""
    export_dir = tmp_path / "centroid_export"
    export_dir.mkdir()
    _export_tiny_centroid(export_dir, max_instances=4, n_nodes=2)
    _write_metadata(export_dir, model_type="centroid", n_nodes=2)
    return export_dir


@pytest.fixture
def centered_instance_export(tmp_path):
    """Export dir for a centered-instance model: model.onnx + metadata.

    Reuses the single-instance ONNX schema since the wrapper output
    is identical (``peaks`` + ``peak_vals``); only ``model_type`` in
    metadata differs.
    """
    export_dir = tmp_path / "centered_instance_export"
    export_dir.mkdir()
    _export_tiny_single_instance(export_dir, n_nodes=2)
    _write_metadata(export_dir, model_type="centered_instance", n_nodes=2)
    return export_dir


@pytest.fixture
def topdown_export(tmp_path):
    """Export dir for a combined top-down model: model.onnx + metadata."""
    export_dir = tmp_path / "topdown_export"
    export_dir.mkdir()
    _export_tiny_topdown(export_dir, max_instances=3, n_nodes=2)
    _write_metadata(export_dir, model_type="topdown", n_nodes=2)
    return export_dir


def test_from_export_dir_centroid_builds_predictor(centroid_export):
    """Centroid export → :class:`ExportedCentroidLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import ExportedCentroidLayer

    predictor = Predictor.from_export_dir(centroid_export, device="cpu")
    assert isinstance(predictor.layer, ExportedCentroidLayer)


def test_from_export_dir_centroid_predict_smoke(centroid_export):
    """Centroid adapter populates ``pred_centroids`` + ``pred_centroid_values``."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(centroid_export, device="cpu")
    images = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    out = outputs_list[0]
    assert out.pred_centroids is not None
    assert out.pred_centroid_values is not None
    assert out.pred_centroids.shape == (1, 4, 2)  # max_instances=4
    assert out.pred_centroid_values.shape == (1, 4)
    # No keypoints emitted by centroid-only adapter (centroids are the prediction).
    assert out.pred_keypoints is None


def test_from_export_dir_centered_instance_builds_predictor(centered_instance_export):
    """Centered-instance export → :class:`ExportedCenteredInstanceLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import ExportedCenteredInstanceLayer

    predictor = Predictor.from_export_dir(centered_instance_export, device="cpu")
    assert isinstance(predictor.layer, ExportedCenteredInstanceLayer)


def test_from_export_dir_centered_instance_predict_smoke(centered_instance_export):
    """Centered-instance adapter populates ``pred_keypoints`` per crop."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(centered_instance_export, device="cpu")
    crops = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=crops, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    out = outputs_list[0]
    assert out.pred_keypoints is not None
    # (B_crops, 1 instance per crop, n_nodes, 2) → (1, 1, 2, 2).
    assert out.pred_keypoints.shape == (1, 1, 2, 2)


def test_from_export_dir_topdown_builds_predictor(topdown_export):
    """Top-down combined export → :class:`ExportedTopDownLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import ExportedTopDownLayer

    predictor = Predictor.from_export_dir(topdown_export, device="cpu")
    assert isinstance(predictor.layer, ExportedTopDownLayer)


def test_from_export_dir_topdown_predict_smoke(topdown_export):
    """Top-down adapter populates centroids + keypoints + instance_valid."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(topdown_export, device="cpu")
    images = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    out = outputs_list[0]
    assert out.pred_keypoints is not None
    assert out.pred_centroids is not None
    assert out.pred_centroid_values is not None
    assert out.pred_peak_values is not None
    assert out.instance_valid is not None
    # max_instances=3, n_nodes=2 → keypoints (1, 3, 2, 2), centroids (1, 3, 2).
    assert out.pred_keypoints.shape == (1, 3, 2, 2)
    assert out.pred_centroids.shape == (1, 3, 2)


def test_centroid_adapter_nan_pads_invalid_slots(centroid_export):
    """Invalid centroid slots (zero-confidence) are NaN'd per Outputs convention."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    # All-zero image → conv outputs ~0 confmap, all topk values <= 0 → all invalid.
    predictor = Predictor.from_export_dir(centroid_export, device="cpu")
    images = np.zeros((1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    out = predictor.predict(provider, make_labels=False)[0]
    # Some slots may be invalid. Wherever instance_valid is False, the
    # corresponding centroid + value must be NaN.
    valid = out.instance_valid[0]
    invalid_mask = ~valid
    if invalid_mask.any():
        invalid_centroids = out.pred_centroids[0][invalid_mask]
        invalid_vals = out.pred_centroid_values[0][invalid_mask]
        assert torch.isnan(invalid_centroids).all()
        assert torch.isnan(invalid_vals).all()


# ──────────────────────────────────────────────────────────────────────
# Bottom-up synthetic export (PR 20)
# ──────────────────────────────────────────────────────────────────────


def _export_tiny_bottomup(
    export_dir: Path,
    n_nodes: int = 2,
    n_edges: int = 1,
    k: int = 4,
) -> Path:
    """Export an ONNX that emits the bottom-up wrapper output schema.

    Outputs:
        peaks (B, n_nodes, k, 2),
        peak_vals (B, n_nodes, k),
        peak_mask (B, n_nodes, k) bool,
        line_scores (B, n_edges, k*k),
        candidate_mask (B, n_edges, k*k) bool.

    The model is intentionally simple — it picks topk peaks per node
    over a 1-conv confmap and assigns synthesized line scores to every
    candidate pair (so grouping has something to chew on).
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, n_nodes, kernel_size=3, padding=1)
            self.k = k
            self.n_nodes = n_nodes
            self.n_edges = n_edges

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())  # (B, n_nodes, H, W)
            B, _C, H, W = cms.shape
            flat = cms.view(B, self.n_nodes, -1)
            vals, idx = flat.topk(self.k, dim=-1)  # (B, n_nodes, k)
            row = (idx // W).float()
            col = (idx % W).float()
            peaks = torch.stack([col, row], dim=-1)  # (B, n_nodes, k, 2)
            peak_mask = vals > 0.0  # (B, n_nodes, k) bool
            # All candidate pairs get a moderate score; mask follows from peak_mask.
            kk = self.k * self.k
            line_scores = torch.full((B, self.n_edges, kk), 0.5)
            # Build candidate_mask = src_mask & dst_mask (matches wrapper).
            edge_src = torch.tensor([0])
            edge_dst = torch.tensor([1])
            src_mask = peak_mask[:, edge_src, :]  # (B, n_edges, k)
            dst_mask = peak_mask[:, edge_dst, :]
            candidate_mask = (
                src_mask.unsqueeze(3).expand(-1, -1, -1, self.k)
                & dst_mask.unsqueeze(2).expand(-1, -1, self.k, -1)
            ).reshape(B, self.n_edges, kk)
            return peaks, vals, peak_mask, line_scores, candidate_mask

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=[
            "peaks",
            "peak_vals",
            "peak_mask",
            "line_scores",
            "candidate_mask",
        ],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _write_metadata_bottomup(
    export_dir: Path,
    n_nodes: int = 2,
    edges: list = None,
    max_peaks_per_node: int = 4,
    input_scale: float = 1.0,
) -> Path:
    """ExportMetadata for bottom-up — adds ``edge_inds`` and ``max_peaks_per_node``."""
    if edges is None:
        edges = [(0, 1)]
    meta = {
        "sleap_nn_version": "0.0.0",
        "export_timestamp": "2026-01-01T00:00:00",
        "export_format": "onnx",
        "model_type": "bottomup",
        "model_name": "test_bottomup",
        "checkpoint_path": "/tmp/fake.ckpt",
        "backbone": "unet",
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "node_names": [f"n{i}" for i in range(n_nodes)],
        "edge_inds": [list(e) for e in edges],
        "input_scale": input_scale,
        "input_channels": 1,
        "output_stride": 1,
        "max_peaks_per_node": max_peaks_per_node,
        "peak_threshold": 0.0,
    }
    path = export_dir / "export_metadata.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def bottomup_export(tmp_path):
    """Export dir for a bottom-up model: model.onnx + metadata."""
    export_dir = tmp_path / "bottomup_export"
    export_dir.mkdir()
    _export_tiny_bottomup(export_dir, n_nodes=2, n_edges=1, k=4)
    _write_metadata_bottomup(
        export_dir, n_nodes=2, edges=[(0, 1)], max_peaks_per_node=4
    )
    return export_dir


def test_from_export_dir_bottomup_builds_predictor(bottomup_export):
    """Bottom-up export → :class:`ExportedBottomUpLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import ExportedBottomUpLayer

    predictor = Predictor.from_export_dir(bottomup_export, device="cpu")
    assert isinstance(predictor.layer, ExportedBottomUpLayer)
    assert predictor.layer.max_peaks_per_node == 4
    assert predictor.layer.node_names == ["n0", "n1"]
    assert predictor.layer.edge_inds == [(0, 1)]


def test_from_export_dir_bottomup_predict_smoke(bottomup_export):
    """Bottom-up adapter runs the full GPU→CPU pipeline.

    Validates the schema-translation glue (fixed-shape wrapper output →
    variable-length ScoredBatch → group_scored_batch → Outputs).
    """
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(
        bottomup_export, device="cpu", min_line_scores=-1.0
    )
    images = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    outputs_list = predictor.predict(provider, make_labels=False)
    out = outputs_list[0]
    # Bottom-up always populates pred_keypoints (NaN-padded if no
    # instances assembled).
    assert out.pred_keypoints is not None
    assert out.pred_keypoints.ndim == 4  # (B, I, N, 2)
    assert out.pred_keypoints.shape[0] == 1
    assert out.pred_keypoints.shape[2] == 2  # n_nodes


def test_from_export_dir_bottomup_forwards_grouping_kwargs(bottomup_export):
    """``min_line_scores`` / ``min_instance_peaks`` / ``max_instances`` flow into the layer."""
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_export_dir(
        bottomup_export,
        device="cpu",
        max_instances=2,
        min_instance_peaks=1,
        min_line_scores=0.1,
    )
    layer = predictor.layer
    assert layer.max_instances == 2
    assert layer.min_instance_peaks == 1
    assert abs(layer.min_line_scores - 0.1) < 1e-9


def test_from_export_dir_bottomup_missing_max_peaks_per_node_raises(tmp_path):
    """Missing ``max_peaks_per_node`` in metadata ⇒ ``ValueError``."""
    from sleap_nn.inference.predictor import Predictor

    export_dir = tmp_path / "bad_bottomup"
    export_dir.mkdir()
    _export_tiny_bottomup(export_dir)
    # Write metadata WITHOUT max_peaks_per_node.
    meta = {
        "sleap_nn_version": "0.0.0",
        "export_timestamp": "2026-01-01T00:00:00",
        "export_format": "onnx",
        "model_type": "bottomup",
        "model_name": "test",
        "checkpoint_path": "/tmp/fake.ckpt",
        "backbone": "unet",
        "n_nodes": 2,
        "n_edges": 1,
        "node_names": ["n0", "n1"],
        "edge_inds": [[0, 1]],
        "input_scale": 1.0,
        "input_channels": 1,
        "output_stride": 1,
    }
    (export_dir / "export_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="max_peaks_per_node"):
        Predictor.from_export_dir(export_dir, device="cpu")


# ──────────────────────────────────────────────────────────────────────
# Multi-class synthetic exports (PR 21)
# ──────────────────────────────────────────────────────────────────────


def _export_tiny_multiclass_topdown(
    export_dir: Path,
    max_instances: int = 3,
    n_nodes: int = 2,
    n_classes: int = 3,
) -> Path:
    """Export ONNX matching ``TopDownMultiClassCombinedONNXWrapper`` schema.

    Outputs: centroids, centroid_vals, peaks, peak_vals, class_logits,
    instance_valid.
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.cls = nn.Conv2d(1, n_classes, kernel_size=1)
            self.max_instances = max_instances
            self.n_nodes = n_nodes
            self.n_classes = n_classes

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())
            B, _C, H, W = cms.shape
            flat = cms.view(B, -1)
            vals, idx = flat.topk(self.max_instances, dim=-1)
            row = (idx // W).float()
            col = (idx % W).float()
            centroids = torch.stack([col, row], dim=-1)
            instance_valid = vals > 0.0
            peaks = centroids.unsqueeze(2).expand(-1, -1, self.n_nodes, -1).clone()
            peak_vals = vals.unsqueeze(-1).expand(-1, -1, self.n_nodes).clone()
            # Synthesize class logits per instance — use a different
            # constant offset per (instance, class) so Hungarian matching
            # has a deterministic preferred assignment.
            base = torch.linspace(0, 1, self.max_instances * self.n_classes).reshape(
                1, self.max_instances, self.n_classes
            )
            class_logits = base.expand(B, -1, -1).clone()
            return centroids, vals, peaks, peak_vals, class_logits, instance_valid

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=[
            "centroids",
            "centroid_vals",
            "peaks",
            "peak_vals",
            "class_logits",
            "instance_valid",
        ],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _export_tiny_multiclass_bottomup(
    export_dir: Path,
    n_nodes: int = 2,
    k: int = 3,
    n_classes: int = 2,
) -> Path:
    """Export ONNX matching ``BottomUpMultiClassONNXWrapper`` schema.

    Outputs: peaks, peak_vals, peak_mask, class_probs.
    """

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, n_nodes, kernel_size=3, padding=1)
            self.k = k
            self.n_nodes = n_nodes
            self.n_classes = n_classes

        def forward(self, x: torch.Tensor):
            cms = self.conv(x.float())
            B, _C, H, W = cms.shape
            flat = cms.view(B, self.n_nodes, -1)
            vals, idx = flat.topk(self.k, dim=-1)
            row = (idx // W).float()
            col = (idx % W).float()
            peaks = torch.stack([col, row], dim=-1)
            peak_mask = vals > 0.0
            # Synthesize per-peak class probabilities — make peak k=0
            # prefer class 0, k=1 prefer class 1, etc., so Hungarian
            # matching has a deterministic answer per (sample, node).
            base = torch.eye(max(self.k, self.n_classes))[: self.k, : self.n_classes]
            class_probs = base.expand(B, self.n_nodes, -1, -1).clone()
            return peaks, vals, peak_mask, class_probs

    onnx_path = export_dir / "model.onnx"
    torch.onnx.export(
        _Tiny(),
        (torch.zeros(1, 1, 16, 16),),
        str(onnx_path),
        input_names=["image"],
        output_names=["peaks", "peak_vals", "peak_mask", "class_probs"],
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _write_metadata_multiclass(
    export_dir: Path,
    model_type: str,
    n_nodes: int = 2,
    n_classes: int = 2,
    max_peaks_per_node: int | None = None,
    edges: list | None = None,
    input_scale: float = 1.0,
) -> Path:
    """ExportMetadata for multiclass — adds ``n_classes``."""
    meta = {
        "sleap_nn_version": "0.0.0",
        "export_timestamp": "2026-01-01T00:00:00",
        "export_format": "onnx",
        "model_type": model_type,
        "model_name": "test_multiclass",
        "checkpoint_path": "/tmp/fake.ckpt",
        "backbone": "unet",
        "n_nodes": n_nodes,
        "n_edges": len(edges) if edges else 0,
        "node_names": [f"n{i}" for i in range(n_nodes)],
        "edge_inds": [list(e) for e in edges] if edges else [],
        "input_scale": input_scale,
        "input_channels": 1,
        "output_stride": 1,
        "n_classes": n_classes,
        "class_names": [f"class_{i}" for i in range(n_classes)],
    }
    if max_peaks_per_node is not None:
        meta["max_peaks_per_node"] = max_peaks_per_node
    path = export_dir / "export_metadata.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def multiclass_topdown_export(tmp_path):
    """Export dir for a multi-class top-down model."""
    export_dir = tmp_path / "mc_topdown_export"
    export_dir.mkdir()
    _export_tiny_multiclass_topdown(export_dir, max_instances=3, n_nodes=2, n_classes=3)
    _write_metadata_multiclass(
        export_dir, model_type="multi_class_topdown", n_nodes=2, n_classes=3
    )
    return export_dir


@pytest.fixture
def multiclass_bottomup_export(tmp_path):
    """Export dir for a multi-class bottom-up model."""
    export_dir = tmp_path / "mc_bottomup_export"
    export_dir.mkdir()
    _export_tiny_multiclass_bottomup(export_dir, n_nodes=2, k=3, n_classes=2)
    _write_metadata_multiclass(
        export_dir,
        model_type="multi_class_bottomup",
        n_nodes=2,
        n_classes=2,
        max_peaks_per_node=3,
    )
    return export_dir


def test_from_export_dir_multiclass_topdown_builds_predictor(
    multiclass_topdown_export,
):
    """multi_class_topdown export → :class:`ExportedTopDownMultiClassLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import ExportedTopDownMultiClassLayer

    predictor = Predictor.from_export_dir(multiclass_topdown_export, device="cpu")
    assert isinstance(predictor.layer, ExportedTopDownMultiClassLayer)
    assert predictor.layer.n_classes == 3


def test_from_export_dir_multiclass_topdown_predict_smoke(
    multiclass_topdown_export,
):
    """Multi-class top-down adapter populates fields with class-ordered slots."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(multiclass_topdown_export, device="cpu")
    images = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    out = predictor.predict(provider, make_labels=False)[0]
    assert out.pred_keypoints is not None
    # I = n_classes = 3
    assert out.pred_keypoints.shape == (1, 3, 2, 2)
    assert out.pred_centroids.shape == (1, 3, 2)
    assert out.pred_class_probs is not None
    assert out.pred_class_probs.shape == (1, 3, 3)
    assert out.instance_valid.shape == (1, 3)


def test_from_export_dir_multiclass_bottomup_builds_predictor(
    multiclass_bottomup_export,
):
    """multi_class_bottomup export → :class:`ExportedBottomUpMultiClassLayer`."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.exported import (
        ExportedBottomUpMultiClassLayer,
    )

    predictor = Predictor.from_export_dir(multiclass_bottomup_export, device="cpu")
    assert isinstance(predictor.layer, ExportedBottomUpMultiClassLayer)
    assert predictor.layer.n_nodes == 2
    assert predictor.layer.n_classes == 2


def test_from_export_dir_multiclass_bottomup_predict_smoke(
    multiclass_bottomup_export,
):
    """Multi-class bottom-up adapter groups peaks by class via Hungarian matching."""
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import NumpyProvider

    predictor = Predictor.from_export_dir(multiclass_bottomup_export, device="cpu")
    images = np.random.randint(0, 256, (1, 1, 16, 16), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=1)

    out = predictor.predict(provider, make_labels=False)[0]
    assert out.pred_keypoints is not None
    # I = n_classes = 2, N = n_nodes = 2.
    assert out.pred_keypoints.shape == (1, 2, 2, 2)
    assert out.pred_class_vectors is not None
    # (B, I, N, C) per Outputs convention.
    assert out.pred_class_vectors.shape == (1, 2, 2, 2)
    assert out.instance_valid.shape == (1, 2)


def test_from_export_dir_multiclass_topdown_missing_n_classes_raises(tmp_path):
    """Missing ``n_classes`` in metadata ⇒ ``ValueError``."""
    from sleap_nn.inference.predictor import Predictor

    export_dir = tmp_path / "bad_mc_topdown"
    export_dir.mkdir()
    _export_tiny_multiclass_topdown(export_dir)
    meta = {
        "sleap_nn_version": "0.0.0",
        "export_timestamp": "2026-01-01T00:00:00",
        "export_format": "onnx",
        "model_type": "multi_class_topdown",
        "model_name": "test",
        "checkpoint_path": "/tmp/fake.ckpt",
        "backbone": "unet",
        "n_nodes": 2,
        "n_edges": 0,
        "node_names": ["n0", "n1"],
        "edge_inds": [],
        "input_scale": 1.0,
        "input_channels": 1,
        "output_stride": 1,
    }
    (export_dir / "export_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="n_classes"):
        Predictor.from_export_dir(export_dir, device="cpu")
