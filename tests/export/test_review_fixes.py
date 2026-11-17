"""Fixes from the reviews of #857, #858 and #860 (export side)."""

import json

import pytest
import torch
from click.testing import CliRunner

from sleap_nn.export.exporters.tensorrt_exporter import (
    BACKBONE_SCOPE,
    check_precision_support,
)


def test_check_precision_support():
    check_precision_support("fp16", "10.16.1.11", (8, 9))
    check_precision_support("bf16", "10.13.0", (8, 0))
    check_precision_support("fp32", "11.5.0.63", (12, 0))  # fp32 / tf32 build on 11
    check_precision_support("tf32", "11.5.0.63", None)
    for p in ("fp16", "bf16"):
        with pytest.raises(RuntimeError, match="TensorRT 11 removed weak typing"):
            check_precision_support(p, "11.5.0.63", (12, 0))
    with pytest.raises(RuntimeError, match="compute capability 8.0"):
        check_precision_support("bf16", "10.16.1.11", (7, 5))
    check_precision_support("fp16", "10.16.1.11", (7, 5))


def test_run_in_batch_chunks_needs_batch_first_outputs():
    from sleap_nn.inference.layers.backends.tensorrt_backend import (
        run_in_batch_chunks,
    )

    x = torch.zeros(10, 1, 2, 2)
    with pytest.raises(RuntimeError, match="not batch-first"):
        run_in_batch_chunks(lambda t: {"a": torch.zeros(3)}, x, 4)
    with pytest.raises(RuntimeError, match="not batch-first"):
        run_in_batch_chunks(lambda t: {"a": torch.tensor(1.0)}, x, 4)
    out = run_in_batch_chunks(lambda t: {"a": t[:, 0, 0, 0]}, x, 4)
    assert out["a"].shape == (10,)


def _export(monkeypatch, ckpt, tmp_path, *extra, fill=None):
    """`sleap-nn export --format tensorrt` with the engine build stubbed out."""
    import sleap_nn.export.exporters as exporters

    seen = {}

    def fake_export_to_tensorrt(model, save_path, **kw):
        seen.update(kw)
        if fill and kw.get("build_info") is not None:
            kw["build_info"].update(fill)
        open(save_path, "wb").close()
        return save_path

    monkeypatch.setattr(exporters, "export_to_tensorrt", fake_export_to_tensorrt)
    from sleap_nn.export.cli import export

    out = tmp_path / "x"
    result = CliRunner().invoke(
        export, [str(ckpt), "-o", str(out), "--format", "tensorrt", *extra]
    )
    return result, seen, out


def test_bf16_scope_follows_the_graph(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    """Peaks in the graph: fp32 outside the backbone; network-only: no such scope."""
    pytest.importorskip("onnx")
    ckpt = minimal_instance_centered_instance_ckpt
    result, seen, _ = _export(monkeypatch, ckpt, tmp_path, "--precision", "bf16")
    assert result.exit_code == 0, result.output
    assert seen["bf16_fp32_outside"] == BACKBONE_SCOPE
    result, seen, _ = _export(
        monkeypatch, ckpt, tmp_path, "--precision", "bf16", "--network-only"
    )
    assert result.exit_code == 0, result.output
    assert seen["bf16_fp32_outside"] is None


def test_metadata_records_the_precision_built(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    """An fp16 request that built fp32 is recorded as fp32."""
    pytest.importorskip("onnx")
    result, _, out = _export(
        monkeypatch,
        minimal_instance_centered_instance_ckpt,
        tmp_path,
        "--precision",
        "fp16",
        fill={"precision_built": "fp32"},
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((out / "model.trt.metadata.json").read_text())
    assert meta["precision"] == "fp32"
    result, _, out = _export(
        monkeypatch,
        minimal_instance_centered_instance_ckpt,
        tmp_path,
        "--precision",
        "bf16",
        fill={"precision_built": "bf16", "bf16_scope": "s", "bf16_fp32_layers": 3},
    )
    meta = json.loads((out / "model.trt.metadata.json").read_text())
    assert (meta["precision"], meta["bf16_scope"], meta["bf16_fp32_layers"]) == (
        "bf16",
        "s",
        3,
    )


def test_opt_batch_size_with_onnx_is_noted(
    minimal_instance_centered_instance_ckpt, tmp_path
):
    pytest.importorskip("onnx")
    from sleap_nn.export.cli import export

    out = tmp_path / "o"
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(out),
            "--format",
            "onnx",
            "--opt-batch-size",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignored for --format onnx" in result.output
    meta = json.loads((out / "export_metadata.json").read_text())
    assert meta["opt_batch_size"] == 4
    assert meta["opt_batch_size_applies_to"] == "tensorrt"


def test_network_only_traces_at_the_size_the_host_sends(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    """Crops are scaled and stride-padded on the host: graph and profile take that size."""
    onnx = pytest.importorskip("onnx")
    import math
    import shutil

    from omegaconf import OmegaConf

    from sleap_nn.export.cli import export

    ckpt = tmp_path / "ckpt"
    shutil.copytree(minimal_instance_centered_instance_ckpt, ckpt)
    cfg_path = ckpt / "training_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.data_config.preprocessing.scale = 0.5
    OmegaConf.save(cfg, cfg_path)
    crop = cfg.data_config.preprocessing.crop_size
    crop = int(crop[0]) if not isinstance(crop, int) else int(crop)
    bt = next(k for k, v in cfg.model_config.backbone_config.items() if v is not None)
    stride = int(cfg.model_config.backbone_config[bt]["max_stride"])
    want = math.ceil(crop * 0.5 / stride) * stride
    assert want < crop  # the case that used to fall below the engine's profile

    result, seen, out = _export(monkeypatch, ckpt, tmp_path, "--network-only")
    assert result.exit_code == 0, result.output
    for k in ("min_shape", "opt_shape", "max_shape"):
        assert tuple(seen[k][2:]) == (want, want), (k, seen[k])
    dims = onnx.load(str(out / "model.onnx")).graph.input[0].type.tensor_type.shape.dim
    assert all(d.dim_value in (0, want) for d in dims[2:])  # dynamic, or the host size
    meta = json.loads((out / "export_metadata.json").read_text())
    assert meta["input_scale"] == 0.5

    for flag in (["--crop-size", "64"], ["--input-scale", "1.0"]):
        result = CliRunner().invoke(
            export, [str(ckpt), "-o", str(tmp_path / "r"), "--network-only", *flag]
        )
        assert result.exit_code != 0 and "do not apply" in result.output
