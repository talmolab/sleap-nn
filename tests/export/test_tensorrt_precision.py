"""TensorRT precision options and backend guards (reviews of #858 / #860)."""

import json

import pytest
import torch
from click.testing import CliRunner

from sleap_nn.export.exporters.tensorrt_exporter import (
    BACKBONE_SCOPE,
    check_precision_support,
)


def test_check_precision_support():
    for p in ("fp32", "tf32", "fp16", "bf16"):  # every precision on 10.x and 11
        check_precision_support(p, "10.16.1.11", (8, 9))
        check_precision_support(p, "11.5.0.63", (12, 0))
    check_precision_support("tf32", "11.5.0.63", None)
    with pytest.raises(RuntimeError, match="compute capability 8.0"):
        check_precision_support("bf16", "10.16.1.11", (7, 5))
    with pytest.raises(RuntimeError, match="compute capability 8.0"):
        check_precision_support("bf16", "11.5.0.63", (7, 5))
    check_precision_support("fp16", "10.16.1.11", (7, 5))
    with pytest.raises(ValueError, match="Unknown TensorRT precision"):
        check_precision_support("int8", "10.16.1.11", (8, 9))


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


def test_strongly_typed_option_reaches_the_build(
    monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path
):
    pytest.importorskip("onnx")
    ckpt = minimal_instance_centered_instance_ckpt
    result, seen, _ = _export(monkeypatch, ckpt, tmp_path, "--precision", "bf16")
    assert result.exit_code == 0, result.output
    assert seen["strongly_typed"] is False  # default: TensorRT 10 builder flags
    result, seen, _ = _export(
        monkeypatch, ckpt, tmp_path, "--precision", "bf16", "--strongly-typed"
    )
    assert result.exit_code == 0, result.output
    assert seen["strongly_typed"] is True
