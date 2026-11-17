"""TensorRT batch profile: `--opt-batch-size` and chunking above an engine's max batch."""

import pytest
import torch
from click.testing import CliRunner


def test_run_in_batch_chunks():
    from sleap_nn.inference.layers.backends.tensorrt_backend import (
        run_in_batch_chunks,
    )

    calls = []

    def run(x):
        calls.append(x.shape[0])
        return {"a": x * 2, "b": x.sum(dim=(1, 2, 3))}

    x = torch.arange(10 * 2 * 3 * 3, dtype=torch.float32).reshape(10, 2, 3, 3)
    whole = run_in_batch_chunks(run, x, None)
    assert calls == [10]
    calls.clear()
    chunked = run_in_batch_chunks(run, x, 4)
    assert calls == [4, 4, 2]
    for k in whole:
        torch.testing.assert_close(chunked[k], whole[k])
    calls.clear()
    run_in_batch_chunks(run, x, 10)
    assert calls == [10]


def _export_profiles(monkeypatch, ckpt, tmp_path, *extra):
    """Run `sleap-nn export --format tensorrt` with the engine build stubbed out."""
    import sleap_nn.export.exporters as exporters

    seen = {}

    def fake_export_to_tensorrt(model, save_path, **kw):
        seen.update(kw)
        open(save_path, "wb").close()
        return save_path

    monkeypatch.setattr(exporters, "export_to_tensorrt", fake_export_to_tensorrt)
    from sleap_nn.export.cli import export

    result = CliRunner().invoke(
        export, [str(ckpt), "-o", str(tmp_path / "x"), "--format", "tensorrt", *extra]
    )
    return result, seen


def test_opt_batch_size(monkeypatch, minimal_instance_centered_instance_ckpt, tmp_path):
    pytest.importorskip("onnx")
    ckpt = minimal_instance_centered_instance_ckpt
    result, seen = _export_profiles(monkeypatch, ckpt, tmp_path)
    assert result.exit_code == 0, result.output
    assert seen["opt_shape"][0] == 1  # unchanged default
    assert seen["max_shape"][0] == 8

    result, seen = _export_profiles(
        monkeypatch, ckpt, tmp_path, "--max-batch-size", "64", "--opt-batch-size", "32"
    )
    assert result.exit_code == 0, result.output
    assert seen["opt_shape"][0] == 32 and seen["max_shape"][0] == 64
    assert tuple(seen["opt_shape"][1:]) == tuple(seen["min_shape"][1:])

    result, _ = _export_profiles(
        monkeypatch, ckpt, tmp_path, "--max-batch-size", "8", "--opt-batch-size", "16"
    )
    assert result.exit_code != 0 and "--opt-batch-size" in result.output
