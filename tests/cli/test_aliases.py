"""Tests for command aliases and routing (PR 10 of #508 / #518).

``sleap-nn track`` uses the legacy ``run_inference`` pipeline.
``sleap-nn predict`` uses the new ``Predictor``-based pipeline and accepts both
trained checkpoints and exported ONNX/TRT model directories.
``sleap-nn infer`` is a deprecated alias that still works but emits a warning.
``sleap-nn export`` exports trained models to ONNX/TensorRT.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_track_routes_to_legacy_run_inference():
    """``sleap-nn track`` routes to the legacy ``run_inference`` pipeline."""
    runner = CliRunner()
    with patch("sleap_nn.legacy_predict.run_inference", return_value=None) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--device",
                "cpu",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()


def test_predict_routes_to_new_predict():
    """``sleap-nn predict`` routes to the new ``predict()`` pipeline."""
    from unittest.mock import MagicMock

    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--device",
                "cpu",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_predict.assert_called_once()


def test_export_is_standalone_command():
    """``sleap-nn export`` is a standalone command taking MODEL_PATHS (no subcommands)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--help"])
    assert result.exit_code == 0, result.output
    assert "MODEL_PATHS" in result.output


def test_export_predict_subcommand_removed():
    """``sleap-nn export predict`` is gone; predict now handles exported models."""
    runner = CliRunner()
    # `export` is standalone now: "predict" is parsed as a (nonexistent) MODEL_PATH
    # directory, so this fails validation rather than dispatching to a subcommand.
    result = runner.invoke(cli, ["export", "predict"])
    assert result.exit_code != 0
    assert "predict" in result.output.lower()


def test_predict_routes_exported_model_to_export_runtime(tmp_path):
    """A --model_paths dir containing model.onnx routes predict to the export runtime."""
    from unittest.mock import MagicMock

    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    # Marker artifact for auto-detection; the actual load is mocked.
    (export_dir / "model.onnx").write_bytes(b"")

    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/video.mp4",
                "--model_paths",
                str(export_dir),
                "--runtime",
                "onnx",
                "--device",
                "cpu",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_predict.assert_called_once()
    kwargs = mock_predict.call_args.kwargs
    assert kwargs.get("export_dir") == str(export_dir)
    assert kwargs.get("model_paths") is None
    assert kwargs.get("runtime") == "onnx"


def test_predict_checkpoint_dir_not_treated_as_export(tmp_path):
    """A --model_paths dir without model.onnx/.trt stays on the checkpoint path."""
    from unittest.mock import MagicMock

    ckpt_dir = tmp_path / "model"
    ckpt_dir.mkdir()
    (ckpt_dir / "best.ckpt").write_bytes(b"")

    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/video.mp4",
                "--model_paths",
                str(ckpt_dir),
                "--device",
                "cpu",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = mock_predict.call_args.kwargs
    assert kwargs.get("export_dir") is None
    assert kwargs.get("model_paths") == [str(ckpt_dir)]


def test_infer_shim_still_works():
    """``sleap-nn infer --help`` still works (exit code 0) via the deprecated shim."""
    runner = CliRunner()
    result = runner.invoke(cli, ["infer", "--help"])
    assert result.exit_code == 0, result.output


def test_infer_shim_emits_deprecation_warning():
    """``sleap-nn infer`` emits a DeprecationWarning pointing to ``predict``."""
    import warnings
    from unittest.mock import MagicMock

    runner = CliRunner()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ):
            result = runner.invoke(
                cli,
                [
                    "infer",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                    "--device",
                    "cpu",
                ],
            )
    assert result.exit_code == 0, result.output
    deprecation_msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("sleap-nn predict" in str(w.message) for w in deprecation_msgs)
