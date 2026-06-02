"""Tests for command aliases and routing (PR 10 of #508 / #518).

``sleap-nn track`` uses the legacy ``run_inference`` pipeline.
``sleap-nn predict`` uses the new ``Predictor``-based pipeline.
``sleap-nn infer`` is a deprecated alias that still works but emits a warning.
``sleap-nn export model`` exports trained models.
``sleap-nn export predict`` runs inference on exported ONNX/TRT models.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_track_routes_to_legacy_run_inference():
    """``sleap-nn track`` routes to the legacy ``run_inference`` pipeline."""
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference", return_value=None) as mock_run:
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


def test_export_predict_under_export_group():
    """``sleap-nn export predict --help`` shows EXPORT_DIR/VIDEO_PATH args.

    The export predict command lives under the ``export`` group and handles
    inference on exported ONNX/TRT models.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "predict", "--help"])
    assert result.exit_code == 0, result.output
    # Marker that this is the export-trained predict, not the inference predict.
    assert "EXPORT_DIR" in result.output or "VIDEO_PATH" in result.output


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
