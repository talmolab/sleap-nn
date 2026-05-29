"""Tests for command aliases and routing (PR 10 of #508 / #518).

``sleap-nn track`` uses the legacy ``run_inference`` pipeline.
``sleap-nn infer`` uses the new ``Predictor``-based pipeline.
``sleap-nn predict`` runs inference on exported ONNX/TRT models.
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


def test_infer_routes_to_new_predict():
    """``sleap-nn infer`` routes to the new ``predict()`` pipeline."""
    from unittest.mock import MagicMock

    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ) as mock_predict:
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
    mock_predict.assert_called_once()


def test_export_predict_top_level_still_works():
    """The legacy top-level ``sleap-nn predict`` (export-trained) is intact.

    PR 10 explicitly does NOT remap top-level ``predict`` — that's a
    follow-up. ``sleap-nn predict --help`` should still render the
    export-trained predict command (with EXPORT_DIR + VIDEO_PATH args).
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0, result.output
    # Marker that this is still the export-trained predict, not infer.
    assert "EXPORT_DIR" in result.output or "VIDEO_PATH" in result.output
