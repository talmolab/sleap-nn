"""Tests for deprecated alias commands (PR 10 of #508 / #518).

``sleap-nn track`` is now a deprecated alias for ``sleap-nn infer``;
emits a ``DeprecationWarning`` once and otherwise reaches the same
implementation. ``sleap-nn predict`` is *not* yet aliased (deferred —
the existing top-level ``predict`` runs inference on exported models
and rerouting it requires the export-group refactor).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_track_emits_deprecation_warning():
    """``sleap-nn track`` runs the legacy flow but emits a DeprecationWarning."""
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference") as mock_run:
        mock_run.return_value = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = runner.invoke(
                cli,
                [
                    "track",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_run.called
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any(
            "sleap-nn track" in str(d.message) and "infer" in str(d.message)
            for d in deprecations
        ), [str(d.message) for d in deprecations]


def test_track_and_infer_reach_same_run_inference_kwargs():
    """``track`` and ``infer`` produce identical kwargs to ``run_inference``."""
    runner = CliRunner()
    args_common = [
        "--data_path",
        "/fake/path.mp4",
        "--model_paths",
        "/fake/model",
        "--device",
        "cpu",
        "--batch_size",
        "2",
        "--peak_threshold",
        "0.15",
    ]

    with patch("sleap_nn.predict.run_inference") as mock_run:
        mock_run.return_value = None
        runner.invoke(cli, ["infer"] + args_common)
        infer_kwargs = dict(mock_run.call_args[1])

    with patch("sleap_nn.predict.run_inference") as mock_run:
        mock_run.return_value = None
        # Suppress the DeprecationWarning during the test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            runner.invoke(cli, ["track"] + args_common)
        track_kwargs = dict(mock_run.call_args[1])

    # Both should reach the same kwargs (modulo the new-flag stripping
    # already done by both paths).
    assert infer_kwargs == track_kwargs


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
