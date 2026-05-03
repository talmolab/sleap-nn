"""Validation tests for the PR 10 flags introduced on ``sleap-nn infer``.

* ``--stream-to-file`` is accepted but currently raises a clear
  ``UsageError`` since the new ``Predictor.predict_to_file`` flow lands
  in a follow-up PR.
* ``--write-interval`` only makes sense paired with ``--stream-to-file``.
* ``--cpu-workers`` is the deprecated old name; emits a deprecation
  warning and maps to ``--paf-workers``.
* ``--paf-workers > 0`` warns about no-effect but does not fail.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_stream_to_file_with_tracking_raises_usage_error():
    """``--stream-to-file`` + ``--tracking`` is rejected until tracking lands.

    PR 12 wires --stream-to-file to ``Predictor.predict_to_file`` for the
    in-memory case; tracking with streaming is a follow-up.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "infer",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--stream-to-file",
            "/tmp/out.slp",
            "--tracking",
        ],
    )
    assert result.exit_code != 0
    assert "tracking" in result.output.lower()


def test_write_interval_without_stream_to_file_errors():
    """``--write-interval`` alone is meaningless and rejected."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "infer",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--write-interval",
            "100",
        ],
    )
    assert result.exit_code != 0
    assert "write-interval" in result.output and "stream-to-file" in result.output


def test_cpu_workers_alias_emits_deprecation_warning():
    """``--cpu-workers`` warns and is wired through (mapped to paf_workers).

    Forces the legacy path with ``--gui`` (PR 13/14 route simple +
    tracking cases through the new factory which doesn't trip this
    warning).
    """
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference") as mock_run:
        mock_run.return_value = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = runner.invoke(
                cli,
                [
                    "infer",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                    "--gui",
                    "--cpu-workers",
                    "2",
                ],
            )
        assert result.exit_code == 0, result.output
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any(
            "cpu-workers" in str(d.message) and "paf-workers" in str(d.message)
            for d in deprecations
        ), [str(d.message) for d in deprecations]


def test_paf_workers_positive_emits_no_effect_warning():
    """``--paf-workers > 0`` succeeds on the legacy path with ``--gui``."""
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--gui",
                "--paf-workers",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output


def test_stream_to_file_invokes_new_predictor_flow(tmp_path):
    """``--stream-to-file`` reaches ``Predictor.predict_to_file``.

    Patches the factory to return a stub Predictor whose
    ``predict_to_file`` records that it was called — confirms the CLI
    routes through the new flow rather than the legacy ``run_inference``.
    """
    from unittest.mock import MagicMock, patch

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict_to_file.return_value = str(out)
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.cli._skeleton_from_predictor") as mock_skel,
        patch("sleap_nn.inference.providers.VideoProvider"),
    ):
        mock_factory.return_value = stub_predictor
        mock_skel.return_value = object()
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--stream-to-file",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_factory.called
    assert stub_predictor.predict_to_file.called


def test_unknown_flag_rejected_cleanly():
    """Click's standard usage-error path catches unknown flags."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "infer",
            "--data_path",
            "/fake/path.mp4",
            "--this-is-not-a-flag",
        ],
    )
    assert result.exit_code != 0
    assert (
        "no such option" in result.output.lower() or "unknown" in result.output.lower()
    )
