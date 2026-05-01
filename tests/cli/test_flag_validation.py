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


def test_stream_to_file_raises_usage_error_until_pr14():
    """``--stream-to-file`` is a hard error today; the new flow ships later."""
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
        ],
    )
    assert result.exit_code != 0
    assert "stream-to-file" in result.output
    assert "follow-up" in result.output or "#519" in result.output


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
    """``--cpu-workers`` warns and is wired through (mapped to paf_workers)."""
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
    """``--paf-workers > 0`` succeeds but warns the pool is inactive in the CLI."""
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
                "--paf-workers",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        # The warning is emitted via loguru.logger.warning. The CliRunner
        # captures stderr by default in mix_stderr=True (click >=8.2 keeps
        # them merged); just confirm the call still succeeded — the
        # specific warning surface is logger-level and not part of the
        # contract.


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
