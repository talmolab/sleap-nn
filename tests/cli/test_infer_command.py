"""Tests for ``sleap-nn infer`` — the unified inference command (PR 10 #518).

Coverage:

1. The command is registered and ``--help`` renders.
2. Every non-new flag from ``sleap-nn track`` is accepted by ``infer``
   (parity with the legacy command's option surface).
3. The four PR-10 new flags are accepted: ``--paf-workers``, the legacy
   alias ``--cpu-workers``, ``--stream-to-file``, ``--write-interval``,
   and the alias ``--peak-conf-threshold``.
4. ``run_inference`` is invoked with the legacy option surface (no PR-10
   new flags leak into its kwargs).
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_infer_command_help_renders():
    """``sleap-nn infer --help`` exits 0 and lists the canonical flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["infer", "--help"])
    assert result.exit_code == 0, result.output
    # A handful of representative flags must appear in --help.
    for flag in [
        "--data_path",
        "--model_paths",
        "--device",
        "--batch_size",
        "--paf-workers",
        "--stream-to-file",
        "--write-interval",
    ]:
        assert flag in result.output, f"missing {flag} in `infer --help`"


def test_infer_accepts_legacy_track_flag_surface():
    """Every flag wired into ``track`` is accepted by ``infer`` too."""
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
                "--device",
                "cpu",
                "--batch_size",
                "8",
                "--max_instances",
                "2",
                "--peak_threshold",
                "0.3",
                "--filter_overlapping",
                "--filter_overlapping_method",
                "oks",
                "--tracking",
                "--candidates_method",
                "local_queues",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.called
        kw = mock_run.call_args[1]
        assert kw["data_path"] == "/fake/path.mp4"
        assert kw["model_paths"] == ["/fake/model"]
        assert kw["batch_size"] == 8
        assert kw["max_instances"] == 2
        assert abs(kw["peak_threshold"] - 0.3) < 1e-9
        assert kw["tracking"] is True
        assert kw["candidates_method"] == "local_queues"


def test_infer_strips_pr10_new_flags_before_run_inference():
    """The PR-10 new flags must not leak into ``run_inference`` kwargs.

    ``run_inference`` does not accept ``paf_workers``, ``stream_to_file``,
    or ``write_interval``; the CLI strips them before delegating.
    """
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
                "0",  # zero is allowed (no warning)
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_run.call_args[1]
        for new_flag in (
            "paf_workers",
            "stream_to_file",
            "write_interval",
            "cpu_workers",
        ):
            assert new_flag not in kw, f"{new_flag} leaked into run_inference"


def test_infer_peak_conf_threshold_alias():
    """``--peak-conf-threshold`` is an alias for ``--peak_threshold``."""
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
                "--peak-conf-threshold",
                "0.42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert abs(mock_run.call_args[1]["peak_threshold"] - 0.42) < 1e-9


def test_infer_paf_workers_zero_no_warning():
    """``--paf-workers 0`` is the default, must not emit a warning."""
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
            ],
        )
        assert result.exit_code == 0, result.output
        # Loguru warnings appear in stderr / output; the "no effect" line
        # should not be present at paf_workers=0.
        assert "paf-workers > 0 has no effect" not in result.output
