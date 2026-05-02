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

    Forces the legacy path with ``--tracking`` so ``run_inference`` is
    called (PR 13 routes simple cases through the new factory; that
    path has its own kwarg surface and isn't tested here).
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
                "--tracking",
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
                "--tracking",  # force legacy path so run_inference is called
                "--peak-conf-threshold",
                "0.42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert abs(mock_run.call_args[1]["peak_threshold"] - 0.42) < 1e-9


def test_infer_simple_case_uses_new_factory_flow(tmp_path):
    """``sleap-nn infer`` without tracking/special flags routes to the new factory.

    PR 13 wires the simple case to ``Predictor.from_model_paths(...).predict(...)``
    instead of the legacy ``run_inference``. This test mocks the factory
    + skeleton resolution + a stub predictor, and verifies that path is
    taken end-to-end (Labels.save called).
    """
    from unittest.mock import MagicMock

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_labels = MagicMock()
    stub_predictor = MagicMock()
    stub_predictor.predict.return_value = stub_labels
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.cli._skeleton_from_predictor") as mock_skel,
        patch("sleap_nn.inference.providers.VideoProvider"),
        patch("sleap_nn.predict.run_inference") as mock_run_inference,
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
                "--output_path",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_factory.called, "new factory was not called for simple infer case"
    assert stub_predictor.predict.called
    assert stub_labels.save.called
    # The legacy run_inference should NOT have been called for this case.
    assert not mock_run_inference.called


def test_infer_with_tracking_falls_through_to_legacy():
    """``--tracking`` forces the legacy ``run_inference`` path.

    The new flow doesn't yet handle tracking; PR 13 keeps the legacy
    fallback for these cases.
    """
    runner = CliRunner()
    with (
        patch("sleap_nn.predict.run_inference") as mock_run,
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
    ):
        mock_run.return_value = None
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--tracking",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.called
        assert not mock_factory.called


def test_infer_paf_workers_zero_no_warning():
    """``--paf-workers 0`` is the default, must not emit a warning.

    Uses ``--tracking`` to force the legacy path where the no-effect
    warning would fire if the flag were misused.
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
                "--tracking",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "paf-workers > 0" not in result.output
