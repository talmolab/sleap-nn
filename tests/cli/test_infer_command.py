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
    """Every flag wired into ``track`` is accepted by ``infer`` too.

    Uses ``--gui`` to force the legacy path so we can read the kwargs
    directly off the ``run_inference`` mock; PR 15 routes
    ``--tracking`` + ``--filter_*`` through the new factory.
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
                "--gui",
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

    Forces the legacy path with ``--gui`` so ``run_inference`` is
    called (PR 13/14 route simple+tracking cases through the new
    factory; that path has its own kwarg surface and isn't tested here).
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
                "--gui",
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
                "--gui",  # force legacy path so run_inference is called
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


def test_infer_with_tracking_uses_new_factory_flow(tmp_path):
    """``--tracking`` now routes through the new factory (PR 14).

    The factory receives a ``tracker_config`` and the legacy
    ``run_inference`` is NOT called.
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
        patch("sleap_nn.predict.run_inference") as mock_run,
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
                "--tracking",
                "--tracking_window_size",
                "7",
                "--candidates_method",
                "local_queues",
                "--max_tracks",
                "3",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_factory.called
        assert not mock_run.called
        cfg = mock_factory.call_args[1]["tracker_config"]
        assert cfg.window_size == 7
        assert cfg.candidates_method == "local_queues"
        assert cfg.max_tracks == 3


def test_infer_with_tracking_plus_filter_uses_new_factory_flow(tmp_path):
    """``--tracking`` + ``--filter_*`` now route through the new factory (PR 15).

    The factory should receive both a ``tracker_config`` and a
    ``filter_config`` reflecting the CLI flags.
    """
    from unittest.mock import MagicMock

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict.return_value = MagicMock()
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.cli._skeleton_from_predictor") as mock_skel,
        patch("sleap_nn.inference.providers.VideoProvider"),
        patch("sleap_nn.predict.run_inference") as mock_run,
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
                "--tracking",
                "--filter_overlapping",
                "--filter_overlapping_method",
                "oks",
                "--filter_overlapping_threshold",
                "0.5",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_factory.called
        assert not mock_run.called
        kw = mock_factory.call_args[1]
        assert kw["tracker_config"] is not None
        assert kw["filter_config"] is not None
        fc = kw["filter_config"]
        assert fc.overlapping is True
        assert fc.overlapping_method == "oks"
        assert abs(fc.overlapping_threshold - 0.5) < 1e-9


def test_infer_with_filter_flags_builds_filter_config(tmp_path):
    """``--filter_min_visible_nodes`` etc. build a ``FilterConfig`` for the new flow."""
    from unittest.mock import MagicMock

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict.return_value = MagicMock()
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.cli._skeleton_from_predictor"),
        patch("sleap_nn.inference.providers.VideoProvider"),
        patch("sleap_nn.predict.run_inference"),
    ):
        mock_factory.return_value = stub_predictor
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--filter_min_visible_nodes",
                "3",
                "--filter_min_mean_node_score",
                "0.4",
                "--filter_min_instance_score",
                "0.6",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        fc = mock_factory.call_args[1]["filter_config"]
        assert fc.min_visible_nodes == 3
        assert abs(fc.min_mean_node_score - 0.4) < 1e-9
        assert abs(fc.min_instance_score - 0.6) < 1e-9


def test_infer_no_empty_frames_passes_clean_flag(tmp_path):
    """``--no_empty_frames`` propagates as ``clean_empty_frames=True`` to predict()."""
    from unittest.mock import MagicMock

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict.return_value = MagicMock()
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.cli._skeleton_from_predictor"),
        patch("sleap_nn.inference.providers.VideoProvider"),
        patch("sleap_nn.predict.run_inference"),
    ):
        mock_factory.return_value = stub_predictor
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--no_empty_frames",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        kw = stub_predictor.predict.call_args[1]
        assert kw["clean_empty_frames"] is True


def test_infer_only_suggested_frames_routes_to_new_flow(tmp_path):
    """``--only_suggested_frames`` goes through the new flow + LabelsProvider."""
    from unittest.mock import MagicMock

    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict.return_value = MagicMock()
    with (
        patch("sleap_nn.inference.factory.from_model_paths") as mock_factory,
        patch("sleap_nn.inference.providers.LabelsProvider") as mock_provider,
        patch("sleap_io.load_slp") as mock_load,
        patch("sleap_nn.predict.run_inference") as mock_run,
    ):
        mock_factory.return_value = stub_predictor
        mock_load.return_value = MagicMock(skeletons=[object()], videos=[object()])
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.slp",
                "--model_paths",
                "/fake/model",
                "--only_suggested_frames",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_factory.called
        assert not mock_run.called
        provider_kwargs = mock_provider.call_args[1]
        assert provider_kwargs["only_suggested_frames"] is True


def test_infer_paf_workers_zero_no_warning():
    """``--paf-workers 0`` is the default, must not emit a warning.

    Uses ``--gui`` to force the legacy path where the no-effect
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
                "--gui",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "paf-workers > 0" not in result.output
