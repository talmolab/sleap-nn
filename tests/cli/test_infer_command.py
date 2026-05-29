"""Tests for ``sleap-nn infer`` -- the unified inference command (PR 10 #518).

Coverage:

1. The command is registered and ``--help`` renders.
2. Every non-new flag from ``sleap-nn track`` is accepted by ``infer``
   (parity with the legacy command's option surface).
3. The four PR-10 new flags are accepted: ``--paf-workers``, the legacy
   alias ``--cpu-workers``, ``--stream-to-file``, ``--write-interval``,
   and the alias ``--peak-conf-threshold``.
4. ``predict()`` from ``sleap_nn.inference.run`` is invoked with the
   correct kwargs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

    Mocks ``predict()`` and asserts the right kwargs propagate through.
    """
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
        assert mock_predict.called
        kw = mock_predict.call_args[1]
        assert kw["batch_size"] == 8
        assert kw["max_instances"] == 2
        assert abs(kw["peak_threshold"] - 0.3) < 1e-9
        # Tracker config came through with the right knob.
        assert kw["tracker_config"].candidates_method == "local_queues"
        # Filter config came through.
        assert kw["filter_config"].overlapping is True
        assert kw["filter_config"].overlapping_method == "oks"


def test_infer_peak_conf_threshold_alias():
    """``--peak-conf-threshold`` is an alias for ``--peak_threshold``."""
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
                "--peak-conf-threshold",
                "0.42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert abs(mock_predict.call_args[1]["peak_threshold"] - 0.42) < 1e-9


def test_infer_simple_case_uses_predict(tmp_path):
    """``sleap-nn infer`` without tracking/special flags routes to ``predict()``.

    PR 27 wires the simple case to ``sleap_nn.inference.run.predict(...)``
    instead of the legacy ``run_inference``. This test mocks ``predict()``
    and verifies that path is taken end-to-end.
    """
    out = tmp_path / "out.slp"
    runner = CliRunner()

    with (
        patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ) as mock_predict,
        patch("sleap_nn.predict.run_inference") as mock_run_inference,
    ):
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
    assert mock_predict.called, "predict() was not called for simple infer case"
    # Source is the first positional arg (a VideoProvider for .mp4 files
    # because the CLI default video_input_format="channels_last" is truthy).
    source = mock_predict.call_args[0][0]
    assert hasattr(source, "video") or isinstance(source, str)
    assert mock_predict.call_args[1]["output_path"] == str(out)
    # The legacy run_inference should NOT have been called for this case.
    assert not mock_run_inference.called


def test_infer_with_tracking_uses_predict(tmp_path):
    """``--tracking`` now routes through ``predict()`` (PR 27).

    The predict call receives a ``tracker_config`` and the legacy
    ``run_inference`` is NOT called.
    """
    out = tmp_path / "out.slp"
    runner = CliRunner()

    with (
        patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ) as mock_predict,
        patch("sleap_nn.predict.run_inference") as mock_run,
    ):
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
        assert mock_predict.called
        assert not mock_run.called
        cfg = mock_predict.call_args[1]["tracker_config"]
        assert cfg.window_size == 7
        assert cfg.candidates_method == "local_queues"
        assert cfg.max_tracks == 3


def test_infer_with_tracking_plus_filter_uses_predict(tmp_path):
    """``--tracking`` + ``--filter_*`` now route through ``predict()`` (PR 27).

    The call should receive both a ``tracker_config`` and a
    ``filter_config`` reflecting the CLI flags.
    """
    out = tmp_path / "out.slp"
    runner = CliRunner()

    with (
        patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ) as mock_predict,
        patch("sleap_nn.predict.run_inference") as mock_run,
    ):
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
        assert mock_predict.called
        assert not mock_run.called
        kw = mock_predict.call_args[1]
        assert kw["tracker_config"] is not None
        assert kw["filter_config"] is not None
        fc = kw["filter_config"]
        assert fc.overlapping is True
        assert fc.overlapping_method == "oks"
        assert abs(fc.overlapping_threshold - 0.5) < 1e-9


def test_infer_with_filter_flags_builds_filter_config(tmp_path):
    """``--filter_min_visible_nodes`` etc. build a ``FilterConfig`` for the new flow."""
    out = tmp_path / "out.slp"
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
        fc = mock_predict.call_args[1]["filter_config"]
        assert fc.min_visible_nodes == 3
        assert abs(fc.min_mean_node_score - 0.4) < 1e-9
        assert abs(fc.min_instance_score - 0.6) < 1e-9


def test_infer_no_empty_frames_passes_clean_flag(tmp_path):
    """``--no_empty_frames`` propagates as ``clean_empty_frames=True`` to predict()."""
    out = tmp_path / "out.slp"
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
                "--no_empty_frames",
                "--output_path",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_predict.call_args[1]
        assert kw["clean_empty_frames"] is True


def test_infer_only_suggested_frames_routes_to_predict(tmp_path):
    """``--only_suggested_frames`` goes through ``predict()`` + LabelsProvider."""
    out = tmp_path / "out.slp"
    runner = CliRunner()

    with (
        patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ) as mock_predict,
        patch("sleap_nn.inference.providers.LabelsProvider") as mock_provider,
        patch("sleap_nn.predict.run_inference") as mock_run,
    ):
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
        assert mock_predict.called
        assert not mock_run.called
        # The source (first positional arg) should be the LabelsProvider instance.
        source = mock_predict.call_args[0][0]
        assert source == mock_provider.return_value
        provider_kwargs = mock_provider.call_args[1]
        assert provider_kwargs["only_suggested_frames"] is True


def test_infer_gui_emits_json_progress(tmp_path):
    """``--gui`` wires a JSON-progress callback through to ``predict()``."""
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
                "--gui",
            ],
        )
        assert result.exit_code == 0, result.output
        cb = mock_predict.call_args[1]["progress_callback"]
        assert cb is not None
        # The callback should emit a JSON line on stdout (final 100%).
        import contextlib
        import io
        import json

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cb(10, 10)
        emitted = buf.getvalue().strip()
        parsed = json.loads(emitted)
        assert parsed["n_processed"] == 10
        assert parsed["n_total"] == 10


def test_infer_without_gui_attaches_rich_progress_callback(tmp_path):
    """No ``--gui`` => ``predict()`` gets a (non-JSON) Rich progress callback (#583)."""
    import io
    import json
    from contextlib import redirect_stdout

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
            ],
        )
        assert result.exit_code == 0, result.output
        cb = mock_predict.call_args[1].get("progress_callback")
        assert cb is not None and callable(cb)
        # It is the Rich callback, NOT the --gui JSON emitter: driving it must
        # not print a JSON line to stdout.
        buf = io.StringIO()
        with redirect_stdout(buf):
            cb(1, 4)
        out = buf.getvalue().strip()
        if out:
            with __import__("pytest").raises(json.JSONDecodeError):
                json.loads(out)


def test_infer_backbone_and_head_ckpt_paths_thread_to_predict(tmp_path):
    """``--backbone_ckpt_path`` / ``--head_ckpt_path`` reach ``predict()``."""
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
                "--backbone_ckpt_path",
                "/fake/backbone.ckpt",
                "--head_ckpt_path",
                "/fake/head.ckpt",
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_predict.call_args[1]
        assert kw["backbone_ckpt_path"] == "/fake/backbone.ckpt"
        assert kw["head_ckpt_path"] == "/fake/head.ckpt"


def test_infer_retrack_only_dispatches_to_predictor_retrack(tmp_path):
    """``--tracking`` + no ``model_paths`` + .slp data -> ``Predictor.retrack``."""
    runner = CliRunner()
    fake_labels = MagicMock()
    fake_labels.videos = []
    fake_labels.skeletons = []
    fake_labels.labeled_frames = []
    tracked = MagicMock()
    with (
        patch("sleap_io.load_slp", return_value=fake_labels),
        patch(
            "sleap_nn.inference.predictor.Predictor.retrack", return_value=tracked
        ) as mock_retrack,
        patch("sleap_nn.predict.run_inference") as mock_legacy,
    ):
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.slp",
                "--tracking",
                "--tracking_window_size",
                "9",
                "--output_path",
                str(tmp_path / "out.slp"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_retrack.called
        assert not mock_legacy.called
        # Tracker config carried the right knob.
        cfg = mock_retrack.call_args[0][1]
        assert cfg.window_size == 9
        # The result was saved.
        tracked.save.assert_called_once()


def test_infer_paf_workers_zero_no_warning(tmp_path):
    """``--paf-workers 0`` is the default, must not emit a warning."""
    runner = CliRunner()
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
            ],
        )
        assert result.exit_code == 0, result.output
        assert "paf-workers > 0" not in result.output


# ─────────────────────────────────────────────────────────────────────────
# #582 — candidates_method defaulting (infer auto-switch + track regression)
# ─────────────────────────────────────────────────────────────────────────


def test_infer_max_tracks_auto_switches_to_local_queues():
    """`infer --max_tracks N` with no explicit method defaults to local_queues.

    Drives the real CLI wiring (#582): the infer option default is None so
    _build_tracker_config can switch fixed_window -> local_queues for max_tracks.
    """
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--tracking",
                "--max_tracks",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = mock_predict.call_args[1]["tracker_config"]
        assert cfg.candidates_method == "local_queues"
        assert cfg.max_tracks == 3


def test_infer_explicit_fixed_window_with_max_tracks_respected():
    """An explicit `--candidates_method fixed_window` is not overridden (#582)."""
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--tracking",
                "--candidates_method",
                "fixed_window",
                "--max_tracks",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args[1]["tracker_config"].candidates_method == (
            "fixed_window"
        )


def test_track_command_candidates_method_defaults_fixed_window():
    """Legacy `track --tracking` keeps the fixed_window default (no None crash).

    Regression guard for #582: the new flow's option was changed to default
    None, which must NOT leak into the legacy `track` command (its option must
    stay 'fixed_window', else run_inference -> Tracker.from_config(None) crashes).
    """
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference", return_value=MagicMock()) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--tracking",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.call_args[1]["candidates_method"] == "fixed_window"


# ─────────────────────────────────────────────────────────────────────────
# #583 — CLI feature preservation
# ─────────────────────────────────────────────────────────────────────────


def test_track_gui_forwards_to_run_inference():
    """`track --gui` forwards gui=True to run_inference (no longer popped). #583."""
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference", return_value=MagicMock()) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--gui",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.call_args[1]["gui"] is True


def test_track_no_gui_defaults_false():
    """Without --gui, track forwards gui=False."""
    runner = CliRunner()
    with patch("sleap_nn.predict.run_inference", return_value=MagicMock()) as mock_run:
        result = runner.invoke(
            cli,
            ["track", "--data_path", "/fake/x.mp4", "--model_paths", "/fake/m"],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.call_args[1]["gui"] is False


def test_infer_paf_knobs_thread_to_predict():
    """The 5 bottom-up PAF knobs reach run.predict (no longer silent no-ops). #583."""
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--max_edge_length_ratio",
                "0.3",
                "--dist_penalty_weight",
                "2.0",
                "--n_points",
                "7",
                "--min_instance_peaks",
                "2",
                "--min_line_scores",
                "0.4",
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_predict.call_args[1]
        assert abs(kw["max_edge_length_ratio"] - 0.3) < 1e-9
        assert abs(kw["dist_penalty_weight"] - 2.0) < 1e-9
        assert kw["n_points"] == 7
        assert kw["min_instance_peaks"] == 2
        assert abs(kw["min_line_scores"] - 0.4) < 1e-9


def test_infer_queue_maxsize_accepted_but_not_forwarded():
    """`--queue_maxsize` is accepted (compat) but never forwarded to predict. #583."""
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--queue_maxsize",
                "64",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "queue_maxsize" not in mock_predict.call_args[1]


def test_infer_queue_maxsize_hidden_in_help():
    """The no-op --queue_maxsize is hidden from `infer --help`. #583."""
    runner = CliRunner()
    result = runner.invoke(cli, ["infer", "--help"])
    assert result.exit_code == 0
    assert "--queue_maxsize" not in result.output


def test_infer_video_index_scopes_and_names_output():
    """`infer --video_index 1` scopes to that video + suffixes the output. #583."""
    import sleap_io as sio

    from sleap_nn.inference.providers import LabelsProvider

    skel = sio.Skeleton(nodes=["a", "b"])
    v0 = sio.Video(filename="a.mp4")
    v1 = sio.Video(filename="b.mp4")
    fake = sio.Labels(videos=[v0, v1], skeletons=[skel], labeled_frames=[])

    runner = CliRunner()
    with (
        patch("sleap_io.load_slp", return_value=fake),
        patch("sleap_nn.inference.run.predict", return_value=MagicMock()) as mock_pred,
    ):
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/x.slp",
                "--model_paths",
                "/fake/model",
                "--video_index",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        source = mock_pred.call_args[0][0]
        assert isinstance(source, LabelsProvider)
        # Output filename is suffixed with the scoped video's name (legacy parity).
        assert mock_pred.call_args[1]["output_path"].endswith("b.predictions.slp")


def test_infer_video_index_out_of_range_errors():
    """An out-of-range --video_index is a clear UsageError. #583."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=["a", "b"])
    fake = sio.Labels(
        videos=[sio.Video(filename="a.mp4")], skeletons=[skel], labeled_frames=[]
    )
    runner = CliRunner()
    with (
        patch("sleap_io.load_slp", return_value=fake),
        patch("sleap_nn.inference.run.predict", return_value=MagicMock()),
    ):
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/x.slp",
                "--model_paths",
                "/fake/model",
                "--video_index",
                "5",
            ],
        )
        assert result.exit_code != 0
        assert "out of range" in result.output


def test_infer_retrack_only_sets_tracking_provenance(tmp_path):
    """Retrack-only writes tracking-only provenance to the saved .slp. #583."""
    import numpy as np
    import sleap_io as sio

    skel = sio.Skeleton(nodes=["head", "tail"])
    video = sio.Video(filename="d.mp4")
    pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    lfs = [
        sio.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                sio.PredictedInstance.from_numpy(
                    points_data=pts, skeleton=skel, score=0.9
                )
            ],
        )
        for i in range(3)
    ]
    fake = sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=lfs,
        provenance={"filename": "/path/preds.slp"},
    )
    out_path = tmp_path / "retracked.slp"
    runner = CliRunner()
    with patch("sleap_io.load_slp", return_value=fake):
        result = runner.invoke(
            cli,
            [
                "infer",
                "--data_path",
                "/fake/preds.slp",
                "--tracking",
                "--tracking_window_size",
                "9",
                "--output_path",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
    reloaded = sio.load_slp(str(out_path))
    assert reloaded.provenance.get("pipeline_type") == "tracking_only"
    assert "sleap_nn_version" in reloaded.provenance
    assert "tracking_start_timestamp" in reloaded.provenance
    assert reloaded.provenance.get("tracking_config", {}).get("window_size") == 9
