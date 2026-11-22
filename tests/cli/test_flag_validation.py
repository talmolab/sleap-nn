"""Validation tests for the PR 10 flags introduced on ``sleap-nn predict``.

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
from unittest.mock import MagicMock, patch

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
            "predict",
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


def test_stream_to_file_with_no_empty_frames_raises_usage_error():
    """``--stream-to-file`` + ``--no_empty_frames`` is rejected (PR 15).

    Streaming writes each batch to disk, so dropping empty frames after
    the fact isn't possible.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--stream-to-file",
            "/tmp/out.slp",
            "--no_empty_frames",
        ],
    )
    assert result.exit_code != 0
    assert "no_empty_frames" in result.output.lower()


def test_stream_to_file_with_embed_raises_usage_error():
    """``--stream-to-file`` + ``--embed true`` is rejected (#652).

    The incremental writer saves with sleap-io defaults, so an embedding
    request would be silently dropped; fail loudly instead.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--stream-to-file",
            "/tmp/out.slp",
            "--embed",
            "true",
        ],
    )
    assert result.exit_code != 0
    assert "embed" in result.output.lower()
    assert "stream-to-file" in result.output


def test_stream_to_file_with_restore_source_videos_raises_usage_error():
    """``--stream-to-file`` + ``--restore_source_videos`` is rejected.

    The incremental writer always saves with ``restore_original_videos=False``
    (PRESERVE_SOURCE), matching the non-streaming default; requesting the
    non-default ``True`` (restore the pre-embedding source video) can't be
    honored there.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--stream-to-file",
            "/tmp/out.slp",
            "--restore_source_videos",
        ],
    )
    assert result.exit_code != 0
    assert "restore_source_videos" in result.output.lower()
    assert "stream-to-file" in result.output


def test_stream_to_file_with_no_restore_source_videos_is_allowed():
    """``--stream-to-file`` + ``--no-restore_source_videos`` (the default) is fine."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--stream-to-file",
            "/tmp/out.slp",
            "--no-restore_source_videos",
        ],
    )
    # Should get past the flag-validation guard (may still fail later for
    # unrelated reasons, e.g. the fake model/data paths not existing).
    assert "restore_source_videos" not in result.output.lower()


def test_write_interval_without_stream_to_file_errors():
    """``--write-interval`` alone is meaningless and rejected."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
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

    The deprecation fires in ``_run_inference_impl`` regardless of which
    backend serves the request -- we just need the impl to not crash.
    """
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = runner.invoke(
                cli,
                [
                    "predict",
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


def test_paf_workers_positive_does_not_warn_on_new_flow():
    """``--paf-workers > 0`` works on the new flow without legacy warnings.

    PR 16 routes everything through the new flow; the old "no effect on
    legacy path" warning is gone.
    """
    runner = CliRunner()
    with patch(
        "sleap_nn.inference.run.predict",
        return_value=MagicMock(),
    ):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--paf-workers",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "paf-workers > 0" not in result.output


def test_stream_to_file_invokes_new_predictor_flow(tmp_path):
    """``--stream-to-file`` reaches ``Predictor.predict_to_file``.

    Patches the factory to return a stub Predictor whose
    ``predict_to_file`` records that it was called -- confirms the CLI
    routes through the new flow rather than the legacy ``run_inference``.
    """
    out = tmp_path / "out.slp"
    runner = CliRunner()

    stub_predictor = MagicMock()
    stub_predictor.predict_to_file.return_value = str(out)
    with (
        patch(
            "sleap_nn.inference.predictor.Predictor.from_model_paths"
        ) as mock_factory,
        patch("sleap_nn.inference.providers.VideoProvider"),
    ):
        mock_factory.return_value = stub_predictor
        result = runner.invoke(
            cli,
            [
                "predict",
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


def test_embed_invalid_choice_rejected():
    """``--embed`` only accepts auto/true/false; an unknown value is rejected (#652)."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--embed",
            "maybe",
        ],
    )
    assert result.exit_code != 0
    assert "embed" in result.output.lower()


def test_embed_choice_case_insensitive(tmp_path):
    """``--embed TRUE`` is normalized to lowercase and forwarded (#652)."""
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
                "--output_path",
                str(tmp_path / "out.slp"),
                "--embed",
                "TRUE",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_predict.call_args[1]["embed"] == "true"


def test_unknown_flag_rejected_cleanly():
    """Click's standard usage-error path catches unknown flags."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--this-is-not-a-flag",
        ],
    )
    assert result.exit_code != 0
    assert (
        "no such option" in result.output.lower() or "unknown" in result.output.lower()
    )


def test_upload_batches_flag_reaches_predict():
    """``--upload_batches`` is passed on to ``run.predict`` (off unless given)."""
    runner = CliRunner()
    for args, expected in (([], False), (["--upload_batches"], True)):
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
                ]
                + args,
            )
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.kwargs["upload_batches"] is expected


def test_track_rejects_upload_batches():
    """`--upload_batches` is a `predict` (new flow) flag; legacy `track` refuses it."""
    runner = CliRunner()
    with patch(
        "sleap_nn.legacy_predict.run_inference", return_value=MagicMock()
    ) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--upload_batches",
            ],
        )
    assert result.exit_code != 0
    assert "upload_batches" in result.output
    assert not mock_run.called


def test_memory_format_flag_reaches_predict():
    """``--memory_format`` is passed on to ``run.predict`` (contiguous by default)."""
    runner = CliRunner()
    for args, expected in (
        ([], "contiguous"),
        (["--memory_format", "channels_last"], "channels_last"),
    ):
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
                ]
                + args,
            )
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.kwargs["memory_format"] == expected


def test_memory_format_flag_rejects_unknown_value():
    """Only ``contiguous`` / ``channels_last`` are accepted."""
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--data_path",
            "/fake/path.mp4",
            "--model_paths",
            "/fake/model",
            "--memory_format",
            "nhwc",
        ],
    )
    assert result.exit_code != 0
    assert "memory_format" in result.output


def test_track_rejects_memory_format():
    """`--memory_format` is a `predict` (new flow) flag; legacy `track` refuses it."""
    runner = CliRunner()
    with patch(
        "sleap_nn.legacy_predict.run_inference", return_value=MagicMock()
    ) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--memory_format",
                "channels_last",
            ],
        )
    assert result.exit_code != 0
    assert "memory_format" in result.output
    assert not mock_run.called


# ── `--precision {32,bf16,fp16}` ─────────────────────────────────────────────


def _flat(result) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (rich-click wraps by width)."""
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ")
    return " ".join(text.split())


def _exported_dir(tmp_path, name: str, with_ckpt: bool = False) -> str:
    """A directory that looks like an exported model (``model.onnx`` inside)."""
    d = tmp_path / name
    d.mkdir()
    (d / "model.onnx").write_bytes(b"")
    if with_ckpt:
        (d / "best.ckpt").write_bytes(b"")
    return str(d)


def test_precision_flag_reaches_predict():
    """``--precision`` is passed on to ``run.predict`` ("32" by default)."""
    runner = CliRunner()
    for args, expected in (
        ([], "32"),
        (["--precision", "32"], "32"),
        (["--precision", "bf16"], "bf16"),
        (["--precision", "fp16"], "fp16"),
        (["--precision", "fp32"], "32"),
        (["--precision", "FP16"], "fp16"),
        (["--precision", "BF16"], "bf16"),
    ):
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
                ]
                + args,
            )
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.kwargs["precision"] == expected


def test_precision_flag_reaches_stream_to_file_predictor(tmp_path):
    """On the ``--stream-to-file`` route the predictor is built with it."""
    out = tmp_path / "out.slp"
    runner = CliRunner()
    for args, expected in (([], "32"), (["--precision", "fp16"], "fp16")):
        stub_predictor = MagicMock()
        stub_predictor.predict_to_file.return_value = str(out)
        with (
            patch(
                "sleap_nn.inference.predictor.Predictor.from_model_paths",
                return_value=stub_predictor,
            ) as mock_factory,
            patch("sleap_nn.inference.providers.VideoProvider"),
        ):
            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                    "--stream-to-file",
                    str(out),
                ]
                + args,
            )
        assert result.exit_code == 0, result.output
        assert mock_factory.call_args.kwargs["precision"] == expected
        assert stub_predictor.predict_to_file.called


def test_precision_flag_rejects_unknown_value():
    """Only ``32`` (or ``fp32``) / ``bf16`` / ``fp16`` are accepted."""
    for value in ("16", "tf32", "half"):
        with patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict:
            result = CliRunner().invoke(
                cli,
                [
                    "predict",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                    "--precision",
                    value,
                ],
            )
        assert result.exit_code != 0
        assert "precision" in result.output
        assert not mock_predict.called


def test_track_rejects_precision():
    """`--precision` is a `predict` (new flow) flag; legacy `track` refuses it."""
    runner = CliRunner()
    with patch(
        "sleap_nn.legacy_predict.run_inference", return_value=MagicMock()
    ) as mock_run:
        result = runner.invoke(
            cli,
            [
                "track",
                "--data_path",
                "/fake/path.mp4",
                "--model_paths",
                "/fake/model",
                "--precision",
                "fp16",
            ],
        )
    assert result.exit_code != 0
    assert "precision" in result.output
    assert not mock_run.called


def test_precision_refused_when_every_model_is_exported(tmp_path):
    """No trained checkpoint to run under it: refused before anything loads.

    A single ``-m`` export directory takes the exported route whatever else it
    holds; an exported pose model with an exported ``--part_model`` likewise has
    no checkpoint.
    """
    one = _exported_dir(tmp_path, "one")
    two = _exported_dir(tmp_path, "two")
    with_ckpt = _exported_dir(tmp_path, "with_ckpt", with_ckpt=True)
    rule = tmp_path / "rule.json"
    rule.write_text("{}")
    for models in (
        ["-m", one],
        ["-m", with_ckpt],
        ["-m", one, "-m", two],
        ["-m", one, "--part_model", two, "--part_crop_rule", str(rule)],
    ):
        with (
            patch(
                "sleap_nn.inference.run.predict", return_value=MagicMock()
            ) as mock_predict,
            patch(
                "sleap_nn.inference.predictor.Predictor.from_model_paths"
            ) as mock_factory,
        ):
            result = CliRunner().invoke(
                cli,
                ["predict", "-i", "/fake/path.mp4", "--precision", "fp16"] + models,
            )
        assert result.exit_code != 0, models
        flat = _flat(result)
        assert "--precision fp16 applies to trained checkpoints only" in flat
        assert "keeps the precision it was exported with" in flat
        assert not mock_predict.called and not mock_factory.called


def test_precision_accepted_when_a_checkpoint_runs_under_it(tmp_path):
    """Checkpoints among exported directories run under it; ``32`` is always fine.

    Among several ``-m`` (and for ``--part_model``) a directory that holds
    ``best.ckpt`` is loaded as that checkpoint.
    """
    one = _exported_dir(tmp_path, "one")
    ckpt_a = _exported_dir(tmp_path, "ckpt_a", with_ckpt=True)
    ckpt_b = _exported_dir(tmp_path, "ckpt_b", with_ckpt=True)
    for args, expected in (
        (["-m", one, "-m", "/fake/checkpoint", "--precision", "bf16"], "bf16"),
        (["-m", ckpt_a, "-m", ckpt_b, "--precision", "fp16"], "fp16"),
        (["-m", one, "--precision", "32"], "32"),
        (["-m", one], "32"),
    ):
        with patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict:
            result = CliRunner().invoke(cli, ["predict", "-i", "/fake/path.mp4"] + args)
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.kwargs["precision"] == expected


def test_precision_refused_on_routes_that_load_no_checkpoint():
    """Tracking-only, SAM masks and the embedding route refuse it by name."""
    reid = "not supported with an `embedding` (re-ID) model"
    for args, has_embedding_model, expected in (
        (
            ["-i", "/fake/labels.slp", "--tracking"],
            False,
            "a tracking-only run loads no model",
        ),
        (
            ["-i", "/fake/labels.slp", "--mask_backend", "sam"],
            False,
            "not supported with --mask_backend",
        ),
        # A lone embedding model, and one fused with a detection model.
        (
            ["-i", "/fake/labels.slp", "-m", "/fake/reid", "--save_embeddings", "slp"],
            True,
            reid,
        ),
        (
            ["-i", "/fake/video.mp4", "-m", "/fake/centroid", "-m", "/fake/reid"]
            + ["--tracking"],
            True,
            reid,
        ),
    ):
        with (
            patch(
                "sleap_nn.inference.run.predict", return_value=MagicMock()
            ) as mock_predict,
            patch("sleap_nn.inference.predictor.Predictor.retrack") as mock_retrack,
            patch(
                "sleap_nn.inference.embedding.predict_embeddings_to_slp"
            ) as mock_embed,
            patch(
                "sleap_nn.cli._has_embedding_model", return_value=has_embedding_model
            ),
        ):
            result = CliRunner().invoke(cli, ["predict", "--precision", "bf16"] + args)
        assert result.exit_code != 0, args
        assert expected in _flat(result), _flat(result)
        assert not (mock_predict.called or mock_retrack.called or mock_embed.called)
