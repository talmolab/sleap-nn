"""The ``.slp`` save options on ``predict`` and ``track``.

``predict`` takes them on every route that writes a ``.slp``.

Unset options must leave every call exactly as before; set options must reach the
save on each route; invalid ones must fail before any model loads.
"""

import re
from unittest.mock import MagicMock, patch

import h5py
import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_nn.cli import cli
from sleap_nn.inference.save_options import SAVE_OPTION_NAMES

SLP = "tests/assets/datasets/minimal_instance.pkg.slp"
VIDEO = "tests/assets/datasets/centered_pair_small.mp4"
CENTROID = "tests/assets/model_ckpts/minimal_instance_centroid"
CENTERED = "tests/assets/model_ckpts/minimal_instance_centered_instance"
SET = ["--slp_version", "v2", "--compression", "gzip", "--unlabeled_as", "absent"]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(result) -> str:
    return " ".join(_ANSI.sub("", result.output).replace("│", " ").split())


def _set_keys(kw) -> set:
    return {k for k in SAVE_OPTION_NAMES if k in kw}


def _format_id(path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["metadata"].attrs["format_id"])


def test_help_lists_the_options():
    for command in ("predict", "track", "detect-boxes"):
        out = CliRunner().invoke(cli, [command, "--help"]).output
        for name in SAVE_OPTION_NAMES:
            assert f"--{name}" in out, (command, name)


# ── in-memory route ──────────────────────────────────────────────────────────


def _in_memory(*extra):
    with patch("sleap_nn.inference.run.predict", return_value=MagicMock()) as p:
        result = CliRunner().invoke(
            cli,
            ["predict", "-i", "/fake/v.mp4", "-m", "/fake/m", "-d", "cpu", *extra],
        )
    return result, p


def test_in_memory_defaults_pass_nothing():
    result, p = _in_memory()
    assert result.exit_code == 0, result.output
    assert _set_keys(p.call_args.kwargs) == set()


def test_in_memory_forwards_set_options():
    result, p = _in_memory(*SET, "--image_format", "jpg", "--image_quality", "80")
    assert result.exit_code == 0, result.output
    kw = p.call_args.kwargs
    assert kw["slp_version"] == "v2" and kw["compression"] == "gzip"
    assert kw["unlabeled_as"] == "absent"
    assert kw["image_format"] == "jpg" and kw["image_quality"] == 80


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--image_quality", "5"], "image_quality needs"),
        (["--image_format", "png", "--image_quality", "50"], "0-9"),
        (["--slp_version", "v9"], "v9"),
    ],
)
def test_invalid_options_fail_before_any_model_loads(extra, match):
    result, p = _in_memory(*extra)
    assert result.exit_code != 0
    assert match in _flat(result)
    assert not p.called


def test_sam_route_forwards_options():
    """``--mask_backend`` goes through ``predict()``; the options ride along."""
    result, p = _in_memory_sam("--slp_version", "v3")
    assert result.exit_code == 0, result.output
    assert p.call_args.kwargs["slp_version"] == "v3"
    assert p.call_args.kwargs["mask_backend"] == "sam"


def _in_memory_sam(*extra):
    with patch("sleap_nn.inference.run.predict", return_value=MagicMock()) as p:
        result = CliRunner().invoke(
            cli,
            ["predict", "-i", SLP, "--mask_backend", "sam", "-d", "cpu", *extra],
        )
    return result, p


# ── tracking-only (retrack) route ────────────────────────────────────────────


def test_retrack_route(tmp_path):
    def _retrack(cli_args):
        with (
            patch(
                "sleap_nn.inference.predictor.Predictor.retrack",
                side_effect=lambda labels, *a, **k: labels,
            ),
            patch("sleap_nn.inference.run.save_predictions") as save,
        ):
            result = CliRunner().invoke(
                cli,
                ["predict", "-i", SLP, "-t", "-o", str(tmp_path / "r.slp")] + cli_args,
            )
        assert result.exit_code == 0, result.output
        return save.call_args.kwargs

    assert _set_keys(_retrack([])) == set()
    kw = _retrack(SET)
    assert kw["slp_version"] == "v2" and kw["compression"] == "gzip"


# ── stream-to-file route ─────────────────────────────────────────────────────


def test_stream_to_file_route(tmp_path):
    def _stream(cli_args):
        predictor = MagicMock()
        with patch(
            "sleap_nn.inference.predictor.Predictor.from_model_paths",
            return_value=predictor,
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "predict",
                    "-i",
                    VIDEO,
                    "-m",
                    "/fake/m",
                    "-d",
                    "cpu",
                    "--stream-to-file",
                    str(tmp_path / "s.slp"),
                ]
                + cli_args,
            )
        assert result.exit_code == 0, result.output
        return predictor.predict_to_file.call_args.kwargs

    assert _set_keys(_stream([])) == set()
    kw = _stream(SET)
    assert kw["slp_version"] == "v2" and kw["unlabeled_as"] == "absent"


def test_stream_to_file_real_v2(tmp_path):
    """A real streamed centroid-only run written in the 2.x layout."""
    out = tmp_path / "s.slp"
    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "-i",
            SLP,
            "-m",
            CENTROID,
            "--centroid_only",
            "-d",
            "cpu",
            "--stream-to-file",
            str(out),
            "--slp_version",
            "v2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _format_id(out) < 3.0
    assert len(sio.load_slp(str(out), open_videos=False)) == 1


# ── embedding (re-ID) route ──────────────────────────────────────────────────


def test_embedding_route_honors_options(minimal_embedding_model_dir, tmp_path):
    def _embed(cli_args):
        with patch(
            "sleap_nn.inference.embedding.predict_embeddings_to_slp",
            return_value=str(tmp_path / "e.slp"),
        ) as run:
            result = CliRunner().invoke(
                cli,
                [
                    "predict",
                    "-m",
                    minimal_embedding_model_dir,
                    "-i",
                    SLP,
                    "--save_embeddings",
                    "slp",
                    "-d",
                    "cpu",
                ]
                + cli_args,
            )
        assert result.exit_code == 0, result.output
        return run.call_args.kwargs

    assert _set_keys(_embed([])) == set()
    kw = _embed(SET)
    assert kw["slp_version"] == "v2" and kw["compression"] == "gzip"


# ── real in-memory run ───────────────────────────────────────────────────────


def test_in_memory_real_v2_round_trip(tmp_path):
    v2, default = tmp_path / "v2.slp", tmp_path / "default.slp"
    base = ["predict", "-i", SLP, "-m", CENTROID, "--centroid_only", "-d", "cpu"]
    for out, extra in ((v2, ["--slp_version", "v2"]), (default, [])):
        result = CliRunner().invoke(cli, base + ["-o", str(out), *extra])
        assert result.exit_code == 0, result.output
    assert _format_id(v2) < 3.0
    # The default is sleap-io's: SLP 3.0 on a sleap-io without `slp_version`; the
    # old layout (whenever the content fits, as here) on one with it ("auto").
    from sleap_nn.inference.save_options import _save_slp_parameters

    if "slp_version" in _save_slp_parameters():
        assert _format_id(default) < 3.0
    else:
        assert _format_id(default) >= 3.0
    a = sio.load_slp(str(v2), open_videos=False)
    b = sio.load_slp(str(default), open_videos=False)
    assert [len(lf.instances) for lf in a] == [len(lf.instances) for lf in b]


# ── legacy track ─────────────────────────────────────────────────────────────


def test_track_defaults_and_forwarding():
    def _track(cli_args):
        with patch("sleap_nn.legacy_predict.run_inference") as run:
            result = CliRunner().invoke(
                cli, ["track", "-i", SLP, "-m", "/fake/m", *cli_args]
            )
        assert result.exit_code == 0, result.output
        return run.call_args.kwargs

    assert _set_keys(_track([])) == set()
    kw = _track(SET)
    assert kw["slp_version"] == "v2" and kw["compression"] == "gzip"
    assert kw["unlabeled_as"] == "absent"


def test_track_invalid_option_fails_before_inference():
    with patch("sleap_nn.legacy_predict.run_inference") as run:
        result = CliRunner().invoke(
            cli, ["track", "-i", SLP, "-m", "/fake/m", "--image_quality", "3"]
        )
    assert result.exit_code != 0 and not run.called
    assert "image_quality needs" in _flat(result)


def test_track_real_v2(tmp_path):
    """The legacy pipeline's save takes the options (a real top-down run)."""
    out = tmp_path / "t.slp"
    result = CliRunner().invoke(
        cli,
        [
            "track",
            "-i",
            SLP,
            "-m",
            CENTROID,
            "-m",
            CENTERED,
            "-d",
            "cpu",
            "-o",
            str(out),
            "--slp_version",
            "v2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _format_id(out) < 3.0
