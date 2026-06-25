"""Tests for remote-URL (`http(s)://`, `s3://`, ...) inputs to ``sleap-nn predict``.

sleap-io 0.8.0 can load ``.slp`` and video directly from URLs. Previously the
CLI ran ``Path(data_path)`` on the input, which mangled ``scheme://`` (collapsing
``//`` and flipping to backslashes on Windows) before it reached sleap-io's
URL-aware loaders. These tests cover the URL detection/routing helpers, the
``--headers`` / ``--stream-mode`` remote-option plumbing through the providers,
and end-to-end routing of a URL through ``predict()`` without mangling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_nn.cli import (
    _build_remote_kwargs,
    _default_predictions_path,
    _is_remote_url,
    _resolve_data_path,
    cli,
)
from sleap_nn.inference.providers import LabelsProvider, VideoProvider


# ── helpers ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://example.com/a.slp", True),
        ("http://example.com/a.mp4", True),
        ("s3://bucket/a.slp", True),
        ("gs://bucket/a.mp4", True),
        ("gcs://bucket/a.mp4", True),
        ("az://container/a.slp", True),
        ("abfs://container/a.slp", True),
        ("/local/abs/a.slp", False),
        ("relative/a.mp4", False),
        ("a.slp", False),
        ("", False),
        (r"C:\data\a.slp", False),  # Windows drive letter is not a URL scheme
        ("C:/data/a.slp", False),
    ],
)
def test_is_remote_url(value, expected):
    assert _is_remote_url(value) is expected


def test_resolve_data_path_local():
    source_str, suffix, is_url = _resolve_data_path("/fake/path.mp4")
    assert is_url is False
    assert suffix == ".mp4"
    assert Path(source_str) == Path("/fake/path.mp4")


def test_resolve_data_path_url_preserves_string_and_parses_suffix():
    url = "https://example.com/data/clip.mp4?token=abc#frag"
    source_str, suffix, is_url = _resolve_data_path(url)
    assert is_url is True
    assert source_str == url  # verbatim — NOT run through Path()
    assert suffix == ".mp4"  # from the URL path, ignoring query/fragment


def test_resolve_data_path_url_slp():
    source_str, suffix, is_url = _resolve_data_path("s3://bucket/proj/labels.slp")
    assert (source_str, suffix, is_url) == ("s3://bucket/proj/labels.slp", ".slp", True)


def test_default_predictions_path_local():
    assert _default_predictions_path("/data/v.mp4", False) == "/data/v.mp4.slp"


def test_default_predictions_path_url_uses_basename_in_cwd():
    # Mirrors the local "[data_path].slp" convention: the URL basename (with its
    # extension) plus ".slp", written in the cwd.
    assert _default_predictions_path("https://h/x/clip.mp4", True) == "clip.mp4.slp"
    assert _default_predictions_path("s3://b/p/labels.slp", True) == "labels.slp.slp"


def test_default_predictions_path_scoped():
    out = _default_predictions_path("/data/multi.slp", False, scoped_video_name="vidA")
    assert Path(out) == Path("/data/multi.vidA.predictions.slp")


def test_build_remote_kwargs_headers_and_stream_mode():
    remote = _build_remote_kwargs(
        {"headers": '{"Authorization": "Bearer t"}', "stream_mode": "stream"}
    )
    assert remote == {"headers": {"Authorization": "Bearer t"}, "stream_mode": "stream"}


def test_build_remote_kwargs_empty():
    assert _build_remote_kwargs({}) == {}
    assert _build_remote_kwargs({"headers": None, "stream_mode": None}) == {}


def test_build_remote_kwargs_invalid_headers_json():
    import click

    with pytest.raises(click.UsageError):
        _build_remote_kwargs({"headers": "not-json"})


def test_build_remote_kwargs_headers_not_object():
    import click

    with pytest.raises(click.UsageError):
        _build_remote_kwargs({"headers": "[1, 2, 3]"})


# ── provider remote-kwargs threading ─────────────────────────────────────────
def test_video_provider_forwards_remote_kwargs():
    fake_video = MagicMock()
    fake_video.__len__.return_value = 3
    with patch("sleap_io.load_video", return_value=fake_video) as mock_load:
        VideoProvider(
            video="https://h/clip.mp4",
            remote_kwargs={"headers": {"A": "B"}, "stream_mode": "stream"},
        )
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == "https://h/clip.mp4"
    assert mock_load.call_args.kwargs == {
        "headers": {"A": "B"},
        "stream_mode": "stream",
    }


def test_labels_provider_forwards_remote_kwargs():
    with patch("sleap_io.load_slp", return_value=sio.Labels()) as mock_load:
        LabelsProvider(
            labels="s3://bucket/labels.slp",
            remote_kwargs={"headers": {"A": "B"}},
        )
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == "s3://bucket/labels.slp"
    assert mock_load.call_args.kwargs == {"headers": {"A": "B"}}


def test_labels_provider_no_remote_kwargs_passes_nothing():
    with patch("sleap_io.load_slp", return_value=sio.Labels()) as mock_load:
        LabelsProvider(labels="local.slp")
    assert mock_load.call_args.kwargs == {}


# ── end-to-end CLI routing ───────────────────────────────────────────────────
def test_predict_remote_video_url_not_path_mangled():
    """A bare remote video URL reaches predict() verbatim (no Path() mangling)."""
    runner = CliRunner()
    url = "https://example.com/data/clip.mp4"
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            ["predict", "--data_path", url, "--model_paths", "/fake/model"],
        )
    assert result.exit_code == 0, result.output
    assert mock_predict.call_args[0][0] == url
    # Output defaults to the URL basename in the cwd (cannot write to the host).
    assert mock_predict.call_args[1]["output_path"] == "clip.mp4.slp"


def test_predict_remote_slp_url_passes_through():
    """A bare remote .slp URL is passed to predict() as the raw string."""
    runner = CliRunner()
    url = "s3://bucket/proj/labels.slp"
    with patch(
        "sleap_nn.inference.run.predict", return_value=MagicMock()
    ) as mock_predict:
        result = runner.invoke(
            cli,
            ["predict", "--data_path", url, "--model_paths", "/fake/model"],
        )
    assert result.exit_code == 0, result.output
    assert mock_predict.call_args[0][0] == url
    assert mock_predict.call_args[1]["output_path"] == "labels.slp.slp"


def test_predict_remote_video_url_with_headers_routes_through_provider():
    """``--headers`` on a video URL routes through a VideoProvider carrying them."""
    runner = CliRunner()
    url = "https://example.com/clip.mp4"
    fake_video = MagicMock()
    fake_video.__len__.return_value = 3
    with (
        patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict,
        # The VideoProvider is built eagerly in the CLI, so stub the remote load.
        patch("sleap_io.load_video", return_value=fake_video) as mock_load_video,
    ):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                url,
                "--model_paths",
                "/fake/model",
                "--headers",
                '{"Authorization": "Bearer t"}',
                "--stream-mode",
                "stream",
            ],
        )
    assert result.exit_code == 0, result.output
    # The remote video was loaded with the forwarded headers/stream_mode
    # (alongside the provider's usual input_format kwarg).
    assert mock_load_video.call_args.args[0] == url
    loaded_kwargs = mock_load_video.call_args.kwargs
    assert loaded_kwargs.get("headers") == {"Authorization": "Bearer t"}
    assert loaded_kwargs.get("stream_mode") == "stream"
    source = mock_predict.call_args[0][0]
    assert isinstance(source, VideoProvider)
    assert source.video == url
    assert source.remote_kwargs == {
        "headers": {"Authorization": "Bearer t"},
        "stream_mode": "stream",
    }


def test_predict_remote_slp_url_with_headers_loads_with_kwargs():
    """``--headers`` on a .slp URL loads it with the remote kwargs up front."""
    runner = CliRunner()
    url = "s3://bucket/labels.slp"
    fake_labels = sio.Labels()
    with (
        patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict,
        patch("sleap_io.load_slp", return_value=fake_labels) as mock_load,
    ):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                url,
                "--model_paths",
                "/fake/model",
                "--headers",
                '{"X": "Y"}',
            ],
        )
    assert result.exit_code == 0, result.output
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == url
    assert mock_load.call_args.kwargs == {"headers": {"X": "Y"}}
    # The loaded Labels object (not the raw URL) is handed to predict().
    assert mock_predict.call_args[0][0] is fake_labels


def test_predict_invalid_headers_errors():
    """Malformed ``--headers`` JSON fails with a clear non-zero exit."""
    runner = CliRunner()
    with patch("sleap_nn.inference.run.predict", return_value=MagicMock()):
        result = runner.invoke(
            cli,
            [
                "predict",
                "--data_path",
                "https://h/clip.mp4",
                "--model_paths",
                "/fake/model",
                "--headers",
                "not-json",
            ],
        )
    assert result.exit_code != 0
    assert "headers" in result.output.lower()
