"""CLI tests for ``--centroid-only`` (PR 25 / epic #508).

Covers:

1. ``--centroid-only`` is exposed in ``sleap-nn infer --help``.
2. Setting ``--centroid-only`` threads ``centroid_only=True`` into the
   ``predict()`` call.
3. Omitting the flag leaves ``centroid_only`` out of the predict kwargs
   (auto-detect handles the single-centroid case).
4. ``--centroid_only`` (underscore variant) is also accepted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def test_centroid_only_flag_in_infer_help():
    """``--centroid-only`` appears in the help output of ``sleap-nn infer``."""
    runner = CliRunner()
    result = runner.invoke(cli, ["infer", "--help"])
    assert result.exit_code == 0, result.output
    assert "--centroid-only" in result.output or "--centroid_only" in result.output


def test_centroid_only_flag_propagates_to_predict():
    """``--centroid-only`` -> ``centroid_only=True`` in ``predict()`` kwargs."""
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
                "/fake/centroid",
                "--model_paths",
                "/fake/centered",
                "--centroid-only",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        assert mock_predict.called
        kw = mock_predict.call_args[1]
        assert kw.get("centroid_only") is True


def test_centroid_only_flag_omitted_is_default_off():
    """Without ``--centroid-only``, the predict call doesn't set centroid_only."""
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
                "/fake/centroid",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_predict.call_args[1]
        # Either absent or explicitly False -- auto-detect path.
        assert kw.get("centroid_only", False) is False


def test_centroid_only_underscore_variant_accepted():
    """``--centroid_only`` (underscore) and ``--centroid-only`` (dash) both work."""
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
                "/fake/centroid",
                "--centroid_only",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        kw = mock_predict.call_args[1]
        assert kw.get("centroid_only") is True
