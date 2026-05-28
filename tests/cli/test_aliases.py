"""Tests for deprecated alias commands (PR 10 of #508 / #518).

``sleap-nn track`` is now a deprecated alias for ``sleap-nn infer``;
emits a ``DeprecationWarning`` once and otherwise reaches the same
implementation. ``sleap-nn predict`` is *not* yet aliased (deferred —
the existing top-level ``predict`` runs inference on exported models
and rerouting it requires the export-group refactor).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from click.testing import CliRunner

from sleap_nn.cli import cli


def _mock_new_flow():
    """Patches that make the new in-memory flow a no-op for fast CLI tests."""
    from unittest.mock import MagicMock

    stub_labels = MagicMock()
    return [
        patch(
            "sleap_nn.inference.run.predict",
            return_value=stub_labels,
        ),
    ]


def test_track_emits_deprecation_warning():
    """``sleap-nn track`` emits a DeprecationWarning before delegating.

    The warning is emitted in the ``track`` command body before any
    impl runs, so the routing destination doesn't matter — we just
    need the impl to not crash.
    """
    runner = CliRunner()
    with (_mock_new_flow()[0] as mock_predict,):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = runner.invoke(
                cli,
                [
                    "track",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                ],
            )
        assert result.exit_code == 0, result.output
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any(
            "sleap-nn track" in str(d.message) and "infer" in str(d.message)
            for d in deprecations
        ), [str(d.message) for d in deprecations]


def test_track_and_infer_reach_same_predict_kwargs():
    """``track`` and ``infer`` produce identical kwargs to ``predict()``.

    PR 27 routes everything through ``sleap_nn.inference.run.predict``, so
    we assert kwarg equality on that call.
    """
    runner = CliRunner()
    args_common = [
        "--data_path",
        "/fake/path.mp4",
        "--model_paths",
        "/fake/model",
        "--device",
        "cpu",
        "--batch_size",
        "2",
        "--peak_threshold",
        "0.15",
    ]

    def _capture(cmd: str):
        from unittest.mock import MagicMock

        with patch(
            "sleap_nn.inference.run.predict",
            return_value=MagicMock(),
        ) as mock_predict:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                runner.invoke(cli, [cmd] + args_common)
            return dict(mock_predict.call_args[1])

    assert _capture("infer") == _capture("track")


def test_export_predict_top_level_still_works():
    """The legacy top-level ``sleap-nn predict`` (export-trained) is intact.

    PR 10 explicitly does NOT remap top-level ``predict`` — that's a
    follow-up. ``sleap-nn predict --help`` should still render the
    export-trained predict command (with EXPORT_DIR + VIDEO_PATH args).
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0, result.output
    # Marker that this is still the export-trained predict, not infer.
    assert "EXPORT_DIR" in result.output or "VIDEO_PATH" in result.output
