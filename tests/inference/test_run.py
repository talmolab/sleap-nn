"""Direct unit tests for ``sleap_nn.inference.run.predict`` (#584 coverage gap).

The CLI tests mock ``run.predict`` as the patch target, so its own body (the
two source guards, build-kwargs assembly, the from_export_dir branch, and the
output_path save) was never executed by fast tests. These tests run that body
with ``Predictor`` mocked so no checkpoints are loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sleap_nn.inference.run import predict


def _mock_predictor():
    """A Predictor stand-in whose predict() returns a MagicMock Labels."""
    predictor = MagicMock()
    predictor.predict.return_value = MagicMock(name="labels")
    return predictor


def test_predict_requires_exactly_one_source():
    """Both or neither of model_paths/export_dir is a ValueError."""
    with pytest.raises(ValueError, match="not both"):
        predict("x.mp4", model_paths=["m"], export_dir="d")
    with pytest.raises(ValueError, match="required"):
        predict("x.mp4")


def test_predict_model_paths_forwards_build_and_override_kwargs():
    """from_model_paths gets construction kwargs (incl. PAF knobs); predict gets
    the prediction-time overrides; conditional kwargs only when set (#584)."""
    pred = _mock_predictor()
    with patch(
        "sleap_nn.inference.predictor.Predictor.from_model_paths",
        return_value=pred,
    ) as mock_factory:
        predict(
            "video.mp4",
            model_paths=["/m"],
            device="cpu",
            batch_size=8,
            paf_workers=2,
            min_line_scores=0.4,
            n_points=7,
            peak_threshold=0.3,
            max_instances=2,
            return_pafs=True,
        )
    bk = mock_factory.call_args[1]
    assert bk["device"] == "cpu"
    assert bk["batch_size"] == 8
    assert bk["paf_workers"] == 2
    # PAF knobs threaded to construction.
    assert abs(bk["min_line_scores"] - 0.4) < 1e-9
    assert bk["n_points"] == 7
    # Conditional kwargs absent when not provided.
    assert "filter_config" not in bk
    assert "tracker_config" not in bk
    assert "anchor_part" not in bk
    # Prediction-time overrides forwarded to predict().
    pk = pred.predict.call_args[1]
    assert abs(pk["peak_threshold"] - 0.3) < 1e-9
    assert pk["max_instances"] == 2
    assert pk["return_pafs"] is True


def test_predict_forwards_conditional_kwargs_when_set():
    """filter_config / tracker_config / anchor_part / centroid_only forwarded
    only when non-None/truthy."""
    from sleap_nn.inference.filters import FilterConfig
    from sleap_nn.inference.tracking import TrackerConfig

    pred = _mock_predictor()
    fc, tc = FilterConfig(), TrackerConfig()
    with patch(
        "sleap_nn.inference.predictor.Predictor.from_model_paths",
        return_value=pred,
    ) as mock_factory:
        predict(
            "video.mp4",
            model_paths=["/m"],
            device="cpu",
            filter_config=fc,
            tracker_config=tc,
            anchor_part="head",
            centroid_only=True,
        )
    bk = mock_factory.call_args[1]
    assert bk["filter_config"] is fc
    assert bk["tracker_config"] is tc
    assert bk["anchor_part"] == "head"
    assert bk["centroid_only"] is True


def test_predict_export_dir_branch():
    """export_dir routes to from_export_dir (not from_model_paths)."""
    pred = _mock_predictor()
    with (
        patch(
            "sleap_nn.inference.predictor.Predictor.from_export_dir",
            return_value=pred,
        ) as mock_export,
        patch("sleap_nn.inference.predictor.Predictor.from_model_paths") as mock_model,
    ):
        predict("video.mp4", export_dir="/exp", device="cpu")
    assert mock_export.called
    assert not mock_model.called


def test_predict_saves_when_output_path_given(tmp_path):
    """output_path triggers labels.save."""
    pred = _mock_predictor()
    out = tmp_path / "out.slp"
    with patch(
        "sleap_nn.inference.predictor.Predictor.from_model_paths",
        return_value=pred,
    ):
        labels = predict(
            "video.mp4", model_paths=["/m"], device="cpu", output_path=str(out)
        )
    labels.save.assert_called_once()


def test_predict_device_auto_resolves(monkeypatch):
    """device='auto' resolves to cpu when no accelerator is available."""
    import torch

    pred = _mock_predictor()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with patch(
        "sleap_nn.inference.predictor.Predictor.from_model_paths",
        return_value=pred,
    ) as mock_factory:
        predict("video.mp4", model_paths=["/m"], device="auto")
    assert mock_factory.call_args[1]["device"] == "cpu"
