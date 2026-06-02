"""Direct unit tests for ``sleap_nn.inference.run.predict`` (#584 coverage gap).

The CLI tests mock ``run.predict`` as the patch target, so its own body (the
two source guards, build-kwargs assembly, the from_export_dir branch, and the
output_path save) was never executed by fast tests. These tests run that body
with ``Predictor`` mocked so no checkpoints are loaded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sleap_io as sio

from sleap_nn.inference.run import (
    predict,
    save_analysis_h5_files,
    save_predictions,
)


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
    the prediction-time overrides; conditional kwargs only when set (#584).
    """
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
    only when non-None/truthy.
    """
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


def test_predict_export_dir_forwards_runtime_and_knobs():
    """runtime + exported-model build knobs reach from_export_dir."""
    pred = _mock_predictor()
    with patch(
        "sleap_nn.inference.predictor.Predictor.from_export_dir",
        return_value=pred,
    ) as mock_export:
        predict(
            "video.mp4",
            export_dir="/exp",
            device="cpu",
            runtime="onnx",
            min_instance_peaks=3,
            min_line_scores=0.5,
            emit_centroid="centroid",
            max_instances=7,
        )
    bk = mock_export.call_args.kwargs
    assert bk["runtime"] == "onnx"
    assert bk["min_instance_peaks"] == 3
    assert bk["min_line_scores"] == 0.5
    assert bk["emit_centroid"] == "centroid"
    assert bk["max_instances"] == 7


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


# ── output_format / analysis HDF5 export ───────────────────────────────


def _toy_predicted_labels(video_filenames, frames_per_video):
    """Build a Labels with empty-instance frames for the given videos.

    ``frames_per_video[i]`` LabeledFrames are attached to video ``i``. The
    analysis-HDF5 path logic only counts frames per video (it does not inspect
    instances), so empty instance lists are sufficient for naming/skip tests.
    """
    skeleton = sio.Skeleton(["A", "B"])
    videos = [sio.Video.from_filename(fn) for fn in video_filenames]
    labeled_frames = []
    for vi, n in enumerate(frames_per_video):
        for frame_idx in range(n):
            labeled_frames.append(
                sio.LabeledFrame(video=videos[vi], frame_idx=frame_idx, instances=[])
            )
    return sio.Labels(
        videos=videos, skeletons=[skeleton], labeled_frames=labeled_frames
    )


def test_save_predictions_invalid_format_raises():
    """An unknown output_format raises ValueError before any write."""
    with pytest.raises(ValueError, match="Invalid output_format"):
        save_predictions(MagicMock(), "out.slp", output_format="csv")


def test_save_predictions_both_writes_slp_and_h5(minimal_instance, tmp_path):
    """output_format='both' writes both a .slp and an analysis .h5 that round-trip."""
    labels = sio.load_slp(minimal_instance.as_posix())
    out = tmp_path / "preds.slp"
    h5_written = save_predictions(labels, out, output_format="both")
    assert out.exists()
    assert h5_written == [tmp_path / "preds.analysis.h5"]
    assert h5_written[0].exists()
    # Both files round-trip back to Labels.
    assert isinstance(sio.load_slp(out.as_posix()), sio.Labels)
    assert isinstance(sio.load_analysis_h5(h5_written[0].as_posix()), sio.Labels)


def test_save_predictions_analysis_h5_only_skips_slp(minimal_instance, tmp_path):
    """output_format='analysis_h5' writes only the .h5 (no .slp)."""
    labels = sio.load_slp(minimal_instance.as_posix())
    out = tmp_path / "preds.slp"
    h5_written = save_predictions(labels, out, output_format="analysis_h5")
    assert not out.exists()
    assert h5_written == [tmp_path / "preds.analysis.h5"]
    assert h5_written[0].exists()


def test_save_analysis_h5_files_one_per_video_with_names(tmp_path):
    """Multi-video predictions write one .h5 per video, with names embedded."""
    labels = _toy_predicted_labels(
        ["/data/sessionA.mp4", "/data/sessionB.mp4"], frames_per_video=[2, 3]
    )
    out = tmp_path / "out.predictions.slp"
    with patch("sleap_nn.inference.run.sio.save_analysis_h5") as mock_save:
        written = save_analysis_h5_files(labels, out)
    assert written == [
        tmp_path / "out.sessionA.analysis.h5",
        tmp_path / "out.sessionB.analysis.h5",
    ]
    assert [c.kwargs["video"] for c in mock_save.call_args_list] == [0, 1]


def test_save_analysis_h5_files_skips_empty_videos(tmp_path):
    """A video with no predicted frames is skipped; the single output is unnamed."""
    labels = _toy_predicted_labels(
        ["/data/sessionA.mp4", "/data/sessionB.mp4"], frames_per_video=[2, 0]
    )
    out = tmp_path / "out.predictions.slp"
    with patch("sleap_nn.inference.run.sio.save_analysis_h5") as mock_save:
        written = save_analysis_h5_files(labels, out)
    # Only the non-empty video 0 is written; single file -> no name embedded.
    assert written == [tmp_path / "out.analysis.h5"]
    assert mock_save.call_count == 1
    assert mock_save.call_args_list[0].kwargs["video"] == 0


def test_save_analysis_h5_files_disambiguates_colliding_names(tmp_path):
    """Videos sharing a filename stem get the video index appended for uniqueness."""
    labels = _toy_predicted_labels(
        ["/a/clip.mp4", "/b/clip.mp4"], frames_per_video=[1, 1]
    )
    out = tmp_path / "out.predictions.slp"
    with patch("sleap_nn.inference.run.sio.save_analysis_h5"):
        written = save_analysis_h5_files(labels, out)
    assert written == [
        tmp_path / "out.clip_0.analysis.h5",
        tmp_path / "out.clip_1.analysis.h5",
    ]
    assert len(set(written)) == 2


def test_predict_forwards_output_format(tmp_path):
    """predict() passes output_format through to save_predictions."""
    pred = _mock_predictor()
    out = tmp_path / "out.slp"
    with (
        patch(
            "sleap_nn.inference.predictor.Predictor.from_model_paths",
            return_value=pred,
        ),
        patch("sleap_nn.inference.run.save_predictions") as mock_save,
    ):
        predict(
            "video.mp4",
            model_paths=["/m"],
            device="cpu",
            output_path=str(out),
            output_format="both",
        )
    assert mock_save.call_args.kwargs["output_format"] == "both"
