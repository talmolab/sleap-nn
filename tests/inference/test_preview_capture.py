"""Tests for in-memory preview frame capture during inference."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import sleap_io as sio
import torch

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.preview import PreviewCollector
from sleap_nn.inference.providers import Batch, NumpyProvider
from sleap_nn.inference.run import save_predictions

_requires_sio_preview = pytest.mark.skipif(
    not hasattr(sio, "PreviewFrame"),
    reason="requires sleap-io with PreviewFrame support",
)


class _StubPoseLayer:
    """Minimal InferenceLayer stub returning deterministic pose predictions."""

    def __init__(self, *, empty_first: bool = True) -> None:
        self.empty_first = empty_first
        self.calls = 0

    def predict(self, images):
        b = len(images)
        kpts = torch.full((b, 1, 2, 2), float("nan"), dtype=torch.float32)
        scores = torch.full((b, 1, 2), float("nan"), dtype=torch.float32)
        inst_scores = torch.full((b, 1), float("nan"), dtype=torch.float32)
        valid = torch.zeros((b, 1), dtype=torch.bool)
        # Leave frame 0 empty when empty_first=True to verify we pick frame 1.
        start = 1 if (self.empty_first and self.calls == 0 and b > 1) else 0
        for i in range(start, b):
            kpts[i, 0] = torch.tensor([[12.0, 24.0], [36.0, 48.0]])
            scores[i, 0] = torch.tensor([0.9, 0.85])
            inst_scores[i, 0] = 0.88
            valid[i, 0] = True
        self.calls += 1
        return Outputs(
            pred_keypoints=kpts,
            pred_peak_values=scores,
            instance_scores=inst_scores,
            instance_valid=valid,
        )


def _make_stub_predictor(empty_first: bool = True) -> Predictor:
    skel = sio.Skeleton(["head", "tail"])
    return Predictor(
        layer=_StubPoseLayer(empty_first=empty_first),
        skeleton=skel,
        batch_size=2,
    )


@_requires_sio_preview
def test_preview_collector_prefers_first_predicted_frame_and_falls_back():
    """PreviewCollector selects first frame with a prediction, else frame 0."""
    video = sio.Video(filename="fake.mp4", open_backend=False)
    imgs = np.zeros((3, 120, 160, 3), dtype=np.uint8)
    imgs[0, :, :, 0] = 50
    imgs[1, :, :, 1] = 200  # frame with prediction
    imgs[2, :, :, 2] = 255
    batch = Batch(
        images=imgs,
        frame_indices=np.array([10, 11, 12], dtype=np.int64),
        video_indices=np.array([0, 0, 0], dtype=np.int64),
    )
    kpts = torch.full((3, 1, 2, 2), float("nan"))
    kpts[1, 0] = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    outputs = Outputs(
        pred_keypoints=kpts,
        frame_indices=torch.tensor([10, 11, 12]),
        video_indices=torch.tensor([0, 0, 0]),
    )

    collector = PreviewCollector()
    collector.observe(batch, outputs)
    assert collector.has_selected

    pfs = collector.to_preview_frames([video])
    assert len(pfs) == 1
    assert pfs[0].video is video
    assert pfs[0].frame_idx == 11
    assert (pfs[0].orig_width, pfs[0].orig_height) == (160, 120)
    assert pfs[0].data[:2] == b"\xff\xd8"

    # Fallback when no predictions exist at all.
    empty_collector = PreviewCollector()
    empty_outputs = Outputs(
        pred_keypoints=torch.full((3, 1, 2, 2), float("nan")),
        frame_indices=torch.tensor([10, 11, 12]),
        video_indices=torch.tensor([0, 0, 0]),
    )
    empty_collector.observe(batch, empty_outputs)
    assert not empty_collector.has_selected
    fallback_pfs = empty_collector.to_preview_frames([video])
    assert len(fallback_pfs) == 1
    assert fallback_pfs[0].frame_idx == 10


@_requires_sio_preview
def test_preview_collector_handles_chw_and_fitted_frame_sizes():
    """CHW tensors and pre-fitted frame_sizes/frame_scales preserve original dims."""
    video = sio.Video(filename="fake.mp4", open_backend=False)
    # CHW 3x200x200 padded frame representing an original 1000x500 image scaled 0.2x
    chw = torch.full((1, 3, 200, 200), 128, dtype=torch.uint8)
    batch = Batch(
        images=chw,
        frame_indices=np.array([5], dtype=np.int64),
        video_indices=np.array([0], dtype=np.int64),
        frame_sizes=np.array([[1000, 500]], dtype=np.int64),
        frame_scales=np.array([[0.2, 0.2]], dtype=np.float32),
    )
    outputs = Outputs(
        pred_bboxes=torch.tensor([[[10.0, 20.0, 100.0, 200.0]]]),
        frame_indices=torch.tensor([5]),
        video_indices=torch.tensor([0]),
    )
    collector = PreviewCollector()
    collector.observe(batch, outputs)
    pfs = collector.to_preview_frames([video])
    assert len(pfs) == 1
    assert pfs[0].frame_idx == 5
    assert (pfs[0].orig_width, pfs[0].orig_height) == (500, 1000)
    assert (pfs[0].width, pfs[0].height) == (100, 200)


@_requires_sio_preview
def test_predictor_predict_and_save_predictions_writes_preview(tmp_path):
    """Predictor.predict() attaches preview_frames and save_predictions() persists /preview."""
    video = sio.Video(filename="nonexistent.mp4", open_backend=False)
    imgs = np.full((3, 96, 128, 3), 90, dtype=np.uint8)
    imgs[1, 10:50, 10:50, 0] = 240
    provider = NumpyProvider(
        images=imgs,
        batch_size=3,
        frame_indices=np.array([0, 1, 2], dtype=np.int64),
    )

    predictor = _make_stub_predictor(empty_first=True)
    labels = predictor.predict(provider, videos=[video], make_labels=True)

    assert len(labels.preview_frames) == 1
    assert labels.preview_frames[0].video is labels.videos[0]
    assert labels.preview_frames[0].frame_idx == 1  # skipped empty frame 0

    out_slp = tmp_path / "preds.slp"
    save_predictions(labels, out_slp, embed="false")
    with h5py.File(out_slp, "r") as f:
        assert "preview" in f
        assert int(f["preview"]["frame_idx"][0]) == 1
        assert int(f["preview"]["orig_width"][0]) == 128
        assert int(f["preview"]["orig_height"][0]) == 96


@_requires_sio_preview
def test_predictor_predict_to_file_writes_preview(tmp_path):
    """Predictor.predict_to_file() captures and writes /preview into the streamed .slp."""
    video = sio.Video(filename="nonexistent.mp4", open_backend=False)
    imgs = np.full((2, 80, 100, 3), 120, dtype=np.uint8)
    provider = NumpyProvider(images=imgs, batch_size=2)
    predictor = _make_stub_predictor(empty_first=True)

    out_path = tmp_path / "streamed.slp"
    predictor.predict_to_file(provider, str(out_path), videos=[video])

    with h5py.File(out_path, "r") as f:
        assert "preview" in f
        assert int(f["preview"]["frame_idx"][0]) == 1


def test_preview_collector_degrades_gracefully_without_sleap_io_preview(monkeypatch):
    """Older sleap-io without PreviewFrame returns [] without error."""
    video = sio.Video(filename="fake.mp4", open_backend=False)
    batch = Batch(
        images=np.zeros((1, 32, 32, 3), dtype=np.uint8),
        frame_indices=np.array([0], dtype=np.int64),
    )
    collector = PreviewCollector()
    collector.observe(batch, Outputs())
    monkeypatch.setattr(sio, "PreviewFrame", None)
    assert collector.to_preview_frames([video]) == []
