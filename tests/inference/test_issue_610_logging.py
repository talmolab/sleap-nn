"""Tests for issue #610 PR-B — inference spin-up + run-summary logging.

The ``Predictor`` now emits three ``loguru`` lines per run (library-wide, so
the ``sleap`` frontend benefits too):

* ``Loaded inference model | ...`` — at ``from_model_paths`` (the spin-up
  header: model type / backbone / nodes / device / thresholds / ...).
* ``Starting inference | ...`` — once the provider is built (source, frames,
  video shape, fps, tracking).
* ``Inference complete | ...`` — after the run (frames, instances/masks,
  elapsed, throughput, tracking, output).

Fast unit tests drive the formatting helpers directly; one integration test
exercises a real ``predict()`` so all three lines are asserted end-to-end.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.providers import NumpyProvider

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
VIDEO = Path(__file__).resolve().parents[1] / "assets" / "datasets" / "small_robot.mp4"
SI_CKPT = CKPT_ROOT / "minimal_instance_single_instance"


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru records into pytest's ``caplog`` (project convention)."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


class _FakeLayer:
    """Minimal non-segmentation layer (so ``_is_segmentation_layer`` is False)."""


# ─────────────────────────────────────────────────────────────────────────
# _describe_source
# ─────────────────────────────────────────────────────────────────────────


def test_describe_source_variants():
    """A str passes through; an object's ``filename`` wins; else the type name."""
    assert Predictor._describe_source("video.mp4") == "video.mp4"
    obj = types.SimpleNamespace(filename="/path/to/clip.mp4")
    assert Predictor._describe_source(obj) == "/path/to/clip.mp4"
    # No str, no filename → falls back to the type name.
    assert Predictor._describe_source(object()) == "object"


# ─────────────────────────────────────────────────────────────────────────
# _log_inference_start
# ─────────────────────────────────────────────────────────────────────────


def test_log_inference_start_basic(caplog):
    """The start line carries source, frame count, video count, tracking."""
    predictor = Predictor(layer=_FakeLayer())
    provider = NumpyProvider(images=np.zeros((5, 1, 4, 4), dtype=np.float32))
    predictor._log_inference_start("clip.mp4", provider, videos=None)
    msg = caplog.text
    assert "Starting inference" in msg
    assert "source=clip.mp4" in msg
    assert "frames=5" in msg
    assert "videos=1" in msg
    assert "tracking=False" in msg


def test_log_inference_start_includes_video_shape_and_fps(caplog):
    """A provider with ``_sio_video`` contributes HxWxC shape + fps."""
    predictor = Predictor(layer=_FakeLayer())
    # A provider exposing _sio_video with shape (N, H, W, C) + fps.
    fake_video = types.SimpleNamespace(shape=(10, 64, 48, 3), fps=30.0)
    provider = types.SimpleNamespace(num_frames=lambda: 10, _sio_video=fake_video)
    predictor._log_inference_start("v.mp4", provider, videos=[fake_video])
    msg = caplog.text
    assert "shape=64x48x3" in msg
    assert "fps=30.0" in msg
    assert "videos=1" in msg


def test_log_inference_start_reports_tracking_enabled(caplog):
    """A set ``tracker_config`` surfaces as ``tracking=True``."""
    predictor = Predictor(layer=_FakeLayer(), tracker_config=object())
    provider = NumpyProvider(images=np.zeros((2, 1, 4, 4), dtype=np.float32))
    predictor._log_inference_start("x", provider, videos=None)
    assert "tracking=True" in caplog.text


def test_log_inference_start_unknown_frame_count(caplog):
    """A length-less provider logs ``frames=?`` rather than -1."""
    predictor = Predictor(layer=_FakeLayer())
    provider = object()  # no num_frames → _safe_num_frames returns -1
    predictor._log_inference_start("src", provider, videos=None)
    assert "frames=?" in caplog.text


# ─────────────────────────────────────────────────────────────────────────
# _log_inference_summary
# ─────────────────────────────────────────────────────────────────────────


def test_log_inference_summary_with_instances(caplog):
    """The summary reports instances + per-frame mean + windowed throughput."""
    predictor = Predictor(layer=_FakeLayer())
    predictor._log_inference_summary(
        n_frames=10, elapsed_s=2.0, n_objects=25, object_label="instances"
    )
    msg = caplog.text
    assert "Inference complete" in msg
    assert "frames=10" in msg
    assert "instances=25 (2.50/frame)" in msg
    assert "throughput=5.0 fps" in msg  # 10 frames / 2.0 s
    assert "tracking=False" in msg


def test_log_inference_summary_streaming_frames_only(caplog):
    """Streaming summary omits object counts but reports output + throughput."""
    predictor = Predictor(layer=_FakeLayer())
    predictor._log_inference_summary(n_frames=4, elapsed_s=1.0, output="out.slp")
    msg = caplog.text
    assert "frames=4" in msg
    assert "instances=" not in msg and "masks=" not in msg
    assert "output=out.slp" in msg
    assert "throughput=4.0 fps" in msg


def test_log_inference_summary_zero_elapsed_no_div_by_zero(caplog):
    """A zero elapsed time reports 0 fps instead of dividing by zero."""
    predictor = Predictor(layer=_FakeLayer())
    predictor._log_inference_summary(n_frames=3, elapsed_s=0.0, n_objects=3)
    assert "throughput=0.0 fps" in caplog.text


def test_log_inference_summary_masks_label(caplog):
    """Segmentation runs report ``masks`` rather than ``instances``."""
    predictor = Predictor(layer=_FakeLayer())
    predictor._log_inference_summary(
        n_frames=2, elapsed_s=1.0, n_objects=6, object_label="masks"
    )
    assert "masks=6 (3.00/frame)" in caplog.text


# ─────────────────────────────────────────────────────────────────────────
# Integration: a real predict() emits all three lines
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (VIDEO.exists() and SI_CKPT.exists()),
    reason="missing single-instance fixtures",
)
def test_predict_emits_spinup_start_and_summary(caplog):
    """End-to-end: from_model_paths + predict log all three observability lines."""
    predictor = Predictor.from_model_paths([str(SI_CKPT)], device="cpu", batch_size=4)
    assert "Loaded inference model" in caplog.text
    assert "type=single_instance" in caplog.text
    assert "backbone=" in caplog.text
    assert "nodes=" in caplog.text

    video = sio.load_video(str(VIDEO))
    labels = predictor.predict(video, frames=list(range(6)))
    msg = caplog.text
    assert "Starting inference" in msg
    assert "frames=6" in msg
    assert "Inference complete" in msg
    assert "instances=" in msg  # in-memory path counts instances
    assert "throughput=" in msg
    assert isinstance(labels, sio.Labels)
