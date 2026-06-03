"""Regression tests for issue #610 (inference CLI observability).

This PR (PR-A) covers two of the four asks:

* **Frame-based progress** — ``Predictor._batch_iter`` reports
  ``(processed_frames, total_frames)`` (a running sum of batch sizes) instead
  of batch indices, so the count / %% / ETA are batch-size-invariant. This
  relies on each ``Provider`` exposing ``num_frames()``.
* **Windowed FPS column** — ``sleap_nn.cli._make_fps_column`` builds a Rich
  column reporting frames/sec over a trailing time window.

(Spin-up logging + run summary are PR-B.)
"""

from __future__ import annotations

import types
from unittest import mock

import numpy as np

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor, _safe_num_frames
from sleap_nn.inference.providers import (
    MultiVideoProvider,
    NumpyProvider,
)

# ─────────────────────────────────────────────────────────────────────────
# Provider.num_frames() — frame count is known up front, batch-size-invariant
# ─────────────────────────────────────────────────────────────────────────


def test_numpy_provider_num_frames_is_batch_size_invariant():
    """``num_frames`` counts frames, not batches — invariant to batch_size."""
    images = np.zeros((7, 1, 4, 4), dtype=np.float32)
    for bs in (1, 3, 4, 10):
        provider = NumpyProvider(images=images, batch_size=bs)
        assert provider.num_frames() == 7  # invariant
        # ...whereas __len__ (batch count) does vary with batch_size:
        assert len(provider) == (7 + bs - 1) // bs


def test_multivideo_provider_num_frames_sums_subproviders():
    """``MultiVideoProvider`` sums its sub-providers' frame counts."""
    a = NumpyProvider(images=np.zeros((5, 1, 4, 4), dtype=np.float32), batch_size=2)
    b = NumpyProvider(images=np.zeros((3, 1, 4, 4), dtype=np.float32), batch_size=2)
    multi = MultiVideoProvider(providers=[a, b])
    assert multi.num_frames() == 8


def test_safe_num_frames_falls_back_to_minus_one():
    """``_safe_num_frames`` returns -1 for providers without ``num_frames``."""

    class _NoFrames:
        def __len__(self):
            return 3

    assert _safe_num_frames(_NoFrames()) == -1
    assert _safe_num_frames(object()) == -1
    assert (
        _safe_num_frames(NumpyProvider(images=np.zeros((4, 1, 2, 2), dtype=np.float32)))
        == 4
    )


# ─────────────────────────────────────────────────────────────────────────
# _batch_iter — progress callback reports cumulative FRAMES against total
# ─────────────────────────────────────────────────────────────────────────


class _FakeLayer:
    """Minimal layer: echoes an empty ``Outputs`` per batch."""

    def predict(self, images):
        return Outputs()


def test_batch_iter_reports_frame_based_progress():
    """Callback gets a running frame sum + total frames (ragged last batch ok)."""
    images = np.zeros((7, 1, 4, 4), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=3)  # batches of 3, 3, 1
    calls: list = []
    predictor = Predictor(layer=_FakeLayer())

    # Identity filter pipeline so we exercise only the counting logic.
    identity = property(lambda self: (lambda outputs: outputs))
    with mock.patch.object(Predictor, "filter_pipeline", identity):
        outs = list(predictor._batch_iter(provider, lambda p, t: calls.append((p, t))))

    assert len(outs) == 3  # one Outputs per batch
    # Cumulative frames, NOT batch indices; total is frame count (7), not 3.
    assert calls == [(3, 7), (6, 7), (7, 7)]


def test_batch_iter_total_is_minus_one_for_lengthless_provider():
    """A provider without ``num_frames`` still drives a running frame count."""

    class _StreamProvider:
        """Length-less provider (e.g. a live source)."""

        def __init__(self, batches):
            self._batches = batches

        def __iter__(self):
            return iter(self._batches)

    batch = types.SimpleNamespace(
        images=np.zeros((2, 1, 4, 4), dtype=np.float32),
        instances=None,
        frame_indices=None,
        video_indices=None,
    )
    provider = _StreamProvider([batch, batch])
    calls: list = []
    predictor = Predictor(layer=_FakeLayer())
    identity = property(lambda self: (lambda outputs: outputs))
    with mock.patch.object(Predictor, "filter_pipeline", identity):
        list(predictor._batch_iter(provider, lambda p, t: calls.append((p, t))))

    # total unknown (-1) but processed still accumulates in frames.
    assert calls == [(2, -1), (4, -1)]


# ─────────────────────────────────────────────────────────────────────────
# _make_fps_column — windowed frames/sec
# ─────────────────────────────────────────────────────────────────────────


def _task(task_id, completed):
    return types.SimpleNamespace(id=task_id, completed=completed)


def test_fps_column_renders_placeholder_until_two_samples():
    """First render (one sample) shows the ``--`` placeholder."""
    from sleap_nn.cli import _make_fps_column

    col = _make_fps_column(time_fn=lambda: 0.0)
    assert "--" in col.render(_task(1, 0)).plain


def test_fps_column_computes_windowed_rate():
    """Rate is (Δframes / Δt) over the retained window."""
    from sleap_nn.cli import _make_fps_column

    times = iter([0.0, 1.0, 2.0])
    col = _make_fps_column(window_s=5.0, time_fn=lambda: next(times))
    col.render(_task(1, 0))  # t=0, c=0  → placeholder
    assert "10.0 fps" in col.render(_task(1, 10)).plain  # (10-0)/(1-0)
    assert "15.0 fps" in col.render(_task(1, 30)).plain  # (30-0)/(2-0)


def test_fps_column_drops_samples_outside_window():
    """Samples older than ``window_s`` are evicted, so the rate is recent."""
    from sleap_nn.cli import _make_fps_column

    times = iter([0.0, 1.0, 10.0, 11.0])
    col = _make_fps_column(window_s=5.0, time_fn=lambda: next(times))
    col.render(_task(1, 0))  # t=0
    col.render(_task(1, 100))  # t=1
    # t=10 evicts the t=0 and t=1 samples (both > 5s old) → only this sample.
    assert "--" in col.render(_task(1, 1000)).plain
    # t=11 with the t=10 sample retained: (1100-1000)/(11-10) = 100 fps.
    assert "100.0 fps" in col.render(_task(1, 1100)).plain


def test_fps_column_keeps_per_task_history():
    """Two tasks (Predicting / Tracking) keep independent sample windows."""
    from sleap_nn.cli import _make_fps_column

    times = iter([0.0, 0.0, 1.0, 1.0])
    col = _make_fps_column(window_s=5.0, time_fn=lambda: next(times))
    col.render(_task(1, 0))  # task 1 @ t=0
    col.render(_task(2, 0))  # task 2 @ t=0
    assert "5.0 fps" in col.render(_task(1, 5)).plain  # task1: (5-0)/(1-0)
    assert "20.0 fps" in col.render(_task(2, 20)).plain  # task2: (20-0)/(1-0)
