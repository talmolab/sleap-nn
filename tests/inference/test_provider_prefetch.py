"""Tests for background-thread frame prefetching in the inference providers.

Restores the legacy ``VideoReader``/``LabelsReader`` producer-consumer
pattern (``sleap_nn/data/providers.py``) that overlapped CPU frame decode
with the GPU forward pass, which was dropped in the inference-pipeline
refactor (#508/#530). Covers: output parity with/without prefetch, real
wall-clock overlap (not just correctness), exception propagation from the
background thread to the consumer, no thread leak on early iterator
exit, ``MultiVideoProvider`` still decoding only one source ahead of
time, and the ``paf_workers``-on-unsupported-layer warning.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.providers import (
    LabelsProvider,
    MultiVideoProvider,
    VideoProvider,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"
VIDEO = DATA_ROOT / "small_robot.mp4"
LABELS = DATA_ROOT / "minimal_instance.pkg.slp"


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


def _active_thread_count() -> int:
    return threading.active_count()


# ─────────────────────────────────────────────────────────────────────────
# VideoProvider
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_prefetch_matches_synchronous_output():
    """prefetch=True yields byte-identical batches to prefetch=False."""
    frames = list(range(20))
    sync_batches = list(
        VideoProvider(str(VIDEO), batch_size=4, frames=frames, prefetch=False)
    )
    pre_batches = list(
        VideoProvider(str(VIDEO), batch_size=4, frames=frames, prefetch=True)
    )
    assert len(sync_batches) == len(pre_batches)
    for a, b in zip(sync_batches, pre_batches):
        np.testing.assert_array_equal(a.images, b.images)
        np.testing.assert_array_equal(a.frame_indices, b.frame_indices)
        np.testing.assert_array_equal(a.video_indices, b.video_indices)


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_prefetch_overlaps_decode_with_consumer_work():
    """Prefetch must actually overlap decode with consumer work, not just be correct.

    Wraps frame access with an artificial per-frame delay (standing in for a
    slow decode) and times the full iterate-and-consume loop (with an
    artificial per-batch consumer delay, standing in for a GPU forward pass)
    against the synchronous baseline. True overlap makes the prefetching
    version meaningfully faster; without it, this is just a no-op wrapper.
    """
    decode_delay = 0.03
    consumer_delay = 0.03
    n_batches = 5

    class SlowVideo:
        def __init__(self, video):
            self._video = video

        def __getitem__(self, i):
            time.sleep(decode_delay / 4)
            return self._video[i]

        def __len__(self):
            return len(self._video)

        def close(self):
            return self._video.close()

        def __deepcopy__(self, memo):
            return SlowVideo(deepcopy(self._video, memo))

    def timed_run(prefetch: bool) -> float:
        provider = VideoProvider(
            str(VIDEO),
            batch_size=4,
            frames=list(range(n_batches * 4)),
            prefetch=prefetch,
        )
        provider._sio_video = SlowVideo(provider._sio_video)
        t0 = time.monotonic()
        for _ in provider:
            time.sleep(consumer_delay)
        return time.monotonic() - t0

    elapsed_prefetch = timed_run(True)
    elapsed_sync = timed_run(False)
    assert elapsed_prefetch < elapsed_sync - 0.05, (
        f"expected prefetch to overlap decode with consumer work: "
        f"prefetch={elapsed_prefetch:.3f}s, sync={elapsed_sync:.3f}s"
    )


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_prefetch_propagates_decode_exception():
    """A mid-stream decode failure must raise on the consumer.

    Not be swallowed as a silent (legacy-bug) end-of-stream.
    """
    provider = VideoProvider(str(VIDEO), batch_size=4, frames=list(range(20)))
    real_video = provider._sio_video

    class FlakyVideo:
        def __init__(self, video):
            self._video = video

        def __getitem__(self, i):
            if i == 8:
                raise RuntimeError("simulated decode failure")
            return self._video[i]

        def __len__(self):
            return len(self._video)

        def close(self):
            return self._video.close()

        def __deepcopy__(self, memo):
            return FlakyVideo(deepcopy(self._video, memo))

    provider._sio_video = FlakyVideo(real_video)

    seen = 0
    with pytest.raises(RuntimeError, match="simulated decode failure"):
        for _ in provider:
            seen += 1
    assert seen == 2  # batches [0-3], [4-7] succeed before frame 8 fails


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_prefetch_fills_queue_and_retries_on_full():
    """A stalled consumer forces the producer's ``queue.Full`` retry loop.

    The producer's ``q.put(batch, timeout=0.5)`` only raises ``Full`` (and
    loops back to retry) once that 0.5s window actually elapses with the
    queue still full, so the consumer must stall past it at least once.
    """
    provider = VideoProvider(
        str(VIDEO), batch_size=1, frames=list(range(4)), queue_maxsize=1
    )
    it = iter(provider)
    next(it)  # producer fills the size-1 queue with the next batch, then blocks
    time.sleep(0.6)  # outlast the 0.5s put timeout at least once
    rest = list(it)
    assert len(rest) == 3


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_prefetch_no_thread_leak_on_early_exit():
    """Breaking out of iteration early must not leak the background thread."""
    before = _active_thread_count()
    provider = VideoProvider(
        str(VIDEO), batch_size=2, frames=list(range(40)), queue_maxsize=1
    )
    it = iter(provider)
    next(it)
    it.close()
    time.sleep(0.2)
    assert _active_thread_count() <= before


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_multivideo_provider_prefetches_one_source_at_a_time():
    """Only one sub-provider's prefetch thread is alive at once.

    Preserves today's sequential, one-source-at-a-time decode behavior.
    """
    p1 = VideoProvider(str(VIDEO), batch_size=2, frames=list(range(6)))
    p2 = VideoProvider(str(VIDEO), batch_size=2, frames=list(range(6)))
    mv = MultiVideoProvider(providers=[p1, p2])

    before = _active_thread_count()
    max_extra = 0
    for _ in mv:
        max_extra = max(max_extra, _active_thread_count() - before)
    time.sleep(0.2)
    assert (
        max_extra == 1
    ), f"expected exactly 1 concurrent prefetch thread, saw {max_extra}"
    assert _active_thread_count() == before


# ─────────────────────────────────────────────────────────────────────────
# LabelsProvider
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not LABELS.exists(), reason="test labels file not present")
def test_labels_provider_prefetch_matches_synchronous_output():
    """prefetch=True yields byte-identical batches to prefetch=False."""
    kwargs = dict(labels=str(LABELS), batch_size=2, only_labeled_frames=False)
    sync_batches = list(LabelsProvider(**kwargs, prefetch=False))
    pre_batches = list(LabelsProvider(**kwargs, prefetch=True))
    assert len(sync_batches) == len(pre_batches)
    for a, b in zip(sync_batches, pre_batches):
        np.testing.assert_array_equal(a.images, b.images)
        np.testing.assert_array_equal(a.frame_indices, b.frame_indices)
        np.testing.assert_array_equal(a.video_indices, b.video_indices)
        if a.instances is None:
            assert b.instances is None
        else:
            np.testing.assert_allclose(a.instances, b.instances, equal_nan=True)


@pytest.mark.skipif(not LABELS.exists(), reason="test labels file not present")
def test_labels_provider_prefetch_propagates_decode_exception():
    """A mid-stream decode failure raises on the consumer, not a silent stop."""
    provider = LabelsProvider(
        labels=str(LABELS), batch_size=2, only_labeled_frames=False
    )
    real_video = provider._sio_labels.videos[0]

    class FlakyVideo:
        def __init__(self, video):
            self._video = video

        def __getitem__(self, i):
            raise RuntimeError("simulated decode failure")

        def close(self):
            return self._video.close()

        def __deepcopy__(self, memo):
            return FlakyVideo(deepcopy(self._video, memo))

    # attrs(slots=True) instances forbid ad hoc attributes; patch the class
    # method for the duration of the test instead.
    orig_map = type(provider)._thread_local_video_map

    def patched_map(self):
        return {k: FlakyVideo(v) for k, v in orig_map(self).items()}

    type(provider)._thread_local_video_map = patched_map
    try:
        with pytest.raises(RuntimeError, match="simulated decode failure"):
            list(provider)
    finally:
        type(provider)._thread_local_video_map = orig_map


@pytest.mark.skipif(not LABELS.exists(), reason="test labels file not present")
def test_labels_provider_prefetch_no_thread_leak_on_early_exit():
    """Breaking out of iteration early must not leak the background thread."""
    before = _active_thread_count()
    provider = LabelsProvider(
        labels=str(LABELS), batch_size=1, only_labeled_frames=False, queue_maxsize=1
    )
    it = iter(provider)
    next(it)
    it.close()
    time.sleep(0.2)
    assert _active_thread_count() <= before


# ─────────────────────────────────────────────────────────────────────────
# Predictor: paf_workers on an unsupported layer warns instead of silently
# no-op'ing.
# ─────────────────────────────────────────────────────────────────────────


class _FakeLayer:
    """Stand-in for any non-``BottomUpLayer`` layer."""


def test_predictor_warns_when_paf_workers_set_on_unsupported_layer(caplog):
    """paf_workers>0 on a non-BottomUpLayer warns instead of silently no-op'ing."""
    Predictor(layer=_FakeLayer(), paf_workers=4)
    assert "paf_workers=4" in caplog.text
    assert "_FakeLayer" in caplog.text


def test_predictor_no_warning_when_paf_workers_zero(caplog):
    """The default paf_workers=0 emits no warning."""
    Predictor(layer=_FakeLayer(), paf_workers=0)
    assert "paf_workers" not in caplog.text
