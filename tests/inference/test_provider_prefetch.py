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

    Asserted by MECHANISM, not by wall clock. The previous version timed a
    prefetch run against a synchronous one and required the prefetch arm to win
    by 50 ms -- smaller than the scheduling jitter on a loaded CI runner, which
    made it a coin flip there (observed on macos-14: prefetch=1.091s vs
    sync=1.082s, red on one run and green on a re-run of the same commit).

    Instead, the slow-decode stub records whether it was decoding *while the
    consumer was busy*. That IS the property under test -- decode overlapping
    consumer work -- and it holds regardless of how fast or loaded the machine
    is: with a background decode thread some frame must be decoded during a
    consumer's turn, and without one, none can be (the synchronous path decodes
    strictly between iterations).
    """
    decode_delay = 0.02
    consumer_delay = 0.02
    n_batches = 5

    def run(prefetch: bool):
        """Return (n_decodes, n_decodes_overlapping_consumer_work, elapsed)."""
        consumer_busy = threading.Event()
        lock = threading.Lock()
        decodes: list[int] = []
        overlapping: list[int] = []

        class SlowVideo:
            def __init__(self, video):
                self._video = video

            def __getitem__(self, i):
                # A decode counts as overlapping if the consumer was busy when it
                # started OR was still busy when it finished -- either way the two
                # ran concurrently.
                busy_at_start = consumer_busy.is_set()
                time.sleep(decode_delay)
                with lock:
                    decodes.append(i)
                    if busy_at_start or consumer_busy.is_set():
                        overlapping.append(i)
                return self._video[i]

            def __len__(self):
                return len(self._video)

            def close(self):
                return self._video.close()

            def __deepcopy__(self, memo):
                return SlowVideo(deepcopy(self._video, memo))

        provider = VideoProvider(
            str(VIDEO),
            batch_size=4,
            frames=list(range(n_batches * 4)),
            prefetch=prefetch,
        )
        provider._sio_video = SlowVideo(provider._sio_video)
        t0 = time.monotonic()
        for _ in provider:
            consumer_busy.set()
            time.sleep(consumer_delay)
            consumer_busy.clear()
        elapsed = time.monotonic() - t0
        with lock:
            return len(decodes), len(overlapping), elapsed

    n_pre, overlap_pre, elapsed_pre = run(prefetch=True)
    n_sync, overlap_sync, elapsed_sync = run(prefetch=False)

    # Both arms must have actually decoded the same frames.
    assert n_pre == n_sync == n_batches * 4

    # The synchronous path cannot overlap: it decodes strictly between the
    # consumer's turns.
    assert overlap_sync == 0, (
        f"synchronous decode overlapped consumer work {overlap_sync} times, which "
        "means the test's own instrumentation is wrong, not the provider"
    )

    # The prefetching path must decode while the consumer is working. Timings are
    # reported for diagnostics only -- they are NOT the assertion.
    assert overlap_pre > 0, (
        "expected prefetch to decode while the consumer was busy, but none of "
        f"{n_pre} decodes overlapped "
        f"(prefetch={elapsed_pre:.3f}s, sync={elapsed_sync:.3f}s)"
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
