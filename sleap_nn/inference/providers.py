"""``Provider`` protocol + concrete data sources for ``Predictor``.

A ``Provider`` yields batches of raw images plus per-batch metadata
(frame indices, video indices, optionally GT instances). The
``Predictor`` consumes these batches and routes them through an
``InferenceLayer``.

Three concrete implementations:

* :class:`NumpyProvider` — emits an in-memory tensor batch as a single
  iterate. Right for testing, real-time loops, or when the caller has
  already loaded frames.
* :class:`VideoProvider` — wraps a video path; yields frames in batches.
  Built on ``sleap_io.Video`` via the existing data-layer reader so we
  don't duplicate decoding.
* :class:`LabelsProvider` — wraps a ``.slp`` file; yields the labeled
  frames + their GT instances (needed for the ``use_gt_centroids`` /
  ``use_gt_peaks`` layer paths).
"""

from __future__ import annotations

import queue
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Iterator, Optional, Protocol, Union

import attrs
import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    import sleap_io as sio


# Sentinel marking end-of-stream on a prefetch queue; a plain `object()` so it
# can never collide with a real `Batch` or exception payload.
_SENTINEL = object()


def _reopen_after_thread_local_copy(video: "sio.Video") -> None:
    """Re-materialize ``video.backend`` after it was closed to make a cheap copy.

    ``_thread_local_video``/``_thread_local_video_map`` close the *shared*
    video (so ``deepcopy`` clones cheap path/config state, not a live handle)
    before handing a private copy to the prefetch thread. Left closed, the
    shared video — which is what ends up in the final output ``Labels`` for
    saving — has ``backend=None``; sleap-io's embedded-image detection in
    ``write_videos`` requires an actually-materialized ``HDF5Video`` instance,
    so a closed embedded video silently falls back to re-serializing stale
    backend metadata (whose ``filename: "."`` self-reference convention is
    only valid inside its *original* file) into the new output file. Reopening
    here restores a live backend on the shared video so save-time embedded
    detection works. Best-effort: if the source is genuinely no longer
    reachable, leave it closed rather than raising out of a prefetch setup path.
    """
    try:
        video.open()
    except Exception as e:
        logger.debug(f"Could not reopen video backend after thread-local copy: {e}")


class _ProducerError:
    """Wraps an exception raised on a prefetch thread for queue delivery.

    `queue.Queue` has no notion of an error channel, so an exception raised
    while decoding on the background thread is boxed and put on the queue
    like any other item; the consumer unwraps and re-raises it. Without this,
    a mid-stream decode failure would look identical to a clean end-of-stream
    (the failure mode the legacy `VideoReader`/`LabelsReader` had).
    """

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


@attrs.frozen
class Batch:
    """One per-batch payload produced by a :class:`Provider`.

    Attributes:
        images: ``(B, ...)`` raw frames (numpy or torch). Shape varies by
            provider; the layer's ``preprocess`` does the canonicalization.
        frame_indices: Optional ``(B,)`` int64 frame indices into the
            source video / labels file.
        video_indices: Optional ``(B,)`` int64 video indices for
            multi-video inputs (constant 0 for single-source providers).
        instances: Optional ``(B, max_instances, n_nodes, 2)`` GT
            instances, populated by :class:`LabelsProvider` for the
            GT-fallback layer paths.
    """

    images: np.ndarray | torch.Tensor
    frame_indices: Optional[np.ndarray] = None
    video_indices: Optional[np.ndarray] = None
    instances: Optional[np.ndarray] = None


class Provider(Protocol):
    """Iterator-of-batches contract that ``Predictor`` consumes."""

    def __iter__(self) -> Iterator[Batch]:
        """Yield ``Batch`` instances until the source is exhausted."""
        ...

    def __len__(self) -> int:
        """Return the total number of batches the provider will yield.

        Used by the ``Predictor`` for progress reporting. Providers over
        unbounded sources (live cameras) may return ``-1`` to signal
        unknown length.
        """
        ...

    def num_frames(self) -> int:
        """Return the total number of frames the provider will yield.

        Used by the ``Predictor`` for frame-based progress reporting, which
        is batch-size-invariant (unlike ``__len__``, which counts batches).
        Providers over unbounded sources may return ``-1`` to signal unknown
        length.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────
# NumpyProvider — emits a pre-loaded tensor as a single batch
# ─────────────────────────────────────────────────────────────────────────


@attrs.define
class VideoProvider:
    """Yield batches from a video file via ``sleap_io.Video``.

    Args:
        video: Path to a video file (``.mp4``, ``.avi``, ``.h5``, etc.) or
            an already-loaded ``sleap_io.Video`` instance.
        batch_size: Number of frames per yielded ``Batch``.
        frames: Optional list of frame indices to read (e.g., ``range(100)``).
            ``None`` reads every frame. Frames are read in the order
            specified — this provider does **not** sort or deduplicate.
        dataset: For HDF5-backed videos, the dataset name (forwarded to
            ``sio.load_video``).
        input_format: For HDF5-backed videos, ``"channels_last"`` or
            ``"channels_first"`` (forwarded to ``sio.load_video``).
        remote_kwargs: Optional mapping of remote-loading options forwarded to
            ``sio.load_video`` when ``video`` is a URL (e.g.
            ``{"headers": {...}, "stream_mode": "..."}``). Ignored for local
            paths and pre-loaded ``sio.Video`` instances.
        prefetch: If ``True`` (default), decode frames on a background
            thread into a bounded queue so CPU decode overlaps the GPU
            forward pass on the consuming (main) thread, instead of
            blocking on decode between batches. Set ``False`` to fall back
            to synchronous, in-line reading (e.g. for deterministic
            single-threaded debugging).
        queue_maxsize: Bound on how many decoded batches may sit in the
            prefetch queue ahead of the consumer. Kept small and explicit
            (like ``paf_workers``' ``max_in_flight``) so a fast decoder
            can't race far ahead of a slow GPU stage and balloon memory.

    Notes:
        Yields raw ``(B, H, W, C)`` frames as ``np.uint8``. The
        layer's ``preprocess`` does the canonicalization
        (e.g., ``ensure_grayscale``).
    """

    video: "Union[str, sio.Video]"
    batch_size: int = 4
    frames: Optional[list[int]] = None
    dataset: Optional[str] = None
    input_format: Optional[str] = None
    remote_kwargs: Optional[dict] = None
    prefetch: bool = True
    queue_maxsize: int = 4

    _sio_video: "Optional[sio.Video]" = attrs.field(
        default=None, init=False, repr=False
    )
    _frame_indices: list[int] = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Resolve the video path → ``sio.Video`` and stash frame indices."""
        import sleap_io as sio

        if isinstance(self.video, sio.Video):
            self._sio_video = self.video
        else:
            kwargs: dict = {}
            if self.dataset is not None:
                kwargs["dataset"] = self.dataset
            if self.input_format is not None:
                kwargs["input_format"] = self.input_format
            if self.remote_kwargs:
                kwargs.update(self.remote_kwargs)
            self._sio_video = sio.load_video(str(self.video), **kwargs)

        n_frames = len(self._sio_video)
        self._frame_indices = (
            list(self.frames) if self.frames is not None else list(range(n_frames))
        )

    def _read_batches(self, video: "sio.Video") -> Iterator[Batch]:
        """Read frames from ``video`` in batches of ``batch_size``.

        Takes the source video as a parameter (rather than always reading
        ``self._sio_video``) so the prefetch thread in ``__iter__`` can pass
        in its own private, thread-local copy instead of sharing a backend
        handle with the main thread.
        """
        for start in range(0, len(self._frame_indices), self.batch_size):
            stop = min(start + self.batch_size, len(self._frame_indices))
            chunk_inds = self._frame_indices[start:stop]
            frames = np.stack([video[i] for i in chunk_inds], axis=0)
            yield Batch(
                images=frames,
                frame_indices=np.asarray(chunk_inds, dtype=np.int64),
                video_indices=np.zeros(len(chunk_inds), dtype=np.int64),
            )

    def _thread_local_video(self) -> "sio.Video":
        """Return a private ``sio.Video`` copy for the prefetch thread.

        Video backends (OpenCV/ffmpeg/HDF5) cache a single stateful reader
        handle and are not safe for concurrent seek+read from two threads —
        sharing ``self._sio_video`` with the background thread can silently
        return the wrong frame. Closing first drops the cached handle so
        ``deepcopy`` clones cheap (path + config) state rather than a live
        handle. The copy lazily reopens its own handle on next access; the
        shared original does not reopen on its own (``backend`` is a plain
        attribute, not a lazy property), so it is explicitly reopened here —
        it is also what ends up in the final output ``Labels`` for saving.
        """
        self._sio_video.close()
        thread_local = deepcopy(self._sio_video)
        _reopen_after_thread_local_copy(self._sio_video)
        return thread_local

    def _prefetch_worker(
        self, video: "sio.Video", q: "queue.Queue", stop_event: threading.Event
    ) -> None:
        """Decode batches from ``video`` onto ``q`` until exhausted or stopped."""
        try:
            for batch in self._read_batches(video):
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        q.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            q.put(_ProducerError(exc))
            return
        q.put(_SENTINEL)

    def __iter__(self) -> Iterator[Batch]:
        """Read frames in batches of ``batch_size`` and yield them.

        When ``prefetch=True`` (default), decoding happens on a background
        thread reading from a private video copy, overlapping CPU decode of
        the next batch with GPU inference on the current one. The thread is
        started here (not at construction) so multi-source callers like
        ``MultiVideoProvider`` only ever have one video being decoded ahead
        of time, matching today's sequential, one-source-at-a-time behavior.
        """
        if not self.prefetch:
            yield from self._read_batches(self._sio_video)
            return

        thread_video = self._thread_local_video()
        q: "queue.Queue" = queue.Queue(maxsize=self.queue_maxsize)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(thread_video, q, stop_event),
            daemon=True,
        )
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    return
                if isinstance(item, _ProducerError):
                    raise item.exc
                yield item
        finally:
            # Unblock the worker if the consumer stops early (exception,
            # break, or GeneratorExit on partial iteration) so it doesn't
            # hang forever on a full queue with no one draining it.
            stop_event.set()
            thread.join(timeout=5.0)

    def __len__(self) -> int:
        """Number of batches; ``ceil(len(frames) / batch_size)``."""
        n = len(self._frame_indices)
        return (n + self.batch_size - 1) // self.batch_size

    def num_frames(self) -> int:
        """Total number of frames this provider will yield."""
        return len(self._frame_indices)

    @property
    def videos(self) -> "list[sio.Video]":
        """The source ``sio.Video``(s), for packaging the output ``Labels``.

        Lets ``Predictor._make_provider`` attach the real video to predicted
        frames when a pre-built provider is passed as the source, instead of a
        ``None`` placeholder that later crashes ``sio.Labels.save`` (#699).
        """
        return [self._sio_video] if self._sio_video is not None else []


@attrs.define
class LabelsProvider:
    """Yield batches from a ``.slp`` file with GT instances attached.

    Used by the GT-fallback layer paths (``CentroidLayer.use_gt_centroids``
    and ``CenteredInstanceLayer.use_gt_peaks``). Each yielded ``Batch``
    carries both the source images **and** the GT instance keypoints
    from the ``.slp`` so the layer can match centroids → GT keypoints
    or build crops from GT centroids without a centroid model.

    Args:
        labels: Path to a ``.slp`` file or an already-loaded
            ``sleap_io.Labels`` instance.
        batch_size: Frames per yielded ``Batch``.
        only_labeled_frames: Yield only frames that have at least one
            user-supplied instance (default ``False``, matching legacy
            ``LabelsReader``). ``Predictor._make_provider`` always passes
            this explicitly (computed from whether the layer needs GT
            instances), so the default only matters for a ``LabelsProvider``
            built directly. Since this is also the highest-priority filter
            (see below), leaving it at a truthy default would silently
            override any other ``only_*``/``exclude_*`` flag a direct caller
            sets without also passing ``only_labeled_frames=False``.
        only_suggested_frames: Yield only frames listed in
            ``labels.suggestions`` that don't already have a user
            instance. Mutually exclusive with the other ``only_*`` /
            ``exclude_*`` modes.
        exclude_user_labeled: Skip any frame that has a user instance.
            Mutually exclusive with ``only_labeled_frames``.
        only_predicted_frames: Yield only frames that already have at
            least one predicted instance.
        remote_kwargs: Optional mapping of remote-loading options forwarded to
            ``sio.load_slp`` when ``labels`` is a URL (e.g.
            ``{"headers": {...}, "stream_mode": "..."}``). Ignored for local
            paths and pre-loaded ``sio.Labels`` instances.
        prefetch: If ``True`` (default), decode frame images on a background
            thread into a bounded queue so CPU decode overlaps the GPU
            forward pass, instead of blocking on decode between batches.
            See :class:`VideoProvider` for the same mechanism.
        queue_maxsize: Bound on how many decoded batches may sit in the
            prefetch queue ahead of the consumer.
        frames: Optional 0-indexed *positions* to keep from the (possibly
            already `only_*`/`exclude_*`-filtered) labeled-frames list, in
            file order -- e.g. ``[0, 1, 2]`` keeps the first three labeled
            frames. NOT a filter on ``LabeledFrame.frame_idx`` values: for a
            `.pkg.slp` with embedded, non-contiguously-sampled frames,
            `frame_idx` is typically NOT sequential (e.g. a cluster-sampled
            training package), so a `frame_idx`-range filter would silently
            match the wrong (often near-empty) subset. Positional selection
            gives a well-defined "first N labeled frames" preview regardless
            of the source video's backing (embedded vs. external). Positions
            beyond the list's length are dropped with a logged warning
            rather than silently ignored.
    """

    labels: "Union[str, sio.Labels]"
    batch_size: int = 4
    only_labeled_frames: bool = False
    only_suggested_frames: bool = False
    exclude_user_labeled: bool = False
    only_predicted_frames: bool = False
    remote_kwargs: Optional[dict] = None
    prefetch: bool = True
    queue_maxsize: int = 4
    frames: Optional[list] = None

    _sio_labels: "Optional[sio.Labels]" = attrs.field(
        default=None, init=False, repr=False
    )
    _labeled_frames: list = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Resolve the labels source and pre-filter the labeled frames."""
        import sleap_io as sio

        if isinstance(self.labels, sio.Labels):
            self._sio_labels = self.labels
        else:
            self._sio_labels = sio.load_slp(
                str(self.labels), **(self.remote_kwargs or {})
            )

        if self.only_labeled_frames and self.exclude_user_labeled:
            raise ValueError(
                "only_labeled_frames=True and exclude_user_labeled=True are "
                "mutually exclusive."
            )

        # Priority order matches legacy LabelsReader exactly (data/providers.py):
        # only_labeled_frames > only_suggested_frames > exclude_user_labeled >
        # only_predicted_frames. When more than one flag is set, whichever comes
        # first here wins -- this order is legacy-parity-load-bearing, not
        # arbitrary; a previous version of this method used a different
        # (effectively reversed) order with no test catching the drift.
        if self.only_labeled_frames:
            # Keep only frames with >=1 USER (ground-truth) instance, and drop
            # predicted-only frames. Legacy LabelsReader restricted GT to user
            # instances; using lf.instances here would feed PredictedInstances
            # into the GT-centroid / GT-peaks paths (#582).
            self._labeled_frames = [
                lf for lf in self._sio_labels.labeled_frames if lf.has_user_instances
            ]
        elif self.only_suggested_frames:
            self._labeled_frames = self._collect_suggested_frames(sio)
        elif self.exclude_user_labeled:
            self._labeled_frames = [
                lf
                for lf in self._sio_labels.labeled_frames
                if not lf.has_user_instances
            ]
        elif self.only_predicted_frames:
            self._labeled_frames = [
                lf
                for lf in self._sio_labels.labeled_frames
                if lf.has_predicted_instances
            ]
        else:
            self._labeled_frames = list(self._sio_labels.labeled_frames)

        if self.frames is not None:
            n = len(self._labeled_frames)
            positions = set(self.frames)
            out_of_range = sorted(p for p in positions if p < 0 or p >= n)
            if out_of_range:
                logger.warning(
                    f"LabelsProvider: {len(out_of_range)} requested frame "
                    f"position(s) out of range for {n} labeled frame(s) "
                    f"(after any only_*/exclude_* filtering) and will be "
                    f"skipped: {out_of_range}"
                )
            self._labeled_frames = [
                lf for i, lf in enumerate(self._labeled_frames) if i in positions
            ]

    def _frame_instances(self, lf) -> list:
        """Instances to expose as GT for a frame.

        In ``only_labeled_frames`` mode (the GT-fallback paths), expose only the
        USER instances so PredictedInstances are never treated as ground truth
        (legacy parity, #582). All other modes expose every instance.
        """
        if self.only_labeled_frames:
            return list(lf.user_instances)
        return list(lf.instances)

    def _collect_suggested_frames(self, sio) -> list:
        """Return new ``LabeledFrame``s for unlabeled suggestions.

        Mirrors the legacy ``LabelsReader`` semantics: walks
        ``labels.suggestions`` and emits a fresh empty ``LabeledFrame``
        for any suggestion whose frame doesn't already have a user
        instance.
        """
        out: list = []
        for suggestion in self._sio_labels.suggestions:
            existing = self._sio_labels.find(suggestion.video, suggestion.frame_idx)
            if not existing or not existing[0].has_user_instances:
                out.append(
                    sio.LabeledFrame(
                        video=suggestion.video, frame_idx=suggestion.frame_idx
                    )
                )
        return out

    def _read_batches(self, image_fn=None) -> Iterator[Batch]:
        """Yield batches; each ``Batch.instances`` carries GT keypoints.

        For frames with no instances (e.g. ``only_suggested_frames``
        emits empty placeholders), ``Batch.instances`` is ``None`` so
        downstream layers that don't need GT (single-instance,
        top-down with centroid model, bottom-up) skip the GT-shaped
        kwargs entirely.

        Args:
            image_fn: Optional ``(LabeledFrame) -> np.ndarray`` override for
                pixel access, used by the prefetch thread to read through a
                thread-local video copy instead of ``lf.image`` (which would
                share a backend handle with whatever the main thread touches).
                Defaults to ``lambda lf: lf.image``.
        """
        if image_fn is None:
            image_fn = lambda lf: lf.image  # noqa: E731

        # Group frames into chunks bounded by batch_size that ALSO share a
        # common image shape. Frames from different videos can differ in
        # resolution, and np.stack requires uniform shape; same-video frames
        # share a shape, so this only shrinks a chunk at a resolution (video)
        # boundary instead of crashing on np.stack (#mixed-resolution .slp).
        n_frames = len(self._labeled_frames)
        start = 0
        while start < n_frames:
            chunk = []
            chunk_imgs = []
            first_shape = None
            idx = start
            while idx < n_frames and len(chunk) < self.batch_size:
                img = image_fn(self._labeled_frames[idx])
                if first_shape is None:
                    first_shape = img.shape
                elif img.shape != first_shape:
                    break
                chunk.append(self._labeled_frames[idx])
                chunk_imgs.append(img)
                idx += 1
            start = idx
            frames = np.stack(chunk_imgs, axis=0)

            inst_lists = [self._frame_instances(lf) for lf in chunk]
            max_inst = max(len(insts) for insts in inst_lists)
            if max_inst == 0:
                instances = None
            else:
                # Pad GT instances to a uniform max_instances per batch so
                # downstream layer code can work with fixed shapes.
                n_nodes = next(
                    (len(insts[0].skeleton.nodes) for insts in inst_lists if insts),
                    1,
                )
                instances = np.full(
                    (len(chunk), max_inst, n_nodes, 2), np.nan, dtype=np.float32
                )
                for i, insts in enumerate(inst_lists):
                    for j, inst in enumerate(insts):
                        pts = np.asarray(inst.numpy(), dtype=np.float32)
                        instances[i, j, : pts.shape[0]] = pts

            frame_idxs = np.array([lf.frame_idx for lf in chunk], dtype=np.int64)
            # Attribute each frame to the index of ITS video in the Labels'
            # video list so multi-video .slp predictions land on the correct
            # video (legacy parity; #530 audit: this was hardcoded to 0, so
            # every frame was mis-assigned to videos[0]).
            vid_index = {id(v): i for i, v in enumerate(self._sio_labels.videos)}
            video_idxs = np.array(
                [vid_index.get(id(lf.video), 0) for lf in chunk], dtype=np.int64
            )
            yield Batch(
                images=frames,
                frame_indices=frame_idxs,
                video_indices=video_idxs,
                instances=instances,
            )

    def _thread_local_video_map(self) -> dict:
        """Map ``id(original video)`` -> private deep copy for the prefetch thread.

        Mirrors :meth:`VideoProvider._thread_local_video`: closing first drops
        each video's cached backend handle so ``deepcopy`` clones cheap
        (path + config) state rather than a live, non-thread-safe handle. Each
        shared original is reopened afterward (see
        ``_reopen_after_thread_local_copy``) since it is also what ends up in
        ``self._sio_labels.videos`` / the final output ``Labels`` for saving.
        """
        mapping = {}
        for video in self._sio_labels.videos:
            video.close()
            mapping[id(video)] = deepcopy(video)
            _reopen_after_thread_local_copy(video)
        return mapping

    def _prefetch_worker(
        self, video_map: dict, q: "queue.Queue", stop_event: threading.Event
    ) -> None:
        """Decode batches via ``video_map`` onto ``q`` until exhausted or stopped."""

        def image_fn(lf):
            return video_map[id(lf.video)][lf.frame_idx]

        try:
            for batch in self._read_batches(image_fn):
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        q.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            q.put(_ProducerError(exc))
            return
        q.put(_SENTINEL)

    def __iter__(self) -> Iterator[Batch]:
        """Yield batches, prefetching frame decode on a background thread.

        See :meth:`VideoProvider.__iter__` for the rationale and shutdown
        semantics (lazy thread start, bounded queue, exception propagation,
        stop-on-early-exit).
        """
        if not self.prefetch:
            yield from self._read_batches()
            return

        video_map = self._thread_local_video_map()
        q: "queue.Queue" = queue.Queue(maxsize=self.queue_maxsize)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(video_map, q, stop_event),
            daemon=True,
        )
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    return
                if isinstance(item, _ProducerError):
                    raise item.exc
                yield item
        finally:
            stop_event.set()
            thread.join(timeout=5.0)

    def __len__(self) -> int:
        """Number of batches over the (filtered) labeled-frame list."""
        n = len(self._labeled_frames)
        return (n + self.batch_size - 1) // self.batch_size

    def num_frames(self) -> int:
        """Total number of frames over the (filtered) labeled-frame list."""
        return len(self._labeled_frames)

    @property
    def videos(self) -> "list[sio.Video]":
        """Source videos of the underlying ``Labels``, for output packaging.

        Frame filtering (``only_suggested_frames`` etc.) narrows which frames
        are yielded, not the video list — predicted frames still reference these
        real videos. Used by ``Predictor._make_provider`` so a pre-built
        provider source doesn't yield a ``None``-video ``Labels`` that crashes
        ``sio.Labels.save`` (#699).
        """
        return list(self._sio_labels.videos)


@attrs.define
class MultiVideoProvider:
    """Concatenate several providers, OFFSETTING per-source video indices.

    Wraps an ordered list of already-built providers (one per input source)
    and yields their batches in order, shifting each batch's
    ``video_indices`` by that source's starting global video index. Each
    sub-provider emits its own local video indices (``VideoProvider`` always
    0; a ``LabelsProvider`` over a multi-video ``.slp`` emits per-frame
    0..N-1), so adding the per-source offset attributes every frame to the
    correct video in the merged multi-video ``.slp`` — and supports both
    single-video and multi-video sources in the list (#582).

    Args:
        providers: Ordered list of per-source :class:`Provider` instances.
            Build these via ``Predictor._make_provider`` so source-type
            dispatch stays in one place.
        video_offsets: Parallel list giving each source's starting index
            into the merged ``videos`` list (i.e. the cumulative video count
            of the preceding sources). Defaults to ``0, 1, 2, ...`` (one
            video per source) when omitted.
    """

    providers: list
    video_offsets: Optional[list] = None

    def _offsets(self) -> list:
        if self.video_offsets is not None:
            return list(self.video_offsets)
        return list(range(len(self.providers)))

    def __iter__(self) -> Iterator[Batch]:
        """Yield each sub-provider's batches with its video offset applied."""
        offsets = self._offsets()
        for provider, offset in zip(self.providers, offsets):
            for batch in provider:
                n = int(batch.images.shape[0])
                if batch.video_indices is not None:
                    local = np.asarray(batch.video_indices, dtype=np.int64)
                    vid = local + offset
                else:
                    vid = np.full(n, offset, dtype=np.int64)
                yield attrs.evolve(batch, video_indices=vid)

    def __len__(self) -> int:
        """Total batches across all sub-providers (or ``-1`` if any unknown)."""
        total = 0
        for provider in self.providers:
            n = len(provider)
            if n < 0:
                return -1
            total += n
        return total

    def num_frames(self) -> int:
        """Total frames across all sub-providers (or ``-1`` if any unknown)."""
        total = 0
        for provider in self.providers:
            n = provider.num_frames()
            if n < 0:
                return -1
            total += n
        return total

    @property
    def videos(self) -> list:
        """Merged source videos across sub-providers, in offset order (#699)."""
        merged: list = []
        for provider in self.providers:
            merged.extend(getattr(provider, "videos", None) or [])
        return merged


@attrs.define
class NumpyProvider:
    """Emit a pre-loaded tensor as one or more batches.

    Args:
        images: ``(N, ...)`` array of frames already in memory. Sliced
            into batches of ``batch_size`` along the leading dim.
        batch_size: Number of frames per yielded ``Batch``. The last
            batch may be smaller if ``N % batch_size != 0``.
        frame_indices: Optional explicit frame indices; defaults to
            ``arange(N)``.
        video_indices: Optional explicit video indices; defaults to all
            zeros (single-video assumption).

    Notes:
        Right for: real-time loops, notebook calls where frames are
        already loaded, integration tests. For video files use
        :class:`VideoProvider` and for ``.slp`` use :class:`LabelsProvider`.
    """

    images: np.ndarray | torch.Tensor
    batch_size: int = 4
    frame_indices: Optional[np.ndarray] = None
    video_indices: Optional[np.ndarray] = None

    def __attrs_post_init__(self) -> None:
        """Default per-frame metadata if the caller didn't provide any."""
        n = int(self.images.shape[0])
        if self.frame_indices is None:
            self.frame_indices = np.arange(n, dtype=np.int64)
        if self.video_indices is None:
            self.video_indices = np.zeros(n, dtype=np.int64)

    def __iter__(self) -> Iterator[Batch]:
        """Yield ``Batch``es of ``batch_size`` frames at a time."""
        n = int(self.images.shape[0])
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            yield Batch(
                images=self.images[start:stop],
                frame_indices=self.frame_indices[start:stop],
                video_indices=self.video_indices[start:stop],
            )

    def __len__(self) -> int:
        """Number of batches; ``ceil(N / batch_size)``."""
        n = int(self.images.shape[0])
        return (n + self.batch_size - 1) // self.batch_size

    def num_frames(self) -> int:
        """Total number of frames in the pre-loaded tensor."""
        return int(self.images.shape[0])
