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

from typing import TYPE_CHECKING, Iterator, Optional, Protocol, Union

import attrs
import numpy as np
import torch

if TYPE_CHECKING:
    import sleap_io as sio


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

    def __iter__(self) -> Iterator[Batch]:
        """Read frames in batches of ``batch_size`` and yield them."""
        for start in range(0, len(self._frame_indices), self.batch_size):
            stop = min(start + self.batch_size, len(self._frame_indices))
            chunk_inds = self._frame_indices[start:stop]
            frames = np.stack([self._sio_video[i] for i in chunk_inds], axis=0)
            yield Batch(
                images=frames,
                frame_indices=np.asarray(chunk_inds, dtype=np.int64),
                video_indices=np.zeros(len(chunk_inds), dtype=np.int64),
            )

    def __len__(self) -> int:
        """Number of batches; ``ceil(len(frames) / batch_size)``."""
        n = len(self._frame_indices)
        return (n + self.batch_size - 1) // self.batch_size

    def num_frames(self) -> int:
        """Total number of frames this provider will yield."""
        return len(self._frame_indices)


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
            user-supplied instance (default ``True``).
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
    """

    labels: "Union[str, sio.Labels]"
    batch_size: int = 4
    only_labeled_frames: bool = True
    only_suggested_frames: bool = False
    exclude_user_labeled: bool = False
    only_predicted_frames: bool = False
    remote_kwargs: Optional[dict] = None

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

        if self.only_suggested_frames:
            self._labeled_frames = self._collect_suggested_frames(sio)
        elif self.only_predicted_frames:
            self._labeled_frames = [
                lf
                for lf in self._sio_labels.labeled_frames
                if lf.has_predicted_instances
            ]
        elif self.exclude_user_labeled:
            self._labeled_frames = [
                lf
                for lf in self._sio_labels.labeled_frames
                if not lf.has_user_instances
            ]
        elif self.only_labeled_frames:
            # Keep only frames with >=1 USER (ground-truth) instance, and drop
            # predicted-only frames. Legacy LabelsReader restricted GT to user
            # instances; using lf.instances here would feed PredictedInstances
            # into the GT-centroid / GT-peaks paths (#582).
            self._labeled_frames = [
                lf for lf in self._sio_labels.labeled_frames if lf.has_user_instances
            ]
        else:
            self._labeled_frames = list(self._sio_labels.labeled_frames)

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

    def __iter__(self) -> Iterator[Batch]:
        """Yield batches; each ``Batch.instances`` carries GT keypoints.

        For frames with no instances (e.g. ``only_suggested_frames``
        emits empty placeholders), ``Batch.instances`` is ``None`` so
        downstream layers that don't need GT (single-instance,
        top-down with centroid model, bottom-up) skip the GT-shaped
        kwargs entirely.
        """
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
                img = self._labeled_frames[idx].image
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

    def __len__(self) -> int:
        """Number of batches over the (filtered) labeled-frame list."""
        n = len(self._labeled_frames)
        return (n + self.batch_size - 1) // self.batch_size

    def num_frames(self) -> int:
        """Total number of frames over the (filtered) labeled-frame list."""
        return len(self._labeled_frames)


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
