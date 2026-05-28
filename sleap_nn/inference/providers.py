"""``Provider`` protocol + concrete data sources for the new ``Predictor``.

A ``Provider`` is the new equivalent of the legacy ``LabelsReader`` /
``VideoReader`` from ``sleap_nn.data.providers``. Two differences:

* It's a protocol, not a base class — no inheritance required, anything
  with the right shape works (existing legacy readers can be wrapped).
* It returns batches of ``np.ndarray`` (raw images) plus per-batch
  metadata (frame indices, video indices, optionally GT instances) —
  not pre-formatted dicts. The new ``Predictor`` does the rest.

Three concrete implementations:

* :class:`NumpyProvider` — emits an in-memory tensor batch as a single
  iterate. Right for testing, real-time loops, or when the caller has
  already loaded frames.
* :class:`VideoProvider` — wraps a video path; yields frames in batches.
  Built on ``sleap_io.Video`` via the existing data-layer reader so we
  don't duplicate decoding.
* :class:`LabelsProvider` — wraps a ``.slp`` file; yields the labeled
  frames + their GT instances (needed for the ``use_gt_centroids`` /
  ``use_gt_peaks`` paths in the new layers).

The latter two land as follow-up commits on this branch — this commit
ships only the protocol + ``NumpyProvider``.
"""

from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable

import attrs
import numpy as np
import torch


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


@runtime_checkable
class Provider(Protocol):
    """Iterator-of-batches contract that the new ``Predictor`` consumes."""

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

    Notes:
        Yields raw ``(B, H, W, C)`` frames as ``np.uint8``. The
        layer's ``preprocess`` does the canonicalization
        (e.g., ``ensure_grayscale``).
    """

    video: object  # str | sio.Video
    batch_size: int = 4
    frames: Optional[list[int]] = None
    dataset: Optional[str] = None
    input_format: Optional[str] = None

    _sio_video: object = attrs.field(default=None, init=False, repr=False)
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
    """

    labels: object  # str | sio.Labels
    batch_size: int = 4
    only_labeled_frames: bool = True
    only_suggested_frames: bool = False
    exclude_user_labeled: bool = False
    only_predicted_frames: bool = False

    _sio_labels: object = attrs.field(default=None, init=False, repr=False)
    _labeled_frames: list = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Resolve the labels source and pre-filter the labeled frames."""
        import sleap_io as sio

        if isinstance(self.labels, sio.Labels):
            self._sio_labels = self.labels
        else:
            self._sio_labels = sio.load_slp(str(self.labels))

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
            self._labeled_frames = [
                lf for lf in self._sio_labels.labeled_frames if len(lf.instances) > 0
            ]
        else:
            self._labeled_frames = list(self._sio_labels.labeled_frames)

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
        for start in range(0, len(self._labeled_frames), self.batch_size):
            stop = min(start + self.batch_size, len(self._labeled_frames))
            chunk = self._labeled_frames[start:stop]
            frames = np.stack([lf.image for lf in chunk], axis=0)

            max_inst = max(len(lf.instances) for lf in chunk)
            if max_inst == 0:
                instances = None
            else:
                # Pad GT instances to a uniform max_instances per batch so
                # downstream layer code can work with fixed shapes.
                n_nodes = next(
                    (
                        len(lf.instances[0].skeleton.nodes)
                        for lf in chunk
                        if lf.instances
                    ),
                    1,
                )
                instances = np.full(
                    (len(chunk), max_inst, n_nodes, 2), np.nan, dtype=np.float32
                )
                for i, lf in enumerate(chunk):
                    for j, inst in enumerate(lf.instances):
                        pts = np.asarray(inst.numpy(), dtype=np.float32)
                        instances[i, j, : pts.shape[0]] = pts

            frame_idxs = np.array([lf.frame_idx for lf in chunk], dtype=np.int64)
            yield Batch(
                images=frames,
                frame_indices=frame_idxs,
                video_indices=np.zeros(len(chunk), dtype=np.int64),
                instances=instances,
            )

    def __len__(self) -> int:
        """Number of batches over the (filtered) labeled-frame list."""
        n = len(self._labeled_frames)
        return (n + self.batch_size - 1) // self.batch_size


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
