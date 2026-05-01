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
