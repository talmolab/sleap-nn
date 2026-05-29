"""``IncrementalLabelsWriter`` — buffered write of a ``.slp``.

The legacy predictor accumulates every frame's predictions in memory
before writing to disk. For long videos (100k frames × hundreds of MB
of confmaps) that's an OOM risk. This writer accepts ``Outputs`` one
batch at a time and ``slim()``s each before converting to
``LabeledFrame``s, so the heavy intermediate tensors (confmaps, PAFs)
are dropped immediately — the dominant memory sink. It finalizes with
an atomic rename so a crash mid-write doesn't corrupt the output.

Two design constraints from the design doc:

* **Atomic rename** — write to ``<path>.tmp``, rename on ``close()``.
  Crash → no corrupted output (just an orphan ``.tmp`` the user can rm).
* **Resume-friendly** — closed writers can be re-opened to append. Not
  implemented in this commit (deferred to a follow-up).

Memory note: ``Outputs`` are slim()-ed per write so heavy tensors are
dropped right away. NOTE: until sleap-io exposes an incremental ``.slp``
append surface, the (lightweight) ``LabeledFrame`` objects accumulate
in memory and the ``.slp`` is written once at :meth:`close`/finalize —
peak memory is therefore O(n_frames) of *slimmed* frames, not
O(``write_interval``). ``write_interval`` currently controls only the
slim/convert cadence, not a per-flush disk write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import attrs


@attrs.define
class IncrementalLabelsWriter:
    """Stream-write ``Outputs`` → ``sio.LabeledFrame`` → ``.slp``.

    Args:
        path: Destination ``.slp`` path. Writes go to ``<path>.tmp`` and
            rename on ``close()``.
        skeleton: ``sio.Skeleton`` for instance conversion.
        videos: Optional list of ``sio.Video`` indexed by ``video_indices``.
        write_interval: Convert + slim buffered ``Outputs`` to
            ``LabeledFrame``s every ``write_interval`` frames; the
            consolidated ``.slp`` is written once at :meth:`close` (no
            per-flush disk write until sleap-io supports incremental append).

    Usage::

        writer = IncrementalLabelsWriter(path="out.slp", skeleton=skel)
        with writer:  # context-manager closes + atomic-renames
            for outputs in predictor.predict_streaming(provider):
                writer.write(outputs)
    """

    path: str
    skeleton: Any
    videos: Optional[List[Any]] = None
    write_interval: int = 500
    # Provenance dict attached to the finalized Labels. May be set after
    # construction (e.g. once end-of-run timestamps are known). #583.
    provenance: Optional[dict] = None

    _buffer: List[Any] = attrs.field(factory=list, init=False, repr=False)
    _all_frames: List[Any] = attrs.field(factory=list, init=False, repr=False)
    _closed: bool = attrs.field(default=False, init=False, repr=False)
    _resolved_videos: Optional[List[Any]] = attrs.field(
        default=None, init=False, repr=False
    )

    @property
    def tmp_path(self) -> Path:
        """The intermediate ``.tmp`` path written to before atomic rename."""
        return Path(self.path).with_suffix(Path(self.path).suffix + ".tmp")

    @property
    def frame_count(self) -> int:
        """Number of ``LabeledFrame``s buffered + accumulated so far."""
        return len(self._all_frames) + len(self._buffer)

    def __enter__(self) -> "IncrementalLabelsWriter":
        """Context manager entry — return self."""
        return self

    def __exit__(self, *exc) -> None:
        """Context manager exit — call :meth:`close` (atomic-renames)."""
        self.close()

    def write(self, outputs: Any) -> None:
        """Buffer an ``Outputs`` for eventual disk flush.

        Args:
            outputs: An :class:`Outputs` from one inference batch. The
                writer slims it (drops heavy intermediates, moves to CPU,
                detaches autograd) before converting to LabeledFrame.
        """
        if self._closed:
            raise RuntimeError("write() called on a closed writer")

        slim = outputs.slim()
        videos = self._resolve_videos()
        sub = slim.to_labels(skeleton=self.skeleton, videos=videos)
        self._buffer.extend(sub.labeled_frames)

        if len(self._buffer) >= self.write_interval:
            self._flush()

    def _resolve_videos(self) -> List[Any]:
        """Return a list of Videos for label conversion (memoized).

        ``sio.Labels.save`` cannot serialize ``LabeledFrame`` objects
        whose ``video`` is ``None``, so when the caller didn't supply a
        list (or supplied ``[None]``) we substitute a single backend-less
        ``Video`` placeholder. The placeholder serializes cleanly and
        stays loadable via ``sio.load_slp``; users typically rebind a
        real ``Video`` after load.

        The result is computed once and cached so ``write()`` (per batch)
        and ``_finalize()`` share the **same** ``Video`` object(s). Minting
        a fresh placeholder per call produced N+1 distinct videos, so every
        frame referenced a ``Video`` absent from the saved ``Labels.videos``
        list — a corrupt multi-video ``.slp`` (#582).
        """
        import sleap_io as sio

        if self._resolved_videos is not None:
            return self._resolved_videos

        if self.videos:
            real = [v for v in self.videos if v is not None]
            if real:
                self._resolved_videos = real
                return self._resolved_videos
        self._resolved_videos = [sio.Video(filename="unknown", backend=None)]
        return self._resolved_videos

    def close(self) -> None:
        """Flush the remaining buffer + atomic-rename ``.tmp`` → final path.

        Idempotent: subsequent calls are no-ops.
        """
        if self._closed:
            return
        if self._buffer:
            self._flush()
        self._finalize()
        self._closed = True

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Move the in-memory buffer onto the on-disk accumulator.

        We don't actually write per-flush to disk in this implementation
        (sleap-io's ``.slp`` format is HDF5; appending one frame at a
        time is high-overhead). Instead we accumulate in
        ``self._all_frames`` and write once at finalization, but the
        caller-visible memory contract (drop ``Outputs`` after
        ``write_interval``) is preserved because we slim() per write
        and only keep ``LabeledFrame`` references.

        A future refactor could switch to true streaming HDF5 writes
        once sleap-io exposes that surface; the public API here doesn't
        change.
        """
        self._all_frames.extend(self._buffer)
        self._buffer.clear()

    def _finalize(self) -> None:
        """Write the ``.tmp`` file and atomically rename to the final path."""
        import sleap_io as sio

        videos = self._resolve_videos()
        labels = sio.Labels(
            labeled_frames=self._all_frames,
            videos=videos,
            skeletons=[self.skeleton],
            provenance=self.provenance or {},
        )
        tmp = self.tmp_path
        tmp.parent.mkdir(parents=True, exist_ok=True)
        # Pass the format explicitly: ``sio.Labels.save`` infers from the
        # filename suffix, but our ``.tmp`` suffix isn't recognized.
        labels.save(str(tmp), format="slp")
        # Atomic rename — no half-written file on the destination path.
        tmp.replace(self.path)
