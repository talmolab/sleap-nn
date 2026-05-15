"""``Predictor`` — high-level orchestrator for the new inference stack.

Composes an :class:`InferenceLayer` (or composed layer like
:class:`TopDownLayer`) with a :class:`Provider` source and a
:class:`FilterPipeline` post-processor. Replaces the legacy
``sleap_nn.inference.predictors.Predictor`` (which is 3964 lines, model-
type-specific, and tightly couples I/O / batching / filtering).

Three usage tiers:

* :meth:`predict` — synchronous, returns ``sio.Labels`` (or a list of
  ``Outputs`` if ``make_labels=False``). Loads everything into memory;
  use for short videos / interactive sessions.
* :meth:`predict_streaming` — yields one ``Outputs`` per batch as a
  generator. Memory stays O(tracker_window).
* :meth:`predict_to_file` — disk-streaming write of a ``.slp`` via the
  forthcoming ``IncrementalLabelsWriter``. Memory stays O(write_interval).

This commit ships the synchronous :meth:`predict`. Streaming +
``predict_to_file`` land as follow-ups on the same branch.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.providers import Provider
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking


def _safe_len(provider: Any) -> int:
    """Return ``len(provider)`` or ``-1`` if the provider doesn't expose ``__len__``."""
    try:
        return len(provider)
    except TypeError:
        return -1


@attrs.define
class Predictor:
    """High-level orchestrator: layer + provider + filter pipeline.

    Args:
        layer: Any object exposing ``predict(image) -> Outputs``. Includes
            every :class:`InferenceLayer` subclass plus composed layers
            like :class:`TopDownLayer`.
        filter_config: Optional post-inference filter config. Default is
            the no-op identity.
        paf_workers: Number of CPU worker processes for the bottom-up
            PAF grouping stage. ``0`` (default) runs grouping inline in
            the main process — the parity path. ``>0`` is only honored
            when ``layer`` is a :class:`BottomUpLayer`; for any other
            layer type the value is ignored. Each worker starts a fresh
            Python interpreter on macOS / Windows (~1s startup cost), so
            keep this off for short videos on those platforms.
        tracker_config: Optional :class:`TrackerConfig`. When set,
            :meth:`predict` runs the tracker on the resulting
            ``sio.Labels`` (requires ``make_labels=True``) before
            returning. A fresh ``Tracker`` is built per call, so no
            state leaks across invocations. ``predict_streaming`` /
            ``predict_to_file`` raise on tracker_config — end-of-stream
            cleanup (``cull_instances`` / ``connect_single_breaks``)
            needs the full LabeledFrame list, which defeats streaming.

    Notes:
        Keeps no state across calls — same predictor can be reused on
        multiple providers safely.
    """

    layer: Any
    filter_config: FilterConfig = attrs.Factory(FilterConfig)
    paf_workers: int = 0
    tracker_config: Optional[TrackerConfig] = None

    @property
    def filter_pipeline(self) -> FilterPipeline:
        """Build a fresh ``FilterPipeline`` from the config (cheap)."""
        return FilterPipeline(self.filter_config)

    # ──────────────────────────────────────────────────────────────────
    # Factory: build a Predictor from one or more checkpoint paths
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_model_paths(cls, model_paths: List[str], **kwargs) -> "Predictor":
        """Build a :class:`Predictor` from one or more model checkpoint paths.

        See :func:`sleap_nn.inference.factory.from_model_paths` for the
        full kwarg surface. This classmethod is a thin alias so existing
        callers can do ``Predictor.from_model_paths(...)`` without
        knowing about the factory module.
        """
        from sleap_nn.inference.factory import from_model_paths

        return from_model_paths(model_paths, **kwargs)

    @classmethod
    def from_export_dir(cls, export_dir: str, **kwargs) -> "Predictor":
        """Build a :class:`Predictor` from an exported ``.onnx`` / ``.trt`` directory.

        See :func:`sleap_nn.inference.factory.from_export_dir` for the
        full kwarg surface.
        """
        from sleap_nn.inference.factory import from_export_dir

        return from_export_dir(export_dir, **kwargs)

    @staticmethod
    def retrack(
        labels: Any,
        tracker_config: TrackerConfig,
        clean_empty_frames: bool = False,
    ) -> Any:
        """Retrack an existing ``sio.Labels`` without running inference.

        Pure tracking — useful when you already have predicted instances
        in a ``.slp`` and just want to (re)apply a tracker. Mirrors the
        legacy ``run_inference(model_paths=None, tracking=True, ...)``
        path. The tracker runs once over the full LabeledFrame list;
        post-tracking cleanup (cull / connect-single-breaks) is applied
        per ``tracker_config``.

        Args:
            labels: A ``sio.Labels`` whose ``predicted_instances`` are
                tracked in-place semantics — this returns a new
                ``Labels`` with tracked instances.
            tracker_config: :class:`TrackerConfig` to drive the tracker.
            clean_empty_frames: When ``True``, drop empty frames from
                the result (matches ``--no_empty_frames``).

        Returns:
            New ``sio.Labels`` with tracks attached.
        """
        out = apply_tracking(labels, tracker_config)
        if clean_empty_frames:
            out.clean(frames=True, skeletons=False)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Synchronous: returns Outputs list or sio.Labels
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        provider: Provider,
        make_labels: bool = False,
        skeleton: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
        clean_empty_frames: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Union[List[Outputs], Any]:
        """Run inference on every batch from ``provider``.

        Args:
            provider: A :class:`Provider` source.
            make_labels: When ``True``, return a ``sio.Labels`` instead of
                a list of ``Outputs``. Requires ``skeleton``.
            skeleton: ``sio.Skeleton`` for label conversion. Required when
                ``make_labels=True``.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for label conversion.
            clean_empty_frames: When ``True`` and ``make_labels=True``,
                drop ``LabeledFrame``s with no instances from the
                returned ``sio.Labels``. Mirrors the legacy
                ``no_empty_frames`` flag.
            progress_callback: Optional ``(processed_batches, total_batches)``
                callback invoked after each batch. ``total_batches`` is
                ``len(provider)`` if the provider implements ``__len__``,
                else ``-1``.

        Returns:
            ``List[Outputs]`` (raw mode) or ``sio.Labels`` (with-labels
            mode).
        """
        outputs_list = list(self._batch_iter(provider, progress_callback))
        if not make_labels:
            if self.tracker_config is not None:
                raise ValueError(
                    "tracker_config requires make_labels=True; the tracker "
                    "operates on sio.PredictedInstance objects."
                )
            return outputs_list
        if skeleton is None:
            raise ValueError("make_labels=True requires `skeleton` to be passed.")
        labels = self._to_labels(
            outputs_list,
            skeleton=skeleton,
            videos=videos,
            anchor_ind=self._packaging_anchor_ind(),
        )
        if self.tracker_config is not None:
            labels = apply_tracking(labels, self.tracker_config)
        if clean_empty_frames:
            labels.clean(frames=True, skeletons=False)
        return labels

    # ──────────────────────────────────────────────────────────────────
    # Streaming: yields one Outputs at a time
    # ──────────────────────────────────────────────────────────────────

    def predict_streaming(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Yield one ``Outputs`` per provider batch.

        Caller-controlled memory: the predictor never materializes the
        full list. Useful for long videos and live cameras.

        When ``paf_workers > 0`` and ``layer`` is a :class:`BottomUpLayer`,
        routes through :meth:`_predict_streaming_pipelined` which runs
        the GPU peak / PAF-scoring stage in this process and ships the
        CPU grouping stage to a :class:`PafGroupingPool`.
        """
        if self.tracker_config is not None:
            raise ValueError(
                "tracker_config is not supported on predict_streaming / "
                "predict_to_file. End-of-stream tracker cleanup needs the "
                "full LabeledFrame list; use predict() instead."
            )
        if self.paf_workers > 0 and self._can_pipeline():
            yield from self._predict_streaming_pipelined(provider, progress_callback)
            return
        yield from self._batch_iter(provider, progress_callback)

    # ──────────────────────────────────────────────────────────────────
    # Disk-streaming: write to a .slp incrementally
    # ──────────────────────────────────────────────────────────────────

    def predict_to_file(
        self,
        provider: Provider,
        path: str,
        skeleton: Any,
        videos: Optional[List[Any]] = None,
        write_interval: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Run inference and stream results to a ``.slp`` file.

        Memory stays O(``write_interval``) — outputs are slimmed and
        converted to LabeledFrames per batch; heavy tensors are dropped
        immediately. The writer atomic-renames a ``.tmp`` to ``path`` on
        successful completion so crashes mid-stream don't corrupt the
        destination.

        Args:
            provider: Frame source.
            path: Destination ``.slp`` path.
            skeleton: ``sio.Skeleton`` for instance conversion.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for the saved labels.
            write_interval: Number of LabeledFrames to buffer before
                a disk flush.
            progress_callback: Optional ``(processed_batches, total_batches)``
                callback invoked after each batch (forwarded to
                :meth:`predict_streaming`).

        Returns:
            The (resolved) destination path string.
        """
        from sleap_nn.inference.writer import IncrementalLabelsWriter

        with IncrementalLabelsWriter(
            path=path,
            skeleton=skeleton,
            videos=videos,
            write_interval=write_interval,
        ) as writer:
            for outputs in self.predict_streaming(provider, progress_callback):
                writer.write(outputs)
        return path

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _batch_iter(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Run ``layer.predict`` + ``FilterPipeline`` per provider batch.

        ``Batch.instances`` is only forwarded to layers whose ``predict``
        signature accepts an ``instances`` keyword — without this guard a
        ``.slp`` source breaks every layer except ``CentroidLayer`` /
        ``TopDownLayer`` (``SingleInstanceLayer.predict`` rejects unknown
        kwargs, etc.). ``LabelsProvider`` always populates ``batch.instances``
        even for layers that don't use them.
        """
        import inspect

        try:
            sig = inspect.signature(self.layer.predict)
            layer_accepts_instances = "instances" in sig.parameters
        except (TypeError, ValueError):  # pragma: no cover — non-introspectable
            layer_accepts_instances = False

        pipeline = self.filter_pipeline
        total = _safe_len(provider)
        for i, batch in enumerate(provider):
            kwargs: dict = {}
            if batch.instances is not None and layer_accepts_instances:
                kwargs["instances"] = (
                    batch.instances
                    if isinstance(batch.instances, torch.Tensor)
                    else torch.from_numpy(batch.instances)
                )
            outputs = self.layer.predict(batch.images, **kwargs)
            outputs = pipeline(outputs)
            outputs = self._stamp_metadata(outputs, batch)
            yield outputs
            if progress_callback is not None:
                progress_callback(i + 1, total)

    # ──────────────────────────────────────────────────────────────────
    # Pipelined bottom-up: GPU stage in main proc, CPU grouping in pool
    # ──────────────────────────────────────────────────────────────────

    def _can_pipeline(self) -> bool:
        """``True`` iff ``layer`` is a :class:`BottomUpLayer` (not multiclass)."""
        # Local import: avoids importing the layer module at predictor load.
        from sleap_nn.inference.layers.bottomup import BottomUpLayer

        return isinstance(self.layer, BottomUpLayer)

    def _predict_streaming_pipelined(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Stream ``Outputs`` with the CPU grouping stage in a worker pool.

        The GPU stage (:meth:`BottomUpLayer._score_pafs_on_gpu`) runs
        synchronously in this process; the CPU stage
        (:func:`group_scored_batch`) is submitted to a
        :class:`PafGroupingPool` and drained in submission order so
        the caller observes the same frame ordering as the inline
        path.
        """
        from sleap_nn.inference.streaming import PafGroupingPool

        pipeline = self.filter_pipeline
        layer = self.layer
        params = layer.grouping_params()
        total = _safe_len(provider)

        # Cache per-batch metadata keyed by submission ordinal so we can
        # restamp it onto the worker-produced Outputs.
        meta: dict[int, Any] = {}
        with PafGroupingPool(
            n_workers=self.paf_workers, grouping_params=params
        ) as pool:
            for ordinal, batch in enumerate(provider):
                x, info = layer.preprocess(batch.images)
                raw = layer.backend(x)
                scored = layer._score_pafs_on_gpu(raw, info)
                pool.submit(ordinal, scored)
                meta[ordinal] = batch
            completed = 0
            for ordinal, outputs in pool.iter_completed():
                outputs = pipeline(outputs)
                outputs = self._stamp_metadata(outputs, meta.pop(ordinal))
                yield outputs
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

    @staticmethod
    def _stamp_metadata(outputs: Outputs, batch: Any) -> Outputs:
        """Attach ``frame_indices`` / ``video_indices`` from the batch."""
        kwargs: dict = {}
        if batch.frame_indices is not None and outputs.frame_indices is None:
            kwargs["frame_indices"] = (
                batch.frame_indices
                if isinstance(batch.frame_indices, torch.Tensor)
                else torch.from_numpy(np.asarray(batch.frame_indices))
            )
        if batch.video_indices is not None and outputs.video_indices is None:
            kwargs["video_indices"] = (
                batch.video_indices
                if isinstance(batch.video_indices, torch.Tensor)
                else torch.from_numpy(np.asarray(batch.video_indices))
            )
        if not kwargs:
            return outputs
        return attrs.evolve(outputs, **kwargs)

    @staticmethod
    def _to_labels(
        outputs_list: List[Outputs],
        skeleton: Any,
        videos: Optional[List[Any]] = None,
        anchor_ind: Optional[int] = None,
    ) -> Any:
        """Concatenate per-batch ``Outputs`` into a single ``sio.Labels``.

        Uses each ``Outputs.to_labels`` (PR 2's minimal implementation) per
        batch and merges the resulting labeled-frame lists. ``anchor_ind``
        is forwarded for centroid-only packaging (no-op otherwise).
        """
        import sleap_io as sio

        videos = list(videos) if videos else [None]
        all_lf: list = []
        for outputs in outputs_list:
            sub = outputs.to_labels(
                skeleton=skeleton, videos=videos, anchor_ind=anchor_ind
            )
            all_lf.extend(sub.labeled_frames)
        valid_videos = [v for v in videos if v is not None]
        return sio.Labels(
            labeled_frames=all_lf,
            videos=valid_videos,
            skeletons=[skeleton],
        )

    def _packaging_anchor_ind(self) -> Optional[int]:
        """Anchor-node slot for centroid-only output packaging.

        Returns ``self.layer.anchor_ind`` when the predictor's layer is a
        ``CentroidLayer`` (centroid-only inference); ``None`` otherwise.
        Forwarded to ``Outputs.to_labels`` so the centroid coordinate is
        placed at the configured skeleton node (or node 0 if unset).
        """
        from sleap_nn.inference.layers.centroid import CentroidLayer

        if isinstance(self.layer, CentroidLayer):
            return self.layer.anchor_ind
        return None
