"""``Predictor`` — high-level orchestrator for the inference stack.

Composes an :class:`InferenceLayer` (or composed layer like
:class:`TopDownLayer`) with a :class:`Provider` source and a
:class:`FilterPipeline` post-processor.

Three usage tiers:

* :meth:`predict` — synchronous, returns ``sio.Labels`` (or a list of
  ``Outputs`` if ``make_labels=False``). Loads everything into memory;
  use for short videos / interactive sessions.
* :meth:`predict_streaming` — yields one ``Outputs`` per batch as a
  generator. Memory stays O(tracker_window).
* :meth:`predict_to_file` — disk-streaming write of a ``.slp`` via
  :class:`IncrementalLabelsWriter`. Memory stays O(write_interval).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.providers import Provider
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

if TYPE_CHECKING:
    import sleap_io as sio


def _safe_len(provider: Any) -> int:
    """Return ``len(provider)`` or ``-1`` if the provider doesn't expose ``__len__``."""
    try:
        return len(provider)
    except TypeError:
        return -1


@attrs.define
class Predictor:
    """High-level orchestrator: layer + source dispatch + filter pipeline.

    Args:
        layer: Any object exposing ``predict(image) -> Outputs``. Includes
            every :class:`InferenceLayer` subclass plus composed layers
            like :class:`TopDownLayer`.
        skeleton: Optional ``sio.Skeleton`` resolved from the training
            config. Populated automatically by
            :func:`~sleap_nn.inference.factory.get_predictor_from_model_paths`.
            Used as the default for ``predict(make_labels=True)`` and
            ``predict_to_file()`` when no explicit ``skeleton`` kwarg is
            passed.
        batch_size: Default batch size for auto-constructed providers when
            ``predict`` / ``predict_streaming`` receive an ``sio.Video``
            or ``sio.Labels`` instead of a pre-built ``Provider``.
        filter_config: Optional post-inference filter config. Default is
            the no-op identity.
        paf_workers: Number of CPU worker processes for the bottom-up
            PAF grouping stage. ``0`` (default) runs grouping inline in
            the main process — the parity path. ``>0`` is only honored
            when ``layer`` is a :class:`BottomUpLayer`; for any other
            layer type the value is ignored.
        tracker_config: Optional :class:`TrackerConfig`. When set,
            :meth:`predict` runs the tracker on the resulting
            ``sio.Labels`` (requires ``make_labels=True``) before
            returning.

    Notes:
        Keeps no state across calls — same predictor can be reused on
        multiple sources safely.
    """

    layer: Any  # TODO: unify layer types under a common Protocol
    skeleton: Optional["sio.Skeleton"] = None
    batch_size: int = 4
    filter_config: FilterConfig = attrs.Factory(FilterConfig)
    paf_workers: int = 0
    tracker_config: Optional[TrackerConfig] = None

    @property
    def filter_pipeline(self) -> FilterPipeline:
        """Build a fresh ``FilterPipeline`` from the config (cheap)."""
        return FilterPipeline(self.filter_config)

    # ──────────────────────────────────────────────────────────────────
    # Source dispatch: sio.Video / sio.Labels / str / Provider
    # ──────────────────────────────────────────────────────────────────

    def _make_provider(
        self,
        source: Any,
        frames: Optional[List[int]] = None,
        **provider_kwargs: Any,
    ) -> tuple["Provider", Optional[List["sio.Video"]]]:
        """Wrap a source into a ``Provider`` + extract videos for label packaging.

        Returns ``(provider, videos)`` where ``videos`` is a list of
        ``sio.Video`` when derivable from the source, else ``None``.
        """
        import sleap_io as sio

        from sleap_nn.inference.providers import (
            LabelsProvider,
            VideoProvider,
        )

        if isinstance(source, (str, np.ndarray)):
            if isinstance(source, str) and source.endswith(".slp"):
                provider = LabelsProvider(
                    labels=source,
                    batch_size=self.batch_size,
                    **provider_kwargs,
                )
                return provider, None
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            return provider, None

        if isinstance(source, sio.Video):
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            return provider, [source]

        if isinstance(source, sio.Labels):
            provider = LabelsProvider(
                labels=source,
                batch_size=self.batch_size,
                **provider_kwargs,
            )
            videos = list(source.videos) if source.videos else None
            return provider, videos

        if hasattr(source, "__iter__"):
            return source, None

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            f"Pass an sio.Video, sio.Labels, file path string, or a Provider."
        )

    @staticmethod
    def retrack(
        labels: "sio.Labels",
        tracker_config: TrackerConfig,
        clean_empty_frames: bool = False,
    ) -> "sio.Labels":
        """Retrack an existing ``sio.Labels`` without running inference.

        Pure tracking -- useful when you already have predicted instances
        in a ``.slp`` and just want to (re)apply a tracker. The tracker
        runs once over the full LabeledFrame list; post-tracking cleanup
        (cull / connect-single-breaks) is applied per ``tracker_config``.

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
        source: Any,
        *,
        make_labels: bool = True,
        frames: Optional[List[int]] = None,
        skeleton: Optional["sio.Skeleton"] = None,
        videos: Optional[List["sio.Video"]] = None,
        clean_empty_frames: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
    ) -> Union[List[Outputs], "sio.Labels"]:
        """Run inference on a source.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`. When a non-Provider source
                is given, a provider is auto-constructed using
                ``self.batch_size``.
            make_labels: When ``True`` (the default), return a
                ``sio.Labels``. Set to ``False`` for a raw
                ``List[Outputs]``.
            frames: Frame indices to predict on. Only used when ``source``
                is an ``sio.Video`` or video path.
            skeleton: ``sio.Skeleton`` for label conversion. Falls back to
                ``self.skeleton`` when ``None``.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for label conversion. Auto-derived from
                the source when possible.
            clean_empty_frames: When ``True`` and ``make_labels=True``,
                drop ``LabeledFrame``s with no instances from the
                returned ``sio.Labels``.
            progress_callback: Optional ``(processed_batches, total_batches)``
                callback invoked after each batch.
            peak_threshold: Override peak threshold for all stages. For
                per-stage control on top-down models, use
                ``centroid_threshold`` / ``keypoint_threshold`` instead.
            centroid_threshold: Override peak threshold for the centroid
                stage only (top-down models).
            keypoint_threshold: Override peak threshold for the centered-
                instance stage only (top-down models).
            max_instances: Override max instances per frame.
            integral_refinement: ``"integral"`` or ``"none"``.
            integral_patch_size: Override integral refinement patch size.
            return_confmaps: Override whether to return confidence maps.
            return_crops: Override whether to return per-instance crops
                (top-down only).

        Returns:
            ``sio.Labels`` (default) or ``List[Outputs]`` (when
            ``make_labels=False``).
        """
        provider, auto_videos = self._make_provider(source, frames=frames)
        if videos is None:
            videos = auto_videos

        with self._postprocess_overrides(
            peak_threshold=peak_threshold,
            centroid_threshold=centroid_threshold,
            keypoint_threshold=keypoint_threshold,
            max_instances=max_instances,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_crops=return_crops,
        ):
            outputs_list = list(self._batch_iter(provider, progress_callback))

        if not make_labels:
            if self.tracker_config is not None:
                raise ValueError(
                    "tracker_config requires make_labels=True; the tracker "
                    "operates on sio.PredictedInstance objects."
                )
            return outputs_list
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None:
            raise ValueError(
                "make_labels=True requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via get_predictor_from_model_paths() "
                "which sets it automatically from the training config."
            )
        labels = self.to_labels(outputs_list, videos=videos)
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
        source: Any,
        *,
        frames: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
    ) -> Iterator[Outputs]:
        """Yield one ``Outputs`` per batch from ``source``.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`.
            frames: Frame indices (only for video sources).
            progress_callback: Optional ``(processed_batches, total_batches)``
                callback.
            peak_threshold: Override peak threshold for all stages.
            centroid_threshold: Override centroid stage threshold (top-down).
            keypoint_threshold: Override centered-instance threshold (top-down).
            max_instances: Override max instances per frame.
            integral_refinement: ``"integral"`` or ``"none"``.
            integral_patch_size: Override integral refinement patch size.
            return_confmaps: Override whether to return confidence maps.
            return_crops: Override whether to return per-instance crops
                (top-down only).
        """
        if self.tracker_config is not None:
            raise ValueError(
                "tracker_config is not supported on predict_streaming / "
                "predict_to_file. End-of-stream tracker cleanup needs the "
                "full LabeledFrame list; use predict() instead."
            )
        provider, _ = self._make_provider(source, frames=frames)
        with self._postprocess_overrides(
            peak_threshold=peak_threshold,
            centroid_threshold=centroid_threshold,
            keypoint_threshold=keypoint_threshold,
            max_instances=max_instances,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_crops=return_crops,
        ):
            if self.paf_workers > 0 and self._can_pipeline():
                yield from self._predict_streaming_pipelined(
                    provider, progress_callback
                )
                return
            yield from self._batch_iter(provider, progress_callback)

    # ──────────────────────────────────────────────────────────────────
    # Disk-streaming: write to a .slp incrementally
    # ──────────────────────────────────────────────────────────────────

    def predict_to_file(
        self,
        source: Any,
        path: str,
        *,
        frames: Optional[List[int]] = None,
        skeleton: Optional["sio.Skeleton"] = None,
        videos: Optional[List["sio.Video"]] = None,
        write_interval: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Run inference and stream results to a ``.slp`` file.

        Memory stays O(``write_interval``) — outputs are slimmed and
        converted to LabeledFrames per batch; heavy tensors are dropped
        immediately.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`.
            path: Destination ``.slp`` path.
            frames: Frame indices (only for video sources).
            skeleton: ``sio.Skeleton`` for instance conversion. Falls back
                to ``self.skeleton`` when ``None``.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for the saved labels.
            write_interval: Number of LabeledFrames to buffer before
                a disk flush.
            progress_callback: Optional callback per batch.

        Returns:
            The (resolved) destination path string.
        """
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None:
            raise ValueError(
                "predict_to_file requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via get_predictor_from_model_paths() "
                "which sets it automatically from the training config."
            )
        from sleap_nn.inference.writer import IncrementalLabelsWriter

        provider, _ = self._make_provider(source, frames=frames)
        with IncrementalLabelsWriter(
            path=path,
            skeleton=self.skeleton,
            videos=videos,
            write_interval=write_interval,
        ) as writer:
            for outputs in self.predict_streaming(
                provider, progress_callback=progress_callback
            ):
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
        """Run ``layer.predict`` + ``FilterPipeline`` per provider batch."""
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
        from sleap_nn.inference.layers.bottomup import BottomUpLayer

        return isinstance(self.layer, BottomUpLayer)

    def _predict_streaming_pipelined(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Stream ``Outputs`` with the CPU grouping stage in a worker pool."""
        from sleap_nn.inference.streaming import PafGroupingPool

        pipeline = self.filter_pipeline
        layer = self.layer
        params = layer.grouping_params()
        total = _safe_len(provider)

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

    def to_labels(
        self,
        outputs_list: List[Outputs],
        videos: Optional[List["sio.Video"]] = None,
    ) -> "sio.Labels":
        """Concatenate per-batch ``Outputs`` into a single ``sio.Labels``."""
        import sleap_io as sio

        skeleton = self.skeleton
        anchor_ind = self._packaging_anchor_ind()
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
        """Anchor-node slot for centroid-only output packaging."""
        from sleap_nn.inference.layers.centroid import CentroidLayer

        if isinstance(self.layer, CentroidLayer):
            return self.layer.anchor_ind
        return None

    # ──────────────────────────────────────────────────────────────────
    # Prediction-time postprocess overrides
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_postprocess_targets(layer: Any) -> list:
        """Return all sub-layers that own a ``postprocess_config``."""
        from sleap_nn.inference.layers.topdown import TopDownLayer

        if isinstance(layer, TopDownLayer):
            targets = [layer.centroid_layer, layer.centered_instance_layer]
        elif hasattr(layer, "postprocess_config"):
            targets = [layer]
        else:
            targets = []
        return targets

    @contextmanager
    def _postprocess_overrides(
        self,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
    ):
        """Context manager that temporarily overrides postprocess configs.

        For top-down layers, ``centroid_threshold`` applies to the centroid
        stage and ``keypoint_threshold`` to the centered-instance stage.
        ``peak_threshold`` sets both when the per-stage kwargs aren't given.
        """
        from sleap_nn.inference.layers.topdown import TopDownLayer

        has_any = any(
            v is not None
            for v in (
                peak_threshold,
                centroid_threshold,
                keypoint_threshold,
                max_instances,
                integral_refinement,
                integral_patch_size,
                return_confmaps,
                return_crops,
            )
        )
        if not has_any:
            yield
            return

        saved: list[tuple[Any, PostprocessConfig]] = []
        saved_return_crops: Optional[bool] = None

        try:
            targets = self._collect_postprocess_targets(self.layer)

            for target in targets:
                old_cfg = target.postprocess_config
                saved.append((target, old_cfg))

                overrides: dict = {}

                # Threshold routing for top-down
                if isinstance(self.layer, TopDownLayer):
                    is_centroid = target is self.layer.centroid_layer
                    if is_centroid:
                        t = centroid_threshold or peak_threshold
                    else:
                        t = keypoint_threshold or peak_threshold
                else:
                    t = peak_threshold

                if t is not None:
                    overrides["peak_threshold"] = t
                if max_instances is not None and hasattr(old_cfg, "max_instances"):
                    overrides["max_instances"] = max_instances
                if integral_refinement is not None:
                    overrides["refinement"] = integral_refinement
                if integral_patch_size is not None:
                    overrides["integral_patch_size"] = integral_patch_size
                if return_confmaps is not None:
                    overrides["return_confmaps"] = return_confmaps

                if overrides:
                    target.postprocess_config = attrs.evolve(old_cfg, **overrides)

            # return_crops lives on TopDownLayer, not on postprocess_config
            if return_crops is not None and isinstance(self.layer, TopDownLayer):
                saved_return_crops = self.layer.return_crops
                self.layer.return_crops = return_crops

            yield
        finally:
            for target, old_cfg in saved:
                target.postprocess_config = old_cfg
            if saved_return_crops is not None:
                self.layer.return_crops = saved_return_crops
