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

from typing import Any, Iterator, List, Optional, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.providers import Provider


@attrs.define
class Predictor:
    """High-level orchestrator: layer + provider + filter pipeline.

    Args:
        layer: Any object exposing ``predict(image) -> Outputs``. Includes
            every :class:`InferenceLayer` subclass plus composed layers
            like :class:`TopDownLayer`.
        filter_config: Optional post-inference filter config. Default is
            the no-op identity.

    Notes:
        Keeps no state across calls — same predictor can be reused on
        multiple providers safely.
    """

    layer: Any
    filter_config: FilterConfig = attrs.Factory(FilterConfig)

    @property
    def filter_pipeline(self) -> FilterPipeline:
        """Build a fresh ``FilterPipeline`` from the config (cheap)."""
        return FilterPipeline(self.filter_config)

    # ──────────────────────────────────────────────────────────────────
    # Synchronous: returns Outputs list or sio.Labels
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        provider: Provider,
        make_labels: bool = False,
        skeleton: Optional[Any] = None,
        videos: Optional[List[Any]] = None,
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

        Returns:
            ``List[Outputs]`` (raw mode) or ``sio.Labels`` (with-labels
            mode).
        """
        outputs_list = list(self._batch_iter(provider))
        if not make_labels:
            return outputs_list
        if skeleton is None:
            raise ValueError("make_labels=True requires `skeleton` to be passed.")
        return self._to_labels(outputs_list, skeleton=skeleton, videos=videos)

    # ──────────────────────────────────────────────────────────────────
    # Streaming: yields one Outputs at a time
    # ──────────────────────────────────────────────────────────────────

    def predict_streaming(self, provider: Provider) -> Iterator[Outputs]:
        """Yield one ``Outputs`` per provider batch.

        Caller-controlled memory: the predictor never materializes the
        full list. Useful for long videos and live cameras.
        """
        yield from self._batch_iter(provider)

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
            for outputs in self.predict_streaming(provider):
                writer.write(outputs)
        return path

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _batch_iter(self, provider: Provider) -> Iterator[Outputs]:
        """Run ``layer.predict`` + ``FilterPipeline`` per provider batch."""
        pipeline = self.filter_pipeline
        for batch in provider:
            kwargs: dict = {}
            if batch.instances is not None:
                kwargs["instances"] = (
                    batch.instances
                    if isinstance(batch.instances, torch.Tensor)
                    else torch.from_numpy(batch.instances)
                )
            outputs = self.layer.predict(batch.images, **kwargs)
            outputs = pipeline(outputs)
            outputs = self._stamp_metadata(outputs, batch)
            yield outputs

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
    ) -> Any:
        """Concatenate per-batch ``Outputs`` into a single ``sio.Labels``.

        Uses each ``Outputs.to_labels`` (PR 2's minimal implementation) per
        batch and merges the resulting labeled-frame lists.
        """
        import sleap_io as sio

        videos = list(videos) if videos else [None]
        all_lf: list = []
        for outputs in outputs_list:
            sub = outputs.to_labels(skeleton=skeleton, videos=videos)
            all_lf.extend(sub.labeled_frames)
        valid_videos = [v for v in videos if v is not None]
        return sio.Labels(
            labeled_frames=all_lf,
            videos=valid_videos,
            skeletons=[skeleton],
        )
