"""Export-adapter layers — thin translators around exported ONNX/TRT models.

The exported wrappers in :mod:`sleap_nn.export.wrappers` bake every
preprocessing + postprocessing step into the ONNX graph, including
``input_scale`` resize, ``output_stride`` rescaling, peak finding,
and (top-down) crop extraction. By the time output keys arrive, peaks
are already in **original-image pixel space** with the right shapes.

The standard :class:`InferenceLayer` subclasses in
``sleap_nn/inference/layers/`` were designed for ``.ckpt`` checkpoints,
where the layer's own ``preprocess`` does the input-scale resize and
``postprocess`` runs the coord ladder (``undo_stride`` /
``undo_input_scale``). Reusing them for export-loaded backends would
double-apply both transforms.

This module ships dedicated adapter layers for the export path, one
per exported model type. Each one:

* skips ``input_scale`` resize (the wrapper did it)
* feeds raw uint8 ``(B, C, H, W)`` tensors to the backend
* skips peak finding (already baked)
* skips the coord ladder (already in original-image space)
* translates the wrapper's output dict into a structured :class:`Outputs`

The classes intentionally don't inherit from :class:`InferenceLayer`
— they're shaped like layers (``predict(image, **kwargs) -> Outputs``)
but the layered preprocess/forward/postprocess pipeline doesn't apply.
"""

from __future__ import annotations

from typing import Any, Optional

import attrs
import numpy as np
import torch

from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.outputs import Outputs


def _to_4d_uint8_tensor(image: Any) -> torch.Tensor:
    """Coerce an image input to ``(B, C, H, W)`` uint8 (channel-first).

    Mirrors :meth:`InferenceLayer._to_4d_float_tensor` shape-wise, but
    keeps the dtype as the runtime expects (``ONNXBackend`` casts to
    its declared input dtype, typically ``uint8``, before invoking the
    session).
    """
    return InferenceLayer._to_4d_float_tensor(image)


@attrs.define
class ExportedSingleInstanceLayer:
    """Adapter for an ONNX/TRT-exported single-instance model.

    The wrapper output schema is:

    * ``peaks``: ``(B, N, 2)`` in original-image (x, y) space
    * ``peak_vals``: ``(B, N)``
    * ``confmaps`` (optional): ``(B, N, H, W)``

    ``Outputs.pred_keypoints`` uses ``(B, I, N, 2)`` so the adapter
    inserts a singleton instance dim.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
        return_confmaps: When ``True`` and the wrapper exports a
            ``confmaps`` output, echo it onto :attr:`Outputs.pred_confmaps`.
    """

    backend: Any
    return_confmaps: bool = False

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        peaks = raw["peaks"]  # (B, N, 2)
        vals = raw["peak_vals"]  # (B, N)
        # Add singleton instance dim per the Outputs convention.
        out = Outputs(
            pred_keypoints=peaks.unsqueeze(1),  # (B, 1, N, 2)
            pred_peak_values=vals.unsqueeze(1),  # (B, 1, N)
        )
        if self.return_confmaps and "confmaps" in raw:
            out = attrs.evolve(out, pred_confmaps=raw["confmaps"])
        return out


@attrs.define
class ExportedCenteredInstanceLayer:
    """Adapter for an ONNX/TRT-exported centered-instance model.

    Standalone centered-instance is invoked on pre-cropped images. The
    wrapper outputs the same shape as single-instance:

    * ``peaks``: ``(B_crops, N, 2)`` in crop (x, y) space
    * ``peak_vals``: ``(B_crops, N)``

    ``Outputs.pred_keypoints`` of shape ``(B_crops, 1, N, 2)``.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
        return_confmaps: Echo ``confmaps`` onto
            :attr:`Outputs.pred_confmaps` when present.
    """

    backend: Any
    return_confmaps: bool = False

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        peaks = raw["peaks"]  # (B_crops, N, 2)
        vals = raw["peak_vals"]
        out = Outputs(
            pred_keypoints=peaks.unsqueeze(1),  # (B_crops, 1, N, 2)
            pred_peak_values=vals.unsqueeze(1),  # (B_crops, 1, N)
        )
        if self.return_confmaps and "confmaps" in raw:
            out = attrs.evolve(out, pred_confmaps=raw["confmaps"])
        return out


@attrs.define
class ExportedCentroidLayer:
    """Adapter for an ONNX/TRT-exported centroid model.

    The wrapper output schema:

    * ``centroids``: ``(B, I, 2)`` in original-image (x, y) space.
      Invalid slots are zero-filled and flagged in ``instance_valid``.
    * ``centroid_vals``: ``(B, I)``
    * ``instance_valid``: ``(B, I)`` bool

    Translates to ``Outputs.pred_centroids`` / ``pred_centroid_values``.
    Invalid slots are turned into ``NaN`` per the ``Outputs`` convention.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
    """

    backend: Any

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        centroids = raw["centroids"].clone()  # (B, I, 2)
        centroid_vals = raw["centroid_vals"].clone()  # (B, I)
        valid = raw["instance_valid"].bool()  # (B, I)

        # Replace invalid zero-padded slots with NaN per Outputs convention.
        invalid = ~valid
        centroids[invalid] = float("nan")
        centroid_vals[invalid] = float("nan")

        return Outputs(
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            instance_valid=valid,
        )


@attrs.define
class ExportedBottomUpLayer:
    """Adapter for an ONNX/TRT-exported bottom-up model.

    The bottom-up wrapper (``BottomUpONNXWrapper``) bakes peak finding +
    PAF line scoring into the graph and returns fixed-shape tensors:

    * ``peaks``: ``(B, n_nodes, k, 2)`` in scaled-input pixel space
      (after multiplication by ``cms_output_stride`` inside the graph).
    * ``peak_vals``: ``(B, n_nodes, k)``
    * ``peak_mask``: ``(B, n_nodes, k)`` bool
    * ``line_scores``: ``(B, n_edges, k*k)``
    * ``candidate_mask``: ``(B, n_edges, k*k)`` bool

    What's still left for the CPU to do: the **grouping** stage —
    matching candidates per edge and assembling peaks into instances.
    The adapter translates the fixed-shape wrapper output into the
    variable-length per-sample :class:`ScoredBatch` format that
    :func:`group_scored_batch` expects, then runs the same grouping
    function the in-flow ``BottomUpLayer`` uses inline.

    Args:
        backend: A :class:`ModelBackend` whose ``does_baked_postproc``
            is ``True``.
        node_names: List of node names from ``ExportMetadata.node_names``.
        edge_inds: List of ``(src_node_idx, dst_node_idx)`` tuples from
            ``ExportMetadata.edge_inds``.
        max_peaks_per_node: ``k`` — must match the wrapper's setting
            (read from ``ExportMetadata.max_peaks_per_node``).
        input_scale: The wrapper's baked-in ``input_scale``. Used to
            unscale predicted instances back to original-image space.
        max_instances: Optional cap on instances per frame.
        min_instance_peaks: Drop assembled instances with fewer peaks.
        min_line_scores: Per-edge match threshold (forwarded to
            :class:`PAFScorer`).
    """

    backend: Any
    node_names: list
    edge_inds: list
    max_peaks_per_node: int
    input_scale: float = 1.0
    max_instances: Optional[int] = None
    min_instance_peaks: float = 0
    min_line_scores: float = 0.25

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend, translate to ``ScoredBatch``, run CPU grouping."""
        from sleap_nn.inference.preprocess_info import PreprocInfo
        from sleap_nn.inference.streaming import GroupingParams, group_scored_batch

        x = _to_4d_uint8_tensor(image)
        B, _C, H, W = x.shape
        raw = self.backend(x)

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=(H, W),
            eff_scale=torch.ones(B),
            input_scale=self.input_scale,
            output_stride=1,  # peaks already in scaled-input pixel space
            pad_amount=(0, 0),
            crop_offsets=None,
        )

        scored = self._build_scored_batch(raw, info)

        # Reconstruct PAFScorer kwargs. ``pafs_stride`` is unused on the
        # CPU grouping path (line scoring already happened in the graph),
        # so 1 is a safe placeholder.
        edges_by_name = [
            (self.node_names[s], self.node_names[d]) for s, d in self.edge_inds
        ]
        params = GroupingParams(
            paf_scorer_kwargs={
                "part_names": list(self.node_names),
                "edges": edges_by_name,
                "pafs_stride": 1,
                "max_edge_length_ratio": 0.25,
                "dist_penalty_weight": 1.0,
                "n_points": 10,
                "min_instance_peaks": self.min_instance_peaks,
                "min_line_scores": self.min_line_scores,
            },
            max_instances=self.max_instances,
        )
        return group_scored_batch(scored, params)

    def _build_scored_batch(self, raw: dict, info: Any) -> Any:
        """Convert wrapper output dict to a :class:`ScoredBatch`."""
        from sleap_nn.inference.streaming import ScoredBatch

        peaks = raw["peaks"]  # (B, n_nodes, k, 2)
        peak_vals = raw["peak_vals"]  # (B, n_nodes, k)
        peak_mask = raw["peak_mask"].bool()  # (B, n_nodes, k)
        line_scores_t = raw["line_scores"]  # (B, n_edges, k*k)
        candidate_mask = raw["candidate_mask"].bool()  # (B, n_edges, k*k)

        B, n_nodes, k, _ = peaks.shape
        n_edges = candidate_mask.shape[1]

        # (n_nodes * k,) — every k consecutive slots map to one node.
        channel_inds_flat = (
            torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, k).reshape(-1)
        )

        cms_peaks_list = []
        cms_peak_vals_list = []
        cms_channel_inds_list = []
        edge_inds_list = []
        edge_peak_inds_list = []
        sample_line_scores_list = []

        for b in range(B):
            mask_flat = peak_mask[b].reshape(-1)  # (n_nodes * k,)
            valid_idx = torch.nonzero(mask_flat, as_tuple=True)[0]  # (n_valid,)

            cms_peaks_list.append(peaks[b].reshape(-1, 2)[valid_idx])
            cms_peak_vals_list.append(peak_vals[b].reshape(-1)[valid_idx])
            cms_channel_inds_list.append(channel_inds_flat[valid_idx])

            # Map global index in (n_nodes * k) layout → compact index in
            # the valid-only list. Invalid slots get -1 (won't be referenced
            # by any candidate since candidate_mask is consistent with
            # peak_mask in the wrapper).
            global_to_compact = torch.full((n_nodes * k,), -1, dtype=torch.int64)
            global_to_compact[valid_idx] = torch.arange(
                len(valid_idx), dtype=torch.int64
            )

            sample_edge_inds = []
            sample_edge_peak_inds = []
            sample_line_scores_b = []
            for e_idx in range(n_edges):
                src_node, dst_node = self.edge_inds[e_idx]
                cm = candidate_mask[b, e_idx]  # (k*k,)
                ls = line_scores_t[b, e_idx]  # (k*k,)
                valid_cand = torch.nonzero(cm, as_tuple=True)[0]
                if valid_cand.numel() == 0:
                    continue
                src_k = valid_cand // k
                dst_k = valid_cand % k
                src_global = src_node * k + src_k
                dst_global = dst_node * k + dst_k
                src_compact = global_to_compact[src_global]
                dst_compact = global_to_compact[dst_global]
                ok = (src_compact >= 0) & (dst_compact >= 0)
                if not ok.any():
                    continue
                sample_edge_inds.append(
                    torch.full((int(ok.sum().item()),), e_idx, dtype=torch.int64)
                )
                sample_edge_peak_inds.append(
                    torch.stack([src_compact[ok], dst_compact[ok]], dim=-1)
                )
                sample_line_scores_b.append(ls[valid_cand][ok])

            if sample_edge_inds:
                edge_inds_list.append(torch.cat(sample_edge_inds))
                edge_peak_inds_list.append(torch.cat(sample_edge_peak_inds))
                sample_line_scores_list.append(torch.cat(sample_line_scores_b))
            else:
                edge_inds_list.append(torch.empty(0, dtype=torch.int64))
                edge_peak_inds_list.append(torch.empty(0, 2, dtype=torch.int64))
                sample_line_scores_list.append(torch.empty(0))

        return ScoredBatch(
            cms_peaks=cms_peaks_list,
            cms_peak_vals=cms_peak_vals_list,
            cms_peak_channel_inds=cms_channel_inds_list,
            edge_inds=edge_inds_list,
            edge_peak_inds=edge_peak_inds_list,
            line_scores=sample_line_scores_list,
            info=info,
            n_samples=B,
            n_nodes=n_nodes,
            skip_paf=False,
        )


@attrs.define
class ExportedTopDownLayer:
    """Adapter for an ONNX/TRT-exported combined top-down model.

    The export bakes both centroid + centered-instance stages plus the
    crop extraction step into a single ONNX graph. Wrapper output:

    * ``centroids``: ``(B, I, 2)`` in original-image (x, y) space
    * ``centroid_vals``: ``(B, I)``
    * ``peaks``: ``(B, I, N, 2)`` in original-image (x, y) space
      (crop offset already added back inside the graph)
    * ``peak_vals``: ``(B, I, N)``
    * ``instance_valid``: ``(B, I)`` bool

    Invalid slots are zero-filled by the wrapper and flagged in
    ``instance_valid``. The adapter NaN-pads invalid slots per the
    ``Outputs`` convention.

    Args:
        backend: A :class:`ModelBackend` whose
            ``does_baked_postproc`` is ``True``.
    """

    backend: Any

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_uint8_tensor(image)
        raw = self.backend(x)
        centroids = raw["centroids"].clone()
        centroid_vals = raw["centroid_vals"].clone()
        peaks = raw["peaks"].clone()
        peak_vals = raw["peak_vals"].clone()
        valid = raw["instance_valid"].bool()

        invalid = ~valid
        centroids[invalid] = float("nan")
        centroid_vals[invalid] = float("nan")
        peaks[invalid] = float("nan")
        peak_vals[invalid] = float("nan")

        return Outputs(
            pred_keypoints=peaks,
            pred_peak_values=peak_vals,
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            instance_valid=valid,
        )
