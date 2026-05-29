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


def _to_4d_tensor_for_export(image: Any) -> torch.Tensor:
    """Coerce an image input to ``(B, C, H, W)`` channel-first, preserving dtype.

    Uses the dtype-preserving :meth:`InferenceLayer._to_4d_tensor` so a uint8
    input stays uint8 (matching the legacy exporter's uint8 NCHW fast path)
    instead of being force-cast to float32. Each backend re-casts to its
    engine's declared input dtype: ``ONNXBackend`` via ``np.astype`` and
    ``TensorRTBackend`` via ``x.to(...)``, so any input dtype is accepted.
    """
    return InferenceLayer._to_4d_tensor(image)


def _raw_to_cpu(raw: dict) -> dict:
    """Move every tensor in a backend output dict to CPU.

    The grouping-based adapters build index/output tensors on the default
    (CPU) device and index them with masks derived from the backend output.
    ``ONNXBackend`` returns CPU tensors, but ``TensorRTBackend`` returns CUDA
    tensors, which makes those mixed-device index/assign ops crash on a GPU
    host (invisible to CPU-only CI). Legacy export inference moved all engine
    outputs to CPU before grouping; mirror that here (#582). No-op on CPU.
    """
    return {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in raw.items()
    }


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
        x = _to_4d_tensor_for_export(image)
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
        x = _to_4d_tensor_for_export(image)
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
        anchor_ind: Skeleton node index the centroid represents (resolved
            from ``ExportMetadata.anchor_part``). Read at packaging time by
            ``Predictor._packaging_anchor_ind`` so the centroid lands on the
            configured anchor node rather than node 0 (#582). ``None`` (old
            exports without ``anchor_part``) keeps the node-0 behavior.
    """

    backend: Any
    anchor_ind: Optional[int] = None

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend and translate to :class:`Outputs`."""
        x = _to_4d_tensor_for_export(image)
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
        peak_conf_threshold: Optional runtime peak-confidence threshold. When
            set, PAF candidate connections whose src or dst peak confidence is
            ``<= peak_conf_threshold`` are dropped before grouping (legacy
            parity for the runtime threshold override, #582). ``None`` keeps
            every candidate the wrapper emitted.
    """

    backend: Any
    node_names: list
    edge_inds: list
    max_peaks_per_node: int
    input_scale: float = 1.0
    max_instances: Optional[int] = None
    min_instance_peaks: float = 0
    min_line_scores: float = 0.25
    peak_conf_threshold: Optional[float] = None

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend, translate to ``ScoredBatch``, run CPU grouping."""
        from sleap_nn.inference.preprocess_info import PreprocInfo
        from sleap_nn.inference.streaming import GroupingParams, group_scored_batch

        x = _to_4d_tensor_for_export(image)
        B, _C, H, W = x.shape
        raw = _raw_to_cpu(self.backend(x))

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
                # Runtime peak-confidence gate: drop candidate connections whose
                # src/dst peak confidence is below the threshold (legacy parity,
                # #582). The wrapper already baked a peak threshold, so this can
                # only tighten further.
                if self.peak_conf_threshold is not None:
                    pv_flat = peak_vals[b].reshape(-1)
                    thr = self.peak_conf_threshold
                    ok = ok & (pv_flat[src_global] > thr) & (pv_flat[dst_global] > thr)
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
        x = _to_4d_tensor_for_export(image)
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


@attrs.define
class ExportedTopDownMultiClassLayer:
    """Adapter for an ONNX/TRT-exported combined top-down multi-class model.

    Like :class:`ExportedTopDownLayer` plus a per-instance class-logits
    output. Wrapper schema:

    * ``centroids``: ``(B, I, 2)`` in original-image space.
    * ``centroid_vals``: ``(B, I)``
    * ``peaks``: ``(B, I, N, 2)`` in original-image space.
    * ``peak_vals``: ``(B, I, N)``
    * ``class_logits``: ``(B, I, n_classes)`` — raw logits.
    * ``instance_valid``: ``(B, I)`` bool.

    The legacy driver applies softmax to ``class_logits`` and runs
    Hungarian matching to assign each valid instance to a unique class.
    The adapter mirrors that and **reorders instances** so each output
    slot ``(b, c, ...)`` holds the instance assigned to class ``c``.
    Slots without a matching instance are NaN-padded.

    Args:
        backend: A :class:`ModelBackend` whose ``does_baked_postproc``
            is ``True``.
        n_classes: Number of identity classes (= ``I`` in the output).
    """

    backend: Any
    n_classes: int

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend, run softmax + Hungarian, build :class:`Outputs`."""
        from sleap_nn.inference.ops.identity import get_class_inds_from_vectors

        x = _to_4d_tensor_for_export(image)
        raw = _raw_to_cpu(self.backend(x))

        centroids = raw["centroids"]
        centroid_vals = raw["centroid_vals"]
        peaks = raw["peaks"]
        peak_vals = raw["peak_vals"]
        class_logits = raw["class_logits"]
        valid = raw["instance_valid"].bool()

        B = peaks.shape[0]
        N = peaks.shape[2]
        C = self.n_classes

        # Allocate class-ordered slots, NaN-padded.
        out_centroids = torch.full((B, C, 2), float("nan"))
        out_centroid_vals = torch.full((B, C), float("nan"))
        out_peaks = torch.full((B, C, N, 2), float("nan"))
        out_peak_vals = torch.full((B, C, N), float("nan"))
        out_class_probs = torch.full((B, C, C), float("nan"))
        out_valid = torch.zeros((B, C), dtype=torch.bool)

        for b in range(B):
            v = valid[b]
            if not v.any():
                continue
            # Softmax of valid instances' logits, then Hungarian matching.
            valid_logits = class_logits[b, v]  # (n_valid, C)
            probs = torch.softmax(valid_logits, dim=1)
            class_inds, _ = get_class_inds_from_vectors(probs)
            valid_idx = torch.nonzero(v, as_tuple=True)[0]  # original instance indices
            for local_i, c in enumerate(class_inds.tolist()):
                if c < 0 or c >= C:
                    continue
                src = int(valid_idx[local_i].item())
                out_centroids[b, c] = centroids[b, src]
                out_centroid_vals[b, c] = centroid_vals[b, src]
                out_peaks[b, c] = peaks[b, src]
                out_peak_vals[b, c] = peak_vals[b, src]
                out_class_probs[b, c] = probs[local_i]
                out_valid[b, c] = True

        return Outputs(
            pred_keypoints=out_peaks,
            pred_peak_values=out_peak_vals,
            pred_centroids=out_centroids,
            pred_centroid_values=out_centroid_vals,
            pred_class_probs=out_class_probs,
            instance_valid=out_valid,
        )


@attrs.define
class ExportedBottomUpMultiClassLayer:
    """Adapter for an ONNX/TRT-exported bottom-up multi-class model.

    Multi-class bottom-up replaces PAF-based grouping with class-map
    grouping: every detected peak gets a class probability vector, and
    Hungarian matching assigns peaks to classes per (sample, node).

    Wrapper schema:

    * ``peaks``: ``(B, n_nodes, k, 2)`` in scaled-input pixel space.
    * ``peak_vals``: ``(B, n_nodes, k)``
    * ``peak_mask``: ``(B, n_nodes, k)`` bool — invalid slots zeroed.
    * ``class_probs``: ``(B, n_nodes, k, n_classes)`` — sampled at peaks.

    The adapter flattens valid peaks per sample, runs
    :func:`group_class_peaks` (the same Hungarian-matching primitive
    the in-flow ``BottomUpMultiClassLayer`` uses), and scatters the
    grouped peaks into a fixed ``(B, n_classes, n_nodes, 2)``.

    Args:
        backend: A :class:`ModelBackend` whose ``does_baked_postproc``
            is ``True``.
        n_nodes: Number of skeleton nodes.
        n_classes: Number of identity classes (= ``I`` in the output).
        input_scale: Wrapper's baked-in ``input_scale``. Used to
            unscale predicted points back to original-image space.
        peak_conf_threshold: Optional runtime peak-confidence threshold.
            When set, peaks whose confidence is ``<= peak_conf_threshold``
            are dropped before class grouping (legacy parity — the legacy
            exported multi-class bottom-up path gated peaks the same way,
            #582). ``None`` keeps every peak the wrapper emitted.
    """

    backend: Any
    n_nodes: int
    n_classes: int
    input_scale: float = 1.0
    peak_conf_threshold: Optional[float] = None

    def predict(self, image: Any, **_kwargs: Any) -> Outputs:
        """Run the backend, flatten + group peaks by class, build ``Outputs``."""
        from sleap_nn.inference.ops.identity import group_class_peaks

        x = _to_4d_tensor_for_export(image)
        raw = _raw_to_cpu(self.backend(x))

        peaks = raw["peaks"]  # (B, n_nodes, k, 2)
        peak_vals = raw["peak_vals"]  # (B, n_nodes, k)
        peak_mask = raw["peak_mask"].bool()  # (B, n_nodes, k)
        # Runtime peak-confidence gate before class grouping (legacy parity).
        if self.peak_conf_threshold is not None:
            peak_mask = peak_mask & (peak_vals > self.peak_conf_threshold)
        class_probs_raw = raw["class_probs"]  # (B, n_nodes, k, n_classes)

        B, n_nodes, k, _ = peaks.shape
        C = self.n_classes

        # Flatten across (B, n_nodes, k) → keep only mask=True entries.
        flat_mask = peak_mask.reshape(-1)  # (B*n_nodes*k,)
        valid_idx = torch.nonzero(flat_mask, as_tuple=True)[0]

        if valid_idx.numel() == 0:
            return Outputs(
                pred_keypoints=torch.full((B, C, n_nodes, 2), float("nan")),
                pred_peak_values=torch.full((B, C, n_nodes), float("nan")),
                instance_valid=torch.zeros((B, C), dtype=torch.bool),
            )

        flat_peaks = peaks.reshape(-1, 2)[valid_idx]  # (n_valid, 2)
        flat_vals = peak_vals.reshape(-1)[valid_idx]
        flat_class_probs = class_probs_raw.reshape(-1, C)[valid_idx]

        # Recover (sample, channel) indices for each surviving peak.
        bnk = torch.arange(B * n_nodes * k)[valid_idx]
        peak_sample_inds = (bnk // (n_nodes * k)).long()
        peak_channel_inds = ((bnk // k) % n_nodes).long()

        # Hungarian-match peaks to classes per (sample, node).
        peak_inds, class_inds = group_class_peaks(
            flat_class_probs,
            peak_sample_inds,
            peak_channel_inds,
            n_samples=B,
            n_channels=n_nodes,
        )

        # Scatter into (B, C, n_nodes, 2) NaN-padded output.
        out_points = torch.full((B, C, n_nodes, 2), float("nan"))
        out_vals = torch.full((B, C, n_nodes), float("nan"))
        out_class_vectors = torch.full((B, C, n_nodes, C), float("nan"))
        for i in range(int(peak_inds.numel())):
            pi = int(peak_inds[i].item())
            ci = int(class_inds[i].item())
            b = int(peak_sample_inds[pi].item())
            n = int(peak_channel_inds[pi].item())
            out_points[b, ci, n] = flat_peaks[pi]
            out_vals[b, ci, n] = flat_vals[pi]
            out_class_vectors[b, ci, n] = flat_class_probs[pi]

        # Unscale to original-image space.
        if self.input_scale != 1.0:
            out_points = out_points / self.input_scale

        # An instance slot is "valid" if any of its nodes got a peak.
        out_valid = ~torch.isnan(out_points[..., 0]).all(dim=-1)

        return Outputs(
            pred_keypoints=out_points,
            pred_peak_values=out_vals,
            pred_class_vectors=out_class_vectors,
            instance_valid=out_valid,
        )
