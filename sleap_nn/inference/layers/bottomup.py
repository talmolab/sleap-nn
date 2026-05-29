"""``BottomUpLayer`` — single-stage multi-instance inference via PAF grouping.

The bottom-up model emits two heads: a multi-instance confidence map
(``MultiInstanceConfmapsHead``) and part-affinity fields
(``PartAffinityFieldsHead``). The layer:

1. Runs the model.
2. Finds local peaks on the confmaps.
3. Scores PAF lines between candidate keypoints.
4. Groups peaks into instances via the existing :class:`PAFScorer`.
5. NaN-pads the variable per-frame instance count to ``max_instances``
   so ``Outputs.pred_keypoints`` retains its canonical
   ``(B, I, N, 2)`` shape.

Steps 1-3 are GPU-friendly tensor ops; step 4 is a CPU-bound
``scipy.linear_sum_assignment`` + BFS instance assembly. The two phases
are split into :meth:`_score_pafs_on_gpu` (GPU) and the free function
:func:`sleap_nn.inference.streaming.group_scored_batch` (CPU). The split
enables a worker pool for the CPU phase; the inline path simply calls
them back-to-back inside :meth:`postprocess`.
"""

from __future__ import annotations

from typing import Optional

import torch

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.paf import PAFScorer
from sleap_nn.inference.ops.peaks import find_local_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.streaming import (
    GroupingParams,
    ScoredBatch,
    group_scored_batch,
)


class BottomUpLayer(InferenceLayer):
    """Bottom-up multi-instance inference layer.

    Args:
        backend: Runtime backend wrapping the bottom-up Lightning module
            (which emits both confmaps and PAFs).
        paf_scorer: Pre-configured :class:`PAFScorer` for instance grouping.
        cms_output_stride: Stride between the confmap output and the
            scaled-input pixels.
        pafs_output_stride: Stride between the PAF output and the
            scaled-input pixels (often equal to ``cms_output_stride``).
        max_instances: Cap on instances per frame. Variable bottom-up
            output is NaN-padded to this fixed shape.
        max_stride: Maximum stride the model requires for input
            divisibility (padding applied bottom-right).
        max_peaks_per_node: Skip PAF scoring entirely if any node has
            more peaks than this — prevents combinatorial explosion on
            noisy early-training models. ``None`` disables.
        preprocess_config / postprocess_config: Standard knobs.
    """

    def __init__(
        self,
        backend: ModelBackend,
        paf_scorer: PAFScorer,
        cms_output_stride: int,
        pafs_output_stride: int,
        max_instances: Optional[int] = None,
        max_stride: int = 1,
        max_peaks_per_node: Optional[int] = None,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with the PAF scorer + stride config."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=cms_output_stride,
            max_stride=max_stride,
        )
        self.paf_scorer = paf_scorer
        self.cms_output_stride = cms_output_stride
        self.pafs_output_stride = pafs_output_stride
        self.max_instances = max_instances
        self.max_peaks_per_node = max_peaks_per_node

    # ──────────────────────────────────────────────────────────────────
    # GPU stage — peaks + PAF line scoring
    # ──────────────────────────────────────────────────────────────────

    def _score_pafs_on_gpu(self, raw_out: dict, info: PreprocInfo) -> ScoredBatch:
        """Run the GPU phase: local-peak find + PAF line scoring.

        Output is a CPU-resident :class:`ScoredBatch` so it can be
        shipped to a worker process verbatim (or fed straight into
        :func:`group_scored_batch` inline).
        """
        cms = raw_out["MultiInstanceConfmapsHead"]
        pafs = raw_out["PartAffinityFieldsHead"].permute(0, 2, 3, 1)  # (B, H, W, 2*E)

        peaks, peak_vals, sample_inds, peak_channel_inds = find_local_peaks(
            cms.detach(),
            threshold=self.postprocess_config.peak_threshold,
            refinement=self.postprocess_config.effective_refinement,
            integral_patch_size=self.postprocess_config.integral_patch_size,
        )
        peaks = peaks * self.cms_output_stride

        B = cms.shape[0]
        n_nodes = cms.shape[1]

        # Per-batch peak lists for the PAF scorer.
        cms_peaks: list[torch.Tensor] = []
        cms_peak_vals: list[torch.Tensor] = []
        cms_peak_channel_inds: list[torch.Tensor] = []
        for b in range(B):
            mask = sample_inds == b
            cms_peaks.append(peaks[mask])
            cms_peak_vals.append(peak_vals[mask].to(torch.float32))
            cms_peak_channel_inds.append(peak_channel_inds[mask])

        # Skip PAF scoring if any node has too many peaks (combinatorial blowup).
        skip_paf = False
        if self.max_peaks_per_node is not None:
            for ch_inds in cms_peak_channel_inds:
                for ni in range(n_nodes):
                    if int((ch_inds == ni).sum().item()) > self.max_peaks_per_node:
                        skip_paf = True
                        break
                if skip_paf:
                    break

        if skip_paf:
            return ScoredBatch(
                cms_peaks=cms_peaks,
                cms_peak_vals=cms_peak_vals,
                cms_peak_channel_inds=cms_peak_channel_inds,
                edge_inds=[],
                edge_peak_inds=[],
                line_scores=[],
                info=info,
                n_samples=B,
                n_nodes=n_nodes,
                skip_paf=True,
                cms=cms.detach() if self.postprocess_config.return_confmaps else None,
                pafs=pafs.detach() if self.postprocess_config.return_pafs else None,
            ).to_cpu()

        edge_inds, edge_peak_inds, line_scores = self.paf_scorer.score_paf_lines(
            pafs, cms_peaks, cms_peak_channel_inds
        )

        scored = ScoredBatch(
            cms_peaks=cms_peaks,
            cms_peak_vals=cms_peak_vals,
            cms_peak_channel_inds=cms_peak_channel_inds,
            edge_inds=edge_inds,
            edge_peak_inds=edge_peak_inds,
            line_scores=line_scores,
            info=info,
            n_samples=B,
            n_nodes=n_nodes,
            skip_paf=False,
            cms=(
                cms.detach()
                if (
                    self.postprocess_config.return_confmaps
                    or self.postprocess_config.return_paf_graph
                )
                else None
            ),
            pafs=(
                pafs.detach()
                if (
                    self.postprocess_config.return_pafs
                    or self.postprocess_config.return_paf_graph
                )
                else None
            ),
        )
        return scored.to_cpu()

    # ──────────────────────────────────────────────────────────────────
    # Layer-level grouping params (passed to inline + pool paths alike)
    # ──────────────────────────────────────────────────────────────────

    def grouping_params(self) -> GroupingParams:
        """Snapshot the layer's grouping configuration as a value type.

        The result is picklable and can be sent to a worker process.
        """
        # Prefer a predict-time override (set on ``postprocess_config`` by
        # ``Predictor._postprocess_overrides``) over the build-time value
        # carried on ``self.max_instances`` (#582).
        max_instances = getattr(self.postprocess_config, "max_instances", None)
        if max_instances is None:
            max_instances = self.max_instances
        return GroupingParams(
            paf_scorer_kwargs={
                "part_names": list(self.paf_scorer.part_names),
                "edges": [tuple(e) for e in self.paf_scorer.edges],
                "pafs_stride": self.paf_scorer.pafs_stride,
                "max_edge_length_ratio": self.paf_scorer.max_edge_length_ratio,
                "dist_penalty_weight": self.paf_scorer.dist_penalty_weight,
                "n_points": self.paf_scorer.n_points,
                "min_instance_peaks": self.paf_scorer.min_instance_peaks,
                "min_line_scores": self.paf_scorer.min_line_scores,
            },
            max_instances=max_instances,
            return_confmaps=self.postprocess_config.return_confmaps,
            return_pafs=self.postprocess_config.return_pafs,
            return_paf_graph=self.postprocess_config.return_paf_graph,
        )

    # ──────────────────────────────────────────────────────────────────
    # Postprocess — inline composition (parity path)
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """GPU peak/PAF scoring + CPU grouping, both in this process."""
        scored = self._score_pafs_on_gpu(raw_out, info)
        return group_scored_batch(scored, self.grouping_params())
