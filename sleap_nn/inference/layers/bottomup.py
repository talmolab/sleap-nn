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

PR 9 (#517) extends this with a CPU worker pool for the PAF grouping
step (currently the bottleneck on bottom-up inference). The current
implementation runs grouping inline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.paf import PAFScorer
from sleap_nn.inference.ops.peaks import find_local_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


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
        )
        self.paf_scorer = paf_scorer
        self.cms_output_stride = cms_output_stride
        self.pafs_output_stride = pafs_output_stride
        self.max_instances = max_instances
        self.max_stride = max_stride
        self.max_peaks_per_node = max_peaks_per_node

    # ──────────────────────────────────────────────────────────────────
    # Preprocess
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Resize, pad to stride, wrap with n_samples dim for Lightning forward."""
        x = self._to_4d_float_tensor(image)
        B, _C, H, W = x.shape

        scaled = (
            resize_image(x, self.preprocess_config.scale)
            if self.preprocess_config.scale != 1.0
            else x
        )
        if self.max_stride != 1:
            scaled = apply_pad_to_stride(scaled, self.max_stride)

        # BottomUpLightningModule.forward squeezes(dim=1) unconditionally.
        scaled_5d = scaled.unsqueeze(1)

        info = PreprocInfo(
            original_size=(H, W),
            processed_size=tuple(scaled.shape[-2:]),
            eff_scale=torch.ones(B),
            input_scale=self.preprocess_config.scale,
            output_stride=self.cms_output_stride,
        )
        return scaled_5d, info

    # ──────────────────────────────────────────────────────────────────
    # Postprocess
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Find peaks, group via PAF scoring, NaN-pad to ``max_instances``."""
        cms = raw_out["MultiInstanceConfmapsHead"]
        pafs = raw_out["PartAffinityFieldsHead"].permute(0, 2, 3, 1)  # (B, H, W, 2*E)

        # Find peaks on the confmaps.
        refinement = (
            self.postprocess_config.refinement
            if self.postprocess_config.refinement != "none"
            else None
        )
        peaks, peak_vals, sample_inds, peak_channel_inds = find_local_peaks(
            cms.detach(),
            threshold=self.postprocess_config.peak_threshold,
            refinement=refinement,
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
            return self._empty_outputs(B, n_nodes, info, cms, pafs)

        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
            edge_inds,
            edge_peak_inds,
            line_scores,
        ) = self.paf_scorer.predict(
            pafs=pafs,
            peaks=cms_peaks,
            peak_vals=cms_peak_vals,
            peak_channel_inds=cms_peak_channel_inds,
        )

        # Apply input-scale + eff-scale corrections per sample.
        if info.input_scale != 1.0:
            predicted_instances = [p / info.input_scale for p in predicted_instances]
        eff = info.eff_scale
        if not torch.all(eff == 1.0):
            predicted_instances = [
                p / eff[i].to(p.device) for i, p in enumerate(predicted_instances)
            ]

        # NaN-pad each batch's instance list into a uniform (B, I, N, 2) tensor.
        max_instances = self.max_instances or self._infer_max_instances(
            predicted_instances
        )
        if max_instances == 0:
            max_instances = 1
        full_kpts = torch.full((B, max_instances, n_nodes, 2), float("nan"))
        full_vals = torch.full((B, max_instances, n_nodes), float("nan"))
        full_scores = torch.full((B, max_instances), float("nan"))
        for b in range(B):
            instances_b = predicted_instances[b]  # (n_b, n_nodes, 2)
            vals_b = predicted_peak_scores[b]  # (n_b, n_nodes)
            scores_b = predicted_instance_scores[b]  # (n_b,)
            n = min(int(instances_b.shape[0]), max_instances)
            if n == 0:
                continue
            full_kpts[b, :n] = instances_b[:n]
            full_vals[b, :n] = vals_b[:n]
            full_scores[b, :n] = scores_b[:n]

        outputs = Outputs(
            pred_keypoints=full_kpts,
            pred_peak_values=full_vals,
            instance_scores=full_scores,
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps:
            outputs = attrs.evolve(outputs, pred_confmaps=cms.detach())
        if self.postprocess_config.return_pafs:
            # Original (B, 2E, H, W) layout; consumers expect this.
            outputs = attrs.evolve(outputs, pred_pafs=pafs.permute(0, 3, 1, 2).detach())
        if self.postprocess_config.return_paf_graph:
            outputs = attrs.evolve(
                outputs,
                pred_paf_graph=(
                    (
                        torch.cat([t for t in cms_peaks], dim=0)
                        if cms_peaks
                        else torch.empty(0, 2)
                    ),
                    (
                        torch.cat([t for t in edge_inds], dim=0)
                        if edge_inds
                        else torch.empty(0, dtype=torch.int32)
                    ),
                    (
                        torch.cat([t for t in edge_peak_inds], dim=0)
                        if edge_peak_inds
                        else torch.empty(0, 2, dtype=torch.int32)
                    ),
                    (
                        torch.cat([t for t in line_scores], dim=0)
                        if line_scores
                        else torch.empty(0)
                    ),
                ),
            )
        return outputs

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_max_instances(predicted_instances: list[torch.Tensor]) -> int:
        """Largest per-batch instance count, or 0 if all empty."""
        return max((int(p.shape[0]) for p in predicted_instances), default=0)

    def _empty_outputs(
        self,
        B: int,
        n_nodes: int,
        info: PreprocInfo,
        cms: torch.Tensor,
        pafs: torch.Tensor,
    ) -> Outputs:
        """Return an Outputs with all-NaN keypoints — the skip-PAF branch."""
        max_instances = self.max_instances or 1
        outputs = Outputs(
            pred_keypoints=torch.full((B, max_instances, n_nodes, 2), float("nan")),
            pred_peak_values=torch.full((B, max_instances, n_nodes), float("nan")),
            instance_scores=torch.full((B, max_instances), float("nan")),
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps:
            outputs = attrs.evolve(outputs, pred_confmaps=cms.detach())
        if self.postprocess_config.return_pafs:
            outputs = attrs.evolve(outputs, pred_pafs=pafs.permute(0, 3, 1, 2).detach())
        return outputs
