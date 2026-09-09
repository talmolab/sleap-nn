"""``BottomUpMultiClassLayer`` — multi-class variant of bottom-up inference.

Same backbone as :class:`BottomUpLayer`, but the model emits
``MultiInstanceConfmapsHead`` + ``ClassMapsHead`` instead of confmaps +
PAFs. Instance grouping is by class identity (via
:func:`classify_peaks_from_maps`) rather than PAF scoring.
"""

from __future__ import annotations

from typing import List, Optional

import attrs
import numpy as np
import torch

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.ops.identity import classify_peaks_from_maps
from sleap_nn.inference.ops.peaks import find_local_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


class BottomUpMultiClassLayer(InferenceLayer):
    """Bottom-up multi-class inference layer.

    Args:
        backend: Runtime backend wrapping the multi-class bottomup
            Lightning module.
        cms_output_stride: Stride for ``MultiInstanceConfmapsHead``.
        class_maps_output_stride: Stride for ``ClassMapsHead``.
        max_stride: Max stride the model requires for input divisibility.
        max_instances: Cap on instances per frame. Since each instance slot is
            a fixed class index, the cap is applied by NaN-masking the
            lowest-scoring class slots (not by reordering), preserving the
            ``slot == class`` identity used for track assignment (legacy parity,
            #582).
        preprocess_config / postprocess_config: Standard knobs.
        class_names: Ordered class names from
            ``multi_class_bottomup.class_maps.classes``. Used by the predictor
            to build the ``sio.Track`` registry for identity packaging. The
            grouped instance slot ``i`` corresponds to ``class_names[i]``.
        class_uuids: Ordered per-class canonical identity UUIDs from
            ``multi_class_bottomup.class_maps.class_uuids`` (parallel to
            ``class_names``). Used by the predictor to build the canonical
            ``sio.Identity`` registry (the train→inference uuid bridge).
            ``None`` for legacy checkpoints (the predictor then mints UUIDs).
    """

    def __init__(
        self,
        backend: ModelBackend,
        cms_output_stride: int,
        class_maps_output_stride: int,
        max_stride: int = 1,
        max_instances: Optional[int] = None,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
        class_names: Optional[List[str]] = None,
        class_uuids: Optional[List[str]] = None,
        class_output: str = "track",
    ) -> None:
        """Compose the layer with the two output strides."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=cms_output_stride,
            max_stride=max_stride,
        )
        self.cms_output_stride = cms_output_stride
        self.class_maps_output_stride = class_maps_output_stride
        self.max_instances = max_instances
        self.class_names = list(class_names) if class_names is not None else None
        self.class_uuids = list(class_uuids) if class_uuids is not None else None
        self.class_output = class_output

    # ──────────────────────────────────────────────────────────────────
    # Postprocess (class-maps based grouping)
    # ──────────────────────────────────────────────────────────────────

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps + class maps to per-class instance keypoints."""
        cms = raw_out["MultiInstanceConfmapsHead"]
        class_maps = raw_out["ClassMapsHead"]  # (B, n_classes, H, W)

        peaks, peak_vals, sample_inds, channel_inds = find_local_peaks(
            cms.detach(),
            threshold=self.postprocess_config.peak_threshold,
            refinement=self.postprocess_config.effective_refinement,
            integral_patch_size=self.postprocess_config.integral_patch_size,
        )
        # Stride-adjust peaks to the input image space, then divide by the
        # class-maps stride so the indices into ``class_maps`` are correct.
        peaks = peaks * self.cms_output_stride
        peaks_for_classmap = peaks / self.class_maps_output_stride

        n_nodes = cms.shape[1]
        instances, peak_scores, class_probs = classify_peaks_from_maps(
            class_maps.detach(),
            peaks_for_classmap,
            peak_vals,
            sample_inds,
            channel_inds,
            n_channels=n_nodes,
        )

        # Lift instances back to image-space and apply input/eff scaling.
        # Shape: (B, n_classes, n_nodes, 2)
        instances = instances * self.class_maps_output_stride
        if info.input_scale != 1.0:
            instances = instances / info.input_scale
        eff = info.eff_scale.to(instances.device)
        if not torch.all(eff == 1.0):
            instances = instances / eff.view(-1, 1, 1, 1)

        # Legacy parity (predictors.py:2987-3010): per instance ``i`` (which
        # IS the class index for class-maps grouping),
        #   score = np.nanmean(confs)        # mean of the confmap peak values
        #   tracking_score = np.nanmean(class_score)  # mean class probability
        # Both are reduced over the node axis to satisfy the (B, I) contract
        # that downstream filters + Outputs.to_instances expect. Carrying them
        # in separate fields keeps ``score`` (instance_scores) distinct from
        # ``tracking_score`` (instance_tracking_scores).
        instance_scores = torch.nanmean(peak_scores, dim=-1)
        instance_tracking_scores = torch.nanmean(class_probs, dim=-1)

        # Cap instances per frame by NaN-masking the lowest-scoring class slots,
        # preserving slot==class identity (legacy parity — legacy sorted by
        # score and truncated to max_instances after class assignment). A
        # predict-time override (postprocess_config) takes precedence over the
        # build-time value (#582).
        max_instances = getattr(self.postprocess_config, "max_instances", None)
        if max_instances is None:
            max_instances = self.max_instances
        if max_instances is not None:
            instances, peak_scores, instance_scores, instance_tracking_scores = (
                self._cap_instances_by_score(
                    instances,
                    peak_scores,
                    instance_scores,
                    instance_tracking_scores,
                    max_instances,
                )
            )

        outputs = Outputs(
            pred_keypoints=instances,
            pred_peak_values=peak_scores,
            instance_scores=instance_scores,
            instance_tracking_scores=instance_tracking_scores,
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps:
            outputs = attrs.evolve(outputs, pred_confmaps=cms.detach())
        if self.postprocess_config.return_class_maps:
            outputs = attrs.evolve(outputs, pred_class_maps=class_maps.detach())
        return outputs

    @staticmethod
    def _cap_instances_by_score(
        instances: torch.Tensor,
        peak_scores: torch.Tensor,
        instance_scores: torch.Tensor,
        instance_tracking_scores: torch.Tensor,
        max_instances: int,
    ) -> tuple:
        """NaN-mask the lowest-scoring class slots beyond ``max_instances``.

        Per frame, the present classes are those with a non-NaN
        ``instance_score`` (a class with no assigned peaks has an all-NaN peak
        row → NaN score). When more than ``max_instances`` classes are present,
        the lowest-scoring ones are masked out (keypoints / peak values /
        scores set to NaN) so the frame yields at most ``max_instances``
        instances, while every surviving instance keeps its original class slot
        (and therefore its track). Mirrors the legacy top-N-by-score truncation
        without reordering.
        """
        B = instance_scores.shape[0]
        for b in range(B):
            scores_b = instance_scores[b]  # (n_classes,)
            valid = ~torch.isnan(scores_b)
            n_valid = int(valid.sum().item())
            if n_valid <= max_instances:
                continue
            # Rank present classes by score (descending) and keep the top
            # ``max_instances``. NaN scores sort last via argsort, matching the
            # legacy ``np.argsort(scores)[::-1]`` selection.
            order = np.argsort(scores_b.detach().cpu().numpy())[::-1]
            keep = torch.as_tensor(order[:max_instances].copy(), dtype=torch.long)
            drop_mask = torch.ones(scores_b.shape[0], dtype=torch.bool)
            drop_mask[keep] = False
            instances[b, drop_mask] = float("nan")
            peak_scores[b, drop_mask] = float("nan")
            instance_scores[b, drop_mask] = float("nan")
            instance_tracking_scores[b, drop_mask] = float("nan")
        return instances, peak_scores, instance_scores, instance_tracking_scores
