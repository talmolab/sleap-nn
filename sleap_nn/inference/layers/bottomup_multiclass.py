"""``BottomUpMultiClassLayer`` — multi-class variant of bottom-up inference.

Same backbone as :class:`BottomUpLayer`, but the model emits
``MultiInstanceConfmapsHead`` + ``ClassMapsHead`` instead of confmaps +
PAFs. Instance grouping is by class identity (via
:func:`classify_peaks_from_maps`) rather than PAF scoring.
"""

from __future__ import annotations

from typing import List, Optional

import attrs
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
        preprocess_config / postprocess_config: Standard knobs.
        class_names: Ordered class names from
            ``multi_class_bottomup.class_maps.classes``. Used by the predictor
            to build the ``sio.Track`` registry for identity packaging. The
            grouped instance slot ``i`` corresponds to ``class_names[i]``.
    """

    def __init__(
        self,
        backend: ModelBackend,
        cms_output_stride: int,
        class_maps_output_stride: int,
        max_stride: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
        class_names: Optional[List[str]] = None,
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
        self.class_names = list(class_names) if class_names is not None else None

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
