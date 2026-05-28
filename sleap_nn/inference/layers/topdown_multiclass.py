"""``TopDownMultiClassLayer`` — multi-class variant of top-down inference.

Composes :class:`CentroidLayer` with a multi-class centered-instance
``CenteredInstanceMultiClassLayer`` (defined in this module). The
multi-class centered-instance model emits keypoint confmaps **plus** a
``ClassVectorsHead`` per crop; the layer adds a class-prob output to
``Outputs.instance_scores`` and a per-node class-index field.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.ops.identity import get_class_inds_from_vectors
from sleap_nn.inference.ops.peaks import find_global_peaks
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


class CenteredInstanceMultiClassLayer(InferenceLayer):
    """Centered-instance + per-instance class vector head.

    Per-crop, returns keypoints (like ``CenteredInstanceLayer``) plus a
    class index and class probability from a softmax-classified
    ``ClassVectorsHead``.

    Args:
        backend: Runtime backend wrapping the multi-class centered-
            instance Lightning module.
        output_stride: Confmap output stride.
        max_stride: Max stride for input divisibility (preprocess pads
            bottom-right).
        preprocess_config / postprocess_config: Standard knobs.

    Notes:
        Class-level ``use_gt_peaks = False``: the multi-class variant does
        not support the GT-keypoints fallback path (you can't match GT
        classes to GT keypoints the way the plain centered-instance layer
        matches centroids → keypoints). The attribute is exposed so
        :class:`TopDownLayer`-derived composers can branch on it
        uniformly without isinstance checks.
    """

    use_gt_peaks: bool = False

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_stride: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with the standard centered-instance config."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
        )
        self.max_stride = max_stride

    def preprocess(self, image: ImageInput) -> Tuple[torch.Tensor, PreprocInfo]:
        """Resize + max-stride pad, wrap to 5D for Lightning forward."""
        x = self._to_4d_float_tensor(image)
        B, _C, H, W = x.shape
        scaled = (
            resize_image(x, self.preprocess_config.scale)
            if self.preprocess_config.scale != 1.0
            else x
        )
        if self.max_stride != 1:
            scaled = apply_pad_to_stride(scaled, self.max_stride)
        scaled_5d = scaled.unsqueeze(1)
        info = PreprocInfo(
            original_size=(H, W),
            processed_size=tuple(scaled.shape[-2:]),
            eff_scale=torch.ones(B),
            input_scale=self.preprocess_config.scale,
            output_stride=self.output_stride,
        )
        return scaled_5d, info

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode confmaps to keypoints; classify via ``ClassVectorsHead``."""
        cms = raw_out["CenteredInstanceConfmapsHead"]
        peak_class_probs = raw_out["ClassVectorsHead"]  # (n_crops, n_classes)

        refinement = (
            self.postprocess_config.refinement
            if self.postprocess_config.refinement != "none"
            else None
        )
        peaks, vals = find_global_peaks(
            cms.detach(),
            threshold=self.postprocess_config.peak_threshold,
            refinement=refinement,
            integral_patch_size=self.postprocess_config.integral_patch_size,
        )
        peaks = peaks * info.output_stride
        if info.input_scale != 1.0:
            peaks = peaks / info.input_scale
        eff = info.eff_scale.to(peaks.device)
        if not torch.all(eff == 1.0):
            peaks = peaks / eff.view(-1, 1, 1)

        class_inds, class_probs = get_class_inds_from_vectors(peak_class_probs)

        # Reshape peaks to canonical (B=n_crops, I=1, N, 2) Outputs shape.
        peaks_BIN2 = peaks.unsqueeze(1)
        vals_BIN = vals.unsqueeze(1)
        class_inds_BIN = class_inds.unsqueeze(1)  # (n_crops, 1)
        class_probs_BI = class_probs.unsqueeze(1)  # (n_crops, 1)

        outputs = Outputs(
            pred_keypoints=peaks_BIN2,
            pred_peak_values=vals_BIN,
            pred_class_inds=class_inds_BIN.unsqueeze(-1).expand(-1, -1, peaks.shape[1]),
            instance_scores=class_probs_BI,
            preprocess_info=info,
        )
        if self.postprocess_config.return_confmaps:
            outputs = attrs.evolve(outputs, pred_confmaps=cms.detach())
        if self.postprocess_config.return_class_vectors:
            outputs = attrs.evolve(
                outputs, pred_class_vectors=peak_class_probs.detach()
            )
        return outputs


class TopDownMultiClassLayer(TopDownLayer):
    """Top-down with per-instance class identity.

    Same composition as :class:`TopDownLayer` (stage 1 = centroid, stage
    2 = centered-instance) but the centered-instance layer is a
    :class:`CenteredInstanceMultiClassLayer` whose output carries class
    indices + probabilities. The composition logic is unchanged — class
    fields propagate through stage 2 unchanged.
    """

    def __init__(
        self,
        centroid_layer: CentroidLayer,
        centered_instance_layer: CenteredInstanceMultiClassLayer,
        crop_size: Tuple[int, int],
        centroid_nms: bool = False,
        centroid_nms_threshold: float = 0.5,
    ) -> None:
        """Forward to ``TopDownLayer`` after type-checking the inner layer."""
        if not isinstance(centered_instance_layer, CenteredInstanceMultiClassLayer):
            raise TypeError(
                "TopDownMultiClassLayer requires a CenteredInstanceMultiClassLayer "
                "for the centered_instance_layer argument; got "
                f"{type(centered_instance_layer).__name__}."
            )
        super().__init__(
            centroid_layer=centroid_layer,
            centered_instance_layer=centered_instance_layer,
            crop_size=crop_size,
            centroid_nms=centroid_nms,
            centroid_nms_threshold=centroid_nms_threshold,
        )
