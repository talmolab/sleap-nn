"""``SegmentationLayer`` — bottom-up instance segmentation inference.

Wraps a trained ``BottomUpSegmentationLightningModule`` (whose ``forward``
returns the three head maps, with sigmoid already applied to the foreground
head). ``postprocess`` groups foreground pixels into instances via the
predicted instance-center offsets, maps each mask back to original-image
resolution, and packages them into ``Outputs.pred_masks``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.segmentation import group_instances_from_offsets


class SegmentationLayer(InferenceLayer):
    """Bottom-up instance-segmentation prediction layer.

    Args:
        backend: Runtime backend wrapping the segmentation Lightning module.
            Its ``forward`` returns ``{"SegmentationHead", "InstanceCenterHead",
            "CenterOffsetHead"}`` with sigmoid already applied to the
            foreground head.
        output_stride: Stride of the head output maps relative to the model
            input (all three heads share it).
        max_stride: Backbone max stride; the input is padded to a multiple of
            it during preprocessing.
        fg_threshold: Foreground probability threshold for binarization.
        min_mask_area: Minimum area (in ORIGINAL-image pixels) for a predicted
            mask to be kept. Masks smaller than this are dropped — useful for
            suppressing tiny spurious blobs (over-segmentation). ``0`` disables
            the filter (only empty masks are dropped).
        preprocess_config / postprocess_config: Standard knobs;
            ``postprocess_config.peak_threshold`` is the instance-center peak
            threshold (overridable via ``Predictor.predict(peak_threshold=...)``).
    """

    _SEG_KEY = "SegmentationHead"
    _CENTER_KEY = "InstanceCenterHead"
    _OFFSET_KEY = "CenterOffsetHead"

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_stride: int = 1,
        fg_threshold: float = 0.5,
        min_mask_area: int = 0,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Store thresholds and standard configs."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config
            or PostprocessConfig(peak_threshold=0.2),
            output_stride=output_stride,
            max_stride=max_stride,
        )
        self.fg_threshold = fg_threshold
        self.min_mask_area = int(min_mask_area)

    @property
    def warmup_input_shape(self):
        """Tiny single-channel warmup shape."""
        return (1, 1, 64, 64)

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Group foreground pixels into instances and package masks.

        Args:
            raw_out: Backend output dict with the three segmentation heads.
            info: Preprocessing metadata for mapping masks back to the
                original image resolution.

        Returns:
            ``Outputs`` with ``pred_masks`` populated (a per-batch list of
            per-instance ``{"mask", "score"}`` dicts at original resolution).
        """
        foreground = raw_out[self._SEG_KEY]  # (B, 1, h, w), already sigmoid
        center_heatmap = raw_out[self._CENTER_KEY]  # (B, 1, h, w)
        offsets = raw_out[self._OFFSET_KEY]  # (B, 2, h, w)

        foreground = foreground.detach().cpu()
        center_heatmap = center_heatmap.detach().cpu()
        offsets = offsets.detach().cpu()

        peak_threshold = self.postprocess_config.peak_threshold
        B = foreground.shape[0]
        pred_masks: List[List[dict]] = []
        for b in range(B):
            instances = group_instances_from_offsets(
                foreground=foreground[b : b + 1],
                center_heatmap=center_heatmap[b : b + 1],
                offsets=offsets[b : b + 1],
                fg_threshold=self.fg_threshold,
                peak_threshold=peak_threshold,
                output_stride=self.output_stride,
            )
            frame_masks: List[dict] = []
            # Drop empty masks and, when ``min_mask_area > 0``, tiny spurious
            # masks below the area floor (in original-image pixels). The
            # ``max(1, ...)`` keeps the empty-mask drop when the filter is off.
            area_floor = max(1, self.min_mask_area)
            for inst in instances:
                mask_full = self._mask_to_original(inst["mask"], info, b)
                if int(mask_full.sum()) < area_floor:
                    continue
                frame_masks.append({"mask": mask_full, "score": float(inst["score"])})
            pred_masks.append(frame_masks)

        return Outputs(pred_masks=pred_masks, preprocess_info=info)

    def _mask_to_original(
        self, mask: np.ndarray, info: PreprocInfo, b: int
    ) -> np.ndarray:
        """Map an output-stride mask (padded/scaled frame) to original resolution.

        Inverts the preprocessing chain (output_stride upsample -> crop the
        bottom-right stride padding -> undo size-match/input scale). For the
        common ``scale=1`` + stride-divisible-input case this collapses to a
        single ``output_stride`` upsample.
        """
        proc_h, proc_w = info.processed_size
        orig_h, orig_w = info.original_size
        if proc_h == 0 or proc_w == 0:
            # No metadata; fall back to an output_stride upsample.
            proc_h = mask.shape[0] * info.output_stride
            proc_w = mask.shape[1] * info.output_stride
        if orig_h == 0 or orig_w == 0:
            orig_h, orig_w = proc_h, proc_w

        t = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
        # 1. Upsample output-stride mask to the processed (padded) resolution.
        if t.shape[-2:] != (proc_h, proc_w):
            t = F.interpolate(t, size=(proc_h, proc_w), mode="nearest")

        # 2. Crop the bottom-right pad: the valid (scaled) image is top-left.
        eff = 1.0
        if info.eff_scale is not None and info.eff_scale.numel() > b:
            eff = float(info.eff_scale[b])
        scale = eff * float(info.input_scale)
        scaled_h = min(proc_h, max(1, int(round(orig_h * scale))))
        scaled_w = min(proc_w, max(1, int(round(orig_w * scale))))
        t = t[:, :, :scaled_h, :scaled_w]

        # 3. Undo the scale: resize the valid region back to original size.
        if (scaled_h, scaled_w) != (orig_h, orig_w):
            t = F.interpolate(t, size=(orig_h, orig_w), mode="nearest")

        return t[0, 0].numpy() > 0.5
