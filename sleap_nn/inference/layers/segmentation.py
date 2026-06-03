"""``SegmentationLayer`` — bottom-up instance segmentation inference.

Wraps a trained ``BottomUpSegmentationLightningModule`` (whose ``forward``
returns the three head maps, with sigmoid already applied to the foreground
head). ``postprocess`` groups foreground pixels into instances via the
predicted instance-center offsets and packages them into ``Outputs.pred_masks``.
By default each mask is kept at the model output-stride resolution with a sio
``scale``/``offset`` carrying the mapping back to image pixels (the #618
~stride^2 RLE win, lossless at model resolution); ``full_res_masks`` restores
the legacy original-resolution upsample.
"""

from __future__ import annotations

import math
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
        max_instances: Optional cap on instances per frame. When more centers
            are detected, only the highest-scoring ``max_instances`` are kept
            before grouping. ``None`` keeps all detected centers. Overridable
            via ``Predictor.predict(max_instances=...)``.
        center_nms_kernel: Odd window size for center-peak NMS. Larger values
            merge nearby duplicate centers (a lever against over-segmentation).
            Default ``3`` (no behavior change).
        mask_cleanup: When ``True``, keep only each mask's largest connected
            component and fill interior holes (suppresses speckle/fragments).
            Default ``False``.
        mask_cleanup_radius: Morphological open->close kernel radius (in
            output-stride pixels) applied during ``mask_cleanup`` before
            keep-largest-CC. ``0`` (default) keeps the keep-largest+fill behavior.
        full_res_masks: When ``True``, encode masks at full ORIGINAL resolution
            (legacy behavior) instead of the model output-stride grid. Default
            ``False``: output-stride encoding is ~stride^2 smaller and lossless
            at model resolution, carrying the image mapping via sio scale/offset.
        mask_output: Output representation — ``"mask"`` (RLE mask, default),
            ``"polygon"`` (Douglas-Peucker ``sio.PredictedROI`` only), or
            ``"both"`` (exact mask + simplified ROI). Consumed at packaging time
            (``Outputs.to_labels``); stored on the layer for the Predictor to read.
        polygon_epsilon: Douglas-Peucker tolerance (fraction of perimeter) for
            ``mask_output`` polygon/both. Default ``0.01``.
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
        max_instances: Optional[int] = None,
        center_nms_kernel: int = 3,
        mask_cleanup: bool = False,
        mask_cleanup_radius: int = 0,
        full_res_masks: bool = False,
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
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
        self.max_instances = max_instances
        self.center_nms_kernel = int(center_nms_kernel)
        self.mask_cleanup = bool(mask_cleanup)
        self.mask_cleanup_radius = int(mask_cleanup_radius)
        self.full_res_masks = bool(full_res_masks)
        self.mask_output = str(mask_output)
        self.polygon_epsilon = float(polygon_epsilon)

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
            ``Outputs`` with ``pred_masks`` populated: a per-batch list of
            per-instance ``{"mask", "score", "scale", "offset"}`` dicts. By
            default ``mask`` is at output-stride resolution and ``scale``/
            ``offset`` map it to image pixels (``image_coord = mask_coord /
            scale + offset``); with ``full_res_masks`` it is at original
            resolution with identity scale/offset.
        """
        foreground = raw_out[self._SEG_KEY]  # (B, 1, h, w), already sigmoid
        center_heatmap = raw_out[self._CENTER_KEY]  # (B, 1, h, w)
        offsets = raw_out[self._OFFSET_KEY]  # (B, 2, h, w)

        foreground = foreground.detach().cpu()
        center_heatmap = center_heatmap.detach().cpu()
        offsets = offsets.detach().cpu()

        peak_threshold = self.postprocess_config.peak_threshold
        # The predict-time override sets ``postprocess_config.max_instances``;
        # fall back to the value the layer was built with (#582 pattern, mirrors
        # BottomUpLayer).
        max_instances = getattr(self.postprocess_config, "max_instances", None)
        if max_instances is None:
            max_instances = getattr(self, "max_instances", None)
        # New knobs are read via ``getattr`` because some tests build the layer
        # through ``__new__`` and set only a subset of attributes.
        full_res_masks = getattr(self, "full_res_masks", False)
        mask_cleanup_radius = getattr(self, "mask_cleanup_radius", 0)
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
                max_instances=max_instances,
                center_nms_kernel=getattr(self, "center_nms_kernel", 3),
                mask_cleanup=getattr(self, "mask_cleanup", False),
                mask_cleanup_radius=mask_cleanup_radius,
            )
            frame_masks: List[dict] = []
            # ``min_mask_area`` is an ORIGINAL-image-pixel floor. ``max(1, ...)``
            # keeps the empty-mask drop when the filter is off.
            area_floor = max(1, self.min_mask_area)
            for inst in instances:
                if full_res_masks:
                    # Legacy path: materialize the original-resolution mask and
                    # filter directly in original pixels (scale/offset identity).
                    mask_out = self._mask_to_original(inst["mask"], info, b)
                    scale, offset = (1.0, 1.0), (0.0, 0.0)
                    if int(mask_out.sum()) < area_floor:
                        continue
                else:
                    # Default: keep the mask at output-stride resolution, carrying
                    # the image mapping via sio scale/offset (the ~stride^2 RLE
                    # win). Convert the original-pixel area floor to stride units
                    # (exact when 1/(sx*sy) is integral, i.e. eff*input_scale==1).
                    mask_out, scale, offset = self._mask_to_stride(
                        inst["mask"], info, b
                    )
                    sx, sy = scale
                    # ceil (not round): the stride floor is the smallest integer
                    # pixel count whose original-pixel area (count / (sx*sy)) still
                    # meets the floor. round() would round the scaled floor DOWN
                    # and over-keep sub-floor masks, breaking the integer-scale
                    # parity with --full_res_masks.
                    area_floor_stride = max(1, math.ceil(area_floor * sx * sy))
                    if int(mask_out.sum()) < area_floor_stride:
                        continue
                frame_masks.append(
                    {
                        "mask": mask_out,
                        "score": float(inst["score"]),
                        "scale": scale,
                        "offset": offset,
                    }
                )
            pred_masks.append(frame_masks)

        return Outputs(pred_masks=pred_masks, preprocess_info=info)

    def _mask_to_stride(self, mask: np.ndarray, info: PreprocInfo, b: int) -> tuple:
        """Crop an output-stride mask to its valid (non-pad) extent + scale/offset.

        Unlike :meth:`_mask_to_original`, this does NOT upsample. The head map
        covers the bottom-right-padded extent, so the valid region is the
        top-left block at stride resolution.

        Two subtleties (both verified against the legacy ``_mask_to_original``
        crop so the default and ``--full_res_masks`` paths keep identical edge
        content):

        * The valid extent is ``ceil(round(orig * s) / stride)`` stride cells,
          NOT ``round(orig * s / stride)``: the trailing partial stride cell
          covers real (non-pad) processed pixels and ``round`` would drop it,
          truncating masks whose object touches the right/bottom edge.
        * The stored sio scale is ``valid / orig`` (the best inverse of the
          crop), NOT the raw ``s / stride`` ratio. ``sio.image_extent`` recovers
          the image size as ``int(valid / scale)``; ``valid / orig`` recovers
          ``orig`` to within +/-1 px (the float division can floor-truncate to
          ``orig - 1`` for ~5% of sizes), whereas the raw ratio truncates by up
          to a full stride cell at large strides. ``image_extent`` is therefore
          still NOT authoritative for the true frame size (see
          :func:`sleap_nn.inference.segmentation_convert.decode_mask_to_image_res`);
          consumers clamp the +/-1.

        ``offset`` is ``(0, 0)`` because every preprocessing pad is bottom-right
        (valid content top-left aligned). Scales are isotropic today
        (eff_scale / input_scale / output_stride are scalars) but stored
        per-axis so each dim inverts its own crop.

        Returns:
            ``(mask_stride_bool, (sx, sy), (ox, oy))``.
        """
        orig_h, orig_w = info.original_size
        stride = float(info.output_stride)
        if orig_h == 0 or orig_w == 0:
            # No metadata: the head map IS the valid extent; map by output_stride.
            sx = sy = 1.0 / stride
            return np.ascontiguousarray(mask, dtype=bool), (sx, sy), (0.0, 0.0)

        eff = 1.0
        if info.eff_scale is not None and info.eff_scale.numel() > b:
            eff = float(info.eff_scale[b])
        s = eff * float(info.input_scale)
        # Valid (non-pad) processed extent in input pixels, then the number of
        # output-stride cells that overlap it (ceil — keeps the trailing partial
        # cell, matching _mask_to_original).
        scaled_h = max(1, int(round(orig_h * s)))
        scaled_w = max(1, int(round(orig_w * s)))
        valid_h = min(mask.shape[0], max(1, math.ceil(scaled_h / stride)))
        valid_w = min(mask.shape[1], max(1, math.ceil(scaled_w / stride)))
        mask_stride = np.ascontiguousarray(mask[:valid_h, :valid_w], dtype=bool)
        # Store the scale that best inverts this crop: image_extent ->
        # int(valid / (valid / orig)) recovers orig within +/-1 px (float
        # truncation can give orig - 1; consumers clamp).
        sx = valid_w / float(orig_w)
        sy = valid_h / float(orig_h)
        return mask_stride, (sx, sy), (0.0, 0.0)

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
