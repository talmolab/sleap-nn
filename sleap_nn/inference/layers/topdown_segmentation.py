"""Top-down (crop-centered) instance-segmentation inference layers (#622).

Two pieces, mirroring the keypoint top-down stack:

* :class:`CenteredInstanceMaskLayer` — the stage-2 analog of
  :class:`~sleap_nn.inference.layers.centered_instance.CenteredInstanceLayer`.
  Runs the trained ``centered_instance_segmentation`` model on per-instance
  crops and returns one boolean foreground mask per crop (at the head's
  output-stride resolution) instead of keypoints. The crop-resolution masks +
  per-crop scores are carried on ``Outputs`` for the composed layer to place.

* :class:`TopDownSegmentationLayer` — subclasses
  :class:`~sleap_nn.inference.layers.topdown.TopDownLayer` to reuse its
  stage-1 (centroid) + sizematch + crop machinery verbatim, overriding only
  the stage-2 emission. Each crop mask is emitted into ``Outputs.pred_masks``
  with the DQ5 offset/scale contract (see :meth:`_run_stage_2`), so
  ``decode_mask_to_image_res`` upsamples the crop mask to the crop's
  image-space size and (offset-aware) top-left pads it to land at the correct
  full-frame location.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.layers.backends.base import ModelBackend
from sleap_nn.inference.layers.base import InferenceLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.ops.crops import crop_bboxes
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.preprocess_info import PreprocInfo


class CenteredInstanceMaskLayer(InferenceLayer):
    """Per-crop foreground-mask prediction layer (top-down stage 2).

    Runs a trained ``centered_instance_segmentation`` model on per-instance
    crops and decodes the ``SegmentationHead`` logits into one boolean
    foreground mask per crop. The structural twin of
    :class:`CenteredInstanceLayer`, but returns masks (on ``Outputs.crops`` as
    a stacked tensor + a per-crop score) rather than keypoints — the composed
    :class:`TopDownSegmentationLayer` places each crop mask back into the frame.

    Args:
        backend: Runtime backend wrapping the seg Lightning module. Its
            ``forward`` returns the ``SegmentationHead`` LOGITS (a single
            tensor, wrapped as ``{"output": ...}`` by ``TorchBackend``).
        output_stride: Head map → crop-pixel stride (default 2).
        max_stride: Backbone max stride; crops are padded to a multiple of it.
        fg_threshold: Foreground probability threshold for binarization
            (sigmoid(logits) > fg_threshold). Default 0.5.
        preprocess_config / postprocess_config: Standard knobs. The crops are
            already sized (extracted from the sizematched image), so only the
            model's own ``input_scale`` is applied here.
    """

    _HEAD_OUTPUT_KEY: str = "SegmentationHead"

    def __init__(
        self,
        backend: ModelBackend,
        output_stride: int,
        max_stride: int = 1,
        fg_threshold: float = 0.5,
        preprocess_config: Optional[PreprocessConfig] = None,
        postprocess_config: Optional[PostprocessConfig] = None,
    ) -> None:
        """Compose the layer with default configs when omitted."""
        super().__init__(
            backend=backend,
            preprocess_config=preprocess_config or PreprocessConfig(),
            postprocess_config=postprocess_config or PostprocessConfig(),
            output_stride=output_stride,
            max_stride=max_stride,
        )
        self.fg_threshold = float(fg_threshold)
        # The parent TopDownLayer.predict inspects this on the stage-2 layer to
        # decide the GT-keypoint branch; a mask layer never takes that branch.
        self.use_gt_peaks = False

    @property
    def warmup_input_shape(self):
        """Tiny single-channel warmup shape."""
        return (1, 1, 64, 64)

    def postprocess(self, raw_out: dict, info: PreprocInfo) -> Outputs:
        """Decode ``SegmentationHead`` logits → one bool mask per crop.

        Args:
            raw_out: Backend output dict carrying the seg-head logits.
            info: Preprocessing metadata (unused for placement — the composed
                layer owns the crop→frame mapping).

        Returns:
            ``Outputs`` with ``crops`` set to the stacked per-crop boolean
            masks ``(N, 1, h, w)`` (float) at output-stride resolution and
            ``instance_scores`` set to the per-crop mean foreground
            probability ``(N, 1)``. The composed layer reads both.
        """
        logits = self._extract_confmaps(raw_out)  # (N, 1, h, w)
        probs = torch.sigmoid(logits.detach())
        masks = (probs > self.fg_threshold).float()  # (N, 1, h, w)
        # Per-crop score: mean foreground probability over the predicted mask.
        # An empty crop scores 0.0 (and is dropped by ``Outputs.to_masks``). Kept
        # on ``instance_scores`` in the (N, 1) layout the composed layer expects.
        denom = masks.sum(dim=(2, 3)).clamp(min=1.0)
        score = (probs * masks).sum(dim=(2, 3)) / denom  # (N, 1)
        return Outputs(crops=masks, instance_scores=score, preprocess_info=info)


class TopDownSegmentationLayer(TopDownLayer):
    """Composed centroid + per-crop-mask two-stage segmentation layer.

    Subclasses :class:`TopDownLayer` to reuse stage 1 (centroid) + sizematch +
    crop extraction verbatim, overriding only :meth:`_run_stage_2` to emit
    per-crop masks into ``Outputs.pred_masks`` instead of keypoints.

    Args:
        centroid_layer: Stage-1 :class:`CentroidLayer` (real model or
            ``use_gt_centroids=True`` for the GT-centroid fallback).
        centered_instance_layer: Stage-2 :class:`CenteredInstanceMaskLayer`.
        crop_size: ``(crop_h, crop_w)`` of the per-instance crop.
        mask_output: Output representation forwarded to ``Outputs.to_labels`` —
            ``"mask"`` (default), ``"polygon"``, or ``"both"``.
        polygon_epsilon: Douglas-Peucker tolerance for polygon/both output.
        centroid_nms / centroid_nms_threshold: Optional centroid dedup before
            stage 2 (inherited).
    """

    def __init__(
        self,
        centroid_layer: CentroidLayer,
        centered_instance_layer: CenteredInstanceMaskLayer,
        crop_size: Tuple[int, int],
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
        centroid_nms: bool = False,
        centroid_nms_threshold: float = 0.5,
    ) -> None:
        """Stash inner layers, crop size, and mask packaging knobs."""
        super().__init__(
            centroid_layer=centroid_layer,
            centered_instance_layer=centered_instance_layer,
            crop_size=crop_size,
            centroid_nms=centroid_nms,
            centroid_nms_threshold=centroid_nms_threshold,
            return_crops=False,
        )
        # Read by Predictor.to_labels / predict_to_file via getattr.
        self.mask_output = str(mask_output)
        self.polygon_epsilon = float(polygon_epsilon)

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: crop → seg model → per-crop mask placement
    # ──────────────────────────────────────────────────────────────────

    def _run_stage_2(
        self,
        image_4d: torch.Tensor,
        centroids: torch.Tensor,
        centroid_vals: torch.Tensor,
        valid_mask: torch.Tensor,
        eff_scale: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Crop around valid centroids, run the seg model, place per-crop masks.

        ``image_4d`` / ``centroids`` are in **sized** space (after the centroid
        layer's sizematcher). For each valid crop we emit one ``pred_masks``
        entry with the DQ5 offset/scale contract::

            offset = (crop_topleft_x / eff, crop_topleft_y / eff)   # image px
            scale  = (eff / output_stride, eff / output_stride)
            score  = centroid confidence for that crop

        where ``eff`` is the per-crop sizematcher scale and ``crop_topleft`` is
        the bbox top-left in sized space (``bboxes[:, 0, :]``). With that scale,
        ``decode_mask_to_image_res`` upsamples the ``(crop/stride)`` mask to the
        crop's image-space size (``image_coord = mask_coord / scale + offset``)
        and then offset-aware top-left pads it, landing the crop at its
        full-frame location.
        """
        B, max_inst, _ = centroids.shape
        crop_h, crop_w = self.crop_size

        valid_idx = valid_mask.nonzero(as_tuple=False)  # (n_valid, 2) — (b, i)
        n_valid = valid_idx.shape[0]

        if eff_scale is None:
            eff_scale = torch.ones(B, dtype=torch.float32, device=centroids.device)
        else:
            eff_scale = eff_scale.to(centroids.device)

        # Empty batch: one empty mask list per frame. Masks ONLY (no
        # pred_centroids) so Outputs.to_labels does not synthesize phantom
        # keypoint instances — matches the bottom-up SegmentationLayer contract.
        if n_valid == 0:
            return Outputs(pred_masks=[[] for _ in range(B)])

        valid_centroids = centroids[valid_idx[:, 0], valid_idx[:, 1]]
        sample_inds = valid_idx[:, 0]  # (n_valid,)
        per_crop_eff_scale = eff_scale[sample_inds]  # (n_valid,)

        bboxes = make_centered_bboxes(valid_centroids, crop_h, crop_w)
        crops = crop_bboxes(image_4d, bboxes, sample_inds)

        # Run the seg model on the crops -> per-crop bool masks (N, 1, h, w).
        stage2_out = self.centered_instance_layer.predict(crops)
        crop_masks = stage2_out.crops  # (n_valid, 1, h, w) float {0,1}
        device = crop_masks.device

        # Normalize metadata onto the model device (GT-centroid path runs
        # centroids/bboxes on CPU while the model runs on cuda/mps).
        centroid_vals = centroid_vals.to(device)
        crop_centroid_vals = centroid_vals[
            valid_idx[:, 0].to(device), valid_idx[:, 1].to(device)
        ]  # (n_valid,)
        # Per-crop score: the inner layer's mean foreground probability (a
        # mask-quality signal for COCO-style AP ranking), falling back to the
        # centroid confidence when unavailable.
        if stage2_out.instance_scores is not None:
            crop_scores = stage2_out.instance_scores.squeeze(1).to(device)
        else:
            crop_scores = crop_centroid_vals
        per_crop_eff_scale = per_crop_eff_scale.to(device)

        # Use the SAME floored top-left that crop_bboxes extracted content from
        # (it floors `trunc(top_left + half) - half`), so the baked image offset
        # lands exactly where the crop pixels came from (removes the up-to-1px
        # shift from using the raw float bbox top-left).
        bboxes = bboxes.to(device)
        half_xy = torch.tensor(
            [crop_w // 2, crop_h // 2], dtype=bboxes.dtype, device=device
        )
        crop_topleft = (bboxes[:, 0, :] + half_xy).to(torch.long) - half_xy.long()

        crop_masks_np = crop_masks.squeeze(1).bool().cpu().numpy()  # (n_valid, h, w)
        crop_topleft_np = crop_topleft.cpu().numpy()  # (n_valid, 2)
        eff_np = per_crop_eff_scale.cpu().numpy()  # (n_valid,)
        scores_np = crop_scores.cpu().numpy()  # (n_valid,)
        samples_np = sample_inds.cpu().numpy()  # (n_valid,)
        stride = float(self.centered_instance_layer.output_stride)
        # The crops are downscaled by the seg model's own input_scale before the
        # head (InferenceLayer.preprocess), so the head mask is at
        # crop*input_scale/stride px. Fold input_scale into the emitted scale to
        # match the bottom-up SegmentationLayer convention (s = eff*input_scale).
        input_scale = float(
            getattr(
                getattr(self.centered_instance_layer, "preprocess_config", None),
                "scale",
                1.0,
            )
        )

        pred_masks: List[List[dict]] = [[] for _ in range(B)]
        for k in range(n_valid):
            b = int(samples_np[k])
            eff = float(eff_np[k])
            # Image-space crop origin: floored sized top-left / eff_scale.
            ox = float(crop_topleft_np[k, 0]) / eff
            oy = float(crop_topleft_np[k, 1]) / eff
            # image_coord = mask_coord / scale + offset, with the crop mask at
            # (crop*input_scale/stride) px over a crop/eff image-px span, so
            # scale (= mask_px / image_px) = (eff * input_scale) / stride.
            sx = (eff * input_scale) / stride
            sy = (eff * input_scale) / stride
            pred_masks[b].append(
                {
                    "mask": np.ascontiguousarray(crop_masks_np[k], dtype=bool),
                    "score": float(scores_np[k]),
                    "scale": (sx, sy),
                    "offset": (ox, oy),
                }
            )

        # Masks ONLY (no pred_centroids/pred_keypoints) so Outputs.to_labels does
        # not synthesize phantom keypoint instances next to the masks (which also
        # broke mask-tracking auto-detection). Mirrors bottom-up SegmentationLayer.
        return Outputs(pred_masks=pred_masks)
