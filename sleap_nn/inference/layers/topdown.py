"""``TopDownLayer`` — composes ``CentroidLayer`` + ``CenteredInstanceLayer``.

Two-stage layer that detects instances by centroid, crops around each
centroid, runs a centered-instance model on the crops, and lifts the
crop-local keypoints back into image space via :func:`add_crop_offset`.

Stage layout (from `12-design-review-and-revised-plan.md` §4.6):

* **Stage A** — :class:`CentroidLayer` decides which centroids survive
  (peak threshold + max_instances cap).
* **Stage B** — this class optionally NaN-prunes centroid slots before
  stage 2 (avoids running the model on garbage crops) and optionally
  applies bbox-IoU NMS on close-together centroids (``centroid_nms``).
* **Stage C** — :class:`CenteredInstanceLayer` predicts keypoints per
  crop. Crop-local peaks are then added to the bbox top-left to land in
  full-image coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import attrs
import torch

from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.layers.base import ImageInput, InferenceLayer
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.ops.coord import add_crop_offset
from sleap_nn.inference.ops.crops import crop_bboxes
from sleap_nn.inference.outputs import Outputs


class TopDownLayer:
    """Composed centroid + centered-instance two-stage inference layer.

    Args:
        centroid_layer: Pre-built :class:`CentroidLayer`.
        centered_instance_layer: Pre-built :class:`CenteredInstanceLayer`.
        crop_size: ``(crop_h, crop_w)`` of the per-instance crop. Must
            match the centered_instance model's training crop size.
        centroid_nms: When ``True``, deduplicate overlapping centroids by
            bbox IoU before running stage 2. Useful for close-together
            animals where the centroid model emits two centroids per
            animal.
        centroid_nms_threshold: bbox-IoU threshold for the centroid NMS.

    Notes:
        Not an :class:`InferenceLayer` subclass — composes two layers
        rather than wrapping a single backend. Same ``predict()`` /
        ``__call__`` surface; no ``preprocess`` / ``postprocess`` because
        the inner layers own those.
    """

    def __init__(
        self,
        centroid_layer: CentroidLayer,
        centered_instance_layer: CenteredInstanceLayer,
        crop_size: Tuple[int, int],
        centroid_nms: bool = False,
        centroid_nms_threshold: float = 0.5,
    ) -> None:
        """Stash the inner layers and crop knobs."""
        self.centroid_layer = centroid_layer
        self.centered_instance_layer = centered_instance_layer
        self.crop_size = crop_size
        self.centroid_nms = centroid_nms
        self.centroid_nms_threshold = centroid_nms_threshold

    def predict(
        self,
        image: ImageInput,
        instances: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Run stage 1 → stage B → stage 2 → coord lift.

        Args:
            image: Full-image input. Same shape contract as
                :class:`InferenceLayer._to_4d_float_tensor`.
            instances: Optional GT instances. Required when either inner
                layer has its corresponding ``use_gt_*`` flag set.

        Returns:
            ``Outputs`` populated with ``pred_keypoints`` (image-space),
            ``pred_peak_values``, ``pred_centroids``,
            ``pred_centroid_values``, and ``instance_bboxes``.
        """
        # Stage 1: centroids.
        centroid_out = (
            self.centroid_layer.predict(image, instances=instances)
            if self.centroid_layer.use_gt_centroids
            else self.centroid_layer.predict(image)
        )
        centroids = centroid_out.pred_centroids
        centroid_vals = centroid_out.pred_centroid_values
        if centroids is None:
            return Outputs()

        B, max_inst, _ = centroids.shape

        # Stage B.1: drop NaN centroids (no model time on garbage crops).
        valid_mask = ~torch.isnan(centroids).any(dim=-1)  # (B, max_inst)

        # Stage B.2: optional centroid-NMS.
        if self.centroid_nms:
            valid_mask = valid_mask & self._centroid_nms_mask(
                centroids, centroid_vals, valid_mask
            )

        # Branch: GT-keypoints path doesn't need a real centered_instance
        # forward — it just matches centroids to GT and emits matched kpts.
        if self.centered_instance_layer.use_gt_peaks:
            if instances is None:
                raise ValueError(
                    "TopDownLayer with use_gt_peaks=True requires `instances`."
                )
            return self.centered_instance_layer.predict(
                crops=None, centroids=centroids, instances=instances
            )

        # Stage 2: crop + run centered-instance model + un-crop.
        # Preserve dtype (uint8 → uint8) so crops + the centered_instance
        # model input match legacy bit-for-bit; normalize_on_gpu inside
        # the Lightning forward handles uint8 → float32 conversion.
        x = self.centroid_layer._to_4d_tensor(image)
        return self._run_stage_2(x, centroids, centroid_vals, valid_mask)

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: crop extraction + centered-instance forward + un-crop
    # ──────────────────────────────────────────────────────────────────

    def _run_stage_2(
        self,
        image_4d: torch.Tensor,
        centroids: torch.Tensor,
        centroid_vals: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Outputs:
        """Crop around valid centroids, run model, lift back to image space."""
        B, max_inst, _ = centroids.shape
        crop_h, crop_w = self.crop_size

        # Flatten valid (b, i) pairs into per-crop indices.
        valid_idx = valid_mask.nonzero(as_tuple=False)  # (n_valid, 2) — (b, i)
        n_valid = valid_idx.shape[0]

        if n_valid == 0:
            # Nothing to crop. Return all-NaN keypoints with the right shape.
            n_nodes = self._infer_n_nodes()
            return Outputs(
                pred_keypoints=torch.full(
                    (B, max_inst, n_nodes, 2), float("nan"), device=centroids.device
                ),
                pred_peak_values=torch.full(
                    (B, max_inst, n_nodes), float("nan"), device=centroids.device
                ),
                pred_centroids=centroids,
                pred_centroid_values=centroid_vals,
            )

        # Per-crop centroid coords (n_valid, 2)
        valid_centroids = centroids[valid_idx[:, 0], valid_idx[:, 1]]
        sample_inds = valid_idx[:, 0]  # (n_valid,)

        # Build bboxes (n_valid, 4, 2) and crop the source image.
        bboxes = make_centered_bboxes(valid_centroids, crop_h, crop_w)
        crops = crop_bboxes(image_4d, bboxes, sample_inds)

        # Run centered-instance model on the crops.
        stage2_out = self.centered_instance_layer.predict(crops)
        # ``stage2_out.pred_keypoints`` shape: ``(n_valid, 1, n_nodes, 2)``;
        # squeeze the ``I=1`` instance dim so ``add_crop_offset`` (which is
        # written for ``(N, n_nodes, 2)``) broadcasts cleanly.
        stage2_kpts_3d = stage2_out.pred_keypoints.squeeze(1)  # (n_valid, n_nodes, 2)
        crop_topleft = bboxes[:, 0, :]  # (n_valid, 2)
        stage2_kpts_img = add_crop_offset(stage2_kpts_3d, crop_topleft)

        # Scatter (n_valid, ...) back into (B, max_inst, ...). Invalid slots
        # stay NaN (the canonical "no peak" sentinel). Allocate on the model's
        # device so the scatter from device-resident stage-2 tensors doesn't
        # raise on non-CPU runtimes (cuda / mps).
        device = stage2_kpts_img.device
        n_nodes = stage2_kpts_img.shape[-2]
        full_kpts = torch.full((B, max_inst, n_nodes, 2), float("nan"), device=device)
        full_vals = torch.full((B, max_inst, n_nodes), float("nan"), device=device)
        full_kpts[valid_idx[:, 0], valid_idx[:, 1]] = stage2_kpts_img
        full_vals[valid_idx[:, 0], valid_idx[:, 1]] = (
            stage2_out.pred_peak_values.squeeze(1)
        )

        # Reshape bboxes back to (B, max_inst, 4, 2) for downstream debug.
        full_bboxes = torch.full((B, max_inst, 4, 2), float("nan"), device=device)
        full_bboxes[valid_idx[:, 0], valid_idx[:, 1]] = bboxes

        return Outputs(
            pred_keypoints=full_kpts,
            pred_peak_values=full_vals,
            pred_centroids=centroids,
            pred_centroid_values=centroid_vals,
            instance_bboxes=full_bboxes,
        )

    # ──────────────────────────────────────────────────────────────────
    # Stage B: optional centroid NMS
    # ──────────────────────────────────────────────────────────────────

    def _centroid_nms_mask(
        self,
        centroids: torch.Tensor,
        centroid_vals: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy bbox-IoU NMS on per-batch centroids, returns a kept-mask."""
        B, max_inst, _ = centroids.shape
        crop_h, crop_w = self.crop_size
        keep = torch.ones_like(valid_mask)

        for b in range(B):
            valid_b = valid_mask[b].nonzero(as_tuple=False).flatten()
            if valid_b.numel() <= 1:
                continue
            ctr_b = centroids[b, valid_b]  # (k, 2)
            val_b = centroid_vals[b, valid_b]  # (k,)
            order = val_b.argsort(descending=True)
            ordered = valid_b[order]
            ordered_ctr = ctr_b[order]
            kept_local: list[int] = []
            kept_ctr: list[torch.Tensor] = []
            for i, ci in enumerate(ordered_ctr):
                if any(
                    self._bbox_iou(ci, kc, crop_h, crop_w) > self.centroid_nms_threshold
                    for kc in kept_ctr
                ):
                    keep[b, ordered[i]] = False
                    continue
                kept_ctr.append(ci)
                kept_local.append(int(ordered[i].item()))
        return keep

    @staticmethod
    def _bbox_iou(c1: torch.Tensor, c2: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """IoU between two centered bboxes of shape ``(h, w)``."""
        half_h, half_w = h / 2.0, w / 2.0
        a_y1, a_x1 = c1[1] - half_h, c1[0] - half_w
        a_y2, a_x2 = c1[1] + half_h, c1[0] + half_w
        b_y1, b_x1 = c2[1] - half_h, c2[0] - half_w
        b_y2, b_x2 = c2[1] + half_h, c2[0] + half_w
        inter_h = (torch.minimum(a_y2, b_y2) - torch.maximum(a_y1, b_y1)).clamp(min=0)
        inter_w = (torch.minimum(a_x2, b_x2) - torch.maximum(a_x1, b_x1)).clamp(min=0)
        inter = inter_h * inter_w
        union = 2.0 * (h * w) - inter
        return inter / union

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _infer_n_nodes(self) -> int:
        """Best-effort guess of the skeleton node count from the inner model.

        Falls back to 1 so any "no detections" early-return still has a
        valid 4D shape.
        """
        try:
            for m in self.centered_instance_layer.backend.model.modules():
                if hasattr(m, "out_channels") and m.out_channels > 1:
                    return int(m.out_channels)
        except Exception:
            pass
        return 1

    def __call__(
        self,
        image: ImageInput,
        instances: Optional[torch.Tensor] = None,
    ) -> Outputs:
        """Alias for :meth:`predict`."""
        return self.predict(image, instances=instances)
