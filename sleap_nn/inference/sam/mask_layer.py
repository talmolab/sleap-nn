"""SAM mask inference layer — the producer that emits ``PredictedSegmentationMask``.

:class:`SamSegmentationLayer` is the full-frame producer. Given a backend, a
prompt mode, and the per-frame poses/centroids, it encodes each frame once,
builds one :class:`~sleap_nn.inference.sam.prompts.SamPrompt` per instance,
asks the backend for masks, and emits ``Outputs.pred_masks`` dicts at the
correct full-frame offset/scale with ``instance=``/``track=`` populated
(PLAN L8). Output collection / SLP packaging is **free** — it reuses
``Outputs.to_masks`` -> ``build_predicted_segmentation_mask`` ->
``labels.save`` exactly like every other seg layer (PLAN §2.5).

It is torch-light: it shells out to a :class:`MaskBackend` (SAM1 here, SAM3
later). The heavy SAM import lives in the backend.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.sam.backends import MaskBackend
from sleap_nn.inference.sam.prompts import (
    SamPrompt,
    prompt_for_instance,
    visible_keypoints,
)


def _frame_gray(image) -> np.ndarray:
    """Coerce a frame/crop to a 2-D ``(H, W)`` uint8 grayscale array.

    Accepts ``(H, W)``, ``(H, W, C)``, or ``(C, H, W)`` numpy/torch input and
    returns the first channel as ``uint8``.

    Args:
        image: A frame or crop in any of the accepted layouts.

    Returns:
        ``(H, W)`` uint8 array.
    """
    arr = image
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # Channels-first (C, H, W) with a small leading dim, else (H, W, C).
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class SamSegmentationLayer:
    """Full-frame SAM mask producer (pose / centroid / box prompts).

    Operates on in-memory ``sio.LabeledFrame`` content (image + pose/centroid
    instances), not on a torch model — there is no trained net here. For each
    frame it encodes the image once via the backend, builds one prompt per
    instance, and emits per-frame ``Outputs.pred_masks`` dicts that the standard
    ``Outputs.to_masks`` path packages into ``sio.PredictedSegmentationMask``.
    Full-frame masks use identity ``scale``/``offset`` (the whole-frame
    representation the P1 prototype produced).

    Args:
        backend: A :class:`MaskBackend` (SAM1 here; SAM3 later).
        prompt_mode: One of ``"pose"`` / ``"centroid"`` / ``"box"``. ``"pose"``
            applies the L3 product rule (pose-if-visible-else-centroid-point).
        anchor_ind: Optional skeleton node index used as the centroid anchor for
            ``prompt_mode="centroid"``; ``None`` uses the mean of visible
            keypoints.
        disjointify_masks: When ``True`` and a frame has >=2 instances, make the
            per-frame masks disjoint via keypoint-Voronoi (harvested #642).
            Default ``False`` (single-instance is the common case; disjointify is
            a multi-instance refinement).
    """

    def __init__(
        self,
        backend: MaskBackend,
        prompt_mode: str = "pose",
        anchor_ind: Optional[int] = None,
        disjointify_masks: bool = False,
    ) -> None:
        """Stash the backend and prompt knobs."""
        if prompt_mode not in ("pose", "centroid", "box"):
            raise ValueError(
                f"SamSegmentationLayer prompt_mode must be 'pose'/'centroid'/'box', "
                f"got {prompt_mode!r}."
            )
        self.backend = backend
        self.prompt_mode = prompt_mode
        self.anchor_ind = anchor_ind
        self.disjointify_masks = bool(disjointify_masks)

    def _instance_centroid(self, kpts_vis: np.ndarray, inst) -> Optional[np.ndarray]:
        """Anchor point for an instance: anchor node if set/visible, else mean."""
        if self.anchor_ind is not None:
            pts = np.asarray(inst.numpy()[:, :2], dtype=np.float32)
            if 0 <= self.anchor_ind < len(pts):
                a = pts[self.anchor_ind]
                if np.isfinite(a).all():
                    return a.astype(np.float32)
        if len(kpts_vis) > 0:
            return kpts_vis.mean(0).astype(np.float32)
        return None

    def masks_for_frame(self, image, instances: Sequence) -> List[dict]:
        """Produce one ``pred_masks`` dict per posed instance for a frame.

        Args:
            image: The frame image (``(H, W)`` / ``(H, W, C)`` / ``(C, H, W)``).
            instances: The frame's ``sio.PredictedInstance`` (or ``sio.Instance``)
                pose/centroid instances. Instances with no visible keypoints (and
                no usable centroid) are skipped.

        Returns:
            A list of ``pred_masks`` dicts ``{"mask", "score", "scale",
            "offset", "instance", "track", "tracking_score"}`` — full-frame masks
            with identity scale/offset and ``instance``/``track`` populated when
            the source instance carries them (PLAN L8).
        """
        gray = _frame_gray(image)
        h, w = gray.shape
        prompts: List[SamPrompt] = []
        kept = []  # (instance, kpts_vis)
        for inst in instances:
            kpts = np.asarray(inst.numpy()[:, :2], dtype=np.float32)
            kpts_vis = visible_keypoints(kpts)
            centroid = self._instance_centroid(kpts_vis, inst)
            try:
                prompt = prompt_for_instance(
                    self.prompt_mode,
                    (h, w),
                    keypoints=kpts_vis if len(kpts_vis) else None,
                    centroid=centroid,
                )
            except ValueError:
                # No usable prompt source for this instance — skip it.
                continue
            prompts.append(prompt)
            kept.append((inst, kpts_vis))

        if not prompts:
            return []

        masks, scores = self.backend.masks(gray, prompts)

        if self.disjointify_masks and len(masks) >= 2:
            from sleap_nn.inference.sam.backends import disjointify

            masks = disjointify(masks, [kv[1] for kv in kept])

        out: List[dict] = []
        for (inst, _kpts), mask, score in zip(kept, masks, scores):
            if mask is None or not mask.any():
                continue
            out.append(
                {
                    "mask": np.ascontiguousarray(mask, dtype=bool),
                    "score": float(score),
                    "scale": (1.0, 1.0),
                    "offset": (0.0, 0.0),
                    "instance": inst if _is_predicted(inst) else None,
                    "track": getattr(inst, "track", None),
                    "tracking_score": _tracking_score(inst),
                }
            )
        return out

    def predict_labels(self, labels) -> "List[List[dict]]":
        """Build ``pred_masks`` for every labeled frame of a ``sio.Labels``.

        Args:
            labels: The source ``sio.Labels`` with pose/centroid instances + image
                data (used as the prompt source).

        Returns:
            A list (one entry per labeled frame) of the frame's ``pred_masks``
            dicts; frames are index-aligned to ``labels.labeled_frames``.
        """
        return [
            self.masks_for_frame(lf.image, lf.instances) for lf in labels.labeled_frames
        ]


def _is_predicted(inst) -> bool:
    """``True`` iff ``inst`` is a ``sio.PredictedInstance`` (vs GT ``Instance``).

    Only predicted instances are set on ``mask.instance`` — a GT instance is a
    user annotation, not a prediction, and pairing a predicted mask to it would
    be misleading provenance.
    """
    import sleap_io as sio

    return isinstance(inst, sio.PredictedInstance)


def _tracking_score(inst) -> Optional[float]:
    """Best-effort tracking score off an instance (``None`` when absent)."""
    ts = getattr(inst, "tracking_score", None)
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None
