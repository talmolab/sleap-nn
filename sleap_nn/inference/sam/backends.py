"""SAM mask backends for prompted instance segmentation (PR-A).

A *backend* owns the model-specific half of mask production: loading the model,
preprocessing an image (grayscale -> CLAHE -> 3-channel), encoding it once, and
turning a list of :class:`~sleap_nn.inference.sam.prompts.SamPrompt` into one
boolean mask + a raw per-model score per prompt.

PR-A ships :class:`SamBackend` (SAM1, ViT-H, Apache-2.0, ungated; the
``sleap_nn[sam]`` extra). The model-specific recipe constants
(:data:`SamBackend.pred_iou_min`, the candidate-rejection factor, the keypoint
box margins, CLAHE) and the candidate-selection / score helpers
(:func:`_pick`, :func:`own_containment`, :func:`disjointify`) are harvested from
the closed #642 (``sleap_nn/data/pseudomasks.py``) and the exp-07 locked recipe,
repurposed to emit a *raw score* rather than to drive a drop-gate (PLAN §1).

Backend selection is **explicit / required** (PLAN L2): there is no default
``mask_backend``; the caller names ``"sam"`` (this backend) and a later PR adds
``"sam3"`` behind the same :class:`MaskBackend` interface. The SAM import is
lazy so the default sleap-nn install never needs ``segment-anything``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sleap_nn.inference.sam.prompts import SamPrompt

# ---------------------------------------------------------------------------
# Locked SAM1 recipe constants (harvested from #642 / exp-07; PLAN §1).
# ---------------------------------------------------------------------------
#: SAM1 candidate rejection: drop candidates whose area exceeds this * box-area
#: (kills SAM's over-confident whole-arena candidate).
SAM_MAX_BOX_AREA_FACTOR: float = 1.5
#: CLAHE parameters applied to the grayscale image before encoding (SAM1).
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
#: SAM1's nominal predicted-IoU floor. Carried as a per-model attribute so a
#: future SAM3 backend can override it (SAM3 uses a recalibrated 0.5, PLAN §2.3);
#: SAM1's raw predicted-IoU is reported as the mask score, not used as a gate.
SAM_PRED_IOU_MIN: float = 0.88


def _to_3ch_clahe(
    img_gray: np.ndarray,
    clahe: bool = True,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Grayscale ``(H, W)`` -> optional CLAHE -> 3-channel uint8 (SAM's input).

    Harvested from #642 ``_sam_instance_masks`` / the prototype ``to_3ch_clahe``.

    Args:
        img_gray: ``(H, W)`` uint8 grayscale frame (or crop).
        clahe: Whether to CLAHE-equalize before replicating to 3 channels.
        clahe_clip_limit: CLAHE clip limit (when ``clahe``).
        clahe_tile_grid: CLAHE tile grid (when ``clahe``).

    Returns:
        ``(H, W, 3)`` uint8 RGB array.
    """
    import cv2

    src = img_gray
    if src.ndim == 3:
        src = src[..., 0]
    src = np.ascontiguousarray(src).astype(np.uint8)
    if clahe:
        src = cv2.createCLAHE(clahe_clip_limit, clahe_tile_grid).apply(src)
    return np.stack([src] * 3, axis=-1).astype(np.uint8)


def _pick(
    masks: np.ndarray,
    scores: np.ndarray,
    box: np.ndarray,
    max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR,
) -> int:
    """Pick the best SAM candidate-mask index (harvested verbatim from #642).

    Rejects candidates whose area exceeds ``max_box_area_factor * box-area``
    (SAM's over-confident whole-arena candidate), then returns the highest
    predicted-IoU survivor; if all are rejected returns the smallest candidate.

    Args:
        masks: ``(n_cands, H, W)`` candidate masks.
        scores: ``(n_cands,)`` SAM predicted-IoU per candidate.
        box: ``[x0, y0, x1, y1]`` reject box.
        max_box_area_factor: Area-rejection multiplier relative to box area.

    Returns:
        Index of the chosen candidate.
    """
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    areas = masks.reshape(len(masks), -1).sum(1).astype(float)
    ok = areas <= max_box_area_factor * box_area
    if ok.any():
        idx = np.where(ok)[0]
        return int(idx[int(np.argmax(scores[idx]))])
    return int(np.argmin(areas))


def own_containment(mask: np.ndarray, kpts: np.ndarray, hw: Tuple[int, int]) -> float:
    """Fraction of an instance's visible keypoints that fall inside ``mask``.

    Harvested from #642 ``_own_containment``. In the inference stack this is a
    *score* (a mask-quality signal surfaced for review), never a drop-gate.

    Args:
        mask: ``(H, W)`` boolean mask.
        kpts: ``(n, 2)`` visible xy keypoints.
        hw: ``(height, width)`` of ``mask``.

    Returns:
        Containment in ``[0, 1]`` (``0.0`` for an empty keypoint set).
    """
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    if len(kpts) == 0:
        return 0.0
    h, w = hw
    inside = 0
    for x, y in kpts:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            inside += 1
    return inside / len(kpts)


def disjointify(
    masks: Sequence[np.ndarray], kpts: Sequence[np.ndarray]
) -> List[np.ndarray]:
    """Make per-instance masks disjoint via keypoint-Voronoi assignment.

    Harvested verbatim from #642 ``_disjointify`` (multi-instance only). Any
    pixel claimed by >=2 masks is assigned to the instance whose nearest visible
    keypoint is closest, so the result is exactly disjoint and each instance
    keeps its own keypoints (they are the Voronoi seeds).

    Args:
        masks: List of ``(H, W)`` boolean masks, one per instance.
        kpts: List of ``(n_i, 2)`` visible xy keypoints, index-aligned to
            ``masks``.

    Returns:
        List of disjoint boolean masks (a shallow copy when uncontested).
    """
    from scipy.ndimage import distance_transform_edt

    n = len(masks)
    if n == 0:
        return []
    h, w = masks[0].shape
    stack = np.stack(masks).astype(bool)
    contested = stack.sum(0) >= 2
    if not contested.any():
        return [m.copy() for m in masks]
    dists = np.full((n, h, w), 1e9, np.float32)
    for i, ks in enumerate(kpts):
        seed = np.ones((h, w), np.uint8)
        for x, y in np.asarray(ks, dtype=np.float32).reshape(-1, 2):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= yi < h and 0 <= xi < w:
                seed[yi, xi] = 0
        if seed.min() == 0:
            dists[i] = distance_transform_edt(seed)
    owner = np.argmin(dists, 0)
    return [np.where(contested & (owner != i), False, stack[i]) for i in range(n)]


def _load_sam_predictor(
    checkpoint: str, model_type: str = "vit_h", device: str = "cuda"
):
    """Lazily load a ``segment_anything.SamPredictor`` (harvested from #642).

    Args:
        checkpoint: Path to the SAM checkpoint (e.g. ``sam_vit_h_4b8939.pth``).
        model_type: SAM model registry key (``"vit_h"``, ``"vit_l"``, ``"vit_b"``).
        device: Torch device to place the model on.

    Returns:
        A ready ``SamPredictor``.

    Raises:
        ImportError: If ``segment-anything`` is not installed.
        ValueError: If ``checkpoint`` is ``None``.
        FileNotFoundError: If ``checkpoint`` does not exist.
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "mask_backend='sam' requires the optional 'segment-anything' "
            'dependency. Install it with `pip install "sleap-nn[sam]"` and '
            "download a SAM checkpoint (e.g. sam_vit_h_4b8939.pth), then pass "
            "sam_checkpoint=/path/to/ckpt."
        ) from e

    if checkpoint is None:
        raise ValueError(
            "mask_backend='sam' requires a SAM checkpoint. Download one "
            "(e.g. sam_vit_h_4b8939.pth) and pass sam_checkpoint=/path/to/ckpt."
        )
    ckpt = Path(checkpoint).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

    sam = sam_model_registry[model_type](checkpoint=ckpt.as_posix()).to(device)
    return SamPredictor(sam)


class MaskBackend(ABC):
    """Abstract prompted-mask backend (the :class:`SamBackend` / SAM3 interface).

    A backend encodes one image and answers a batch of prompts on it. The
    composed inference layer (:mod:`sleap_nn.inference.sam.mask_layer`) owns the
    crop/frame geometry; the backend owns only the model call. Selection is
    explicit (PLAN L2) — see :func:`sleap_nn.inference.sam.get_mask_backend`.
    """

    #: Per-model nominal predicted-IoU floor (SAM1: 0.88; SAM3 recalibrates).
    pred_iou_min: float = SAM_PRED_IOU_MIN

    @abstractmethod
    def masks(
        self, image: np.ndarray, prompts: Sequence[SamPrompt]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Encode ``image`` once and answer each prompt with a mask + raw score.

        Args:
            image: ``(H, W)`` grayscale (or ``(H, W, C)``) image to encode.
            prompts: Per-instance prompts in image space.

        Returns:
            ``(masks, scores)``: a list of ``(H, W)`` boolean masks (one per
            prompt) and the list of raw per-model scores.
        """
        raise NotImplementedError


class SamBackend(MaskBackend):
    """SAM1 (ViT-H) prompted-mask backend (the ``sleap_nn[sam]`` extra).

    Wraps a lazily loaded ``segment_anything.SamPredictor``. For one frame:
    CLAHE-equalize + 3-channel replicate, ``set_image`` once, then per prompt
    call ``predict(..., multimask_output=True)`` and select via :func:`_pick`.
    The raw SAM predicted-IoU of the chosen candidate is the mask score (PLAN
    §2.3 — store the raw per-model score; no drop-gate).

    Args:
        predictor: A ready ``SamPredictor`` (e.g. from :func:`_load_sam_predictor`
            or injected for testing). When ``None``, :meth:`from_checkpoint`
            builds one.
        clahe: Whether to CLAHE-equalize before encoding.
        max_box_area_factor: Candidate-rejection factor (:func:`_pick`).
        clahe_clip_limit: CLAHE clip limit.
        clahe_tile_grid: CLAHE tile grid.
        pred_iou_min: Nominal predicted-IoU floor carried for parity with SAM3;
            SAM1 reports the raw score and does not gate on it.
    """

    def __init__(
        self,
        predictor,
        clahe: bool = True,
        max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR,
        clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
        clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
        pred_iou_min: float = SAM_PRED_IOU_MIN,
    ) -> None:
        """Stash the predictor and the (model-specific) recipe knobs."""
        self.predictor = predictor
        self.clahe = bool(clahe)
        self.max_box_area_factor = float(max_box_area_factor)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.pred_iou_min = float(pred_iou_min)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        **kwargs,
    ) -> "SamBackend":
        """Build a backend by lazily loading a SAM checkpoint.

        Args:
            checkpoint: Path to the SAM checkpoint.
            model_type: SAM model registry key.
            device: Torch device for the model.
            **kwargs: Forwarded to :class:`SamBackend` (e.g. ``clahe``).

        Returns:
            A ready :class:`SamBackend`.
        """
        predictor = _load_sam_predictor(
            checkpoint, model_type=model_type, device=device
        )
        return cls(predictor, **kwargs)

    def masks(
        self, image: np.ndarray, prompts: Sequence[SamPrompt]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Encode ``image`` once, run each prompt, return masks + raw scores.

        Args:
            image: ``(H, W)`` grayscale (or ``(H, W, C)``) image / crop.
            prompts: Per-instance :class:`SamPrompt` in ``image`` pixel space.

        Returns:
            ``(masks, scores)`` with one ``(H, W)`` boolean mask + raw
            predicted-IoU per prompt. An empty prompt list returns ``([], [])``.
        """
        img = np.asarray(image)
        if img.ndim == 3:
            img = img[..., 0]
        h, w = img.shape[:2]
        rgb = _to_3ch_clahe(
            img,
            clahe=self.clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid=self.clahe_tile_grid,
        )
        self.predictor.set_image(rgb)

        out_masks: List[np.ndarray] = []
        out_scores: List[float] = []
        for prompt in prompts:
            point_coords = (
                prompt.point_coords.astype(np.float32)
                if prompt.point_coords is not None
                else None
            )
            point_labels = (
                prompt.point_labels.astype(np.int32)
                if prompt.point_labels is not None
                else None
            )
            box = prompt.box.astype(np.float32) if prompt.box is not None else None
            ms, sc, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
            b = _pick(ms, sc, prompt.reject_box, self.max_box_area_factor)
            out_masks.append(ms[b].astype(bool))
            out_scores.append(float(sc[b]))
        # Defensive: a degenerate single-px prompt could yield a (h, w) mismatch
        # only if SAM is given a wrong-size image; guard the contract here.
        for m in out_masks:
            if m.shape[:2] != (h, w):
                raise ValueError(f"SAM returned a {m.shape} mask for a {(h, w)} image.")
        return out_masks, out_scores
