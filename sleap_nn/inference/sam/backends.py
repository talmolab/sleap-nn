"""SAM mask backends for prompted instance segmentation (PR-A).

A *backend* owns the model-specific half of mask production: loading the model,
preprocessing an image (grayscale -> CLAHE -> 3-channel), encoding it once, and
turning a list of :class:`~sleap_nn.inference.sam.prompts.SamPrompt` into one
boolean mask + a raw per-model score per prompt.

PR-A ships :class:`SamBackend` (SAM1, ViT-H, Apache-2.0, ungated; the
``sleap_nn[sam]`` extra). PR-B adds :class:`Sam3Backend` (Meta SAM 3, gated
``facebook/sam3`` via ``transformers``; the ``sleap_nn[sam3]`` extra). The
model-specific recipe constants (:data:`SamBackend.pred_iou_min`, the
candidate-rejection factor, the keypoint box margins, CLAHE) and the
candidate-selection / score helpers (:func:`_pick`, :func:`own_containment`,
:func:`disjointify`) are harvested from the closed #642
(``sleap_nn/data/pseudomasks.py``) and the exp-07 locked recipe, repurposed to
emit a *raw score* rather than to drive a drop-gate (PLAN §1).

Backend selection is **explicit / required** (PLAN L2): there is no default
``mask_backend``; the caller names ``"sam"`` (SAM1) or ``"sam3"`` (SAM3) and both
honor the same :class:`MaskBackend` interface. The heavy imports
(``segment-anything`` / ``transformers``) are lazy so the default sleap-nn
install never needs either.

SAM3 specifics (NEVER shared with SAM1; harvested from the closed #643): its
predicted-IoU is on a lower scale, so the per-model floor is recalibrated to
:attr:`Sam3Backend.pred_iou_min` (``0.5``, **not** SAM1's ``0.88``), and its raw
masks are speckly/fragmented, so each is passed through :func:`_cleanup_speckle`
(morphological open + close + keep-keypoint-component) before it is returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sleap_nn.inference.sam.prompts import SamPrompt


def _to_3ch_clahe(
    img_gray: np.ndarray,
    clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
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
    max_box_area_factor: float = 1.5,
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


def _cleanup_speckle(
    mask: np.ndarray,
    kpts: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """De-speckle a (SAM3) mask, keeping the keypoint-connected blob.

    Harvested verbatim from #643 ``_cleanup_speckle``. Raw SAM3 masks are
    fragmented into many tiny connected components. This applies a morphological
    open (drop specks) + close (fill pinholes), then keeps only the connected
    component(s) that contain one of the instance's visible keypoints — falling
    back to the largest component if the cleanup detached every keypoint, or to
    the untouched mask if opening erased it entirely. The result is a single
    coherent silhouette that retains ~97% of the original area while collapsing
    the component count to ~1. Mandatory for SAM3 (PLAN §2.3); SAM1 masks are
    already solid and never run through this.

    Args:
        mask: ``(H, W)`` boolean mask for one instance.
        kpts: ``(n, 2)`` visible xy keypoints for that instance (the seeds whose
            connected component is kept).
        radius: Morphological structuring-element radius (px) for open/close.

    Returns:
        Cleaned ``(H, W)`` boolean mask.
    """
    import cv2
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mm = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k)
    labels_arr, n = ndimage.label(mm)
    if n == 0:
        # Opening erased everything (a very thin mask) -> keep the raw mask.
        return mask
    h, w = mm.shape
    keep = set()
    for x, y in np.asarray(kpts, dtype=np.float32).reshape(-1, 2):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= yi < h and 0 <= xi < w and labels_arr[yi, xi] > 0:
            keep.add(int(labels_arr[yi, xi]))
    if not keep:
        # Cleanup detached every keypoint -> keep the largest component.
        sizes = ndimage.sum(np.ones_like(labels_arr), labels_arr, range(1, n + 1))
        keep = {int(np.argmax(sizes)) + 1}
    return np.isin(labels_arr, list(keep))


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


def _load_sam3(model_id: str = "facebook/sam3", device: str = "cuda"):
    """Lazily load the SAM3 image visual-prompt model + processor (#643).

    Uses the transformers ``Sam3TrackerModel`` / ``Sam3TrackerProcessor`` *image*
    visual-prompt path (NOT the video-tracking or text classes; those land in
    PR-D). The model is loaded in ``bfloat16`` on ``device`` and set to
    ``.eval()``.

    ``facebook/sam3`` is a **gated** Hugging Face model: authentication comes
    from a cached HF token (``huggingface-cli login`` /
    ``~/.cache/huggingface/token``) or the ``HF_TOKEN`` env var, handled
    transparently by ``transformers``. The two-venv env reality (PLAN §2.3, DQ6):
    SAM3 needs ``transformers>=5`` + torch cu130 + the gated weights, which may
    not co-install with the sleap-nn venv; install it on its own or run the SAM3
    image path out-of-process and assemble the ``.slp`` in the sleap-nn venv. See
    ``docs/guides/sam-inference-segmentation.md``.

    Args:
        model_id: Hugging Face model id (default ``"facebook/sam3"``, the gated
            SAM3 image visual-prompt path).
        device: Torch device to place the model on.

    Returns:
        Tuple ``(model, processor)``.

    Raises:
        ImportError: If ``transformers`` (with SAM3 support) is not installed.
    """
    import torch

    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "mask_backend='sam3' requires the optional 'transformers' dependency "
            "with SAM3 support (transformers>=5.0). Install it with "
            '`pip install "sleap-nn[sam3]"`. Note that facebook/sam3 is a GATED '
            "model: run `huggingface-cli login` (or set HF_TOKEN) with an account "
            "that has been granted access. SAM3 may need its own environment "
            "(transformers>=5 + torch cu130 + the gated weights); see "
            "docs/guides/sam-inference-segmentation.md for the two-venv handoff."
        ) from e

    model = (
        Sam3TrackerModel.from_pretrained(model_id)
        .to(device, dtype=torch.bfloat16)
        .eval()
    )
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    return model, processor


class MaskBackend(ABC):
    """Abstract prompted-mask backend (the :class:`SamBackend` / SAM3 interface).

    A backend encodes one image and answers a batch of prompts on it. The
    composed inference layer (:mod:`sleap_nn.inference.sam.mask_layer`) owns the
    crop/frame geometry; the backend owns only the model call. Selection is
    explicit (PLAN L2) — see :func:`sleap_nn.inference.sam.get_mask_backend`.
    """

    #: Per-model nominal predicted-IoU floor. Defaults to SAM1's ``0.88``;
    #: :class:`Sam3Backend` overrides it with a recalibrated ``0.5`` (SAM3's
    #: predicted-IoU is on a lower scale, PLAN §2.3). Carried as a per-model
    #: attribute so SAM3 can override it; SAM1's raw predicted-IoU is reported as
    #: the mask score, not used as a gate.
    pred_iou_min: float = 0.88

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
        max_box_area_factor: float = 1.5,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        pred_iou_min: float = 0.88,
    ) -> None:
        """Stash the predictor and the (model-specific) recipe knobs.

        The recipe defaults are the locked SAM1 values (harvested from #642 /
        exp-07; PLAN §1): ``max_box_area_factor=1.5`` drops candidates whose area
        exceeds ``1.5 * box-area`` (kills SAM's over-confident whole-arena
        candidate, see :func:`_pick`); ``clahe_clip_limit=3.0`` /
        ``clahe_tile_grid=(8, 8)`` are the CLAHE parameters applied to the
        grayscale image before encoding; ``pred_iou_min=0.88`` is SAM1's nominal
        predicted-IoU floor, reported (not gated) and carried for SAM3 parity.
        """
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


def _cleanup_seed(prompt: SamPrompt) -> np.ndarray:
    """The keypoints used to seed :func:`_cleanup_speckle` for one SAM3 prompt.

    The positive points (pose keypoints / centroid / crop center) are the natural
    component seeds. A box-only prompt (``box`` mode) has no points, so its box
    center is used instead so the cleanup still keeps the central blob.

    Args:
        prompt: A built :class:`~sleap_nn.inference.sam.prompts.SamPrompt`.

    Returns:
        ``(n, 2)`` float32 xy seed keypoints (``n >= 1``).
    """
    if prompt.point_coords is not None and len(prompt.point_coords):
        return np.asarray(prompt.point_coords, dtype=np.float32).reshape(-1, 2)
    box = np.asarray(prompt.reject_box, dtype=np.float32).reshape(4)
    return np.array(
        [[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]], dtype=np.float32
    )


class Sam3Backend(MaskBackend):
    """SAM3 (Meta SAM 3) prompted-mask backend (the ``sleap_nn[sam3]`` extra).

    Wraps a lazily loaded transformers ``Sam3TrackerModel`` + ``Sam3TrackerProcessor``
    image visual-prompt pair. Honors the same :class:`MaskBackend` surface as
    :class:`SamBackend`, but two SAM3 specifics are mandatory and NEVER shared
    with SAM1 (PLAN §2.3, harvested from #643):

    * **Recalibrated floor.** SAM3's ``iou_scores`` (predicted-IoU) are on a
      LOWER scale than SAM1 (median ~0.68 vs SAM1's ~0.95). SAM1's ``0.88`` floor
      applied verbatim would drop ~100% of SAM3 masks as a pure calibration
      artifact, so the per-model :attr:`pred_iou_min` defaults to ``0.5``
      (~SAM1's ``0.88`` in percentile terms), never SAM1's ``0.88``. As with SAM1
      the raw chosen-candidate score is reported, not gated on.
    * **Speckle cleanup.** Raw SAM3 masks are speckly/fragmented (median ~14
      connected components per mask vs SAM1's 1), with ~97% of the area in the
      keypoint-connected component. The speckle is cosmetic, so each chosen mask
      is passed through :func:`_cleanup_speckle` (morphological open + close +
      keep-keypoint-component, -> median 1 component, ~97% area retained) before
      it is returned. Mandatory for SAM3; SAM1 masks are already solid.

    Unlike SAM1's per-prompt loop, SAM3 runs **all prompts for the frame in a
    single batched forward pass** (each prompt is one object), matching #643's
    ``_sam3_instance_masks``. The candidate selection (:func:`_pick`) and the
    raw-score contract are identical to SAM1.

    Args:
        model: A ready ``Sam3TrackerModel`` (e.g. from :func:`_load_sam3` or
            injected for testing).
        processor: The matching ``Sam3TrackerProcessor``.
        device: Torch device the prompt tensors are moved to.
        clahe: Whether to CLAHE-equalize before encoding.
        max_box_area_factor: Candidate-rejection factor (:func:`_pick`).
        clahe_clip_limit: CLAHE clip limit.
        clahe_tile_grid: CLAHE tile grid.
        cleanup_radius: Speckle-cleanup morphological radius (px).
        pred_iou_min: Per-model nominal predicted-IoU floor (default ``0.5``,
            recalibrated; NEVER SAM1's ``0.88``); reported, not gated.
    """

    #: SAM3's recalibrated predicted-IoU floor. NEVER SAM1's ``0.88`` (SAM3's
    #: predicted-IoU is on a lower scale; the ``0.5`` value is ~SAM1's ``0.88`` in
    #: percentile terms). Reported as the per-model score, not gated on.
    pred_iou_min: float = 0.5

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        clahe: bool = True,
        max_box_area_factor: float = 1.5,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        cleanup_radius: int = 3,
        pred_iou_min: float = 0.5,
    ) -> None:
        """Stash the model/processor and the (SAM3-specific) recipe knobs.

        The SAM1-shared recipe defaults match :class:`SamBackend`
        (``max_box_area_factor=1.5``, ``clahe_clip_limit=3.0``,
        ``clahe_tile_grid=(8, 8)``). The SAM3-specific defaults are
        ``cleanup_radius=3`` (the morphological open + close radius (px) for the
        mandatory speckle cleanup) and ``pred_iou_min=0.5`` (the recalibrated
        floor; NEVER SAM1's ``0.88``, since SAM3's predicted-IoU is on a lower
        scale).
        """
        self.model = model
        self.processor = processor
        self.device = str(device)
        self.clahe = bool(clahe)
        self.max_box_area_factor = float(max_box_area_factor)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.cleanup_radius = int(cleanup_radius)
        self.pred_iou_min = float(pred_iou_min)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam3",
        device: str = "cuda",
        **kwargs,
    ) -> "Sam3Backend":
        """Build a backend by lazily loading the gated SAM3 model + processor.

        Args:
            model_id: Hugging Face model id (default ``"facebook/sam3"``).
            device: Torch device for the model.
            **kwargs: Forwarded to :class:`Sam3Backend` (e.g. ``clahe``).

        Returns:
            A ready :class:`Sam3Backend`.

        Raises:
            ImportError: If ``transformers`` (with SAM3 support) is absent.
        """
        model, processor = _load_sam3(model_id=model_id, device=device)
        return cls(model, processor, device=device, **kwargs)

    def masks(
        self, image: np.ndarray, prompts: Sequence[SamPrompt]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Encode ``image`` once, run all prompts batched, return masks + scores.

        Mirrors #643's ``_sam3_instance_masks``: one batched forward pass over all
        prompts (each prompt is an object), :func:`_pick` to choose a candidate,
        :func:`_cleanup_speckle` to de-fragment, and the raw chosen predicted-IoU
        as the per-mask score.

        Args:
            image: ``(H, W)`` grayscale (or ``(H, W, C)``) image / crop.
            prompts: Per-instance :class:`SamPrompt` in ``image`` pixel space.

        Returns:
            ``(masks, scores)`` with one ``(H, W)`` boolean mask + raw
            predicted-IoU per prompt (on SAM3's lower scale). An empty prompt list
            returns ``([], [])``.
        """
        import torch

        prompts = list(prompts)
        img = np.asarray(image)
        if img.ndim == 3:
            img = img[..., 0]
        img = np.ascontiguousarray(img).astype(np.uint8)
        h, w = img.shape[:2]

        out_masks: List[np.ndarray] = [np.zeros((h, w), bool) for _ in prompts]
        out_scores: List[float] = [0.0 for _ in prompts]
        if not prompts:
            return out_masks, out_scores

        rgb = _to_3ch_clahe(
            img,
            clahe=self.clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid=self.clahe_tile_grid,
        )

        # Build batched per-object prompts (one image, each prompt an object).
        # Mirror SAM1 (``SamBackend.masks``): forward only a prompt's real
        # ``box`` — never ``reject_box``, which exists solely for the
        # candidate-rejection heuristic (:func:`_pick`). Point-only prompts
        # (e.g. ``centroid`` mode, ``box is None``) carry no box; feeding them
        # ``reject_box`` would hand SAM3 a whole-frame "segment everything" box
        # and make SAM3 diverge from SAM1 on identical input.
        obj_points: List[List[List[float]]] = []
        obj_labels: List[List[int]] = []
        obj_boxes: List[List[float]] = []
        any_box = False
        for prompt in prompts:
            pc = prompt.point_coords
            pl = prompt.point_labels
            if pc is not None and len(pc):
                obj_points.append([[float(x), float(y)] for x, y in pc])
                labels = [int(v) for v in pl] if pl is not None else [1] * len(pc)
                obj_labels.append(labels)
            else:
                obj_points.append([])
                obj_labels.append([])
            if prompt.box is not None:
                obj_boxes.append([float(v) for v in np.asarray(prompt.box).reshape(4)])
                any_box = True
            else:
                obj_boxes.append([])

        processor_kwargs = dict(
            images=rgb,
            input_points=[obj_points],
            input_labels=[obj_labels],
            return_tensors="pt",
        )
        # Only forward boxes when a prompt actually has one (pose / box modes);
        # a frame of point-only prompts forwards no boxes at all.
        if any_box:
            processor_kwargs["input_boxes"] = [obj_boxes]
        inputs = self.processor(**processor_kwargs).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        post = self.processor.post_process_masks(
            out.pred_masks, original_sizes=inputs["original_sizes"], binarize=True
        )[
            0
        ]  # (n_obj, n_cand, H, W) bool
        post = np.asarray(post.cpu().numpy()).astype(bool)
        scores = np.asarray(out.iou_scores.float().cpu().numpy()[0])  # (n_obj, n_cand)

        for j, prompt in enumerate(prompts):
            cand_masks = post[j]
            cand_scores = scores[j]
            b = _pick(
                cand_masks, cand_scores, prompt.reject_box, self.max_box_area_factor
            )
            mask = _cleanup_speckle(
                cand_masks[b], _cleanup_seed(prompt), self.cleanup_radius
            )
            out_masks[j] = mask.astype(bool)
            out_scores[j] = float(cand_scores[b])

        for m in out_masks:
            if m.shape[:2] != (h, w):
                raise ValueError(
                    f"SAM3 returned a {m.shape} mask for a {(h, w)} image."
                )
        return out_masks, out_scores
