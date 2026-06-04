"""Pose -> per-instance segmentation pseudomask generation.

Bottom-up instance-segmentation models train on ``.slp`` files whose labeled
frames already carry per-instance ground-truth
``sio.UserSegmentationMask`` silhouettes. This module produces that GT from
pose keypoints in an explicit, offline prep step, writing a new ``.slp`` whose
frames carry both the original keypoint instances (retained for eval
frame-pairing) and the generated masks.

Five GT sources are supported, selected per dataset:

* ``"skeleton"`` (default, no extra deps): rasterize each instance's skeleton
  into a dilated boolean silhouette using only ``cv2`` + ``numpy``. Robust on
  every body plan; the silhouette is a coarse over-approximation of the animal.
* ``"sam"`` (optional): prompt Segment Anything 1 (ViT-H by default) with each
  instance's visible keypoints to produce a tight, true silhouette, then apply
  a quality filter (predicted-IoU / own-keypoint-containment / area-ratio). SAM
  masks are truer on compact animals (e.g. mice) but degrade badly on tiny or
  elongated ones (e.g. flies), so the source must be a per-dataset choice.
* ``"hybrid"`` (optional): SAM1 with a per-instance dilated-skeleton fallback
  whenever the SAM mask fails the quality filter, so every instance still gets
  GT and the 1:1 instance<->mask pairing is preserved.
* ``"sam3"`` (optional): the same keypoint+box recipe via Meta SAM 3's image
  visual-prompt path (``transformers`` ``Sam3Tracker*``). On mice it matches
  SAM1 silhouette-for-silhouette (mask IoU ~0.85 vs SAM1) after a speckle
  cleanup; it is a higher-ceiling but heavier and *gated* opt-in (SAM1 stays
  the default SAM path). Two SAM3 specifics are handled internally: a
  recalibrated predicted-IoU floor (SAM3 scores are lower-scaled) and a
  morphological speckle cleanup of the fragmented raw masks.
* ``"hybrid_sam3"`` (optional): SAM3 with the per-instance dilated-skeleton
  fallback, analogous to ``"hybrid"``.

The ``sam``/``hybrid`` paths require the optional ``segment-anything``
dependency (``pip install "sleap-nn[sam]"``) plus a SAM checkpoint; the
``sam3``/``hybrid_sam3`` paths require ``transformers`` with SAM3 support
(``pip install "sleap-nn[sam3]"``) and access to the gated ``facebook/sam3``
model (``huggingface-cli login``). All heavy deps are imported / loaded lazily
so the default skeleton path stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import sleap_io as sio
from loguru import logger

# ---------------------------------------------------------------------------
# Locked SAM recipe constants (module defaults).
#
# These remain the single source of truth for the default recipe; the
# :class:`PseudomaskGenerator` dataclass fields default to these values (so the
# constants and the dataclass can never drift), and the free helper functions
# keep them as keyword-argument defaults for back-compat.
# ---------------------------------------------------------------------------
#: Minimum SAM predicted-IoU for a SAM mask to be accepted.
PRED_IOU_MIN = 0.88
#: Minimum fraction of an instance's visible keypoints that must fall inside
#: its own SAM mask for the mask to be accepted.
OWN_CONTAIN_MIN = 0.9
#: SAM-area / skeleton-area must lie within ``[AREA_RATIO_MIN, AREA_RATIO_MAX]``.
AREA_RATIO_MIN = 0.2
AREA_RATIO_MAX = 2.5
#: SAM keypoint-box margin: ``max(SAM_BOX_MARGIN_MIN, SAM_BOX_MARGIN_FRAC*side)``.
SAM_BOX_MARGIN_FRAC = 0.6
SAM_BOX_MARGIN_MIN = 15.0
#: Reject SAM candidates whose area exceeds ``SAM_MAX_BOX_AREA_FACTOR * box-area``
#: (kills SAM's over-confident whole-arena candidate).
SAM_MAX_BOX_AREA_FACTOR = 1.5
#: CLAHE parameters applied to the grayscale image before prompting SAM.
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
#: Skeleton-rasterizer defaults (shared by the rasterizer + the generator).
NODE_RADIUS = 4
EDGE_THICKNESS = 4
DILATE_FRAC = 0.10
MIN_DILATE = 4

# ---------------------------------------------------------------------------
# SAM3 recipe constants (module defaults).
#
# As with the SAM1 constants above, these remain the single source of truth for
# the SAM3 recipe; the :class:`PseudomaskGenerator` dataclass fields
# (``sam3_model_id``, ``sam3_pred_iou_min``, ``sam3_cleanup_radius``) default to
# these values (so the two can never drift), and :func:`_sam3_instance_masks`
# keeps them as keyword-argument defaults for back-compat.
# ---------------------------------------------------------------------------
# SAM3 (Meta SAM 3, via the transformers ``Sam3Tracker*`` image visual-prompt
# path) is an alternative mask backend that uses the IDENTICAL keypoint+box
# prompt/select/disjointify recipe as SAM1. It differs from SAM1 in two ways
# that the defaults below account for:
#   1. ``iou_scores`` (predicted-IoU) are on a LOWER scale than SAM1 (median
#      ~0.68 vs SAM1's ~0.95). SAM1's ``PRED_IOU_MIN=0.88`` floor applied
#      verbatim would drop ~100% of SAM3 masks as a pure calibration artifact,
#      not a quality signal, so the SAM3 floor is recalibrated to 0.5
#      (~SAM1's 0.88 in percentile terms; keeps ~92% of mice instances).
#   2. Raw SAM3 masks are speckly/fragmented (median ~14 connected components/
#      mask vs SAM1's 1), with ~97% of the area in the keypoint-connected
#      component. The speckle is cosmetic and removed by a morphological
#      open+close + keep-keypoint-connected-component cleanup (-> median 1
#      component, ~97% area retained), applied before the quality filter.
#: Minimum SAM3 predicted-IoU (recalibrated; see note above).
SAM3_PRED_IOU_MIN = 0.5
#: Morphological radius (px) for the SAM3 speckle open+close cleanup.
SAM3_CLEANUP_RADIUS = 3
#: Default HF model id for the SAM3 image visual-prompt path.
SAM3_MODEL_ID = "facebook/sam3"


# ---------------------------------------------------------------------------
# Skeleton rasterizer (dependency-free: cv2 + numpy only).
# ---------------------------------------------------------------------------
def rasterize_instance_mask(
    points: np.ndarray,
    edge_inds: Iterable[Tuple[int, int]],
    hw: Tuple[int, int],
    node_radius: int = 4,
    edge_thickness: int = 4,
    dilate_frac: float = 0.10,
    min_dilate: int = 4,
) -> np.ndarray:
    """Rasterize one instance's skeleton into a dilated boolean mask.

    Each visible node is drawn as a filled disk and each visible edge as a thick
    line; the union is dilated by ``max(min_dilate, round(dilate_frac * diag))``
    where ``diag`` is the keypoint bounding-box diagonal, so the silhouette gains
    body proportional to the instance size.

    Args:
        points: ``(n_nodes, 2)`` xy keypoints (may contain ``NaN`` for missing).
        edge_inds: Iterable of ``(src_idx, dst_idx)`` node-index pairs.
        hw: ``(height, width)`` of the target mask.
        node_radius: Radius (px) of the filled disk drawn at each visible node.
        edge_thickness: Line thickness (px) for each visible edge.
        dilate_frac: Dilation radius as a fraction of the instance bbox diagonal.
        min_dilate: Minimum dilation radius (px).

    Returns:
        Boolean array of shape ``hw`` -- the dilated skeleton silhouette.
    """
    h, w = hw
    canvas = np.zeros((h, w), dtype=np.uint8)

    # Edges first (so node disks sit on top of the lines).
    for s, d in edge_inds:
        ps, pd = points[s], points[d]
        if np.isnan(ps).any() or np.isnan(pd).any():
            continue
        cv2.line(
            canvas,
            (int(round(ps[0])), int(round(ps[1]))),
            (int(round(pd[0])), int(round(pd[1]))),
            color=1,
            thickness=edge_thickness,
            lineType=cv2.LINE_8,
        )

    # Nodes.
    for p in points:
        if np.isnan(p).any():
            continue
        cv2.circle(
            canvas,
            (int(round(p[0])), int(round(p[1]))),
            node_radius,
            color=1,
            thickness=-1,
        )

    # Dilate proportionally to the instance size so the silhouette has body.
    valid = points[~np.isnan(points).any(axis=1)]
    diag = (
        float(np.linalg.norm(valid.max(axis=0) - valid.min(axis=0)))
        if len(valid) >= 2
        else 0.0
    )
    dilate = max(min_dilate, int(round(dilate_frac * diag)))
    if dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
        )
        canvas = cv2.dilate(canvas, k)

    return canvas.astype(bool)


# ---------------------------------------------------------------------------
# SAM helpers (locked recipe). All dependency-free except _load_sam_predictor.
# ---------------------------------------------------------------------------
def _kpt_box(
    pos: np.ndarray,
    hw: Tuple[int, int],
    margin_frac: float = SAM_BOX_MARGIN_FRAC,
    margin_min: float = SAM_BOX_MARGIN_MIN,
) -> np.ndarray:
    """Compute a padded keypoint bounding box ``[x0, y0, x1, y1]`` clamped to ``hw``.

    Args:
        pos: ``(n, 2)`` visible xy keypoints (no NaNs).
        hw: ``(height, width)`` of the frame.
        margin_frac: Box margin as a fraction of the box side length.
        margin_min: Minimum box margin (px) per axis.

    Returns:
        ``float32`` array ``[x0, y0, x1, y1]``.
    """
    x0, y0 = pos.min(0)
    x1, y1 = pos.max(0)
    mx = max(margin_min, margin_frac * (x1 - x0))
    my = max(margin_min, margin_frac * (y1 - y0))
    return np.array(
        [
            max(0, x0 - mx),
            max(0, y0 - my),
            min(hw[1] - 1, x1 + mx),
            min(hw[0] - 1, y1 + my),
        ],
        np.float32,
    )


def _pick(
    masks: np.ndarray,
    scores: np.ndarray,
    box: np.ndarray,
    max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR,
) -> int:
    """Pick the best SAM candidate mask index.

    Rejects candidates whose area exceeds ``max_box_area_factor * box-area``
    (SAM's over-confident whole-arena candidate), then returns the highest
    predicted-IoU survivor; if all are rejected returns the smallest candidate.

    Args:
        masks: ``(n_cands, H, W)`` candidate masks.
        scores: ``(n_cands,)`` SAM predicted-IoU per candidate.
        box: ``[x0, y0, x1, y1]`` prompt box.
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


def _disjointify(
    masks: Sequence[np.ndarray], kpts: Sequence[np.ndarray]
) -> List[np.ndarray]:
    """Make per-instance masks disjoint via keypoint-Voronoi assignment.

    Any pixel claimed by >=2 masks is assigned to the instance whose nearest
    visible keypoint is closest, so the result is exactly disjoint and each
    instance never loses its own keypoints (they are the Voronoi seeds).

    Args:
        masks: List of ``(H, W)`` boolean masks, one per instance.
        kpts: List of ``(n_i, 2)`` visible xy keypoints, index-aligned to ``masks``.

    Returns:
        List of disjoint boolean masks.
    """
    from scipy.ndimage import distance_transform_edt

    n = len(masks)
    h, w = masks[0].shape
    stack = np.stack(masks).astype(bool)
    contested = stack.sum(0) >= 2
    if not contested.any():
        return [m.copy() for m in masks]
    dists = np.full((n, h, w), 1e9, np.float32)
    for i, ks in enumerate(kpts):
        seed = np.ones((h, w), np.uint8)
        for x, y in ks:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                seed[yi, xi] = 0
        if seed.min() == 0:
            dists[i] = distance_transform_edt(seed)
    owner = np.argmin(dists, 0)
    return [np.where(contested & (owner != i), False, stack[i]) for i in range(n)]


def _sam_instance_masks(
    predictor,
    img_gray: np.ndarray,
    kpts: Sequence[np.ndarray],
    clahe: bool = True,
    max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR,
    box_margin_frac: float = SAM_BOX_MARGIN_FRAC,
    box_margin_min: float = SAM_BOX_MARGIN_MIN,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> Tuple[List[np.ndarray], List[float]]:
    """Run the locked SAM recipe on one frame.

    Per instance: CLAHE-equalize the grayscale image (replicated to 3 channels),
    prompt SAM with the instance's visible keypoints (positive) plus a padded
    keypoint box (no negatives), ``multimask_output=True``, pick via
    :func:`_pick`. Then make all masks disjoint via :func:`_disjointify`.

    Args:
        predictor: A loaded ``segment_anything.SamPredictor``.
        img_gray: ``(H, W)`` uint8 grayscale frame.
        kpts: List of ``(n_i, 2)`` visible xy keypoints per instance.
        clahe: Whether to CLAHE-equalize before prompting SAM.
        max_box_area_factor: Passed through to :func:`_pick`.
        box_margin_frac: Keypoint-box margin fraction passed to :func:`_kpt_box`.
        box_margin_min: Keypoint-box minimum margin passed to :func:`_kpt_box`.
        clahe_clip_limit: CLAHE clip limit applied when ``clahe`` is true.
        clahe_tile_grid: CLAHE tile grid applied when ``clahe`` is true.

    Returns:
        Tuple ``(masks, pred_ious)`` where ``masks`` is a list of disjoint
        boolean ``(H, W)`` arrays and ``pred_ious`` the per-instance SAM
        predicted-IoU of the chosen candidate.
    """
    if clahe:
        src = cv2.createCLAHE(clahe_clip_limit, clahe_tile_grid).apply(img_gray)
    else:
        src = img_gray
    predictor.set_image(np.stack([src] * 3, -1).astype(np.uint8))
    h, w = img_gray.shape
    masks: List[np.ndarray] = []
    pred_ious: List[float] = []
    for ks in kpts:
        if len(ks) == 0:
            masks.append(np.zeros((h, w), bool))
            pred_ious.append(0.0)
            continue
        box = _kpt_box(
            ks,
            img_gray.shape,
            margin_frac=box_margin_frac,
            margin_min=box_margin_min,
        )
        ms, sc, _ = predictor.predict(
            point_coords=ks.astype(np.float32),
            point_labels=np.ones(len(ks), np.int32),
            box=box,
            multimask_output=True,
        )
        b = _pick(ms, sc, box, max_box_area_factor)
        masks.append(ms[b].astype(bool))
        pred_ious.append(float(sc[b]))
    masks = _disjointify(masks, kpts)
    return masks, pred_ious


def _cleanup_speckle(
    mask: np.ndarray,
    kpts: np.ndarray,
    radius: int = SAM3_CLEANUP_RADIUS,
) -> np.ndarray:
    """Remove speckle from a (SAM3) mask, keeping the keypoint-connected blob.

    Raw SAM3 masks are fragmented into many tiny connected components. This
    applies a morphological open (drop specks) + close (fill pinholes), then
    keeps only the connected component(s) that contain one of the instance's
    visible keypoints (falling back to the largest component if the cleanup
    detached every keypoint, or to the untouched mask if opening erased it
    entirely). The result is a single coherent silhouette that retains ~97% of
    the original mask area while collapsing the component count to ~1.

    Args:
        mask: ``(H, W)`` boolean mask for one instance.
        kpts: ``(n, 2)`` visible xy keypoints for that instance (the seeds whose
            connected component is kept).
        radius: Morphological structuring-element radius (px) for open/close.

    Returns:
        Cleaned ``(H, W)`` boolean mask.
    """
    from scipy import ndimage

    if not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mm = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k)
    labels_arr, n = ndimage.label(mm)
    if n == 0:
        # Opening erased everything (very thin mask) -> keep the raw mask.
        return mask
    h, w = mm.shape
    keep = set()
    for x, y in kpts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and labels_arr[yi, xi] > 0:
            keep.add(int(labels_arr[yi, xi]))
    if not keep:
        # Cleanup detached every keypoint -> keep the largest component.
        sizes = ndimage.sum(np.ones_like(labels_arr), labels_arr, range(1, n + 1))
        keep = {int(np.argmax(sizes)) + 1}
    return np.isin(labels_arr, list(keep))


def _load_sam3(model_id: str = SAM3_MODEL_ID, device: str = "cuda"):
    """Lazily load the SAM3 image visual-prompt model + processor.

    Uses the transformers ``Sam3TrackerModel`` / ``Sam3TrackerProcessor`` image
    visual-prompt path (NOT the video-tracking or text/PCS classes). The model
    is loaded in ``bfloat16`` on ``device`` and set to ``.eval()``.

    ``facebook/sam3`` is a GATED Hugging Face model: authentication comes from a
    cached HF token (``huggingface-cli login`` / ``~/.cache/huggingface/token``)
    or the ``HF_TOKEN`` env var, handled transparently by ``transformers``.

    Args:
        model_id: Hugging Face model id.
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
            "The 'sam3'/'hybrid_sam3' pseudomask source requires the optional "
            "'transformers' dependency with SAM3 support (transformers>=5.0). "
            'Install it with `pip install "sleap-nn[sam3]"`. Note that '
            "facebook/sam3 is a gated model: run `huggingface-cli login` "
            "(or set HF_TOKEN) with an account that has been granted access."
        ) from e

    model = (
        Sam3TrackerModel.from_pretrained(model_id)
        .to(device, dtype=torch.bfloat16)
        .eval()
    )
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    return model, processor


def _sam3_instance_masks(
    model,
    processor,
    img_gray: np.ndarray,
    kpts: Sequence[np.ndarray],
    device: str = "cuda",
    clahe: bool = True,
    max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR,
    box_margin_frac: float = SAM_BOX_MARGIN_FRAC,
    box_margin_min: float = SAM_BOX_MARGIN_MIN,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
    cleanup_radius: int = SAM3_CLEANUP_RADIUS,
) -> Tuple[List[np.ndarray], List[float]]:
    """Run the SAM3 image visual-prompt recipe on one frame.

    Mirrors :func:`_sam_instance_masks` (same keypoint+box prompt, oversized-
    candidate rejection via :func:`_pick`, and keypoint-Voronoi
    :func:`_disjointify`) but uses SAM3's ``Sam3TrackerModel`` image path, which
    runs ALL instances of the frame in a single batched forward pass, and adds a
    per-mask speckle :func:`_cleanup_speckle` before disjointify (SAM3 masks are
    fragmented where SAM1's are solid).

    Args:
        model: The SAM3 ``Sam3TrackerModel`` from :func:`_load_sam3`.
        processor: The SAM3 ``Sam3TrackerProcessor`` from :func:`_load_sam3`.
        img_gray: ``(H, W)`` uint8 grayscale frame.
        kpts: List of ``(n_i, 2)`` visible xy keypoints per instance.
        device: Torch device the prompts are moved to.
        clahe: Whether to CLAHE-equalize before prompting SAM3.
        max_box_area_factor: Passed through to :func:`_pick`.
        box_margin_frac: Keypoint-box margin fraction passed to :func:`_kpt_box`.
        box_margin_min: Keypoint-box minimum margin passed to :func:`_kpt_box`.
        clahe_clip_limit: CLAHE clip limit applied when ``clahe`` is true.
        clahe_tile_grid: CLAHE tile grid applied when ``clahe`` is true.
        cleanup_radius: Speckle-cleanup morphological radius (px).

    Returns:
        Tuple ``(masks, pred_ious)`` where ``masks`` is a list of disjoint,
        de-speckled boolean ``(H, W)`` arrays and ``pred_ious`` the per-instance
        SAM3 predicted-IoU (on SAM3's lower scale; see ``SAM3_PRED_IOU_MIN``).
    """
    import torch

    if clahe:
        src = cv2.createCLAHE(clahe_clip_limit, clahe_tile_grid).apply(img_gray)
    else:
        src = img_gray
    rgb = np.stack([src] * 3, -1).astype(np.uint8)
    h, w = img_gray.shape

    # Build batched per-object prompts (one image, each instance a separate
    # object). Instances with no visible keypoints get an empty mask.
    obj_points: List[List[List[float]]] = []
    obj_labels: List[List[int]] = []
    obj_boxes: List[List[float]] = []
    valid_idx: List[int] = []
    boxes_by_idx: dict = {}
    for i, ks in enumerate(kpts):
        if len(ks) == 0:
            continue
        box = _kpt_box(
            np.asarray(ks, np.float32),
            (h, w),
            margin_frac=box_margin_frac,
            margin_min=box_margin_min,
        )
        obj_points.append([[float(x), float(y)] for x, y in ks])
        obj_labels.append([1] * len(ks))  # all positive, no negatives
        obj_boxes.append([float(v) for v in box])
        boxes_by_idx[i] = box
        valid_idx.append(i)

    masks: List[np.ndarray] = [np.zeros((h, w), bool) for _ in kpts]
    pred_ious: List[float] = [0.0 for _ in kpts]
    if not valid_idx:
        return _disjointify(masks, kpts), pred_ious

    inputs = processor(
        images=rgb,
        input_points=[obj_points],
        input_labels=[obj_labels],
        input_boxes=[obj_boxes],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    post = processor.post_process_masks(
        out.pred_masks, original_sizes=inputs["original_sizes"], binarize=True
    )[
        0
    ]  # (n_obj, n_cand, H, W) bool
    post = post.cpu().numpy().astype(bool)
    scores = out.iou_scores.float().cpu().numpy()[0]  # (n_obj, n_cand)

    for j, i in enumerate(valid_idx):
        box = boxes_by_idx[i]
        b = _pick(post[j], scores[j], box, max_box_area_factor)
        m = _cleanup_speckle(
            post[j][b], np.asarray(kpts[i], np.float32), cleanup_radius
        )
        masks[i] = m
        pred_ious[i] = float(scores[j][b])

    masks = _disjointify(masks, kpts)
    return masks, pred_ious


def _load_sam_predictor(
    checkpoint: str, model_type: str = "vit_h", device: str = "cuda"
):
    """Lazily load a ``segment_anything.SamPredictor`` from a checkpoint.

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
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "The 'sam'/'hybrid' pseudomask source requires the optional "
            "'segment-anything' dependency. Install it with "
            '`pip install "sleap-nn[sam]"` and download a SAM checkpoint '
            "(e.g. sam_vit_h_4b8939.pth), then pass --sam-checkpoint /path/to/ckpt."
        ) from e

    if checkpoint is None:
        raise ValueError(
            "A SAM checkpoint is required for the 'sam'/'hybrid' source. "
            "Download one (e.g. sam_vit_h_4b8939.pth) and pass "
            "--sam-checkpoint /path/to/ckpt."
        )
    ckpt = Path(checkpoint).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

    sam = sam_model_registry[model_type](checkpoint=ckpt.as_posix()).to(device)
    return SamPredictor(sam)


def _own_containment(mask: np.ndarray, kpts: np.ndarray, hw: Tuple[int, int]) -> float:
    """Fraction of an instance's visible keypoints that fall inside ``mask``."""
    if len(kpts) == 0:
        return 0.0
    h, w = hw
    inside = 0
    for x, y in kpts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            inside += 1
    return inside / len(kpts)


# ---------------------------------------------------------------------------
# Top-level generator.
# ---------------------------------------------------------------------------
@dataclass
class PseudomaskGenerator:
    """Configurable pose -> per-instance pseudomask generator.

    Bundles every tunable of the skeleton/SAM1/SAM3/hybrid recipe as a field with
    the same defaults as the module-level constants (the single source of truth),
    so callers can construct one generator and reuse it. The free helper functions
    (:func:`rasterize_instance_mask`, :func:`_kpt_box`, :func:`_pick`,
    :func:`_disjointify`, :func:`_sam_instance_masks`, :func:`_sam3_instance_masks`,
    :func:`_cleanup_speckle`, :func:`_own_containment`) remain available and
    receive the relevant fields when :meth:`run` calls them.

    Attributes:
        source: GT source, one of ``"skeleton"`` (default), ``"sam"``,
            ``"hybrid"``, ``"sam3"``, ``"hybrid_sam3"``. ``"sam"``/``"hybrid"``
            require ``segment-anything`` + a checkpoint; ``"sam3"``/
            ``"hybrid_sam3"`` require ``transformers`` + the gated
            ``facebook/sam3`` model.
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM1 checkpoint (required for sam/hybrid).
        sam_model_type: SAM1 model registry key.
        sam3_model_id: HF model id for the SAM3 path (sam3/hybrid_sam3).
        device: Torch device for the segmentation model.
        pred_iou_min: Quality filter: minimum predicted-IoU. ``None`` (default)
            selects the source-appropriate floor in :meth:`run`: ``PRED_IOU_MIN``
            (0.88) for SAM1 and ``sam3_pred_iou_min`` (0.5) for SAM3, whose
            predicted-IoU is on a lower scale.
        own_containment_min: Quality filter: minimum own-keypoint containment.
        area_ratio_range: Quality filter: ``(min, max)`` for model/skeleton area.
        sam_box_margin_frac: Keypoint-box margin as a fraction of the side.
        sam_box_margin_min: Keypoint-box minimum margin (px).
        sam_max_box_area_factor: Reject candidates above this * box-area.
        clahe_clip_limit: CLAHE clip limit applied before prompting the model.
        clahe_tile_grid: CLAHE tile grid applied before prompting the model.
        sam3_cleanup_radius: SAM3 speckle-cleanup morphological radius (px).
    """

    source: str = "skeleton"
    node_radius: int = NODE_RADIUS
    edge_thickness: int = EDGE_THICKNESS
    dilate_frac: float = DILATE_FRAC
    min_dilate: int = MIN_DILATE
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    sam3_model_id: str = SAM3_MODEL_ID
    device: str = "cuda"
    pred_iou_min: Optional[float] = None
    own_containment_min: float = OWN_CONTAIN_MIN
    area_ratio_range: Tuple[float, float] = (AREA_RATIO_MIN, AREA_RATIO_MAX)
    sam_box_margin_frac: float = SAM_BOX_MARGIN_FRAC
    sam_box_margin_min: float = SAM_BOX_MARGIN_MIN
    sam_max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_tile_grid: Tuple[int, int] = field(default_factory=lambda: CLAHE_TILE_GRID)
    sam3_pred_iou_min: float = SAM3_PRED_IOU_MIN
    sam3_cleanup_radius: int = SAM3_CLEANUP_RADIUS

    def run(self, labels: sio.Labels) -> sio.Labels:
        """Generate per-instance segmentation pseudomasks from pose keypoints.

        For every labeled frame with at least one posed instance, generates a
        boolean silhouette per instance and attaches it as a
        ``sio.UserSegmentationMask``. The original keypoint instances are
        retained (needed for downstream eval frame-pairing) and embedded images
        are preserved on save.

        Args:
            labels: Source ``sio.Labels`` carrying keypoint instances + images.

        Returns:
            New ``sio.Labels`` with per-frame ``UserSegmentationMask`` objects.

        Raises:
            ValueError: If ``source`` is not one of the supported values.
            ImportError: If ``source`` needs SAM/SAM3 but its dependency is
                absent.
        """
        source = self.source
        if source not in ("skeleton", "sam", "hybrid", "sam3", "hybrid_sam3"):
            raise ValueError(
                f"Unknown pseudomask source {source!r}; expected one of "
                "'skeleton', 'sam', 'hybrid', 'sam3', 'hybrid_sam3'."
            )

        skel = labels.skeletons[0]
        edge_inds = list(skel.edge_inds)
        skel_kw = dict(
            node_radius=self.node_radius,
            edge_thickness=self.edge_thickness,
            dilate_frac=self.dilate_frac,
            min_dilate=self.min_dilate,
        )

        use_sam1 = source in ("sam", "hybrid")
        use_sam3 = source in ("sam3", "hybrid_sam3")
        use_sam = use_sam1 or use_sam3
        is_hybrid = source in ("hybrid", "hybrid_sam3")

        # Source-appropriate predicted-IoU floor (SAM3's scores are
        # lower-scaled, so SAM1's 0.88 is recalibrated to keep mask quality
        # comparable).
        pred_iou_min = self.pred_iou_min
        if pred_iou_min is None:
            pred_iou_min = self.sam3_pred_iou_min if use_sam3 else PRED_IOU_MIN

        predictor = None
        sam3_model = sam3_processor = None
        if use_sam1:
            predictor = _load_sam_predictor(
                self.sam_checkpoint,
                model_type=self.sam_model_type,
                device=self.device,
            )
        elif use_sam3:
            sam3_model, sam3_processor = _load_sam3(
                model_id=self.sam3_model_id, device=self.device
            )

        area_min, area_max = self.area_ratio_range

        lfs = [lf for lf in labels.labeled_frames if len(lf.instances) > 0]
        logger.info(
            f"Generating '{source}' pseudomasks for {len(lfs)} labeled frames "
            f"(skeleton: {len(skel.nodes)} nodes / {len(edge_inds)} edges)."
        )

        new_lfs: List[sio.LabeledFrame] = []
        n_masks = 0
        n_sam = 0
        n_fallback = 0
        n_dropped = 0
        for lf in lfs:
            video = lf.video
            h, w = video.shape[1], video.shape[2]

            # Per-instance skeleton masks + keypoints, index-aligned to kept
            # instances. Instances with no pose or an empty skeleton mask are
            # skipped (mirrors offline scratch behavior).
            kept_instances = []
            skel_masks: List[np.ndarray] = []
            kpts_all: List[np.ndarray] = []
            for inst in lf.instances:
                pts = inst.numpy()[:, :2]
                if np.isnan(pts).all():
                    continue
                sm = rasterize_instance_mask(pts, edge_inds, (h, w), **skel_kw)
                if not sm.any():
                    continue
                kept_instances.append(inst)
                skel_masks.append(sm)
                kpts_all.append(pts[~np.isnan(pts).any(axis=1)].astype(np.float32))

            if not skel_masks:
                continue

            if not use_sam:
                final_masks = skel_masks
            else:
                img = np.asarray(lf.image)
                if img.ndim == 3:
                    img = img[..., 0]
                if use_sam3:
                    sam_masks, pred_ious = _sam3_instance_masks(
                        sam3_model,
                        sam3_processor,
                        img,
                        kpts_all,
                        device=self.device,
                        max_box_area_factor=self.sam_max_box_area_factor,
                        box_margin_frac=self.sam_box_margin_frac,
                        box_margin_min=self.sam_box_margin_min,
                        clahe_clip_limit=self.clahe_clip_limit,
                        clahe_tile_grid=self.clahe_tile_grid,
                        cleanup_radius=self.sam3_cleanup_radius,
                    )
                else:
                    sam_masks, pred_ious = _sam_instance_masks(
                        predictor,
                        img,
                        kpts_all,
                        max_box_area_factor=self.sam_max_box_area_factor,
                        box_margin_frac=self.sam_box_margin_frac,
                        box_margin_min=self.sam_box_margin_min,
                        clahe_clip_limit=self.clahe_clip_limit,
                        clahe_tile_grid=self.clahe_tile_grid,
                    )
                final_masks = []
                for i in range(len(skel_masks)):
                    sm = skel_masks[i]
                    mm = sam_masks[i]
                    sam_area = int(mm.sum())
                    skel_area = max(1, int(sm.sum()))
                    area_ratio = sam_area / skel_area
                    oc = _own_containment(mm, kpts_all[i], (h, w))
                    keep = (
                        sam_area > 0
                        and pred_ious[i] >= pred_iou_min
                        and oc >= self.own_containment_min
                        and area_min <= area_ratio <= area_max
                    )
                    if keep:
                        final_masks.append(mm)
                        n_sam += 1
                    elif is_hybrid:
                        final_masks.append(sm)  # per-instance skeleton fallback
                        n_fallback += 1
                    else:  # plain sam/sam3: drop the rejected instance entirely
                        final_masks.append(None)
                        n_dropped += 1

            masks_obj = []
            kept_for_frame = []
            for inst, m in zip(kept_instances, final_masks):
                if m is None:
                    continue
                masks_obj.append(sio.UserSegmentationMask.from_numpy(m.astype(bool)))
                kept_for_frame.append(inst)
            if not masks_obj:
                continue
            n_masks += len(masks_obj)
            new_lfs.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=lf.frame_idx,
                    instances=kept_for_frame,  # retain poses for eval pairing
                    masks=masks_obj,
                )
            )

        out_labels = sio.Labels(
            videos=labels.videos, skeletons=[skel], labeled_frames=new_lfs
        )
        mpf = n_masks / max(1, len(new_lfs))
        logger.info(
            f"Built {len(new_lfs)} frames, {n_masks} masks ({mpf:.2f} masks/frame)."
        )
        if use_sam:
            model_name = "SAM3" if use_sam3 else "SAM"
            logger.info(
                f"{model_name} masks: {n_sam}; skeleton-fallback: {n_fallback}; "
                f"dropped: {n_dropped}."
            )
        return out_labels


def generate_pseudomasks(
    labels: sio.Labels,
    source: str = "skeleton",
    *,
    node_radius: int = NODE_RADIUS,
    edge_thickness: int = EDGE_THICKNESS,
    dilate_frac: float = DILATE_FRAC,
    min_dilate: int = MIN_DILATE,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    sam3_model_id: str = SAM3_MODEL_ID,
    device: str = "cuda",
    pred_iou_min: Optional[float] = None,
    own_containment_min: float = OWN_CONTAIN_MIN,
    area_ratio_range: Tuple[float, float] = (AREA_RATIO_MIN, AREA_RATIO_MAX),
) -> sio.Labels:
    """Generate per-instance segmentation pseudomasks from pose keypoints.

    Thin wrapper that builds a :class:`PseudomaskGenerator` from the given
    keyword arguments and calls :meth:`PseudomaskGenerator.run`. Preserved as the
    stable public function entry point.

    Args:
        labels: Source ``sio.Labels`` carrying keypoint instances + image data.
        source: GT source, one of ``"skeleton"`` (default), ``"sam"``,
            ``"hybrid"``, ``"sam3"``, ``"hybrid_sam3"``. The ``sam*`` sources
            prompt a foundation segmentation model with each instance's visible
            keypoints + a keypoint box, then quality-filter the result; the
            ``hybrid*`` variants fall back to the dilated-skeleton mask per
            instance whenever the model mask is rejected (so every instance
            keeps GT), while the plain ``sam``/``sam3`` variants drop rejected
            instances. ``sam``/``hybrid`` use SAM1 (``segment-anything`` +
            checkpoint); ``sam3``/``hybrid_sam3`` use SAM3 (``transformers``,
            gated ``facebook/sam3``).
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM1 checkpoint (required for sam/hybrid).
        sam_model_type: SAM1 model registry key.
        sam3_model_id: HF model id for the SAM3 path (sam3/hybrid_sam3).
        device: Torch device for the segmentation model.
        pred_iou_min: Quality filter minimum predicted-IoU. ``None`` (default)
            selects the source-appropriate floor: ``PRED_IOU_MIN`` (0.88) for
            SAM1 and ``SAM3_PRED_IOU_MIN`` (0.5) for SAM3 -- SAM3's predicted-IoU
            is on a lower scale, so SAM1's 0.88 is recalibrated to keep mask
            quality comparable (see ``SAM3_PRED_IOU_MIN``).
        own_containment_min: Quality filter: minimum own-keypoint containment.
        area_ratio_range: Quality filter: ``(min, max)`` for model/skeleton area.

    Returns:
        New ``sio.Labels`` with per-frame ``UserSegmentationMask`` objects.

    Raises:
        ValueError: If ``source`` is not one of the supported values.
        ImportError: If ``source`` needs SAM/SAM3 but its dependency is absent.
    """
    return PseudomaskGenerator(
        source=source,
        node_radius=node_radius,
        edge_thickness=edge_thickness,
        dilate_frac=dilate_frac,
        min_dilate=min_dilate,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        sam3_model_id=sam3_model_id,
        device=device,
        pred_iou_min=pred_iou_min,
        own_containment_min=own_containment_min,
        area_ratio_range=area_ratio_range,
    ).run(labels)


def make_pseudomasks_cli(
    src: str,
    out: str,
    source: str = "skeleton",
    *,
    node_radius: int = NODE_RADIUS,
    edge_thickness: int = EDGE_THICKNESS,
    dilate_frac: float = DILATE_FRAC,
    min_dilate: int = MIN_DILATE,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    sam3_model_id: str = SAM3_MODEL_ID,
    device: str = "cuda",
    overlay: Optional[str] = None,
) -> sio.Labels:
    """Load ``src``, generate ``source`` pseudomasks, save embedded to ``out``.

    Args:
        src: Path to a source pose ``.slp`` / ``.pkg.slp`` (with image data).
        out: Output ``.pkg.slp`` path (saved with embedded images).
        source: GT source: ``"skeleton"`` (default), ``"sam"``, ``"hybrid"``,
            ``"sam3"``, ``"hybrid_sam3"``.
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM1 checkpoint (required for sam/hybrid).
        sam_model_type: SAM1 model registry key.
        sam3_model_id: HF model id for the SAM3 path (sam3/hybrid_sam3).
        device: Torch device for the segmentation model.
        overlay: Optional path to write a colored mask overlay PNG of frame 0.

    Returns:
        The generated ``sio.Labels`` (also saved to ``out``).
    """
    src_path = Path(src).expanduser().as_posix()
    logger.info(f"Loading {src_path} ...")
    labels = sio.load_slp(src_path)

    out_labels = PseudomaskGenerator(
        source=source,
        node_radius=node_radius,
        edge_thickness=edge_thickness,
        dilate_frac=dilate_frac,
        min_dilate=min_dilate,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        sam3_model_id=sam3_model_id,
        device=device,
    ).run(labels)

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving embedded -> {out_path} ...")
    out_labels.save(out_path.as_posix(), embed=True)

    if overlay:
        _save_overlay(out_labels, Path(overlay).expanduser())

    return out_labels


def _save_overlay(labels: sio.Labels, path: Path) -> None:
    """Render image + colored per-instance mask overlay for the first frame."""
    if not labels.labeled_frames:
        logger.warning("No labeled frames; skipping overlay.")
        return
    lf = labels.labeled_frames[0]
    img = np.asarray(lf.image)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1).astype(np.float32)
    else:
        rgb = img.astype(np.float32)
    colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (255, 80, 255),
        (80, 255, 255),
    ]
    for i, m in enumerate(lf.masks):
        mm = np.asarray(m.data, dtype=bool)
        c = np.array(colors[i % len(colors)], dtype=np.float32)
        rgb[mm] = 0.5 * rgb[mm] + 0.5 * c
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), rgb[..., ::-1].astype(np.uint8))  # RGB->BGR
    logger.info(f"Overlay -> {path} ({len(lf.masks)} masks on frame {lf.frame_idx}).")
