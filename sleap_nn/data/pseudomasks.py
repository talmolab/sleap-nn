"""Pose -> per-instance segmentation pseudomask generation.

Bottom-up instance-segmentation models train on ``.slp`` files whose labeled
frames already carry per-instance ground-truth
``sio.UserSegmentationMask`` silhouettes. This module produces that GT from
pose keypoints in an explicit, offline prep step, writing a new ``.slp`` whose
frames carry both the original keypoint instances (retained for eval
frame-pairing) and the generated masks.

Three GT sources are supported, selected per dataset:

* ``"skeleton"`` (default, no extra deps): rasterize each instance's skeleton
  into a dilated boolean silhouette using only ``cv2`` + ``numpy``. Robust on
  every body plan; the silhouette is a coarse over-approximation of the animal.
* ``"sam"`` (optional): prompt Segment Anything (ViT-H by default) with each
  instance's visible keypoints to produce a tight, true silhouette, then apply
  a quality filter (predicted-IoU / own-keypoint-containment / area-ratio). SAM
  masks are truer on compact animals (e.g. mice) but degrade badly on tiny or
  elongated ones (e.g. flies), so the source must be a per-dataset choice.
* ``"hybrid"`` (optional): SAM with a per-instance dilated-skeleton fallback
  whenever the SAM mask fails the quality filter, so every instance still gets
  GT and the 1:1 instance<->mask pairing is preserved.

The SAM / hybrid paths require the optional ``segment-anything`` dependency
(``pip install "sleap-nn[sam]"``) plus a SAM checkpoint; both are imported /
loaded lazily so the default skeleton path stays dependency-free.
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

    Bundles every tunable of the skeleton/SAM/hybrid recipe as a field with the
    same defaults as the module-level constants (the single source of truth), so
    callers can construct one generator and reuse it. The free helper functions
    (:func:`rasterize_instance_mask`, :func:`_kpt_box`, :func:`_pick`,
    :func:`_disjointify`, :func:`_sam_instance_masks`, :func:`_own_containment`)
    remain available and receive the relevant fields when :meth:`run` calls them.

    Attributes:
        source: GT source, one of ``"skeleton"`` (default), ``"sam"``,
            ``"hybrid"``. ``"sam"``/``"hybrid"`` require ``segment-anything`` +
            a checkpoint.
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM checkpoint (required for sam/hybrid).
        sam_model_type: SAM model registry key.
        device: Torch device for SAM.
        pred_iou_min: SAM quality filter: minimum predicted-IoU.
        own_containment_min: SAM quality filter: minimum own-keypoint containment.
        area_ratio_range: SAM quality filter: ``(min, max)`` for SAM/skeleton area.
        sam_box_margin_frac: SAM keypoint-box margin as a fraction of the side.
        sam_box_margin_min: SAM keypoint-box minimum margin (px).
        sam_max_box_area_factor: Reject SAM candidates above this * box-area.
        clahe_clip_limit: CLAHE clip limit applied before prompting SAM.
        clahe_tile_grid: CLAHE tile grid applied before prompting SAM.
    """

    source: str = "skeleton"
    node_radius: int = NODE_RADIUS
    edge_thickness: int = EDGE_THICKNESS
    dilate_frac: float = DILATE_FRAC
    min_dilate: int = MIN_DILATE
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    device: str = "cuda"
    pred_iou_min: float = PRED_IOU_MIN
    own_containment_min: float = OWN_CONTAIN_MIN
    area_ratio_range: Tuple[float, float] = (AREA_RATIO_MIN, AREA_RATIO_MAX)
    sam_box_margin_frac: float = SAM_BOX_MARGIN_FRAC
    sam_box_margin_min: float = SAM_BOX_MARGIN_MIN
    sam_max_box_area_factor: float = SAM_MAX_BOX_AREA_FACTOR
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_tile_grid: Tuple[int, int] = field(default_factory=lambda: CLAHE_TILE_GRID)

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
            ImportError: If ``source`` needs SAM but ``segment-anything`` is
                absent.
        """
        source = self.source
        if source not in ("skeleton", "sam", "hybrid"):
            raise ValueError(
                f"Unknown pseudomask source {source!r}; expected one of "
                "'skeleton', 'sam', 'hybrid'."
            )

        skel = labels.skeletons[0]
        edge_inds = list(skel.edge_inds)
        skel_kw = dict(
            node_radius=self.node_radius,
            edge_thickness=self.edge_thickness,
            dilate_frac=self.dilate_frac,
            min_dilate=self.min_dilate,
        )

        predictor = None
        use_sam = source in ("sam", "hybrid")
        if use_sam:
            predictor = _load_sam_predictor(
                self.sam_checkpoint,
                model_type=self.sam_model_type,
                device=self.device,
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
                        and pred_ious[i] >= self.pred_iou_min
                        and oc >= self.own_containment_min
                        and area_min <= area_ratio <= area_max
                    )
                    if keep:
                        final_masks.append(mm)
                        n_sam += 1
                    elif source == "hybrid":
                        final_masks.append(sm)  # per-instance skeleton fallback
                        n_fallback += 1
                    else:  # source == "sam": drop the rejected instance entirely
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
            logger.info(
                f"SAM masks: {n_sam}; skeleton-fallback: {n_fallback}; "
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
    device: str = "cuda",
    pred_iou_min: float = PRED_IOU_MIN,
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
            ``"hybrid"``. ``"sam"``/``"hybrid"`` require ``segment-anything`` +
            a checkpoint.
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM checkpoint (required for sam/hybrid).
        sam_model_type: SAM model registry key.
        device: Torch device for SAM.
        pred_iou_min: SAM quality filter: minimum predicted-IoU.
        own_containment_min: SAM quality filter: minimum own-keypoint containment.
        area_ratio_range: SAM quality filter: ``(min, max)`` for SAM/skeleton area.

    Returns:
        New ``sio.Labels`` with per-frame ``UserSegmentationMask`` objects.

    Raises:
        ValueError: If ``source`` is not one of the supported values.
        ImportError: If ``source`` needs SAM but ``segment-anything`` is absent.
    """
    return PseudomaskGenerator(
        source=source,
        node_radius=node_radius,
        edge_thickness=edge_thickness,
        dilate_frac=dilate_frac,
        min_dilate=min_dilate,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
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
    device: str = "cuda",
    overlay: Optional[str] = None,
) -> sio.Labels:
    """Load ``src``, generate ``source`` pseudomasks, save embedded to ``out``.

    Args:
        src: Path to a source pose ``.slp`` / ``.pkg.slp`` (with image data).
        out: Output ``.pkg.slp`` path (saved with embedded images).
        source: GT source: ``"skeleton"`` (default), ``"sam"``, ``"hybrid"``.
        node_radius: Skeleton-rasterizer node disk radius (px).
        edge_thickness: Skeleton-rasterizer edge line thickness (px).
        dilate_frac: Skeleton-rasterizer dilation as a fraction of bbox diagonal.
        min_dilate: Skeleton-rasterizer minimum dilation radius (px).
        sam_checkpoint: Path to the SAM checkpoint (required for sam/hybrid).
        sam_model_type: SAM model registry key.
        device: Torch device for SAM.
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
