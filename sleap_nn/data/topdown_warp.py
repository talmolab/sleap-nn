"""Single-warp centered-instance data path (``preprocessing.crop_pipeline="warp"``).

The stock :class:`~sleap_nn.data.custom_datasets.CenteredInstanceDataset` decodes a
frame, size-matches it to ``max_hw`` (resize + pad of the whole frame), cuts an
oversized pre-crop, augments it, re-crops it and renders float32 confidence maps in
the worker. For large multi-dataset training (hundreds of thousands of people from
frames of many sizes) that costs ~8 ms of CPU per sample and ~70 MB of float32 per
256-sample batch through the worker -> main-process pipe.

This module does the same job in one resample:

1. :func:`build_instance_table` flattens the labels into numpy arrays once (image
   path, box, keypoints on the model's node list, per-node loss weight, labels
   index). Workers never touch ``sio.Labels`` objects (unless a video is not a list
   of image files, in which case frames are read through the labels as usual).
2. :class:`WarpCenteredInstanceDataset` reads the frame and builds ONE affine
   (box window -> ``crop_size`` crop, with the rotation / scale / translation / flip
   augmentation folded in) and calls ``cv2.warpAffine`` from the full frame straight
   to the output crop: no size matcher, no pre-crop, no full-frame copies.
   Without augmentation the affine is exactly the box-scaled window of
   :func:`~sleap_nn.data.instance_cropping.bbox_crop_windows`, i.e. the same pixels
   the top-down inference layer cuts with
   :func:`~sleap_nn.data.instance_cropping.crop_windows`.
3. Samples are uint8 crops + crop keypoints (+ per-node weights when they can
   differ from 1). With ``gpu_confmaps`` the confidence maps are rendered on the
   GPU in the training step (:func:`render_confmaps`, the same ``make_confmaps``
   math), so no float maps cross the process boundary. For distillation with a
   ``teacher_crop_size``, training samples also carry ``teacher_image``: the same
   window and augmentation draw cut at the teacher's resolution from the same
   decoded frame (a second warp, no second decode).
4. :class:`BalancedBatchSampler` composes every batch from fixed per-file counts
   (``labels_idx``), with an iteration-based epoch length, per-rank streams and a
   resumable position (:class:`BalancedDataLoader`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sleap_io as sio
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from sleap_nn.data.bboxes import (
    bbox_category_name,
    frame_user_bboxes,
    resolve_bbox_source,
)
from sleap_nn.data.confidence_maps import make_confmaps
from sleap_nn.data.instance_cropping import (
    BBoxCropSpec,
    bbox_crop_windows,
    crop_hw,
    window_corners,
)
from sleap_nn.data.node_weights import (
    UNLABELED_POINTS_MODES,
    get_labeled_nodes_weights,
    instance_node_weights,
)
from sleap_nn.data.utils import make_grid_vectors

try:  # cv2 is a hard dependency of the warp path only.
    import cv2

    cv2.setNumThreads(1)
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Instance table
# ---------------------------------------------------------------------------


@dataclass
class InstanceFilter:
    """Train-time instance filters (``data_config.instance_filter``).

    Attributes:
        min_labeled_nodes: Keep an instance only if at least this many of
            ``count_nodes`` have a location (and are labeled by its file).
        count_nodes: Node names counted by ``min_labeled_nodes``; ``None`` counts
            every model node (e.g. the 12 COCO limb joints for "6 of 12 limbs").
        min_bbox_size: Keep an instance only if ``sqrt(box area)`` in frame pixels is
            at least this.
    """

    min_labeled_nodes: int = 0
    count_nodes: Optional[List[str]] = None
    min_bbox_size: float = 0.0

    @classmethod
    def from_config(cls, cfg) -> Optional["InstanceFilter"]:
        """Build from a ``data_config.instance_filter`` DictConfig (or ``None``)."""
        if cfg is None:
            return None
        return cls(
            min_labeled_nodes=int(cfg.get("min_labeled_nodes", 0) or 0),
            count_nodes=(
                list(cfg.get("count_nodes")) if cfg.get("count_nodes") else None
            ),
            min_bbox_size=float(cfg.get("min_bbox_size", 0.0) or 0.0),
        )


@dataclass
class InstanceTable:
    """All training instances as flat arrays (cheap to fork/pickle to workers)."""

    labels_idx: np.ndarray  # (n,) int16
    lf_idx: np.ndarray  # (n,) int32
    video_idx: np.ndarray  # (n,) int32
    frame_idx: np.ndarray  # (n,) int32
    path_id: np.ndarray  # (n,) int32 into `paths`, -1 = read through the labels
    paths: np.ndarray  # (n_images,) bytes
    box: np.ndarray  # (n, 4) float32 x1, y1, x2, y2 in frame px
    kpts: np.ndarray  # (n, K, 2) float32, NaN = no location
    weight: np.ndarray  # (n, K) float32: 0 = node not labeled by this file
    num_instances: np.ndarray  # (n,) int16 user instances in the frame
    filtered: Dict[str, int] = field(default_factory=dict)
    # Crop box origin counts: "gt" and, with detector boxes, "detector".
    box_sources: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of samples (or batches)."""
        return len(self.labels_idx)


def _image_path(video: sio.Video, frame_idx: int) -> Optional[str]:
    """The image file of a frame when the video is a list of images, else None."""
    fn = video.filename
    if isinstance(fn, (list, tuple)):
        return fn[frame_idx]
    if isinstance(fn, str) and fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return fn
    return None


def _linked_detector_boxes(
    lf: sio.LabeledFrame,
    categories: Optional[set],
    detector: Optional[str],
    min_score: float,
) -> Dict[int, sio.BoundingBox]:
    """Predicted boxes of a frame linked to an instance: ``{id(instance): box}``.

    Only non-empty boxes of ``categories`` (any if ``None``), from ``detector`` (box
    ``source``; any if ``None``) and scoring at least ``min_score`` count; when
    several are linked to one instance the highest-scoring one wins.
    """
    out: Dict[int, sio.BoundingBox] = {}
    for b in getattr(lf, "bboxes", None) or []:
        inst = getattr(b, "instance", None)
        if (
            inst is None
            or not b.is_predicted
            or b.is_empty
            or b.width <= 0
            or b.height <= 0
        ):
            continue
        if categories is not None and bbox_category_name(b) not in categories:
            continue
        if detector is not None and b.source != detector:
            continue
        score = float(getattr(b, "score", 0.0) or 0.0)
        if score < min_score:
            continue
        prev = out.get(id(inst))
        if prev is None or score > float(getattr(prev, "score", 0.0) or 0.0):
            out[id(inst)] = b
    return out


def build_instance_table(
    labels_list: Sequence[sio.Labels],
    node_names: Sequence[str],
    bbox_source: str = "auto",
    bbox_padding: float = 0.0,
    bbox_categories: Optional[Sequence[str]] = None,
    instance_filter: Optional[InstanceFilter] = None,
    user_instances_only: bool = True,
    supervise_occluded: bool = True,
    crop_bbox_source: str = "gt",
    crop_bbox_detector: Optional[str] = None,
    crop_bbox_min_score: float = 0.0,
    crop_bbox_fallback: str = "gt",
) -> InstanceTable:
    """Flatten labels into an :class:`InstanceTable` on the model's node list.

    Nodes are matched by name, so files with different skeletons (COCO-17, AIC-14,
    a union skeleton) can train one model; a model node missing from a file gets
    weight 0 for that file's instances. Boxes follow
    :func:`~sleap_nn.data.bboxes.instance_crop_bbox` (``bboxes``: the user box
    linked to the instance; else the tight keypoint box + ``bbox_padding``), except
    that a linked box whose category is not in ``bbox_categories`` (e.g. MPII's
    *head* boxes) is ignored. Per-node weights combine the file's
    ``provenance["labeled_nodes"]`` and the points' own ``labeled`` flags (when
    sleap-io provides them) exactly as the stock datasets do
    (:func:`~sleap_nn.data.node_weights.instance_node_weights`).

    With ``crop_bbox_source="detector"`` the crop box is instead the predicted box
    linked to the instance (a detector box matched to the ground truth; filtered by
    ``bbox_categories``, ``crop_bbox_detector`` and ``crop_bbox_min_score``), and an
    instance without one falls back to the box above (``crop_bbox_fallback="gt"``)
    or is dropped (``"drop"``).

    Raises:
        ValueError: If ``bbox_source`` resolves to ``masks`` (not supported here),
            ``crop_bbox_source`` / ``crop_bbox_fallback`` is unknown, or no instance
            is left.
    """
    if crop_bbox_source not in ("gt", "detector"):
        raise ValueError(
            f"crop_bbox_source must be 'gt' or 'detector', got {crop_bbox_source!r}."
        )
    if crop_bbox_fallback not in ("gt", "drop"):
        raise ValueError(
            f"crop_bbox_fallback must be 'gt' or 'drop', got {crop_bbox_fallback!r}."
        )
    use_det = crop_bbox_source == "detector"
    K = len(node_names)
    count_idx = None
    if instance_filter is not None and instance_filter.count_nodes:
        count_idx = np.array(
            [
                node_names.index(n)
                for n in instance_filter.count_nodes
                if n in node_names
            ]
        )
    cats = set(bbox_categories) if bbox_categories else None
    paths: List[str] = []
    path_ids: Dict[str, int] = {}
    cols = {
        k: []
        for k in (
            "labels_idx",
            "lf_idx",
            "video_idx",
            "frame_idx",
            "path_id",
            "box",
            "kpts",
            "weight",
            "num_instances",
        )
    }
    n_small = n_few = n_empty = n_no_det = 0
    box_sources = {"gt": 0, "detector": 0} if use_det else {"gt": 0}
    for li, labels in enumerate(labels_list):
        sk_names = labels.skeletons[0].node_names
        src = np.array([sk_names.index(n) if n in sk_names else -1 for n in node_names])
        has = src >= 0
        w_sk = get_labeled_nodes_weights(labels, sk_names)
        source = resolve_bbox_source(labels, bbox_source)
        if source == "masks":
            raise ValueError(
                "crop_pipeline='warp' takes crop boxes from user boxes or keypoints; "
                "bbox_source='masks' is not supported."
            )
        video_index = {id(v): i for i, v in enumerate(labels.videos)}
        for lfi, lf in enumerate(labels.labeled_frames):
            insts = lf.user_instances if user_instances_only else lf.instances
            if not insts:
                continue
            linked = {}
            if source == "bboxes":
                for b in frame_user_bboxes(lf):
                    inst = getattr(b, "instance", None)
                    if inst is not None and (
                        cats is None or bbox_category_name(b) in cats
                    ):
                        linked[id(inst)] = b
            det_linked = (
                _linked_detector_boxes(
                    lf, cats, crop_bbox_detector, float(crop_bbox_min_score)
                )
                if use_det
                else {}
            )
            path = _image_path(lf.video, lf.frame_idx)
            pid = -1
            if path is not None:
                pid = path_ids.setdefault(path, len(paths))
                if pid == len(paths):
                    paths.append(path)
            for inst in insts:
                w_inst, pts = instance_node_weights(
                    inst, w_sk, inst.numpy(), supervise_occluded=supervise_occluded
                )
                k = np.full((K, 2), np.nan, np.float32)
                k[has] = pts[src[has]]
                w_file = np.zeros(K, np.float32)
                w_file[has] = w_inst[src[has]]
                k[w_file == 0] = np.nan
                vis = np.isfinite(k).all(-1)
                if not vis.any():
                    n_empty += 1
                    continue
                b = det_linked.get(id(inst))
                if b is None and use_det and crop_bbox_fallback == "drop":
                    n_no_det += 1
                    continue
                box_from = "detector" if b is not None else "gt"
                if b is None:
                    b = linked.get(id(inst))
                if b is not None and not b.is_empty:
                    box = np.asarray(b.bounds if b.is_rotated else b.xyxy, np.float32)
                else:
                    f = k[vis]
                    box = np.concatenate(
                        [f.min(0) - bbox_padding, f.max(0) + bbox_padding]
                    )
                if instance_filter is not None:
                    n_lab = vis[count_idx].sum() if count_idx is not None else vis.sum()
                    if n_lab < instance_filter.min_labeled_nodes:
                        n_few += 1
                        continue
                    area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
                    if math.sqrt(area) < instance_filter.min_bbox_size:
                        n_small += 1
                        continue
                box_sources[box_from] += 1
                cols["labels_idx"].append(li)
                cols["lf_idx"].append(lfi)
                cols["video_idx"].append(video_index[id(lf.video)])
                cols["frame_idx"].append(lf.frame_idx)
                cols["path_id"].append(pid)
                cols["box"].append(box)
                cols["kpts"].append(k)
                cols["weight"].append(w_file)
                cols["num_instances"].append(len(insts))
    if not cols["labels_idx"]:
        raise ValueError("No training instances left after filtering.")
    table = InstanceTable(
        labels_idx=np.asarray(cols["labels_idx"], np.int16),
        lf_idx=np.asarray(cols["lf_idx"], np.int32),
        video_idx=np.asarray(cols["video_idx"], np.int32),
        frame_idx=np.asarray(cols["frame_idx"], np.int32),
        path_id=np.asarray(cols["path_id"], np.int32),
        # Fixed-width bytes: no per-string Python objects for workers to refcount.
        paths=np.array([p.encode() for p in paths], dtype=bytes),
        box=np.stack(cols["box"]).astype(np.float32),
        kpts=np.stack(cols["kpts"]),
        weight=np.stack(cols["weight"]),
        num_instances=np.asarray(cols["num_instances"], np.int16),
        filtered={
            "no_visible_node": n_empty,
            "few_labeled_nodes": n_few,
            "small_box": n_small,
            **({"no_detector_box": n_no_det} if use_det else {}),
        },
        box_sources=box_sources,
    )
    return table


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

#: Batch key of distillation's dual-resolution crop: the sample's window at the
#: teacher's resolution (``teacher_crop_size``), read by
#: :class:`~sleap_nn.training.distillation.Distiller`.
TEACHER_IMAGE_KEY = "teacher_image"

#: ``geometric.affine_sampling`` values (see :class:`WarpAug`).
AFFINE_SAMPLINGS = ("uniform", "mmpose")


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def window_affine(window: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """3x3 frame -> crop affine of a box-scaled window, in keypoint coordinates.

    Keypoints put pixel centers at integer coordinates and windows are in pixel-edge
    coordinates, so this is
    :func:`~sleap_nn.data.instance_cropping.points_to_crop` as a matrix:
    ``c = (p + 0.5 - window[:2]) * scale - 0.5``. ``cv2.warpAffine`` with it samples
    the frame where ``crop_windows`` does.
    """
    sx, sy = float(scale[0]), float(scale[1])
    return (
        _translate(-0.5, -0.5)
        @ np.diag([sx, sy, 1.0])
        @ _translate(0.5 - float(window[0]), 0.5 - float(window[1]))
    )


#: Substrings of node names that :class:`WarpAug` half-body treats as lower body
#: when ``half_body_lower_nodes`` is not given (COCO / AIC / Halpe style names).
LOWER_BODY_KEYWORDS = ("hip", "knee", "ankle", "heel", "toe", "foot")


def _truncnorm(rng: np.random.Generator) -> float:
    """One draw of a standard normal truncated to ``[-1, 1]`` (rejection)."""
    while True:
        v = rng.standard_normal()
        if -1.0 <= v <= 1.0:
            return float(v)


def _odd_kernel(rng: np.random.Generator, lo: int, hi: int) -> int:
    """A random odd kernel size in ``[lo, hi]`` (albumentations ``blur_limit``)."""
    choices = [k for k in range(int(lo), int(hi) + 1) if k % 2 == 1 and k >= 3]
    return int(rng.choice(choices)) if choices else 3


def center_size_window(
    center: np.ndarray,
    size: np.ndarray,
    crop_hw: Tuple[int, int],
    aspect: str = "square",
    min_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window + scale of a ``size`` box at ``center``, fit to the crop's aspect.

    The numpy, padding-free counterpart of
    :func:`~sleap_nn.data.instance_cropping.bbox_crop_windows` (same aspect fit and
    ``min_size`` floor) for a box given as a center and a ``(w, h)`` size in
    keypoint coordinates -- what mmpose's ``TopdownAffine`` does with the
    half-body ``bbox_center`` / ``bbox_scale``.

    Args:
        center: ``(2,)`` box center (keypoint coordinates).
        size: ``(2,)`` box ``(width, height)``.
        crop_hw: Crop ``(height, width)``.
        aspect: ``"stretch"`` (one scale per axis) or anything else (one scale).
        min_size: Smallest window side in frame pixels, or ``None``.

    Returns:
        ``(window, scale)``: ``(4,)`` edge-coordinate window and ``(2,)`` scale.
    """
    ch, cw = crop_hw
    out_wh = np.array([cw, ch], np.float64)
    floor = max(float(min_size or 0.0), 1.0)
    padded = np.maximum(np.asarray(size, np.float64), 1e-6)
    if aspect == "stretch":
        sides = np.maximum(padded, floor)
    else:
        s = min((out_wh / padded).min(), min(ch, cw) / floor)
        sides = out_wh / s
    c = np.asarray(center, np.float64) + 0.5
    window = np.concatenate([c - sides / 2, c + sides / 2]).astype(np.float32)
    return window, (out_wh / sides).astype(np.float32)


@dataclass
class WarpAug:
    """Warp-path augmentation: geometry in the crop affine, photometry on the crop.

    Geometry (:meth:`sample`): rotation and scale act about the crop center;
    translation shifts the window by ``translate_*`` times the padded box's width /
    height (mmpose's ``RandomBBoxTransform`` shift), not the crop's; ``flip_p``
    mirrors the crop and swaps left/right nodes through ``flip_idx``. With
    ``affine_sampling="uniform"`` (default) the draws are uniform, ``scale > 1``
    zooms in and the shift is applied after rotation / scale, as in the stock path.
    With ``"mmpose"`` they follow ``RandomBBoxTransform``: truncated-normal draws
    (``[-1, 1]`` sigma) around the middle of each range, ``scale_*`` is the *box*
    scale (content scale = 1 / box scale) and the shift moves the box before it is
    rotated and scaled, so an mmpose config's numbers can be used as they are.

    Half-body (:meth:`half_body_window`, mmpose ``RandomHalfBody``): with
    ``half_body_p``, an instance with at least ``half_body_min_total`` visible
    nodes is cropped to its upper or lower body (upper preferred with
    ``half_body_upper_prob``; each half needs ``half_body_min_upper`` /
    ``half_body_min_lower`` visible nodes): the box is centered on the mean of the
    chosen nodes, ``half_body_padding`` times their extent, fit to the crop aspect.
    It replaces the crop window, so it costs nothing extra (still one warp).

    Photometric (:meth:`intensity`, on the uint8 RGB crop, in this order):
    contrast / brightness (contrast pivots on 127.5, as in the stock path), HSV
    shift (mmdet ``YOLOXHSVRandomAug``), box blur (albumentations ``Blur``), median
    blur, Gaussian blur, uniform / Gaussian noise, coarse dropout (albumentations
    ``CoarseDropout``, fill 0) and random erase (random-colour box, as in the stock
    pipeline). Holes and erases leave the keypoint targets untouched (as mmpose /
    albumentations do): the model learns to infer occluded joints.
    """

    rotation: Tuple[float, float] = (0.0, 0.0)
    rotation_p: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)
    scale_p: float = 0.0
    translate: Tuple[float, float] = (0.0, 0.0)
    translate_p: float = 0.0
    flip_p: float = 0.0
    affine_sampling: str = "uniform"
    # Half-body.
    half_body_p: float = 0.0
    half_body_min_total: int = 9
    half_body_min_upper: int = 2
    half_body_min_lower: int = 3
    half_body_padding: float = 1.5
    half_body_upper_prob: float = 0.7
    half_body_lower_nodes: Optional[List[str]] = None
    # Photometric.
    contrast: Tuple[float, float] = (1.0, 1.0)
    contrast_p: float = 0.0
    brightness: Tuple[float, float] = (1.0, 1.0)
    brightness_p: float = 0.0
    hsv_delta: Tuple[float, float, float] = (5.0, 30.0, 30.0)
    hsv_p: float = 0.0
    blur_kernel: Tuple[int, int] = (3, 7)
    blur_p: float = 0.0
    median_blur_kernel: Tuple[int, int] = (3, 7)
    median_blur_p: float = 0.0
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    gaussian_blur_p: float = 0.0
    uniform_noise: Tuple[float, float] = (0.0, 0.04)  # 0-1 scale
    uniform_noise_p: float = 0.0
    gaussian_noise: Tuple[float, float] = (0.0, 0.0)  # mean, std (0-1 scale)
    gaussian_noise_p: float = 0.0
    coarse_dropout_holes: Tuple[int, int] = (1, 1)
    coarse_dropout_size: Tuple[float, float] = (0.2, 0.4)  # fraction of crop h / w
    coarse_dropout_p: float = 0.0
    erase_scale: Tuple[float, float] = (0.0001, 0.01)
    erase_ratio: Tuple[float, float] = (1.0, 1.0)
    erase_p: float = 0.0
    # Bound to the model's nodes by :meth:`bind_nodes`.
    upper_idx: Optional[np.ndarray] = field(default=None, repr=False)
    lower_idx: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_configs(cls, geometric, intensity) -> "WarpAug":
        """From the ``augmentation_config.geometric`` / ``.intensity`` sections.

        Raises:
            ValueError: On ``mixup_p > 0`` (not implemented; it is not silently
                ignored) or an unknown ``affine_sampling``.
        """
        a = cls()
        if geometric is not None:
            g = dict(geometric)
            affine_p = float(g.get("affine_p", 0.0) or 0.0)

            def p(key):
                v = g.get(key)
                return affine_p if v is None else float(v)

            def f(key, default):
                v = g.get(key)
                return default if v is None else v

            a.rotation = (
                float(f("rotation_min", 0.0)),
                float(f("rotation_max", 0.0)),
            )
            a.rotation_p = p("rotation_p")
            a.scale = (float(f("scale_min", 1.0)), float(f("scale_max", 1.0)))
            a.scale_p = p("scale_p")
            a.translate = (
                float(f("translate_width", 0.0)),
                float(f("translate_height", 0.0)),
            )
            a.translate_p = p("translate_p")
            a.flip_p = float(f("flip_p", 0.0))
            a.affine_sampling = str(f("affine_sampling", "uniform"))
            if a.affine_sampling not in AFFINE_SAMPLINGS:
                raise ValueError(
                    f"affine_sampling must be one of {AFFINE_SAMPLINGS}, got "
                    f"{a.affine_sampling!r}."
                )
            a.half_body_p = float(f("half_body_p", 0.0))
            a.half_body_min_total = int(f("half_body_min_total_keypoints", 9))
            a.half_body_min_upper = int(f("half_body_min_upper_keypoints", 2))
            a.half_body_min_lower = int(f("half_body_min_lower_keypoints", 3))
            a.half_body_padding = float(f("half_body_padding", 1.5))
            a.half_body_upper_prob = float(f("half_body_upper_prob", 0.7))
            lower = g.get("half_body_lower_nodes")
            a.half_body_lower_nodes = list(lower) if lower else None
            a.coarse_dropout_p = float(f("coarse_dropout_p", 0.0))
            a.coarse_dropout_holes = (
                int(f("coarse_dropout_min_holes", 1)),
                int(f("coarse_dropout_max_holes", 1)),
            )
            a.coarse_dropout_size = (
                float(f("coarse_dropout_min_size", 0.2)),
                float(f("coarse_dropout_max_size", 0.4)),
            )
            a.erase_p = float(f("erase_p", 0.0))
            a.erase_scale = (
                float(f("erase_scale_min", 0.0001)),
                float(f("erase_scale_max", 0.01)),
            )
            a.erase_ratio = (
                float(f("erase_ratio_min", 1.0)),
                float(f("erase_ratio_max", 1.0)),
            )
            if float(f("mixup_p", 0.0)) > 0:
                raise ValueError(
                    "crop_pipeline='warp' does not implement mixup (mixup_p > 0). "
                    "Neither ViTPose nor RTMPose uses it; set mixup_p: 0."
                )
        if intensity is not None:
            i = dict(intensity)

            def fi(key, default):
                v = i.get(key)
                return default if v is None else v

            a.contrast = (
                float(fi("contrast_min", 1.0)),
                float(fi("contrast_max", 1.0)),
            )
            a.contrast_p = float(fi("contrast_p", 0.0))
            a.brightness = (
                float(fi("brightness_min", 1.0)),
                float(fi("brightness_max", 1.0)),
            )
            a.brightness_p = float(fi("brightness_p", 0.0))
            a.hsv_p = float(fi("hsv_p", 0.0))
            a.hsv_delta = (
                float(fi("hsv_hue_delta", 5.0)),
                float(fi("hsv_saturation_delta", 30.0)),
                float(fi("hsv_value_delta", 30.0)),
            )
            a.blur_p = float(fi("blur_p", 0.0))
            a.blur_kernel = (
                int(fi("blur_kernel_min", 3)),
                int(fi("blur_kernel_max", 7)),
            )
            a.median_blur_p = float(fi("median_blur_p", 0.0))
            a.median_blur_kernel = (
                int(fi("median_blur_kernel_min", 3)),
                int(fi("median_blur_kernel_max", 7)),
            )
            a.gaussian_blur_p = float(fi("gaussian_blur_p", 0.0))
            a.gaussian_blur_sigma = (
                float(fi("gaussian_blur_sigma_min", 0.1)),
                float(fi("gaussian_blur_sigma_max", 2.0)),
            )
            a.uniform_noise_p = float(fi("uniform_noise_p", 0.0))
            a.uniform_noise = (
                float(fi("uniform_noise_min", 0.0)),
                float(fi("uniform_noise_max", 0.04)),
            )
            a.gaussian_noise = (
                float(fi("gaussian_noise_mean", 0.0)),
                float(fi("gaussian_noise_std", 0.0)),
            )
            a.gaussian_noise_p = float(fi("gaussian_noise_p", 0.0))
        return a

    def bind_nodes(self, node_names: Sequence[str]) -> "WarpAug":
        """Resolve the half-body upper / lower node indices on the model's nodes."""
        if self.half_body_lower_nodes:
            lower_set = set(self.half_body_lower_nodes)
            lower = [n in lower_set for n in node_names]
        else:
            lower = [
                any(k in n.lower() for k in LOWER_BODY_KEYWORDS) for n in node_names
            ]
        lower = np.asarray(lower, bool)
        self.lower_idx = np.flatnonzero(lower)
        self.upper_idx = np.flatnonzero(~lower)
        if self.half_body_p > 0 and (
            not len(self.lower_idx) or not len(self.upper_idx)
        ):
            logger.warning(
                "half_body_p > 0 but the nodes do not split into an upper and a lower "
                f"body (lower: {[node_names[j] for j in self.lower_idx]}); set "
                "half_body_lower_nodes."
            )
        return self

    def _draw(self, rng: np.random.Generator, lo: float, hi: float) -> float:
        if self.affine_sampling == "mmpose":
            return (lo + hi) / 2.0 + _truncnorm(rng) * (hi - lo) / 2.0
        return float(rng.uniform(lo, hi))

    def sample(
        self,
        rng: np.random.Generator,
        crop_h: int,
        crop_w: int,
        box_wh: Optional[Sequence[float]] = None,
    ):
        """Return ``(A, flipped)``: a 3x3 crop -> crop affine and the flip flag.

        Args:
            rng: The worker's generator.
            crop_h: Crop height.
            crop_w: Crop width.
            box_wh: The padded box's ``(width, height)`` in crop pixels, the unit of
                the translation; ``None`` uses the crop size.
        """
        cx, cy = (crop_w - 1) / 2.0, (crop_h - 1) / 2.0
        theta, s, tx, ty = 0.0, 1.0, 0.0, 0.0
        mm = self.affine_sampling == "mmpose"
        if self.rotation_p > 0 and rng.random() < self.rotation_p:
            theta = math.radians(self._draw(rng, *self.rotation))
        if self.scale_p > 0 and rng.random() < self.scale_p:
            s = self._draw(rng, *self.scale)
            if mm:  # mmpose scales the box: content scale is its inverse.
                s = 1.0 / max(s, 1e-6)
        if self.translate_p > 0 and rng.random() < self.translate_p:
            bw, bh = (crop_w, crop_h) if box_wh is None else box_wh
            tx = self._draw(rng, -1.0, 1.0) * self.translate[0] * float(bw)
            ty = self._draw(rng, -1.0, 1.0) * self.translate[1] * float(bh)
        c, si = math.cos(theta) * s, math.sin(theta) * s
        rs = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        if mm:
            # The box center moves first; the crop is then cut around it.
            A = _translate(cx, cy) @ rs @ _translate(-cx - tx, -cy - ty)
        else:
            A = _translate(cx + tx, cy + ty) @ rs @ _translate(-cx, -cy)
        flipped = self.flip_p > 0 and rng.random() < self.flip_p
        if flipped:
            A = np.array([[-1.0, 0.0, crop_w - 1.0], [0.0, 1.0, 0.0], [0, 0, 1.0]]) @ A
        return A, flipped

    def half_body_nodes(
        self, rng: np.random.Generator, visible: np.ndarray
    ) -> Optional[np.ndarray]:
        """Select half-body nodes (mmpose ``RandomHalfBody``); None = no half-body."""
        if self.half_body_p <= 0 or self.upper_idx is None:
            return None
        if visible.sum() < self.half_body_min_total:
            return None
        if rng.random() > self.half_body_p:
            return None
        up = self.upper_idx[visible[self.upper_idx]]
        lo = self.lower_idx[visible[self.lower_idx]]
        prefer_upper = rng.random() < self.half_body_upper_prob
        ok_up = len(up) >= self.half_body_min_upper
        ok_lo = len(lo) >= self.half_body_min_lower
        if not ok_up and not ok_lo:
            return None
        if not ok_lo:
            return up
        if not ok_up:
            return lo
        return up if prefer_upper else lo

    def half_body_window(
        self,
        rng: np.random.Generator,
        kpts: np.ndarray,
        visible: np.ndarray,
        crop_hw: Tuple[int, int],
        aspect: str = "square",
        min_size: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The half-body window of an instance, or None (no half-body).

        Returns:
            ``(window, scale, box_wh)``: the window and scale as
            :func:`center_size_window` returns them, and the half-body box size in
            crop pixels (the unit of the shift).
        """
        idx = self.half_body_nodes(rng, visible)
        if idx is None:
            return None
        pts = kpts[idx].astype(np.float64)
        size = (pts.max(0) - pts.min(0)) * self.half_body_padding
        window, scale = center_size_window(pts.mean(0), size, crop_hw, aspect, min_size)
        return window, scale, (np.maximum(size, 1e-6) * scale).astype(np.float32)

    def intensity(
        self,
        crop: np.ndarray,
        rng: np.random.Generator,
        teacher: Optional[np.ndarray] = None,
        teacher_rng: Optional[np.random.Generator] = None,
    ):
        """Photometric ops on a uint8 ``(H, W, C)`` RGB (or gray) crop.

        Args:
            crop: The crop.
            rng: The worker's generator. Every draw that decides an op comes from
                it.
            teacher: Optional second crop of the same window at another resolution
                (the distillation teacher's, see
                :class:`WarpCenteredInstanceDataset` ``teacher_crop_size``). It gets
                the same draws, with the spatial ones mapped to its pixel grid: blur
                kernels and sigma scale with the size ratio, and dropout holes and
                the erase box cover the same part of the window. Noise is drawn per
                pixel at its own resolution (same strength) from ``teacher_rng``,
                so ``crop`` and ``rng``'s stream are what they are without it.
            teacher_rng: Generator of the teacher's noise; ``None`` uses ``rng``.

        Returns:
            The augmented crop, or ``(crop, teacher)`` when ``teacher`` is given.
        """
        imgs = [crop] if teacher is None else [crop, teacher]
        gens = [rng] if teacher is None else [rng, teacher_rng or rng]
        H, W = crop.shape[:2]
        # (y, x) pixel-grid ratio of each image to `crop`.
        ratios = [(im.shape[0] / H, im.shape[1] / W) for im in imgs]
        alpha, beta = 1.0, 0.0
        if self.contrast_p > 0 and rng.random() < self.contrast_p:
            # About mid-grey, like the stock path (not the crop mean).
            c = rng.uniform(*self.contrast)
            alpha, beta = c, (1.0 - c) * 127.5
        if self.brightness_p > 0 and rng.random() < self.brightness_p:
            b = rng.uniform(*self.brightness)
            alpha, beta = alpha * b, beta * b
        if alpha != 1.0 or beta != 0.0:
            # A clipped lookup table (cv2.convertScaleAbs would fold values below
            # 0 back up through abs()).
            lut = np.clip(np.arange(256, dtype=np.float32) * alpha + beta, 0, 255)
            lut = np.rint(lut).astype(np.uint8)
            imgs = [cv2.LUT(im, lut) for im in imgs]
        imgs = [_hwc(im) for im in imgs]
        if self.hsv_p > 0 and imgs[0].shape[-1] == 3 and rng.random() < self.hsv_p:
            lut = self._hsv_lut(rng)
            if lut is not None:
                imgs = [_apply_hsv_lut(im, lut) for im in imgs]
        if self.blur_p > 0 and rng.random() < self.blur_p:
            k = _odd_kernel(rng, *self.blur_kernel)
            imgs = [
                cv2.blur(im, (_scaled_odd_kernel(k, rx), _scaled_odd_kernel(k, ry)))
                for im, (ry, rx) in zip(imgs, ratios)
            ]
        if self.median_blur_p > 0 and rng.random() < self.median_blur_p:
            k = _odd_kernel(rng, *self.median_blur_kernel)
            imgs = [
                cv2.medianBlur(
                    np.ascontiguousarray(im),
                    max(3, _scaled_odd_kernel(k, (ry + rx) / 2.0)),
                )
                for im, (ry, rx) in zip(imgs, ratios)
            ]
        if self.gaussian_blur_p > 0 and rng.random() < self.gaussian_blur_p:
            sigma = rng.uniform(*self.gaussian_blur_sigma)
            imgs = [
                cv2.GaussianBlur(im, (0, 0), sigma * rx, sigmaY=sigma * ry)
                for im, (ry, rx) in zip(imgs, ratios)
            ]
        imgs = [_hwc(im) for im in imgs]
        if self.uniform_noise_p > 0 and rng.random() < self.uniform_noise_p:
            lo, hi = int(self.uniform_noise[0] * 255), int(self.uniform_noise[1] * 255)
            out = []
            for im, g in zip(imgs, gens):
                noise = g.integers(lo, hi + 1, im.shape, dtype=np.int16)
                out.append(
                    np.clip(im.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                )
            imgs = out
        if self.gaussian_noise_p > 0 and rng.random() < self.gaussian_noise_p:
            m, sd = self.gaussian_noise
            out = []
            for im, g in zip(imgs, gens):
                noise = g.standard_normal(im.shape, dtype=np.float32)
                noise = noise * np.float32(sd * 255.0) + np.float32(m * 255.0)
                out.append(np.clip(im + noise, 0, 255).astype(np.uint8))
            imgs = out
        if self.coarse_dropout_p > 0 and rng.random() < self.coarse_dropout_p:
            rects = self._dropout_rects(H, W, rng)
            imgs = [_fill_rects(im, rects, r, 0) for im, r in zip(imgs, ratios)]
        if self.erase_p > 0 and rng.random() < self.erase_p:
            erase = self._erase_rect(H, W, imgs[0].shape[-1], rng)
            if erase is not None:
                rect, colour = erase
                imgs = [
                    _fill_rects(im, [rect], r, colour) for im, r in zip(imgs, ratios)
                ]
        return imgs[0] if teacher is None else (imgs[0], imgs[1])

    def _hsv_lut(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        # mmdet YOLOXHSVRandomAug: gains in [-delta, delta], each of h/s/v kept with
        # p=0.5, truncated to int; hue wraps mod 180 (OpenCV 8-bit hue), s/v clip.
        # Done as one 3-channel LUT on the HSV image (None: all gains are 0).
        gains = rng.uniform(-1, 1, 3) * np.asarray(self.hsv_delta)
        gains = (gains * rng.integers(0, 2, 3)).astype(np.int16)
        if not gains.any():
            return None
        x = np.arange(256, dtype=np.int16)
        return np.stack(
            [
                (x + gains[0]) % 180,
                np.clip(x + gains[1], 0, 255),
                np.clip(x + gains[2], 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)[:, None, :]

    def _dropout_rects(
        self, H: int, W: int, rng: np.random.Generator
    ) -> List[Tuple[int, int, int, int]]:
        # albumentations CoarseDropout with fractional sizes: integer hole sides in
        # [min * side, max * side], uniformly placed. Holes are (y, x, h, w).
        lo, hi = self.coarse_dropout_size
        n_holes = int(
            rng.integers(self.coarse_dropout_holes[0], self.coarse_dropout_holes[1] + 1)
        )
        rects = []
        for _ in range(n_holes):
            h = int(rng.integers(int(lo * H), int(hi * H) + 1))
            w = int(rng.integers(int(lo * W), int(hi * W) + 1))
            y = int(rng.integers(0, max(H - h, 0) + 1))
            x = int(rng.integers(0, max(W - w, 0) + 1))
            rects.append((y, x, h, w))
        return rects

    def _erase_rect(self, H: int, W: int, C: int, rng: np.random.Generator):
        # Stock `_apply_random_erase`: area fraction + aspect ratio, random colour.
        # Returns ((y, x, h, w), colour), or None when the box does not fit.
        area = rng.uniform(*self.erase_scale) * H * W
        ratio = rng.uniform(*self.erase_ratio)
        eh, ew = int(math.sqrt(area * ratio)), int(math.sqrt(area / ratio))
        if eh >= H or ew >= W or eh < 1 or ew < 1:
            return None
        y, x = int(rng.integers(0, H - eh)), int(rng.integers(0, W - ew))
        return (y, x, eh, ew), rng.integers(0, 256, C, dtype=np.uint8)


def _hwc(img: np.ndarray) -> np.ndarray:
    """``(H, W, C)``: cv2 drops the channel axis of single-channel images."""
    return img[:, :, None] if img.ndim == 2 else img


def _apply_hsv_lut(crop: np.ndarray, lut: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return cv2.cvtColor(cv2.LUT(hsv, lut), cv2.COLOR_HSV2RGB)


def _scaled_odd_kernel(k: int, r: float) -> int:
    """The odd kernel size nearest ``k * r`` (``k`` itself at ``r == 1``).

    A ``k``-pixel kernel of a crop, on a crop of the same window ``r`` x as fine.
    """
    if r == 1.0:
        return int(k)
    return max(1, 2 * int(round((k * r - 1.0) / 2.0)) + 1)


def _fill_rects(
    img: np.ndarray,
    rects: Sequence[Tuple[int, int, int, int]],
    ratio: Tuple[float, float],
    value,
) -> np.ndarray:
    """A copy of ``img`` with ``(y, x, h, w)`` boxes filled with ``value``.

    Boxes are in the pixels of a crop of the same window ``ratio`` (y, x) x
    coarser than ``img``. A box covers pixel edges ``y .. y + h``; on ``img``'s grid that is
    ``y * ry .. (y + h) * ry``, rounded to whole pixels, so boxes of crops of one
    window at two resolutions cover the same part of the window.
    """
    ry, rx = ratio
    img = img.copy()
    for y, x, h, w in rects:
        y0, x0 = int(round(y * ry)), int(round(x * rx))
        y1, x1 = int(round((y + h) * ry)), int(round((x + w) * rx))
        img[y0:y1, x0:x1] = value
    return img


def flip_index(
    skeleton_symmetries: Sequence[Tuple[str, str]], node_names: Sequence[str]
):
    """Node permutation that swaps symmetric (left/right) nodes after a flip."""
    idx = np.arange(len(node_names))
    pos = {n: i for i, n in enumerate(node_names)}
    for a, b in skeleton_symmetries:
        if a in pos and b in pos:
            idx[pos[a]], idx[pos[b]] = pos[b], pos[a]
    return idx


def skeleton_symmetry_names(labels_list: Sequence[sio.Labels]) -> List[Tuple[str, str]]:
    """Union of the (name, name) symmetry pairs of every file's skeleton."""
    pairs = set()
    for labels in labels_list:
        for sk in labels.skeletons:
            for sym in sk.symmetries:
                a, b = (n.name for n in sym.nodes)
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def merged_skeleton(
    labels_list: Sequence[sio.Labels], node_names: Sequence[str]
) -> sio.Skeleton:
    """A skeleton on ``node_names`` with the edges and symmetries of every file."""
    edges, syms = [], set()
    for labels in labels_list:
        for sk in labels.skeletons:
            for e in sk.edges:
                pair = (e.source.name, e.destination.name)
                seen = {tuple(sorted(x)) for x in edges}
                if (
                    all(n in node_names for n in pair)
                    and tuple(sorted(pair)) not in seen
                ):
                    edges.append(pair)
    for a, b in skeleton_symmetry_names(labels_list):
        if a in node_names and b in node_names:
            syms.add((a, b))
    name = "+".join(
        n
        for n in dict.fromkeys(sk.name for lb in labels_list for sk in lb.skeletons)
        if n
    )
    return sio.Skeleton(
        nodes=list(node_names), edges=edges, symmetries=sorted(syms), name=name
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WarpCenteredInstanceDataset(Dataset):
    """Box-scaled centered-instance crops cut by one affine warp from the frame.

    Args:
        labels: Training labels (any mix of skeletons; nodes matched by name).
        crop_size: Model input, an int (square) or ``(height, width)``.
        confmap_head_config: The ``centered_instance.confmaps`` head config
            (``part_names``, ``sigma``, ``output_stride``).
        max_stride: The backbone's max stride; ``crop_size`` must be divisible by it.
        bbox_crop: Window rule (:class:`BBoxCropSpec`); default square, padding 0.1.
        geometric_aug / intensity_aug: The usual augmentation config sections.
        apply_aug: Augment (training) or not (validation: exact inference crops).
        gpu_confmaps: Skip rendering confidence maps in the worker; the
            LightningModule renders them on the GPU from ``instance``.
        bbox_categories: Only linked user boxes of these categories are crop boxes.
        instance_filter: Train-time :class:`InstanceFilter`.
        ensure_rgb / ensure_grayscale: As in the other datasets.
        user_instances_only: Train on user instances only.
        stage2_aug: Optional ``(geometric, intensity)`` sections of a second
            augmentation stage (e.g. RTMPose's weaker last epochs). The active
            stage is :attr:`aug_stage`, a shared-memory tensor that
            :class:`~sleap_nn.training.callbacks.AugmentationStageCallback` flips in
            the main process and persistent workers read per sample.
        return_flip: Add ``flipped`` / ``half_body`` / ``aug_stage`` flags to
            samples (debugging / visualization).
        return_geometry: Add the crop geometry (``crop_window``, ``crop_scale``,
            ``instance_bbox``, ...) validation needs; default: when not augmenting.
        base_seed: Augmentation seed (mixed with the rank and the worker's torch
            seed).
        unlabeled_points: ``data_config.unlabeled_points``: ``"absent"`` trains a
            point with no location in the crop as absent (all-zero map),
            ``"ignore"`` gives it zero loss weight (ViTPose's ``target_weight``).
        node_loss_mask: ``data_config.node_loss_mask``. With it on, samples carry
            ``node_weights`` whenever weights can differ from 1 (a model node some
            file lacks or does not label, or ``unlabeled_points="ignore"``); off,
            every node of every sample is trained.
        supervise_occluded: ``data_config.supervise_occluded`` (see
            :mod:`sleap_nn.data.node_weights`).
        crop_bbox_source: ``"gt"`` or ``"detector"`` (the predicted box linked to
            each instance; see :func:`build_instance_table`).
        crop_bbox_detector / crop_bbox_min_score / crop_bbox_fallback: Which
            predicted boxes count, and what an instance without one does.
        crop_bbox_jitter: Training only (``apply_aug``): standard deviation of the
            independent Gaussian shift of each box edge, as a fraction of the box's
            width / height, applied before the window rule.
        teacher_crop_size: Distillation (``trainer_config.distillation
            .teacher_crop_size``): also cut the SAME window at this size (an int or
            ``(height, width)``, same aspect ratio as ``crop_size``) and return it
            as ``teacher_image``. Same decode, same affine draw (box jitter,
            half-body, rotation, scale, shift, flip) composed with a rescale of
            the crop's pixel grid, and the same photometric draw
            (:meth:`WarpAug.intensity` with ``teacher``). ``None`` (default): no
            second crop, samples unchanged.
    """

    def __init__(
        self,
        labels: List[sio.Labels],
        crop_size,
        confmap_head_config,
        max_stride: int = 1,
        bbox_crop: Optional[BBoxCropSpec] = None,
        geometric_aug=None,
        intensity_aug=None,
        apply_aug: bool = False,
        gpu_confmaps: bool = False,
        bbox_categories: Optional[Sequence[str]] = None,
        instance_filter: Optional[InstanceFilter] = None,
        ensure_rgb: bool = False,
        ensure_grayscale: bool = False,
        user_instances_only: bool = True,
        stage2_aug=None,
        return_flip: bool = False,
        return_geometry: Optional[bool] = None,
        base_seed: int = 0,
        unlabeled_points: str = "absent",
        node_loss_mask: bool = True,
        supervise_occluded: bool = True,
        crop_bbox_source: str = "gt",
        crop_bbox_detector: Optional[str] = None,
        crop_bbox_min_score: float = 0.0,
        crop_bbox_fallback: str = "gt",
        crop_bbox_jitter: float = 0.0,
        teacher_crop_size=None,
    ):
        """Initialize."""
        if cv2 is None:  # pragma: no cover
            raise ImportError("crop_pipeline='warp' needs opencv-python.")
        self.crop_h, self.crop_w = crop_hw(crop_size)
        if self.crop_h % max_stride or self.crop_w % max_stride:
            raise ValueError(
                f"crop_size {crop_size} must be divisible by max_stride {max_stride} "
                "for crop_pipeline='warp' (crops are not padded)."
            )
        self.teacher_hw = self.teacher_affine = None
        if teacher_crop_size is not None:
            th, tw = crop_hw(teacher_crop_size)
            ry, rx = th / self.crop_h, tw / self.crop_w
            if min(th, tw) <= 0 or abs(ry / rx - 1.0) > 0.01:
                raise ValueError(
                    f"teacher_crop_size {teacher_crop_size} must be a positive "
                    f"[height, width] with the aspect ratio of crop_size {crop_size} "
                    "(the teacher sees the same window at another resolution)."
                )
            self.teacher_hw = (th, tw)
            # Student crop -> teacher crop (keypoint coordinates: pixel centers at
            # integers), the same window on a grid (ry, rx) x as fine.
            self.teacher_affine = (
                _translate(-0.5, -0.5) @ np.diag([rx, ry, 1.0]) @ _translate(0.5, 0.5)
            )
        self.node_names = list(confmap_head_config.part_names)
        self.sigma = float(confmap_head_config.sigma)
        self.output_stride = int(confmap_head_config.output_stride)
        self.bbox_crop = bbox_crop or BBoxCropSpec()
        self.apply_aug = apply_aug
        self.augs = [
            (
                WarpAug.from_configs(geometric_aug, intensity_aug)
                if apply_aug
                else WarpAug()
            ).bind_nodes(self.node_names)
        ]
        if apply_aug and stage2_aug is not None:
            self.augs.append(
                WarpAug.from_configs(*stage2_aug).bind_nodes(self.node_names)
            )
        # Active augmentation stage, in shared memory so persistent (forked or
        # spawned) workers see the main process's switch (they never observe plain
        # attribute writes made after they started).
        self.aug_stage = torch.zeros((), dtype=torch.long).share_memory_()
        self.gpu_confmaps = gpu_confmaps
        self.ensure_rgb = ensure_rgb
        self.ensure_grayscale = ensure_grayscale
        self.return_flip = return_flip
        self.return_geometry = (
            (not apply_aug) if return_geometry is None else return_geometry
        )
        self.base_seed = base_seed
        # Global DDP rank, mixed into the augmentation seed (set by the loader
        # builder): every rank seeds torch identically, so without it all ranks
        # would draw the same augmentation stream.
        self.rank = 0
        if unlabeled_points not in UNLABELED_POINTS_MODES:
            raise ValueError(
                f"unlabeled_points must be one of {UNLABELED_POINTS_MODES}, got "
                f"{unlabeled_points!r}."
            )
        self.unlabeled_points = unlabeled_points
        self.flip_idx = flip_index(skeleton_symmetry_names(labels), self.node_names)
        if (
            any(a.flip_p > 0 for a in self.augs)
            and (self.flip_idx == np.arange(len(self.flip_idx))).all()
        ):
            logger.warning(
                "flip_p > 0 but no symmetric node pairs: flips swap nothing."
            )
        self.table = build_instance_table(
            labels,
            self.node_names,
            bbox_source=self.bbox_crop.bbox_source,
            bbox_padding=self.bbox_crop.bbox_padding,
            bbox_categories=bbox_categories,
            instance_filter=instance_filter,
            user_instances_only=user_instances_only,
            supervise_occluded=supervise_occluded,
            crop_bbox_source=crop_bbox_source,
            crop_bbox_detector=crop_bbox_detector,
            crop_bbox_min_score=crop_bbox_min_score,
            crop_bbox_fallback=crop_bbox_fallback,
        )
        if crop_bbox_jitter < 0:
            raise ValueError(f"crop_bbox_jitter must be >= 0, got {crop_bbox_jitter}.")
        self.crop_bbox_jitter = float(crop_bbox_jitter) if apply_aug else 0.0
        # Per-node weights ride along only when they can differ from 1, so a run on
        # fully labeled files with `unlabeled_points: absent` uses the plain loss.
        self.emit_node_weights = node_loss_mask and (
            unlabeled_points == "ignore" or bool((self.table.weight != 1).any())
        )
        if not node_loss_mask:
            # Train every node, as absent where a file has no location for it.
            self.table.weight = np.ones_like(self.table.weight)
        # Precompute the un-augmented windows (also what inference cuts).
        windows, scales, keep_half = bbox_crop_windows(
            torch.from_numpy(self.table.box),
            (self.crop_h, self.crop_w),
            self.bbox_crop.aspect,
            self.bbox_crop.padding,
            self.bbox_crop.min_size,
        )
        self.windows = windows.numpy()
        self.scales = scales.numpy()
        # The padded box in crop pixels: the unit of the translation augmentation.
        self.box_wh = (2.0 * keep_half).numpy()
        if self.bbox_crop.aspect == "letterbox":
            logger.warning("crop_pipeline='warp' does not letterbox; using 'square'.")
        #: Number of training files (``labels_idx`` groups, filtered or not).
        self.num_files = len(labels)
        # Only kept when some frame is not a plain image file.
        self.labels_list = labels if (self.table.path_id < 0).any() else None
        self._rng = self._teacher_rng = None
        self._pid = None
        f = self.table.filtered
        logger.info(
            f"Warp dataset: {len(self.table)} instances from {len(labels)} file(s) "
            f"(per file {np.bincount(self.table.labels_idx).tolist()}); dropped "
            f"{f['no_visible_node']} without a visible node, {f['few_labeled_nodes']} "
            f"with too few labeled nodes, {f['small_box']} with small boxes."
        )
        if "detector" in self.table.box_sources:
            logger.info(
                f"Warp dataset crop boxes: {self.table.box_sources['detector']} from "
                f"the detector, {self.table.box_sources['gt']} from the GT "
                f"(fallback); dropped {f.get('no_detector_box', 0)} without a "
                "detector box."
            )

    def __len__(self) -> int:
        """Number of samples (or batches)."""
        return len(self.table)

    @property
    def aug(self) -> WarpAug:
        """The augmentation of the active stage (:attr:`aug_stage`)."""
        return self.augs[min(int(self.aug_stage), len(self.augs) - 1)]

    def set_aug_stage(self, stage: int) -> None:
        """Switch the augmentation stage (visible to running workers)."""
        self.aug_stage.fill_(int(stage))

    @property
    def labels_idx(self) -> np.ndarray:
        """``labels_idx`` of every sample (for :class:`BalancedBatchSampler`)."""
        return self.table.labels_idx

    def _rng_for_process(self) -> np.random.Generator:
        # One generator per worker process, seeded from the worker's torch seed so
        # workers draw different augmentations.
        if self._rng is None or self._pid != os.getpid():
            seed = (self.base_seed, int(self.rank), torch.initial_seed())
            self._rng = np.random.default_rng(seed)
            # The teacher crop's own noise stream, so the student's stream (and
            # crops) are the same with and without a teacher crop.
            self._teacher_rng = np.random.default_rng(seed + (1,))
            self._pid = os.getpid()
        return self._rng

    def _read(self, i: int) -> np.ndarray:
        t = self.table
        pid = t.path_id[i]
        if pid >= 0:
            if self.ensure_grayscale:
                flag = cv2.IMREAD_GRAYSCALE
            elif self.ensure_rgb:
                flag = cv2.IMREAD_COLOR
            else:  # the file's own channels, like the stock path
                flag = cv2.IMREAD_UNCHANGED
            img = cv2.imread(t.paths[pid].decode(), flag)
            if img is None:
                raise IOError(f"Could not read {t.paths[pid].decode()}")
            if img.ndim == 2:
                img = img[:, :, None]
            elif img.shape[-1] == 4:
                img = img[:, :, :3]  # drop alpha
            return img  # BGR or gray
        lf = self.labels_list[t.labels_idx[i]][int(t.lf_idx[i])]
        img = lf.image
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[-1] == 3:
            img = img[:, :, ::-1]  # RGB -> BGR like cv2.imread
        if self.ensure_grayscale and img.shape[-1] == 3:
            img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)[
                :, :, None
            ]
        return np.ascontiguousarray(img)

    def _jittered_window(self, i: int, rng: np.random.Generator):
        """Window, scale and padded-box size of sample ``i``'s box, edges jittered.

        Each edge moves by ``N(0, crop_bbox_jitter)`` times the box's width (x1, x2)
        or height (y1, y2); the jittered box then goes through the same window rule
        (:func:`~sleap_nn.data.instance_cropping.bbox_crop_windows`) as the
        un-jittered one.
        """
        box = self.table.box[i].astype(np.float64)
        w = max(box[2] - box[0], 1.0)
        h = max(box[3] - box[1], 1.0)
        box = box + rng.normal(0.0, self.crop_bbox_jitter, 4) * np.array([w, h, w, h])
        # Keep the corners ordered (a jittered box never collapses below 1 px).
        x1, x2 = min(box[0], box[2]), max(box[0], box[2], min(box[0], box[2]) + 1.0)
        y1, y2 = min(box[1], box[3]), max(box[1], box[3], min(box[1], box[3]) + 1.0)
        windows, scales, keep_half = bbox_crop_windows(
            torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            (self.crop_h, self.crop_w),
            self.bbox_crop.aspect,
            self.bbox_crop.padding,
            self.bbox_crop.min_size,
        )
        return windows[0].numpy(), scales[0].numpy(), (2.0 * keep_half[0]).numpy()

    def _warp(self, img: np.ndarray, M: np.ndarray, hw: Tuple[int, int]):
        """``(h, w, C)`` RGB (or gray) uint8 crop of the BGR frame through ``M``."""
        crop = cv2.warpAffine(
            img,
            M[:2].astype(np.float64),
            (hw[1], hw[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if crop.ndim == 2:
            crop = crop[:, :, None]
        if crop.shape[-1] == 3:
            if self.ensure_grayscale:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[:, :, None]
            else:
                # BGR -> RGB as a contiguous array: the photometric ops (and the
                # CHW copy below) are ~2x slower on a reversed-channel view.
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        elif self.ensure_rgb:
            crop = np.repeat(crop, 3, axis=-1)
        return crop

    def __getitem__(self, i: int) -> Dict:
        """Return the sample at index ``i``."""
        t = self.table
        i = int(i)
        H, W = self.crop_h, self.crop_w
        window, scale, box_wh = self.windows[i], self.scales[i], self.box_wh[i]
        flipped = half_body = False
        rng = None
        stage = int(self.aug_stage)
        aug = self.augs[min(stage, len(self.augs) - 1)]
        if self.apply_aug:
            rng = self._rng_for_process()
            if self.crop_bbox_jitter > 0:
                window, scale, box_wh = self._jittered_window(i, rng)
            if aug.half_body_p > 0:
                vis = np.isfinite(t.kpts[i]).all(-1) & (t.weight[i] > 0)
                hb = aug.half_body_window(
                    rng,
                    t.kpts[i],
                    vis,
                    (H, W),
                    "stretch" if self.bbox_crop.aspect == "stretch" else "square",
                    self.bbox_crop.min_size,
                )
                if hb is not None:
                    (window, scale, box_wh), half_body = hb, True
        M = window_affine(window, scale)
        if self.apply_aug:
            A, flipped = aug.sample(rng, H, W, box_wh)
            M = A @ M
        img = self._read(i)
        orig_hw = img.shape[:2]
        crop = self._warp(img, M, (H, W))
        teacher = None
        if self.teacher_hw is not None:
            # The same window (and augmentation draw) at the teacher's size, from
            # the same decoded frame.
            teacher = self._warp(img, self.teacher_affine @ M, self.teacher_hw)
        if rng is not None:
            if teacher is None:
                crop = aug.intensity(crop, rng)
            else:
                crop, teacher = aug.intensity(crop, rng, teacher, self._teacher_rng)

        k = t.kpts[i].astype(np.float64) @ M[:2, :2].T + M[:2, 2]
        k = k.astype(np.float32)
        w = t.weight[i]
        if flipped:
            k, w = k[self.flip_idx], w[self.flip_idx]
        # Points pushed out of the crop get an empty map (like `filter_oob_points`).
        oob = (k[:, 0] < 0) | (k[:, 1] < 0) | (k[:, 0] > W - 1) | (k[:, 1] > H - 1)
        k[oob] = np.nan
        if self.unlabeled_points == "ignore":
            # ViTPose/mmpose `target_weight`: points that are not labeled visible
            # (NaN, e.g. COCO v=0) or that left the crop get no loss, instead of
            # being trained as absent (an all-zero map).
            w = w * np.isfinite(k).all(axis=-1)

        sample = {
            "instance_image": torch.from_numpy(
                np.ascontiguousarray(crop.transpose(2, 0, 1))
            ).unsqueeze(
                0
            ),  # (1, C, H, W) uint8
            "instance": torch.from_numpy(k).unsqueeze(0),  # (1, K, 2)
            "labels_idx": int(t.labels_idx[i]),
            "video_idx": int(t.video_idx[i]),
            "frame_idx": int(t.frame_idx[i]),
        }
        if teacher is not None:
            sample[TEACHER_IMAGE_KEY] = torch.from_numpy(
                np.ascontiguousarray(teacher.transpose(2, 0, 1))
            ).unsqueeze(
                0
            )  # (1, C, Ht, Wt) uint8
        if self.emit_node_weights:
            # (1, K) like the stock datasets' `node_weights`.
            sample["node_weights"] = torch.from_numpy(
                np.ascontiguousarray(w, dtype=np.float32)
            ).unsqueeze(0)
        if self.return_geometry:
            # Frame -> crop geometry of the un-augmented crop, for mapping
            # validation predictions back to the frame. Left out of training
            # batches: every extra tensor is one more shared-memory segment the
            # main process has to receive per batch (~25% of its throughput).
            win = torch.from_numpy(self.windows[i]).reshape(1, 4)
            sample.update(
                {
                    "num_instances": int(t.num_instances[i]),
                    "orig_size": torch.tensor([orig_hw], dtype=torch.float32),
                    "eff_scale": torch.tensor(1.0),
                    "crop_window": win,
                    "crop_scale": torch.from_numpy(self.scales[i]).reshape(1, 2),
                    "input_scale": torch.tensor(1.0),
                    "instance_bbox": window_corners(win) - 0.5,
                }
            )
        if self.return_flip:
            sample["flipped"] = flipped
            sample["half_body"] = half_body
            sample["aug_stage"] = stage
        if not self.gpu_confmaps:
            sample["confidence_maps"] = render_confmaps(
                sample["instance"], (H, W), self.sigma, self.output_stride
            )  # (1, K, h, w) on CPU
        return sample


# ---------------------------------------------------------------------------
# Targets and loss
# ---------------------------------------------------------------------------


def render_confmaps(
    instance: torch.Tensor, crop_size, sigma: float, output_stride: int
) -> torch.Tensor:
    """Confidence maps of crop keypoints on their device (CPU or GPU).

    Same grid and Gaussian as :func:`~sleap_nn.data.confidence_maps.generate_confmaps`
    (``make_confmaps`` on ``make_grid_vectors``), so CPU and GPU targets agree.

    Args:
        instance: ``(B, 1, K, 2)`` or ``(B, K, 2)`` crop keypoints, NaN = no point.
        crop_size: Crop ``(h, w)`` (or an int).
        sigma: Head sigma (output-grid units, as in the config).
        output_stride: Head output stride.

    Returns:
        ``(B, K, h / stride, w / stride)`` float32 maps.
    """
    h, w = crop_hw(crop_size)
    pts = instance.reshape(instance.shape[0], -1, 2).float()
    xv, yv = make_grid_vectors(h, w, output_stride)
    return make_confmaps(
        pts, xv.to(pts.device), yv.to(pts.device), sigma * output_stride
    )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def resolve_batch_counts(
    counts: Optional[Sequence[int]],
    weights: Optional[Sequence[float]],
    batch_size: int,
    n_groups: int,
) -> List[int]:
    """Per-file samples per batch from explicit ``counts`` or ``weights``.

    ``weights`` are normalized and rounded (largest remainder) to sum to
    ``batch_size``; with neither, every file gets an equal share.
    """
    if counts is not None:
        counts = [int(c) for c in counts]
        if len(counts) != n_groups:
            raise ValueError(
                f"balanced_sampler.counts has {len(counts)} entries for "
                f"{n_groups} training files."
            )
        return counts
    w = np.ones(n_groups) if weights is None else np.asarray(weights, np.float64)
    if len(w) != n_groups:
        raise ValueError(
            f"balanced_sampler.weights has {len(w)} entries for "
            f"{n_groups} training files."
        )
    raw = w / w.sum() * batch_size
    out = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - out))[: batch_size - out.sum()]:
        out[j] += 1
    return out.tolist()


class BalancedBatchSampler(Sampler):
    """Batches with a fixed number of samples from each training file.

    ``counts[g]`` samples of group ``g`` (``labels_idx``) go in every batch, drawn by
    reshuffled passes over the group, so small files are oversampled. One "epoch"
    is ``batches_per_epoch`` batches (budget training by iterations); the per-group
    passes carry over between epochs. Under DDP each rank draws its own stream
    (seed + rank) over all samples, like the embedding sampler, so ranks need no
    coordination and Lightning must not replace the sampler
    (``use_distributed_sampler=False``).

    The stream is a pure function of ``(seed, rank)`` and the global batch index:
    pass ``k`` over group ``g`` is ``permutation(seed, rank, g, k)``, and batch ``b``
    takes positions ``[b * counts[g], (b + 1) * counts[g])`` of that group's
    concatenated passes. So resuming needs only the number of batches delivered
    (``position``, kept up to date by :class:`BalancedDataLoader`): each ``__iter__``
    yields from ``position`` to the end of its epoch.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        counts: Sequence[int],
        batches_per_epoch: int,
        seed: int = 0,
        rank: int = 0,
    ):
        """Initialize."""
        group_ids = np.asarray(group_ids)
        self.counts = [int(c) for c in counts]
        if sum(self.counts) <= 0:
            raise ValueError("balanced_sampler counts must sum to at least 1.")
        self.groups = [np.flatnonzero(group_ids == g) for g in range(len(self.counts))]
        for g, (c, idx) in enumerate(zip(self.counts, self.groups)):
            if c < 0:
                raise ValueError(f"Group {g} has a negative count {c}.")
            if c > 0 and len(idx) == 0:
                raise ValueError(f"Group {g} has count {c} but no samples.")
        self.batch_size = sum(self.counts)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.rank = int(rank)
        #: Global index of the next batch to deliver (batches already trained on).
        self.position = 0
        self._perm_cache: Dict[int, Tuple[int, np.ndarray]] = {}

    def _pass(self, g: int, k: int) -> np.ndarray:
        """Permutation ``k`` of group ``g`` (cached: passes are read in order)."""
        cached = self._perm_cache.get(g)
        if cached is None or cached[0] != k:
            rng = np.random.default_rng((self.seed, self.rank, g, k))
            cached = (k, rng.permutation(self.groups[g]))
            self._perm_cache[g] = cached
        return cached[1]

    def batch(self, b: int) -> np.ndarray:
        """Sample indices of global batch ``b`` (group by group)."""
        parts = []
        for g, c in enumerate(self.counts):
            n = len(self.groups[g])
            pos, end = b * c, (b + 1) * c
            while pos < end:
                k, off = divmod(pos, n)
                m = min(end - pos, n - off)
                parts.append(self._pass(g, k)[off : off + m])
                pos += m
        return np.concatenate(parts)

    def __iter__(self):
        """Yield the batches from ``position`` to the end of its epoch."""
        start = int(self.position)
        end = (start // self.batches_per_epoch + 1) * self.batches_per_epoch
        for b in range(start, end):
            yield self.batch(b).tolist()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.batches_per_epoch

    def state_dict(self) -> Dict[str, int]:
        """The resumable state: the number of batches delivered."""
        return {"position": int(self.position)}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        """Resume from :meth:`state_dict`."""
        self.position = int(state["position"])


class BalancedDataLoader(DataLoader):
    """A ``DataLoader`` over a :class:`BalancedBatchSampler` that Lightning resumes.

    It counts the batches it hands to the training loop (in the main process, so
    worker prefetching does not count) into the sampler's ``position``, and
    exposes ``state_dict`` / ``load_state_dict``, which Lightning stores in and
    restores from the checkpoint's fit-loop state. Every rank resumes rank 0's
    position (ranks deliver the same number of batches) in its own stream. A
    checkpoint saved mid-epoch can be one batch ahead (Lightning fetches one batch
    in advance), so at most one batch is skipped on resume.
    """

    def __iter__(self):
        """Iterate the batches, advancing the sampler's position per batch."""
        sampler = self.batch_sampler
        for batch in super().__iter__():
            sampler.position += 1
            yield batch

    def state_dict(self) -> Dict[str, int]:
        """The sampler's resumable state."""
        return self.batch_sampler.state_dict()

    def load_state_dict(self, state: Dict[str, int]) -> None:
        """Restore the sampler's state."""
        self.batch_sampler.load_state_dict(state)
