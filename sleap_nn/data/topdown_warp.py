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
   math), so no float maps cross the process boundary.
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


def build_instance_table(
    labels_list: Sequence[sio.Labels],
    node_names: Sequence[str],
    bbox_source: str = "auto",
    bbox_padding: float = 0.0,
    bbox_categories: Optional[Sequence[str]] = None,
    instance_filter: Optional[InstanceFilter] = None,
    user_instances_only: bool = True,
    supervise_occluded: bool = True,
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

    Raises:
        ValueError: If ``bbox_source`` resolves to ``masks`` (not supported here) or
            no instance is left.
    """
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
    n_small = n_few = n_empty = 0
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
        },
    )
    return table


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


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


@dataclass
class WarpAug:
    """Augmentation of the warp path: geometry folded into the crop affine.

    Rotation and scale act about the crop center. Translation shifts the window by
    ``translate_*`` times the padded box's width / height (mmpose's
    ``RandomBBoxTransform`` shift), not the crop's. ``flip_p`` mirrors the crop and
    swaps left/right nodes through ``flip_idx``. Contrast pivots on 127.5 and
    brightness scales, as in the stock path (one lookup table on the uint8 crop).
    """

    rotation: Tuple[float, float] = (0.0, 0.0)
    rotation_p: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)
    scale_p: float = 0.0
    translate: Tuple[float, float] = (0.0, 0.0)
    translate_p: float = 0.0
    flip_p: float = 0.0
    contrast: Tuple[float, float] = (1.0, 1.0)
    contrast_p: float = 0.0
    brightness: Tuple[float, float] = (1.0, 1.0)
    brightness_p: float = 0.0
    gaussian_noise: Tuple[float, float] = (0.0, 0.0)  # mean, std (0-1 scale)
    gaussian_noise_p: float = 0.0

    @classmethod
    def from_configs(cls, geometric, intensity) -> "WarpAug":
        """From the ``augmentation_config.geometric`` / ``.intensity`` sections."""
        a = cls()
        if geometric is not None:
            g = dict(geometric)
            affine_p = float(g.get("affine_p", 0.0) or 0.0)

            def p(key):
                v = g.get(key)
                return affine_p if v is None else float(v)

            a.rotation = (
                float(g.get("rotation_min", 0.0)),
                float(g.get("rotation_max", 0.0)),
            )
            a.rotation_p = p("rotation_p")
            a.scale = (float(g.get("scale_min", 1.0)), float(g.get("scale_max", 1.0)))
            a.scale_p = p("scale_p")
            a.translate = (
                float(g.get("translate_width", 0.0)),
                float(g.get("translate_height", 0.0)),
            )
            a.translate_p = p("translate_p")
            a.flip_p = float(g.get("flip_p", 0.0) or 0.0)
            unsupported = [k for k in ("erase_p", "mixup_p") if (g.get(k) or 0) > 0]
            if unsupported:
                logger.warning(f"crop_pipeline='warp' ignores {unsupported}.")
        if intensity is not None:
            i = dict(intensity)
            a.contrast = (
                float(i.get("contrast_min", 1.0)),
                float(i.get("contrast_max", 1.0)),
            )
            a.contrast_p = float(i.get("contrast_p", 0.0) or 0.0)
            a.brightness = (
                float(i.get("brightness_min", 1.0)),
                float(i.get("brightness_max", 1.0)),
            )
            a.brightness_p = float(i.get("brightness_p", 0.0) or 0.0)
            a.gaussian_noise = (
                float(i.get("gaussian_noise_mean", 0.0)),
                float(i.get("gaussian_noise_std", 0.0)),
            )
            a.gaussian_noise_p = float(i.get("gaussian_noise_p", 0.0) or 0.0)
            if (i.get("uniform_noise_p") or 0) > 0:
                logger.warning("crop_pipeline='warp' ignores uniform_noise_p.")
        return a

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
        if self.rotation_p > 0 and rng.random() < self.rotation_p:
            theta = math.radians(rng.uniform(*self.rotation))
        if self.scale_p > 0 and rng.random() < self.scale_p:
            s = rng.uniform(*self.scale)
        if self.translate_p > 0 and rng.random() < self.translate_p:
            bw, bh = (crop_w, crop_h) if box_wh is None else box_wh
            tx = rng.uniform(-1, 1) * self.translate[0] * float(bw)
            ty = rng.uniform(-1, 1) * self.translate[1] * float(bh)
        c, si = math.cos(theta) * s, math.sin(theta) * s
        rs = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        A = _translate(cx + tx, cy + ty) @ rs @ _translate(-cx, -cy)
        flipped = self.flip_p > 0 and rng.random() < self.flip_p
        if flipped:
            A = np.array([[-1.0, 0.0, crop_w - 1.0], [0.0, 1.0, 0.0], [0, 0, 1.0]]) @ A
        return A, flipped

    def intensity(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Contrast / brightness / gaussian noise on a uint8 crop."""
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
            crop = cv2.LUT(crop, np.rint(lut).astype(np.uint8))
            if crop.ndim == 2:
                crop = crop[:, :, None]
        if self.gaussian_noise_p > 0 and rng.random() < self.gaussian_noise_p:
            m, sd = self.gaussian_noise
            noise = rng.normal(m * 255.0, sd * 255.0, crop.shape).astype(np.float32)
            crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return crop


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
        return_flip: Add a ``flipped`` flag to samples (debugging / visualization).
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
        return_flip: bool = False,
        return_geometry: Optional[bool] = None,
        base_seed: int = 0,
        unlabeled_points: str = "absent",
        node_loss_mask: bool = True,
        supervise_occluded: bool = True,
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
        self.node_names = list(confmap_head_config.part_names)
        self.sigma = float(confmap_head_config.sigma)
        self.output_stride = int(confmap_head_config.output_stride)
        self.bbox_crop = bbox_crop or BBoxCropSpec()
        self.apply_aug = apply_aug
        self.aug = (
            WarpAug.from_configs(geometric_aug, intensity_aug)
            if apply_aug
            else WarpAug()
        )
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
            self.aug.flip_p > 0
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
        )
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
        self._rng = None
        self._pid = None
        f = self.table.filtered
        logger.info(
            f"Warp dataset: {len(self.table)} instances from {len(labels)} file(s) "
            f"(per file {np.bincount(self.table.labels_idx).tolist()}); dropped "
            f"{f['no_visible_node']} without a visible node, {f['few_labeled_nodes']} "
            f"with too few labeled nodes, {f['small_box']} with small boxes."
        )

    def __len__(self) -> int:
        """Number of samples (or batches)."""
        return len(self.table)

    @property
    def labels_idx(self) -> np.ndarray:
        """``labels_idx`` of every sample (for :class:`BalancedBatchSampler`)."""
        return self.table.labels_idx

    def _rng_for_process(self) -> np.random.Generator:
        # One generator per worker process, seeded from the worker's torch seed so
        # workers draw different augmentations.
        if self._rng is None or self._pid != os.getpid():
            self._rng = np.random.default_rng(
                (self.base_seed, int(self.rank), torch.initial_seed())
            )
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

    def __getitem__(self, i: int) -> Dict:
        """Return the sample at index ``i``."""
        t = self.table
        i = int(i)
        H, W = self.crop_h, self.crop_w
        M = window_affine(self.windows[i], self.scales[i])
        flipped = False
        rng = None
        if self.apply_aug:
            rng = self._rng_for_process()
            A, flipped = self.aug.sample(rng, H, W, self.box_wh[i])
            M = A @ M
        img = self._read(i)
        orig_hw = img.shape[:2]
        crop = cv2.warpAffine(
            img,
            M[:2].astype(np.float64),
            (W, H),
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
                crop = crop[:, :, ::-1]  # BGR -> RGB
        elif self.ensure_rgb:
            crop = np.repeat(crop, 3, axis=-1)
        if rng is not None:
            crop = self.aug.intensity(crop, rng)

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
