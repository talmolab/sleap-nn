"""Prediction galleries for W&B: panel selection, prediction and drawing.

Used by :class:`~sleap_nn.training.callbacks.WandBGalleryCallback`. Everything here is
pure (numpy / OpenCV / torch, no Lightning state), so it can be tested on its own:

* :func:`select_panel` picks a fixed, stratified validation panel from per-sample
  metadata (no image decoding).
* :func:`predict_boxes` / :func:`predict_keypoints` run one no-grad forward over a
  batch of panel images with the model's own decoders.
* :func:`bbox_wandb_image` builds a ``wandb.Image`` with native box overlays
  (ground truth and predictions, scores attached so the W&B viewer can filter them);
  :func:`draw_keypoints` draws keypoints and skeleton edges with OpenCV.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

#: Roles of the panel samples, in display order within a file.
ROLES = ("typical", "hard")

# Node colors (RGB), cycled by node index within a file.
_PALETTE = np.array(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
    ],
    dtype=np.uint8,
)
_GT_COLOR = (0, 255, 0)
_ERR_COLOR = (255, 64, 64)
_PRED_EDGE_COLOR = (255, 191, 0)


def name_seed(name: str) -> int:
    """A seed that depends only on ``name`` (stable across runs and processes)."""
    return zlib.crc32(str(name).encode("utf-8"))


def _quotas(counts: np.ndarray, size: int) -> np.ndarray:
    """Split ``size`` slots over groups of ``counts`` samples (water-filling).

    Every group gets an equal share (the remainder goes to the larger groups first),
    capped at its size; slots a small group cannot use go to the others.
    """
    counts = np.asarray(counts, dtype=int)
    quota = np.zeros_like(counts)
    left = int(min(size, counts.sum()))
    while left > 0:
        open_ = np.flatnonzero(quota < counts)
        share = max(left // len(open_), 1)
        # Larger groups (then lower index) take the remainder first.
        order = sorted(open_, key=lambda g: (-counts[g], g))
        for g in order:
            if left == 0:
                break
            add = min(share, counts[g] - quota[g], left)
            quota[g] += add
            left -= add
    return quota


def select_panel(
    groups: np.ndarray,
    names: Sequence[str],
    size: int = 64,
    hardness: Optional[np.ndarray] = None,
) -> List[Tuple[int, str]]:
    """A fixed, stratified panel of sample indices.

    Each group (validation file) gets an equal share of ``size`` (a group with fewer
    samples shows all of them and leaves its slots to the others). Within a group, a
    quarter of its share (at least one, when it has two or more slots and
    ``hardness`` is given) are its hardest samples, the rest a seeded draw. The
    draw's seed is :func:`name_seed` of the group's name, so a group's picks do not
    depend on the other groups, the training seed or the process. With more groups
    than ``size``, ``size`` groups spread evenly over the list get one sample each.

    Args:
        groups: ``(n,)`` group (file) index of every sample.
        names: Group names, indexed by group.
        size: Maximum panel size.
        hardness: Optional ``(n,)`` score; higher is harder (ties: lower index).

    Returns:
        ``[(sample_index, role), ...]`` in display order: by group, then typical
        (by index) before hard (hardest first). ``role`` is ``"typical"`` or
        ``"hard"``.
    """
    groups = np.asarray(groups, dtype=int)
    if len(groups) == 0 or size <= 0:
        return []
    present = [g for g in range(len(names)) if np.any(groups == g)]
    if len(present) > size:
        keep = np.unique(np.linspace(0, len(present) - 1, size).round().astype(int))
        present = [present[i] for i in keep]
    members = [np.flatnonzero(groups == g) for g in present]
    quota = _quotas(np.array([len(m) for m in members]), size)

    panel = []
    for g, idx, q in zip(present, members, quota):
        if q <= 0:
            continue
        n_hard = 0
        if hardness is not None and q >= 2:
            n_hard = max(1, q // 4)
        hard = []
        if n_hard:
            h = np.asarray(hardness, dtype=np.float64)[idx]
            # Hardest first; ties keep the lower index (stable sort on -h).
            order = np.argsort(-h, kind="stable")
            hard = [int(idx[i]) for i in order[:n_hard]]
        rest = np.setdiff1d(idx, hard)
        rng = np.random.default_rng(name_seed(names[g]))
        n_typ = min(int(q) - len(hard), len(rest))
        typical = sorted(int(i) for i in rng.choice(rest, n_typ, replace=False))
        panel += [(i, "typical") for i in typical] + [(i, "hard") for i in hard]
    return panel


def box_hardness(boxes_per_sample: Sequence[np.ndarray]) -> np.ndarray:
    """Detector hardness: most boxes first, then the smallest box (``sqrt`` area).

    Args:
        boxes_per_sample: Per sample, its ``(n, 4)`` xyxy ground-truth boxes.

    Returns:
        ``(n_samples,)`` scores, higher = harder.
    """
    out = np.zeros(len(boxes_per_sample), dtype=np.float64)
    for i, b in enumerate(boxes_per_sample):
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
        b = b[np.isfinite(b).all(axis=1)]
        if not len(b):
            out[i] = -np.inf
            continue
        side = np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0, None))
        out[i] = len(b) * 1e6 - side.min()
    return out


def instance_hardness(boxes: np.ndarray) -> np.ndarray:
    """Pose hardness: the smallest animal (box ``sqrt`` area, source pixels) first."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    side = np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0, None))
    return np.where(np.isfinite(side), -side, -np.inf)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_boxes(module, images, score_threshold: float = 0.05):
    """Decoded boxes of a ``_BBoxLightningModel`` for a batch of panel frames.

    Args:
        module: The detector's LightningModule (its ``forward`` and ``decoder``).
        images: ``(B, 1, C, H, W)`` uint8 model-input frames on the module's device.
        score_threshold: Keep detections down to this score (the W&B viewer's score
            slider filters further).

    Returns:
        Per frame ``(boxes (n, 4) xyxy model-input px, scores (n,))`` numpy arrays.
    """
    import torch

    with torch.no_grad():
        dec_in = module.forward(images)
        boxes, scores, _ = module.decoder(
            dec_in, img_hw=tuple(images.shape[-2:]), score_threshold=score_threshold
        )
    boxes, scores = boxes.float().cpu().numpy(), scores.float().cpu().numpy()
    out = []
    for b, s in zip(boxes, scores):
        valid = np.isfinite(s)
        out.append((b[valid], s[valid]))
    return out


def predict_keypoints(module, images):
    """Crop keypoints of a top-down centered-instance module, every node predicted.

    Uses the module's own peak finder settings (refinement, DARK sigma, stride) with
    threshold 0, so every node gets a location and the error is not lowered by
    abstaining on hard points.

    Args:
        module: The centered-instance LightningModule.
        images: ``(B, 1, C, H, W)`` uint8 crops on the module's device.

    Returns:
        ``(points (B, K, 2) crop px, values (B, K))`` numpy arrays.
    """
    import torch
    from sleap_nn.inference.peak_finding import find_global_peaks

    layer = module.instance_peaks_inf_layer
    with torch.no_grad():
        cms = module.forward(images).float()
        pts, vals = find_global_peaks(
            cms,
            threshold=0.0,
            refinement=layer.refinement,
            integral_patch_size=layer.integral_patch_size,
            dark_blur_sigma=layer.dark_blur_sigma,
        )
    pts = pts * layer.output_stride
    if getattr(layer, "input_scale", 1.0) != 1.0:
        pts = pts / layer.input_scale
    return pts.cpu().numpy(), vals.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def to_rgb_uint8(image) -> np.ndarray:
    """``(C, H, W)`` / ``(1, C, H, W)`` tensor or array -> ``(H, W, 3)`` uint8 RGB."""
    a = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    while a.ndim > 3:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        scale = 255.0 if a.max() <= 1.0 else 1.0
        a = np.clip(a * scale, 0, 255).astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.ascontiguousarray(a[..., :3])


def downscale(image: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """Shrink ``image`` so its long side is at most ``long_side``; ``(image, scale)``."""
    h, w = image.shape[:2]
    s = min(1.0, float(long_side) / max(h, w))
    if s >= 1.0:
        return image, 1.0
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), s


def _box_entries(boxes, scale, class_ids=None, scores=None, names=None) -> List[dict]:
    out = []
    for j, b in enumerate(np.asarray(boxes, dtype=np.float64).reshape(-1, 4)):
        if not np.isfinite(b).all():
            continue
        x1, y1, x2, y2 = (b * scale).tolist()
        cid = int(class_ids[j]) if class_ids is not None else 0
        entry = {
            "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
            "domain": "pixel",
            "class_id": max(cid, 0),
        }
        if scores is not None:
            s = float(scores[j])
            entry["scores"] = {"score": s}
            entry["box_caption"] = f"{s:.2f}"
        elif names:
            entry["box_caption"] = names[max(cid, 0)] if cid < len(names) else ""
        out.append(entry)
    return out


def encode_display_jpeg(
    image: np.ndarray, path: str, long_side: int, quality: int = 85
):
    """Downscale an RGB frame to ``long_side`` and write it as JPEG.

    Returns:
        ``(scale, (height, width))`` of the written image.
    """
    small, s = downscale(image, long_side)
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise IOError(f"JPEG encoding failed for {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return s, small.shape[:2]


def bbox_wandb_image(
    image: Optional[np.ndarray],
    gt_boxes: np.ndarray,
    caption: str,
    categories: Sequence[str],
    pred_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    gt_classes: Optional[np.ndarray] = None,
    long_side: int = 640,
    max_preds: int = 50,
    cached: Optional[Tuple[str, float]] = None,
):
    """A JPEG ``wandb.Image`` of a frame with native W&B box overlays.

    Args:
        image: ``(H, W, 3)`` uint8 RGB frame (model-input pixels); unused with
            ``cached``.
        gt_boxes: ``(n, 4)`` xyxy ground-truth boxes in ``image`` pixels.
        caption: Image caption.
        categories: Class names (``class_id`` indexes them).
        pred_boxes: Predicted ``(m, 4)`` boxes, or ``None`` (training inputs show
            their targets only).
        pred_scores: Scores of ``pred_boxes``; the ``max_preds`` highest are kept.
        gt_classes: Ground-truth class ids (default all 0).
        long_side: The image is downscaled to this long side before encoding.
        max_preds: Maximum predicted boxes per image (highest scores).
        cached: ``(jpeg_path, scale)`` of the frame already downscaled and encoded
            by :func:`encode_display_jpeg` (validation panels are encoded once).

    Returns:
        A ``wandb.Image``.
    """
    import wandb

    if cached is not None:
        data, s = cached
    else:
        data, s = downscale(image, long_side)
    class_labels = {i: str(c) for i, c in enumerate(categories or ["object"])}
    boxes = {
        "ground_truth": {
            "box_data": _box_entries(
                gt_boxes, s, gt_classes, names=list(class_labels.values())
            ),
            "class_labels": class_labels,
        }
    }
    if pred_boxes is not None:
        pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores).reshape(-1)
        keep = np.argsort(-pred_scores, kind="stable")[:max_preds]
        boxes["predictions"] = {
            "box_data": _box_entries(
                pred_boxes[keep], s, None, scores=pred_scores[keep]
            ),
            "class_labels": class_labels,
        }
    if cached is not None:
        return wandb.Image(data, caption=caption, boxes=boxes)
    return wandb.Image(data, caption=caption, boxes=boxes, file_type="jpg")


def draw_keypoints(
    image: np.ndarray,
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]] = (),
    gt: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    pred_values: Optional[np.ndarray] = None,
    min_value: float = 0.2,
) -> np.ndarray:
    """Draw ground truth and predicted keypoints of one crop with OpenCV.

    Only ``nodes`` (the crop's own skeleton's channels) are drawn. With predictions:
    the predicted skeleton (amber edges, a dot per node in its own color, a cross
    where the peak value is below ``min_value``), ground truth as hollow green
    circles, and a red line from each ground-truth point to its prediction. Without
    predictions (training targets): the ground-truth skeleton in green.

    Args:
        image: ``(H, W, 3)`` uint8 RGB crop (not modified).
        nodes: Model channel indices to draw.
        edges: ``(src, dst)`` model channel index pairs.
        gt: ``(K, 2)`` ground truth in crop px (NaN = none), or ``None``.
        pred: ``(K, 2)`` predictions in crop px, or ``None``.
        pred_values: ``(K,)`` peak values, or ``None``.
        min_value: Peak value below which a prediction is drawn as a cross.

    Returns:
        A new ``(H, W, 3)`` uint8 image.
    """
    out = np.ascontiguousarray(image.copy())
    nodes = [int(n) for n in nodes]
    node_set = set(nodes)
    color = {
        n: tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
        for i, n in enumerate(nodes)
    }

    def pt(a, n):
        if a is None:
            return None
        p = a[n]
        if not np.isfinite(p).all():
            return None
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    aa = cv2.LINE_AA
    skel, skel_color = (pred, _PRED_EDGE_COLOR) if pred is not None else (gt, _GT_COLOR)
    for a, b in edges:
        if a in node_set and b in node_set:
            pa, pb = pt(skel, a), pt(skel, b)
            if pa and pb:
                cv2.line(out, pa, pb, skel_color, 1, aa)
    for n in nodes:
        g, p = pt(gt, n), pt(pred, n)
        if g and p:
            cv2.line(out, g, p, _ERR_COLOR, 1, aa)
    for n in nodes:
        g, p = pt(gt, n), pt(pred, n)
        if g:
            cv2.circle(out, g, 4 if pred is not None else 2, _GT_COLOR, 1, aa)
        if p:
            low = pred_values is not None and not (float(pred_values[n]) >= min_value)
            if low:
                cv2.drawMarker(out, p, color[n], cv2.MARKER_TILTED_CROSS, 6, 1, aa)
            else:
                cv2.circle(out, p, 2, color[n], -1, aa)
    return out


def keypoint_error(gt: np.ndarray, pred: np.ndarray, nodes: Sequence[int]) -> float:
    """Median distance (crop px) between ``gt`` and ``pred`` over ``nodes`` with GT."""
    nodes = np.asarray(list(nodes), dtype=int)
    if not len(nodes):
        return float("nan")
    d = np.linalg.norm(np.asarray(gt)[nodes] - np.asarray(pred)[nodes], axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("nan")


def keypoint_wandb_image(image: np.ndarray, caption: str):
    """A JPEG ``wandb.Image`` of a drawn crop."""
    import wandb

    return wandb.Image(image, caption=caption, file_type="jpg")


def caption(
    name: str,
    lab: Optional[str],
    video_idx: int,
    frame_idx: int,
    extra: str = "",
    role: Optional[str] = None,
) -> str:
    """``"<dataset> | <lab> | v<video> f<frame> | <role> | <extra>"``."""
    parts = [str(name)]
    if lab:
        parts.append(str(lab))
    parts.append(f"v{int(video_idx)} f{int(frame_idx)}")
    if role:
        parts.append(role)
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def file_edges(skeletons, model_nodes: Sequence[str]) -> List[Tuple[int, int]]:
    """Edges of a file's skeletons as model channel index pairs (both ends known)."""
    index = {n: i for i, n in enumerate(model_nodes)}
    out = []
    for sk in skeletons or []:
        for e in getattr(sk, "edges", []):
            a = index.get(e.source.name)
            b = index.get(e.destination.name)
            if a is not None and b is not None:
                out.append((a, b))
    return out


def file_display_names(labels_list, fallback: Sequence[str]) -> List[str]:
    """Per labels file, ``provenance["dataset"]`` if all are set and unique, else ``fallback``."""
    names = []
    for labels in labels_list:
        prov = getattr(labels, "provenance", None) or {}
        names.append(prov.get("dataset"))
    if all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names):
        return names
    return list(fallback)


def file_labs(
    labels_list, names: Sequence[str], labs: Optional[Dict[str, str]]
) -> List[Optional[str]]:
    """Per labels file, ``labs[name]`` if given, else ``provenance["lab"]``, else ``None``."""
    labs = dict(labs or {})
    out = []
    for labels, name in zip(labels_list, names):
        prov = getattr(labels, "provenance", None) or {}
        out.append(labs.get(name) or prov.get("lab"))
    return out
