"""Write detector boxes into training labels (``sleap-nn detect-boxes``).

Detector-box training crops (``preprocessing.crop_bbox_source: detector``) need,
in the training ``.slp``, a ``PredictedBoundingBox`` linked to each pose instance:
the box a detector finds for that person. This module runs or loads person
detections for every labeled frame of a labels file, matches them to the frame's
ground-truth people and writes a copy of the labels with the boxes added.

**Detections** (:class:`FrameDetections`, one entry per labeled frame) come from:

* a ``bbox_*`` model (e.g. a ``bbox_detr`` checkpoint or an imported RF-DETR) run
  over the labeled frames (:func:`detect_with_layer`), optionally in full-height
  tiles for very wide frames (:class:`TileSpec`, JRDB's 3760 x 480 panoramas);
* a COCO detection-results JSON matched by ``image_id`` or file name
  (:func:`load_coco_detections`);
* the raw chunk files of the human project's D-FINE pass
  (``dets/<model>/<set>/chunk_*.npz`` + its image list;
  :func:`load_detection_chunks`), so existing detections can be replayed.

**Matching** (:func:`match_detections`, a port of the human project's
``detector/match.py``): per frame, Hungarian matching on IoU between the
detections scoring at least ``score_threshold`` (0.3) and the ground-truth people,
keeping pairs with IoU above ``min_iou`` (0.1). Ground-truth people
(:func:`gt_people`) are every non-crowd user box of the person category (people
without keypoints too, so their detections are not handed to a labeled
neighbour) plus the keypoint extent of each pose instance with no linked person
box (e.g. MPII, whose boxes are head boxes). Crowd boxes are ignored.

**Output:** every detection scoring at least ``score_threshold`` is appended to its
frame as a ``PredictedBoundingBox`` with the file's single person ``Category``
(one shared object: a per-box category makes sleap-io store and, on load, dedupe
one category per box), ``source`` = the detector id, its ``score`` and
``metadata["match_iou"]``. A detection matched to a pose instance is linked to it
(``box.instance``); unmatched detections and detections matched to keypoint-less
people stay unlinked. Frames, instances, masks, existing boxes (at the same
indices) and provenance are kept, ``/human_meta`` is copied verbatim, and the
match statistics go into ``provenance["detector_boxes"]`` and a JSON next to the
output (:func:`write_detector_labels`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
import sleap_io as sio
from loguru import logger
from scipy.optimize import linear_sum_assignment

from sleap_nn.data.bboxes import is_crowd_bbox

DEFAULT_CATEGORY = "person"
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MIN_IOU = 0.1


@attrs.define
class FrameDetections:
    """Person detections for the labeled frames of one labels file.

    Attributes:
        boxes: Per labeled frame (in ``labels.labeled_frames`` order), ``(K, 4)``
            float64 ``x1, y1, x2, y2`` boxes in frame pixels.
        scores: Per labeled frame, ``(K,)`` detection scores.
        detector: Detector id, stored as the boxes' ``source``.
        image_hw: Per labeled frame, the ``(height, width)`` of its image when the
            source knows it (model runs and detection chunks), else ``None``.
        info: Source-specific counts (e.g. detections whose image is not in the
            labels), recorded in the match statistics.
    """

    boxes: List[np.ndarray]
    scores: List[np.ndarray]
    detector: str
    image_hw: Optional[List[Optional[Tuple[int, int]]]] = None
    info: Dict[str, object] = attrs.field(factory=dict)

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.boxes)


def _empty_detections(n_frames: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return (
        [np.zeros((0, 4), np.float64) for _ in range(n_frames)],
        [np.zeros(0, np.float64) for _ in range(n_frames)],
    )


def frame_image_path(lf: sio.LabeledFrame) -> Optional[str]:
    """The image file of a labeled frame of an image-list video, else ``None``."""
    fn = lf.video.filename
    if isinstance(fn, (list, tuple)):
        return str(fn[lf.frame_idx]) if 0 <= lf.frame_idx < len(fn) else None
    return None


# ---------------------------------------------------------------------------
# Detection sources
# ---------------------------------------------------------------------------


def load_coco_detections(
    labels: sio.Labels,
    detections: Union[str, Path, List[dict]],
    category_id: Optional[int] = 1,
    detector: Optional[str] = None,
) -> FrameDetections:
    """Detections from a COCO results JSON, attached to the labels' frames.

    Entries are ``{"image_id", "bbox": [x, y, w, h], "score", "category_id"}``. An
    entry is matched to frames by ``file_name`` (full path, else base name) when it
    has one, else by ``image_id`` (the frame's ``/human_meta`` image id, or the
    trailing digits of its image file name, as
    :func:`~sleap_nn.evaluation_coco.frame_image_ids` reads them). Several frames
    with one image (e.g. two skeleton views of a file) all get its detections.
    A ``bbox_xyxy`` entry, written by :func:`save_coco_detections`, is used instead
    of ``bbox`` so a round trip keeps the boxes bit-exact. Detections keep the
    file's order within a frame.

    Args:
        labels: Labels whose frames the detections belong to.
        detections: Path to the JSON results file, or the parsed list.
        category_id: Keep only entries of this category (entries without one are
            kept); ``None`` keeps all.
        detector: Detector id for the boxes' ``source``; defaults to the JSON
            file's stem (``"detections"`` for a parsed list).

    Returns:
        The :class:`FrameDetections`; ``info["n_unmatched_detections"]`` counts
        entries whose image is not in ``labels``.
    """
    from sleap_nn.evaluation_coco import frame_image_ids

    if detector is None:
        detector = (
            Path(detections).stem
            if isinstance(detections, (str, Path))
            else "detections"
        )
    if not isinstance(detections, list):
        with open(detections) as f:
            detections = json.load(f)
    frames = labels.labeled_frames
    by_path: Dict[str, List[int]] = {}
    by_name: Dict[str, List[int]] = {}
    for i, lf in enumerate(frames):
        p = frame_image_path(lf)
        if p is not None:
            by_path.setdefault(p, []).append(i)
            by_name.setdefault(Path(p).name, []).append(i)
    by_id: Optional[Dict[int, List[int]]] = None

    per_frame: List[List[Tuple[List[float], float]]] = [[] for _ in frames]
    unmatched = 0
    for det in detections:
        if category_id is not None and det.get("category_id", category_id) != (
            category_id
        ):
            continue
        if "file_name" in det:
            fn = str(det["file_name"])
            targets = by_path.get(fn) or by_name.get(Path(fn).name) or []
        else:
            if by_id is None:
                by_id = {}
                for i, iid in enumerate(frame_image_ids(labels)):
                    if iid is not None:
                        by_id.setdefault(int(iid), []).append(i)
            targets = by_id.get(int(det["image_id"]), [])
        if not targets:
            unmatched += 1
            continue
        if "bbox_xyxy" in det:
            xyxy = [float(v) for v in det["bbox_xyxy"]]
        else:
            x, y, w, h = (float(v) for v in det["bbox"])
            xyxy = [x, y, x + w, y + h]
        for i in targets:
            per_frame[i].append((xyxy, float(det["score"])))
    boxes = [
        np.array([b for b, _ in fr], np.float64).reshape(-1, 4) for fr in per_frame
    ]
    scores = [np.array([s for _, s in fr], np.float64) for fr in per_frame]
    if unmatched:
        logger.warning(f"{unmatched} detections have no labeled frame; ignored.")
    return FrameDetections(
        boxes=boxes,
        scores=scores,
        detector=str(detector),
        info={"n_unmatched_detections": unmatched},
    )


def load_detection_chunks(
    labels: sio.Labels,
    chunk_dir: Union[str, Path],
    image_list: Union[str, Path],
    detector: Optional[str] = None,
) -> FrameDetections:
    """Detections from the human project's resumable detector pass (``detect.py``).

    ``chunk_dir`` holds ``chunk_*.npz`` files with flat per-detection arrays
    ``img_idx`` (index into ``image_list``), ``boxes`` (xyxy, raw-pixel
    orientation) and ``scores``, per-image ``image_idx`` / ``image_hw``, and the
    detector id ``model``. ``image_list`` is the text file of image paths the pass
    ran over (one per line); frames are matched to it by their image path. Within
    a frame, detections keep the chunk files' order, so matching ties break as
    they did in ``match.py``.

    Raises:
        ValueError: If no chunk is found, the chunks do not cover every listed
            image (an incomplete pass) or a labeled frame's image is not listed.
    """
    files = sorted(Path(chunk_dir).glob("chunk_*.npz"))
    if not files:
        raise ValueError(f"No chunk_*.npz files in {chunk_dir}.")
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "model"}
    model_id = str(parts[0]["model"])
    paths = Path(image_list).read_text().splitlines()
    paths = [p for p in paths if p]
    if len(np.unique(d["image_idx"])) != len(paths):
        raise ValueError(
            f"The chunks cover {len(np.unique(d['image_idx']))} of the "
            f"{len(paths)} images of {image_list}: the detector pass is incomplete."
        )
    path_to_idx = {p: i for i, p in enumerate(paths)}
    order = np.argsort(d["img_idx"], kind="stable")
    starts = np.searchsorted(d["img_idx"][order], np.arange(len(paths) + 1))
    hw = np.zeros((len(paths), 2), np.int64)
    hw[d["image_idx"]] = d["image_hw"]

    boxes, scores, image_hw = [], [], []
    for lf in labels.labeled_frames:
        p = frame_image_path(lf)
        i = path_to_idx.get(p) if p is not None else None
        if i is None:
            raise ValueError(
                f"Frame {lf.frame_idx} of {lf.video.filename if p is None else p} "
                f"is not in {image_list}."
            )
        sel = order[starts[i] : starts[i + 1]]
        boxes.append(d["boxes"][sel].astype(np.float64))
        scores.append(d["scores"][sel])
        image_hw.append((int(hw[i, 0]), int(hw[i, 1])))
    return FrameDetections(
        boxes=boxes,
        scores=scores,
        detector=detector or model_id,
        image_hw=image_hw,
        info={"n_chunks": len(files)},
    )


def save_coco_detections(
    labels: sio.Labels,
    detections: FrameDetections,
    path: Union[str, Path],
    category_id: int = 1,
) -> int:
    """Write detections as a COCO results JSON that :func:`load_coco_detections` reads.

    Each entry has ``image_id`` (when the frame has one), ``file_name`` (when the
    frame is an image file), ``bbox`` (xywh), ``bbox_xyxy`` (the exact box),
    ``score`` and ``category_id``. Returns the number of entries.
    """
    from sleap_nn.evaluation_coco import frame_image_ids

    ids = frame_image_ids(labels)
    out = []
    for lf, iid, boxes, scores in zip(
        labels.labeled_frames, ids, detections.boxes, detections.scores
    ):
        fn = frame_image_path(lf)
        for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist()):
            entry = {}
            if iid is not None:
                entry["image_id"] = int(iid)
            if fn is not None:
                entry["file_name"] = fn
            entry.update(
                bbox=[x1, y1, x2 - x1, y2 - y1],
                bbox_xyxy=[x1, y1, x2, y2],
                score=float(s),
                category_id=category_id,
            )
            out.append(entry)
    Path(path).write_text(json.dumps(out))
    return len(out)


@attrs.define
class TileSpec:
    """Full-height horizontal tiling for very wide frames (``detect_tiled.py``).

    A frame wider than ``width`` is cut into full-height tiles ``width`` px wide at
    ``stride`` (the last tile flush with the right edge: 9 tiles of 640 for a
    3760-px JRDB panorama), each tile is detected on its own, and the tile boxes,
    shifted back to frame coordinates, are merged by :func:`merge_tile_detections`.
    Frames no wider than ``width`` are one tile (still merged, i.e. NMS-ed).

    Attributes:
        width: Tile width in frame pixels.
        stride: Horizontal tile stride.
        nms_iou: Suppress a box whose IoU with a kept box exceeds this.
        contain: Suppress a box touching an inner tile edge when more than this
            fraction of it lies inside a kept box (a truncated person).
        edge_margin: A box within this many pixels of an inner tile edge touches it.
    """

    width: int = 640
    stride: int = 390
    nms_iou: float = 0.6
    contain: float = 0.7
    edge_margin: float = 2.0

    def starts(self, frame_width: int) -> List[int]:
        """Left edges of the tiles of a frame ``frame_width`` px wide."""
        if frame_width <= self.width:
            return [0]
        xs = list(range(0, frame_width - self.width, self.stride))
        return xs + [frame_width - self.width]


def _iou_one(box: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IoU of ``box`` with each of ``boxes`` and the fraction of ``box`` inside each."""
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    a = np.prod(np.clip(box[2:] - box[:2], 0, None))
    a2 = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), 1)
    return inter / np.maximum(a + a2 - inter, 1e-9), inter / max(a, 1e-9)


def merge_tile_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    edge: np.ndarray,
    nms_iou: float = 0.6,
    contain: float = 0.7,
    max_detections: int = 300,
) -> np.ndarray:
    """Edge-aware greedy NMS over one frame's tile detections.

    Boxes are visited boxes not touching an inner tile edge first, then by score.
    A box is suppressed if its IoU with a kept box exceeds ``nms_iou``, or if it
    touches an inner tile edge (``edge``; it may be a truncated person) and more
    than ``contain`` of it lies inside a kept box.

    Returns:
        Indices of the kept boxes, by descending score.
    """
    order = np.lexsort((-scores, edge))
    kept: List[int] = []
    for i in order:
        if kept:
            iou, frac_in = _iou_one(boxes[i], boxes[kept])
            if (iou > nms_iou).any() or (edge[i] and (frac_in > contain).any()):
                continue
        kept.append(int(i))
        if len(kept) >= max_detections:
            break
    kept_arr = np.array(kept, int)
    return kept_arr[np.argsort(-scores[kept_arr], kind="stable")]


def load_bbox_layer(
    model_path: Union[str, Path],
    device: str = "cpu",
    min_score: float = 0.01,
    max_detections: int = 300,
):
    """Load a ``bbox_*`` model directory as a :class:`~sleap_nn.inference.layers.bbox.BBoxLayer`.

    ``min_score`` / ``max_detections`` become the layer's score threshold and
    per-image cap.

    Raises:
        ValueError: If the model is not a box detector.
    """
    from sleap_nn.inference.layers.bbox import BBoxLayer, BBoxStageLayer
    from sleap_nn.inference.predictor import Predictor

    predictor = Predictor.from_model_paths(
        [str(model_path)],
        device=device,
        peak_threshold=min_score,
        max_instances=max_detections,
    )
    layer = predictor.layer
    if not isinstance(layer, BBoxLayer) or isinstance(layer, BBoxStageLayer):
        raise ValueError(
            f"{model_path} is not a bounding-box detector (bbox_* model); got a "
            f"{type(layer).__name__}."
        )
    return layer


def _category_index(layer, category: str) -> int:
    names = list(layer.categories)
    if category in names:
        return names.index(category)
    if len(names) == 1:
        logger.info(f"Single-class detector {names}: using its class as {category!r}.")
        return 0
    raise ValueError(f"The detector has no {category!r} class (classes: {names}).")


def _layer_frame_detections(out, category_index: int):
    """Per-image ``(boxes (K, 4) float64, scores (K,))`` of one class from ``Outputs``."""
    boxes = out.pred_bboxes.detach().cpu().numpy().astype(np.float64)
    scores = out.pred_bbox_scores.detach().cpu().numpy()
    classes = out.pred_bbox_classes.detach().cpu().numpy()
    res = []
    for b in range(boxes.shape[0]):
        keep = np.isfinite(scores[b]) & (classes[b] == category_index)
        order = np.argsort(-scores[b][keep], kind="stable")
        res.append((boxes[b][keep][order], scores[b][keep][order].astype(np.float64)))
    return res


def _predict_chunked(layer, images: np.ndarray, batch_size: int, category_index: int):
    res = []
    for s in range(0, len(images), batch_size):
        res += _layer_frame_detections(
            layer.predict(images[s : s + batch_size]), category_index
        )
    return res


def _detect_tiled(
    layer,
    images: np.ndarray,
    tiling: TileSpec,
    batch_size: int,
    category_index: int,
    max_detections: int,
):
    """Tiled detections of a ``(B, H, W, C)`` batch of same-size frames."""
    W = images.shape[2]
    tw = min(tiling.width, W)
    xs = tiling.starts(W)
    tiles = np.stack([img[:, x0 : x0 + tw] for img in images for x0 in xs])
    per_tile = _predict_chunked(layer, tiles, batch_size, category_index)
    res = []
    for b in range(len(images)):
        bx_l, sc_l, ed_l = [], [], []
        for t, x0 in enumerate(xs):
            bx, sc = per_tile[b * len(xs) + t]
            bx = bx.copy()
            bx[:, [0, 2]] += x0
            e = np.zeros(len(bx), bool)
            if x0 > 0:
                e |= bx[:, 0] <= x0 + tiling.edge_margin
            if x0 + tw < W:
                e |= bx[:, 2] >= x0 + tw - tiling.edge_margin
            bx_l.append(bx), sc_l.append(sc), ed_l.append(e)
        bx = np.concatenate(bx_l).reshape(-1, 4)
        sc = np.concatenate(sc_l)
        k = merge_tile_detections(
            bx, sc, np.concatenate(ed_l), tiling.nms_iou, tiling.contain, max_detections
        )
        res.append((bx[k], sc[k]))
    return res


def _image_size(path: Optional[str]) -> Tuple[int, int]:
    """``(width, height)`` from an image file's header, ``(0, 0)`` if unreadable."""
    from PIL import Image

    if path is None:
        return (0, 0)
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return (0, 0)


def _size_ordered(labels: sio.Labels) -> sio.Labels:
    """A view of ``labels`` with the same frames grouped by video and image size."""
    from concurrent.futures import ThreadPoolExecutor

    frames = labels.labeled_frames
    with ThreadPoolExecutor(8) as pool:
        sizes = list(pool.map(_image_size, (frame_image_path(lf) for lf in frames)))
    vid_index = {id(v): i for i, v in enumerate(labels.videos)}
    order = sorted(
        range(len(frames)), key=lambda i: (vid_index[id(frames[i].video)], sizes[i])
    )
    return sio.Labels(
        labeled_frames=[frames[i] for i in order],
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
    )


def detect_with_layer(
    labels: sio.Labels,
    layer,
    detector: str,
    category: str = DEFAULT_CATEGORY,
    batch_size: int = 8,
    tiling: Optional[TileSpec] = None,
    max_detections: int = 300,
    order_by_size: bool = True,
    progress_every: int = 500,
) -> FrameDetections:
    """Run a box-detection layer over every labeled frame of ``labels``.

    Frames are decoded on a background thread (``LabelsProvider``) in batches of
    same-size frames; with ``order_by_size`` they are visited grouped by image
    size (read from the image file headers), so files of mixed-size photos (COCO,
    AIC) still fill their batches. Each image's detections of ``category`` (a single-class
    detector's only class whatever its name) are kept, by descending score. With
    ``tiling``, every frame is detected in full-height tiles (:class:`TileSpec`)
    and the tile boxes merged; the layer's own cap applies per tile and
    ``max_detections`` after merging.

    Args:
        labels: Labels to detect on (with openable videos).
        layer: A :class:`~sleap_nn.inference.layers.bbox.BBoxLayer` (see
            :func:`load_bbox_layer`).
        detector: Detector id for the boxes' ``source``.
        category: Detector class to keep.
        batch_size: Frames (or tiles) per forward pass.
        tiling: Tile wide frames, or ``None``.
        max_detections: Per-frame cap after merging tiles.
        order_by_size: Visit image-file frames grouped by image size.
        progress_every: Log progress every this many frames.

    Returns:
        The :class:`FrameDetections` (in ``labels.labeled_frames`` order whatever
        the visiting order); frames whose image could not be read have
        none (``info["n_unreadable_frames"]``).
    """
    from sleap_nn.inference.providers import LabelsProvider

    cls = _category_index(layer, category)
    frames = labels.labeled_frames
    vid_index = {id(v): i for i, v in enumerate(labels.videos)}
    pos: Dict[Tuple[int, int], int] = {}
    for i, lf in enumerate(frames):
        pos.setdefault((vid_index[id(lf.video)], int(lf.frame_idx)), i)
    boxes, scores = _empty_detections(len(frames))
    image_hw: List[Optional[Tuple[int, int]]] = [None] * len(frames)
    source = _size_ordered(labels) if order_by_size else labels
    provider = LabelsProvider(labels=source, batch_size=batch_size)
    t0, done, next_log = time.time(), 0, progress_every
    for batch in provider:
        images = np.asarray(batch.images)
        if tiling is not None:
            res = _detect_tiled(layer, images, tiling, batch_size, cls, max_detections)
        else:
            res = _predict_chunked(layer, images, batch_size, cls)
        for b, (bx, sc) in enumerate(res):
            i = pos[int(batch.video_indices[b]), int(batch.frame_indices[b])]
            boxes[i], scores[i] = bx, sc
            image_hw[i] = (int(images.shape[1]), int(images.shape[2]))
        done += len(images)
        if done >= next_log:
            next_log += progress_every
            rate = done / max(time.time() - t0, 1e-9)
            logger.info(f"detected {done}/{len(frames)} frames, {rate:.1f} frames/s")
    unreadable = sum(hw is None for hw in image_hw)
    return FrameDetections(
        boxes=boxes,
        scores=scores,
        detector=detector,
        image_hw=image_hw,
        info={
            "n_unreadable_frames": unreadable,
            "tiling": attrs.asdict(tiling) if tiling is not None else None,
        },
    )


# ---------------------------------------------------------------------------
# Matching (a port of the human project's detector/match.py)
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes ``a`` (N, 4) and ``b`` (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), 1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), 1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def keypoint_extent(inst: sio.Instance) -> Optional[np.ndarray]:
    """The xyxy extent of an instance's labeled points with coordinates (occluded too)."""
    pts = inst.numpy(invisible_as_nan=False)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) == 0:
        return None
    return np.array([*pts.min(0), *pts.max(0)], np.float64)


def _is_category(box: sio.BoundingBox, category: str) -> bool:
    return box.category is not None and box.category.name == category


def gt_people(
    lf: sio.LabeledFrame, category: str = DEFAULT_CATEGORY
) -> Tuple[np.ndarray, List[Optional[sio.Instance]], List[str]]:
    """Ground-truth people of a frame, the targets detections are matched to.

    Every non-crowd ``category`` user box (kind ``"box"`` when it is linked to a
    pose instance, else ``"nokpt"``), then the keypoint extent of each user pose
    instance with no linked box and at least one located point
    (``"kpt_extent"``).

    Returns:
        ``(boxes (G, 4), instance-or-None per row, kind per row)``.
    """
    boxes, insts, kinds = [], [], []
    linked = set()
    for b in lf.bboxes:
        if (
            isinstance(b, sio.UserBoundingBox)
            and _is_category(b, category)
            and not is_crowd_bbox(b)
        ):
            boxes.append([b.x1, b.y1, b.x2, b.y2])
            insts.append(b.instance)
            kinds.append("box" if b.instance is not None else "nokpt")
            if b.instance is not None:
                linked.add(id(b.instance))
    for inst in lf.user_instances:
        if id(inst) in linked:
            continue
        ext = keypoint_extent(inst)
        if ext is None:
            continue
        boxes.append(ext.tolist())
        insts.append(inst)
        kinds.append("kpt_extent")
    return np.array(boxes, np.float64).reshape(-1, 4), insts, kinds


def match_frame(
    gt: np.ndarray, det: np.ndarray, min_iou: float = DEFAULT_MIN_IOU
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian on IoU; ``(gt_idx, det_idx, iou)`` of the pairs with IoU > ``min_iou``."""
    iou = iou_matrix(gt, det)
    if iou.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] > min_iou
    return r[keep], c[keep], iou[r[keep], c[keep]]


def summarize_matches(rows: Sequence[Tuple[bool, float, str]]) -> dict:
    """Match statistics over the pose instances: ``rows`` of (matched, IoU, kind)."""
    m = np.array([r[0] for r in rows], bool)
    iou = np.array([r[1] for r in rows], np.float64)
    kinds = np.array([r[2] for r in rows])
    out = {
        "people": int(len(m)),
        "matched": int(m.sum()),
        "unmatched": int((~m).sum()),
        "unmatched_frac": float((~m).mean()) if len(m) else 0.0,
        "recall_iou50": float((iou >= 0.5).mean()) if len(m) else 0.0,
        "recall_iou75": float((iou >= 0.75).mean()) if len(m) else 0.0,
        "mean_iou_matched": float(iou[m].mean()) if m.any() else 0.0,
        "median_iou_matched": float(np.median(iou[m])) if m.any() else 0.0,
    }
    by_kind = {}
    for k in np.unique(kinds):
        s = kinds == k
        by_kind[str(k)] = {
            "people": int(s.sum()),
            "recall_iou50": float((iou[s] >= 0.5).mean()),
            "mean_iou_matched": float(iou[s & m].mean()) if (s & m).any() else 0.0,
            "unmatched": int((s & ~m).sum()),
        }
    out["by_gt_box_kind"] = by_kind
    return out


def shared_category(labels: sio.Labels, name: str) -> sio.Category:
    """The labels' ``Category`` called ``name`` (one shared object), else a new one."""
    found = next((c for c in labels.categories if c.name == name), None)
    return found if found is not None else sio.Category(name=name)


def match_detections(
    labels: sio.Labels,
    detections: FrameDetections,
    category: str = DEFAULT_CATEGORY,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_iou: float = DEFAULT_MIN_IOU,
    add_boxes: bool = True,
    replace_predicted: bool = False,
) -> dict:
    """Match detections to each frame's ground-truth people and add them, in place.

    See the module docstring for the rule. Detections scoring below
    ``score_threshold`` are neither matched nor stored.

    Args:
        labels: Labels to add the boxes to (modified in place).
        detections: Detections aligned with ``labels.labeled_frames``.
        category: Ground-truth person category, also the new boxes' category.
        score_threshold: Minimum detection score to match and store.
        min_iou: Keep matched pairs with IoU above this.
        add_boxes: Add the boxes (``False``: statistics only; ``boxes_added``
            then counts the boxes that would be stored).
        replace_predicted: Remove every existing ``PredictedBoundingBox`` first.

    Returns:
        Match statistics (also what :func:`write_detector_labels` stores).
    """
    frames = labels.labeled_frames
    if len(detections) != len(frames):
        raise ValueError(
            f"{len(detections)} detection frames for {len(frames)} labeled frames."
        )
    person = shared_category(labels, category)
    rows: List[Tuple[bool, float, str]] = []
    n_added = n_linked = n_nokpt_matched = oob = n_removed = 0
    for i, lf in enumerate(frames):
        scores = np.asarray(detections.scores[i])
        keep = scores >= score_threshold
        det_boxes = np.asarray(detections.boxes[i], np.float64)[keep]
        det_scores = scores[keep]

        gt, insts, kinds = gt_people(lf, category)
        gi, di, ious = match_frame(gt, det_boxes, min_iou)
        det_to_inst, matched, det_iou = {}, {}, {}
        for g, d, v in zip(gi, di, ious):
            matched[g] = v
            det_iou[d] = float(v)
            if insts[g] is not None:
                det_to_inst[d] = insts[g]
            else:
                n_nokpt_matched += 1
        hw = detections.image_hw[i] if detections.image_hw is not None else None
        for g, (inst, kind) in enumerate(zip(insts, kinds)):
            if inst is None:
                continue
            rows.append((g in matched, matched.get(g, 0.0), kind))
            ext = keypoint_extent(inst)
            if hw is not None and ext is not None:
                H, W = hw
                if ext[2] > W + 2 or ext[3] > H + 2:
                    oob += 1

        n_added += len(det_boxes)
        n_linked += len(det_to_inst)
        if not add_boxes:
            continue
        if replace_predicted:
            n_before = len(lf.bboxes)
            lf.bboxes = [b for b in lf.bboxes if not b.is_predicted]
            n_removed += n_before - len(lf.bboxes)
        for d in range(len(det_boxes)):
            x1, y1, x2, y2 = (float(v) for v in det_boxes[d])
            inst = det_to_inst.get(d)
            lf.bboxes.append(
                sio.PredictedBoundingBox(
                    x1,
                    y1,
                    x2,
                    y2,
                    instance=inst,
                    category=person,
                    score=float(det_scores[d]),
                    source=detections.detector,
                    # IoU with the matched GT person (None: unmatched detection).
                    metadata={"match_iou": det_iou.get(d)},
                )
            )

    return {
        "detector": detections.detector,
        "category": category,
        "det_thr": score_threshold,
        "min_iou": min_iou,
        "rule": "Hungarian on IoU per image, pairs kept if IoU > min_iou (D4)",
        "gt_boxes": "linked person box, else keypoint extent; non-crowd keypoint-less "
        "person boxes are matching targets but not counted",
        "frames": len(frames),
        "boxes_added": n_added,
        "boxes_linked": n_linked,
        "predicted_boxes_removed": n_removed,
        "dets_matched_to_kptless_people": n_nokpt_matched,
        "people_kpts_outside_image": oob if detections.image_hw is not None else None,
        "detections": detections.info,
        **summarize_matches(rows),
    }


def _labels_name(labels_path: Union[str, Path]) -> str:
    name = Path(labels_path).name
    return name[: -len(".slp")] if name.endswith(".slp") else Path(name).stem


def default_output_path(labels_path: Union[str, Path], tag: str) -> Path:
    """``<dir>/<name>.<tag>.slp`` for the labels file ``<dir>/<name>.slp``."""
    return Path(labels_path).with_name(f"{_labels_name(labels_path)}.{tag}.slp")


def write_detector_labels(
    labels: sio.Labels,
    stats: dict,
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Save labels with detector boxes, their statistics and the source's ``/human_meta``.

    Stores ``stats`` in ``provenance["detector_boxes"]`` (and the source path in
    ``provenance["detector_boxes_source"]``), saves ``output_path`` without
    embedding images, copies the source file's ``/human_meta`` group verbatim
    (frames keep their order, so its per-frame rows still apply) and writes
    ``stats`` to ``output_path`` with a ``.json`` suffix.

    Returns:
        The stats JSON path.

    Raises:
        FileExistsError: If ``output_path`` exists and ``overwrite`` is not set.
        ValueError: If the boxes carry more distinct ``Category`` objects than the
            labels have categories (a per-box category).
    """
    import h5py

    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("The output would overwrite the input labels.")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists (pass overwrite to replace).")
        output_path.unlink()
    n_cats = len(
        {id(b.category) for lf in labels for b in lf.bboxes if b.category is not None}
    )
    if n_cats > len(labels.categories) + 1:
        raise ValueError(f"{n_cats} distinct box Category objects: share one.")
    labels.provenance["detector_boxes"] = stats
    labels.provenance["detector_boxes_source"] = str(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(output_path), embed=False)
    with h5py.File(source_path, "r") as fs, h5py.File(output_path, "a") as fo:
        if "human_meta" in fs and "human_meta" not in fo:
            fs.copy("human_meta", fo)
    stats_path = output_path.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=1))
    return stats_path


def detect_boxes(
    labels_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    model: Optional[Union[str, Path]] = None,
    detections: Optional[Union[str, Path]] = None,
    detection_chunks: Optional[Union[str, Path]] = None,
    image_list: Optional[Union[str, Path]] = None,
    detector: Optional[str] = None,
    tag: Optional[str] = None,
    category: str = DEFAULT_CATEGORY,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_iou: float = DEFAULT_MIN_IOU,
    json_category_id: Optional[int] = 1,
    device: str = "cpu",
    batch_size: int = 8,
    min_score: float = 0.01,
    max_detections: int = 300,
    tiling: Optional[TileSpec] = None,
    replace_predicted: bool = False,
    save_detections: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    layer_factory: Optional[Callable] = None,
) -> Tuple[dict, Optional[Path]]:
    """Detect (or load) person boxes for a labels file, match them and write the result.

    Exactly one detection source: ``model`` (a ``bbox_*`` model directory),
    ``detections`` (a COCO results JSON) or ``detection_chunks`` (+
    ``image_list``). The output defaults to ``<name>.<tag>.slp`` next to the input
    (``tag``: ``bboxdetr`` for a model, ``dets`` otherwise).

    Args:
        labels_path: Input labels ``.slp``.
        output_path: Output ``.slp``.
        model: Box-detector model directory to run over the labeled frames.
        detections: COCO detection-results JSON.
        detection_chunks: Directory of ``chunk_*.npz`` detection files.
        image_list: Image list of ``detection_chunks``.
        detector: Detector id (box ``source``); defaults to the model directory
            name, the JSON stem or the chunks' recorded model id.
        tag: Output name tag.
        category: Person category (ground truth, detector class and new boxes).
        score_threshold: Minimum score to match and store.
        min_iou: Keep matched pairs with IoU above this.
        json_category_id: COCO category id kept from ``detections``.
        device: Torch device for ``model``.
        batch_size: Frames (or tiles) per forward pass.
        min_score: Detector score floor (``model`` only; also what
            ``save_detections`` keeps).
        max_detections: Per-image detection cap (``model`` only).
        tiling: Tile wide frames (``model`` only).
        replace_predicted: Remove existing predicted boxes first.
        save_detections: Also write the raw detections (every score above
            ``min_score``) as a COCO results JSON, to re-match without the model.
        dry_run: Statistics only; write nothing.
        overwrite: Replace an existing output.
        layer_factory: ``(model, device, min_score, max_detections) -> BBoxLayer``
            override (tests).

    Returns:
        ``(stats, output path or None for a dry run)``.
    """
    sources = [s is not None for s in (model, detections, detection_chunks)]
    if sum(sources) != 1:
        raise ValueError("Give exactly one of model, detections, detection_chunks.")
    if detection_chunks is not None and image_list is None:
        raise ValueError("detection_chunks needs image_list.")
    if tiling is not None and model is None:
        raise ValueError("Tiling applies to a model run only.")
    tag = tag or ("bboxdetr" if model is not None else "dets")
    out = Path(output_path) if output_path else default_output_path(labels_path, tag)
    if not dry_run and out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass overwrite to replace).")

    t0 = time.time()
    labels = sio.load_slp(str(labels_path), open_videos=model is not None)
    logger.info(f"Loaded {len(labels)} frames from {labels_path}.")
    if model is not None:
        factory = layer_factory or load_bbox_layer
        layer = factory(model, device, min_score, max_detections)
        dets = detect_with_layer(
            labels,
            layer,
            detector=detector or Path(model).resolve().name,
            category=category,
            batch_size=batch_size,
            tiling=tiling,
            max_detections=max_detections,
        )
    elif detections is not None:
        dets = load_coco_detections(
            labels, detections, category_id=json_category_id, detector=detector
        )
    else:
        dets = load_detection_chunks(
            labels, detection_chunks, image_list, detector=detector
        )
    n_dets = int(sum(len(s) for s in dets.scores))
    logger.info(
        f"{n_dets} detections for {len(dets)} frames ({time.time() - t0:.0f}s)."
    )
    if save_detections is not None:
        n = save_coco_detections(labels, dets, save_detections)
        logger.info(f"Wrote {n} detections to {save_detections}.")

    stats = match_detections(
        labels,
        dets,
        category=category,
        score_threshold=score_threshold,
        min_iou=min_iou,
        add_boxes=not dry_run,
        replace_predicted=replace_predicted,
    )
    stats = {"dataset": _labels_name(labels_path), **stats}
    if dry_run:
        return stats, None
    write_detector_labels(labels, stats, labels_path, out, overwrite=overwrite)
    logger.info(f"Wrote {out} ({time.time() - t0:.0f}s).")
    return stats, out
