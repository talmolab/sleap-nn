"""``PoseSeededTopDownLayer`` -- run a box detector only when needed on video (experimental).

A box-seeded top-down pipeline (a ``bbox_*`` detector, then a centered-instance
model on box-scaled crops) runs its detector on every frame. On video, most
frames could instead reuse what the previous frame found: each animal's crop box
for frame ``t`` is built from its own predicted keypoints at frame ``t - 1``,
and the detector runs only

* on the first frame of a stream (a video, or after a jump in frame numbers),
* on a schedule, every ``detect_every`` frames,
* on the next frame when an animal is lost (its points' mean score falls below
  ``lost_score`` and below ``lost_ratio`` x its mean score at its last detection,
  or its box is more than ``max_outside`` outside the frame),
* when two carried boxes converge on one animal (IoU above ``dup_iou``; the
  weaker one is dropped at once),
* when fewer animals are carried than the last detector run kept.

The box from keypoints is the tight box of the animal's predicted points (with
``box_points="conf"`` only those scoring at least ``conf_threshold``, when at
least ``min_points`` do), padded on every side by ``(box_scale - 1) / 2`` x its
longest side, like the labelled boxes detectors are trained on. Fit
``box_scale`` so these boxes match the detector's (the detector box's longest side
over the tight box's longest side, for the same animals' predictions). A square
box-scaled crop (``crop_mode="bbox"``, ``crop_bbox_aspect: square``) reads only a
box's centre and longest side.

With ``async_detect`` (live video) a due detector run goes to a background
thread on its own CUDA stream and the frame is cropped from the carried boxes;
when the detector's boxes are ready (one or more frames later) they are matched
to the boxes the animals had when it was asked for, and unmatched ones start new
animals. Only a stream with nothing to carry (its first frame, or every animal
lost) waits for the detector. Outputs then depend on timing, so runs are not
repeatable; this is for frames arriving live, where a detector frame would
otherwise stall the stream.

On a detector frame the outputs are exactly the plain pipeline's; with
``keep_unmatched``, carried animals that no detection matches (IoU below
``match_iou``) are also cropped from their own boxes and kept unless lost or
duplicated. On the other frames the boxes in the outputs are the boxes built
from keypoints, scored with the animal's last detector score.

Experimental: on some videos (small animals, crowds) carried crops make more
head-to-tail flips and jitter than running the detector on every frame; check a
clip against the plain pipeline before relying on it.

Frames are tied into streams by ``frame_indices`` / ``video_indices``: a frame
continues the stream whose last frame is at most ``max_gap`` frames before it
in the same video. Frames of one stream are processed in order (one round per
frame); frames of different streams in one batch are processed together.
"""

from __future__ import annotations

import copy
from collections import Counter
from contextlib import nullcontext as _nullcontext
from typing import Any, Dict, List, Optional

import attrs
import numpy as np
import torch

from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.outputs import Outputs

# Outputs fields with an instance axis (dim 1) that are padded when frames with
# different instance counts are stacked back into one batch.
_INSTANCE_FIELDS = (
    "crops",
    "pred_keypoints",
    "pred_crop_keypoints",
    "pred_peak_values",
    "pred_centroids",
    "pred_centroid_values",
    "pred_bboxes",
    "pred_bbox_scores",
    "pred_bbox_classes",
    "instance_scores",
    "instance_valid",
    "instance_bboxes",
    "instance_tracking_scores",
    "pred_class_vectors",
    "pred_class_inds",
    "pred_class_probs",
    "pred_embeddings",
)


@attrs.define
class SeedConfig:
    """Settings of :class:`PoseSeededTopDownLayer` (see the module docstring)."""

    detect_every: int = attrs.field(default=30)
    lost_score: float = 0.25
    lost_ratio: float = 0.75
    max_outside: float = 0.5
    dup_iou: float = 0.6
    box_scale: float = 1.19
    box_points: str = attrs.field(default="conf")
    conf_threshold: float = 0.2
    min_points: int = 4
    motion: bool = False
    max_gap: int = 1
    match_iou: float = 0.3
    keep_unmatched: bool = False
    async_detect: bool = False
    emit_tracks: bool = False
    fix_flips: bool = False
    flip_nodes: tuple = ("nose", "tail_base")
    flip_margin: float = 0.1

    @detect_every.validator
    def _check_every(self, attribute, value):
        if int(value) < 1:
            raise ValueError(f"detect_every must be >= 1; got {value}.")

    @box_points.validator
    def _check_points(self, attribute, value):
        if value not in ("all", "conf"):
            raise ValueError(f"box_points must be 'all' or 'conf'; got {value!r}.")

    @classmethod
    def from_any(cls, value: Any) -> "SeedConfig":
        """A ``SeedConfig`` from itself, a dict, or an int (``detect_every``)."""
        if isinstance(value, SeedConfig):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return cls(detect_every=int(value))
        raise TypeError(f"Cannot build a SeedConfig from {type(value).__name__}.")

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict (for provenance)."""
        d = attrs.asdict(self)
        d["flip_nodes"] = list(d["flip_nodes"])
        return d


def box_from_points(
    points: np.ndarray, scores: Optional[np.ndarray], config: SeedConfig
) -> Optional[np.ndarray]:
    """``(4,)`` xyxy crop box from one instance's ``(N, 2)`` points, or ``None``.

    The tight box of the chosen points, padded on every side by
    ``(box_scale - 1) / 2`` x its longest side: its longest side is ``box_scale`` x
    the tight box's, and its centre is the tight box's (all that a square
    box-scaled crop reads), while its shape stays a detector-like rectangle.
    """
    pts = np.asarray(points, dtype=np.float64)
    ok = np.isfinite(pts).all(-1)
    if config.box_points == "conf" and scores is not None:
        s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
        sel = ok & (s >= config.conf_threshold)
        if sel.sum() >= config.min_points:
            ok = sel
    if ok.sum() < 2:
        return None
    lo, hi = pts[ok].min(0), pts[ok].max(0)
    side = float((hi - lo).max())
    if side <= 0:
        return None
    pad = (config.box_scale - 1.0) / 2 * side
    return np.array([lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _outside_fraction(box: np.ndarray, height: int, width: int) -> float:
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    w = max(0.0, min(box[2], width) - max(box[0], 0.0))
    h = max(0.0, min(box[3], height) - max(box[1], 0.0))
    return 1.0 - (w * h) / area


@attrs.define
class _Animal:
    box: np.ndarray  # crop box for the next frame (xyxy)
    score: float  # last detector score
    ref_score: float  # mean point score at the last detection
    center: Optional[np.ndarray] = None  # tight-box centre at this frame
    points: Optional[np.ndarray] = None  # (N, 2) output points at this frame
    scores: Optional[np.ndarray] = None  # (N,) their scores
    track_id: int = -1


@attrs.define
class _Stream:
    video: int
    last_frame: int
    animals: List[_Animal] = attrs.field(factory=list)
    since_detect: int = 0
    detect_count: int = 0
    detect_next: bool = True
    reason: str = "start"
    pending: Any = None  # async_detect: the detector's future
    pending_frame: int = -1
    pending_carried: Optional[np.ndarray] = None  # carried boxes when it was asked for
    pending_tracks: List[int] = attrs.field(factory=list)  # and their animals' ids


def _take(out: Outputs, i: int) -> Outputs:
    """Batch slot ``i`` of ``out`` as a one-frame ``Outputs``."""
    kw = {}
    for f in attrs.fields(Outputs):
        v = getattr(out, f.name)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            kw[f.name] = v[i : i + 1]
        elif f.name in ("frame_indices", "video_indices", "preprocess_info"):
            kw[f.name] = None
        else:
            kw[f.name] = v
    return Outputs(**kw)


def _pad_value(t: torch.Tensor):
    if t.dtype == torch.bool:
        return False
    if t.is_floating_point():
        return float("nan")
    return -1


def _cat_instances(a: Outputs, b: Outputs) -> Outputs:
    """One-frame ``a`` with ``b``'s instance slots appended after its own."""
    kw: Dict[str, Any] = {}
    for f in attrs.fields(Outputs):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if (
            f.name in _INSTANCE_FIELDS
            and isinstance(va, torch.Tensor)
            and isinstance(vb, torch.Tensor)
            and va.ndim >= 2
        ):
            kw[f.name] = torch.cat([va, vb.to(va.device)], dim=1)
        elif f.name in _INSTANCE_FIELDS and isinstance(va, torch.Tensor):
            pad = torch.full(
                (1, _n_slots(b)) + tuple(va.shape[2:]),
                _pad_value(va),
                dtype=va.dtype,
                device=va.device,
            )
            kw[f.name] = torch.cat([va, pad], dim=1)
        else:
            kw[f.name] = va
    return Outputs(**kw)


def _n_slots(o: Outputs) -> int:
    """Instance slots of a one-frame ``Outputs``."""
    return 0 if o.pred_keypoints is None else int(o.pred_keypoints.shape[1])


def _stack(frames: List[Outputs]) -> Outputs:
    """Stack one-frame ``Outputs`` into one batch, padding the instance axis."""
    kw: Dict[str, Any] = {}
    for f in attrs.fields(Outputs):
        vals = [getattr(o, f.name) for o in frames]
        tensors = [v for v in vals if isinstance(v, torch.Tensor)]
        if not tensors:
            kw[f.name] = vals[0] if all(v is vals[0] for v in vals) else None
            continue
        if len(tensors) != len(vals):
            raise ValueError(f"Outputs.{f.name} is set on some frames only.")
        device = tensors[0].device
        if f.name in _INSTANCE_FIELDS and tensors[0].ndim >= 2:
            n = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                t = t.to(device)
                if t.shape[1] < n:
                    pad = torch.full(
                        (t.shape[0], n - t.shape[1]) + tuple(t.shape[2:]),
                        _pad_value(t),
                        dtype=t.dtype,
                        device=device,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            kw[f.name] = torch.cat(padded, dim=0)
        else:
            kw[f.name] = torch.cat([t.to(device) for t in tensors], dim=0)
    return Outputs(**kw)


class PoseSeededTopDownLayer(TopDownLayer):
    """A box-seeded :class:`TopDownLayer` that runs its detector only when needed.

    Args:
        base: The box-seeded top-down layer (detector stage 1, box-scaled crops).
        seed_stage: A given-box stage 1
            (:class:`~sleap_nn.inference.layers.bbox.BBoxStageLayer` with
            ``use_given_boxes=True``) that feeds the boxes built from keypoints.
        config: :class:`SeedConfig`.
    """

    def __init__(self, base: TopDownLayer, seed_stage: Any, config: SeedConfig):
        """Take over ``base``'s stages and crop settings."""
        super().__init__(
            centroid_layer=base.centroid_layer,
            centered_instance_layer=base.centered_instance_layer,
            crop_size=base.crop_size,
            centroid_nms=base.centroid_nms,
            centroid_nms_threshold=base.centroid_nms_threshold,
            return_crops=base.return_crops,
            bbox_crop=base.bbox_crop,
            crop_max_hw=base.crop_max_hw,
        )
        self.seed_stage = seed_stage
        self.seed_config = config
        self.reset_streams()

    def reset_streams(self) -> None:
        """Forget every stream (call at the start of each prediction run)."""
        self._streams: List[_Stream] = []
        self.seed_stats: Counter = Counter()
        # (video index, frame index) -> track id of each output slot (-1: none)
        self.track_log: Dict[tuple, List[int]] = {}
        self._next_track = 0
        if getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False)
        self._executor = None
        self._det_stream = None

    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        image,
        instances: Optional[torch.Tensor] = None,
        instance_boxes: Optional[torch.Tensor] = None,
        frame_boxes: Optional[torch.Tensor] = None,
        frame_box_scores: Optional[torch.Tensor] = None,
        frame_indices=None,
        video_indices=None,
    ) -> Outputs:
        """Predict a batch of frames, seeding crops from earlier frames' keypoints.

        Without ``frame_indices`` every frame runs the detector (the plain
        :class:`TopDownLayer`).
        """
        if frame_indices is None:
            return super().predict(
                image,
                instances=instances,
                instance_boxes=instance_boxes,
                frame_boxes=frame_boxes,
                frame_box_scores=frame_box_scores,
            )
        fidx = np.asarray(
            frame_indices.cpu() if torch.is_tensor(frame_indices) else frame_indices
        ).astype(np.int64)
        n = len(fidx)
        vidx = (
            np.zeros(n, dtype=np.int64)
            if video_indices is None
            else np.asarray(
                video_indices.cpu() if torch.is_tensor(video_indices) else video_indices
            ).astype(np.int64)
        )
        height, width = self.centroid_layer._to_4d_tensor(image).shape[-2:]

        # Assign each frame to a stream and a round (its position in that stream
        # within this batch), so that frames of one stream run in order.
        stream_of: List[_Stream] = []
        rounds: List[int] = []
        depth: Dict[int, int] = {}
        for i in range(n):
            s = self._find_stream(int(vidx[i]), int(fidx[i]))
            if s is None:
                s = _Stream(video=int(vidx[i]), last_frame=int(fidx[i]) - 1)
                self._streams.append(s)
            # Within the batch the stream advances as frames are claimed.
            r = depth.get(id(s), 0)
            depth[id(s)] = r + 1
            s.last_frame = int(fidx[i])  # claim this frame for the stream
            stream_of.append(s)
            rounds.append(r)

        per_frame: List[Optional[Outputs]] = [None] * n
        for r in range(max(rounds) + 1 if rounds else 0):
            idx = [i for i in range(n) if rounds[i] == r]
            if self.seed_config.async_detect:
                for i in idx:
                    self._async_step(stream_of[i], image, i, int(fidx[i]))
            det = [i for i in idx if stream_of[i].detect_next]
            seed = [i for i in idx if not stream_of[i].detect_next]
            if det:
                out = TopDownLayer.predict(self, self._subset(image, det))
                outs = [_take(out, k) for k in range(len(det))]
                n_det = [
                    o.pred_keypoints.shape[1] if o.pred_keypoints is not None else 0
                    for o in outs
                ]
                extras = [[] for _ in det]
                if self.seed_config.keep_unmatched:
                    extras = [
                        self._unmatched(stream_of[i], outs[k])
                        for k, i in enumerate(det)
                    ]
                    ks = [k for k in range(len(det)) if extras[k]]
                    if ks:
                        x = self._seeded(
                            image, [det[k] for k in ks], [extras[k] for k in ks]
                        )
                        for m, k in enumerate(ks):
                            outs[k] = _cat_instances(outs[k], _take(x, m))
                for k, i in enumerate(det):
                    per_frame[i] = outs[k]
                    self._update(
                        stream_of[i],
                        outs[k],
                        True,
                        height,
                        width,
                        vouched=n_det[k],
                        extra_prev=extras[k],
                        frame=int(fidx[i]),
                    )
            if seed:
                out = self._seeded(image, seed, [stream_of[i].animals for i in seed])
                for k, i in enumerate(seed):
                    per_frame[i] = _take(out, k)
                    self._update(
                        stream_of[i],
                        per_frame[i],
                        False,
                        height,
                        width,
                        frame=int(fidx[i]),
                    )
        if len(self._streams) > 4096:
            self._streams = self._streams[-4096:]
        return _stack(per_frame)

    # ──────────────────────────────────────────────────────────────────

    def _detect_boxes(self, frame) -> tuple:
        """The detector on one frame (background thread): ``(boxes (K, 4), scores (K,))``."""
        stream = self._det_stream
        ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            out = self.centroid_layer.predict(frame)
            if out.pred_bboxes is None or out.pred_centroids is None:
                return np.zeros((0, 4)), np.zeros((0,))
            valid = ~torch.isnan(out.pred_centroids).any(dim=-1)
            if self.centroid_nms:  # as TopDownLayer.predict keeps them
                valid = valid & self._centroid_nms_mask(
                    out.pred_centroids, out.pred_centroid_values, valid
                )
            valid = valid[0].cpu().numpy()
            boxes = out.pred_bboxes[0].detach().float().cpu().numpy()
            scores = out.pred_bbox_scores[0].detach().float().cpu().numpy()
        ok = valid & np.isfinite(boxes).all(-1) & np.isfinite(scores)
        return boxes[ok], scores[ok]

    def _async_step(self, s: _Stream, image, i: int, frame: int) -> None:
        """async_detect: merge a finished detector run, or start one that is due."""
        if s.pending is not None and s.pending.done():
            boxes, scores = s.pending.result()
            self._merge_detections(s, boxes, scores)
            s.pending = None
            s.detect_next, s.reason = False, ""
        if not s.detect_next or not s.animals:
            return  # nothing due, or nothing to carry: the detector runs in line
        if s.pending is None:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(max_workers=1)
                if torch.cuda.is_available():
                    self._det_stream = torch.cuda.Stream()
            frame_img = self._subset(image, [i])
            frame_img = (
                frame_img.copy()
                if isinstance(frame_img, np.ndarray)
                else frame_img.clone()
            )
            s.pending = self._executor.submit(self._detect_boxes, frame_img)
            s.pending_frame = frame
            s.pending_carried = np.stack([a.box for a in s.animals])
            s.pending_tracks = [a.track_id for a in s.animals]
            self.seed_stats["detector_frames"] += 1
            self.seed_stats[f"detector_{s.reason}"] += 1
            self.seed_stats["detector_async"] += 1
            s.since_detect = -1  # this frame's update counts it as the detector frame
        s.detect_next = False  # this frame is cropped from the carried boxes

    def _merge_detections(
        self, s: _Stream, boxes: np.ndarray, scores: np.ndarray
    ) -> None:
        """Merge a background detector run into the carried animals.

        A detection matched to a carried box (as it was when the detector was
        asked) re-anchors that animal: its next crop takes the detector's box,
        moved by how far the animal has moved since. Detections no carried animal
        accounts for start new animals; carried animals the detector missed stay
        (as with ``keep_unmatched``).
        """
        carried = (
            s.pending_carried if s.pending_carried is not None else np.zeros((0, 4))
        )
        matched = set()
        if len(boxes) and len(carried):
            from scipy.optimize import linear_sum_assignment

            iou = _iou_matrix(boxes, carried)
            rows, cols = linear_sum_assignment(-iou)
            by_track = {a.track_id: a for a in s.animals if a.track_id >= 0}
            for r, c in zip(rows, cols):
                if iou[r, c] < self.seed_config.match_iou:
                    continue
                matched.add(r)
                a = by_track.get(s.pending_tracks[c]) if c < len(s.pending_tracks) else None
                if a is None:
                    continue  # lost since
                shift = (a.box[:2] + a.box[2:]) / 2 - (carried[c, :2] + carried[c, 2:]) / 2
                a.box = boxes[r].astype(np.float64) + np.tile(shift, 2)
                a.score = float(scores[r])
                self.seed_stats["async_reanchored"] += 1
        now = np.stack([a.box for a in s.animals]) if s.animals else np.zeros((0, 4))
        added = 0
        for k in range(len(boxes)):
            if k in matched:
                continue
            if len(now) and _iou_matrix(boxes[k : k + 1], now).max() > self.seed_config.dup_iou:
                continue  # already carried under another box
            s.animals.append(  # its reference score is set by its first pose
                _Animal(
                    box=boxes[k].astype(np.float64),
                    score=float(scores[k]),
                    ref_score=float("nan"),
                )
            )
            added += 1
        self.seed_stats["async_added"] += added
        s.detect_count = len(s.animals)

    def _seeded(self, image, idx: List[int], animals: List[List[_Animal]]) -> Outputs:
        """The pose stage on frames ``idx``, cropped from these animals' boxes."""
        kmax = max(1, max(len(a) for a in animals))
        boxes = np.full((len(idx), kmax, 4), np.nan, dtype=np.float32)
        scores = np.full((len(idx), kmax), np.nan, dtype=np.float32)
        for k, group in enumerate(animals):
            for j, a in enumerate(group):
                boxes[k, j] = a.box
                scores[k, j] = a.score
        view = copy.copy(self)
        view.centroid_layer = self.seed_stage
        return TopDownLayer.predict(
            view,
            self._subset(image, idx),
            frame_boxes=torch.from_numpy(boxes),
            frame_box_scores=torch.from_numpy(scores),
        )

    def _unmatched(self, s: _Stream, out: Outputs) -> List[_Animal]:
        """Carried animals that no detection of this frame matches."""
        if not s.animals:
            return []
        carried = np.stack([a.box for a in s.animals])
        det = (
            out.pred_bboxes[0].detach().float().cpu().numpy()
            if out.pred_bboxes is not None
            else np.zeros((0, 4))
        )
        det = det[np.isfinite(det).all(-1)]
        if not len(det):
            return list(s.animals)
        from scipy.optimize import linear_sum_assignment

        iou = _iou_matrix(carried, det)
        rows, cols = linear_sum_assignment(-iou)
        matched = {
            r for r, c in zip(rows, cols) if iou[r, c] >= self.seed_config.match_iou
        }
        return [a for j, a in enumerate(s.animals) if j not in matched]

    @staticmethod
    def _subset(image, idx: List[int]):
        if isinstance(image, np.ndarray):
            return image[idx]
        if torch.is_tensor(image):
            return image[torch.as_tensor(idx, device=image.device)]
        return [image[i] for i in idx]

    def _find_stream(self, video: int, frame: int) -> Optional[_Stream]:
        gap = max(1, int(self.seed_config.max_gap))
        for s in reversed(self._streams):
            if s.video == video and 0 < frame - s.last_frame <= gap:
                return s
        return None

    flip_index: Optional[tuple] = (
        None  # (head, tail) node indices, set by the predictor
    )

    @staticmethod
    def _drop_slot(out: Outputs, i: int) -> None:
        """Blank instance slot ``i`` of a one-frame ``Outputs`` (it is not written)."""
        for name in _INSTANCE_FIELDS:
            t = getattr(out, name)
            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[1] > i:
                t[0, i] = _pad_value(t)

    def _previous(
        self, old: List[_Animal], kpts: np.ndarray, vals: np.ndarray, detected: bool
    ) -> Dict[int, _Animal]:
        """The carried animal each output slot continues, by slot or by box IoU.

        On a seeded frame slot ``i`` is carried animal ``i``. On a detector frame
        the detections' boxes are matched to the boxes the carried animals would
        have used (Hungarian, IoU >= ``match_iou``).
        """
        if not old:
            return {}
        if not detected:
            return {i: old[i] for i in range(min(len(old), kpts.shape[0]))}
        boxes, slots = [], []
        for i in range(kpts.shape[0]):
            if np.isnan(kpts[i]).all():
                continue
            b = box_from_points(kpts[i], vals[i], self.seed_config)
            if b is not None:
                boxes.append(b)
                slots.append(i)
        if not boxes:
            return {}
        from scipy.optimize import linear_sum_assignment

        iou = _iou_matrix(np.stack(boxes), np.stack([a.box for a in old]))
        rows, cols = linear_sum_assignment(-iou)
        return {
            slots[r]: old[c]
            for r, c in zip(rows, cols)
            if iou[r, c] >= self.seed_config.match_iou
        }

    def _fix_flip(
        self, pts: np.ndarray, vals: np.ndarray, prev: _Animal
    ) -> Optional[np.ndarray]:
        """``pts`` turned 180 degrees if they flipped head to tail since ``prev``.

        A flip: the head point lands near the previous tail point and the tail
        point near the previous head point (closer swapped than kept, by half).
        It is undone (points rotated 180 degrees about the head-tail midpoint)
        unless the head and tail points now score higher than before by
        ``flip_margin`` (then the earlier frame was the wrong one).
        """
        h, t = self.flip_index
        q = prev.points
        if q is None or not (
            np.isfinite(pts[[h, t]]).all() and np.isfinite(q[[h, t]]).all()
        ):
            return None
        same = np.linalg.norm(pts[h] - q[h]) + np.linalg.norm(pts[t] - q[t])
        swap = np.linalg.norm(pts[h] - q[t]) + np.linalg.norm(pts[t] - q[h])
        if not swap < 0.5 * same:
            return None
        now = float(np.nanmean(vals[[h, t]]))
        before = (
            float(np.nanmean(prev.scores[[h, t]])) if prev.scores is not None else 0.0
        )
        if now > before + self.seed_config.flip_margin:
            self.seed_stats["flips_kept"] += 1
            return None
        self.seed_stats["flips_undone"] += 1
        c = (pts[h] + pts[t]) / 2
        return 2 * c - pts

    def _update(
        self,
        s: _Stream,
        out: Outputs,
        detected: bool,
        height: int,
        width: int,
        vouched: Optional[int] = None,
        extra_prev: Optional[List[_Animal]] = None,
        frame: Optional[int] = None,
    ) -> None:
        """Carry one frame's instances into the stream and decide the next frame.

        On a detector frame, slots from ``vouched`` on hold carried animals the
        detector did not find (``extra_prev``, in order): they are judged like
        seeded animals, and dropped from the outputs too when lost or duplicated.
        """
        cfg = self.seed_config
        self.seed_stats["frames"] += 1
        if detected:
            self.seed_stats["detector_frames"] += 1
            self.seed_stats[f"detector_{s.reason}"] += 1
        if out.pred_keypoints is None:
            kpts = np.zeros((0, 0, 2))
            vals = np.zeros((0, 0))
        else:
            kpts = out.pred_keypoints[0].detach().float().cpu().numpy()
            vals = (
                out.pred_peak_values[0].detach().float().cpu().numpy()
                if out.pred_peak_values is not None
                else np.full(kpts.shape[:2], np.nan)
            )
        bscores = (
            out.pred_bbox_scores[0].detach().float().cpu().numpy()
            if out.pred_bbox_scores is not None
            else np.ones(kpts.shape[0])
        )
        old = s.animals
        n_v = kpts.shape[0] if vouched is None else vouched
        prevs = self._previous(old, kpts[:n_v], vals[:n_v], detected)
        for j, a in enumerate(extra_prev or []):
            prevs[n_v + j] = a
        animals: List[_Animal] = []
        slots: List[int] = []
        mean_scores: List[float] = []
        lost = 0
        for i in range(kpts.shape[0]):
            if np.isnan(kpts[i]).all():
                continue
            prev = prevs.get(i)
            seeded = (not detected) or i >= n_v
            if cfg.fix_flips and prev is not None and self.flip_index is not None:
                fixed = self._fix_flip(kpts[i], vals[i], prev)
                if fixed is not None:
                    kpts[i] = fixed
                    out.pred_keypoints[0, i] = torch.as_tensor(
                        fixed, dtype=out.pred_keypoints.dtype
                    ).to(out.pred_keypoints.device)
            box = box_from_points(kpts[i], vals[i], cfg)
            m = float(np.nanmean(vals[i])) if np.isfinite(vals[i]).any() else 0.0
            if box is None:
                lost += 1
                if i >= n_v:
                    self._drop_slot(out, i)
                continue
            ref = (
                m
                if (prev is None or not seeded or not np.isfinite(prev.ref_score))
                else prev.ref_score
            )
            if seeded and (
                (m < cfg.lost_score and m < cfg.lost_ratio * ref)
                or _outside_fraction(box, height, width) > cfg.max_outside
            ):
                lost += 1
                if i >= n_v:
                    self._drop_slot(out, i)
                continue
            center = (box[:2] + box[2:]) / 2
            if cfg.motion and prev is not None and prev.center is not None:
                box = box + np.tile(center - prev.center, 2)
            score = float(bscores[i]) if np.isfinite(bscores[i]) else 0.0
            if prev is not None and prev.track_id >= 0:
                tid = prev.track_id
            else:
                tid = self._next_track
                self._next_track += 1
            animals.append(
                _Animal(
                    box=box,
                    score=score,
                    ref_score=ref,
                    center=center,
                    points=kpts[i].copy(),
                    scores=vals[i].copy(),
                    track_id=tid,
                )
            )
            mean_scores.append(m)
            slots.append(i)
        # Two boxes on one animal: keep the detector's, then the higher-scoring one.
        dups = 0
        if len(animals) > 1:
            iou = _iou_matrix(
                np.stack([a.box for a in animals]), np.stack([a.box for a in animals])
            )
            keep = np.ones(len(animals), dtype=bool)
            rank = [
                (slot < n_v and detected, m_) for slot, m_ in zip(slots, mean_scores)
            ]
            order = sorted(range(len(animals)), key=lambda q: rank[q], reverse=True)
            for a_i in order:
                if not keep[a_i]:
                    continue
                for b_i in order:
                    if b_i != a_i and keep[b_i] and iou[a_i, b_i] > cfg.dup_iou:
                        keep[b_i] = False
                        dups += 1
            for q, k in enumerate(keep):
                if not k and slots[q] >= n_v and detected:
                    self._drop_slot(out, slots[q])
            kept_slots = [q for q, k in zip(slots, keep) if k]
            animals = [a for a, k in zip(animals, keep) if k]
            slots = kept_slots
        s.animals = animals
        if cfg.emit_tracks and out.pred_keypoints is not None:
            # One id per instance the frame will write (its non-empty slots, in order).
            now = out.pred_keypoints[0].detach().float().cpu().numpy()
            ids = {slot: a.track_id for slot, a in zip(slots, animals)}
            key = (s.video, s.last_frame if frame is None else int(frame))
            self.track_log[key] = [
                ids.get(i, -1)
                for i in range(now.shape[0])
                if not np.isnan(now[i]).all()
            ]
        if detected:
            s.since_detect = 0
            s.detect_count = len(animals)
        else:
            s.since_detect += 1
        reason = None
        if s.since_detect + 1 >= cfg.detect_every:
            reason = "schedule"
        elif not detected and lost:
            reason = "lost"
        elif not detected and dups:
            reason = "duplicate"
        elif len(animals) < s.detect_count:
            reason = "fewer"
        self.seed_stats["lost"] += lost
        self.seed_stats["duplicates_dropped"] += dups
        s.detect_next = reason is not None
        s.reason = reason or ""
