"""Per-frame detection arrays for the MOT trackers.

:func:`extract_detections` turns one frame's sleap-io detections (pose
``Instance`` / ``PredictedInstance`` or ``PredictedSegmentationMask``) into the
vectorized :class:`FrameDetections` every tracker, the calibration and the
identity linker operate on. The source objects are kept (``sources``) so tracks
can be written back onto them.

Geometry conventions (shared by every consumer -- never mix them):

- ``raw_boxes`` is the tight extent of the visible keypoints (or the mask's
  bounding box), ``xyxy`` in image pixels.
- ``boxes`` is ``raw_boxes`` padded by ``box_padding * max(w, h)`` on each side.
- ``centroids`` is the CENTER OF ``raw_boxes`` -- the same reference point the
  Kalman filters track (box center). Do not use the node median here: mixing the
  two reference points injects a fake per-frame displacement.
"""

from typing import List, Optional, Sequence, Tuple

import attrs
import numpy as np
import sleap_io as sio

from sleap_nn.tracking.mot.params import SCORE_MODES
from sleap_nn.tracking.utils import get_embedding, is_segmentation_mask

# Valid `carrier` values of `extract_detections`.
CARRIERS = ("pose", "mask")

# A raw box narrower than this (px) on either axis is degenerate: a single
# visible node, coincident nodes, or nodes collinear on an image axis.
_MIN_BOX_SIDE = 1.0


@attrs.define
class FrameDetections:
    """Vectorized detections of one frame.

    Attributes:
        frame_idx: Frame index of the detections.
        sources: The original sleap-io objects, in input order.
        boxes: ``(n, 4)`` float64 padded ``xyxy`` boxes; NaN rows where invalid.
        raw_boxes: ``(n, 4)`` float64 unpadded extents; NaN rows where invalid.
        centroids: ``(n, 2)`` float64 raw-box centers; NaN rows where invalid.
        keypoints: ``(n, K, 2)`` float64 keypoints with NaN for missing nodes, or
            ``None`` for masks.
        scores: ``(n,)`` float64 confidences in ``[0, 1]``.
        embeddings: ``(n, D)`` float32 L2-normalized appearance vectors with zero
            rows where missing, or ``None`` when no detection has one.
        has_embedding: ``(n,)`` bool.
        valid: ``(n,)`` bool: the detection has usable geometry.
        n_visible: ``(n,)`` int visible nodes (masks: 1 if non-empty else 0).
        n_nodes: Nodes per detection ``K`` (masks: 1).
    """

    frame_idx: int
    sources: list
    boxes: np.ndarray
    raw_boxes: np.ndarray
    centroids: np.ndarray
    keypoints: Optional[np.ndarray]
    scores: np.ndarray
    embeddings: Optional[np.ndarray]
    has_embedding: np.ndarray
    valid: np.ndarray
    n_visible: np.ndarray
    n_nodes: int

    def __len__(self) -> int:
        """Number of detections."""
        return len(self.sources)

    def subset(self, idx: np.ndarray) -> "FrameDetections":
        """Return the detections at ``idx`` (index array or bool mask), copied.

        Args:
            idx: Integer indices (any order) or a boolean mask of length
                ``len(self)``.

        Returns:
            A new :class:`FrameDetections` holding copies of the selected rows.
            ``embeddings`` is ``None`` when none of the selected detections has
            a vector (same invariant as :func:`extract_detections`).

        Raises:
            ValueError: If a boolean mask has the wrong length.
        """
        idx = np.asarray(idx)
        if idx.dtype == bool:
            if idx.shape != (len(self),):
                raise ValueError(
                    f"Boolean subset mask has shape {idx.shape}, expected "
                    f"({len(self)},)."
                )
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.intp, copy=False).reshape(-1)

        has_embedding = self.has_embedding[idx].copy()
        embeddings = None
        if self.embeddings is not None and has_embedding.any():
            embeddings = self.embeddings[idx].copy()

        return FrameDetections(
            frame_idx=self.frame_idx,
            sources=[self.sources[i] for i in idx],
            boxes=self.boxes[idx].copy(),
            raw_boxes=self.raw_boxes[idx].copy(),
            centroids=self.centroids[idx].copy(),
            keypoints=None if self.keypoints is None else self.keypoints[idx].copy(),
            scores=self.scores[idx].copy(),
            embeddings=embeddings,
            has_embedding=has_embedding,
            valid=self.valid[idx].copy(),
            n_visible=self.n_visible[idx].copy(),
            n_nodes=self.n_nodes,
        )

    @classmethod
    def empty(cls, frame_idx: int, n_nodes: int = 1) -> "FrameDetections":
        """Return an empty frame (zero detections) with correctly shaped arrays.

        Args:
            frame_idx: Frame index.
            n_nodes: Nodes per detection; sets the ``(0, n_nodes, 2)`` keypoint
                shape.

        Returns:
            A :class:`FrameDetections` with ``len() == 0`` and no embeddings.
        """
        n_nodes = int(n_nodes)
        return cls(
            frame_idx=int(frame_idx),
            sources=[],
            boxes=np.zeros((0, 4), dtype=np.float64),
            raw_boxes=np.zeros((0, 4), dtype=np.float64),
            centroids=np.zeros((0, 2), dtype=np.float64),
            keypoints=np.zeros((0, n_nodes, 2), dtype=np.float64),
            scores=np.zeros((0,), dtype=np.float64),
            embeddings=None,
            has_embedding=np.zeros((0,), dtype=bool),
            valid=np.zeros((0,), dtype=bool),
            n_visible=np.zeros((0,), dtype=np.int64),
            n_nodes=n_nodes,
        )


def _instance_score(obj) -> float:
    """The object's own ``score``; 1.0 if it has none or it is not finite."""
    score = getattr(obj, "score", None)
    if score is None:
        return 1.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 1.0
    return score if np.isfinite(score) else 1.0


def _pose_arrays(objs: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack pose keypoints and point scores, one ``numpy()`` call per object.

    Args:
        objs: Pose instances (user or predicted).

    Returns:
        ``(xy, point_scores, is_predicted)``: ``(n, K, 2)`` keypoints with NaN for
        missing nodes (NaN-padded to the longest skeleton ``K``), ``(n, K)`` point
        scores (NaN for user instances) and ``(n,)`` bool.
    """
    arrays: List[np.ndarray] = []
    is_predicted = np.zeros(len(objs), dtype=bool)
    for i, obj in enumerate(objs):
        if isinstance(obj, sio.PredictedInstance):
            # (K, 3): x, y, point score (invisible nodes are NaN in x/y).
            arrays.append(np.asarray(obj.numpy(scores=True), dtype=np.float64))
            is_predicted[i] = True
        else:
            arrays.append(np.asarray(obj.numpy(), dtype=np.float64))
    k_max = max((a.shape[0] for a in arrays), default=0)
    xy = np.full((len(objs), k_max, 2), np.nan, dtype=np.float64)
    point_scores = np.full((len(objs), k_max), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        k = a.shape[0]
        xy[i, :k] = a[:, :2]
        if a.shape[1] > 2:
            point_scores[i, :k] = a[:, 2]
    return xy, point_scores, is_predicted


def _pose_geometry(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw ``xyxy`` extents of the visible (finite) keypoints, vectorized.

    Returns:
        ``(raw_boxes, n_visible, visible)``; rows with no visible node are NaN.
    """
    visible = np.isfinite(xy).all(axis=-1)  # (n, K)
    n_visible = visible.sum(axis=1).astype(np.int64)
    # Masked min/max (no all-NaN-slice warnings): invisible nodes never win.
    lo = np.where(visible[..., None], xy, np.inf).min(axis=1, initial=np.inf)
    hi = np.where(visible[..., None], xy, -np.inf).max(axis=1, initial=-np.inf)
    raw = np.concatenate([lo, hi], axis=1)
    raw[n_visible == 0] = np.nan
    return raw, n_visible, visible


def _pose_scores(
    objs: Sequence,
    point_scores: np.ndarray,
    visible: np.ndarray,
    is_predicted: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    """Per-detection pose confidence (before the final finite/clip pass)."""
    n = len(objs)
    if score_mode == "none":
        return np.ones(n, dtype=np.float64)
    if score_mode == "instance":
        return np.array([_instance_score(o) for o in objs], dtype=np.float64)
    # "auto": mean point score over the VISIBLE nodes of predicted poses (an
    # invisible node's score says nothing about the detection); the instance
    # score when no visible node has a finite point score, or when every one is
    # exactly 0 (`PredictedInstance.from_numpy` without `point_scores` stores
    # zeros: no information, not zero confidence); 1.0 for user poses.
    has_ps = visible & np.isfinite(point_scores)
    n_ps = has_ps.sum(axis=1)
    ps = np.where(has_ps, point_scores, 0.0)
    mean_ps = ps.sum(axis=1) / np.maximum(n_ps, 1)
    informative = (n_ps > 0) & (np.abs(ps).sum(axis=1) > 0)
    scores = np.ones(n, dtype=np.float64)
    for i in np.flatnonzero(is_predicted):
        scores[i] = mean_ps[i] if informative[i] else _instance_score(objs[i])
    return scores


def _mask_geometry(objs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Raw ``xyxy`` boxes from ``mask.bbox`` (image space) and non-empty flags."""
    raw = np.full((len(objs), 4), np.nan, dtype=np.float64)
    nonempty = np.zeros(len(objs), dtype=bool)
    for i, obj in enumerate(objs):
        # `area` is a cheap sum over the RLE runs; only decode `bbox` when needed.
        if int(obj.area) <= 0:
            continue
        x, y, w, h = (float(v) for v in obj.bbox)
        raw[i] = (x, y, x + w, y + h)
        nonempty[i] = True
    return raw, nonempty


def _usable_vector(obj) -> Optional[Tuple[np.ndarray, float]]:
    """A detection's appearance vector (float64, flat) and its L2 norm, if usable.

    THE "usable vector" rule, shared by the trackers (:func:`extract_detections`)
    and the up-front checks of ``apply_tracking``
    (:func:`has_usable_embedding`): a vector that is absent, empty, non-finite
    or zero-norm is unusable (``None``).
    """
    vec = get_embedding(obj)
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        return None
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return vec, norm


def has_usable_embedding(obj) -> bool:
    """Whether ``obj`` carries a usable appearance vector (see :func:`_usable_vector`).

    A frame-level check can additionally drop a usable vector whose dimension
    differs from the frame's (:func:`extract_detections`); per detection, only
    presence, finiteness and a non-zero norm can be judged.
    """
    return _usable_vector(obj) is not None


def _embedding_arrays(
    objs: Sequence, embedding_dim: Optional[int]
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Collect re-L2-normalized appearance vectors.

    A vector that is absent, empty, non-finite, zero-norm (:func:`_usable_vector`),
    or of a dimension other than ``embedding_dim`` (``None``: the first usable
    vector's) counts as missing.

    Returns:
        ``(embeddings, has_embedding)``; ``embeddings`` is ``None`` when no
        object carries a usable vector.
    """
    has = np.zeros(len(objs), dtype=bool)
    rows: List[Optional[np.ndarray]] = [None] * len(objs)
    dim = embedding_dim
    for i, obj in enumerate(objs):
        usable = _usable_vector(obj)
        if usable is None:
            continue
        vec, norm = usable
        if dim is not None and vec.size != dim:
            continue
        rows[i] = vec / norm
        has[i] = True
        if dim is None:
            dim = vec.size
    if not has.any():
        return None, has
    emb = np.zeros((len(objs), dim), dtype=np.float32)
    for i in np.flatnonzero(has):
        emb[i] = rows[i]
    return emb, has


def extract_detections(
    objs: Sequence,
    frame_idx: int,
    *,
    carrier: str = "pose",
    box_padding: float = 0.1,
    pseudo_box_size: Optional[float] = None,
    score_mode: str = "auto",
    embedding_dim: Optional[int] = None,
    min_box_size: Optional[float] = None,
) -> FrameDetections:
    """Build :class:`FrameDetections` from one frame's sleap-io detections.

    Geometry: a pose's raw box is the extent of its visible (finite) nodes; a
    mask's is ``mask.bbox`` (image space), and an empty mask is invalid. Each
    object is dispatched with
    :func:`~sleap_nn.tracking.utils.is_segmentation_mask`. A valid detection is
    DEGENERATE when it is a pose with a single visible node or its raw box is
    narrower than 1 px on either axis: its raw and padded boxes then both become
    the square of side ``pseudo_box_size`` centered on its centroid (with
    ``pseudo_box_size=None``, the raw box is instead floored at 1 px per axis
    around its center and padded as usual). Centroids are raw-box centers.

    Scores (``score_mode``), then non-finite -> 1.0 and clipped to ``[0, 1]``:

    - ``"auto"``: predicted pose -> mean point score over its visible nodes
      (``obj.score`` if none is finite); predicted mask -> ``obj.score``; user
      ``Instance`` / user mask (no ``score``) -> 1.0.
    - ``"instance"``: ``obj.score`` (1.0 if absent).
    - ``"none"``: 1.0 everywhere.

    Args:
        objs: The frame's detections: pose instances (``carrier="pose"``) or
            segmentation masks (``carrier="mask"``).
        frame_idx: Frame index of the detections.
        carrier: ``"pose"`` or ``"mask"``.
        box_padding: Fraction of ``max(w, h)`` added to each side of the box.
        pseudo_box_size: Side (px) of the square box used for degenerate
            geometry (a single visible node, or a raw box narrower than 1 px).
            ``None`` floors the box at 1 px instead.
        score_mode: ``"auto"``, ``"instance"`` or ``"none"`` (see
            :class:`sleap_nn.tracking.mot.params.MOTParams`).
        embedding_dim: Expected appearance-vector dimension; vectors of another
            dimension are treated as missing. ``None`` accepts the first seen.
        min_box_size: Minimum side (px) of the PADDED box on each axis; narrower
            boxes are widened symmetrically around their center. Keeps IoU
            meaningful for thin poses (nodes nearly collinear along one axis).
            Raw boxes (the OKS scale) are not changed. ``None`` disables it.

    Returns:
        The frame's :class:`FrameDetections`. ``keypoints`` is ``None`` for the
        mask carrier; on the pose carrier a (stray) mask gets an all-NaN row.

    Raises:
        ValueError: On an unknown ``carrier`` or ``score_mode``.
    """
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}.")
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}."
        )
    objs = list(objs)
    n = len(objs)
    is_mask = np.array([is_segmentation_mask(o) for o in objs], dtype=bool)
    pose_idx = np.flatnonzero(~is_mask)
    mask_idx = np.flatnonzero(is_mask)

    raw = np.full((n, 4), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    n_visible = np.zeros(n, dtype=np.int64)
    scores = np.ones(n, dtype=np.float64)
    single_node = np.zeros(n, dtype=bool)
    xy = None

    # Pose detections: one numpy() call per object, everything else vectorized.
    if len(pose_idx) > 0:
        pose_objs = [objs[i] for i in pose_idx]
        xy, point_scores, is_predicted = _pose_arrays(pose_objs)
        pose_raw, pose_n_visible, visible = _pose_geometry(xy)
        raw[pose_idx] = pose_raw
        valid[pose_idx] = pose_n_visible > 0
        n_visible[pose_idx] = pose_n_visible
        single_node[pose_idx] = pose_n_visible == 1
        scores[pose_idx] = _pose_scores(
            pose_objs, point_scores, visible, is_predicted, score_mode
        )

    # Mask detections: geometry from the image-space bbox; score is the mask's
    # own (a user mask has no `score` -> 1.0) in both "auto" and "instance".
    if len(mask_idx) > 0:
        mask_objs = [objs[i] for i in mask_idx]
        mask_raw, nonempty = _mask_geometry(mask_objs)
        raw[mask_idx] = mask_raw
        valid[mask_idx] = nonempty
        n_visible[mask_idx] = nonempty.astype(np.int64)
        if score_mode != "none":
            scores[mask_idx] = [_instance_score(o) for o in mask_objs]

    scores = np.clip(np.where(np.isfinite(scores), scores, 1.0), 0.0, 1.0)
    valid &= np.isfinite(raw).all(axis=1)
    raw[~valid] = np.nan

    # Keypoints (pose carrier only), NaN-padded to the longest skeleton.
    keypoints = None
    n_nodes = 1
    if carrier == "pose":
        n_nodes = max(1, 0 if xy is None else xy.shape[1])
        keypoints = np.full((n, n_nodes, 2), np.nan, dtype=np.float64)
        if xy is not None:
            keypoints[pose_idx, : xy.shape[1]] = xy

    if n == 0:
        out = FrameDetections.empty(frame_idx, n_nodes=n_nodes)
        out.keypoints = keypoints
        return out

    # Padding, then degenerate geometry (overrides the padded box).
    wh = raw[:, 2:] - raw[:, :2]
    centroids = 0.5 * (raw[:, :2] + raw[:, 2:])
    pad = float(box_padding) * wh.max(axis=1)
    boxes = raw + pad[:, None] * np.array([-1.0, -1.0, 1.0, 1.0])
    degenerate = valid & (single_node | (wh < _MIN_BOX_SIDE).any(axis=1))
    if degenerate.any():
        c = centroids[degenerate]
        if pseudo_box_size is not None and np.isfinite(pseudo_box_size):
            # Square of the calibrated body size: raw (OKS scale) and padded.
            half = np.full_like(c, 0.5 * max(float(pseudo_box_size), _MIN_BOX_SIDE))
            raw[degenerate] = np.concatenate([c - half, c + half], axis=1)
            boxes[degenerate] = raw[degenerate]
        else:
            # Standalone callers: 1-px floor per axis, then the usual padding.
            half = 0.5 * np.maximum(wh[degenerate], _MIN_BOX_SIDE)
            raw[degenerate] = np.concatenate([c - half, c + half], axis=1)
            dpad = float(box_padding) * 2.0 * half.max(axis=1)
            boxes[degenerate] = raw[degenerate] + dpad[:, None] * np.array(
                [-1.0, -1.0, 1.0, 1.0]
            )

    boxes = _widened(boxes, valid, min_box_size)

    embeddings, has_embedding = _embedding_arrays(objs, embedding_dim)

    return FrameDetections(
        frame_idx=int(frame_idx),
        sources=objs,
        boxes=boxes,
        raw_boxes=raw,
        centroids=centroids,
        keypoints=keypoints,
        scores=scores,
        embeddings=embeddings,
        has_embedding=has_embedding,
        valid=valid,
        n_visible=n_visible,
        n_nodes=int(n_nodes),
    )


def _widened(boxes: np.ndarray, valid: np.ndarray, min_box_size) -> np.ndarray:
    """``boxes`` widened symmetrically to at least ``min_box_size`` per axis."""
    if min_box_size is None or not np.isfinite(min_box_size) or min_box_size <= 0:
        return boxes
    side = boxes[:, 2:] - boxes[:, :2]
    grow = 0.5 * np.maximum(float(min_box_size) - side, 0.0)
    grow[~valid] = 0.0
    return boxes + np.concatenate([-grow, grow], axis=1)


def normalize_scores(dets: FrameDetections, normalizer: Optional[float]) -> None:
    """Divide ``dets.scores`` by ``normalizer`` and clip to ``[0, 1]``, in place.

    The per-video score normalization of ``MOTParams.normalize_scores`` (the
    normalizer is :func:`~sleap_nn.tracking.mot.calibration.score_normalizer`:
    the video's score q95). Apply it exactly once per extraction.

    Args:
        dets: Detections to modify.
        normalizer: Positive divisor; ``None``, non-finite or non-positive is a
            no-op.
    """
    if normalizer is None or not np.isfinite(normalizer) or normalizer <= 0:
        return
    if len(dets):
        dets.scores = np.clip(
            np.asarray(dets.scores, dtype=np.float64) / float(normalizer), 0.0, 1.0
        )


def widen_boxes(dets: FrameDetections, min_box_size: Optional[float]) -> None:
    """Floor ``dets.boxes`` (padded) at ``min_box_size`` per axis, in place.

    Same rule as ``extract_detections(min_box_size=...)``, for detections that
    were already extracted. Raw boxes are not changed.

    Args:
        dets: Detections to modify.
        min_box_size: Minimum padded side (px); ``None`` or non-positive is a no-op.
    """
    if len(dets):
        dets.boxes = _widened(dets.boxes, dets.valid, min_box_size)
