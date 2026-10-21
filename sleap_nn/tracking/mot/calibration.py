"""Per-video self-calibration for the MOT trackers.

:func:`calibrate` looks at every detection of a video BEFORE tracking it (all
vectors already exist in sleap-nn's offline tracking flow) and measures:

- **Geometry:** body size, nearest-neighbor spacing, the processing stride and
  per-step motion, from *anchor tracklets* -- detections chained frame-to-frame
  only where the association is unambiguous (mutual nearest neighbors, well
  separated from any competitor), cut at any doubt.
- **Scores:** detection-confidence quantiles.
- **Appearance** (when detections carry re-ID vectors): the distributions of the
  exact DECISION STATISTICS the trackers and the identity linker use, measured on
  synthetic trials over anchor tracklets. Two anchors that overlap in time are
  guaranteed to be different animals, so co-occurring anchors provide true
  negatives; later vectors of the same anchor provide positives. Calibrating the
  statistic that is thresholded (EMA cosine, gallery max, tracklet-to-gallery
  top-k mean) -- not single-pair cosines -- keeps the thresholds' error rates
  meaningful.

When there is not enough evidence (no vectors, a single animal, too few trials)
the appearance part reports ``status != "ok"`` and every ``"auto"`` threshold
disables appearance rather than falling back to a constant tuned on person re-ID.
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import attrs
import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.detections import FrameDetections

# Query length (vectors) of the linking statistic. The identity linker scores a
# gap link / swap with exactly this many query vectors, so its calibrated
# thresholds describe the same statistic.
LINK_QUERY_LEN = 10
# Margins tried when calibrating the linking rule, smallest first.
LINK_MARGIN_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
# Minimum calibration trials (g = 1) for the appearance part to be "ok".
MIN_TRIALS = 100
# Minimum anchor-tracklet length.
MIN_ANCHOR_LEN = 10
# Required precision of the calibrated linking rule.
LINK_PRECISION = 0.99
# Minimum linking recall for appearance to be used for linking.
MIN_LINK_RECALL = 0.2
# Minimum g=1 rank-1 for appearance to be used in association.
MIN_ASSOC_RANK1 = 0.95
# Warn when this fraction of same-frame negatives look like positives.
NEG_COLLISION_WARN = 0.05
# Score quantile levels reported in `VideoCalibration.score_quantiles`.
SCORE_QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Split points of the appearance trials: every TRIAL_STRIDE-th vector of an anchor.
TRIAL_STRIDE = 10
# `k` of the linking statistic (mean of the top-k gallery cosines per query).
LINK_TOPK = 5
# Floor of the association margin `m_assoc`.
MIN_ASSOC_MARGIN = 0.02

# Anchor chaining: a link must be shorter than this fraction of the video's
# nearest-neighbor spacing, and the runner-up (in both directions) at least
# _ANCHOR_RATIO times farther.
_ANCHOR_MAX_NN_FRACTION = 0.5
_ANCHOR_RATIO = 2.0
# Two same-frame detections whose intersection covers more than this fraction of
# the smaller box are duplicates (split masks, double detections).
_DUPLICATE_IOMIN = 0.5
# Body size / spacing when nothing can be measured (px).
_DEFAULT_SIZE = 50.0
# Element budget of the batched (frames x N x N) temporaries.
_CHUNK_ELEMS = 1 << 20


@attrs.frozen
class VideoCalibration:
    """Calibration of one video. Every field is a plain Python type.

    Attributes:
        status: Appearance-calibration status: ``"ok"``, ``"no_vectors"`` or
            ``"insufficient"``.
        n_frames: Frames seen.
        n_detections: Valid detections seen.
        frame_step: Most common positive frame-index delta (1 if unknown).
        regular_stride: Whether at least half of the deltas equal ``frame_step``.
        body_size: Median raw-box diagonal of valid multi-node detections (px);
            falls back to ``nn_distance``, then 50.
        nn_distance: Median same-frame nearest-neighbor centroid distance (px),
            ignoring duplicate pairs (IoMin > 0.5); falls back to ``body_size``.
        step_p95: 95th percentile per-step centroid displacement along anchors (px).
        step_p99: 99th percentile per-step centroid displacement along anchors (px).
        score_quantiles: Score quantiles at 0.05, 0.25, 0.5, 0.75 and 0.95.
        n_anchors: Anchor tracklets found.
        n_trials: Appearance trials run at ``g = 1``.
        assoc_rank1: Fraction of ``g = 1`` trials where the true track's EMA
            cosine beats every co-occurring competitor's.
        m_assoc: Association margin: ``max(0.02, q10)`` of (true - best
            competitor) EMA cosine at ``g = 1``.
        ema_cos_gate: q95 of competitor EMA cosines at ``g = 1``.
        gallery_cos_gate: q95 of competitor gallery-max cosines at ``g = 1``.
        mu_pos: Median true EMA cosine at ``g = 1``.
        mu_neg: Median competitor EMA cosine at ``g = 1``.
        link_thresh: q99 of competitor linking statistics at the long gap.
        link_margin: Smallest margin in :data:`LINK_MARGIN_GRID` giving linking
            precision ``>= LINK_PRECISION`` on the trials, or ``None``.
        link_recall: Recall of the calibrated linking rule on the trials.
        neg_collision_rate: Fraction of same-frame negative pairs whose cosine is
            at or above the q05 of ``g = 1`` positives.
        appearance_for_association: ``status == "ok"`` and ``assoc_rank1 >=
            MIN_ASSOC_RANK1``.
        appearance_for_linking: ``status == "ok"``, ``link_margin is not None``
            and ``link_recall >= MIN_LINK_RECALL``.
    """

    status: str = "no_vectors"
    n_frames: int = 0
    n_detections: int = 0
    frame_step: int = 1
    regular_stride: bool = True
    body_size: float = 50.0
    nn_distance: float = 50.0
    step_p95: float = float("nan")
    step_p99: float = float("nan")
    score_quantiles: tuple = ()
    n_anchors: int = 0
    n_trials: int = 0
    assoc_rank1: Optional[float] = None
    m_assoc: Optional[float] = None
    ema_cos_gate: Optional[float] = None
    gallery_cos_gate: Optional[float] = None
    mu_pos: Optional[float] = None
    mu_neg: Optional[float] = None
    link_thresh: Optional[float] = None
    link_margin: Optional[float] = None
    link_recall: Optional[float] = None
    neg_collision_rate: Optional[float] = None
    appearance_for_association: bool = False
    appearance_for_linking: bool = False

    def as_dict(self) -> dict:
        """JSON-serializable dict (NaN becomes ``None``)."""
        return {k: _json_safe(v) for k, v in attrs.asdict(self, recurse=False).items()}

    def summary(self) -> str:
        """One-line human-readable summary for logs."""
        stride = "regular" if self.regular_stride else "irregular"
        geometry = (
            f"{self.n_frames} frames, {self.n_detections} detections, frame step "
            f"{self.frame_step} ({stride}), body {_fmt(self.body_size, 1)} px, "
            f"NN {_fmt(self.nn_distance, 1)} px, step p95/p99 "
            f"{_fmt(self.step_p95, 1)}/{_fmt(self.step_p99, 1)} px, "
            f"{self.n_anchors} anchors"
        )
        if self.status != "ok":
            appearance = (
                f"appearance {self.status} ({self.n_trials} trials): association "
                "and linking by geometry only"
            )
        else:
            appearance = (
                f"appearance ok ({self.n_trials} trials): rank-1 "
                f"{_fmt(self.assoc_rank1)}, m_assoc {_fmt(self.m_assoc)}, EMA gate "
                f"{_fmt(self.ema_cos_gate)}, gallery gate "
                f"{_fmt(self.gallery_cos_gate)}, link thresh "
                f"{_fmt(self.link_thresh)} margin {_fmt(self.link_margin, 2)} "
                f"recall {_fmt(self.link_recall)}, negative collisions "
                f"{_fmt(self.neg_collision_rate)}; association "
                f"{'on' if self.appearance_for_association else 'off'}, linking "
                f"{'on' if self.appearance_for_linking else 'off'}"
            )
        return f"MOT calibration: {geometry}; {appearance}"


def _json_safe(value: Any) -> Any:
    """Cast to JSON-native Python types; non-finite floats become ``None``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(v) for v in value]
    return value


def _fmt(value: Optional[float], digits: int = 3) -> str:
    """Format an optional float for :meth:`VideoCalibration.summary`."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _opt_float(value: Any) -> Optional[float]:
    """``float(value)`` or ``None`` when missing or not finite."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------
# Flattened detections
# --------------------------------------------------------------------------


@attrs.define
class _Flat:
    """The video's VALID detections, flattened in (frame, input order) order.

    Attributes:
        frame_pos: ``(M,)`` position of each detection's frame in ``frames``.
        offsets: ``(F,)`` first flat index of each frame.
        counts: ``(F,)`` valid detections per frame.
        centroids: ``(M, 2)`` raw-box centers.
        raw_boxes: ``(M, 4)`` raw ``xyxy`` boxes.
        scores: ``(M,)`` detection scores.
        sized: ``(M,)`` usable for the body size (masks, or poses with at least
            two visible nodes).
        vectors: ``(M, D)`` float32 unit vectors (zero rows where missing) or
            ``None`` when no detection has one.
        has_vec: ``(M,)`` bool.
    """

    frame_pos: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    raw_boxes: np.ndarray
    scores: np.ndarray
    sized: np.ndarray
    vectors: Optional[np.ndarray]
    has_vec: np.ndarray


def _vector_dim(frames: Sequence[FrameDetections], valid: List[np.ndarray]) -> int:
    """Most common appearance-vector dimension over the valid detections (0: none)."""
    weights: Dict[int, int] = {}
    for fd, m in zip(frames, valid):
        if fd.embeddings is None or np.ndim(fd.embeddings) != 2:
            continue
        n = int(np.count_nonzero(np.asarray(fd.has_embedding, dtype=bool) & m))
        if n:
            d = int(np.asarray(fd.embeddings).shape[1])
            weights[d] = weights.get(d, 0) + n
    if not weights:
        return 0
    # Ties -> the smaller dimension (deterministic).
    return max(sorted(weights), key=lambda d: weights[d])


def _flatten(frames: Sequence[FrameDetections]) -> _Flat:
    """Flatten the valid detections of ``frames`` (see :class:`_Flat`)."""
    valid = [np.asarray(fd.valid, dtype=bool).reshape(-1) for fd in frames]
    counts = np.array([int(m.sum()) for m in valid], dtype=np.int64)
    offsets = (np.cumsum(counts) - counts).astype(np.int64)
    n_total = int(counts.sum())
    frame_pos = np.repeat(np.arange(len(frames), dtype=np.int64), counts)

    def cat(arrays: List[np.ndarray], width: int) -> np.ndarray:
        if not arrays:
            return np.zeros((0, width), dtype=np.float64)
        return np.concatenate(arrays).astype(np.float64).reshape(-1, width)

    centroids = cat(
        [np.asarray(fd.centroids).reshape(-1, 2)[m] for fd, m in zip(frames, valid)],
        2,
    )
    raw_boxes = cat(
        [np.asarray(fd.raw_boxes).reshape(-1, 4)[m] for fd, m in zip(frames, valid)],
        4,
    )
    scores = cat(
        [np.asarray(fd.scores).reshape(-1)[m] for fd, m in zip(frames, valid)], 1
    ).reshape(-1)
    sized = np.concatenate(
        [
            (
                np.ones(int(m.sum()), dtype=bool)
                if fd.keypoints is None
                else (np.asarray(fd.n_visible).reshape(-1)[m] >= 2)
            )
            for fd, m in zip(frames, valid)
        ]
        or [np.zeros(0, dtype=bool)]
    )

    dim = _vector_dim(frames, valid)
    vectors = None
    has_vec = np.zeros(n_total, dtype=bool)
    if dim > 0:
        vectors = np.zeros((n_total, dim), dtype=np.float32)
        for f, (fd, m) in enumerate(zip(frames, valid)):
            if fd.embeddings is None or counts[f] == 0:
                continue
            emb = np.asarray(fd.embeddings)
            if emb.ndim != 2 or emb.shape[1] != dim:
                continue
            sl = slice(offsets[f], offsets[f] + counts[f])
            vectors[sl] = emb[m]
            has_vec[sl] = np.asarray(fd.has_embedding, dtype=bool).reshape(-1)[m]
        # Defensive renormalization; non-finite or zero rows count as missing.
        finite = np.isfinite(vectors).all(axis=1)
        vectors[~finite] = 0.0
        norms = np.linalg.norm(vectors, axis=1)
        has_vec &= finite & (norms > 1e-12)
        vectors[has_vec] /= norms[has_vec, None]
        vectors[~has_vec] = 0.0
        if not has_vec.any():
            vectors = None

    return _Flat(
        frame_pos=frame_pos,
        offsets=offsets,
        counts=counts,
        centroids=centroids,
        raw_boxes=raw_boxes,
        scores=scores,
        sized=sized,
        vectors=vectors,
        has_vec=has_vec,
    )


def _batches(
    frames: np.ndarray,
    flat: _Flat,
    width: int,
    offset: int = 0,
    pair_width: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Dense ``(c, width)`` flat-index blocks of frames holding ``width`` detections.

    Each row lists the detections of frame ``frames[i] + offset``. Blocks keep
    ``c * width * pair_width`` (the size of a pairwise temporary; ``pair_width``
    defaults to ``width``) within the element budget, so two calls with the same
    ``width * pair_width`` yield aligned blocks.
    """
    pair_width = width if pair_width is None else pair_width
    step = max(1, _CHUNK_ELEMS // max(1, width * pair_width))
    cols = np.arange(width, dtype=np.int64)[None, :]
    for start in range(0, len(frames), step):
        yield flat.offsets[frames[start : start + step] + offset][:, None] + cols


def _second_smallest(dist: np.ndarray, axis: int) -> np.ndarray:
    """Second-smallest value along ``axis`` (``inf`` when there is none)."""
    if dist.shape[axis] < 2:
        shape = list(dist.shape)
        del shape[axis]
        return np.full(shape, np.inf)
    return np.take(np.partition(dist, 1, axis=axis), 1, axis=axis)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _frame_steps(frame_idx: np.ndarray, time_base: str) -> Tuple[int, bool, np.ndarray]:
    """Processing stride and the steps between consecutive processed frames.

    Returns:
        ``(frame_step, regular_stride, steps)`` with ``steps`` the ``(F - 1,)``
        number of time steps between consecutive frames (>= 1), resolved like
        :class:`~sleap_nn.tracking.mot.params.MOTParams` ``time_base``.
    """
    deltas = np.diff(frame_idx)
    positive = deltas[deltas > 0]
    frame_step, regular = 1, True
    if positive.size:
        values, counts = np.unique(positive, return_counts=True)
        frame_step = int(values[np.argmax(counts)])  # ties -> smallest delta
        regular = bool(np.count_nonzero(positive == frame_step) >= 0.5 * positive.size)
    mode = time_base
    if mode == "auto":
        mode = "frames" if regular else "steps"
    if mode == "frames":
        steps = np.maximum(1, np.rint(deltas / frame_step)).astype(np.int64)
    else:
        steps = np.ones(len(deltas), dtype=np.int64)
    return frame_step, regular, steps


def _same_frame_pairs(
    flat: _Flat,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Same-frame structure of the detections.

    Frames are processed in groups of equal detection count, so a single crowded
    frame does not inflate the work on every other frame.

    Returns:
        ``(nn_dist, duplicate, neg_cos)``: per-detection nearest same-frame
        non-duplicate neighbor distance (``inf`` when alone), per-detection
        duplicate flag (another same-frame detection with IoMin above the
        duplicate threshold), and the cosines of every non-duplicate same-frame
        pair that both carry a vector (``None`` without vectors).
    """
    nn_dist = np.full(len(flat.frame_pos), np.inf)
    duplicate = np.zeros(len(flat.frame_pos), dtype=bool)
    neg_cos: List[np.ndarray] = [np.zeros(0, dtype=np.float32)]
    for k in np.unique(flat.counts[flat.counts >= 2]).tolist():
        eye = np.eye(k, dtype=bool)
        upper = np.triu(np.ones((k, k), dtype=bool), k=1)
        for idx in _batches(np.flatnonzero(flat.counts == k), flat, k):
            # Duplicates: IoMin = intersection / smaller area (raw boxes).
            box = flat.raw_boxes[idx]  # (c, k, 4)
            lo = np.maximum(box[:, :, None, :2], box[:, None, :, :2])
            hi = np.minimum(box[:, :, None, 2:], box[:, None, :, 2:])
            inter = np.clip(hi - lo, 0.0, None).prod(axis=-1)
            area = np.clip(box[..., 2:] - box[..., :2], 0.0, None).prod(axis=-1)
            min_area = np.minimum(area[:, :, None], area[:, None, :])
            with np.errstate(invalid="ignore", divide="ignore"):
                iomin = np.where(min_area > 0, inter / min_area, 0.0)
            dup_pair = (iomin > _DUPLICATE_IOMIN) & ~eye
            duplicate[idx] = dup_pair.any(axis=2)

            # Spacing between DIFFERENT animals: duplicate pairs do not count (a
            # double detection would otherwise shrink the spacing to ~0).
            cen = flat.centroids[idx]  # (c, k, 2)
            dist = np.linalg.norm(cen[:, :, None, :] - cen[:, None, :, :], axis=-1)
            dist = np.where(np.isfinite(dist) & ~eye & ~dup_pair, dist, np.inf)
            nn_dist[idx] = dist.min(axis=2)

            if flat.vectors is not None:
                vec = flat.vectors[idx]  # (c, k, D)
                has = flat.has_vec[idx]
                cos = np.einsum("cnd,cmd->cnm", vec, vec)
                ok = has[:, :, None] & has[:, None, :] & upper & ~dup_pair
                neg_cos.append(cos[ok])
    if flat.vectors is None:
        return nn_dist, duplicate, None
    return nn_dist, duplicate, np.concatenate(neg_cos)


def _anchor_links(flat: _Flat, duplicate: np.ndarray, max_dist: float) -> np.ndarray:
    """Unambiguous frame-to-frame links between consecutive processed frames.

    A detection ``r`` of frame ``f`` links to ``c`` of frame ``f + 1`` when they
    are mutual nearest neighbors, closer than ``max_dist``, the runner-up of each
    (in both directions) is at least ``_ANCHOR_RATIO`` times farther (and
    strictly farther), and neither is a same-frame duplicate. Frame pairs are
    processed in groups of equal ``(count_f, count_f+1)``.

    Returns:
        ``(M,)`` flat index of each detection's successor, -1 where none.
    """
    nxt = np.full(len(flat.frame_pos), -1, dtype=np.int64)
    if len(flat.counts) < 2:
        return nxt
    k_a, k_b = flat.counts[:-1], flat.counts[1:]
    pair_key = k_a * (int(flat.counts.max()) + 1) + k_b
    for key in np.unique(pair_key[(k_a > 0) & (k_b > 0)]).tolist():
        f = np.flatnonzero(pair_key == key)
        ka, kb = int(k_a[f[0]]), int(k_b[f[0]])
        rows = np.arange(ka)
        for src_blk, dst_blk in zip(
            _batches(f, flat, ka, pair_width=kb),
            _batches(f, flat, kb, offset=1, pair_width=ka),
        ):
            ca = flat.centroids[src_blk]  # (c, ka, 2) frame f
            cb = flat.centroids[dst_blk]  # (c, kb, 2) frame f + 1
            dist = np.linalg.norm(ca[:, :, None, :] - cb[:, None, :, :], axis=-1)
            dist = np.where(np.isfinite(dist), dist, np.inf)

            best_col = dist.argmin(axis=2)  # (c, ka): nearest in f + 1 per row
            best_row = dist.argmin(axis=1)  # (c, kb): nearest in f per column
            d1 = np.take_along_axis(dist, best_col[:, :, None], axis=2)[..., 0]
            row_second = _second_smallest(dist, axis=2)
            col_second = np.take_along_axis(
                _second_smallest(dist, axis=1), best_col, axis=1
            )
            mutual = np.take_along_axis(best_row, best_col, axis=1) == rows[None, :]
            dst = np.take_along_axis(dst_blk, best_col, axis=1)
            ok = np.isfinite(d1) & mutual & (d1 < max_dist)
            ok &= (row_second >= _ANCHOR_RATIO * d1) & (row_second > d1)
            ok &= (col_second >= _ANCHOR_RATIO * d1) & (col_second > d1)
            ok &= ~duplicate[src_blk] & ~duplicate[dst]
            nxt[src_blk[ok]] = dst[ok]
    return nxt


def _chain_roots(nxt: np.ndarray) -> np.ndarray:
    """Root (first detection) of each detection's chain, by pointer doubling."""
    n_total = len(nxt)
    ptr = np.arange(n_total, dtype=np.int64)
    linked = nxt >= 0
    ptr[nxt[linked]] = np.flatnonzero(linked)  # predecessor (or self)
    while True:
        nxt_ptr = ptr[ptr]
        if np.array_equal(nxt_ptr, ptr):
            return ptr
        ptr = nxt_ptr


# --------------------------------------------------------------------------
# Appearance trials
# --------------------------------------------------------------------------


@attrs.define
class _AnchorVectors:
    """Vector-bearing anchor detections, grouped by anchor and sorted by time.

    Attributes:
        vectors: ``(V, D)`` float32 unit vectors.
        ema: ``(V, D)`` running EMA of each anchor's vectors up to each position.
        steps: ``(V,)`` time step of each vector.
        key: ``(V,)`` sort key ``anchor * key_scale + step`` (strictly increasing).
        key_scale: Multiplier separating anchors in ``key``.
        offsets: ``(A,)`` first position of each anchor.
        lengths: ``(A,)`` vectors per anchor.
    """

    vectors: np.ndarray
    ema: np.ndarray
    steps: np.ndarray
    key: np.ndarray
    key_scale: int
    offsets: np.ndarray
    lengths: np.ndarray


def _anchor_vectors(
    flat: _Flat,
    anchor_of: np.ndarray,
    n_anchors: int,
    step_idx: np.ndarray,
    ema_alpha: float,
) -> Optional[_AnchorVectors]:
    """Collect anchor vectors and their running EMA (see :class:`_AnchorVectors`)."""
    sel = np.flatnonzero((anchor_of >= 0) & flat.has_vec)
    if sel.size == 0:
        return None
    anchor = anchor_of[sel]
    steps = step_idx[flat.frame_pos[sel]]
    order = np.lexsort((steps, anchor))
    sel, anchor, steps = sel[order], anchor[order], steps[order]
    vectors = flat.vectors[sel]
    lengths = np.bincount(anchor, minlength=n_anchors).astype(np.int64)
    offsets = (np.cumsum(lengths) - lengths).astype(np.int64)
    key_scale = int(step_idx[-1]) + 2
    key = anchor.astype(np.int64) * key_scale + steps

    # Running EMA, exactly as `AppearanceMemory` (seed, then renormalized
    # a * ema + (1 - a) * vec), vectorized across anchors one position at a time.
    ema = np.empty_like(vectors)
    by_len = np.argsort(-lengths, kind="stable")
    sorted_len = lengths[by_len]
    alpha = np.float32(ema_alpha)
    for k in range(int(sorted_len[0]) if len(sorted_len) else 0):
        # Anchors with more than k vectors (lengths sorted descending).
        active = by_len[: int(np.searchsorted(-sorted_len, -k, side="left"))]
        pos = offsets[active] + k
        if k == 0:
            ema[pos] = vectors[pos]
            continue
        mixed = alpha * ema[pos - 1] + (1 - alpha) * vectors[pos]
        norms = np.linalg.norm(mixed, axis=1, keepdims=True)
        keep = norms[:, 0] > 1e-12
        ema[pos] = np.where(
            keep[:, None], mixed / np.maximum(norms, 1e-12), ema[pos - 1]
        )
    return _AnchorVectors(
        vectors=vectors,
        ema=ema,
        steps=steps,
        key=key,
        key_scale=key_scale,
        offsets=offsets,
        lengths=lengths,
    )


@attrs.define
class _TrialStats:
    """Decision statistics of the trials of one gap (true track first).

    Attributes:
        true_ema: ``(T,)`` EMA cosine of the query to the true track.
        true_gal: ``(T,)`` gallery-max cosine to the true track.
        true_link: ``(T,)`` linking statistic to the true track.
        lag1: ``(T,)`` single-pair cosine of the query to the true track's latest
            vector.
        comp_ema: Per trial, ``(k,)`` EMA cosines to the competitors.
        comp_gal: Per trial, ``(k,)`` gallery-max cosines to the competitors.
        comp_link: Per trial, ``(k,)`` linking statistics to the competitors.
    """

    true_ema: np.ndarray
    true_gal: np.ndarray
    true_link: np.ndarray
    lag1: np.ndarray
    comp_ema: List[np.ndarray]
    comp_gal: List[np.ndarray]
    comp_link: List[np.ndarray]

    def __len__(self) -> int:
        """Number of trials."""
        return len(self.true_ema)


# Per-trial statistics `_run_trials` collects, keyed by their `_TrialStats` field.
_TRIAL_KEYS = tuple(a.name for a in attrs.fields(_TrialStats))


def _run_trials(
    av: _AnchorVectors,
    gap: int,
    *,
    gallery_size: int,
    query_len: int,
    max_trials: int,
    rng: np.random.Generator,
) -> _TrialStats:
    """Run the appearance trials at one gap.

    For anchor ``a`` and split position ``j`` (every :data:`TRIAL_STRIDE`-th
    vector) at step ``t``: the true track's memory is ``a``'s last
    ``gallery_size`` vectors at or before ``t - gap`` (and its running EMA
    there); the query is ``a``'s ``query_len`` vectors from ``j``. Competitors
    are the other anchors with a vector inside the true gallery's time window:
    they overlap ``a`` in time, so they are different animals. Trials without a
    competitor are dropped.
    """
    n_anchors = len(av.lengths)
    first_step = np.full(n_anchors, np.iinfo(np.int64).max)
    last_step = np.full(n_anchors, np.iinfo(np.int64).min)
    has = av.lengths > 0
    first_step[has] = av.steps[av.offsets[has]]
    last_step[has] = av.steps[av.offsets[has] + av.lengths[has] - 1]

    # Candidate split points (anchor, position) with a non-empty true gallery.
    cand_anchor: List[np.ndarray] = []
    cand_pos: List[np.ndarray] = []
    for a in np.flatnonzero(has):
        lo, n = int(av.offsets[a]), int(av.lengths[a])
        steps_a = av.steps[lo : lo + n]
        first = int(np.searchsorted(steps_a, steps_a[0] + gap, side="left"))
        js = np.arange(first, n, TRIAL_STRIDE, dtype=np.int64)
        cand_anchor.append(np.full(len(js), a, dtype=np.int64))
        cand_pos.append(lo + js)
    anchors = np.concatenate(cand_anchor) if cand_anchor else np.zeros(0, np.int64)
    positions = np.concatenate(cand_pos) if cand_pos else np.zeros(0, np.int64)
    if len(anchors) > max_trials:
        pick = np.sort(rng.choice(len(anchors), size=max(0, max_trials), replace=False))
        anchors, positions = anchors[pick], positions[pick]

    g_offsets = np.arange(gallery_size, dtype=np.int64)
    out: Dict[str, list] = {k: [] for k in _TRIAL_KEYS}
    for a, pos in zip(anchors.tolist(), positions.tolist()):
        t = int(av.steps[pos])
        w1 = t - gap
        lo_a = int(av.offsets[a])
        end_a = int(np.searchsorted(av.key, a * av.key_scale + w1, side="right")) - 1
        start_a = max(lo_a, end_a - gallery_size + 1)
        w0 = int(av.steps[start_a])

        # Competitors: other anchors with a vector in [w0, w1].
        maybe = np.flatnonzero((first_step <= w1) & (last_step >= w0))
        maybe = maybe[maybe != a]
        if maybe.size == 0:
            continue
        ends = np.searchsorted(av.key, maybe * av.key_scale + w1, side="right") - 1
        ok = (ends >= av.offsets[maybe]) & (av.steps[np.maximum(ends, 0)] >= w0)
        if not ok.any():
            continue
        comp, comp_end = maybe[ok], ends[ok]

        ends_all = np.concatenate([[end_a], comp_end])
        starts_all = np.maximum(
            np.concatenate([[lo_a], av.offsets[comp]]), ends_all - gallery_size + 1
        )
        n_gal = ends_all - starts_all + 1
        in_gal = g_offsets[None, :] < n_gal[:, None]  # (C, gs)
        gidx = np.where(in_gal, starts_all[:, None] + g_offsets[None, :], 0)
        gal = av.vectors[gidx]  # (C, gs, D)
        q_end = min(pos + query_len, lo_a + int(av.lengths[a]))
        queries = av.vectors[pos:q_end]  # (Q, D)

        cos = np.einsum("qd,cgd->cqg", queries, gal).astype(np.float64)
        cos = np.where(in_gal[:, None, :], cos, -np.inf)
        s_gal = cos[:, 0, :].max(axis=1)
        top = -np.sort(-cos, axis=2)  # descending, padding (-inf) last
        k_eff = np.minimum(LINK_TOPK, n_gal)
        top_sum = np.take_along_axis(
            np.cumsum(top, axis=2),
            np.broadcast_to((k_eff - 1)[:, None, None], (len(k_eff), top.shape[1], 1)),
            axis=2,
        )[..., 0]
        s_link = (top_sum / k_eff[:, None]).mean(axis=1)
        s_ema = av.ema[ends_all].astype(np.float64) @ queries[0].astype(np.float64)

        out["true_ema"].append(s_ema[0])
        out["true_gal"].append(s_gal[0])
        out["true_link"].append(s_link[0])
        out["lag1"].append(cos[0, 0, n_gal[0] - 1])
        out["comp_ema"].append(s_ema[1:])
        out["comp_gal"].append(s_gal[1:])
        out["comp_link"].append(s_link[1:])

    return _TrialStats(
        true_ema=np.asarray(out["true_ema"], dtype=np.float64),
        true_gal=np.asarray(out["true_gal"], dtype=np.float64),
        true_link=np.asarray(out["true_link"], dtype=np.float64),
        lag1=np.asarray(out["lag1"], dtype=np.float64),
        comp_ema=out["comp_ema"],
        comp_gal=out["comp_gal"],
        comp_link=out["comp_link"],
    )


def _link_rule(stats: _TrialStats) -> Dict[str, Optional[float]]:
    """Calibrate the linking rule ``best >= thresh and best - second >= margin``.

    ``thresh`` is the q99 of the competitors' linking statistics; ``margin`` the
    smallest grid value reaching :data:`LINK_PRECISION` (``None`` if none does).
    """
    pooled = np.concatenate(stats.comp_link)
    thresh = float(np.quantile(pooled, 0.99))
    best_comp = np.array([c.max() for c in stats.comp_link])
    # Highest and runner-up over {true} + competitors, per trial.
    second_comp = np.array(
        [np.sort(c)[-2] if len(c) > 1 else -np.inf for c in stats.comp_link]
    )
    true = stats.true_link
    correct = true > best_comp  # ties are not a correct decision
    best = np.maximum(true, best_comp)
    second = np.where(
        correct, best_comp, np.maximum(np.minimum(true, best_comp), second_comp)
    )
    margin, recall = None, None
    for m in LINK_MARGIN_GRID:
        accept = (best >= thresh) & (best - second >= m)
        n_accept = int(np.count_nonzero(accept))
        if n_accept == 0:
            continue
        n_correct = int(np.count_nonzero(accept & correct))
        if n_correct / n_accept >= LINK_PRECISION:
            margin, recall = float(m), n_correct / len(true)
            break
    return {"link_thresh": thresh, "link_margin": margin, "link_recall": recall}


def _appearance(
    flat: _Flat,
    anchor_of: np.ndarray,
    n_anchors: int,
    step_idx: np.ndarray,
    neg_cos: Optional[np.ndarray],
    *,
    track_buffer: int,
    gallery_size: int,
    query_len: int,
    ema_alpha: float,
    max_trials: int,
    seed: int,
) -> Dict[str, Any]:
    """Appearance part of the calibration (fields of :class:`VideoCalibration`)."""
    if flat.vectors is None:
        return {"status": "no_vectors"}
    av = _anchor_vectors(flat, anchor_of, n_anchors, step_idx, ema_alpha)
    if av is None:
        return {"status": "insufficient"}
    kwargs = dict(gallery_size=gallery_size, query_len=query_len, max_trials=max_trials)
    seed = int(seed) & 0xFFFFFFFF
    short = _run_trials(av, 1, rng=np.random.default_rng((seed, 1)), **kwargs)
    if len(short) < MIN_TRIALS:
        return {"status": "insufficient", "n_trials": len(short)}

    comp_max = np.array([c.max() for c in short.comp_ema])
    pooled_ema = np.concatenate(short.comp_ema)
    res: Dict[str, Any] = {
        "status": "ok",
        "n_trials": len(short),
        "assoc_rank1": float(np.mean(short.true_ema > comp_max)),
        "m_assoc": max(
            MIN_ASSOC_MARGIN, float(np.quantile(short.true_ema - comp_max, 0.1))
        ),
        "ema_cos_gate": float(np.quantile(pooled_ema, 0.95)),
        "gallery_cos_gate": float(np.quantile(np.concatenate(short.comp_gal), 0.95)),
        "mu_pos": float(np.median(short.true_ema)),
        "mu_neg": float(np.median(pooled_ema)),
    }
    if neg_cos is not None and neg_cos.size:
        pos_q05 = float(np.quantile(short.lag1, 0.05))
        res["neg_collision_rate"] = float(np.mean(neg_cos >= pos_q05))
    res["appearance_for_association"] = res["assoc_rank1"] >= MIN_ASSOC_RANK1

    long_gap = max(1, int(track_buffer))
    if long_gap == 1:
        long = short
    else:
        rng = np.random.default_rng((seed, long_gap))
        long = _run_trials(av, long_gap, rng=rng, **kwargs)
    if len(long) >= MIN_TRIALS:
        res.update(_link_rule(long))
        res["appearance_for_linking"] = (
            res["link_margin"] is not None and res["link_recall"] >= MIN_LINK_RECALL
        )
    return res


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def calibrate(
    frames: Sequence[FrameDetections],
    *,
    track_buffer: int = 30,
    gallery_size: int = 30,
    query_len: int = LINK_QUERY_LEN,
    ema_alpha: float = 0.8,
    max_trials: int = 2000,
    seed: int = 0,
    time_base: str = "auto",
) -> VideoCalibration:
    """Calibrate a video from all of its frames' detections.

    Procedure (deterministic given ``seed``):

    1. **Geometry:** ``frame_step`` / ``regular_stride`` from the frame indices;
       ``body_size`` (median raw-box diagonal of valid masks and poses with >= 2
       visible nodes), ``nn_distance`` (median same-frame nearest-neighbor
       centroid distance, duplicate pairs ignored) and the score quantiles of
       the valid detections.
    2. **Anchor tracklets:** detections of consecutive processed frames are
       chained when they are mutual nearest neighbors closer than ``0.5 *
       nn_distance``, the runner-up is >= 2x farther in both directions, and
       neither is a same-frame duplicate (IoMin > 0.5); chains are cut at any
       violation and kept when >= :data:`MIN_ANCHOR_LEN` long. ``step_p95`` /
       ``step_p99`` are per-STEP displacements along the anchor links.
    3. **Appearance trials** at ``g = 1`` and ``g = track_buffer`` steps: for
       anchor ``a`` split every :data:`TRIAL_STRIDE` vectors at step ``t``, the
       true memory is ``a``'s last ``gallery_size`` vectors at or before ``t - g``
       plus its running EMA there (seeded and renormalized like
       :class:`~sleap_nn.tracking.mot.appearance.AppearanceMemory`), the query
       is ``a``'s next ``query_len`` vectors from ``t``, and the competitors are
       the anchors with a vector in the true gallery's time window (they share
       a frame with ``a``, so they are different animals), with memories built
       the same way. Statistics: ``s_ema = cos(q, ema)``, ``s_gal = max cos(q,
       gallery)`` (``q`` the first query vector) and ``s_link`` = mean over the
       query of the top-5 mean gallery cosine. See :class:`VideoCalibration` for
       the thresholds. At most ``max_trials`` per gap (subsampled with
       ``seed``). ``status`` is
       ``"no_vectors"`` without vectors and ``"insufficient"`` with fewer than
       :data:`MIN_TRIALS` ``g = 1`` trials having a co-occurring competitor (e.g.
       a single animal); linking thresholds also need :data:`MIN_TRIALS` trials
       at the long gap.

    Logs a one-line INFO summary, and a WARNING when more than
    :data:`NEG_COLLISION_WARN` of the same-frame pairs of different animals look
    like the same animal (the embedding may encode class/sex, not identity).

    Args:
        frames: The video's :class:`FrameDetections`, in temporal order.
        track_buffer: Long-gap length (steps) for the linking statistic.
        gallery_size: Gallery length used in the trials.
        query_len: Query tracklet length used for the linking statistic.
        ema_alpha: EMA momentum used for the EMA statistic.
        max_trials: Cap on trials per gap (subsampled deterministically).
        seed: Seed for subsampling.
        time_base: How time advances between processed frames, as
            :class:`~sleap_nn.tracking.mot.params.MOTParams` ``time_base``
            (``"auto"``, ``"frames"`` or ``"steps"``); sets the step unit of the
            displacement statistics and the gaps.

    Returns:
        The :class:`VideoCalibration`. Never raises on degenerate input (empty
        video, single animal, no vectors).

    Raises:
        ValueError: On an unknown ``time_base``.
    """
    if time_base not in ("auto", "frames", "steps"):
        raise ValueError(
            f"time_base must be 'auto', 'frames' or 'steps', got {time_base!r}."
        )
    frames = list(frames)
    gallery_size = max(1, int(gallery_size))
    query_len = max(1, int(query_len))
    max_trials = max(0, int(max_trials))

    frame_idx = np.array([int(fd.frame_idx) for fd in frames], dtype=np.int64)
    frame_step, regular, steps = _frame_steps(frame_idx, time_base)
    step_idx = np.concatenate([[0], np.cumsum(steps)]).astype(np.int64)
    flat = _flatten(frames)
    n_total = len(flat.frame_pos)

    # Same-frame geometry: spacing, duplicates, negative cosines.
    nn_dist, duplicate, neg_cos = _same_frame_pairs(flat)
    finite_nn = nn_dist[np.isfinite(nn_dist)]
    nn_distance = float(np.median(finite_nn)) if finite_nn.size else float("nan")
    diag = np.hypot(*(flat.raw_boxes[:, 2:] - flat.raw_boxes[:, :2]).T)
    diag = diag[flat.sized & np.isfinite(diag) & (diag > 0)]
    body_size = float(np.median(diag)) if diag.size else float("nan")
    if not np.isfinite(body_size):
        body_size = nn_distance if np.isfinite(nn_distance) else _DEFAULT_SIZE
    if not np.isfinite(nn_distance):
        nn_distance = body_size
    finite_scores = flat.scores[np.isfinite(flat.scores)]
    score_quantiles = (
        tuple(float(q) for q in np.quantile(finite_scores, SCORE_QUANTILE_LEVELS))
        if finite_scores.size
        else ()
    )

    # Anchor tracklets.
    nxt = _anchor_links(flat, duplicate, _ANCHOR_MAX_NN_FRACTION * nn_distance)
    roots = _chain_roots(nxt)
    lengths = np.bincount(roots, minlength=n_total)
    kept = lengths[roots] >= MIN_ANCHOR_LEN
    anchor_roots = np.unique(roots[kept])
    anchor_of = np.full(n_total, -1, dtype=np.int64)
    anchor_of[kept] = np.searchsorted(anchor_roots, roots[kept])
    n_anchors = len(anchor_roots)

    step_p95 = step_p99 = float("nan")
    src = np.flatnonzero((nxt >= 0) & kept)
    if src.size:
        disp = np.linalg.norm(flat.centroids[nxt[src]] - flat.centroids[src], axis=1)
        disp = disp / steps[flat.frame_pos[src]]
        step_p95, step_p99 = (float(v) for v in np.percentile(disp, [95, 99]))

    appearance = _appearance(
        flat,
        anchor_of,
        n_anchors,
        step_idx,
        neg_cos,
        track_buffer=track_buffer,
        gallery_size=gallery_size,
        query_len=query_len,
        ema_alpha=ema_alpha,
        max_trials=max_trials,
        seed=int(seed),
    )

    cal = VideoCalibration(
        status=str(appearance.get("status", "no_vectors")),
        n_frames=len(frames),
        n_detections=int(n_total),
        frame_step=int(frame_step),
        regular_stride=bool(regular),
        body_size=float(body_size),
        nn_distance=float(nn_distance),
        step_p95=step_p95,
        step_p99=step_p99,
        score_quantiles=score_quantiles,
        n_anchors=int(n_anchors),
        n_trials=int(appearance.get("n_trials", 0)),
        assoc_rank1=_opt_float(appearance.get("assoc_rank1")),
        m_assoc=_opt_float(appearance.get("m_assoc")),
        ema_cos_gate=_opt_float(appearance.get("ema_cos_gate")),
        gallery_cos_gate=_opt_float(appearance.get("gallery_cos_gate")),
        mu_pos=_opt_float(appearance.get("mu_pos")),
        mu_neg=_opt_float(appearance.get("mu_neg")),
        link_thresh=_opt_float(appearance.get("link_thresh")),
        link_margin=_opt_float(appearance.get("link_margin")),
        link_recall=_opt_float(appearance.get("link_recall")),
        neg_collision_rate=_opt_float(appearance.get("neg_collision_rate")),
        appearance_for_association=bool(
            appearance.get("appearance_for_association", False)
        ),
        appearance_for_linking=bool(appearance.get("appearance_for_linking", False)),
    )
    logger.info(cal.summary())
    if (
        cal.neg_collision_rate is not None
        and cal.neg_collision_rate > NEG_COLLISION_WARN
    ):
        logger.warning(
            f"MOT calibration: {cal.neg_collision_rate:.1%} of same-frame pairs of "
            "different animals are as similar as the same animal one step apart "
            f"(more than {NEG_COLLISION_WARN:.0%}): the embedding may encode "
            "class/sex rather than identity."
        )
    return cal
