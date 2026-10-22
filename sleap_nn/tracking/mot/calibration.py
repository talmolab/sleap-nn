"""Per-video self-calibration for the MOT trackers.

:func:`calibrate` looks at every detection of a video BEFORE tracking it (all
vectors already exist in sleap-nn's offline tracking flow) and measures:

- **Geometry:** body size, nearest-neighbor spacing, the processing stride and
  per-step motion, from *anchor tracklets* -- detections chained frame-to-frame
  only where the association is unambiguous (mutual nearest neighbors, well
  separated from any competitor), cut at any doubt.
- **Time base:** the stride and how time advances between processed frames,
  through the shared step helpers (:func:`stride_from_counts`,
  :func:`resolve_time_base`, :func:`steps_between`, :func:`frame_steps`) that the
  trackers and the identity linker use too, so all of them count the same steps.
- **Scores:** detection-confidence quantiles (:func:`score_normalizer` turns
  them into the per-video score normalizer of ``MOTParams.normalize_scores``).
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

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.detections import FrameDetections

# Query length (vectors) of the linking statistic. The identity linker scores a
# gap link / swap with UP TO this many query vectors; the linking rule is
# calibrated for every query length from 1 to this (`link_thresh_by_len`), so a
# short query is held to the threshold of its own length.
LINK_QUERY_LEN = 10
# Gallery length (vectors) of the linking statistic: the identity linker's end
# galleries and pre-event galleries. The long-gap trials always use it, whatever
# the trackers' own gallery length, so `link_thresh` / `link_margin` describe
# the statistic the linker computes.
LINK_GALLERY_LEN = 30
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
# Floor of the per-video score normalizer (`score_normalizer`).
SCORE_NORMALIZER_FLOOR = 1e-6
# Maximum spacing (vectors) of the appearance trials' split points along an
# anchor. The actual stride adapts to the evidence (`_trial_stride`): short clips
# use every vector so they still reach MIN_TRIALS, long ones space the trials out.
MAX_TRIAL_STRIDE = 10
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
# Motion-derived body size of extent-less (single-node) detections:
# MOTION_BODY_FACTOR x step_p99, at least MIN_MOTION_BODY px (see
# `resolve_body_size`). The trackers' centroid similarity (scale body / 2) then
# reaches 0.3 at a displacement of ~6 x step_p99.
MOTION_BODY_FACTOR = 8.0
MIN_MOTION_BODY = 16.0
# Where `VideoCalibration.body_size` came from.
BODY_SIZE_SOURCES = ("measured", "motion", "nn_distance", "default", "override")
# Resolved time bases (`MOTParams.time_base` minus "auto").
RESOLVED_TIME_BASES = ("frames", "steps")
# Element budget of the batched (frames x N x N) temporaries.
_CHUNK_ELEMS = 1 << 20


@attrs.frozen
class VideoCalibration:
    """Calibration of one video. Every field is a plain Python type.

    Attributes:
        status: Appearance-calibration status: ``"ok"``, ``"no_vectors"``,
            ``"insufficient"`` or ``"skipped"`` (``calibrate(appearance=False)``:
            nobody reads the appearance part, so it was not computed).
        n_frames: Frames seen.
        n_detections: Valid detections seen.
        frame_step: Most common positive frame-index delta (1 if unknown).
        regular_stride: Whether at least half of the deltas equal ``frame_step``.
        time_base: The RESOLVED time base, ``"frames"`` or ``"steps"`` (see
            :func:`resolve_time_base`): with :func:`frame_steps` (``frame_step``
            in the ``"frames"`` time base) it gives every processed frame's step
            position, exactly as the trackers count steps.
        body_size: Body size (px), from the fallback chain of
            :func:`resolve_body_size`: the median raw-box diagonal of valid
            masks and multi-node poses; else (single-node poses) a
            motion-derived size; else ``nn_distance``; else 50. A
            ``body_size`` given to :func:`calibrate` replaces all of them.
        body_size_source: Where ``body_size`` came from: ``"measured"``,
            ``"motion"``, ``"nn_distance"``, ``"default"`` or ``"override"``.
        nn_distance: Median same-frame nearest-neighbor centroid distance (px),
            ignoring duplicate pairs (IoMin > 0.5); falls back to ``body_size``.
        step_p95: 95th percentile per-step centroid displacement along anchors (px).
        step_p99: 99th percentile per-step centroid displacement along anchors (px).
        score_quantiles: Score quantiles at 0.05, 0.25, 0.5, 0.75 and 0.95, of
            the RAW detection scores (before any normalization).
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
        link_thresh: q99 of competitor linking statistics at the long gap, for
            a full query (:data:`LINK_QUERY_LEN` vectors, gallery
            :data:`LINK_GALLERY_LEN`).
        link_margin: Smallest margin in :data:`LINK_MARGIN_GRID` giving linking
            precision ``>= LINK_PRECISION`` on the trials, or ``None``.
        link_recall: Recall of the calibrated linking rule on the trials.
        link_thresh_by_len: ``link_thresh`` for a query of ``n`` vectors at index
            ``n - 1`` (``n = 1 .. LINK_QUERY_LEN``; the last equals
            ``link_thresh``): a short query is noisier, so it gets its own,
            stricter threshold. Empty when not calibrated.
        link_margin_by_len: ``link_margin`` per query length, likewise (``None``
            where no grid margin reaches the precision at that length: such a
            query is too short for appearance to decide).
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
    time_base: str = "frames"
    body_size: float = 50.0
    body_size_source: str = "default"
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
    link_thresh_by_len: tuple = ()
    link_margin_by_len: tuple = ()
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
            f"{self.frame_step} ({stride}, time base {self.time_base}), body "
            f"{_fmt(self.body_size, 1)} px ({self.body_size_source}), "
            f"NN {_fmt(self.nn_distance, 1)} px, step p95/p99 "
            f"{_fmt(self.step_p95, 1)}/{_fmt(self.step_p99, 1)} px, "
            f"{self.n_anchors} anchors"
        )
        if self.status == "skipped":
            appearance = "appearance not calibrated (not used)"
        elif self.status != "ok":
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


def _flatten(frames: Sequence[FrameDetections], vectors: bool = True) -> _Flat:
    """Flatten the valid detections of ``frames`` (see :class:`_Flat`).

    With ``vectors=False`` the appearance vectors are left out (``vectors`` is
    ``None``), skipping their per-detection work.
    """
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

    dim = _vector_dim(frames, valid) if vectors else 0
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


# Shared step helpers. The trackers (`BaseMOTTracker._advance`, online and after
# `prepare`), `calibrate` and the identity linker all count time with these, so
# a gap is the same number of steps everywhere.


def stride_from_counts(counts: Mapping[int, int]) -> Tuple[int, bool]:
    """Processing stride from the counts of the positive frame-index deltas.

    The stride (``frame_step``) is the most common positive delta (ties: the
    smallest); it is regular when at least half of the deltas equal it. This is
    the rule of :func:`calibrate` (over the whole video) and of an uncalibrated
    tracker (over the deltas seen so far).

    Args:
        counts: ``{delta: count}``; non-positive deltas and counts are ignored.

    Returns:
        ``(frame_step, regular)``; ``(1, True)`` when there is no positive delta.
    """
    kept = {int(d): int(c) for d, c in counts.items() if int(d) > 0 and int(c) > 0}
    if not kept:
        return 1, True
    frame_step = min(kept, key=lambda d: (-kept[d], d))
    return frame_step, bool(kept[frame_step] >= 0.5 * sum(kept.values()))


def frame_stride(frame_indices: Sequence[int]) -> Tuple[int, bool]:
    """``(frame_step, regular)`` of a sequence of frame indices.

    See :func:`stride_from_counts` (applied to the positive consecutive deltas).
    """
    deltas = np.diff(np.asarray(frame_indices, dtype=np.int64).reshape(-1))
    values, counts = np.unique(deltas[deltas > 0], return_counts=True)
    return stride_from_counts(dict(zip(values.tolist(), counts.tolist())))


def resolve_time_base(time_base: str, regular: bool) -> str:
    """Resolve ``MOTParams.time_base`` to ``"frames"`` or ``"steps"``.

    Args:
        time_base: ``"auto"``, ``"frames"`` or ``"steps"``.
        regular: Whether the processed frames are regularly strided.

    Returns:
        ``time_base`` itself when explicit; for ``"auto"``, ``"frames"`` when the
        stride is regular, else ``"steps"``.

    Raises:
        ValueError: On an unknown ``time_base``.
    """
    if time_base == "auto":
        return "frames" if regular else "steps"
    if time_base not in RESOLVED_TIME_BASES:
        raise ValueError(
            f"time_base must be 'auto', 'frames' or 'steps', got {time_base!r}."
        )
    return time_base


def steps_between(
    delta: Union[int, float, np.ndarray], frame_step: int
) -> Union[int, np.ndarray]:
    """Time steps between two processed frames ``delta`` frame indices apart.

    The ``"frames"`` time base: ``max(1, round(delta / frame_step))``, rounding
    half to even (``np.rint``, like Python's ``round``) -- the ONE rounding rule
    of the trackers, :func:`calibrate` and the identity linker. A non-positive
    ``delta`` (a repeated or backwards frame) counts one step. In the
    ``"steps"`` time base every processed frame is one step and this is not used.

    Args:
        delta: Frame-index difference(s) to the previous processed frame.
        frame_step: The stride (floored at 1).

    Returns:
        An ``int`` for a scalar ``delta``, else an ``int64`` array.
    """
    frame_step = max(1, int(frame_step))
    steps = np.maximum(
        1.0, np.rint(np.asarray(delta, dtype=np.float64) / frame_step)
    ).astype(np.int64)
    return int(steps) if steps.ndim == 0 else steps


def frame_steps(
    frame_indices: Sequence[int], time_base: str, frame_step: int = 1
) -> np.ndarray:
    """Step position of each processed frame (the first at step 0).

    ``"frames"``: consecutive frames are :func:`steps_between` steps apart, so
    a strided video behaves like a dense one and genuine gaps still count;
    ``"steps"``: one step per processed frame. With a calibration, call it as
    ``frame_steps(idx, cal.time_base, cal.frame_step)``.

    Args:
        frame_indices: Frame indices of the processed frames, in processing order.
        time_base: The RESOLVED time base, ``"frames"`` or ``"steps"``.
        frame_step: The stride (used in the ``"frames"`` time base).

    Returns:
        ``(F,)`` int64 step positions.

    Raises:
        ValueError: On a time base other than ``"frames"`` / ``"steps"``.
    """
    if time_base not in RESOLVED_TIME_BASES:
        raise ValueError(
            f"frame_steps needs a resolved time base ('frames' or 'steps'), got "
            f"{time_base!r}."
        )
    idx = np.asarray(frame_indices, dtype=np.float64).reshape(-1)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    if time_base == "steps":
        return np.arange(len(idx), dtype=np.int64)
    inc = np.asarray(steps_between(np.diff(idx), frame_step), dtype=np.int64)
    return np.concatenate([[0], np.cumsum(inc)]).astype(np.int64)


def resolve_body_size(
    measured: float, step_p99: float, nn_distance: float
) -> Tuple[float, str]:
    """The body size (px) and its source (the shared fallback chain).

    Used by :func:`calibrate` and by the uncalibrated trackers:

    1. ``measured`` (median raw-box diagonal of sized detections) ->
       ``"measured"``.
    2. Else (single-node detections have no extent) the per-step motion:
       ``MOTION_BODY_FACTOR * step_p99``, at least :data:`MIN_MOTION_BODY` and at
       most ``nn_distance`` (when known) -> ``"motion"``. Inter-animal spacing is
       not a body length: in a sparse arena it would open the centroid gate
       (scale ``body / 2``) most of the way to the next animal.
    3. Else ``nn_distance`` -> ``"nn_distance"``.
    4. Else 50 px -> ``"default"``.

    Args:
        measured: Measured body size (NaN if none).
        step_p99: 99th percentile per-step displacement (NaN if unknown).
        nn_distance: Same-frame nearest-neighbor spacing (NaN if unknown).

    Returns:
        ``(body_size, source)``.
    """

    def ok(v: float) -> bool:
        return v is not None and bool(np.isfinite(v)) and v > 0

    if ok(measured):
        return float(measured), "measured"
    if step_p99 is not None and np.isfinite(step_p99) and step_p99 >= 0:
        body = max(MOTION_BODY_FACTOR * float(step_p99), MIN_MOTION_BODY)
        if ok(nn_distance):
            body = min(body, float(nn_distance))
        return float(body), "motion"
    if ok(nn_distance):
        return float(nn_distance), "nn_distance"
    return _DEFAULT_SIZE, "default"


def score_normalizer(calibration: Optional["VideoCalibration"]) -> float:
    """Per-video score normalizer of ``MOTParams.normalize_scores``.

    ``max(q95, SCORE_NORMALIZER_FLOOR)`` of the calibration's (raw) score
    quantiles; 1.0 without a calibration or scores. Dividing a detection score by
    it (then clipping to ``[0, 1]``, see
    :func:`~sleap_nn.tracking.mot.detections.normalize_scores`) maps a typical
    detection of the video to ~0.85-1.0 whatever the model's score scale.

    Args:
        calibration: The video's calibration, or ``None``.

    Returns:
        The normalizer (> 0).
    """
    if calibration is None or len(calibration.score_quantiles) < len(
        SCORE_QUANTILE_LEVELS
    ):
        return 1.0
    q95 = float(calibration.score_quantiles[SCORE_QUANTILE_LEVELS.index(0.95)])
    if not np.isfinite(q95):
        return 1.0
    return max(q95, SCORE_NORMALIZER_FLOOR)


def link_rule_tables(
    calibration: Optional["VideoCalibration"], query_len: int = LINK_QUERY_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """The calibrated linking rule for each query length.

    Entry ``n - 1`` is the ``(threshold, margin)`` for a query of ``n`` vectors
    (``n = 1 .. query_len``; longer calibrated tables are cut, shorter ones
    repeat their last entry). Without per-length tables (a hand-built or older
    calibration) the full-length ``link_thresh`` / ``link_margin`` apply to every
    length. A length whose margin is unknown (no grid margin reached the
    precision there) gets NaN for BOTH: appearance cannot decide such a query.

    Args:
        calibration: The video's calibration, or ``None``.
        query_len: Longest query.

    Returns:
        ``(thresholds, margins)``, two ``(query_len,)`` float arrays (NaN where
        undecidable).
    """
    q = max(1, int(query_len))
    thresh = np.full(q, np.nan)
    margin = np.full(q, np.nan)
    if calibration is None:
        return thresh, margin

    def fill(out: np.ndarray, table: Sequence, scalar: Optional[float]) -> None:
        if len(table):
            for n in range(q):
                v = table[min(n, len(table) - 1)]
                out[n] = np.nan if v is None else float(v)
        elif scalar is not None:
            out[:] = float(scalar)

    fill(thresh, tuple(calibration.link_thresh_by_len), calibration.link_thresh)
    fill(margin, tuple(calibration.link_margin_by_len), calibration.link_margin)
    bad = ~(np.isfinite(thresh) & np.isfinite(margin))
    thresh[bad] = np.nan
    margin[bad] = np.nan
    return thresh, margin


def _keypoint_extents(fd: FrameDetections) -> FrameDetections:
    """``fd`` with pose raw boxes = visible-keypoint extents floored at 1 px.

    Undoes pseudo boxes (``extract_detections(pseudo_box_size=...)``) so the
    duplicate test sees what :func:`calibrate`'s first pass sees. Masks and rows
    without a visible keypoint keep their raw box. Returns ``fd`` itself when
    there is nothing to change (no copy of the other arrays either way).
    """
    if fd.keypoints is None or len(fd) == 0:
        return fd
    kp = np.asarray(fd.keypoints, dtype=np.float64)
    vis = np.isfinite(kp).all(axis=-1)
    has = vis.any(axis=1) & np.asarray(fd.valid, dtype=bool)
    if not has.any():
        return fd
    lo = np.where(vis[..., None], kp, np.inf).min(axis=1)
    hi = np.where(vis[..., None], kp, -np.inf).max(axis=1)
    center = 0.5 * (lo[has] + hi[has])
    half = 0.5 * np.maximum(hi[has] - lo[has], 1.0)
    raw = np.array(fd.raw_boxes, dtype=np.float64, copy=True)
    raw[has] = np.concatenate([center - half, center + half], axis=1)
    return attrs.evolve(fd, raw_boxes=raw)


def pair_geometry(
    prev: Optional[FrameDetections],
    cur: FrameDetections,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same-frame spacing of ``cur`` and its unambiguous links to ``prev``.

    The online counterpart of :func:`calibrate`'s geometry (used by uncalibrated
    trackers), with the same rules: duplicate pairs (IoMin > 0.5) are not
    neighbors, and a ``prev -> cur`` link needs mutual nearest neighbors closer
    than ``max_dist`` whose runner-ups are at least twice farther.

    Args:
        prev: The previous processed frame's detections (``None``: no links).
        cur: This frame's detections. Pose pseudo boxes (degenerate geometry
            extracted with a ``pseudo_box_size``) are replaced by the keypoint
            extent floored at 1 px, the geometry :func:`calibrate` sees.
        max_dist: Link gate (px). ``None``: half the median same-frame spacing
            of the two frames when measurable, else no gate.

    Returns:
        ``(nn, links)``: the finite nearest-neighbor distances of ``cur``'s valid
        detections and the lengths (px) of the unambiguous links.
    """
    frames = [_keypoint_extents(f) for f in ([cur] if prev is None else [prev, cur])]
    flat = _flatten(frames)
    flat.vectors = None  # Geometry only.
    nn_dist, duplicate, _ = _same_frame_pairs(flat)
    nn = nn_dist[flat.frame_pos == len(frames) - 1]
    nn = nn[np.isfinite(nn)]
    if prev is None:
        return nn, np.zeros(0, dtype=np.float64)
    if max_dist is None:
        finite = nn_dist[np.isfinite(nn_dist)]
        max_dist = (
            _ANCHOR_MAX_NN_FRACTION * np.median(finite) if finite.size else np.inf
        )
    nxt = _anchor_links(flat, duplicate, float(max_dist))
    src = np.flatnonzero(nxt >= 0)
    links = np.linalg.norm(flat.centroids[nxt[src]] - flat.centroids[src], axis=1)
    return nn, links


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
        true_link_n: ``(T, query_len)`` linking statistic to the true track using
            only the first ``n`` query vectors, at column ``n - 1`` (the last
            column is ``true_link``).
        comp_link_n: Per trial, ``(k, query_len)``: the same for the competitors.
    """

    true_ema: np.ndarray
    true_gal: np.ndarray
    true_link: np.ndarray
    lag1: np.ndarray
    comp_ema: List[np.ndarray]
    comp_gal: List[np.ndarray]
    comp_link: List[np.ndarray]
    true_link_n: np.ndarray
    comp_link_n: List[np.ndarray]

    def __len__(self) -> int:
        """Number of trials."""
        return len(self.true_ema)


# Per-trial statistics `_run_trials` collects, keyed by their `_TrialStats` field.
_TRIAL_KEYS = tuple(a.name for a in attrs.fields(_TrialStats))


def _trial_stride(n_positions: int, max_trials: int) -> int:
    """Spacing of the trial split points along each anchor (deterministic).

    ``clip(n_positions // max_trials, 1, MAX_TRIAL_STRIDE)``: a short clip uses
    every candidate position so it can still reach :data:`MIN_TRIALS` (a
    400-frame clip of two animals has ~800), a long one spaces the trials out
    (less correlated, less work) before any subsampling to ``max_trials``.

    Args:
        n_positions: Candidate split positions over all anchors.
        max_trials: Cap on the trials.

    Returns:
        The stride, in ``[1, MAX_TRIAL_STRIDE]``.
    """
    stride = int(n_positions) // max(1, int(max_trials))
    return int(min(max(stride, 1), MAX_TRIAL_STRIDE))


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

    For anchor ``a`` and split position ``j`` (every ``stride``-th vector, see
    :func:`_trial_stride`) at step ``t``: the true track's memory is ``a``'s last
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
    spans: List[Tuple[int, int, int, int]] = []
    for a in np.flatnonzero(has):
        lo, n = int(av.offsets[a]), int(av.lengths[a])
        steps_a = av.steps[lo : lo + n]
        first = int(np.searchsorted(steps_a, steps_a[0] + gap, side="left"))
        spans.append((int(a), lo, first, n))
    stride = _trial_stride(
        sum(max(0, n - first) for _, _, first, n in spans), max_trials
    )
    cand_anchor: List[np.ndarray] = []
    cand_pos: List[np.ndarray] = []
    for a, lo, first, n in spans:
        js = np.arange(first, n, stride, dtype=np.int64)
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
        per_q = top_sum / k_eff[:, None]  # (C, Q)
        # Linking statistic of the first n queries, n = 1 .. query_len (a query
        # cut short by the anchor's end keeps its full-length value).
        s_link_n = np.cumsum(per_q, axis=1) / np.arange(1, per_q.shape[1] + 1)
        if s_link_n.shape[1] < query_len:
            pad = np.repeat(s_link_n[:, -1:], query_len - s_link_n.shape[1], axis=1)
            s_link_n = np.concatenate([s_link_n, pad], axis=1)
        s_link = s_link_n[:, -1]
        s_ema = av.ema[ends_all].astype(np.float64) @ queries[0].astype(np.float64)

        out["true_ema"].append(s_ema[0])
        out["true_gal"].append(s_gal[0])
        out["true_link"].append(s_link[0])
        out["lag1"].append(cos[0, 0, n_gal[0] - 1])
        out["comp_ema"].append(s_ema[1:])
        out["comp_gal"].append(s_gal[1:])
        out["comp_link"].append(s_link[1:])
        out["true_link_n"].append(s_link_n[0])
        out["comp_link_n"].append(s_link_n[1:])

    return _TrialStats(
        true_ema=np.asarray(out["true_ema"], dtype=np.float64),
        true_gal=np.asarray(out["true_gal"], dtype=np.float64),
        true_link=np.asarray(out["true_link"], dtype=np.float64),
        lag1=np.asarray(out["lag1"], dtype=np.float64),
        comp_ema=out["comp_ema"],
        comp_gal=out["comp_gal"],
        comp_link=out["comp_link"],
        true_link_n=np.asarray(out["true_link_n"], dtype=np.float64).reshape(
            -1, query_len
        ),
        comp_link_n=out["comp_link_n"],
    )


def _link_rule_at(
    true: np.ndarray, comp: List[np.ndarray]
) -> Tuple[float, Optional[float], Optional[float]]:
    """Calibrate ``best >= thresh and best - second >= margin`` for one statistic.

    ``thresh`` is the q99 of the competitors' linking statistics; ``margin`` the
    smallest grid value reaching :data:`LINK_PRECISION` (``None`` if none does).

    Args:
        true: ``(T,)`` statistic of the true track per trial.
        comp: Per trial, ``(k,)`` statistics of the competitors.

    Returns:
        ``(thresh, margin, recall)``.
    """
    pooled = np.concatenate(comp)
    thresh = float(np.quantile(pooled, 0.99))
    best_comp = np.array([c.max() for c in comp])
    # Highest and runner-up over {true} + competitors, per trial.
    second_comp = np.array([np.sort(c)[-2] if len(c) > 1 else -np.inf for c in comp])
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
    return thresh, margin, recall


def _link_rule(stats: _TrialStats) -> Dict[str, Any]:
    """Calibrate the linking rule per query length (see :func:`_link_rule_at`).

    The rule is calibrated on the statistic of the first ``n`` query vectors for
    every ``n`` (``link_thresh_by_len`` / ``link_margin_by_len``): a 1-vector
    query is much noisier than a 10-vector one, so one threshold for all lengths
    would let short queries of different animals through far more often than the
    calibrated rate. ``link_thresh`` / ``link_margin`` / ``link_recall`` are the
    full-length values.
    """
    q = stats.true_link_n.shape[1]
    by_len = [
        _link_rule_at(stats.true_link_n[:, n], [c[:, n] for c in stats.comp_link_n])
        for n in range(q)
    ]
    thresh, margin, recall = _link_rule_at(stats.true_link, stats.comp_link)
    return {
        "link_thresh": thresh,
        "link_margin": margin,
        "link_recall": recall,
        "link_thresh_by_len": tuple(float(t) for t, _, _ in by_len),
        "link_margin_by_len": tuple(m for _, m, _ in by_len),
    }


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

    # The linking statistic is the linker's: gallery LINK_GALLERY_LEN whatever
    # the trackers' own gallery length (`gallery_size`).
    long_gap = max(1, int(track_buffer))
    if long_gap == 1 and gallery_size == LINK_GALLERY_LEN:
        long = short
    else:
        rng = np.random.default_rng((seed, long_gap))
        long = _run_trials(
            av,
            long_gap,
            rng=rng,
            gallery_size=LINK_GALLERY_LEN,
            query_len=query_len,
            max_trials=max_trials,
        )
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
    body_size: Optional[float] = None,
    appearance: bool = True,
) -> VideoCalibration:
    """Calibrate a video from all of its frames' detections.

    Procedure (deterministic given ``seed``):

    1. **Geometry:** ``frame_step`` / ``regular_stride`` from the frame indices
       and the resolved ``time_base`` (:func:`frame_stride`,
       :func:`resolve_time_base`; step positions from :func:`frame_steps`);
       ``nn_distance`` (median same-frame nearest-neighbor centroid distance,
       duplicate pairs ignored) and the score quantiles of the valid detections.
    2. **Anchor tracklets:** detections of consecutive processed frames are
       chained when they are mutual nearest neighbors closer than ``0.5 *
       nn_distance`` (no distance gate when no frame holds two detections: a
       lone detection has no competitor), the runner-up is >= 2x farther in
       both directions, and neither is a same-frame duplicate (IoMin > 0.5);
       chains are cut at any violation and kept when >= :data:`MIN_ANCHOR_LEN`
       long. ``step_p95`` / ``step_p99`` are per-STEP displacements along the
       anchor links.
    3. **Body size** (:func:`resolve_body_size`): the median raw-box diagonal of
       valid masks and poses with >= 2 visible nodes; else (single-node poses)
       ``MOTION_BODY_FACTOR * step_p99`` clipped to ``[MIN_MOTION_BODY,
       nn_distance]``; else ``nn_distance``; else 50 px. ``body_size_source``
       records which; a WARNING is logged when only the spacing or the default
       was left.
    4. **Appearance trials** at ``g = 1`` and ``g = track_buffer`` steps: for
       anchor ``a`` split every ``stride`` vectors (``clip(candidates //
       max_trials, 1, MAX_TRIAL_STRIDE)``) at step ``t``, the
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
       at the long gap. The long-gap (linking) trials always use the linker's
       gallery length :data:`LINK_GALLERY_LEN` and calibrate the rule for every
       query length up to ``query_len`` (``link_thresh_by_len``).

    Logs a one-line INFO summary, and a WARNING when more than
    :data:`NEG_COLLISION_WARN` of the same-frame pairs of different animals look
    like the same animal (the embedding may encode class/sex, not identity).

    Args:
        frames: The video's :class:`FrameDetections`, in temporal order.
        track_buffer: Long-gap length (steps) for the linking statistic.
        gallery_size: Gallery length of the ``g = 1`` trials: the length of the
            tracker's appearance memory, which ``gallery_cos_gate`` describes
            (DeepSORT: ``gallery_budget``). The linking trials use
            :data:`LINK_GALLERY_LEN` regardless.
        query_len: Query tracklet length used for the linking statistic.
        ema_alpha: EMA momentum used for the EMA statistic.
        max_trials: Cap on trials per gap (subsampled deterministically).
        seed: Seed for subsampling.
        time_base: How time advances between processed frames, as
            :class:`~sleap_nn.tracking.mot.params.MOTParams` ``time_base``
            (``"auto"``, ``"frames"`` or ``"steps"``); sets the step unit of the
            displacement statistics and the gaps. The resolved value is
            ``VideoCalibration.time_base``.
        body_size: A known body size (px, e.g. ``MOTParams.body_size``) that
            replaces the measured one (``body_size_source="override"``).
        appearance: Calibrate the appearance part. ``False`` (a geometry-only
            tracker without identity linking: nothing reads it) skips the
            vector work entirely and reports ``status="skipped"``; the geometry
            is identical either way.

    Returns:
        The :class:`VideoCalibration`. Never raises on degenerate input (empty
        video, single animal, no vectors).

    Raises:
        ValueError: On an unknown ``time_base``.
    """
    if time_base not in ("auto",) + RESOLVED_TIME_BASES:
        raise ValueError(
            f"time_base must be 'auto', 'frames' or 'steps', got {time_base!r}."
        )
    frames = list(frames)
    gallery_size = max(1, int(gallery_size))
    query_len = max(1, int(query_len))
    max_trials = max(0, int(max_trials))

    frame_idx = np.array([int(fd.frame_idx) for fd in frames], dtype=np.int64)
    frame_step, regular = frame_stride(frame_idx)
    resolved_time_base = resolve_time_base(time_base, regular)
    step_idx = frame_steps(frame_idx, resolved_time_base, frame_step)
    steps = np.diff(step_idx)
    flat = _flatten(frames, vectors=bool(appearance))
    n_total = len(flat.frame_pos)

    # Same-frame geometry: spacing, duplicates, negative cosines.
    nn_dist, duplicate, neg_cos = _same_frame_pairs(flat)
    finite_nn = nn_dist[np.isfinite(nn_dist)]
    nn_measured = float(np.median(finite_nn)) if finite_nn.size else float("nan")
    diag = np.hypot(*(flat.raw_boxes[:, 2:] - flat.raw_boxes[:, :2]).T)
    diag = diag[flat.sized & np.isfinite(diag) & (diag > 0)]
    measured_body = float(np.median(diag)) if diag.size else float("nan")
    finite_scores = flat.scores[np.isfinite(flat.scores)]
    score_quantiles = (
        tuple(float(q) for q in np.quantile(finite_scores, SCORE_QUANTILE_LEVELS))
        if finite_scores.size
        else ()
    )

    # Anchor tracklets. Without any same-frame pair (one animal), a link has no
    # competitor: no distance gate.
    max_link = (
        _ANCHOR_MAX_NN_FRACTION * nn_measured if np.isfinite(nn_measured) else np.inf
    )
    nxt = _anchor_links(flat, duplicate, max_link)
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

    # Body size (fallback chain), then the spacing's own fallback.
    if body_size is not None and np.isfinite(body_size) and body_size > 0:
        body, body_source = float(body_size), "override"
    else:
        body, body_source = resolve_body_size(measured_body, step_p99, nn_measured)
    nn_distance = nn_measured if np.isfinite(nn_measured) else body

    if appearance:
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
    else:
        appearance = {"status": "skipped"}

    cal = VideoCalibration(
        status=str(appearance.get("status", "no_vectors")),
        n_frames=len(frames),
        n_detections=int(n_total),
        frame_step=int(frame_step),
        regular_stride=bool(regular),
        time_base=resolved_time_base,
        body_size=float(body),
        body_size_source=body_source,
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
        link_thresh_by_len=tuple(
            _opt_float(v) for v in appearance.get("link_thresh_by_len", ())
        ),
        link_margin_by_len=tuple(
            _opt_float(v) for v in appearance.get("link_margin_by_len", ())
        ),
        neg_collision_rate=_opt_float(appearance.get("neg_collision_rate")),
        appearance_for_association=bool(
            appearance.get("appearance_for_association", False)
        ),
        appearance_for_linking=bool(appearance.get("appearance_for_linking", False)),
    )
    logger.info(cal.summary())
    if n_total and body_source in ("nn_distance", "default"):
        logger.warning(
            "MOT calibration: no body size could be measured (no detection with "
            "an extent, e.g. single-node poses) and too little unambiguous motion "
            f"to size it from; using the {body_source.replace('_', ' ')} "
            f"({body:.1f} px) as the body size, which sets the centroid-similarity "
            "gate and pseudo-box size. Pass the animal length in px as "
            "MOTParams.body_size (--mot_param body_size=<px>) if it is far off."
        )
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
