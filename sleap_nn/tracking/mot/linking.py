"""Identity linking: object-level long-term memory for tracked videos.

:func:`link_tracklets` runs once per video AFTER tracking (any tracker, the
``classic`` one included). It treats every track as an identity made of
tracklets and:

1. **Corrects swaps at crossings** (needs calibrated appearance): around each
   ambiguity event (two or more identities' boxes overlapping), it compares each
   identity's appearance gallery from before the event with the vectors right
   after it, and applies a permutation only when appearance prefers it by the
   calibrated margin.
2. **Links tracklets across gaps** (re-identification): an identity that ends
   and one that starts later, that never co-occur (cannot-link) and are
   spatially plausible given the gap, are merged when their appearance agrees
   (calibrated threshold and margin), or -- without usable appearance -- when the
   match is spatially unique within a short gap (an identity's own later
   fragments, reachable through the candidate, do not count as competitors).
3. **Enforces a closed set** when the number of animals ``N`` is known: the
   ``N`` best-supported identities that are present together become anchors
   (one per animal), and every other tracklet is assigned to a compatible
   anchor -- one it never co-occurs with, apart from frames where it merely
   duplicates the anchor's detection -- by spatial continuity and appearance.
   Only duplicate detections and tracklets that fit no anchor are untracked;
   a real fragment is never dropped because some other, short false-positive
   identity happened to co-occur with it.

Every vector already exists when tracking runs in sleap-nn (offline flow), so
linking decides with full tracklets on both sides of every gap -- more evidence
than an online re-ID decision at track confirmation could use.

Conventions used throughout:

- **Identity**: one distinct ``sio.Track`` object among the tracked detections.
- **Tracklet**: a maximal run of an identity's detections with no gap longer
  than ``track_buffer`` steps.
- **Time** is counted in steps, exactly as the trackers count them: the
  calibration's resolved ``time_base`` and stride through
  :func:`~sleap_nn.tracking.mot.calibration.frame_steps` (frame-index deltas
  divided by ``frame_step`` in the ``"frames"`` time base, one step per
  processed frame in ``"steps"``).
- **Gated vector**: the appearance vector of a detection that is confident
  (score ``>= high_threshold``), mostly visible (at least half of its nodes) and
  isolated (padded-box IoU below 0.1 against every other detection of its
  frame). Only gated vectors enter galleries: crops of overlapping animals are
  contaminated.
- **Linking statistic** ``s_link(Q, G)``: the mean over query vectors ``q`` of
  the mean of the top-5 cosines of ``q`` against gallery ``G`` -- the statistic
  :func:`~sleap_nn.tracking.mot.calibration.calibrate` calibrates the linking
  rule on, for every query length (``link_thresh_by_len``): a decision on an
  ``n``-vector query uses the threshold and margin of ``n`` vectors.
- **Cannot-link**: two identities whose temporal extents overlap are never
  linked by the open-set rules (an identity is linked only to one that starts
  after it ends); identities that share a frame are different animals for
  certain, which is what makes them the competitors that appearance margins are
  measured against.
- **Duplicate**: two same-frame detections whose raw boxes overlap by at least
  ``DUPLICATE_IOMIN`` of the smaller one, or whose centroids are closer than
  ``DUPLICATE_CENTROID_BODY`` body sizes -- one animal detected twice.

Scaling: the open set keeps a sparse set of candidate pairs per identity (every
pair inside the spatial-only window plus the best appearance candidates), not
dense identity x identity matrices, and the closed set is linear in the number
of tracklets times ``N``.
"""

import heapq
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import attrs
import numpy as np
import sleap_io as sio
from loguru import logger
from scipy.optimize import linear_sum_assignment

from sleap_nn.tracking.mot.association import iou_matrix
from sleap_nn.tracking.mot.calibration import (
    DEFAULT_BODY_SIZE,
    LINK_GALLERY_LEN,
    LINK_QUERY_LEN,
    LINK_TOPK,
    RESOLVED_TIME_BASES,
    VideoCalibration,
    calibrate,
    frame_steps,
    link_rule_tables,
    resolve_time_base,
)
from sleap_nn.tracking.mot.detections import (
    FrameDetections,
    extract_detections,
    normalize_scores,
)

# Gated vectors in an identity's end gallery (last of the earlier identity in a
# gap link) and in the pre-event gallery for swap correction. `calibrate()` runs
# its linking trials with exactly this gallery length.
GALLERY_SIZE = LINK_GALLERY_LEN
# Gated vectors in a query: the first of the later identity in a gap link, and the
# first after an ambiguity event for swap correction (at most; a shorter query is
# decided with the calibrated rule of its own length).
LINK_QUERY_SIZE = LINK_QUERY_LEN
SWAP_QUERY_SIZE = LINK_QUERY_SIZE
# Top-k cosines averaged per query vector in the linking statistic: the
# calibration's own `k` (its linking trials compute exactly this statistic).
LINK_TOP_K = LINK_TOPK
# Isolation test of the gallery quality gate (padded-box IoU, strict <).
ISOLATION_IOU = 0.1
# Minimum fraction of visible nodes for a gated vector.
MIN_VISIBLE_FRACTION = 0.5
# Padded-box IoU at or above which two identities are in an ambiguity event.
AMBIGUITY_IOU = 0.1
# Ambiguity events over more identities than this are not swap-corrected.
MAX_EVENT_IDENTITIES = 4
# Duplicate detections (closed set): raw-box intersection over the smaller box
# at least this, or centroids closer than DUPLICATE_CENTROID_BODY body sizes.
DUPLICATE_IOMIN = 0.5
DUPLICATE_CENTROID_BODY = 0.3
# A tracklet still fits an anchor when at most this fraction of its detections
# share a frame with the anchor's without duplicating them (a false positive on
# one side); more, and they are two animals.
MAX_CONFLICT_FRACTION = 0.05
# Spatial gate: radius = body + GATE_STEP_MULT * s * f(gap), with f(gap) = gap up
# to GATE_LINEAR_STEPS steps and sqrt(GATE_LINEAR_STEPS * gap) beyond (a long
# absence is a random walk, not a straight line), and s = step_p95 floored at
# GATE_MIN_STEP_BODY body sizes per step (a calibration clip where the animals
# barely move must not shrink the gate to the body size).
GATE_STEP_MULT = 3.0
GATE_LINEAR_STEPS = 10
GATE_MIN_STEP_BODY = 0.05
# Spatial-only (no appearance) links are accepted up to this many track buffers.
SPATIAL_ONLY_BUFFERS = 4
# Appearance candidates kept per identity (each side) in the open set:
# clip(2 x the most identities tracked in one frame, MIN, MAX), by a cheap
# mean-vector proxy of the linking statistic. Pairs inside the spatial-only
# window are always kept (up to MAX_WINDOW_CANDIDATES per identity).
MIN_APPEARANCE_CANDIDATES = 8
MAX_APPEARANCE_CANDIDATES = 64
MAX_WINDOW_CANDIDATES = 256
# Per-decision notes kept in a LinkingResult (the counts are always exact).
MAX_DECISION_NOTES = 50

# Comparison slack for margins computed from float32 cosines.
_EPS = 1e-9
# Element budget of the blocked (rows x identities) appearance-proxy pass.
_BLOCK_ELEMS = 1 << 20


@attrs.define
class LinkingResult:
    """Outcome of :func:`link_tracklets` for one video.

    Attributes:
        n_identities_before: Identities (tracks) before linking.
        n_identities: Identities after linking.
        n_merges: Gap links applied (appearance or spatial).
        n_forced_merges: Closed-set assignments of a tracklet to an anchor.
        n_swaps: Swap corrections applied.
        n_untracked: Detections set to untracked by the closed-set rule.
        notes: Human-readable decisions worth logging.
    """

    n_identities_before: int = 0
    n_identities: int = 0
    n_merges: int = 0
    n_forced_merges: int = 0
    n_swaps: int = 0
    n_untracked: int = 0
    notes: List[str] = attrs.field(factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict."""
        return attrs.asdict(self)


def _s_link(queries: np.ndarray, gallery: np.ndarray, k: int = LINK_TOP_K) -> float:
    """Linking statistic: mean over queries of the mean top-``k`` gallery cosine.

    Args:
        queries: ``(nq, D)`` L2-normalized vectors.
        gallery: ``(ng, D)`` L2-normalized vectors.
        k: Cosines averaged per query (all of them when ``ng < k``).

    Returns:
        The statistic, or NaN when either side is empty.
    """
    if len(queries) == 0 or len(gallery) == 0:
        return float("nan")
    cos = (
        np.asarray(queries, dtype=np.float64) @ np.asarray(gallery, dtype=np.float64).T
    )
    kk = min(k, cos.shape[1])
    top = -np.partition(-cos, kk - 1, axis=1)[:, :kk]
    return float(top.mean())


def _step_positions(frame_idx: np.ndarray, calibration: VideoCalibration) -> np.ndarray:
    """Time (in steps) of each processed frame, starting at 0.

    The calibration's resolved ``time_base`` and stride, through the shared
    :func:`~sleap_nn.tracking.mot.calibration.frame_steps` -- the trackers count
    the same steps. A calibration without a resolved time base falls back to the
    ``"auto"`` rule on its ``regular_stride``.
    """
    time_base = calibration.time_base
    if time_base not in RESOLVED_TIME_BASES:
        time_base = resolve_time_base("auto", bool(calibration.regular_stride))
    steps = frame_steps(frame_idx, time_base, calibration.frame_step)
    return steps.astype(np.float64)


def _has_degenerate_pose(dets: FrameDetections) -> bool:
    """Whether any valid pose is a single node or narrower than 1 px (floored)."""
    if dets.keypoints is None or len(dets) == 0:
        return False
    wh = dets.raw_boxes[:, 2:] - dets.raw_boxes[:, :2]
    small = (wh <= 1.0 + 1e-9).any(axis=1)
    return bool((dets.valid & ((dets.n_visible == 1) | small)).any())


def _embedding_dim(dets: Sequence[FrameDetections]) -> Optional[int]:
    """Most common appearance-vector dimension over frames (first seen on ties)."""
    dims = Counter(int(d.embeddings.shape[1]) for d in dets if d.embeddings is not None)
    return dims.most_common(1)[0][0] if dims else None


def _ranges(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(owner, index)`` for the concatenation of ``range(lo[k], hi[k])``."""
    counts = np.maximum(np.asarray(hi) - np.asarray(lo), 0)
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(counts)), counts)
    if total == 0:
        return owner, np.zeros(0, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    index = np.arange(total) - np.repeat(starts, counts) + np.repeat(lo, counts)
    return owner, index.astype(np.int64)


@attrs.define(eq=False)
class _Event:
    """An ambiguity event: a maximal span of overlapping identities (private).

    Attributes:
        start: First processed-frame position.
        end: Last processed-frame position.
        labels: Identity labels involved (labels at event construction).
        dets: Detection indices involved in an overlap.
    """

    start: int
    end: int
    labels: set
    dets: list


@attrs.define(eq=False)
class _Fit:
    """How a tracklet fits one anchor in the closed set (private).

    Attributes:
        anchor: The anchor identity.
        kind: ``"ordered"`` (before or after the anchor), ``"insertion"``
            (inside one of its gaps) or ``"duplicate"`` (the tracklet only
            occupies frames where the anchor has a detection too).
        rest: The tracklet's detections that would join the anchor.
        shared_t: The tracklet's detections in frames where the anchor has one.
        shared_u: The anchor's detections there (aligned with ``shared_t``).
        n_conflict: Shared frames that are not duplicates (two objects).
        n_far: Conflicts more than a body size apart (two animals, rather than a
            false positive next to the animal).
        gated: Every junction with the anchor passes the spatial gate.
        tier: 0 appearance agrees, 1 unavailable / undecidable, 2 disagrees.
        cost: Worst junction's geometric cost (see :meth:`_Linker._junction_cost`).
        gap: Longest junction gap (steps).
        worst: ``(distance, radius)`` of the worst gate violation, if any.
    """

    anchor: int
    kind: str
    rest: np.ndarray
    shared_t: np.ndarray
    shared_u: np.ndarray
    n_conflict: int = 0
    n_far: int = 0
    gated: bool = True
    tier: int = 1
    cost: float = float("inf")
    gap: float = 0.0
    worst: Optional[Tuple[float, float]] = None


class _Linker:
    """Mutable state of one :func:`link_tracklets` run (private).

    Holds every TRACKED detection of the video in temporal order (detection index
    order is time order) and its current identity label (-1: untracked by the
    closed set), and per identity its member detections, liveness, temporal
    extent, endpoint centroids and gated-vector lists. The open set works on a
    sparse list of candidate pairs (rows: the earlier identity ``i``, columns:
    the later one ``j``) that a merge of ``j`` into ``i`` replaces for ``i`` and
    ``j`` only.
    """

    def __init__(
        self,
        frames: List[Tuple[int, list]],
        dets: List[FrameDetections],
        calibration: VideoCalibration,
        *,
        track_buffer: int,
        spatial_gate_frames: int,
        high_threshold: float,
        body_size: Optional[float] = None,
    ):
        """Collect the tracked detections and build identity state."""
        self.cal = calibration
        self.track_buffer = int(track_buffer)
        self.spatial_gate_frames = float(spatial_gate_frames)
        body = calibration.body_size if body_size is None else body_size
        self.body = float(body) if body is not None else float("nan")
        if not np.isfinite(self.body) or self.body <= 0:
            self.body = DEFAULT_BODY_SIZE
        p95 = calibration.step_p95
        p95 = float(p95) if p95 is not None else float("nan")
        # Per-step motion scale of the gate (NaN: no gate).
        self.step_scale = (
            max(p95, GATE_MIN_STEP_BODY * self.body) if np.isfinite(p95) else p95
        )
        self.thresh_n, self.margin_n = link_rule_tables(calibration, LINK_QUERY_SIZE)
        self.notes: List[str] = []
        self._n_decision_notes = 0

        self.frame_idx = np.array([f for f, _ in frames], dtype=np.int64)
        frame_steps_ = _step_positions(self.frame_idx, calibration)
        self.dim = _embedding_dim(dets)
        self._collect(dets, frame_steps_, high_threshold)
        self._build_identities()

    # ------------------------------------------------------------ collection
    def _collect(
        self, dets: List[FrameDetections], frame_steps_: np.ndarray, high: float
    ) -> None:
        """Gather tracked detections, gated vectors, overlaps and duplicates."""
        objs: List[Any] = []
        pos: List[int] = []
        centroids: List[np.ndarray] = []
        scores: List[float] = []
        gated: List[bool] = []
        vectors: List[Optional[np.ndarray]] = []
        labels: List[int] = []
        overlaps: List[Tuple[int, int]] = []
        dups: List[Tuple[int, int]] = []
        self.tracks: List[sio.Track] = []
        track_index: Dict[int, int] = {}
        dup_dist = DUPLICATE_CENTROID_BODY * self.body

        for p, d in enumerate(dets):
            tracked = [
                i
                for i, o in enumerate(d.sources)
                if getattr(o, "track", None) is not None
            ]
            if not tracked:
                continue
            iou = iou_matrix(d.boxes, d.boxes)
            np.fill_diagonal(iou, 0.0)
            isolated = (iou < ISOLATION_IOU).all(axis=1)
            visible = d.n_visible / max(int(d.n_nodes), 1) >= MIN_VISIBLE_FRACTION
            has_vec = d.has_embedding.copy()
            if d.embeddings is None or d.embeddings.shape[1] != self.dim:
                has_vec[:] = False
            ok = d.valid & (d.scores >= high) & visible & isolated & has_vec

            first = len(objs)
            for i in tracked:
                obj = d.sources[i]
                key = id(obj.track)
                if key not in track_index:
                    track_index[key] = len(self.tracks)
                    self.tracks.append(obj.track)
                objs.append(obj)
                pos.append(p)
                centroids.append(d.centroids[i])
                scores.append(float(d.scores[i]))
                gated.append(bool(ok[i]))
                vectors.append(d.embeddings[i] if ok[i] else None)
                labels.append(track_index[key])
            if len(tracked) < 2:
                continue

            t = np.asarray(tracked)
            # Overlaps between DIFFERENT identities (ambiguity events).
            sub = iou[np.ix_(t, t)]
            for a, b in zip(*np.nonzero(np.triu(sub >= AMBIGUITY_IOU, k=1))):
                if labels[first + a] != labels[first + b]:
                    overlaps.append((first + int(a), first + int(b)))
            # Duplicates (one animal detected twice), any identities.
            box = np.asarray(d.raw_boxes, dtype=np.float64)[t]
            lo = np.maximum(box[:, None, :2], box[None, :, :2])
            hi = np.minimum(box[:, None, 2:], box[None, :, 2:])
            inter = np.clip(hi - lo, 0.0, None).prod(axis=-1)
            area = np.clip(box[:, 2:] - box[:, :2], 0.0, None).prod(axis=-1)
            small = np.minimum(area[:, None], area[None, :])
            cen = np.asarray(d.centroids, dtype=np.float64)[t]
            # Single-node poses have no extent (a pseudo box at most): centroid
            # distance only.
            sized = (
                np.ones(len(t), dtype=bool)
                if d.keypoints is None
                else np.asarray(d.n_visible)[t] >= 2
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                iomin = np.where(small > 0, inter / small, 0.0)
                iomin = np.where(sized[:, None] & sized[None, :], iomin, 0.0)
                dist = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
                dup = (np.nan_to_num(iomin) >= DUPLICATE_IOMIN) | (dist < dup_dist)
            dup &= d.valid[t][:, None] & d.valid[t][None, :]
            for a, b in zip(*np.nonzero(np.triu(dup, k=1))):
                dups.append((first + int(a), first + int(b)))

        n = len(objs)
        self.objs = objs
        self.pos = np.asarray(pos, dtype=np.int64)
        self.step = frame_steps_[self.pos] if n else np.zeros(0, dtype=np.float64)
        self.centroids = (
            np.asarray(centroids, dtype=np.float64).reshape(n, 2)
            if n
            else np.zeros((0, 2))
        )
        self.valid_c = np.isfinite(self.centroids).all(axis=1)
        # Arena extent: diagonal of the centroids' bounding box (>= body).
        c = self.centroids[self.valid_c]
        self.arena = max(
            float(np.hypot(*np.ptp(c, axis=0))) if len(c) else 0.0, self.body
        )
        self.scores = np.asarray(scores, dtype=np.float64)
        self.gated = np.asarray(gated, dtype=bool)
        self.label = np.asarray(labels, dtype=np.int64)
        self.vectors = None
        if self.gated.any():
            self.vectors = np.zeros((n, self.dim), dtype=np.float32)
            for i in np.flatnonzero(self.gated):
                self.vectors[i] = vectors[i]
        self.overlaps = np.asarray(overlaps, dtype=np.int64).reshape(-1, 2)
        self.in_event = np.zeros(n, dtype=bool)
        self.in_event[self.overlaps.reshape(-1)] = True
        dup_pairs = np.asarray(dups, dtype=np.int64).reshape(-1, 2)
        self.dup_pairs = dup_pairs
        # Sorted keys (low * n + high) of the duplicate pairs, for lookups.
        self._dup_keys = np.sort(dup_pairs.min(axis=1) * max(n, 1) + dup_pairs.max(1))

    def _build_identities(self) -> None:
        """Per-identity member lists (time-ordered detection indices)."""
        m = len(self.tracks)
        order = np.argsort(self.label, kind="stable")
        bounds = np.searchsorted(self.label[order], np.arange(m + 1))
        self.members = [order[bounds[k] : bounds[k + 1]] for k in range(m)]
        self.alive = np.ones(m, dtype=bool)
        names = [str(t.name) if t.name is not None else "" for t in self.tracks]
        self.names = names
        # Rank of each name (ties broken by first appearance) for vectorized sorts.
        rank = sorted(range(m), key=lambda k: (names[k], k))
        self.name_rank = np.empty(m, dtype=np.int64)
        self.name_rank[rank] = np.arange(m)
        # Identity owning each detection before the closed set (Track naming).
        self.owner = self.label.copy()
        self._mcache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def note(self, text: str, decision: bool = True) -> None:
        """Record a note (per-decision notes are capped at MAX_DECISION_NOTES)."""
        if decision:
            self._n_decision_notes += 1
            if self._n_decision_notes > MAX_DECISION_NOTES:
                return
        self.notes.append(text)

    def _is_dup(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Whether detections ``a[k]`` and ``b[k]`` are duplicates of each other."""
        if len(self._dup_keys) == 0 or len(a) == 0:
            return np.zeros(len(a), dtype=bool)
        n = max(len(self.objs), 1)
        key = np.minimum(a, b) * n + np.maximum(a, b)
        at = np.searchsorted(self._dup_keys, key)
        at = np.minimum(at, len(self._dup_keys) - 1)
        return self._dup_keys[at] == key

    # ------------------------------------------------------- swap correction
    def ambiguity_events(self) -> List[_Event]:
        """Maximal spans of processed frames where identities' boxes overlap.

        Overlapping identities in one frame form a component; a component joins
        any event that shares an identity and was active in the same or the
        previous processed frame, so a chain of crossings is one event. Linear in
        the number of overlaps (they are collected in frame order).
        """
        if len(self.overlaps) == 0:
            return []
        frame_pos = self.pos[self.overlaps[:, 0]]
        order = np.argsort(frame_pos, kind="stable")
        frame_pos = frame_pos[order]
        edges_all = self.overlaps[order].tolist()
        labels = self.label.tolist()
        cuts = np.flatnonzero(np.diff(frame_pos)) + 1
        starts = np.concatenate([[0], cuts]).tolist()
        stops = np.concatenate([cuts, [len(frame_pos)]]).tolist()

        events: Dict[int, _Event] = {}  # live events (insertion ordered)
        active: List[_Event] = []
        for s, e in zip(starts, stops):
            p = int(frame_pos[s])
            edges = edges_all[s:e]
            # Components of this frame's overlap graph over identity labels.
            parent: Dict[int, int] = {}

            def find(x: int) -> int:
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in edges:
                ra, rb = find(labels[a]), find(labels[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            comps: Dict[int, _Event] = {}
            seen: Dict[int, set] = {}
            for a, b in edges:
                for d in (a, b):
                    lab = labels[d]
                    root = find(lab)
                    ev = comps.get(root)
                    if ev is None:
                        ev = comps[root] = _Event(p, p, set(), [])
                        seen[root] = set()
                    ev.labels.add(lab)
                    if d not in seen[root]:
                        seen[root].add(d)
                        ev.dets.append(d)

            active = [ev for ev in active if ev.end == p - 1]
            current: List[_Event] = []
            for root in sorted(comps):
                ev = comps[root]
                # Absorb every live event sharing an identity (to a fixpoint: an
                # absorbed event can bring identities shared with another). The
                # older event's detections come first.
                while True:
                    other = next(
                        (x for x in active + current if x.labels & ev.labels), None
                    )
                    if other is None:
                        break
                    other.start = min(other.start, ev.start)
                    other.end = max(other.end, ev.end)
                    other.labels |= ev.labels
                    other.dets.extend(ev.dets)
                    events.pop(id(ev), None)
                    if any(x is other for x in active):
                        active = [x for x in active if x is not other]
                    else:
                        current = [x for x in current if x is not other]
                    ev = other
                current.append(ev)
                events[id(ev)] = ev
            active = current
        out = list(events.values())
        out.sort(key=lambda ev: (ev.start, ev.end, min(ev.dets)))
        return out

    def _member_cache(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(positions, gated indices, in-event indices)`` of ``k``'s members."""
        cached = self._mcache.get(k)
        if cached is None:
            m = self.members[k]
            cached = (
                self.pos[m],
                np.flatnonzero(self.gated[m]),
                np.flatnonzero(self.in_event[m]),
            )
            self._mcache[k] = cached
        return cached

    def _pre_gallery(self, k: int, start: int) -> np.ndarray:
        """Last GALLERY_SIZE gated vectors of identity ``k`` before ``start``."""
        mpos, gidx, _ = self._member_cache(k)
        cut = int(np.searchsorted(mpos, start, side="left"))
        n_gated = int(np.searchsorted(gidx, cut, side="left"))
        sel = self.members[k][gidx[max(0, n_gated - GALLERY_SIZE) : n_gated]]
        return self.vectors[sel]

    def _post_query(self, k: int, end: int) -> np.ndarray:
        """First SWAP_QUERY_SIZE gated vectors of ``k`` after ``end``.

        Collection stops at ``k``'s next detection involved in an overlap (its
        next ambiguity event).
        """
        mpos, gidx, busy = self._member_cache(k)
        cut = int(np.searchsorted(mpos, end, side="right"))
        nb = int(np.searchsorted(busy, cut, side="left"))
        stop = int(busy[nb]) if nb < len(busy) else len(mpos)
        lo = int(np.searchsorted(gidx, cut, side="left"))
        hi = int(np.searchsorted(gidx, stop, side="left"))
        sel = self.members[k][gidx[lo : min(hi, lo + SWAP_QUERY_SIZE)]]
        return self.vectors[sel]

    def _rule(self, n_query: int) -> Tuple[float, float]:
        """Calibrated ``(threshold, margin)`` for an ``n_query``-vector query."""
        q = len(self.thresh_n)
        n = min(max(int(n_query), 1), q) - 1
        return float(self.thresh_n[n]), float(self.margin_n[n])

    def correct_swaps(self) -> int:
        """Swap correction over every ambiguity event, in time order.

        Each moved identity is decided with the calibrated rule of its post-event
        query's length; an event with a query too short to decide is skipped.

        Returns:
            Number of events whose identities were permuted.
        """
        n_swaps = 0
        n_large = 0
        for ev in self.ambiguity_events():
            ks = sorted({int(self.label[d]) for d in ev.dets})
            if len(ks) > MAX_EVENT_IDENTITIES:
                n_large += 1
                continue
            pre = {k: self._pre_gallery(k, ev.start) for k in ks}
            post = {k: self._post_query(k, ev.end) for k in ks}
            ks = [k for k in ks if len(pre[k]) and len(post[k])]
            if len(ks) < 2:
                continue
            # s[l, k]: identity l's post-event vectors vs identity k's pre-event
            # gallery. The best permutation relabels post-event l as perm[l].
            s = np.array([[_s_link(post[lab], pre[k]) for k in ks] for lab in ks])
            rows, cols = linear_sum_assignment(-s)
            moved = [r for r, c in zip(rows, cols) if r != c]
            if not moved:
                continue
            rules = {r: self._rule(len(post[ks[r]])) for r in moved}
            if not all(np.isfinite(t) for t, _ in rules.values()):
                continue
            gain = float(s[rows, cols].sum() - np.trace(s))
            pair_ok = all(
                s[r, cols[r]] >= rules[r][0]
                and s[r, cols[r]] - s[r, r] >= rules[r][1] - _EPS
                for r in moved
            )
            need = sum(rules[r][1] for r in moved)
            if gain < need - _EPS or not pair_ok:
                continue
            perm = {ks[r]: ks[cols[r]] for r in moved}
            self._relabel_after(perm, ev.end)
            n_swaps += 1
            pairs = ", ".join(
                f"{self.names[a]}->{self.names[b]}" for a, b in perm.items()
            )
            self.note(
                f"swap corrected after frames {self.frame_idx[ev.start]}-"
                f"{self.frame_idx[ev.end]}: {pairs} (gain {gain:.3f})"
            )
        if n_large:
            self.note(
                f"{n_large} ambiguity event(s) over more than {MAX_EVENT_IDENTITIES} "
                "identities were not swap-corrected",
                decision=False,
            )
        return n_swaps

    def _relabel_after(self, perm: Dict[int, int], end: int) -> None:
        """Give identity ``perm[l]`` the detections of ``l`` after position ``end``."""
        heads, tails = {}, {}
        for lab in perm:
            m = self.members[lab]
            cut = int(np.searchsorted(self._member_cache(lab)[0], end, side="right"))
            heads[lab], tails[lab] = m[:cut], m[cut:]
        for lab, dst in perm.items():
            self.members[dst] = np.concatenate([heads[dst], tails[lab]])
            self.label[tails[lab]] = dst
        for lab in perm:
            self._mcache.pop(lab, None)

    # ---------------------------------------------------------- gap linking
    def init_linking(self, use_appearance: bool) -> None:
        """Per-identity summaries and the sparse candidate pairs of the open set."""
        m = len(self.tracks)
        self.use_app = bool(use_appearance) and self.vectors is not None
        self.label = self.label.copy()
        self.owner = self.label.copy()
        self.first_step = np.full(m, np.inf)
        self.last_step = np.full(m, -np.inf)
        self.start_c = np.full((m, 2), np.nan)
        self.end_c = np.full((m, 2), np.nan)
        self.n_dets = np.zeros(m, dtype=np.int64)
        self.score_sum = np.zeros(m, dtype=np.float64)
        self.gated_members: List[np.ndarray] = []
        empty = np.zeros((0, self.dim or 0), dtype=np.float32)
        self.heads: List[np.ndarray] = [empty] * m
        self.tails: List[np.ndarray] = [empty] * m
        self.n_head = np.zeros(m, dtype=np.int64)
        self.n_tail = np.zeros(m, dtype=np.int64)
        for k in range(m):
            mem = self.members[k]
            self.gated_members.append(mem[self.gated[mem]])
            if len(mem) == 0:
                self.alive[k] = False
                continue
            self._set_extent(k)
            self.n_dets[k] = len(mem)
            self.score_sum[k] = self.scores[mem].sum()
            if self.use_app:
                self._set_galleries(k)

        # Sparse candidate pairs (capacity-doubling arrays).
        self._np = 0
        self._cap = 0
        self.pi = np.zeros(0, dtype=np.int64)
        self.pj = np.zeros(0, dtype=np.int64)
        self.pgap = np.zeros(0)
        self.pdist = np.zeros(0)
        self.pscore = np.zeros(0)
        self.pwin = np.zeros(0, dtype=bool)
        self.papp = np.zeros(0, dtype=bool)
        self.pspat = np.zeros(0, dtype=bool)
        self.plive = np.zeros(0, dtype=bool)
        self.pverdict = np.zeros(0, dtype=np.int8)
        self._n_dead = 0
        self.row_pairs: List[List[int]] = [[] for _ in range(m)]
        self.col_pairs: List[List[int]] = [[] for _ in range(m)]
        self._init_pairs()

    def _set_extent(self, k: int) -> None:
        """First/last step and first/last valid centroid of identity ``k``."""
        mem = self.members[k]
        self.first_step[k] = self.step[mem[0]]
        self.last_step[k] = self.step[mem[-1]]
        ok = mem[self.valid_c[mem]]
        if len(ok):
            self.start_c[k] = self.centroids[ok[0]]
            self.end_c[k] = self.centroids[ok[-1]]

    def _set_galleries(self, k: int) -> None:
        """Head (first LINK_QUERY_SIZE) and tail (last GALLERY_SIZE) gated vectors."""
        g = self.gated_members[k]
        head, tail = g[:LINK_QUERY_SIZE], g[-GALLERY_SIZE:]
        self.heads[k] = self.vectors[head]
        self.tails[k] = self.vectors[tail]
        self.n_head[k], self.n_tail[k] = len(head), len(tail)

    def _radius(self, gap: np.ndarray) -> np.ndarray:
        """Spatial-gate radius (px) for gaps in steps (NaN when uncalibrated)."""
        gap = np.asarray(gap, dtype=np.float64)
        eff = np.where(
            gap <= GATE_LINEAR_STEPS,
            gap,
            np.sqrt(GATE_LINEAR_STEPS * np.maximum(gap, 0.0)),
        )
        return self.body + GATE_STEP_MULT * self.step_scale * eff

    def _gated(self, gap: np.ndarray) -> np.ndarray:
        """Whether a junction over ``gap`` steps is spatially gated at all."""
        return (np.asarray(gap) <= self.spatial_gate_frames) & np.isfinite(
            self.step_scale
        )

    def _gate_ok(self, gap: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Spatial feasibility of junctions (``dist`` NaN fails a gated junction)."""
        gap = np.asarray(gap, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            near = np.asarray(dist) <= self._radius(np.maximum(gap, 0.0))
        return ~self._gated(gap) | near

    def _junction_cost(self, gap: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Geometric cost of junctions: ``4.5 z^2 + 2 ln(r)``, ``z = dist / r``.

        ``r`` is the gate radius at that gap (the gate sits near 3 sigma). Up to
        constants this is the negative log-likelihood ratio of a random-walk
        continuation against an unrelated animal landing there: the endpoint
        distance normalized by the gate, plus the price of the gate's growth with
        the gap (a long gap leaves room for a coincidence). The radius is capped
        at the arena (the extent of every centroid): beyond it the continuation
        is anywhere, as likely as a coincidence, so a long gap is not priced
        above an unrelated landing. Without a calibrated motion scale:
        ``4.5 (dist / body)^2 + ln(gap)``. Non-finite -> ``inf``.
        """
        gap = np.maximum(np.asarray(gap, dtype=np.float64), 1.0)
        dist = np.asarray(dist, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.isfinite(self.step_scale):
                radius = np.minimum(self._radius(gap), self.arena)
                c = 4.5 * (dist / radius) ** 2 + 2.0 * np.log(radius)
            else:
                c = 4.5 * (dist / self.body) ** 2 + np.log(gap)
        return np.where(np.isfinite(c), c, np.inf)

    # ------------------------------------------------------ sparse candidates
    def _max_concurrent(self) -> int:
        """Most identities tracked in one processed frame."""
        live = self.label >= 0
        if not live.any():
            return 1
        return int(np.bincount(self.pos[live]).max())

    def _init_pairs(self) -> None:
        """Candidate pairs: the spatial-only window plus the appearance best."""
        alive = np.flatnonzero(self.alive)
        if len(alive) < 2:
            return
        order = alive[np.argsort(self.first_step[alive], kind="stable")]
        firsts = self.first_step[order]
        window = SPATIAL_ONLY_BUFFERS * self.track_buffer
        lo = np.searchsorted(firsts, self.last_step[alive], side="right")
        hi = np.searchsorted(firsts, self.last_step[alive] + window, side="right")
        hi = np.minimum(hi, lo + MAX_WINDOW_CANDIDATES)
        owner, index = _ranges(lo, hi)
        ii = [alive[owner]]
        jj = [order[index]]
        if self.use_app:
            ai, aj = self._appearance_candidates()
            ii.append(ai)
            jj.append(aj)
        ii, jj = np.concatenate(ii), np.concatenate(jj)
        m = len(self.tracks)
        keys = np.unique(ii * m + jj)
        self._add_pairs(keys // m, keys % m)

    def _appearance_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Best ordered, spatially feasible pairs by a mean-vector proxy.

        For every identity, its ``K`` best successors (by the cosine between its
        mean end-gallery vector and their mean head vector) and its ``K`` best
        predecessors, ``K = clip(2 x max concurrent identities, MIN, MAX)``:
        the true continuation of a fragment is among the best-matching ones,
        and so is its strongest competitor. Computed in row blocks.
        """
        none = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        rows = np.flatnonzero(self.alive & (self.n_tail > 0))
        cols = np.flatnonzero(self.alive & (self.n_head > 0))
        if len(rows) == 0 or len(cols) == 0:
            return none

        def mean_unit(vs: List[np.ndarray]) -> np.ndarray:
            mu = np.stack([v.mean(axis=0) for v in vs]).astype(np.float32)
            norm = np.linalg.norm(mu, axis=1, keepdims=True)
            return mu / np.maximum(norm, 1e-12)

        tm = mean_unit([self.tails[k] for k in rows])
        hm = mean_unit([self.heads[k] for k in cols])
        k_app = int(
            np.clip(
                2 * self._max_concurrent(),
                MIN_APPEARANCE_CANDIDATES,
                MAX_APPEARANCE_CANDIDATES,
            )
        )
        mc = len(cols)
        kr = min(k_app, mc)
        col_val = np.full((k_app, mc), -np.inf, dtype=np.float32)
        col_idx = np.full((k_app, mc), -1, dtype=np.int64)
        out_i: List[np.ndarray] = []
        out_j: List[np.ndarray] = []
        block = max(1, _BLOCK_ELEMS // max(mc, 1))
        for b0 in range(0, len(rows), block):
            r = rows[b0 : b0 + block]
            sim = tm[b0 : b0 + block] @ hm.T  # (b, mc)
            gap = self.first_step[cols][None, :] - self.last_step[r][:, None]
            dist = np.linalg.norm(
                self.start_c[cols][None, :, :] - self.end_c[r][:, None, :], axis=-1
            )
            ok = (gap > 0) & self._gate_ok(gap, dist)
            sim = np.where(ok, sim, -np.inf).astype(np.float32)
            # Each row's best successors.
            top = np.argpartition(-sim, kr - 1, axis=1)[:, :kr]
            val = np.take_along_axis(sim, top, axis=1)
            keep = np.isfinite(val)
            out_i.append(np.repeat(r, kr).reshape(-1, kr)[keep])
            out_j.append(cols[top[keep]])
            # Each column's best predecessors so far.
            if len(r) > k_app:
                ctop = np.argpartition(-sim, k_app - 1, axis=0)[:k_app]
            else:
                ctop = np.broadcast_to(np.arange(len(r))[:, None], (len(r), mc))
            cval = np.take_along_axis(sim, ctop, axis=0)
            all_val = np.concatenate([col_val, cval], axis=0)
            all_idx = np.concatenate([col_idx, r[ctop]], axis=0)
            best = np.argpartition(-all_val, k_app - 1, axis=0)[:k_app]
            col_val = np.take_along_axis(all_val, best, axis=0)
            col_idx = np.take_along_axis(all_idx, best, axis=0)
        keep = np.isfinite(col_val) & (col_idx >= 0)
        out_i.append(col_idx[keep])
        out_j.append(np.broadcast_to(cols[None, :], col_val.shape)[keep])
        return np.concatenate(out_i), np.concatenate(out_j)

    def _score_pairs(self, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        """Linking statistic of pairs: ``j``'s head against ``i``'s end gallery.

        NaN where either side has no gated vector (or appearance is off).
        Batched per earlier identity.
        """
        out = np.full(len(ii), np.nan)
        if not self.use_app or len(ii) == 0:
            return out
        ok = np.flatnonzero((self.n_tail[ii] > 0) & (self.n_head[jj] > 0))
        if len(ok) == 0:
            return out
        ok = ok[np.argsort(ii[ok], kind="stable")]
        cuts = np.flatnonzero(np.diff(ii[ok])) + 1
        for grp in np.split(ok, cuts):
            tail = self.tails[int(ii[grp[0]])].astype(np.float64)
            heads = [self.heads[int(j)] for j in jj[grp]]
            nq = np.array([len(h) for h in heads])
            cos = np.concatenate(heads).astype(np.float64) @ tail.T
            nt = cos.shape[1]
            kk = min(LINK_TOP_K, nt)
            top = np.partition(cos, nt - kk, axis=1)[:, nt - kk :]
            per_q = top.mean(axis=1)
            starts = np.cumsum(nq) - nq
            out[grp] = np.add.reduceat(per_q, starts) / nq
        return out

    def _add_pairs(
        self,
        ii: np.ndarray,
        jj: np.ndarray,
        known: Optional[Dict[int, float]] = None,
        known_side: int = 1,
    ) -> None:
        """Add the ordered, spatially feasible pairs among ``(ii, jj)``.

        Args:
            ii: Earlier identities.
            jj: Later identities.
            known: Linking statistics still valid for some pairs, keyed by the
                pair's ``jj`` (``known_side=1``) or ``ii`` (``0``) identity;
                the others are computed.
            known_side: Which side keys ``known``.
        """
        ii = np.asarray(ii, dtype=np.int64)
        jj = np.asarray(jj, dtype=np.int64)
        if len(ii) == 0:
            return
        gap = self.first_step[jj] - self.last_step[ii]
        dist = np.linalg.norm(self.start_c[jj] - self.end_c[ii], axis=1)
        ok = (
            self.alive[ii]
            & self.alive[jj]
            & (ii != jj)
            & (gap > 0)
            & self._gate_ok(gap, dist)
        )
        ii, jj, gap, dist = ii[ok], jj[ok], gap[ok], dist[ok]
        n_new = len(ii)
        if n_new == 0:
            return
        if self._n_dead > max(4096, self._np // 2):
            self._compact()
        need = self._np + n_new
        if need > self._cap:
            cap = max(need, 2 * self._cap, 64)
            for name in self._PAIR_FIELDS:
                arr = getattr(self, name)
                grown = np.zeros(cap, dtype=arr.dtype)
                grown[: self._np] = arr[: self._np]
                setattr(self, name, grown)
            verdict = np.full(cap, -1, dtype=np.int8)
            verdict[: self._np] = self.pverdict[: self._np]
            self.pverdict = verdict
            self._cap = cap
        sl = slice(self._np, need)
        self.pi[sl], self.pj[sl] = ii, jj
        self.pgap[sl], self.pdist[sl] = gap, dist
        if known:
            keys = (jj if known_side == 1 else ii).tolist()
            have = np.array([k in known for k in keys], dtype=bool)
            score = np.full(n_new, np.nan)
            score[have] = [known[k] for k, h in zip(keys, have) if h]
            score[~have] = self._score_pairs(ii[~have], jj[~have])
            self.pscore[sl] = score
        else:
            self.pscore[sl] = self._score_pairs(ii, jj)
        self.pwin[sl] = gap <= SPATIAL_ONLY_BUFFERS * self.track_buffer
        # Static per pair (a merge replaces the pairs of the identities it
        # changes): an appearance candidate clears the threshold of j's head
        # length; a pair without appearance may link spatially in the window.
        has_app = np.isfinite(self.pscore[sl])
        n_q = np.clip(self.n_head[jj], 1, len(self.thresh_n))
        with np.errstate(invalid="ignore"):
            self.papp[sl] = has_app & (self.pscore[sl] >= self.thresh_n[n_q - 1])
        self.pspat[sl] = self.pwin[sl] & ~has_app
        self.plive[sl] = True
        self.pverdict[sl] = -1
        for p, (a, b) in enumerate(zip(ii.tolist(), jj.tolist()), start=self._np):
            self.row_pairs[a].append(p)
            self.col_pairs[b].append(p)
        self._np = need

    _PAIR_FIELDS = (
        "pi",
        "pj",
        "pgap",
        "pdist",
        "pscore",
        "pwin",
        "papp",
        "pspat",
        "plive",
    )

    def _compact(self) -> None:
        """Drop dead pairs from the pair arrays (and re-index the pair lists)."""
        keep = np.flatnonzero(self.plive[: self._np])
        for name in self._PAIR_FIELDS + ("pverdict",):
            arr = getattr(self, name)
            setattr(self, name, arr[keep].copy())
        self._np = self._cap = len(keep)
        self._n_dead = 0
        m = len(self.tracks)
        self.row_pairs = [[] for _ in range(m)]
        self.col_pairs = [[] for _ in range(m)]
        for p, (a, b) in enumerate(zip(self.pi.tolist(), self.pj.tolist())):
            self.row_pairs[a].append(p)
            self.col_pairs[b].append(p)

    def _live(self, lists: List[List[int]], k: int) -> np.ndarray:
        """Live pair indices of ``lists[k]`` (dead entries are pruned)."""
        idx = np.asarray(lists[k], dtype=np.int64)
        if len(idx):
            idx = idx[self.plive[idx]]
            lists[k] = idx.tolist()
        return idx

    def _cooccur(self, ks: np.ndarray, k: int) -> np.ndarray:
        """Whether identities ``ks`` overlap identity ``k`` in time (and live)."""
        ks = np.asarray(ks, dtype=np.int64)
        return self.alive[ks] & ~(
            (self.last_step[ks] < self.first_step[k])
            | (self.last_step[k] < self.first_step[ks])
        )

    def best_gap_link(self) -> Optional[Tuple[int, int, str, float]]:
        """The best gap link the open-set rules accept, or ``None``.

        Appearance (pair has galleries on both sides): accept when ``s_link`` is
        at least the calibrated threshold for the later identity's head length
        and its margin over every competing candidate -- for the same later
        identity, those that cannot-link with the earlier one; for the same
        earlier identity, those that cannot-link with the later one -- is at
        least that length's margin (no competitor: threshold only). Spatial only:
        accept when the gap is at most SPATIAL_ONLY_BUFFERS track buffers and the
        pair is unique on the frontier of that window: ``j`` is the only window
        successor of ``i`` starting before ``i``'s nearest window successor ends
        (a later one -- e.g. the animal's own next fragment -- can follow
        through it), ``i`` the only window predecessor of ``j`` ending after
        ``j``'s latest window predecessor starts, and no host (an identity with
        either inside one of its gaps) competes. Accepted links are ranked by
        (appearance before spatial, score desc, gap asc, start of j asc, names
        asc), and a link that would skip over another candidate of the same
        identity is deferred (see the stages below). Cost per call: linear in
        the live candidate pairs (a few per identity), not in identities
        squared.

        Returns:
            ``(i, j, kind, score)`` with ``kind`` ``"appearance"`` or
            ``"spatial"``, or ``None`` when nothing is accepted.
        """
        n, m = self._np, len(self.tracks)
        live = self.plive[:n]
        app = np.flatnonzero(live & self.papp[:n])

        # Spatial-only: unique on the window's frontier. A window successor of i
        # that starts after i's nearest window successor ends is no competitor
        # (it can follow through that one, e.g. the animal's own later
        # fragment); likewise for predecessors of j.
        w = np.flatnonzero(live & self.pspat[:n])
        wi, wj = self.pi[w], self.pj[w]
        end_first_w = np.full(m, np.inf)
        np.minimum.at(end_first_w, wi, self.last_step[wj])
        start_last_w = np.full(m, -np.inf)
        np.maximum.at(start_last_w, wj, self.first_step[wi])
        row_f = self.first_step[wj] <= end_first_w[wi]
        col_f = self.last_step[wi] >= start_last_w[wj]
        row_n = np.bincount(wi[row_f], minlength=m)
        col_n = np.bincount(wj[col_f], minlength=m)
        spatial = w[row_f & col_f & (row_n[wi] == 1) & (col_n[wj] == 1)]

        cp = np.concatenate([app, spatial])
        if len(cp) == 0:
            return None
        ci, cj = self.pi[cp], self.pj[cp]
        c_spatial = np.concatenate(
            [np.zeros(len(app), dtype=bool), np.ones(len(spatial), dtype=bool)]
        )
        c_score = np.nan_to_num(self.pscore[cp], nan=0.0)

        def accepted(k: int) -> bool:
            p = int(cp[k])
            if self.pverdict[p] < 0:
                i, j = int(ci[k]), int(cj[k])
                ok = (
                    self._spatially_unique(i, j)
                    if c_spatial[k]
                    else self._margin_ok(i, j, float(self.pscore[p]))
                )
                self.pverdict[p] = int(ok)
            return bool(self.pverdict[p])

        def ranked(sel: np.ndarray) -> np.ndarray:
            k = np.flatnonzero(sel)
            keys = (
                self.name_rank[cj[k]],
                self.name_rank[ci[k]],
                self.first_step[cj[k]],
                self.pgap[cp[k]],
                -c_score[k],
                c_spatial[k],
            )
            return k[np.lexsort(keys)]

        # Stage A: the "frontier" -- candidates that skip no other candidate of
        # the same identity (no i -> k with k ending before j starts, no k -> j
        # with k starting after i ends). Merging a skip link first would nest
        # the skipped identity inside the merged identity's gap, where it could
        # never be linked. The smallest-gap candidate is always on the frontier.
        end_first = np.full(m, np.inf)
        np.minimum.at(end_first, ci, self.last_step[cj])
        start_last = np.full(m, -np.inf)
        np.maximum.at(start_last, cj, self.first_step[ci])
        frontier = (self.first_step[cj] <= end_first[ci]) & (
            self.last_step[ci] >= start_last[cj]
        )
        for k in ranked(frontier):
            if accepted(int(k)):
                return self._link(int(ci[k]), int(cj[k]), bool(c_spatial[k]), cp[k])

        # Stage B (rare): every frontier candidate failed. Accept a link that
        # skips only REJECTED candidates (none of the identities it skips over
        # can be linked to either end).
        index = {int(p): k for k, p in enumerate(cp.tolist())}
        for k in ranked(~frontier):
            if not accepted(int(k)):
                continue
            i, j = int(ci[k]), int(cj[k])
            skipped = [
                index.get(int(p))
                for p in self._live(self.row_pairs, i)
                if self.last_step[self.pj[p]] < self.first_step[j]
            ] + [
                index.get(int(p))
                for p in self._live(self.col_pairs, j)
                if self.first_step[self.pi[p]] > self.last_step[i]
            ]
            if any(s is not None and accepted(s) for s in skipped):
                continue
            return self._link(i, j, bool(c_spatial[k]), cp[k])
        return None

    def _link(
        self, i: int, j: int, spatial: bool, p: int
    ) -> Tuple[int, int, str, float]:
        """``best_gap_link`` return value for the chosen pair."""
        if spatial:
            return i, j, "spatial", 0.0
        return i, j, "appearance", float(self.pscore[p])

    def _margin_ok(self, i: int, j: int, score: float) -> bool:
        """Appearance margin of ``i -> j`` over its proven-different competitors.

        Competitors: candidates for the same ``j`` that cannot-link with ``i``,
        and candidates of the same ``i`` that cannot-link with ``j``. With none,
        only the threshold applies (already checked by the caller). The margin is
        the calibrated one for ``j``'s head length.
        """
        others = [-np.inf]
        col = self._live(self.col_pairs, j)
        col = col[(self.pi[col] != i) & np.isfinite(self.pscore[col])]
        col = col[self._cooccur(self.pi[col], i)]
        if len(col):
            others.append(float(self.pscore[col].max()))
        row = self._live(self.row_pairs, i)
        row = row[(self.pj[row] != j) & np.isfinite(self.pscore[row])]
        row = row[self._cooccur(self.pj[row], j)]
        if len(row):
            others.append(float(self.pscore[row].max()))
        # Identities that could take j's start (or i's end) inside one of their
        # gaps compete too -- e.g. an animal whose own fragments were already
        # joined around j.
        cl_i = self._cooccur(np.arange(len(self.tracks)), i)
        cl_j = self._cooccur(np.arange(len(self.tracks)), j)
        for h, gap1, _, d1, _ in self._hosts(j):
            if h != i and cl_i[h] and self._gate_ok(gap1, d1):
                others.append(self._junction_score(h, j, "start"))
        for h, _, gap2, _, d2 in self._hosts(i):
            if h != j and cl_j[h] and self._gate_ok(gap2, d2):
                others.append(self._junction_score(h, i, "end"))
        best_other = np.nanmax(np.asarray(others))
        if not np.isfinite(best_other):
            return True
        _, margin = self._rule(self.n_head[j])
        return bool(score - best_other >= margin - _EPS)

    def _spatially_unique(self, i: int, j: int) -> bool:
        """No host could take ``j``'s start or ``i``'s end within the window.

        Complements the frontier uniqueness of the spatial-only rule: an
        identity with ``j`` (or ``i``) inside one of its gaps is a feasible
        predecessor (successor) too.
        """
        window = SPATIAL_ONLY_BUFFERS * self.track_buffer
        for h, gap1, _, d1, _ in self._hosts(j):
            if h != i and gap1 <= window and self._gate_ok(gap1, d1):
                return False
        for h, _, gap2, _, d2 in self._hosts(i):
            if h != j and gap2 <= window and self._gate_ok(gap2, d2):
                return False
        return True

    def _gap_fits(
        self, h: int, cand: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Which ``cand`` fit entirely inside one gap of host ``h``, with junctions.

        Args:
            h: Host identity.
            cand: Candidate guest identities.

        Returns:
            ``(guests, gap1, gap2, d1, d2)``: the guests that never co-occur with
            ``h`` and lie between two of its detections, the steps from ``h``'s
            detection before each guest to the guest's start (``gap1``) and from
            the guest's end to ``h``'s next detection (``gap2``), and the
            distances between the matching centroids (``h``'s nearest valid
            detections; NaN if none).
        """
        none = np.zeros(0, dtype=np.int64)
        empty = (none, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        mem = self.members[h]
        if not self.alive[h] or len(mem) < 2 or len(cand) == 0:
            return empty
        cand = np.asarray(cand, dtype=np.int64)
        inside = (
            self.alive[cand]
            & (self.first_step[cand] > self.first_step[h])
            & (self.last_step[cand] < self.last_step[h])
        )
        cand = cand[inside]
        st = self.step[mem]
        lo = np.searchsorted(st, self.first_step[cand], side="left")
        hi = np.searchsorted(st, self.last_step[cand], side="right")
        fits = lo == hi
        cand, lo = cand[fits], lo[fits]
        if len(cand) == 0:
            return empty
        gap1 = self.first_step[cand] - st[lo - 1]
        gap2 = st[lo] - self.last_step[cand]
        vm = mem[self.valid_c[mem]]
        if len(vm) == 0:
            nan = np.full(len(cand), np.nan)
            return cand, gap1, gap2, nan, nan.copy()
        vs = self.step[vm]
        p = np.searchsorted(vs, self.first_step[cand], side="left") - 1
        q = np.searchsorted(vs, self.last_step[cand], side="right")
        prev_c = np.where(
            (p >= 0)[:, None], self.centroids[vm[np.clip(p, 0, len(vm) - 1)]], np.nan
        )
        next_c = np.where(
            (q < len(vm))[:, None],
            self.centroids[vm[np.clip(q, 0, len(vm) - 1)]],
            np.nan,
        )
        d1 = np.linalg.norm(self.start_c[cand] - prev_c, axis=1)
        d2 = np.linalg.norm(next_c - self.end_c[cand], axis=1)
        return cand, gap1, gap2, d1, d2

    def _guests(self, h: int) -> np.ndarray:
        """Live identities fitting entirely inside one gap of ``h``."""
        return self._gap_fits(h, np.flatnonzero(self.alive))[0]

    def _hosts(self, k: int) -> List[Tuple[int, float, float, float, float]]:
        """Live identities ``h`` with ``k`` inside one of their gaps.

        Returns:
            ``(h, gap1, gap2, d1, d2)`` per host (see :meth:`_gap_fits`).
        """
        hosts = np.flatnonzero(
            self.alive
            & (self.first_step < self.first_step[k])
            & (self.last_step > self.last_step[k])
        )
        out = []
        for h in hosts:
            fit = self._gap_fits(int(h), np.array([k]))
            if len(fit[0]):
                out.append((int(h),) + tuple(float(v[0]) for v in fit[1:]))
        return out

    def _junction_score(self, h: int, k: int, side: str) -> float:
        """Linking statistic of ``k`` against host ``h`` across one junction.

        ``side="start"``: ``k``'s first gated vectors against ``h``'s gated
        vectors right before ``k``; ``"end"``: ``h``'s gated vectors right after
        ``k`` against ``k``'s last ones.
        """
        g = self.gated_members[h]
        gs = self.step[g]
        gk = self.gated_members[k]
        if side == "start":
            before = g[: np.searchsorted(gs, self.first_step[k], side="left")]
            return _s_link(
                self.vectors[gk[:LINK_QUERY_SIZE]],
                self.vectors[before[-GALLERY_SIZE:]],
            )
        after = g[np.searchsorted(gs, self.last_step[k], side="right") :]
        return _s_link(
            self.vectors[after[:LINK_QUERY_SIZE]], self.vectors[gk[-GALLERY_SIZE:]]
        )

    def _neighborhood(self, k: int, mark: np.ndarray) -> None:
        """Mark ``k``, its candidate predecessors/successors and its guests."""
        mark[k] = True
        mark[self.pi[self._live(self.col_pairs, k)]] = True
        mark[self.pj[self._live(self.row_pairs, k)]] = True
        mark[self._guests(k)] = True

    def merge(self, i: int, j: int) -> None:
        """Merge identity ``j`` into the EARLIER identity ``i`` (keeps ``i``'s Track).

        ``j`` starts after ``i`` ends (an open-set gap link). Replaces every
        candidate pair of ``i`` and ``j`` with the merged identity's, and resets
        the cached verdicts of every pair whose competitors may have changed: a
        pair's verdict depends only on the candidates of its two identities,
        those identities' co-occurring competitors and hosts, so every pair with
        an endpoint in the neighborhood (candidates and guests) of ``i`` or ``j``
        -- before or after the merge -- is reset.
        """
        m = len(self.tracks)
        dirty = np.zeros(m, dtype=bool)
        self._neighborhood(i, dirty)
        self._neighborhood(j, dirty)
        rows = [self.pj[self._live(self.row_pairs, k)] for k in (i, j)]
        cols = [self.pi[self._live(self.col_pairs, k)] for k in (i, j)]
        # Statistics that survive the merge: (a, i) keeps i's head, (j, b) has
        # the merged identity's tail when j's tail was full (or j has i's).
        old_head = self.gated_members[i][:LINK_QUERY_SIZE]
        old_tail = self.gated_members[j][-GALLERY_SIZE:]
        col_i = self._live(self.col_pairs, i)
        row_j = self._live(self.row_pairs, j)
        head_scores = dict(zip(self.pi[col_i].tolist(), self.pscore[col_i].tolist()))
        tail_scores = dict(zip(self.pj[row_j].tolist(), self.pscore[row_j].tolist()))
        for k in (i, j):
            for lst in (self.row_pairs[k], self.col_pairs[k]):
                idx = np.asarray(lst, dtype=np.int64)
                self._n_dead += int(np.count_nonzero(self.plive[idx]))
                self.plive[idx] = False
            self.row_pairs[k], self.col_pairs[k] = [], []

        self.alive[j] = False
        self.members[i] = np.concatenate([self.members[i], self.members[j]])
        self.members[j] = self.members[j][:0]
        self.label[self.members[i]] = i
        self.owner[self.members[i]] = i
        self.gated_members[i] = np.concatenate(
            [self.gated_members[i], self.gated_members[j]]
        )
        self._set_extent(i)
        self.n_dets[i] += self.n_dets[j]
        self.score_sum[i] += self.score_sum[j]
        if self.use_app:
            self._set_galleries(i)
        self._mcache.pop(i, None)
        self._mcache.pop(j, None)

        # The merged identity keeps j's successors and i's predecessors.
        succ = np.unique(np.concatenate(rows))
        pred = np.unique(np.concatenate(cols))
        succ = succ[(succ != i) & (succ != j)]
        pred = pred[(pred != i) & (pred != j)]
        g = self.gated_members[i]
        same_tail = np.array_equal(old_tail, g[-GALLERY_SIZE:])
        same_head = np.array_equal(old_head, g[:LINK_QUERY_SIZE])
        self._add_pairs(
            np.full(len(succ), i), succ, tail_scores if same_tail else None, 1
        )
        self._add_pairs(
            pred, np.full(len(pred), i), head_scores if same_head else None, 0
        )

        self._neighborhood(i, dirty)
        for k in np.flatnonzero(dirty).tolist():
            self.pverdict[self._live(self.row_pairs, k)] = -1
            self.pverdict[self._live(self.col_pairs, k)] = -1

    def n_spatial_ambiguous(self) -> int:
        """Live spatial-only window pairs left unlinked (ambiguous)."""
        live = np.flatnonzero(self.plive[: self._np])
        return int(np.count_nonzero(self.pwin[live] & ~np.isfinite(self.pscore[live])))

    # ------------------------------------------------------------ closed set
    def _select_anchors(self, n_max: int) -> List[int]:
        """The ``n_max`` identities that best represent the animals.

        Per processed frame, the identities present (a detection duplicating a
        better-supported identity's in that frame does not count) are ranked by
        support (detection count); the anchors are the top ``n_max`` of the
        frame where the most identities -- up to ``n_max`` -- are present
        together, with the largest total support (earliest on ties). Present
        together, they are certainly different animals. Fewer than ``n_max``
        present anywhere: the rest by support.
        """
        alive = np.flatnonzero(self.alive)
        support = self.n_dets.astype(np.float64)
        tracked = np.flatnonzero(self.label >= 0)
        lab = self.label[tracked]
        present = np.ones(len(tracked), dtype=bool)
        if len(self.dup_pairs):
            a, b = self.dup_pairs[:, 0], self.dup_pairs[:, 1]
            la, lb = self.label[a], self.label[b]
            ok = (la >= 0) & (lb >= 0) & (la != lb)
            a, b, la, lb = a[ok], b[ok], la[ok], lb[ok]
            # The less supported side of a duplicate pair is dominated there.
            a_wins = (support[la] > support[lb]) | (
                (support[la] == support[lb]) & (self.name_rank[la] < self.name_rank[lb])
            )
            loser = np.where(a_wins, b, a)
            where = np.searchsorted(tracked, loser)
            present[where] = False
        t, lab = tracked[present], lab[present]
        p = self.pos[t]
        order = np.lexsort((self.name_rank[lab], -support[lab], p))
        p, lab = p[order], lab[order]
        starts = np.searchsorted(p, p, side="left")
        take = (np.arange(len(p)) - starts) < n_max
        n_pos = int(self.pos.max()) + 1 if len(self.pos) else 1
        counts = np.bincount(p[take], minlength=n_pos)
        sums = np.bincount(p[take], weights=support[lab[take]], minlength=n_pos)
        anchors: List[int] = []
        if counts.any():
            best = int(np.lexsort((np.arange(n_pos), -sums, -counts))[0])
            for k in lab[take & (p == best)].tolist():
                if k not in anchors:
                    anchors.append(int(k))
        if len(anchors) < n_max:
            rest = [int(k) for k in alive if int(k) not in anchors]
            rest.sort(key=lambda k: (-support[k], self.name_rank[k]))
            anchors += rest[: n_max - len(anchors)]
        return anchors

    def _tracklets(self, k: int) -> List[np.ndarray]:
        """``k``'s tracklets: runs of its detections split at gaps > track_buffer."""
        mem = self.members[k]
        if len(mem) == 0:
            return []
        cuts = np.flatnonzero(np.diff(self.step[mem]) > max(self.track_buffer, 0)) + 1
        return np.split(mem, cuts)

    def _fit(self, t: np.ndarray, a: int) -> Optional[_Fit]:
        """How tracklet ``t`` fits anchor ``a`` (``None``: they co-occur).

        In a frame both hold a detection, ``t``'s either duplicates ``a``'s (one
        animal detected twice) or conflicts with it (two objects). ``t`` and
        ``a`` are incompatible when more than ``MAX_CONFLICT_FRACTION`` of
        ``t``'s detections conflict (they are different animals); a few
        conflicts are a false positive on one side, resolved like duplicates.
        The detections of the other frames (``rest``) would join ``a``; every
        junction between them and ``a``'s detections (consecutive detections of
        the merged timeline from different sides) is checked against the
        spatial gate and costed.
        """
        amem = self.members[a]
        ap, ast, ga, gs = self._anchor[a]
        tp = self.pos[t]
        none = np.zeros(0, dtype=np.int64)
        shared_t = shared_u = none
        shared = np.zeros(len(t), dtype=bool)
        n_conflict = n_far = 0
        if len(ap):
            at = np.minimum(np.searchsorted(ap, tp), len(ap) - 1)
            shared = ap[at] == tp
            shared_t, shared_u = t[shared], amem[at[shared]]
            if len(shared_t):
                conflict = ~self._is_dup(shared_t, shared_u)
                n_conflict = int(np.count_nonzero(conflict))
                if n_conflict > MAX_CONFLICT_FRACTION * len(t):
                    return None
                gap = np.linalg.norm(
                    self.centroids[shared_t[conflict]]
                    - self.centroids[shared_u[conflict]],
                    axis=1,
                )
                n_far = int(np.count_nonzero(~(gap < self.body)))
        rest = t[~shared]
        if len(rest) == 0:
            return _Fit(
                a,
                "duplicate",
                rest,
                shared_t,
                shared_u,
                n_conflict,
                n_far,
                cost=-np.inf,
            )
        fit = _Fit(a, "ordered", rest, shared_t, shared_u, n_conflict, n_far)

        # Junctions with a's detections around (and inside) rest's span.
        lo = int(np.searchsorted(ast, self.step[rest[0]], side="left")) - 1
        hi = int(np.searchsorted(ast, self.step[rest[-1]], side="right"))
        if lo >= 0 and hi < len(amem):
            fit.kind = "insertion"
        local = amem[max(lo, 0) : min(hi + 1, len(amem))]
        seq = np.concatenate([local, rest])
        src = np.concatenate([np.zeros(len(local), bool), np.ones(len(rest), bool)])
        keep = self.valid_c[seq]
        seq, src = seq[keep], src[keep]
        order = np.argsort(self.step[seq], kind="stable")
        seq, src = seq[order], src[order]
        tr = np.flatnonzero(src[1:] != src[:-1])
        if len(tr):
            x, y = seq[tr], seq[tr + 1]
            gaps = self.step[y] - self.step[x]
            dists = np.linalg.norm(self.centroids[y] - self.centroids[x], axis=1)
            ok = self._gate_ok(gaps, dists)
            cost = self._junction_cost(gaps, dists)
            fit.gated = bool(ok.all())
            fit.cost = float(cost.max())
            fit.gap = float(gaps.max())
            if not fit.gated:
                bad = np.flatnonzero(~ok)
                w = bad[np.argmax(dists[bad] - self._radius(gaps[bad]))]
                fit.worst = (float(dists[w]), float(self._radius(gaps[w])))

        if self.use_app:
            rg = rest[self.gated[rest]]
            scores, n_q = [], []
            if len(rg):
                before = ga[: np.searchsorted(gs, self.step[rest[0]], side="left")]
                if len(before):
                    q = rg[:LINK_QUERY_SIZE]
                    scores.append(
                        _s_link(self.vectors[q], self.vectors[before[-GALLERY_SIZE:]])
                    )
                    n_q.append(len(q))
                after = ga[np.searchsorted(gs, self.step[rest[-1]], side="right") :]
                if len(after):
                    q = after[:LINK_QUERY_SIZE]
                    scores.append(
                        _s_link(self.vectors[q], self.vectors[rg[-GALLERY_SIZE:]])
                    )
                    n_q.append(len(q))
            if scores:
                thresh, _ = self._rule(min(n_q))
                if np.isfinite(thresh):
                    fit.tier = 0 if float(np.mean(scores)) >= thresh else 2
        return fit

    def _fit_key(self, f: _Fit) -> tuple:
        """Preference order of fits (smaller is better).

        Appearance first, then whether any conflicting frame holds two
        detections more than a body size apart (two animals at once), then the
        geometric cost. A conflict next to the animal is as likely a false
        positive (``MAX_CONFLICT_FRACTION`` bounds those) and does not rank a
        fit down.
        """
        return (f.tier, f.n_far > 0, f.cost, f.gap, self.name_rank[f.anchor])

    def _choose(self, fits: List[_Fit], n_dets: int) -> Optional[_Fit]:
        """The anchor a tracklet joins (``None``: untrack it).

        A duplicate of an anchor first; then the best gated fit by (appearance
        tier, far conflicts, geometric cost, gap, name); else -- past the spatial
        gate, as a last resort before losing the detections -- the best fit
        whose appearance agrees, or for a substantial tracklet (more than
        ``track_buffer`` detections; short far-away false positives stay
        untracked) the best fit appearance does not contradict.
        """
        if not fits:
            return None
        dups = [f for f in fits if f.kind == "duplicate"]
        if dups:
            return min(dups, key=self._fit_key)

        def best(cands: List[_Fit]) -> Optional[_Fit]:
            return min(cands, key=self._fit_key) if cands else None

        gated = best([f for f in fits if f.gated])
        if gated is not None:
            return gated
        agree = best([f for f in fits if f.tier == 0])
        if agree is not None:
            return agree
        if n_dets > self.track_buffer:
            return best([f for f in fits if f.tier == 1])
        return None

    def _decision(
        self, fits: List[Optional[_Fit]], n_dets: int
    ) -> Tuple[tuple, Optional[_Fit]]:
        """A tracklet's next decision and its priority (smaller goes first).

        Untracking (no anchor fits) goes first -- it is final, anchors only gain
        detections. Then forced assignments (a single viable anchor), then the
        rest by confidence: decisive when the runner-up is of a worse class
        (duplicate, gated, appearance tier, far conflicts), else by the geometric-cost
        margin over it. Longer tracklets first within a priority class.
        """
        viable = [f for f in fits if f is not None]
        choice = self._choose(viable, n_dets)
        if choice is None:
            return (0, 0.0, -n_dets), None
        runner = self._choose([f for f in viable if f is not choice], n_dets)
        if runner is None:
            return (1, 0.0, -n_dets), choice

        def cls(f: _Fit) -> tuple:
            return (f.kind != "duplicate", not f.gated, f.tier, f.n_far > 0)

        if cls(choice) != cls(runner):
            return (2, -np.inf, -n_dets), choice
        return (2, -(runner.cost - choice.cost), -n_dets), choice

    def _join(self, fit: _Fit) -> Tuple[int, int]:
        """Apply ``fit``: ``rest`` joins the anchor; shared frames are resolved.

        In a frame both hold a detection (a duplicate, or a rare conflict) the
        anchor keeps the higher-scoring detection (its own on ties); the other
        is untracked.

        Returns:
            ``(untracked, swapped)``: detections untracked, and shared frames
            where the tracklet's detection replaced the anchor's.
        """
        a = fit.anchor
        swap = self.scores[fit.shared_t] > self.scores[fit.shared_u] + _EPS
        out_t, in_t = fit.shared_t[~swap], fit.shared_t[swap]
        out_u = fit.shared_u[swap]
        self.label[out_t] = -1
        self.label[out_u] = -1
        joined = np.concatenate([fit.rest, in_t])
        self.label[joined] = a
        mem = self.members[a]
        if len(out_u):
            mem = mem[~np.isin(mem, out_u)]
        self.members[a] = np.sort(np.concatenate([mem, joined]), kind="stable")
        self.gated_members[a] = self.members[a][self.gated[self.members[a]]]
        self._cache_anchor(a)
        self.n_dets[a] = len(self.members[a])
        return len(out_t) + len(out_u), len(in_t)

    def _cache_anchor(self, a: int) -> None:
        """Positions / steps of anchor ``a``'s detections and gated detections."""
        mem, ga = self.members[a], self.gated_members[a]
        self._anchor[a] = (self.pos[mem], self.step[mem], ga, self.step[ga])

    def close_set(self, n_max: int) -> Tuple[int, int]:
        """Reduce the live identities to at most ``n_max`` (anchors).

        The anchors are chosen by :meth:`_select_anchors`. Every tracklet of the
        other identities then joins the anchor it fits best (:meth:`_fit`,
        :meth:`_choose`), most constrained first (:meth:`_decision`): a tracklet
        that co-occurs with all anchors but one has no choice, and joining it
        extends that anchor in time, which constrains the tracklets around it in
        turn -- so ambiguous geometry is decided late, across short junctions.
        Shared frames are resolved per frame, and only a tracklet that fits no
        anchor -- it co-occurs with every one, or only a far, short or
        appearance-contradicted junction is left -- is untracked.

        A join re-fits the tracklets within ``track_buffer`` steps of the joined
        one against that anchor at once (their conflicts and nearest junctions
        changed); any other stale fit is refreshed when its tracklet comes up,
        and the tracklet waits again if its priority got worse.

        Returns:
            ``(n_assigned_tracklets, n_untracked_detections)``.
        """
        n_assigned = n_untracked = 0
        alive = np.flatnonzero(self.alive)
        if len(alive) <= n_max:
            return 0, 0
        self.owner = self.label.copy()
        anchors = self._select_anchors(n_max)
        self._anchor: Dict[int, tuple] = {}
        for a in anchors:
            self._cache_anchor(a)
        self.note(
            f"closed set: anchors {', '.join(self.names[a] for a in anchors)}",
            decision=False,
        )
        pieces: List[np.ndarray] = []
        what: List[str] = []
        for k in alive:
            k = int(k)
            if k in self._anchor:
                continue
            tracklets = self._tracklets(k)
            for t in tracklets:
                pieces.append(t)
                label = self.names[k]
                if len(tracklets) > 1:
                    label += (
                        f" (frames {self.frame_idx[self.pos[t[0]]]}-"
                        f"{self.frame_idx[self.pos[t[-1]]]})"
                    )
                what.append(label)
        n_pieces = len(pieces)
        p_first = np.array([self.step[t[0]] for t in pieces], dtype=np.float64)
        p_last = np.array([self.step[t[-1]] for t in pieces], dtype=np.float64)
        version = np.zeros(len(anchors), dtype=np.int64)  # per anchor
        fits = [[self._fit(t, a) for a in anchors] for t in pieces]
        seen = np.zeros((n_pieces, len(anchors)), dtype=np.int64)  # versions used
        stamp = np.zeros(n_pieces, dtype=np.int64)
        done = np.zeros(n_pieces, dtype=bool)

        def push(u: int) -> None:
            stamp[u] += 1
            prio, _ = self._decision(fits[u], len(pieces[u]))
            heapq.heappush(heap, (prio, p_first[u], u, int(stamp[u])))

        heap: List[tuple] = []
        for u in range(n_pieces):
            push(u)
        look = ("appearance agrees", "no appearance", "appearance disagrees")
        while heap:
            prio, _, u, st = heapq.heappop(heap)
            if done[u] or st != stamp[u]:
                continue
            stale = np.flatnonzero(seen[u] != version)
            if len(stale):
                for ai in stale.tolist():
                    fits[u][ai] = self._fit(pieces[u], anchors[ai])
                    seen[u, ai] = version[ai]
                if self._decision(fits[u], len(pieces[u]))[0] > prio:
                    push(u)
                    continue
            done[u] = True
            t = pieces[u]
            _, fit = self._decision(fits[u], len(t))
            if fit is None:
                self.label[t] = -1
                n_untracked += len(t)
                viable = [f for f in fits[u] if f is not None]
                if not viable:
                    reason = "co-occurs with every anchor"
                else:
                    near = min(viable, key=self._fit_key)
                    reason = f"no fitting anchor: {self.names[near.anchor]}"
                    if near.worst is not None:
                        d, r = near.worst
                        reason += f" is past the spatial gate ({d:.0f} px > {r:.0f} px)"
                    else:
                        reason += f" ({look[near.tier]})"
                self.note(
                    f"closed set: untracked {what[u]} ({len(t)} detections, {reason})"
                )
                continue

            a = fit.anchor
            n_out, n_swapped = self._join(fit)
            n_untracked += n_out
            name_a = self.names[a]
            if fit.kind == "duplicate":
                self.note(
                    f"closed set: {what[u]} duplicates {name_a} ({n_out} duplicate "
                    "detection(s) untracked"
                    + (
                        f"; {n_swapped} higher-scoring one(s) kept on {name_a})"
                        if n_swapped
                        else ")"
                    )
                )
            else:
                n_assigned += 1
                extra = "" if fit.gated else ", past the spatial gate"
                if len(fit.shared_t):
                    extra += f", {len(fit.shared_t)} shared frame(s) resolved"
                self.note(
                    f"closed set: forced {what[u]} -> {name_a} ({fit.kind}, "
                    f"{look[fit.tier]}, geometric cost {fit.cost:.2f}, gap "
                    f"{fit.gap:g} steps{extra})"
                )
            # Re-fit now the tracklets around the join (conflicts, short
            # junctions); farther ones are refreshed when they come up.
            ai = anchors.index(a)
            version[ai] += 1
            reach = max(self.track_buffer, 1)
            near = np.flatnonzero(
                ~done & (p_last >= p_first[u] - reach) & (p_first <= p_last[u] + reach)
            )
            for v in near.tolist():
                fits[v][ai] = self._fit(pieces[v], a)
                seen[v, ai] = version[ai]
                push(v)
        for k in alive:
            if int(k) not in self._anchor:
                self.alive[k] = False
                self.members[k] = self.members[k][:0]
        return n_assigned, n_untracked

    def final_tracks(self) -> Dict[int, int]:
        """Track (index into ``tracks``) of every live identity.

        Each identity takes the Track its earliest detection carried before the
        closed set (the earlier identity's, as for gap links), unless an
        earlier-starting identity already took it; then its own.
        """
        out: Dict[int, int] = {}
        used = set()
        alive = [int(k) for k in np.flatnonzero(self.alive) if len(self.members[k])]
        alive.sort(key=lambda k: int(self.members[k].min()))
        for k in alive:
            o = int(self.owner[int(self.members[k].min())])
            if o in used or o < 0:
                o = k
            out[k] = o
            used.add(o)
        return out

    def n_alive(self) -> int:
        """Live identities."""
        return int(self.alive.sum())


def link_tracklets(
    frames: Sequence[Tuple[int, Sequence[Any]]],
    *,
    carrier: str = "pose",
    calibration: Optional[VideoCalibration] = None,
    max_tracks: Optional[int] = None,
    track_buffer: int = 30,
    swap_correction: bool = True,
    spatial_gate_frames: int = 300,
    box_padding: float = 0.1,
    high_threshold: float = 0.5,
    score_mode: str = "auto",
    score_normalizer: Optional[float] = None,
    time_base: str = "auto",
    body_size: Optional[float] = None,
    track_objects: Optional[Dict[int, sio.Track]] = None,
) -> LinkingResult:
    """Link a tracked video's identities across gaps and crossings, in place.

    Steps, in order (see the module docstring for the definitions):

    1. **Swap correction** (when ``swap_correction`` and the calibration reports
       ``appearance_for_linking``): for each ambiguity event over at most
       ``MAX_EVENT_IDENTITIES`` identities, the post-event vectors of each
       identity are scored against every identity's pre-event gallery; the best
       permutation is applied to the moved identities' detections after the event
       when its total gain is at least the calibrated margin per moved identity
       and every moved pair clears the calibrated threshold and beats its
       unswapped pair by the margin (the rule of each query's own length).
    2. **Gap linking**, greedily best-first (appearance before spatial, score
       desc, gap asc, start asc, name asc) until nothing is accepted, between an
       identity that ends and one that starts later within the spatial gate
       ``body + 3 * s * (gap if gap <= 10 else sqrt(10 * gap))`` with ``s`` the
       calibrated ``step_p95`` floored at 0.05 body per step (no gate beyond
       ``spatial_gate_frames`` steps or when ``step_p95`` is unknown). Pairs with
       gated vectors on both sides (and calibrated appearance) are decided by
       appearance (the threshold for the later identity's head length, then the
       margin over the competing candidates that provably are other animals);
       others are linked only when the gap is at most 4 track buffers and the
       pair is unique on that window's frontier (a candidate that starts after
       the nearest one ends can follow through it, so it is no competitor). A
       link that would skip over another candidate of the same identity
       waits for it (so that fragment is not stranded inside the merged
       identity's gap).
    3. **Closed set** (``max_tracks``): while more identities remain, the
       ``max_tracks`` best-supported identities present together become
       anchors; every other tracklet, most constrained first (a single viable
       anchor before a choice; decisive choices before close calls), joins the
       anchor it never co-occurs with (frames where it only duplicates the
       anchor's detection are resolved: the higher-scoring detection stays),
       ranked by appearance agreement, then a gap-aware geometric cost of its
       junctions (gated first; past the gate only when appearance agrees, or for
       a tracklet longer than ``track_buffer`` that appearance does not
       contradict). A tracklet that fits no anchor is untracked.

    Args:
        frames: ``(frame_idx, objects)`` per frame in temporal order: the tracked
            detections (``.track`` set or ``None``) of ONE video.
        carrier: ``"pose"`` or ``"mask"``.
        calibration: The video's calibration (computes one from ``frames`` if
            ``None``, with ``time_base`` and ``body_size``). Appearance is used
            only where it reports ``appearance_for_linking``. Its resolved
            ``time_base`` and stride count the steps.
        max_tracks: Known number of animals: closed-set linking.
        track_buffer: Gap (steps) below which a track's detections form one
            tracklet; also scales the spatial-only rule.
        swap_correction: Correct swaps at crossings (appearance only).
        spatial_gate_frames: Gaps longer than this (steps) are not spatially gated.
        box_padding: Box padding used for overlap tests.
        high_threshold: Minimum score for a vector to enter a gallery.
        score_mode: Detection score mode (see ``MOTParams.score_mode``).
        score_normalizer: The tracker's per-video score normalizer
            (``MOTParams.normalize_scores``): scores are divided by it and
            clipped to ``[0, 1]`` before the ``high_threshold`` gate, so the
            gate sees the scores the tracker saw. ``None``: raw scores.
        time_base: ``MOTParams.time_base``, used when ``calibration`` is
            ``None`` (a given calibration carries its resolved time base).
        body_size: A known body size (px, ``MOTParams.body_size``): pseudo-box
            size, duplicate and gate scale (default: the calibration's).
        track_objects: The tracker's ``{public_id: sio.Track}``; updated to drop
            merged-away / untracked identities when given.

    Returns:
        A :class:`LinkingResult`. Merged identities keep the EARLIER identity's
        ``sio.Track``. Never raises on degenerate input.
    """
    result = LinkingResult()
    frames = [(int(f), list(objs)) for f, objs in frames]
    if not any(
        getattr(o, "track", None) is not None for _, objs in frames for o in objs
    ):
        result.notes.append("no tracked detections: nothing to link")
        return result

    if body_size is not None and not (np.isfinite(body_size) and body_size > 0):
        body_size = None

    def extract(pseudo: Optional[float]) -> List[FrameDetections]:
        return [
            extract_detections(
                objs,
                f,
                carrier=carrier,
                box_padding=box_padding,
                pseudo_box_size=pseudo,
                score_mode=score_mode,
            )
            for f, objs in frames
        ]

    if calibration is None:
        # Raw geometry and scores, as the trackers' `prepare` calibrates them.
        dets = extract(None)
        calibration = calibrate(
            dets,
            track_buffer=track_buffer,
            time_base=time_base,
            body_size=body_size,
        )
        if carrier == "pose" and any(_has_degenerate_pose(d) for d in dets):
            # Degenerate poses (single node) get the calibrated body-size box, as
            # in the trackers, so overlap tests see a realistic extent.
            dets = extract(float(calibration.body_size))
    else:
        dets = extract(
            float(calibration.body_size) if body_size is None else float(body_size)
        )
    for d in dets:
        normalize_scores(d, score_normalizer)

    linker = _Linker(
        frames,
        dets,
        calibration,
        track_buffer=track_buffer,
        spatial_gate_frames=spatial_gate_frames,
        high_threshold=high_threshold,
        body_size=body_size,
    )
    n_before = len(linker.tracks)
    result.n_identities_before = n_before

    n_tracklets = 0
    for mem in linker.members:
        if len(mem):
            n_tracklets += 1 + int(
                (np.diff(linker.step[mem]) > max(int(track_buffer), 0)).sum()
            )
    linker.note(
        f"{n_before} identities in {n_tracklets} tracklets; appearance calibration "
        f"{calibration.status}, used for linking: {calibration.appearance_for_linking}",
        decision=False,
    )
    if time_base != "auto" and time_base != calibration.time_base:
        linker.note(
            f"time_base={time_base!r} differs from the calibration's "
            f"{calibration.time_base!r}: steps are counted in the calibration's",
            decision=False,
        )
    appearance = (
        bool(calibration.appearance_for_linking)
        and bool(np.isfinite(linker.thresh_n).any())
        and linker.vectors is not None
    )

    # 1. Swap correction at crossings.
    if swap_correction and appearance:
        result.n_swaps = linker.correct_swaps()
    elif swap_correction:
        linker.note(
            "swap correction skipped: appearance is not calibrated for linking",
            decision=False,
        )

    # 2. Open-set gap linking.
    linker.init_linking(appearance)
    while True:
        best = linker.best_gap_link()
        if best is None:
            break
        i, j, kind, score = best
        gap = linker.first_step[j] - linker.last_step[i]
        detail = f"score {score:.3f}" if kind == "appearance" else "unique"
        linker.note(
            f"linked {linker.names[j]} -> {linker.names[i]} ({kind}, {detail}, "
            f"gap {gap:g} steps)"
        )
        linker.merge(i, j)
        result.n_merges += 1
    n_ambiguous = linker.n_spatial_ambiguous()
    if n_ambiguous:
        # Kept with the summary notes (decision notes stay last).
        linker.notes.insert(
            1,
            f"{n_ambiguous} spatial-only junction(s) left unlinked: more than one "
            "feasible candidate within the window (no usable appearance)",
        )

    # 3. Closed set.
    if max_tracks is not None and int(max_tracks) < 1:
        linker.note(f"max_tracks={max_tracks} ignored (must be >= 1)", decision=False)
    elif max_tracks is not None and linker.n_alive() > int(max_tracks):
        result.n_forced_merges, result.n_untracked = linker.close_set(int(max_tracks))

    # Write back: each live identity's detections take its final Track.
    final = linker.final_tracks()
    for d, obj in enumerate(linker.objs):
        k = int(linker.label[d])
        if k < 0 or k not in final:
            obj.track = None
            obj.tracking_score = 0.0
            continue
        track = linker.tracks[final[k]]
        if obj.track is not track:
            obj.track = track

    kept = {id(linker.tracks[t]) for t in final.values()}
    removed = {id(t) for t in linker.tracks} - kept
    if track_objects is not None and removed:
        for key in [k for k, t in track_objects.items() if id(t) in removed]:
            del track_objects[key]

    result.n_identities = len(final)
    if linker._n_decision_notes > MAX_DECISION_NOTES:
        linker.notes.append(
            f"... {linker._n_decision_notes - MAX_DECISION_NOTES} more decision(s) "
            "not listed"
        )
    result.notes.extend(linker.notes)

    logger.info(
        f"Identity linking: {n_before} -> {result.n_identities} identities "
        f"({result.n_merges} gap links, {result.n_forced_merges} forced, "
        f"{result.n_swaps} swaps corrected, {result.n_untracked} detections "
        "untracked)."
    )
    return result
