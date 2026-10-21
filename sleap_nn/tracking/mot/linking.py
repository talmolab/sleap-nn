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
   match is spatially unique within a short gap.
3. **Enforces a closed set** when the number of animals is known: keeps merging
   the best feasible pairs (including an identity that fits inside a gap of
   another) until at most ``max_tracks`` identities remain; identities that
   co-occur with all others (duplicates / false positives) are untracked,
   duplicates first.

Every vector already exists when tracking runs in sleap-nn (offline flow), so
linking decides with full tracklets on both sides of every gap -- more evidence
than an online re-ID decision at track confirmation could use.

Conventions used throughout:

- **Identity**: one distinct ``sio.Track`` object among the tracked detections.
- **Tracklet**: a maximal run of an identity's detections with no gap longer
  than ``track_buffer`` steps.
- **Time** is counted in steps: frame-index deltas divided by the calibrated
  ``frame_step`` when the processed frames are regularly strided, else one step
  per processed frame (the trackers' time base).
- **Gated vector**: the appearance vector of a detection that is confident
  (score ``>= high_threshold``), mostly visible (at least half of its nodes) and
  isolated (padded-box IoU below 0.1 against every other detection of its
  frame). Only gated vectors enter galleries: crops of overlapping animals are
  contaminated.
- **Linking statistic** ``s_link(Q, G)``: the mean over query vectors ``q`` of
  the mean of the top-5 cosines of ``q`` against gallery ``G`` -- the statistic
  :func:`~sleap_nn.tracking.mot.calibration.calibrate` calibrates
  ``link_thresh`` / ``link_margin`` on.
- **Cannot-link**: two identities whose temporal extents overlap are never
  linked by the open-set rules (an identity is linked only to one that starts
  after it ends); identities that share a frame are different animals for
  certain, which is what makes them the competitors that appearance margins are
  measured against.
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
    LINK_QUERY_LEN,
    VideoCalibration,
    calibrate,
)
from sleap_nn.tracking.mot.detections import FrameDetections, extract_detections

# Gated vectors in an identity's end gallery (last of the earlier identity in a
# gap link) and in the pre-event gallery for swap correction.
GALLERY_SIZE = 30
# Gated vectors in a query: the first of the later identity in a gap link, and the
# first after an ambiguity event for swap correction. `calibrate()` is run with
# the same query length, so the calibrated thresholds describe this statistic.
LINK_QUERY_SIZE = LINK_QUERY_LEN
SWAP_QUERY_SIZE = LINK_QUERY_SIZE
# Top-k cosines averaged per query vector in the linking statistic.
LINK_TOP_K = 5
# Isolation test of the gallery quality gate (padded-box IoU, strict <).
ISOLATION_IOU = 0.1
# Minimum fraction of visible nodes for a gated vector.
MIN_VISIBLE_FRACTION = 0.5
# Padded-box IoU at or above which two identities are in an ambiguity event.
AMBIGUITY_IOU = 0.1
# Ambiguity events over more identities than this are not swap-corrected.
MAX_EVENT_IDENTITIES = 4
# Padded-box IoU at or above which two co-occurring detections are duplicates.
DUPLICATE_IOU = 0.5
# An identity is a duplicate (untracked first in a closed set) when at least this
# fraction of its frames overlaps another identity at DUPLICATE_IOU.
DUPLICATE_FRACTION = 0.5
# Spatial gate: radius = body + GATE_STEP_MULT * step_p95 * f(gap), with f(gap) =
# gap up to GATE_LINEAR_STEPS steps and sqrt(GATE_LINEAR_STEPS * gap) beyond
# (a long absence is a random walk, not a straight line).
GATE_STEP_MULT = 3.0
GATE_LINEAR_STEPS = 10
# Spatial-only (no appearance) links are accepted up to this many track buffers.
SPATIAL_ONLY_BUFFERS = 4
# Per-decision notes kept in a LinkingResult (the counts are always exact).
MAX_DECISION_NOTES = 50

# Comparison slack for margins computed from float32 cosines.
_EPS = 1e-9


@attrs.define
class LinkingResult:
    """Outcome of :func:`link_tracklets` for one video.

    Attributes:
        n_identities_before: Identities (tracks) before linking.
        n_identities: Identities after linking.
        n_merges: Gap links applied (appearance or spatial).
        n_forced_merges: Closed-set merges applied below the open-set rule.
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

    Regular stride: each frame-index delta counts ``max(1, round(delta /
    frame_step))`` steps, so a strided video behaves like a dense one and genuine
    gaps still count; irregular input: one step per processed frame.
    """
    n = len(frame_idx)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if calibration.regular_stride:
        frame_step = max(1, int(calibration.frame_step))
        inc = np.maximum(1.0, np.rint(np.diff(frame_idx) / frame_step))
    else:
        inc = np.ones(n - 1, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(inc)])


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


class _Linker:
    """Mutable state of one :func:`link_tracklets` run (private).

    Holds every TRACKED detection of the video in temporal order (detection index
    order is time order), a current identity label per detection, and per
    identity a union-find parent (merges), liveness, temporal extent, endpoint
    centroids and gated-vector lists. Pairwise matrices over identities (rows:
    the earlier identity ``i``, columns: the later one ``j``) are kept up to date
    incrementally: a merge of ``j`` into ``i`` only changes row/column ``i``.
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
    ):
        """Collect the tracked detections and build identity state."""
        self.cal = calibration
        self.track_buffer = int(track_buffer)
        self.spatial_gate_frames = float(spatial_gate_frames)
        self.body = float(calibration.body_size)
        if not np.isfinite(self.body) or self.body <= 0:
            self.body = 50.0
        p95 = calibration.step_p95
        self.step_p95 = float(p95) if p95 is not None else float("nan")
        self.notes: List[str] = []
        self._n_decision_notes = 0

        self.frame_idx = np.array([f for f, _ in frames], dtype=np.int64)
        frame_steps = _step_positions(self.frame_idx.astype(np.float64), calibration)
        self.dim = _embedding_dim(dets)
        self._collect(dets, frame_steps, high_threshold)
        self._build_identities()

    # ------------------------------------------------------------ collection
    def _collect(
        self, dets: List[FrameDetections], frame_steps: np.ndarray, high: float
    ) -> None:
        """Gather tracked detections, gated vectors and cross-identity overlaps."""
        objs: List[Any] = []
        pos: List[int] = []
        centroids: List[np.ndarray] = []
        scores: List[float] = []
        gated: List[bool] = []
        vectors: List[Optional[np.ndarray]] = []
        labels: List[int] = []
        overlaps: List[Tuple[int, int, float]] = []
        self.tracks: List[sio.Track] = []
        track_index: Dict[int, int] = {}

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

            # Overlaps between DIFFERENT identities (ambiguity / duplicates).
            t = np.asarray(tracked)
            sub = iou[np.ix_(t, t)]
            for a, b in zip(*np.nonzero(np.triu(sub >= AMBIGUITY_IOU, k=1))):
                if labels[first + a] != labels[first + b]:
                    overlaps.append((first + int(a), first + int(b), float(sub[a, b])))

        n = len(objs)
        self.objs = objs
        self.pos = np.asarray(pos, dtype=np.int64)
        self.step = frame_steps[self.pos] if n else np.zeros(0, dtype=np.float64)
        self.centroids = (
            np.asarray(centroids, dtype=np.float64).reshape(n, 2)
            if n
            else np.zeros((0, 2))
        )
        self.valid_c = np.isfinite(self.centroids).all(axis=1)
        self.scores = np.asarray(scores, dtype=np.float64)
        self.gated = np.asarray(gated, dtype=bool)
        self.label = np.asarray(labels, dtype=np.int64)
        self.vectors = None
        if self.gated.any():
            self.vectors = np.zeros((n, self.dim), dtype=np.float32)
            for i in np.flatnonzero(self.gated):
                self.vectors[i] = vectors[i]
        ov = np.asarray([(a, b) for a, b, _ in overlaps], dtype=np.int64)
        self.overlaps = ov.reshape(-1, 2)
        self.overlap_iou = np.asarray([v for _, _, v in overlaps], dtype=np.float64)
        self.in_event = np.zeros(n, dtype=bool)
        self.in_event[self.overlaps.reshape(-1)] = True

    def _build_identities(self) -> None:
        """Per-identity member lists (time-ordered detection indices)."""
        m = len(self.tracks)
        self.members = [np.flatnonzero(self.label == k) for k in range(m)]
        self.parent = np.arange(m)
        self.alive = np.ones(m, dtype=bool)
        self.untracked = np.zeros(m, dtype=bool)
        names = [str(t.name) if t.name is not None else "" for t in self.tracks]
        self.names = names
        # Rank of each name (ties broken by first appearance) for vectorized sorts.
        order = sorted(range(m), key=lambda k: (names[k], k))
        self.name_rank = np.empty(m, dtype=np.int64)
        self.name_rank[order] = np.arange(m)

    def note(self, text: str, decision: bool = True) -> None:
        """Record a note (per-decision notes are capped at MAX_DECISION_NOTES)."""
        if decision:
            self._n_decision_notes += 1
            if self._n_decision_notes > MAX_DECISION_NOTES:
                return
        self.notes.append(text)

    # ------------------------------------------------------- swap correction
    def ambiguity_events(self) -> List[_Event]:
        """Maximal spans of processed frames where identities' boxes overlap.

        Overlapping identities in one frame form a component; a component joins
        any event that shares an identity and was active in the same or the
        previous processed frame, so a chain of crossings is one event.
        """
        if len(self.overlaps) == 0:
            return []
        frame_pos = self.pos[self.overlaps[:, 0]]
        events: List[_Event] = []
        active: List[_Event] = []
        for p in np.unique(frame_pos):
            edges = self.overlaps[frame_pos == p]
            # Components of this frame's overlap graph over identity labels.
            parent: Dict[int, int] = {}

            def find(x: int) -> int:
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in edges:
                ra, rb = find(int(self.label[a])), find(int(self.label[b]))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            comps: Dict[int, _Event] = {}
            for a, b in edges:
                for d in (int(a), int(b)):
                    lab = int(self.label[d])
                    ev = comps.setdefault(find(lab), _Event(p, p, set(), []))
                    ev.labels.add(lab)
                    if d not in ev.dets:
                        ev.dets.append(d)

            active = [e for e in active if e.end == p - 1]
            current: List[_Event] = []
            for root in sorted(comps):
                ev = comps[root]
                # Absorb every live event sharing an identity (to a fixpoint: an
                # absorbed event can bring identities shared with another).
                while True:
                    other = next(
                        (e for e in active + current if e.labels & ev.labels), None
                    )
                    if other is None:
                        break
                    ev.start = min(ev.start, other.start)
                    ev.labels |= other.labels
                    ev.dets = other.dets + ev.dets
                    pool = active if any(e is other for e in active) else current
                    pool[:] = [e for e in pool if e is not other]
                    events[:] = [e for e in events if e is not other]
                current.append(ev)
                events.append(ev)
            active = current
        events.sort(key=lambda e: (e.start, e.end, min(e.dets)))
        return events

    def _pre_gallery(self, k: int, start: int) -> np.ndarray:
        """Last GALLERY_SIZE gated vectors of identity ``k`` before ``start``."""
        m = self.members[k]
        m = m[self.pos[m] < start]
        m = m[self.gated[m]][-GALLERY_SIZE:]
        return self.vectors[m]

    def _post_query(self, k: int, end: int) -> np.ndarray:
        """First SWAP_QUERY_SIZE gated vectors of ``k`` after ``end``.

        Collection stops at ``k``'s next detection involved in an overlap (its
        next ambiguity event).
        """
        m = self.members[k]
        m = m[self.pos[m] > end]
        busy = np.flatnonzero(self.in_event[m])
        if len(busy):
            m = m[: busy[0]]
        m = m[self.gated[m]][:SWAP_QUERY_SIZE]
        return self.vectors[m]

    def correct_swaps(self) -> int:
        """Swap correction over every ambiguity event, in time order.

        Returns:
            Number of events whose identities were permuted.
        """
        thresh = float(self.cal.link_thresh)
        margin = float(self.cal.link_margin)
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
            gain = float(s[rows, cols].sum() - np.trace(s))
            pair_ok = all(
                s[r, cols[r]] >= thresh and s[r, cols[r]] - s[r, r] >= margin - _EPS
                for r in moved
            )
            if gain < margin * len(moved) - _EPS or not pair_ok:
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
            cut = int(np.searchsorted(self.pos[m], end, side="right"))
            heads[lab], tails[lab] = m[:cut], m[cut:]
        for lab, dst in perm.items():
            self.members[dst] = np.concatenate([heads[dst], tails[lab]])
            self.label[tails[lab]] = dst

    # ---------------------------------------------------------- gap linking
    def init_linking(self, use_appearance: bool) -> None:
        """Per-identity summaries and the pairwise matrices used by linking."""
        m = len(self.tracks)
        self.use_app = bool(use_appearance) and self.vectors is not None
        self.first_step = np.full(m, np.inf)
        self.last_step = np.full(m, -np.inf)
        self.start_c = np.full((m, 2), np.nan)
        self.end_c = np.full((m, 2), np.nan)
        self.n_dets = np.zeros(m, dtype=np.int64)
        self.score_sum = np.zeros(m, dtype=np.float64)
        self.gated_members: List[np.ndarray] = []
        for k in range(m):
            mem = self.members[k]
            self.gated_members.append(mem[self.gated[mem]])
            if len(mem) == 0:
                self.alive[k] = False
                continue
            self._set_extent(k)
            self.n_dets[k] = len(mem)
            self.score_sum[k] = self.scores[mem].sum()

        self.gap = np.zeros((m, m))
        self.dist = np.full((m, m), np.nan)
        # Cached appearance-margin verdicts (-1: unknown), invalidated by merges.
        self.verdict = np.full((m, m), -1, dtype=np.int8)
        self.feasible = np.zeros((m, m), dtype=bool)
        self.score = np.full((m, m), np.nan)
        if self.use_app:
            self.heads = np.zeros((m, LINK_QUERY_SIZE, self.dim), dtype=np.float32)
            self.tails = np.zeros((m, GALLERY_SIZE, self.dim), dtype=np.float32)
            self.n_head = np.zeros(m, dtype=np.int64)
            self.n_tail = np.zeros(m, dtype=np.int64)
            for k in range(m):
                self._set_galleries(k)
        for k in range(m):
            self._update_geometry_row(k)
        for k in range(m):
            if self.use_app:
                self._update_score_col(k)

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
        self.heads[k] = 0.0
        self.tails[k] = 0.0
        self.heads[k, : len(head)] = self.vectors[head]
        self.tails[k, : len(tail)] = self.vectors[tail]
        self.n_head[k], self.n_tail[k] = len(head), len(tail)

    def _radius(self, gap: np.ndarray) -> np.ndarray:
        """Spatial-gate radius (px) for gaps in steps (NaN when uncalibrated)."""
        gap = np.asarray(gap, dtype=np.float64)
        eff = np.where(
            gap <= GATE_LINEAR_STEPS,
            gap,
            np.sqrt(GATE_LINEAR_STEPS * np.maximum(gap, 0.0)),
        )
        return self.body + GATE_STEP_MULT * self.step_p95 * eff

    def _gated(self, gap: np.ndarray) -> np.ndarray:
        """Whether a junction over ``gap`` steps is spatially gated at all."""
        return (np.asarray(gap) <= self.spatial_gate_frames) & np.isfinite(
            self.step_p95
        )

    def _gate_ok(self, gap: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Spatial feasibility of junctions (``dist`` NaN fails a gated junction)."""
        gap = np.asarray(gap, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            near = np.asarray(dist) <= self._radius(np.maximum(gap, 0.0))
        return ~self._gated(gap) | near

    def _update_geometry_row(self, i: int) -> None:
        """Recompute ``gap``/``dist``/``feasible`` for ``i`` as the EARLIER side."""
        if not self.alive[i]:
            self.feasible[i, :] = False
            self.feasible[:, i] = False
            return
        order = self.alive & (self.last_step[i] < self.first_step)
        gap = np.where(order, self.first_step - self.last_step[i], 0.0)
        dist = np.linalg.norm(self.start_c - self.end_c[i], axis=1)
        self.gap[i] = gap
        self.dist[i] = dist
        self.feasible[i] = order & self._gate_ok(gap, dist)
        self.feasible[i, i] = False

    def _update_geometry_col(self, j: int) -> None:
        """Recompute ``gap``/``dist``/``feasible`` for ``j`` as the LATER side."""
        if not self.alive[j]:
            self.feasible[:, j] = False
            return
        order = self.alive & (self.last_step < self.first_step[j])
        gap = np.where(order, self.first_step[j] - self.last_step, 0.0)
        dist = np.linalg.norm(self.start_c[j] - self.end_c, axis=1)
        self.gap[:, j] = gap
        self.dist[:, j] = dist
        self.feasible[:, j] = order & self._gate_ok(gap, dist)
        self.feasible[j, j] = False

    def _update_score_row(self, i: int) -> None:
        """``score[i, j]`` for ``i``'s feasible successors ``j`` (NaN elsewhere).

        Each successor's head (queries) against ``i``'s tail (gallery).
        """
        self.score[i] = np.nan
        nt = int(self.n_tail[i])
        cols = np.flatnonzero(self.feasible[i] & (self.n_head > 0))
        if nt == 0 or len(cols) == 0:
            return
        heads = self.heads[cols].reshape(-1, self.dim)
        cos = heads @ self.tails[i, :nt].T
        kk = min(LINK_TOP_K, nt)
        top = np.partition(cos, nt - kk, axis=1)[:, nt - kk :]  # the kk largest
        per_q = top.mean(axis=1, dtype=np.float64).reshape(len(cols), LINK_QUERY_SIZE)
        valid = np.arange(LINK_QUERY_SIZE)[None, :] < self.n_head[cols, None]
        self.score[i, cols] = (
            np.where(valid, per_q, 0.0).sum(axis=1) / self.n_head[cols]
        )

    def _update_score_col(self, j: int) -> None:
        """``score[i, j]`` for ``j``'s feasible predecessors ``i`` (NaN elsewhere).

        ``j``'s head (queries) against each predecessor's tail (gallery).
        """
        self.score[:, j] = np.nan
        nh = int(self.n_head[j])
        rows = np.flatnonzero(self.feasible[:, j] & (self.n_tail > 0))
        if nh == 0 or len(rows) == 0:
            return
        tails = self.tails[rows].reshape(-1, self.dim)
        cos = (self.heads[j, :nh] @ tails.T).reshape(nh, len(rows), GALLERY_SIZE)
        empty = np.arange(GALLERY_SIZE)[None, :] >= self.n_tail[rows, None]
        cos[:, empty] = -np.inf
        top = np.partition(cos, GALLERY_SIZE - LINK_TOP_K, axis=2)[
            ..., GALLERY_SIZE - LINK_TOP_K :
        ]  # the LINK_TOP_K largest (padding -inf when a tail is shorter)
        top[~np.isfinite(top)] = 0.0
        n_top = np.minimum(LINK_TOP_K, self.n_tail[rows])
        s = top.sum(axis=2, dtype=np.float64) / n_top[None, :]
        self.score[rows, j] = s.mean(axis=0)

    def _cannot_link(self, k: int) -> np.ndarray:
        """``(M,)`` bool: live identities whose temporal extent overlaps ``k``'s."""
        return self.alive & ~(
            (self.last_step < self.first_step[k])
            | (self.last_step[k] < self.first_step)
        )

    def _rank_order(
        self, ii: np.ndarray, jj: np.ndarray, tier: np.ndarray, primary: np.ndarray
    ) -> np.ndarray:
        """Pair indices sorted by (tier, primary, gap, start of j, names) asc."""
        keys = (
            self.name_rank[jj],
            self.name_rank[ii],
            self.first_step[jj],
            self.gap[ii, jj],
            primary,
            tier,
        )
        return np.lexsort(keys)

    def best_gap_link(self) -> Optional[Tuple[int, int, str, float]]:
        """The best gap link the open-set rules accept, or ``None``.

        Appearance (pair has galleries on both sides): accept when ``s_link >=
        link_thresh`` and its margin over every competing candidate -- for the
        same later identity, those that cannot-link with the earlier one; for the
        same earlier identity, those that cannot-link with the later one -- is at
        least ``link_margin`` (no competitor: threshold only). Spatial only:
        accept when the gap is at most SPATIAL_ONLY_BUFFERS track buffers and the
        pair is the only feasible predecessor/successor within that window.
        Accepted links are ranked by (appearance before spatial, score desc,
        gap asc, start of j asc, names asc), and a link that would skip over
        another candidate of the same identity is deferred (see the stages
        below).

        Returns:
            ``(i, j, kind, score)`` with ``kind`` ``"appearance"`` or
            ``"spatial"``, or ``None`` when nothing is accepted.
        """
        feasible = self.feasible
        has_app = np.isfinite(self.score) if self.use_app else np.zeros_like(feasible)
        cand = np.zeros_like(feasible)
        if self.use_app:
            with np.errstate(invalid="ignore"):
                cand = feasible & has_app & (self.score >= self.cal.link_thresh)
        window = feasible & (self.gap <= SPATIAL_ONLY_BUFFERS * self.track_buffer)
        spatial = (
            window
            & ~has_app
            & (window.sum(axis=0)[None, :] == 1)
            & (window.sum(axis=1)[:, None] == 1)
        )
        cand |= spatial
        if not cand.any():
            return None

        def accepted(i: int, j: int) -> bool:
            if spatial[i, j]:
                return self._spatially_unique(i, j)
            if self.verdict[i, j] < 0:
                self.verdict[i, j] = self._margin_ok(i, j)
            return bool(self.verdict[i, j])

        # Stage A: the "frontier" -- candidates that skip no other candidate of
        # the same identity (no i -> k with k ending before j starts, no k -> j
        # with k starting after i ends). Merging a skip link first would nest
        # the skipped identity inside the merged identity's gap, where it could
        # never be linked. The smallest-gap candidate is always on the frontier.
        end_first = np.where(cand, self.last_step[None, :], np.inf).min(axis=1)
        start_last = np.where(cand, self.first_step[:, None], -np.inf).max(axis=0)
        frontier = (
            cand
            & (self.first_step[None, :] <= end_first[:, None])
            & (self.last_step[:, None] >= start_last[None, :])
        )
        ii, jj = np.nonzero(frontier)
        for c in self._rank_order(
            ii, jj, spatial[ii, jj], -self._score_or_zero(ii, jj)
        ):
            if accepted(int(ii[c]), int(jj[c])):
                return self._link(int(ii[c]), int(jj[c]), spatial)

        # Stage B (rare): every frontier candidate failed its margin. Accept a
        # link that skips only REJECTED candidates (none of the identities it
        # skips over can be linked to either end).
        ii, jj = np.nonzero(cand & ~frontier)
        for c in self._rank_order(
            ii, jj, spatial[ii, jj], -self._score_or_zero(ii, jj)
        ):
            i, j = int(ii[c]), int(jj[c])
            if not accepted(i, j):
                continue
            skipped_succ = np.flatnonzero(
                cand[i] & (self.last_step < self.first_step[j])
            )
            skipped_pred = np.flatnonzero(
                cand[:, j] & (self.first_step > self.last_step[i])
            )
            if any(accepted(i, int(k)) for k in skipped_succ) or any(
                accepted(int(k), j) for k in skipped_pred
            ):
                continue
            return self._link(i, j, spatial)
        return None

    def _score_or_zero(self, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        """Linking statistic of pairs, 0 where unavailable (spatial candidates)."""
        return np.nan_to_num(self.score[ii, jj], nan=0.0)

    def _link(self, i: int, j: int, spatial: np.ndarray) -> Tuple[int, int, str, float]:
        """``best_gap_link`` return value for the chosen pair."""
        if spatial[i, j]:
            return i, j, "spatial", 0.0
        return i, j, "appearance", float(self.score[i, j])

    def _margin_ok(self, i: int, j: int) -> bool:
        """Appearance margin of ``i -> j`` over its proven-different competitors.

        Competitors: candidates for the same ``j`` that cannot-link with ``i``,
        and candidates of the same ``i`` that cannot-link with ``j``. With none,
        only the threshold applies (already checked by the caller).
        """
        cl_i, cl_j = self._cannot_link(i), self._cannot_link(j)
        col = self.feasible[:, j] & np.isfinite(self.score[:, j]) & cl_i
        col[i] = False
        row = self.feasible[i, :] & np.isfinite(self.score[i, :]) & cl_j
        row[j] = False
        others = [
            self.score[col, j].max(initial=-np.inf),
            self.score[i, row].max(initial=-np.inf),
        ]
        # Identities that could take j's start (or i's end) inside one of their
        # gaps compete too -- e.g. an animal whose own fragments were already
        # joined around j.
        for h, gap1, _, d1, _ in self._hosts(j):
            if h != i and cl_i[h] and self._gate_ok(gap1, d1):
                others.append(self._junction_score(h, j, "start"))
        for h, _, gap2, _, d2 in self._hosts(i):
            if h != j and cl_j[h] and self._gate_ok(gap2, d2):
                others.append(self._junction_score(h, i, "end"))
        best_other = np.nanmax(np.asarray(others + [-np.inf]))
        if not np.isfinite(best_other):
            return True
        return bool(self.score[i, j] - best_other >= self.cal.link_margin - _EPS)

    def _spatially_unique(self, i: int, j: int) -> bool:
        """No host could take ``j``'s start or ``i``'s end within the window.

        Complements the ordered-pair uniqueness of the spatial-only rule: an
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

    # ------------------------------------------------------------ closed set
    def _rank(
        self, score: np.ndarray, gaps: List[np.ndarray], dists: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forced-merge ranking ``(tier, primary)``, both ascending.

        Tier: 0 when appearance agrees (``s_link >= link_thresh``), 1 when it is
        unavailable, 2 when it disagrees. Primary (geometry, every tier): the
        worst junction's ``4.5 z^2 + 2 ln(r)`` with ``r`` the gate radius at that
        gap and ``z = dist / r`` (the gate sits near 3 sigma). Up to constants
        this is the negative log-likelihood ratio of a random-walk continuation
        against an unrelated animal landing there: the endpoint distance
        normalized by the gate, plus the price of the gate's growth with the gap
        (a long gap leaves room for a coincidence). Without a calibrated
        ``step_p95``: ``4.5 (dist / body)^2 + ln(gap)``. Appearance thus decides
        which merges are plausible and geometry which continuation is likeliest;
        ranking forced merges by the appearance score alone lets a weak embedder
        chain wrong merges that strand fragments.
        """
        cost = np.full(len(score), -np.inf)
        for gap, dist in zip(gaps, dists):
            gap = np.maximum(np.asarray(gap, dtype=np.float64), 1.0)
            dist = np.asarray(dist, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                if np.isfinite(self.step_p95):
                    radius = self._radius(gap)
                    c = 4.5 * (dist / radius) ** 2 + 2.0 * np.log(radius)
                else:
                    c = 4.5 * (dist / self.body) ** 2 + np.log(gap)
            cost = np.maximum(cost, np.where(np.isfinite(c), c, np.inf))
        tier = np.ones(len(score), dtype=np.int64)
        if self.use_app:
            with np.errstate(invalid="ignore"):
                tier[score >= self.cal.link_thresh] = 0
                tier[score < self.cal.link_thresh] = 2
        return tier, cost

    def _entries(
        self,
        ii: np.ndarray,
        jj: np.ndarray,
        tier: np.ndarray,
        primary: np.ndarray,
        gap_key: np.ndarray,
        kind: str,
    ) -> List[tuple]:
        """Heap entries ``(tier, primary, gap, start_j, name_i, name_j, i, j, ...)``."""
        return [
            (t, p, g, f, ri, rj, i, j, int(self.version[i]), int(self.version[j]), kind)
            for t, p, g, f, ri, rj, i, j in zip(
                np.asarray(tier).tolist(),
                np.asarray(primary).tolist(),
                np.asarray(gap_key).tolist(),
                self.first_step[jj].tolist(),
                self.name_rank[ii].tolist(),
                self.name_rank[jj].tolist(),
                np.asarray(ii).tolist(),
                np.asarray(jj).tolist(),
            )
        ]

    def _ordered_entries(self, ii: np.ndarray, jj: np.ndarray) -> List[tuple]:
        """Heap entries for feasible ordered pairs (``i`` ends before ``j`` starts)."""
        if len(ii) == 0:
            return []
        gap = self.gap[ii, jj]
        score = self.score[ii, jj] if self.use_app else np.full(len(ii), np.nan)
        tier, primary = self._rank(score, [gap], [self.dist[ii, jj]])
        return self._entries(ii, jj, tier, primary, gap, "ordered")

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
        inside = (
            self.alive[cand]
            & (self.first_step[cand] > self.first_step[h])
            & (self.last_step[cand] < self.last_step[h])
        )
        cand = np.asarray(cand)[inside]
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

    def _insertion_entries(
        self, i: int, cand: Optional[np.ndarray] = None
    ) -> List[tuple]:
        """Heap entries for identities fitting entirely inside one gap of ``i``.

        Such an identity never co-occurs with ``i``; the merge is feasible when
        both junctions (``i`` -> it, it -> ``i``) pass the spatial gate.

        Args:
            i: Host identity.
            cand: Candidate guests (default: every live identity).
        """
        cand = np.flatnonzero(self.alive) if cand is None else cand
        cand, gap1, gap2, d1, d2 = self._gap_fits(i, cand)
        ok = self._gate_ok(gap1, d1) & self._gate_ok(gap2, d2)
        cand, gap1, gap2, d1, d2 = cand[ok], gap1[ok], gap2[ok], d1[ok], d2[ok]
        if len(cand) == 0:
            return []
        score = np.full(len(cand), np.nan)
        if self.use_app:
            score = np.array([self._insertion_score(i, int(j)) for j in cand])
        tier, primary = self._rank(score, [gap1, gap2], [d1, d2])
        ii = np.full(len(cand), i)
        return self._entries(
            ii, cand, tier, primary, np.maximum(gap1, gap2), "insertion"
        )

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
                self.vectors[gk[:GALLERY_SIZE]], self.vectors[before[-GALLERY_SIZE:]]
            )
        after = g[np.searchsorted(gs, self.last_step[k], side="right") :]
        return _s_link(
            self.vectors[after[:GALLERY_SIZE]], self.vectors[gk[-GALLERY_SIZE:]]
        )

    def _insertion_score(self, i: int, j: int) -> float:
        """Mean linking statistic over the two junctions of inserting ``j`` into ``i``."""
        scores = [self._junction_score(i, j, side) for side in ("start", "end")]
        scores = [v for v in scores if np.isfinite(v)]
        return float(np.mean(scores)) if scores else float("nan")

    def _entries_for(self, k: int) -> List[tuple]:
        """Every closed-set candidate involving ``k`` (after ``k`` changed)."""
        out = self._ordered_entries(
            np.full(int(self.feasible[k].sum()), k), np.flatnonzero(self.feasible[k])
        )
        out += self._ordered_entries(
            np.flatnonzero(self.feasible[:, k]),
            np.full(int(self.feasible[:, k].sum()), k),
        )
        out += self._insertion_entries(k)
        hosts = np.flatnonzero(
            self.alive
            & (self.first_step < self.first_step[k])
            & (self.last_step > self.last_step[k])
        )
        for h in hosts:
            out += self._insertion_entries(int(h), np.array([k]))
        return out

    def close_set(self, n_max: int) -> Tuple[int, int]:
        """Reduce the live identities to at most ``n_max``.

        Repeatedly applies the best feasible merge -- an ordered pair, or an
        identity that fits entirely inside one gap of another (merges can nest
        fragments inside a merged identity's gaps) -- ranked by :meth:`_rank`,
        then gap, start and names. When no merge is feasible (the remaining
        identities co-occur), untracks one identity (:meth:`untrack_one`).

        Returns:
            ``(n_forced_merges, n_untracked_detections)``.
        """
        n_forced = n_untracked = 0
        self.version = np.zeros(len(self.tracks), dtype=np.int64)
        heap = self._ordered_entries(*np.nonzero(self.feasible))
        for k in np.flatnonzero(self.alive):
            heap += self._insertion_entries(int(k))
        heapq.heapify(heap)
        while self.n_alive() > n_max:
            best = None
            while heap:
                entry = heapq.heappop(heap)
                i, j, vi, vj = entry[6:10]
                if (
                    self.alive[i]
                    and self.alive[j]
                    and self.version[i] == vi
                    and self.version[j] == vj
                ):
                    best = entry
                    break
            if best is None:
                k = self.untrack_one()
                n_untracked += int(self.n_dets[k])
                continue
            tier, primary, gap = best[0], best[1], best[2]
            i, j, kind = best[6], best[7], best[10]
            look = ("appearance agrees", "no appearance", "appearance disagrees")
            self.note(
                f"closed set: forced {self.names[j]} -> {self.names[i]} ({kind}, "
                f"{look[tier]}, geometric cost {primary:.2f}, gap {gap:g} steps)"
            )
            self.merge(i, j)
            n_forced += 1
            self.version[i] += 1
            self.version[j] += 1
            for entry in self._entries_for(i):
                heapq.heappush(heap, entry)
        return n_forced, n_untracked

    def merge(self, i: int, j: int) -> None:
        """Merge identity ``j`` into the EARLIER-starting ``i`` (keeps ``i``'s Track).

        ``j`` either starts after ``i`` ends or fits inside one of its gaps; the
        two never co-occur. Detection indices are in temporal order, so sorting
        the union keeps every per-identity list time-ordered.
        """
        # Competitor structure of i and j BEFORE the merge (margin-verdict cache).
        before = [
            (
                self._cannot_link(k),
                self.feasible[:, k].copy(),
                self.feasible[k].copy(),
                self._guests(k),
            )
            for k in (i, j)
        ]
        self.parent[j] = i
        self.alive[j] = False
        self.members[i] = np.sort(np.concatenate([self.members[i], self.members[j]]))
        self.gated_members[i] = np.sort(
            np.concatenate([self.gated_members[i], self.gated_members[j]])
        )
        self._set_extent(i)
        self.n_dets[i] += self.n_dets[j]
        self.score_sum[i] += self.score_sum[j]
        self._remove(j)
        self._update_geometry_row(i)
        self._update_geometry_col(i)
        if self.use_app:
            self._set_galleries(i)
            self._update_score_row(i)
            self._update_score_col(i)
        # A cached margin verdict of (a, b) changes only if a or b is i or j, or
        # if i or j (before or after the merge) is one of its competitors: x is
        # a competitor of (a, b) when b is x's successor and a co-occurs with x,
        # or a is x's predecessor and b co-occurs with x.
        # A host x competes for its guests' starts and ends the same way.
        after = (
            self._cannot_link(i),
            self.feasible[:, i],
            self.feasible[i],
            self._guests(i),
        )
        for cl, pred, succ, guests in before + [after]:
            self.verdict[np.ix_(cl, succ)] = -1
            self.verdict[np.ix_(pred, cl)] = -1
            self.verdict[np.ix_(cl, guests)] = -1
            self.verdict[np.ix_(guests, cl)] = -1
        self.verdict[[i, j], :] = -1
        self.verdict[:, [i, j]] = -1

    def _remove(self, k: int) -> None:
        """Drop identity ``k`` from every pairwise matrix."""
        self.feasible[k, :] = False
        self.feasible[:, k] = False
        self.score[k, :] = np.nan
        self.score[:, k] = np.nan

    def roots(self) -> np.ndarray:
        """Current identity (union-find root) of every tracked detection."""
        r = self.parent[self.label]
        while True:
            nxt = self.parent[r]
            if np.array_equal(nxt, r):
                return r
            r = nxt

    def duplicate_fraction(self) -> np.ndarray:
        """Per identity: fraction of its frames overlapping another at DUPLICATE_IOU."""
        m = len(self.tracks)
        dup = self.overlaps[self.overlap_iou >= DUPLICATE_IOU]
        if len(dup) == 0:
            return np.zeros(m, dtype=np.float64)
        roots = self.roots()
        # (identity, frame) pairs encoded as one integer key.
        n_pos = int(self.pos.max()) + 1
        n_frames = np.bincount(
            np.unique(roots * n_pos + self.pos) // n_pos, minlength=m
        )
        a, b = dup[:, 0], dup[:, 1]
        ra, rb = roots[a], roots[b]
        keep = (ra != rb) & self.alive[ra] & self.alive[rb]
        hit_keys = np.unique(
            np.concatenate(
                [
                    ra[keep] * n_pos + self.pos[a[keep]],
                    rb[keep] * n_pos + self.pos[b[keep]],
                ]
            )
        )
        hits = np.bincount(hit_keys // n_pos, minlength=m)
        return hits / np.maximum(n_frames, 1)

    def untrack_one(self) -> int:
        """Untrack the most duplicated identity, else the one with fewest detections.

        Returns:
            The untracked identity.
        """
        alive = np.flatnonzero(self.alive)
        frac = self.duplicate_fraction()[alive]
        mean_score = self.score_sum[alive] / np.maximum(self.n_dets[alive], 1)
        dup = frac >= DUPLICATE_FRACTION
        if dup.any():
            cand = alive[dup]
            b = np.lexsort((-self.name_rank[cand], self.n_dets[cand], -frac[dup]))[0]
            k = int(cand[b])
            reason = f"duplicate ({frac[dup][b]:.0%} of frames overlap another)"
        else:
            b = np.lexsort((-self.name_rank[alive], mean_score, self.n_dets[alive]))[0]
            k = int(alive[b])
            reason = "fewest detections"
        self.alive[k] = False
        self.untracked[k] = True
        self._remove(k)
        self.note(
            f"closed set: untracked {self.names[k]} ({int(self.n_dets[k])} "
            f"detections, {reason})"
        )
        return k

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
    track_objects: Optional[Dict[int, sio.Track]] = None,
) -> LinkingResult:
    """Link a tracked video's identities across gaps and crossings, in place.

    Steps, in order (see the module docstring for the definitions):

    1. **Swap correction** (when ``swap_correction`` and the calibration reports
       ``appearance_for_linking``): for each ambiguity event over at most
       ``MAX_EVENT_IDENTITIES`` identities, the post-event vectors of each
       identity are scored against every identity's pre-event gallery; the best
       permutation is applied to the moved identities' detections after the event
       when its total gain is at least ``link_margin`` per moved identity and
       every moved pair clears ``link_thresh`` and beats its unswapped pair by
       ``link_margin``.
    2. **Gap linking**, greedily best-first (appearance before spatial, score
       desc, gap asc, start asc, name asc) until nothing is accepted, between an
       identity that ends and one that starts later within the spatial gate
       ``body + 3 * step_p95 * (gap if gap <= 10 else sqrt(10 * gap))`` (no gate
       beyond ``spatial_gate_frames`` steps or when ``step_p95`` is unknown).
       Pairs with gated vectors on both sides (and calibrated appearance) are
       decided by appearance (``link_thresh``, then a ``link_margin`` over the
       competing candidates that provably are other animals); others are linked
       only when the gap is at most 4 track buffers and the pair is mutually the
       only feasible candidate in that window. A link that would skip over
       another candidate of the same identity waits for it (so that fragment is
       not stranded inside the merged identity's gap).
    3. **Closed set** (``max_tracks``): while more identities remain, force the
       best feasible merge -- an ordered pair, or an identity that fits inside
       one gap of another, every junction spatially gated -- ranked by
       appearance agreement, then a gap-aware geometric cost; when none is
       feasible (the identities co-occur), untrack a duplicate identity (at least
       half of its frames overlapping another at IoU >= 0.5; the most duplicated
       first), else the one with the fewest detections.

    Args:
        frames: ``(frame_idx, objects)`` per frame in temporal order: the tracked
            detections (``.track`` set or ``None``) of ONE video.
        carrier: ``"pose"`` or ``"mask"``.
        calibration: The video's calibration (computes one from ``frames`` if
            ``None``). Appearance is used only where it reports
            ``appearance_for_linking``.
        max_tracks: Known number of animals: closed-set linking.
        track_buffer: Gap (steps) below which a track's detections form one
            tracklet; also scales the spatial-only rule.
        swap_correction: Correct swaps at crossings (appearance only).
        spatial_gate_frames: Gaps longer than this (steps) are not spatially gated.
        box_padding: Box padding used for overlap tests.
        high_threshold: Minimum score for a vector to enter a gallery.
        score_mode: Detection score mode (see ``MOTParams.score_mode``).
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

    pseudo = None if calibration is None else float(calibration.body_size)
    dets = [
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
        calibration = calibrate(dets, track_buffer=track_buffer)
        if carrier == "pose" and any(_has_degenerate_pose(d) for d in dets):
            # Degenerate poses (single node) get the calibrated body-size box, as
            # in the trackers, so overlap tests see a realistic extent.
            dets = [
                extract_detections(
                    objs,
                    f,
                    carrier=carrier,
                    box_padding=box_padding,
                    pseudo_box_size=float(calibration.body_size),
                    score_mode=score_mode,
                )
                for f, objs in frames
            ]

    linker = _Linker(
        frames,
        dets,
        calibration,
        track_buffer=track_buffer,
        spatial_gate_frames=spatial_gate_frames,
        high_threshold=high_threshold,
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
    appearance = (
        bool(calibration.appearance_for_linking)
        and calibration.link_thresh is not None
        and calibration.link_margin is not None
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
        gap = linker.gap[i, j]
        detail = f"score {score:.3f}" if kind == "appearance" else "unique"
        linker.note(
            f"linked {linker.names[j]} -> {linker.names[i]} ({kind}, {detail}, "
            f"gap {gap:g} steps)"
        )
        linker.merge(i, j)
        result.n_merges += 1

    # 3. Closed set.
    if max_tracks is not None and int(max_tracks) < 1:
        linker.note(f"max_tracks={max_tracks} ignored (must be >= 1)", decision=False)
    elif max_tracks is not None and linker.n_alive() > int(max_tracks):
        result.n_forced_merges, result.n_untracked = linker.close_set(int(max_tracks))

    # Write back: merged detections take the root identity's Track (the earliest).
    roots = linker.roots()
    for d, obj in enumerate(linker.objs):
        k = int(roots[d])
        if linker.untracked[k]:
            obj.track = None
            obj.tracking_score = 0.0
        elif obj.track is not linker.tracks[k]:
            obj.track = linker.tracks[k]

    removed = {
        id(linker.tracks[k])
        for k in range(n_before)
        if linker.parent[k] != k or linker.untracked[k]
    }
    if track_objects is not None and removed:
        for key in [k for k, t in track_objects.items() if id(t) in removed]:
            del track_objects[key]

    result.n_identities = linker.n_alive()
    if linker._n_decision_notes > MAX_DECISION_NOTES:
        linker.notes.append(
            f"... {linker._n_decision_notes - MAX_DECISION_NOTES} more decision(s) "
            "not listed"
        )
    result.notes.extend(linker.notes)

    changed = result.n_merges + result.n_forced_merges + result.n_swaps
    log = logger.info if changed or result.n_untracked else logger.debug
    log(
        f"Identity linking: {n_before} -> {result.n_identities} identities "
        f"({result.n_merges} gap links, {result.n_forced_merges} forced, "
        f"{result.n_swaps} swaps corrected, {result.n_untracked} detections "
        "untracked)."
    )
    return result
