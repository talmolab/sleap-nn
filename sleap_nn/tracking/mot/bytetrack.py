"""ByteTrack (Zhang et al., 2022) and BoT-SORT (Aharon et al., 2022).

ByteTrack associates every detection box: high-score detections first, then the
remaining tracked tracks against LOW-score detections (occluded animals), then
unconfirmed tracks. BoT-SORT adds a ``[cx, cy, w, h]`` Kalman filter and fuses
re-ID appearance into the first stages -- here only inside the geometric gate and
only where appearance discriminates. Original implementations from the published
algorithms.

Algorithm (per frame, after every live track is predicted to this step; ``sim`` is
the geometric similarity of a track's prediction to a detection, ``s`` the
detection score):

1. Split the detections with the references' strict comparisons:
   ``D_hi = {s > track_high_threshold}`` and ``D_lo = {track_low_threshold < s <
   track_high_threshold}``. Everything else (including ``s`` exactly at the high
   threshold) is ignored. Only ``D_hi`` detections scoring at least
   ``new_track_threshold`` may start a track.
2. **Stage 1:** confirmed and lost tracks vs ``D_hi``. Cost ``1 - sim * s`` (score
   fusion, ``fuse_score``) or ``1 - sim``; cost cap ``1 - min_match_similarity``.
   A lost track that matches is re-activated with its id.
3. **Stage 2:** the tracks still unmatched that were matched at the previous step
   (``CONFIRMED``, not ``LOST``) vs ``D_lo``: cost ``1 - sim`` (never fused), cap
   ``1 - low_stage_min_similarity``. This is what keeps an occluded animal --
   whose detection score drops -- on its track. Tracks unmatched after this stage
   become ``LOST``.
4. **Stage 3:** tentative tracks vs the ``D_hi`` detections left after stage 1,
   stage-1 cost, cap ``1 - unconfirmed_min_similarity``. A tentative track left
   unmatched dies (``tentative_dies_on_miss``).
5. **Duplicate removal** (reference ``remove_duplicate_stracks``): every pair of a
   track matched this frame and a lost track whose predicted similarity to that
   detection exceeds ``duplicate_similarity`` is a duplicate; the one with the
   shorter history (``last_frame - start_frame``; ties remove the matched one) is
   removed. A removed matched track's detection stays untracked and does not start
   a track. Detections that would start a track count as matched tracks with an
   empty history, so a would-be track duplicating a lost track is not started.

BoT-SORT changes stages 1 and 3 only (stage 2 stays geometric)::

    d_geo = 1 - sim * s           (or 1 - sim without fuse_score)
    far   = sim < proximity_threshold
    cos   = cosine(track appearance EMA, detection vector)
    use appearance where: cos >= gate, not far, and the cell's row AND column
          cosine margins (top-1 minus top-2 among in-gate cells) >= margin
    C     = min(d_geo, (1 - cos) / 2) + 1e-3 * d_geo   where appearance is used
          = d_geo                                        elsewhere

The gate is ``appearance_threshold`` (calibrated EMA gate under ``"auto"``,
disabled when the video's calibration says appearance does not discriminate; see
:meth:`BaseMOTTracker._appearance_gate`); the margin is the calibrated
``m_assoc`` (0 under the ``paper`` preset). Appearance therefore never creates a
match outside the geometric gate, and non-discriminative vectors leave the
tracker identical to its geometric (ByteTrack-style) self.

Adaptations to sleap-nn (documented deviations from the references):

- The geometric similarity is the tracker's resolved kind (IoU of padded boxes,
  OKS of the predicted pose, or centroid proximity), also for duplicate removal.
  The reference duplicate test is the IoU of the updated tracked box vs the lost
  box; here the matched track's detection stands in for its updated box.
- Tracks are labeled from ``min_hits`` matches on (retroactively), never on the
  first frame alone; BoT-SORT's reference also emits unconfirmed tracks, these
  stay unlabeled until confirmed.
- The geometric tie-break ``1e-3 * d_geo`` only applies where the appearance cost
  is in use, so BoT-SORT without usable appearance reproduces ByteTrack's costs
  exactly.
- BoT-SORT with no appearance vectors at all associates on geometry alone (with a
  warning) unless ``strict_vectors`` is set, in which case :meth:`prepare` raises.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.association import (
    appearance_margins,
    fuse_score,
    linear_assignment,
)
from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections
from sleap_nn.tracking.mot.params import MOTParams
from sleap_nn.tracking.mot.track import MOTTrack, TrackState
from sleap_nn.tracking.utils import get_embedding

# Weight of the geometric cost added to BoT-SORT's appearance-based cost, so that
# pairs with equally good appearance are ordered by geometry.
GEOMETRY_TIE_BREAK = 1e-3


class ByteTracker(BaseMOTTracker):
    """ByteTrack tracker (see :mod:`sleap_nn.tracking.mot.bytetrack`).

    Class attributes:
        size_velocity: Kalman state indices zeroed before predicting a track that
            was not matched at the previous step, per filter (ByteTrack: the
            height velocity; the area velocity for ``xysr``). Selects
            ``lost_zero_velocity`` for the filter in use.
    """

    name = "bytetrack"
    default_kalman = "xyah"
    tentative_dies_on_miss = True
    hits_start = 1
    lost_zero_velocity = (7,)
    size_velocity: ClassVar[Dict[str, Tuple[int, ...]]] = {
        "xyah": (7,),
        "xywh": (7,),
        "xysr": (6,),
    }

    def __init__(
        self,
        params: MOTParams,
        carrier: str = "pose",
        n_nodes: Optional[int] = None,
    ):
        """Create a tracker for one video.

        Args:
            params: Resolved :class:`~sleap_nn.tracking.mot.params.MOTParams`.
            carrier: ``"pose"`` or ``"mask"``.
            n_nodes: Skeleton size for pose detections (``None`` if unknown).
        """
        super().__init__(params, carrier=carrier, n_nodes=n_nodes)
        # Zero the velocities that exist in the filter actually in use (a
        # `kalman` override changes the state layout).
        self.lost_zero_velocity = self.size_velocity[self.kalman_kind]

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """BYTE association: high, low and unconfirmed stages, then de-duplication.

        Args:
            dets: This frame's detections.

        Returns:
            The matches (with geometric similarities), the detections that must
            not start a track and the tracks removed as duplicates.
        """
        p = self.params
        scores = np.asarray(dets.scores, dtype=np.float64)
        valid = np.asarray(dets.valid, dtype=bool)
        high = valid & (scores > p.track_high_threshold)
        low = (
            valid & (scores > p.track_low_threshold) & (scores < p.track_high_threshold)
        )
        # Only high-score detections may start a track, and only from the
        # new-track threshold on.
        spawnable = high & (scores >= p.new_track_threshold)
        result = AssociationResult(
            no_spawn={int(j) for j in np.flatnonzero(~spawnable)}
        )

        tracks = self._live()
        if not tracks:
            return result
        sim = self._geometric_similarity(tracks, dets)
        rows = {t.tid: i for i, t in enumerate(tracks)}

        # Stage 1: confirmed and lost tracks vs high-score detections.
        pool = [t for t in tracks if t.state != TrackState.TENTATIVE]
        left_pool, left_high = self._assign(
            pool,
            np.flatnonzero(high),
            dets,
            sim,
            rows,
            max_cost=1.0 - p.min_match_similarity,
            result=result,
            fused=True,
        )

        # Stage 2: tracks matched at the previous step vs low-score detections
        # (geometry only, never fused). Unmatched ones become LOST in the base.
        tracked = [t for t in left_pool if t.state == TrackState.CONFIRMED]
        self._assign(
            tracked,
            np.flatnonzero(low),
            dets,
            sim,
            rows,
            max_cost=1.0 - p.low_stage_min_similarity,
            result=result,
            fused=False,
        )

        # Stage 3: tentative tracks vs the high-score detections left over.
        tentative = [t for t in tracks if t.state == TrackState.TENTATIVE]
        _, left_high = self._assign(
            tentative,
            left_high,
            dets,
            sim,
            rows,
            max_cost=1.0 - p.unconfirmed_min_similarity,
            result=result,
            fused=True,
        )

        self._remove_duplicates(tracks, sim, rows, dets, left_high, spawnable, result)
        return result

    # ------------------------------------------------------------- helpers
    def _assign(
        self,
        tracks: Sequence[MOTTrack],
        det_idx: np.ndarray,
        dets: FrameDetections,
        sim: np.ndarray,
        rows: Dict[int, int],
        *,
        max_cost: float,
        result: AssociationResult,
        fused: bool,
    ) -> Tuple[List[MOTTrack], np.ndarray]:
        """One assignment stage; appends its matches to ``result``.

        Args:
            tracks: The stage's tracks.
            det_idx: The stage's detection indices.
            dets: This frame's detections.
            sim: ``(n_live, n_dets)`` geometric similarity of every live track.
            rows: Live track ``tid`` -> row of ``sim``.
            max_cost: Cost cap of a match.
            result: Receives the matches.
            fused: Use the stage-1/3 cost (:meth:`_fused_cost`) instead of
                ``1 - sim``.

        Returns:
            The unmatched tracks and the unmatched detection indices.
        """
        tracks = list(tracks)
        det_idx = np.asarray(det_idx, dtype=np.int64)
        if not tracks or len(det_idx) == 0:
            return tracks, det_idx
        sub = sim[np.ix_([rows[t.tid] for t in tracks], det_idx)]
        cost = self._fused_cost(tracks, det_idx, dets, sub) if fused else 1.0 - sub
        pairs, left_rows, left_cols = linear_assignment(cost, max_cost=max_cost)
        for r, c in pairs:
            result.matches.append((tracks[r], int(det_idx[c]), float(sub[r, c])))
        return [tracks[r] for r in left_rows], det_idx[left_cols]

    def _fused_cost(
        self,
        tracks: Sequence[MOTTrack],
        det_idx: np.ndarray,
        dets: FrameDetections,
        sim: np.ndarray,
    ) -> np.ndarray:
        """Stage-1/3 cost: ``1 - sim * score`` (``fuse_score``) or ``1 - sim``.

        Args:
            tracks: The stage's tracks (rows).
            det_idx: The stage's detection indices (columns).
            dets: This frame's detections.
            sim: ``(len(tracks), len(det_idx))`` geometric similarity.

        Returns:
            The ``(len(tracks), len(det_idx))`` cost matrix.
        """
        if self.params.fuse_score:
            sim = fuse_score(sim, np.asarray(dets.scores, dtype=np.float64)[det_idx])
        return 1.0 - sim

    def _remove_duplicates(
        self,
        tracks: Sequence[MOTTrack],
        sim: np.ndarray,
        rows: Dict[int, int],
        dets: FrameDetections,
        left_high: np.ndarray,
        spawnable: np.ndarray,
        result: AssociationResult,
    ) -> None:
        """Tracked-vs-lost duplicate removal (shorter history loses), in place.

        The tracked side holds this frame's matched tracks (box: the matched
        detection) and the detections about to start a track (empty history); the
        lost side holds the live tracks left unmatched (box: their prediction)
        that survive this step. A pair whose geometric similarity exceeds
        ``duplicate_similarity`` drops the one with the shorter history; ties drop
        the tracked one (reference semantics). Every pair is decided
        independently.

        Args:
            tracks: Every live track (rows of ``sim``).
            sim: ``(n_live, n_dets)`` geometric similarity.
            rows: Live track ``tid`` -> row of ``sim``.
            dets: This frame's detections.
            left_high: High-score detections left unmatched by every stage.
            spawnable: ``(n_dets,)`` detections allowed to start a track.
            result: Updated in place (matches, ``no_spawn``, ``removed``).
        """
        p = self.params
        matched_tids = {t.tid for t, _, _ in result.matches}
        lost = [
            t
            for t in tracks
            if t.state != TrackState.TENTATIVE
            and t.tid not in matched_tids
            and t.time_since_update <= p.track_buffer
        ]
        if not lost:
            return
        # Tracked side: (matched track or None for a would-be new track, det, history).
        tracked: List[Tuple[Optional[MOTTrack], int, int]] = [
            (t, int(j), int(dets.frame_idx) - int(t.start_frame))
            for t, j, _ in result.matches
        ]
        tracked += [(None, int(j), 0) for j in left_high if spawnable[j]]
        if not tracked:
            return

        dup = sim[np.ix_([rows[t.tid] for t in lost], [j for _, j, _ in tracked])]
        drop_tracked, drop_lost = set(), set()
        for q, k in np.argwhere(dup > p.duplicate_similarity):
            lost_history = int(lost[q].last_frame) - int(lost[q].start_frame)
            if tracked[k][2] > lost_history:
                drop_lost.add(int(q))
            else:
                drop_tracked.add(int(k))
        if not drop_tracked and not drop_lost:
            return

        removed = {t.tid: t for t in result.removed}
        for k in sorted(drop_tracked):
            track, det_idx, _ = tracked[k]
            # The detection stays untracked: no match, and no new track from it.
            result.no_spawn.add(det_idx)
            if track is not None:
                removed.setdefault(track.tid, track)
        for q in sorted(drop_lost):
            removed.setdefault(lost[q].tid, lost[q])
        result.matches = [m for m in result.matches if m[0].tid not in removed]
        result.removed = list(removed.values())
        logger.debug(
            f"{self.name}: frame {dets.frame_idx}: removed {len(removed)} duplicate "
            f"track(s) (tids {sorted(removed)})."
        )


class BoTSORTTracker(ByteTracker):
    """BoT-SORT tracker (see :mod:`sleap_nn.tracking.mot.bytetrack`).

    Attributes:
        strict_vectors: When ``True``, :meth:`prepare` raises if the video's
            detections carry no appearance vector at all (the "no silent no-op"
            rule of the CLI path); when ``False`` (default, library use) the
            tracker warns and associates on geometry alone.
    """

    name = "botsort"
    default_kalman = "xywh"
    uses_appearance = True
    lost_zero_velocity = (6, 7)
    # BoT-SORT zeroes the width and height velocities (aspect and height for
    # `xyah`; the area for `xysr`, whose aspect has no velocity).
    size_velocity: ClassVar[Dict[str, Tuple[int, ...]]] = {
        "xywh": (6, 7),
        "xyah": (6, 7),
        "xysr": (6,),
    }

    def __init__(
        self,
        params: MOTParams,
        carrier: str = "pose",
        n_nodes: Optional[int] = None,
    ):
        """Create a tracker for one video.

        Args:
            params: Resolved :class:`~sleap_nn.tracking.mot.params.MOTParams`.
            carrier: ``"pose"`` or ``"mask"``.
            n_nodes: Skeleton size for pose detections (``None`` if unknown).
        """
        super().__init__(params, carrier=carrier, n_nodes=n_nodes)
        self.strict_vectors = False
        self._warned_no_vectors = False

    def prepare(self, frames: Sequence[Tuple[int, Sequence]]) -> None:
        """Per-video pre-pass (see :meth:`BaseMOTTracker.prepare`).

        Args:
            frames: ``(frame_idx, objects)`` per frame, in temporal order.

        Raises:
            ValueError: If ``strict_vectors`` is set and the frames hold
                detections but none carries an appearance vector.
        """
        frames = [(int(fi), list(objs)) for fi, objs in frames]
        if self.strict_vectors:
            objs = [obj for _, frame_objs in frames for obj in frame_objs]
            if objs and not any(_has_vector(obj) for obj in objs):
                raise ValueError(
                    "botsort associates by appearance, but none of the tracked "
                    "detections carries an appearance vector (identity_embedding). "
                    "Run an embedding model first (or track the carrier that holds "
                    "the vectors), or use --tracker bytetrack for the same "
                    "algorithm without appearance."
                )
        super().prepare(frames)

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """BYTE association with BoT-SORT's gated appearance cost (stages 1 and 3).

        Args:
            dets: This frame's detections.

        Returns:
            See :meth:`ByteTracker._associate`.
        """
        if (
            self.calibration is None
            and not self._vectors_seen
            and not self._warned_no_vectors
            and bool(np.any(dets.valid))
        ):
            self._warned_no_vectors = True
            logger.warning(
                "botsort: the detections carry no appearance vectors; associating "
                "on geometry alone until vectors appear."
            )
        return super()._associate(dets)

    def _association_margin(self) -> float:
        """Row/column cosine margin required for appearance to decide a pair.

        The calibrated ``m_assoc`` (see :meth:`BaseMOTTracker._appearance_margin`);
        0 under the ``paper`` preset (the published algorithm has no margin).
        """
        if self.params.preset == "paper":
            return 0.0
        return self._appearance_margin()

    def _fused_cost(
        self,
        tracks: Sequence[MOTTrack],
        det_idx: np.ndarray,
        dets: FrameDetections,
        sim: np.ndarray,
    ) -> np.ndarray:
        """Stage-1/3 cost: geometric cost, lowered by gated decisive appearance.

        Args:
            tracks: The stage's tracks (rows).
            det_idx: The stage's detection indices (columns).
            dets: This frame's detections.
            sim: ``(len(tracks), len(det_idx))`` geometric similarity (unfused).

        Returns:
            The ``(len(tracks), len(det_idx))`` cost matrix.
        """
        d_geo = super()._fused_cost(tracks, det_idx, dets, sim)
        gate = self._appearance_gate("ema")
        if gate is None or dets.embeddings is None:
            return d_geo
        cos = self._appearance_cos(tracks, dets, "ema")[:, det_idx]
        # Appearance only inside the geometric (proximity) gate.
        near = sim >= self.params.proximity_threshold
        in_gate = np.isfinite(cos) & (cos >= gate) & near
        if not in_gate.any():
            return d_geo
        # ...and only where it discriminates among the in-gate candidates.
        row_margin, col_margin = appearance_margins(np.where(in_gate, cos, np.nan))
        margin = self._association_margin()
        decisive = (
            in_gate & (row_margin[:, None] >= margin) & (col_margin[None, :] >= margin)
        )
        d_emb = np.where(decisive, 0.5 * (1.0 - np.nan_to_num(cos)), 1.0)
        return np.where(
            decisive,
            np.minimum(d_geo, d_emb) + GEOMETRY_TIE_BREAK * d_geo,
            d_geo,
        )


def _has_vector(obj) -> bool:
    """Whether ``obj`` carries a usable (finite, non-zero) appearance vector."""
    try:
        vec = get_embedding(obj)
    except (AttributeError, TypeError, ValueError):
        return False
    if vec is None:
        return False
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    return bool(vec.size) and bool(np.isfinite(vec).all()) and bool(np.any(vec != 0))
