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
   track matched this frame and a lost track whose predicted box overlaps that
   detection's box with IoU above ``duplicate_similarity`` is a duplicate; the one
   with the shorter history (``last_frame - start_frame``; ties remove the
   matched one) is removed. A removed matched track's detection stays untracked
   and does not start a track. Detections that would start a track count as
   matched tracks with an empty history, so a would-be track duplicating a lost
   track is not started.

BoT-SORT changes stages 1 and 3 only (stage 2 stays geometric)::

    d_geo = 1 - sim * s           (or 1 - sim without fuse_score)
    far   = sim < proximity_threshold
    cos   = cosine(track appearance EMA, detection vector)
    in_gate = cos >= gate and not far
    use appearance where: in_gate, and the cell is the top in-gate cosine of its
          row AND of its column, each by a top-1 minus top-2 margin > 0 and
          >= margin (``paper`` preset: every in-gate cell, as published)
    C     = min(d_geo, (1 - cos) / 2) + 1e-3 * d_geo   where appearance is used
          = d_geo                                        elsewhere

The gate is ``appearance_threshold`` (calibrated EMA gate under ``"auto"``,
disabled when the video's calibration says appearance does not discriminate; see
:meth:`BaseMOTTracker._appearance_gate`); the margin is the calibrated
``m_assoc``. Appearance therefore never creates a match outside the geometric
gate, never lowers the cost of a pair it ranks below another candidate, and
non-discriminative vectors leave the tracker identical to its geometric
(ByteTrack-style) self.

Adaptations to sleap-nn (documented deviations from the references):

- The geometric similarity is the tracker's resolved kind (IoU of padded boxes,
  OKS of the predicted pose, or centroid proximity).
- Pose IoU compares the same physical extent on both sides. A keypoint box only
  spans the VISIBLE nodes, so occlusion -- exactly when a detection's score drops
  into ``D_lo`` -- shrinks it, while the track's Kalman box keeps the full body
  (the references' detectors predict full-extent boxes even under occlusion).
  The similarity of a pose pair is therefore the larger of the Kalman-box IoU
  and the IoU of the padded boxes of the nodes visible in BOTH the track's
  predicted pose (last pose shifted by the Kalman motion) and the detection
  (:func:`shared_node_iou`; at least 2 shared nodes). The published thresholds
  (stage-2 0.5, fused stage-1 0.2) keep their IoU meaning.
- Duplicate removal uses the IoU of the lost track's predicted box and the
  matched detection's box (the reference test; the detection stands in for the
  updated tracked box) for the ``iou`` and ``centroid`` similarities, and the
  OKS of the predicted pose for ``oks`` (a left/right-flipped pose overlaps its
  original box). The centroid kernel must not be used here: at 0.85 it admits
  ~3.5x the displacement an IoU of 0.85 does, which removed the tracks of
  nearby animals after a single missed detection.
- Tracks are labeled from ``min_hits`` matches on (retroactively), never on the
  first frame alone; BoT-SORT's reference also emits unconfirmed tracks, these
  stay unlabeled until confirmed.
- The geometric tie-break ``1e-3 * d_geo`` only applies where the appearance cost
  is in use, so BoT-SORT without usable appearance reproduces ByteTrack's costs
  exactly.
- BoT-SORT with no usable appearance vector (``FrameDetections.has_embedding``)
  associates on geometry alone, with a warning; refusing vector-less input is
  ``apply_tracking``'s job.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.association import (
    decisive_cells,
    fuse_score,
    iou_matrix,
    linear_assignment,
)
from sleap_nn.tracking.mot.base import (
    MIN_BOX_FRACTION,
    AssociationResult,
    BaseMOTTracker,
)
from sleap_nn.tracking.mot.detections import FrameDetections
from sleap_nn.tracking.mot.params import MOTParams
from sleap_nn.tracking.mot.track import MOTTrack, TrackState

# Weight of the geometric cost added to BoT-SORT's appearance-based cost, so that
# pairs with equally good appearance are ordered by geometry.
GEOMETRY_TIE_BREAK = 1e-3

# `shared_node_iou`: fewest nodes visible on both sides of a pair, and the
# narrowest extent (px) of those nodes, for their boxes to be compared. A single
# node or coincident nodes carry no extent (the Kalman-box IoU decides); nodes
# collinear on an image axis are compared when the boxes are floored.
MIN_SHARED_NODES = 2
MIN_SHARED_EXTENT = 1.0


def _pad_nodes(kp: np.ndarray, n_nodes: int) -> np.ndarray:
    """NaN-pad ``(n, K, 2)`` keypoints to ``n_nodes`` nodes."""
    kp = np.asarray(kp, dtype=np.float64)
    if kp.shape[1] >= n_nodes:
        return kp
    out = np.full((kp.shape[0], n_nodes, 2), np.nan, dtype=np.float64)
    out[:, : kp.shape[1]] = kp
    return out


def _padded_extent(
    lo: np.ndarray,
    hi: np.ndarray,
    box_padding: float,
    min_box_size: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad ``(..., 2)`` extents like ``extract_detections`` and floor their sides."""
    pad = float(box_padding) * (hi - lo).max(axis=-1, keepdims=True)
    lo, hi = lo - pad, hi + pad
    if min_box_size is not None and np.isfinite(min_box_size) and min_box_size > 0:
        grow = 0.5 * np.maximum(float(min_box_size) - (hi - lo), 0.0)
        lo, hi = lo - grow, hi + grow
    return lo, hi


def shared_node_iou(
    track_keypoints: np.ndarray,
    det_keypoints: np.ndarray,
    *,
    box_padding: float = 0.1,
    min_box_size: Optional[float] = None,
) -> np.ndarray:
    """IoU of keypoint boxes built from the nodes visible on BOTH sides of a pair.

    For each (track, detection) pair, both boxes span only the nodes visible in
    the track's pose AND in the detection, and are padded and floored with the
    detection rules (``box_padding * max(w, h)`` per side, then at least
    ``min_box_size`` per axis). A partial (occluded) detection is thus compared
    with the same part of the track's pose instead of its full-body box.

    Args:
        track_keypoints: ``(n, K, 2)`` track poses (NaN = not visible).
        det_keypoints: ``(m, K', 2)`` detection poses (NaN = not visible).
        box_padding: Padding per side, as a fraction of ``max(w, h)``.
        min_box_size: Minimum padded side (px) per axis; ``None`` disables it.

    Returns:
        ``(n, m)`` IoU in ``[0, 1]``; NaN where fewer than
        :data:`MIN_SHARED_NODES` nodes are shared, or where the shared nodes of
        either side span less than :data:`MIN_SHARED_EXTENT` px on every axis
        (on either axis when ``min_box_size`` is ``None``: nothing then gives the
        thin axis a size).
    """
    trk = np.asarray(track_keypoints, dtype=np.float64)
    det = np.asarray(det_keypoints, dtype=np.float64)
    n, m = len(trk), len(det)
    out = np.full((n, m), np.nan, dtype=np.float64)
    if n == 0 or m == 0:
        return out
    k = max(trk.shape[1], det.shape[1])
    trk, det = _pad_nodes(trk, k), _pad_nodes(det, k)

    # (n, m, K): node visible on both sides of the pair.
    shared = np.isfinite(trk).all(-1)[:, None, :] & np.isfinite(det).all(-1)[None]
    sel = shared[..., None]
    lo_t = np.where(sel, trk[:, None], np.inf).min(axis=2)  # (n, m, 2)
    hi_t = np.where(sel, trk[:, None], -np.inf).max(axis=2)
    lo_d = np.where(sel, det[None], np.inf).min(axis=2)
    hi_d = np.where(sel, det[None], -np.inf).max(axis=2)
    floored = min_box_size is not None and np.isfinite(min_box_size)
    floored = floored and min_box_size > 0
    with np.errstate(invalid="ignore"):  # inf - inf where nothing is shared
        wide_t = (hi_t - lo_t) >= MIN_SHARED_EXTENT
        wide_d = (hi_d - lo_d) >= MIN_SHARED_EXTENT
    if floored:
        wide_t, wide_d = wide_t.any(axis=-1), wide_d.any(axis=-1)
    else:
        wide_t, wide_d = wide_t.all(axis=-1), wide_d.all(axis=-1)
    ok = (shared.sum(axis=-1) >= MIN_SHARED_NODES) & wide_t & wide_d
    if not ok.any():
        return out

    lo_t, hi_t, lo_d, hi_d = lo_t[ok], hi_t[ok], lo_d[ok], hi_d[ok]
    lo_t, hi_t = _padded_extent(lo_t, hi_t, box_padding, min_box_size)
    lo_d, hi_d = _padded_extent(lo_d, hi_d, box_padding, min_box_size)
    overlap = np.clip(np.minimum(hi_t, hi_d) - np.maximum(lo_t, lo_d), 0.0, None)
    inter = overlap.prod(axis=-1)
    union = (hi_t - lo_t).prod(axis=-1) + (hi_d - lo_d).prod(axis=-1) - inter
    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 0)
    out[ok] = np.clip(iou, 0.0, 1.0)
    return out


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

    def _score_cuts(self) -> List[Tuple[str, bool, str]]:
        """Base cuts, plus the STRICT high-score split.

        Only ``s > track_high_threshold`` (``D_hi``) may start a track.
        """
        return super()._score_cuts() + [
            (
                "track_high_threshold",
                True,
                "only detections scoring above it can start a track, so most "
                "detections can never start one. Lower --track_high_threshold",
            )
        ]

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
    def _geometric_similarity(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections
    ) -> np.ndarray:
        """Base similarity; pose IoU also scores the shared visible nodes.

        For the ``iou`` similarity on the pose carrier, each pair scores the
        larger of the Kalman-box IoU and :func:`shared_node_iou` of the track's
        predicted pose (last pose shifted by the Kalman center's motion) and the
        detection, so a partly occluded detection is compared with the same part
        of the track (see the module docstring). Other kinds are unchanged.

        Args:
            tracks: Live tracks (rows), already predicted to this step.
            dets: This frame's detections.

        Returns:
            ``(len(tracks), len(dets))`` similarity in ``[0, 1]``.
        """
        tracks = list(tracks)
        sim = super()._geometric_similarity(tracks, dets)
        if (
            sim.size == 0
            or self.carrier != "pose"
            or dets.keypoints is None
            or self._similarity(dets) != "iou"
        ):
            return sim
        means = np.stack([t.mean for t in tracks])
        predicted, _ = self._predicted_poses(tracks, means)
        shared = shared_node_iou(
            predicted,
            dets.keypoints,
            box_padding=self.params.box_padding,
            min_box_size=MIN_BOX_FRACTION * self._body_size(),
        )
        return self._finish_similarity(np.fmax(sim, shared), dets)

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

    def _duplicate_similarity(
        self,
        lost: Sequence[MOTTrack],
        cols: Sequence[int],
        sim: np.ndarray,
        rows: Dict[int, int],
        dets: FrameDetections,
    ) -> np.ndarray:
        """``(len(lost), len(cols))`` overlap compared with ``duplicate_similarity``.

        The IoU of each lost track's predicted (Kalman) box and the detection's
        padded box -- the reference test -- for the ``iou`` and ``centroid``
        similarities (single-node detections carry body-sized pseudo boxes); the
        association OKS for ``oks``.
        """
        if self._similarity(dets) == "oks":
            return sim[np.ix_([rows[t.tid] for t in lost], list(cols))]
        boxes = self.kf.xyxy_from_mean(np.stack([t.mean for t in lost]))
        return iou_matrix(boxes, np.asarray(dets.boxes, dtype=np.float64)[list(cols)])

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
        that survive this step. A pair whose overlap
        (:meth:`_duplicate_similarity`) exceeds ``duplicate_similarity`` drops the
        one with the shorter history; ties drop the tracked one (reference
        semantics). Every pair is decided independently.

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

        dup = self._duplicate_similarity(
            lost, [j for _, j, _ in tracked], sim, rows, dets
        )
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
    """BoT-SORT tracker (see :mod:`sleap_nn.tracking.mot.bytetrack`)."""

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
        self._warned_no_vectors = False

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
                "botsort: the detections carry no usable appearance vectors; "
                "associating on geometry alone until vectors appear."
            )
        return super()._associate(dets)

    def _fused_cost(
        self,
        tracks: Sequence[MOTTrack],
        det_idx: np.ndarray,
        dets: FrameDetections,
        sim: np.ndarray,
    ) -> np.ndarray:
        """Stage-1/3 cost: geometric cost, lowered by gated decisive appearance.

        Under the ``paper`` preset every in-gate cell takes the appearance cost
        (published min-fusion); otherwise only :func:`decisive_cells` with the
        calibrated margin ``m_assoc`` do.

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
        if self.params.preset == "paper":
            decisive = in_gate
        else:
            # ...and only where it discriminates: the cell is the top in-gate
            # cosine of its row and of its column, by the calibrated margin.
            decisive = decisive_cells(cos, in_gate, self._appearance_margin())
        d_emb = np.where(decisive, 0.5 * (1.0 - np.nan_to_num(cos)), 1.0)
        return np.where(
            decisive,
            np.minimum(d_geo, d_emb) + GEOMETRY_TIE_BREAK * d_geo,
            d_geo,
        )
