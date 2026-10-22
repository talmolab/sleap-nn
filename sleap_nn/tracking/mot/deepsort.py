"""DeepSORT (Wojke et al., 2017) and StrongSORT (Du et al., 2023).

DeepSORT adds appearance to SORT: a matching cascade over confirmed and lost
tracks by time since update, scored by the minimum cosine distance to each
track's gallery of re-ID vectors and gated by Kalman (Mahalanobis) distance,
followed by a geometric stage for tentative and just-lost tracks. StrongSORT
replaces the gallery with an EMA feature, the cascade with one global
assignment, blends motion into the cost, and scales the measurement noise by
detection confidence (NSA Kalman). Original implementations from the published
algorithms.

DeepSORT, per frame (after every live track is predicted to this step):

1. Detections scoring at or below ``track_low_threshold`` are ignored entirely:
   never matched, never start a track (the reference's ``min_confidence``
   prefilter).
2. **Matching cascade** over the confirmed tracks (``CONFIRMED`` and ``LOST``; the
   reference keeps missed confirmed tracks in the cascade, a matched ``LOST``
   track becomes ``CONFIRMED`` again with the same id). Levels ``l = 1 ..
   track_buffer`` take the tracks with ``time_since_update == l`` (``l = 1``:
   matched at the previous step), most recently updated first, each level
   assigned to the detections the earlier levels left over. The cost of a pair is
   ``1 - max cosine`` between the detection's vector and the track's gallery
   (the EMA while the gallery is empty; a missing vector costs 1), pairs outside
   the Kalman gate cost ``INFEASIBLE_COST``, and the cost cap is ``1 - gate``
   where ``gate`` is the minimum cosine (``appearance_threshold``; ``"auto"``:
   the calibrated ``gallery_cos_gate``).
3. **Geometric stage** over the tentative tracks plus the cascade's unmatched
   tracks with ``time_since_update == 1``: cost ``1 - similarity`` (the tracker's
   geometric similarity, IoU by default), cap ``1 - min_match_similarity``
   (0.7 in IoU distance, as published; DeepSORT's only geometric threshold).
4. Lifecycle (base class, DeepSORT settings): spawned tracks start with one hit,
   are confirmed after ``min_hits`` matches, die at their first miss while
   tentative, and are removed after more than ``track_buffer`` unmatched steps.

StrongSORT changes step 2 into one global assignment over every confirmed track
with cost ``lambda * (1 - EMA cosine) + (1 - lambda) * d^2`` where ``d^2`` is
the raw squared Mahalanobis distance; gated pairs have their appearance cost set
to ``INFEASIBLE_COST`` before the blend, and the cost cap is ``(1 - gate) +
0.05`` (the published allowance for the motion term; no allowance when
``mc_lambda == 1``). Its gate is the calibrated ``ema_cos_gate`` under
``"auto"``, since the EMA cosine is the statistic it thresholds. Its Kalman
filter is NSA (``params.nsa``): the base update passes the detection score as the
confidence that scales the measurement noise.

Adaptations to animal pose/mask detections (documented deviations):

- The Kalman gate uses the box CENTER only (2 dof, chi-square 0.95 = 5.9915):
  the aspect ratio and height of a keypoint/mask box change with body rotation
  and node dropout, which the 4-dof gate would reject.
- The appearance memory is quality-gated (see
  :mod:`~sleap_nn.tracking.mot.appearance`), always seeded with the spawn vector,
  and scored on its EMA while the gallery is empty.
- A detection "has a vector" exactly when ``FrameDetections.has_embedding`` says
  so (finite, non-zero, of the video's dimension): one rule for the appearance
  stage, the memories and the "no vectors" decision. A vector that fails it
  (e.g. all NaN from an embedder that failed on a blank crop) counts as missing,
  and ``prepare()`` warns with the count.
- When appearance cannot decide -- the detections carry no usable vectors, or
  ``appearance_threshold="auto"`` and the video's calibration says the embedding
  does not separate the animals (or there was no ``prepare()``) -- the cascade
  is skipped and the geometric stage also takes every confirmed track with
  ``time_since_update <= track_buffer`` (logged once at INFO). The tracker then
  degrades to geometric matching instead of losing every track at its first
  miss. The same holds per track for a confirmed track whose memory holds no
  vector (none of its detections carried one).
- These trackers do not refuse vector-less input themselves (library loops and
  the generic tracker contract keep working); ``requires_vectors`` is what lets
  ``apply_tracking`` and the CLI reject such input up front.
"""

from typing import ClassVar, List, Sequence, Tuple

import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.appearance import AppearanceMemory
from sleap_nn.tracking.mot.association import (
    CHI2INV95,
    INFEASIBLE_COST,
    linear_assignment,
)
from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections
from sleap_nn.tracking.mot.track import MOTTrack, TrackState

# Position-only (2-dof) Mahalanobis gate: chi-square 0.95 quantile.
GATING_THRESHOLD = CHI2INV95[2]

# StrongSORT: the appearance cost cap is raised by this much when motion is
# blended into the cost (the reference's `max_cosine_distance += 0.05` under MC).
MC_COST_ALLOWANCE = 0.05


def _has_appearance(track: MOTTrack) -> bool:
    """Whether a track's appearance memory holds at least one vector."""
    return track.appearance is not None and not track.appearance.is_empty


class DeepSORTTracker(BaseMOTTracker):
    """DeepSORT tracker (see :mod:`sleap_nn.tracking.mot.deepsort`).

    Attributes:
        appearance_statistic: Appearance statistic thresholded by the
            appearance stage (``"gallery_max"``: DeepSORT's nearest neighbor in
            the gallery; ``"ema"``: StrongSORT's EMA feature). It also selects
            the calibrated gate under ``appearance_threshold="auto"``.
    """

    name = "deepsort"
    default_kalman = "xyah"
    uses_appearance = True
    requires_vectors = True
    tentative_dies_on_miss = True
    hits_start = 1

    appearance_statistic: ClassVar[str] = "gallery_max"

    # Set once the "appearance disabled" INFO message has been logged.
    _degraded_logged: bool = False

    def _score_cuts(self) -> List[Tuple[str, bool, str]]:
        """Base cuts, plus ``track_low_threshold``.

        A detection scoring at or below it is ignored (never matched, never
        starts a track).
        """
        return super()._score_cuts() + [
            (
                "track_low_threshold",
                True,
                "they are ignored (never matched, never start a track). Lower "
                "--track_low_threshold",
            )
        ]

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """Appearance stage over confirmed tracks, then a geometric stage.

        Args:
            dets: This frame's detections.

        Returns:
            The matches (with their geometric similarity); low-score detections
            are excluded from spawning.
        """
        p = self.params
        # Low-score detections are invisible (never matched, never spawn).
        ignored = np.asarray(dets.scores) <= p.track_low_threshold
        result = AssociationResult(no_spawn={int(j) for j in np.flatnonzero(ignored)})
        cols = np.flatnonzero(np.asarray(dets.valid, dtype=bool) & ~ignored)
        tracks = self._live()
        if not tracks or len(cols) == 0:
            return result

        # Geometric similarity of every live track's prediction to the candidates:
        # the geometric stage's score and every match's tracking score.
        sim = self._geometric_similarity(tracks, dets)[:, cols]
        row_of = {t.tid: i for i, t in enumerate(tracks)}
        free = np.ones(len(cols), dtype=bool)

        tentative = [t for t in tracks if t.state == TrackState.TENTATIVE]
        confirmed = [
            t for t in tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        gate = self._appearance_gate(self.appearance_statistic)
        if gate is None:
            self._log_degraded()
            pool, coasting = [], confirmed
        else:
            pool = [t for t in confirmed if _has_appearance(t)]
            coasting = [t for t in confirmed if not _has_appearance(t)]

        # Stage 1: appearance (cascade / global assignment) over confirmed tracks.
        matched = self._match_appearance(pool, dets, cols, gate, free) if pool else []
        matched_tids = {t.tid for t, _ in matched}

        # Stage 2: geometry over tentative tracks, the appearance stage's
        # unmatched tracks updated at the previous step and, when appearance
        # cannot decide for them, every coasting confirmed track.
        geo_pool = (
            tentative
            + [
                t
                for t in pool
                if t.tid not in matched_tids and t.time_since_update == 1
            ]
            + [t for t in coasting if t.time_since_update <= p.track_buffer]
        )
        geo_pool.sort(key=lambda t: t.tid)
        free_idx = np.flatnonzero(free)
        if geo_pool and len(free_idx):
            sub = sim[np.ix_([row_of[t.tid] for t in geo_pool], free_idx)]
            min_sim = p.min_match_similarity
            cost = np.where(sub >= min_sim, 1.0 - sub, np.inf)
            pairs, _, _ = linear_assignment(cost, max_cost=1.0 - min_sim)
            for r, c in pairs:
                matched.append((geo_pool[int(r)], int(free_idx[int(c)])))
                free[free_idx[int(c)]] = False

        for track, c in matched:
            result.matches.append(
                (track, int(cols[c]), float(sim[row_of[track.tid], c]))
            )
        return result

    # ------------------------------------------------------- appearance stage
    def _match_appearance(
        self,
        pool: List[MOTTrack],
        dets: FrameDetections,
        cols: np.ndarray,
        gate: float,
        free: np.ndarray,
    ) -> List[Tuple[MOTTrack, int]]:
        """DeepSORT matching cascade (most recently updated tracks first).

        Args:
            pool: Confirmed tracks with an appearance memory, in ``tid`` order.
            dets: This frame's detections.
            cols: Candidate detection indices (into ``dets``).
            gate: Minimum gallery-max cosine of a match.
            free: ``(len(cols),)`` availability of the candidates; matched
                candidates are set to False in place.

        Returns:
            ``(track, candidate position)`` per match (positions index ``cols``).
        """
        cost = self._appearance_cost(pool, dets, cols)
        max_cost = 1.0 - gate
        tsu = np.array([t.time_since_update for t in pool])
        matches: List[Tuple[MOTTrack, int]] = []
        # Levels 1..track_buffer; only the levels that hold tracks do anything.
        for level in np.unique(tsu):
            if not 1 <= level <= self.params.track_buffer:
                continue
            free_idx = np.flatnonzero(free)
            if len(free_idx) == 0:
                break
            rows = np.flatnonzero(tsu == level)
            pairs, _, _ = linear_assignment(
                cost[np.ix_(rows, free_idx)], max_cost=max_cost
            )
            for r, c in pairs:
                matches.append((pool[int(rows[int(r)])], int(free_idx[int(c)])))
                free[free_idx[int(c)]] = False
        return matches

    def _appearance_cost(
        self, pool: Sequence[MOTTrack], dets: FrameDetections, cols: np.ndarray
    ) -> np.ndarray:
        """Gated cosine distance ``1 - cos`` of ``pool`` to the candidates.

        A missing vector (either side) costs 1; a pair outside the Kalman gate
        costs ``INFEASIBLE_COST``.

        Args:
            pool: Tracks (rows).
            dets: This frame's detections.
            cols: Candidate detection indices (columns).

        Returns:
            ``(len(pool), len(cols))`` costs.
        """
        cos = self._appearance_cos(pool, dets, self.appearance_statistic)[:, cols]
        cost = np.where(np.isfinite(cos), 1.0 - cos, 1.0)
        cost[self._mahalanobis(pool, dets, cols) > GATING_THRESHOLD] = INFEASIBLE_COST
        return cost

    def _mahalanobis(
        self, pool: Sequence[MOTTrack], dets: FrameDetections, cols: np.ndarray
    ) -> np.ndarray:
        """Squared position-only Mahalanobis distances (un-scaled measurement noise).

        Args:
            pool: Tracks (rows), already predicted to this step.
            dets: This frame's detections.
            cols: Candidate detection indices (columns).

        Returns:
            ``(len(pool), len(cols))`` squared distances (``inf`` for non-finite
            boxes or states).
        """
        boxes = np.asarray(dets.boxes, dtype=np.float64)[cols]
        return self.kf.gating_distances(
            [t.mean for t in pool], [t.cov for t in pool], boxes, only_position=True
        )

    # ------------------------------------------------------------- lifecycle
    def _make_appearance_memory(self) -> AppearanceMemory:
        """A track memory whose gallery holds ``gallery_budget`` vectors (``nn_budget``)."""
        p = self.params
        return AppearanceMemory(
            ema_alpha=p.ema_alpha,
            gallery_size=p.gallery_budget,
            adaptive=p.adaptive_ema,
            score_floor=p.track_high_threshold,
        )

    def _log_degraded(self) -> None:
        """Log once that the appearance stage is off (geometric matching only)."""
        if self._degraded_logged:
            return
        self._degraded_logged = True
        reason = self._appearance_summary()["reason"]
        logger.info(
            f"{self.name}: appearance is disabled for association ({reason}); "
            "skipping the appearance stage and matching every confirmed track "
            "by geometry."
        )


class StrongSORTTracker(DeepSORTTracker):
    """StrongSORT tracker (see :mod:`sleap_nn.tracking.mot.deepsort`)."""

    name = "strongsort"
    appearance_statistic: ClassVar[str] = "ema"

    def _match_appearance(
        self,
        pool: List[MOTTrack],
        dets: FrameDetections,
        cols: np.ndarray,
        gate: float,
        free: np.ndarray,
    ) -> List[Tuple[MOTTrack, int]]:
        """One global assignment on the motion-blended (MC) cost.

        Args:
            pool: Confirmed tracks with an appearance memory, in ``tid`` order.
            dets: This frame's detections.
            cols: Candidate detection indices (into ``dets``).
            gate: Minimum EMA cosine of a match (before the MC allowance).
            free: ``(len(cols),)`` availability of the candidates; matched
                candidates are set to False in place.

        Returns:
            ``(track, candidate position)`` per match (positions index ``cols``).
        """
        free_idx = np.flatnonzero(free)
        if len(free_idx) == 0:
            return []
        lam = float(self.params.mc_lambda)
        max_cost = 1.0 - gate + (MC_COST_ALLOWANCE if lam < 1.0 else 0.0)
        cost = self._mc_cost(pool, dets, cols[free_idx])
        pairs, _, _ = linear_assignment(cost, max_cost=max_cost)
        matches = []
        for r, c in pairs:
            matches.append((pool[int(r)], int(free_idx[int(c)])))
            free[free_idx[int(c)]] = False
        return matches

    def _mc_cost(
        self, pool: Sequence[MOTTrack], dets: FrameDetections, cols: np.ndarray
    ) -> np.ndarray:
        """StrongSORT motion-blended cost ``lambda * d_app + (1 - lambda) * d^2``.

        ``d_app = 1 - EMA cosine`` (1 when a vector is missing) and ``d^2`` is the
        raw squared position-only Mahalanobis distance. Pairs with ``d^2`` above
        the gate get ``d_app = INFEASIBLE_COST`` BEFORE the blend.

        Args:
            pool: Tracks (rows).
            dets: This frame's detections.
            cols: Candidate detection indices (columns).

        Returns:
            ``(len(pool), len(cols))`` costs.
        """
        cos = self._appearance_cos(pool, dets, self.appearance_statistic)[:, cols]
        d_app = np.where(np.isfinite(cos), 1.0 - cos, 1.0)
        d2 = self._mahalanobis(pool, dets, cols)
        # A non-finite distance (invalid state) is simply out of the gate.
        d2 = np.where(np.isfinite(d2), d2, INFEASIBLE_COST)
        d_app[d2 > GATING_THRESHOLD] = INFEASIBLE_COST
        lam = float(self.params.mc_lambda)
        return lam * d_app + (1.0 - lam) * d2

    def _make_appearance_memory(self) -> AppearanceMemory:
        """The base memory (EMA statistic; the gallery only feeds the quality gate)."""
        return BaseMOTTracker._make_appearance_memory(self)
