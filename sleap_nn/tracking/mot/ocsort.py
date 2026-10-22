"""OC-SORT (Cao et al., 2023) and Deep OC-SORT (Maggiolino et al., 2023).

Observation-centric SORT: re-updates a re-found track's Kalman state along a
virtual trajectory over the gap (ORU), adds a velocity-direction consistency
term to the association (OCM) and recovers tracks from their last observation
(OCR). Deep OC-SORT adds a score-adaptive appearance EMA and an appearance term
weighted by how discriminative it is per row/column (adaptive weighting).
Original implementations from the published algorithms.

OC-SORT, per frame (after every live track is predicted to this step):

1. Detections are split by score: ``D_hi`` (``s > track_high_threshold``) and,
   only when ``use_byte``, ``D_lo`` (``track_low_threshold < s <=
   track_high_threshold``). Everything else is ignored. Only ``D_hi`` detections
   start tracks.
2. **Stage 1** -- one pool of every live track (tentative, confirmed and lost
   alike, as in SORT) against ``D_hi``. A pair is admissible on GEOMETRY ALONE:
   similarity of the track's prediction ``>= min_match_similarity``. If the
   admissible pairs form a one-to-one pattern they are taken as is; otherwise
   the assignment maximizes ``similarity + OCM`` over the matrix and drops the
   inadmissible pairs (:func:`~sleap_nn.tracking.mot.association.
   linear_assignment_masked`). OCM (observation-centric momentum) rewards
   detections lying in the direction the track was moving: the angle between the
   track's direction (from its observation ``delta_t`` steps back to its last
   observation) and the direction from that earlier observation to the
   detection, weighted by ``inertia`` and the detection score.
3. **BYTE** (``use_byte`` only) -- the remaining tracks' predictions against
   ``D_lo``, admissible at ``>= min_match_similarity``.
4. **OCR** (observation-centric recovery) -- the remaining ``D_hi`` against the
   remaining tracks' LAST OBSERVATIONS (not their predictions), admissible at
   ``>= min_match_similarity``: recovers tracks whose prediction drifted while
   the animal stopped or turned.
5. Lifecycle (base class): a track's state is frozen at its first miss and,
   when it is re-matched, re-updated along a virtual trajectory linearly
   interpolated between its last observation and the new one (ORU) before the
   real update. SORT-family counters (``hits`` start at 0, confirmation on
   ``hit_streak >= min_hits``, tentative tracks coast) and removal after more
   than ``track_buffer`` unmatched steps.

Deep OC-SORT is OC-SORT (``D_hi`` only, no BYTE stage) on a ``[cx, cy, w, h]``
Kalman filter whose size velocities are zeroed while a track is frozen, with:

- a score-adaptive appearance EMA (base class memory, ``adaptive_ema``: a
  detection at the score floor ``track_high_threshold`` does not move the EMA),
  updated under the appearance quality gate;
- a STANDARDIZED appearance term in the stage-1 objective, ``similarity + OCM +
  W * E'`` with ``E' = clip(2 (cos - mu_neg) / (mu_pos - mu_neg) - 1, -1, 1)``
  (``cos``: cosine of the detection vector to the track's EMA; ``mu_pos`` /
  ``mu_neg``: the video's calibrated median true / competitor EMA cosines), so
  the term means the same thing whatever the embedding model's cosine scale.
  When appearance is enabled for association but the video has no usable
  ``mu_pos`` / ``mu_neg`` spread -- an explicit ``appearance_threshold`` on a
  video that is uncalibrated (no ``prepare()``) or too short / with too few
  animals to calibrate -- ``E'`` is replaced by the raw EMA cosine, the
  reference's term (logged once; ``resolved["appearance"]["association_term"]``
  records which term ran). The term is 0 (neutral) where either side has no
  vector, where the pair has no geometric overlap, and everywhere when
  appearance is disabled. ``W`` is the adaptive weight (``w_assoc_emb`` plus
  half the row and column top-1/top-2 margins of ``E'``, each capped at
  ``aw_param``) or the constant ``w_assoc_emb`` with ``aw_off``. Appearance
  orders COMPETING candidates but never vetoes an uncontested one: on a pair
  that is the only admissible candidate of both its row and its column a
  negative ``W * E'`` is clamped to 0 (otherwise the masked assignment trades
  it for an inadmissible, zero-evidence cell whenever an unrelated ambiguity
  elsewhere in the frame stops the one-to-one shortcut);
- an optional appearance **rescue** (``rescue``): a pair below
  ``min_match_similarity`` but at or above ``rescue_min_similarity`` is also
  admissible when appearance is DECISIVE: its EMA cosine clears the appearance
  gate and is the top of its row and column among the geometrically plausible
  pairs by at least the calibrated margin ``m_assoc``. The one-to-one shortcut is
  evaluated on this extended admissibility.

The OCR stage is geometric only, as in the reference. ``tracking_score`` is
always the geometric similarity of the matched pair (prediction or last
observation), never the composite objective.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from sleap_nn.tracking.mot.association import (
    adaptive_weight,
    appearance_margins,
    linear_assignment_masked,
    ocm_matrix,
    one_to_one_pattern,
)
from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections
from sleap_nn.tracking.mot.params import MOTParams
from sleap_nn.tracking.mot.track import MOTTrack
from sleap_nn.tracking.utils import get_embedding

# Smallest calibrated (mu_pos - mu_neg) spread that can standardize cosines.
_MIN_APPEARANCE_SPREAD = 1e-6

# XYWH state indices of the size velocities (v_w, v_h), zeroed while frozen.
_XYWH_SIZE_VELOCITY = (6, 7)

# The appearance term Deep OC-SORT adds to its stage-1 objective (provenance).
_TERM_STANDARDIZED = "standardized"  # E' from the calibrated mu_pos / mu_neg.
_TERM_RAW_COSINE = "raw_cosine"  # The reference's raw EMA cosine (no spread).
_TERM_OFF = "off"  # Appearance disabled for association.


class OCSORTTracker(BaseMOTTracker):
    """OC-SORT tracker (see :mod:`sleap_nn.tracking.mot.ocsort`)."""

    name = "ocsort"
    default_kalman = "xysr"
    tentative_dies_on_miss = False
    hits_start = 0
    confirm_on_streak = True
    use_oru = True
    # OCM reads `velocity` / `observations` (the base keeps them on request).
    keeps_observation_history = True

    # Whether the optional BYTE (low-score) stage is available to this family.
    _supports_byte = True

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """Stage 1 (geometry-gated, OCM-ordered), optional BYTE, then OCR.

        Args:
            dets: This frame's detections.

        Returns:
            The matches (each with its geometric similarity); every detection
            outside ``D_hi`` is excluded from spawning.
        """
        p = self.params
        scores = np.asarray(dets.scores, dtype=np.float64)
        usable = np.asarray(dets.valid, dtype=bool) & np.isfinite(scores)
        high = usable & (scores > p.track_high_threshold)
        low = (
            usable
            & (scores > p.track_low_threshold)
            & (scores <= p.track_high_threshold)
        )
        result = AssociationResult(
            no_spawn={int(j) for j in np.flatnonzero(~high)},
        )
        self._appearance_notice(dets)

        tracks = self._live()
        hi_idx = np.flatnonzero(high)
        if not tracks or (len(hi_idx) == 0 and not self._byte_enabled()):
            return result
        sim_all = self._geometric_similarity(tracks, dets)

        # Stage 1: every live track vs D_hi.
        free_rows = np.arange(len(tracks))
        free_hi = hi_idx
        if len(hi_idx):
            sim = sim_all[:, hi_idx]
            valid, score = self._first_stage(tracks, dets, hi_idx, sim)
            pairs = one_to_one_pattern(valid)
            if pairs is None:
                pairs, _, _ = linear_assignment_masked(score, valid)
            free_rows, free_hi = self._record(
                result, tracks, free_rows, hi_idx, pairs, sim
            )

        # BYTE: remaining tracks' predictions vs D_lo (before OCR).
        lo_idx = np.flatnonzero(low)
        if self._byte_enabled() and len(free_rows) and len(lo_idx):
            sim = sim_all[np.ix_(free_rows, lo_idx)]
            pairs, _, _ = linear_assignment_masked(sim, sim >= p.min_match_similarity)
            free_rows, _ = self._record(result, tracks, free_rows, lo_idx, pairs, sim)

        # OCR: remaining D_hi vs remaining tracks' last observations.
        if len(free_rows) and len(free_hi):
            remaining = [tracks[int(i)] for i in free_rows]
            sim = self._last_observation_similarity(remaining, dets)[:, free_hi]
            pairs, _, _ = linear_assignment_masked(sim, sim >= p.min_match_similarity)
            self._record(result, tracks, free_rows, free_hi, pairs, sim)
        return result

    # ------------------------------------------------------------ stages
    def _byte_enabled(self) -> bool:
        """Whether the BYTE (low-score) stage runs."""
        return self._supports_byte and bool(self.params.use_byte)

    def _first_stage(
        self,
        tracks: Sequence[MOTTrack],
        dets: FrameDetections,
        cols: np.ndarray,
        sim: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Admissibility and objective of stage 1.

        Args:
            tracks: The live tracks (rows).
            dets: This frame's detections.
            cols: Detection indices of the columns (``D_hi``).
            sim: ``(n_tracks, len(cols))`` geometric similarity of the
                predictions.

        Returns:
            ``(valid, score)``: admissibility on geometry alone and the
            objective ``sim + OCM`` the assignment maximizes.
        """
        valid = sim >= self.params.min_match_similarity
        return valid, sim + self._momentum(tracks, dets, cols)

    def _momentum(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections, cols: np.ndarray
    ) -> np.ndarray:
        """``(n_tracks, len(cols))`` observation-centric momentum (OCM) term.

        A track contributes only once it has a velocity direction (two
        observations) and an earlier observation to measure the direction to
        each detection from.
        """
        n = len(tracks)
        dirs = np.zeros((n, 2), dtype=np.float64)
        prev = np.full((n, 4), np.nan, dtype=np.float64)
        has_dir = np.zeros(n, dtype=bool)
        for i, track in enumerate(tracks):
            if track.velocity is None:
                continue
            obs = self._previous_observation(track)
            if obs is None:
                continue
            dirs[i] = np.asarray(track.velocity, dtype=np.float64).reshape(2)
            prev[i] = np.asarray(obs, dtype=np.float64).reshape(4)
            has_dir[i] = True
        return ocm_matrix(
            dirs,
            prev,
            has_dir,
            np.asarray(dets.boxes)[cols],
            np.asarray(dets.scores)[cols],
            self.params.inertia,
        )

    def _previous_observation(self, track: MOTTrack) -> Optional[np.ndarray]:
        """The track's observation ``delta_t`` steps back, for the OCM direction.

        The oldest observation within the last ``delta_t`` steps, else the latest
        observation (``None`` if the track has none), as in the reference.
        """
        for dt in range(self.params.delta_t, 0, -1):
            obs = track.observations.get(self._step - dt)
            if obs is not None:
                return obs
        return track.last_observation

    @staticmethod
    def _record(
        result: AssociationResult,
        tracks: Sequence[MOTTrack],
        rows: np.ndarray,
        cols: np.ndarray,
        pairs: np.ndarray,
        sim: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append a stage's matches and return the still-unmatched rows/columns.

        Args:
            result: The frame's association result (appended to).
            tracks: Every live track; ``rows`` index into it.
            rows: Track indices of the stage's rows.
            cols: Detection indices of the stage's columns.
            pairs: ``(k, 2)`` matched ``(row, col)`` positions in the stage.
            sim: The stage's geometric similarity (the tracking score).

        Returns:
            ``(free_rows, free_cols)``: the unmatched track and detection indices.
        """
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        for r, c in pairs:
            result.matches.append(
                (tracks[int(rows[r])], int(cols[c]), float(sim[r, c]))
            )
        free_rows = np.setdiff1d(np.arange(len(rows)), pairs[:, 0])
        free_cols = np.setdiff1d(np.arange(len(cols)), pairs[:, 1])
        return np.asarray(rows)[free_rows], np.asarray(cols)[free_cols]

    def _appearance_notice(self, dets: FrameDetections) -> None:
        """Hook for appearance trackers to log once (OC-SORT: nothing)."""


class DeepOCSORTTracker(OCSORTTracker):
    """Deep OC-SORT tracker (see :mod:`sleap_nn.tracking.mot.ocsort`)."""

    name = "deepocsort"
    default_kalman = "xywh"
    uses_appearance = True

    # Deep OC-SORT associates high-score detections only (no BYTE stage).
    _supports_byte = False

    def __init__(
        self,
        params: MOTParams,
        carrier: str = "pose",
        n_nodes: Optional[int] = None,
    ):
        """Create a tracker for one video.

        Args:
            params: Resolved parameters (see :meth:`MOTParams.for_tracker`).
            carrier: ``"pose"`` or ``"mask"``.
            n_nodes: Skeleton size for pose detections (``None`` if unknown; then
                taken from the detections).

        Raises:
            ValueError: On an unknown carrier, or parameters resolved for another
                tracker.
        """
        super().__init__(params, carrier=carrier, n_nodes=n_nodes)
        # The reference stops the box-size velocities of a frozen (missed) track
        # on its [cx, cy, w, h] filter; other filters keep their velocities.
        self.lost_zero_velocity = (
            _XYWH_SIZE_VELOCITY if self.kalman_kind == "xywh" else ()
        )
        if params.use_byte:
            logger.warning(
                "deepocsort has no BYTE (low-score) stage; use_byte is ignored."
            )
        self._appearance_noticed = False
        self._raw_term_noticed = False

    def prepare(self, frames: Sequence[Tuple[int, Sequence[Any]]]) -> None:
        """Per-video pre-pass (see :meth:`BaseMOTTracker.prepare`).

        Without any appearance vector Deep OC-SORT is OC-SORT on a ``[cx, cy, w,
        h]`` filter: with ``appearance_threshold="auto"`` this is logged and the
        video is tracked on geometry alone; an explicit (float)
        ``appearance_threshold`` asks for appearance that does not exist and
        raises. An explicit threshold on a video whose calibration cannot
        standardize the cosines (too few trials) logs that the raw EMA cosine
        is used instead of ``E'``.

        Args:
            frames: ``(frame_idx, objects)`` per frame, in temporal order, holding
                exactly the objects ``track()`` will receive.

        Raises:
            ValueError: An explicit ``appearance_threshold`` was given but no
                detection carries an appearance vector.
        """
        frames = [(int(fi), list(objs)) for fi, objs in frames]
        n_objects = sum(len(objs) for _, objs in frames)
        if (
            n_objects > 0
            and self.params.appearance_threshold != "auto"
            and not any(
                get_embedding(obj) is not None for _, objs in frames for obj in objs
            )
        ):
            raise ValueError(
                "deepocsort was given an explicit appearance_threshold "
                f"({self.params.appearance_threshold}) but none of the "
                f"{n_objects} detections to track carries an appearance vector. "
                "Run an embedding model first, or use --tracker ocsort (the same "
                "algorithm without appearance)."
            )
        super().prepare(frames)
        self._raw_term_notice()

    # ------------------------------------------------------------ stages
    def _first_stage(
        self,
        tracks: Sequence[MOTTrack],
        dets: FrameDetections,
        cols: np.ndarray,
        sim: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1 with the standardized appearance term and the rescue rule.

        Args:
            tracks: The live tracks (rows).
            dets: This frame's detections.
            cols: Detection indices of the columns (``D_hi``).
            sim: ``(n_tracks, len(cols))`` geometric similarity of the
                predictions.

        Returns:
            ``(valid, score)``: ``sim >= min_match_similarity`` extended by
            decisive-appearance rescues, and ``sim + OCM + W * E'`` (``E'``: the
            standardized term, or the raw EMA cosine when the video has no
            usable calibrated spread), where a negative ``W * E'`` on a pair
            that is the only admissible candidate of both its row and its
            column is clamped to 0.
        """
        valid, score = super()._first_stage(tracks, dets, cols, sim)
        cos = self._appearance_cos(tracks, dets, "ema")[:, cols]
        if not np.isfinite(cos).any():
            return valid, score  # No appearance evidence: plain OC-SORT.

        valid = valid | self._rescue(cos, sim)
        appearance = self._association_appearance(cos)
        # As in the reference, appearance only counts where geometry overlaps.
        appearance[sim <= 0.0] = 0.0
        if appearance.any():
            p = self.params
            if p.aw_off:
                weight = np.full(appearance.shape, float(p.w_assoc_emb))
            else:
                weight = adaptive_weight(
                    appearance, aw_param=p.aw_param, base=p.w_assoc_emb
                )
            term = weight * appearance
            # Appearance orders competing candidates; it must not veto an
            # uncontested one. The masked assignment maximizes over the WHOLE
            # matrix, so a lone admissible pair scored below an inadmissible,
            # zero-evidence cell (a spare false positive, a lost track) would be
            # traded for it and dropped -- but only when some unrelated pair in
            # the frame stops the one-to-one shortcut. Such a pair keeps at least
            # its OC-SORT score.
            lone = (
                valid
                & (valid.sum(axis=1, keepdims=True) == 1)
                & (valid.sum(axis=0, keepdims=True) == 1)
            )
            score = score + np.where(lone, np.maximum(term, 0.0), term)
        return valid, score

    def _association_term(self) -> str:
        """Which appearance term stage 1 adds to its objective.

        Returns:
            ``"off"`` when appearance is disabled for association (no vectors,
            or ``"auto"`` without a calibration enabling it); ``"standardized"``
            (``E'``) when the calibration has a usable ``mu_pos`` / ``mu_neg``
            spread; otherwise ``"raw_cosine"``: the reference's raw EMA cosine
            (e.g. an explicit ``appearance_threshold`` on a video that is
            uncalibrated or has too few trials to calibrate).
        """
        if self._appearance_gate("ema") is None:
            return _TERM_OFF
        if self._appearance_spread() is None:
            return _TERM_RAW_COSINE
        return _TERM_STANDARDIZED

    def _association_appearance(self, cos: np.ndarray) -> np.ndarray:
        """The appearance values of the stage-1 term (see :meth:`_association_term`).

        Args:
            cos: ``(n, m)`` EMA cosines, NaN where unavailable.

        Returns:
            ``(n, m)`` values in ``[-1, 1]``: the standardized ``E'``, or the raw
            cosine without a usable calibrated spread; 0 where a cosine is
            missing and everywhere when appearance is off.
        """
        term = self._association_term()
        if term == _TERM_STANDARDIZED:
            return self._standardized_appearance(cos)
        out = np.zeros(cos.shape, dtype=np.float64)
        if term == _TERM_RAW_COSINE:
            ok = np.isfinite(cos)
            out[ok] = np.clip(cos[ok], -1.0, 1.0)
        return out

    def _standardized_appearance(self, cos: np.ndarray) -> np.ndarray:
        """``E' = clip(2 (cos - mu_neg) / (mu_pos - mu_neg) - 1, -1, 1)``.

        ``mu_pos`` / ``mu_neg`` are the video's calibrated median true and
        competitor EMA cosines (one step apart). Missing cosines (no vector on
        either side) give 0, the neutral value; so does every pair when the video
        has no usable calibration (uncalibrated, or a degenerate spread).

        Args:
            cos: ``(n, m)`` EMA cosines, NaN where unavailable.

        Returns:
            ``(n, m)`` standardized appearance in ``[-1, 1]``.
        """
        out = np.zeros(cos.shape, dtype=np.float64)
        spread = self._appearance_spread()
        if spread is None:
            return out
        mu_neg, width = spread
        ok = np.isfinite(cos)
        out[ok] = np.clip(2.0 * (cos[ok] - mu_neg) / width - 1.0, -1.0, 1.0)
        return out

    def _appearance_spread(self) -> Optional[Tuple[float, float]]:
        """``(mu_neg, mu_pos - mu_neg)`` from the calibration, or ``None``."""
        cal = self.calibration
        if cal is None or cal.mu_pos is None or cal.mu_neg is None:
            return None
        mu_pos, mu_neg = float(cal.mu_pos), float(cal.mu_neg)
        if not (np.isfinite(mu_pos) and np.isfinite(mu_neg)):
            return None
        if mu_pos - mu_neg < _MIN_APPEARANCE_SPREAD:
            return None
        return mu_neg, mu_pos - mu_neg

    def _rescue(self, cos: np.ndarray, sim: np.ndarray) -> np.ndarray:
        """Pairs admissible through decisive appearance despite weak geometry.

        A pair is rescued when ``rescue`` is on, its geometric similarity is at
        least ``rescue_min_similarity``, its EMA cosine clears the appearance gate
        (``"auto"``: the calibrated EMA gate; a float: that value) and it is the
        strict top of its row and of its column -- among the pairs geometrically
        plausible enough to be matched at all -- by at least the calibrated
        margin ``m_assoc``.

        Args:
            cos: ``(n, m)`` EMA cosines (NaN where unavailable).
            sim: ``(n, m)`` geometric similarities.

        Returns:
            ``(n, m)`` bool mask of rescued pairs (all False when disabled).
        """
        p = self.params
        rescued = np.zeros(cos.shape, dtype=bool)
        gate = self._appearance_gate("ema")
        if not p.rescue or gate is None or cos.size == 0:
            return rescued
        floor = min(p.rescue_min_similarity, p.min_match_similarity)
        plausible = (sim >= floor) & np.isfinite(cos)
        if not plausible.any():
            return rescued
        ranked = np.where(plausible, cos, np.nan)
        row_margin, col_margin = appearance_margins(ranked)
        filled = np.where(plausible, cos, -np.inf)
        row_top = filled.max(axis=1)
        col_top = filled.max(axis=0)
        margin = max(self._appearance_margin(), 0.0)
        decisive_row = np.nan_to_num(row_margin, nan=0.0) > 0.0
        decisive_row &= np.nan_to_num(row_margin, nan=0.0) >= margin
        decisive_col = np.nan_to_num(col_margin, nan=0.0) > 0.0
        decisive_col &= np.nan_to_num(col_margin, nan=0.0) >= margin
        rescued = (
            plausible
            & (sim >= p.rescue_min_similarity)
            & (filled >= gate)
            & (filled >= row_top[:, None])
            & (filled >= col_top[None, :])
            & decisive_row[:, None]
            & decisive_col[None, :]
        )
        return rescued

    def _appearance_notice(self, dets: FrameDetections) -> None:
        """Log once how a run associates when the standardized term is unavailable.

        Covers the two prepare-less cases where appearance is unavailable: no
        appearance vectors (so far), or ``appearance_threshold="auto"`` without a
        calibration (with ``prepare()`` the base class reports the decision);
        and, once, any run whose stage-1 term falls back to the raw cosine.
        """
        if len(dets) == 0:
            return
        if not self._appearance_noticed and self.calibration is None:
            self._appearance_noticed = True
            if not self._vectors_seen:
                logger.warning(
                    "deepocsort: the tracked detections carry no appearance "
                    "vectors; associating on geometry alone (OC-SORT on a [cx, cy, "
                    "w, h] filter)."
                )
            elif self._appearance_gate("ema") is None:
                logger.warning(
                    "deepocsort: appearance_threshold='auto' needs the per-video "
                    "calibration of prepare(); without it appearance is disabled "
                    "and association uses geometry alone. Call prepare() (the "
                    "calibrated, standardized appearance term) or pass an explicit "
                    "appearance_threshold (the reference's raw-cosine term)."
                )
        self._raw_term_notice()

    def _raw_term_notice(self) -> None:
        """Log once when stage 1 uses the raw EMA cosine instead of ``E'``."""
        if self._raw_term_noticed or self._association_term() != _TERM_RAW_COSINE:
            return
        self._raw_term_noticed = True
        logger.warning(
            f"deepocsort: {self._raw_term_reason()}, so appearance cosines cannot "
            "be standardized; the stage-1 appearance term is the raw EMA cosine "
            "(as in the reference Deep OC-SORT) instead of the standardized E'."
        )

    def _raw_term_reason(self) -> str:
        """Why the calibration gives no usable ``mu_pos`` / ``mu_neg`` spread."""
        cal = self.calibration
        if cal is None:
            return "the video is uncalibrated (prepare() was not called)"
        detail = f"status={cal.status!r}, {cal.n_trials} trials"
        if cal.mu_pos is not None and cal.mu_neg is not None:
            detail += f", mu_pos={cal.mu_pos:.3g}, mu_neg={cal.mu_neg:.3g}"
        return (
            f"the video's calibration has no usable true/competitor cosine spread "
            f"({detail})"
        )

    def _appearance_summary(self) -> Dict[str, Any]:
        """JSON-safe record of the appearance decisions, with the stage-1 term.

        Adds ``association_term`` (``"standardized"``, ``"raw_cosine"`` or
        ``"off"``, see :meth:`_association_term`) to the base record and, for the
        raw-cosine fallback, why the calibrated term is unavailable.
        """
        summary = super()._appearance_summary()
        term = self._association_term()
        summary["association_term"] = term
        if term == _TERM_RAW_COSINE:
            summary["reason"] = (
                f"{summary['reason']}; {self._raw_term_reason()}: the stage-1 "
                "term is the raw EMA cosine"
            )
        return summary
