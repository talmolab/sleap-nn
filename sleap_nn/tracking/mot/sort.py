"""SORT (Bewley et al., 2016) for animal pose/mask detections.

Simple Online and Realtime Tracking: a constant-velocity Kalman filter per track
(``[cx, cy, s, r]`` state) and a single Hungarian assignment on geometric
similarity per frame. Original implementation from the published algorithm.

Algorithm (per frame, after every live track is predicted to this step):

1. Detections scoring at or below ``track_low_threshold`` are ignored entirely:
   they are never matched and never start a track (SORT itself has no score logic;
   the reference expects the caller to filter detections).
2. One pool holds every live track (tentative, confirmed and lost alike). A pair
   is admissible when its geometric similarity is at least
   ``min_match_similarity`` (0.3). If the admissible pairs already form a
   one-to-one pattern they are taken as is; otherwise a Hungarian assignment on
   ``1 - similarity`` (cost cap ``1 - min_match_similarity``) decides.
3. Lifecycle (base class, SORT-family settings): a new track starts with
   ``hits = hit_streak = 0`` (the spawn detection does not count), is confirmed
   once ``hit_streak >= min_hits`` (``min_hits=3``: four consecutive detections),
   survives misses while tentative, and every track is removed after more than
   ``track_buffer`` unmatched steps. Unmatched detections scoring at least
   ``new_track_threshold`` start tracks.
"""

import numpy as np

from sleap_nn.tracking.mot.association import linear_assignment, one_to_one_pattern
from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections


class SORTTracker(BaseMOTTracker):
    """SORT tracker (see :mod:`sleap_nn.tracking.mot.sort`)."""

    name = "sort"
    default_kalman = "xysr"
    tentative_dies_on_miss = False
    hits_start = 0
    confirm_on_streak = True

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """Single-pool, single-stage assignment on geometric similarity.

        Args:
            dets: This frame's detections.

        Returns:
            The matches; low-score detections are excluded from spawning.
        """
        p = self.params
        # Low-score detections are invisible to SORT (never matched, never spawn).
        ignored = np.asarray(dets.scores) <= p.track_low_threshold
        result = AssociationResult(no_spawn={int(j) for j in np.flatnonzero(ignored)})
        candidates = np.flatnonzero(np.asarray(dets.valid, dtype=bool) & ~ignored)
        tracks = self._live()
        if not tracks or len(candidates) == 0:
            return result

        sim = self._geometric_similarity(tracks, dets)[:, candidates]
        valid = sim >= p.min_match_similarity
        pairs = one_to_one_pattern(valid)
        if pairs is None:
            # `1 - sim` is monotone, so the admissible set is exactly `valid`.
            cost = np.where(valid, 1.0 - sim, np.inf)
            pairs, _, _ = linear_assignment(cost, max_cost=1.0 - p.min_match_similarity)
        for row, col in pairs:
            result.matches.append(
                (tracks[int(row)], int(candidates[int(col)]), float(sim[row, col]))
            )
        return result
