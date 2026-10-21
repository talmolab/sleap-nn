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

from typing import Optional, Sequence

import attrs

from sleap_nn.tracking.mot.detections import FrameDetections

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
        nn_distance: Median same-frame nearest-neighbor centroid distance (px);
            falls back to ``body_size``.
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
        raise NotImplementedError

    def summary(self) -> str:
        """One-line human-readable summary for logs."""
        raise NotImplementedError


def calibrate(
    frames: Sequence[FrameDetections],
    *,
    track_buffer: int = 30,
    gallery_size: int = 30,
    query_len: int = 5,
    ema_alpha: float = 0.8,
    max_trials: int = 2000,
    seed: int = 0,
) -> VideoCalibration:
    """Calibrate a video from all of its frames' detections.

    Args:
        frames: The video's :class:`FrameDetections`, in temporal order.
        track_buffer: Long-gap length (steps) for the linking statistic.
        gallery_size: Gallery length used in the trials.
        query_len: Query tracklet length used for the linking statistic.
        ema_alpha: EMA momentum used for the EMA statistic.
        max_trials: Cap on trials per gap (subsampled deterministically).
        seed: Seed for subsampling.

    Returns:
        The :class:`VideoCalibration`. Never raises on degenerate input (empty
        video, single animal, no vectors).
    """
    raise NotImplementedError
