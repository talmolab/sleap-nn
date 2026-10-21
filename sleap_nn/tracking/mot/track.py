"""Track state for the MOT trackers."""

import enum
from typing import Any, Dict, List, Optional, Tuple

import attrs
import numpy as np

from sleap_nn.tracking.mot.appearance import AppearanceMemory


class TrackState(enum.IntEnum):
    """Lifecycle state of a :class:`MOTTrack`.

    ``TENTATIVE`` tracks are not yet confirmed (fewer than ``min_hits``
    matches); ``CONFIRMED`` tracks were matched at the latest step; ``LOST``
    tracks are confirmed but currently unmatched (kept for re-association for
    ``track_buffer`` steps); ``REMOVED`` tracks are finished.
    """

    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


@attrs.define(eq=False)
class MOTTrack:
    """One track of a MOT tracker.

    Attributes:
        tid: Internal id, unique within one tracker instance.
        state: Lifecycle state.
        mean: Kalman state mean.
        cov: Kalman state covariance.
        hits: Total matches (``hits_start`` at spawn).
        hit_streak: Consecutive steps with a match.
        age: Steps since spawn.
        time_since_update: Steps since the last match.
        start_frame: Frame index of the first detection.
        last_frame: Frame index of the latest matched detection.
        step: The tracker's step counter at the latest match.
        last_box: Latest observed padded ``xyxy`` box.
        last_raw_box: Latest observed unpadded extent.
        last_keypoints: Latest observed keypoints (pose carrier) or ``None``.
        kf_center_at_update: Kalman posterior center right after the latest
            update; the predicted pose is ``last_keypoints + (predicted center -
            kf_center_at_update)`` so both terms use the same reference point.
        last_score: Latest matched detection score.
        observations: OC-SORT observation history ``{step: xyxy}`` (bounded).
        last_observation: Latest observed ``xyxy`` (OC-SORT), or ``None``.
        velocity: OC-SORT unit velocity direction ``(dy, dx)`` or ``None``.
        frozen: OC-SORT state saved at the first miss ``(mean, cov, step)`` or
            ``None``.
        appearance: Appearance memory (appearance-using trackers) or ``None``.
        members: ``(frame_idx, source_object, similarity)`` of every assigned
            detection, for retroactive labeling and identity linking.
        public_id: Public id (``sio.Track`` key) once confirmed, else ``None``.
    """

    tid: int
    state: TrackState
    mean: np.ndarray
    cov: np.ndarray
    start_frame: int
    last_frame: int
    step: int
    last_box: np.ndarray
    last_raw_box: np.ndarray
    kf_center_at_update: np.ndarray
    last_keypoints: Optional[np.ndarray] = None
    last_score: float = 1.0
    hits: int = 1
    hit_streak: int = 1
    age: int = 0
    time_since_update: int = 0
    observations: Dict[int, np.ndarray] = attrs.field(factory=dict)
    last_observation: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    frozen: Optional[Tuple[np.ndarray, np.ndarray, int]] = None
    appearance: Optional[AppearanceMemory] = None
    members: List[Tuple[int, Any, float]] = attrs.field(factory=list)
    public_id: Optional[int] = None

    @property
    def is_live(self) -> bool:
        """Whether the track can still be matched (not removed)."""
        return self.state != TrackState.REMOVED

    @property
    def is_confirmed(self) -> bool:
        """Whether the track has been confirmed (``CONFIRMED`` or ``LOST``)."""
        return self.state in (TrackState.CONFIRMED, TrackState.LOST)
