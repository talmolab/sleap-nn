"""SORT (Bewley et al., 2016) for animal pose/mask detections.

Simple Online and Realtime Tracking: a constant-velocity Kalman filter per track
(``[cx, cy, s, r]`` state) and a single Hungarian assignment on geometric
similarity per frame. Original implementation from the published algorithm.
"""

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
        raise NotImplementedError
