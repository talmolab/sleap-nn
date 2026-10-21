"""ByteTrack (Zhang et al., 2022) and BoT-SORT (Aharon et al., 2022).

ByteTrack associates every detection box: high-score detections first, then the
remaining tracked tracks against LOW-score detections (occluded animals), then
unconfirmed tracks. BoT-SORT adds a ``[cx, cy, w, h]`` Kalman filter and fuses
re-ID appearance into the first stages -- here only inside the geometric gate and
only where appearance discriminates. Original implementations from the published
algorithms.
"""

from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections


class ByteTracker(BaseMOTTracker):
    """ByteTrack tracker (see :mod:`sleap_nn.tracking.mot.bytetrack`)."""

    name = "bytetrack"
    default_kalman = "xyah"
    tentative_dies_on_miss = True
    hits_start = 1
    lost_zero_velocity = (7,)

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        raise NotImplementedError


class BoTSORTTracker(ByteTracker):
    """BoT-SORT tracker (see :mod:`sleap_nn.tracking.mot.bytetrack`)."""

    name = "botsort"
    default_kalman = "xywh"
    uses_appearance = True
    lost_zero_velocity = (6, 7)
