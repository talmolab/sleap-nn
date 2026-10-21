"""OC-SORT (Cao et al., 2023) and Deep OC-SORT (Maggiolino et al., 2023).

Observation-centric SORT: re-updates a re-found track's Kalman state along a
virtual trajectory over the gap (ORU), adds a velocity-direction consistency
term to the association (OCM) and recovers tracks from their last observation
(OCR). Deep OC-SORT adds a score-adaptive appearance EMA and an appearance term
weighted by how discriminative it is per row/column (adaptive weighting).
Original implementations from the published algorithms.
"""

from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections


class OCSORTTracker(BaseMOTTracker):
    """OC-SORT tracker (see :mod:`sleap_nn.tracking.mot.ocsort`)."""

    name = "ocsort"
    default_kalman = "xysr"
    tentative_dies_on_miss = False
    hits_start = 0
    confirm_on_streak = True
    use_oru = True

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        raise NotImplementedError


class DeepOCSORTTracker(OCSORTTracker):
    """Deep OC-SORT tracker (see :mod:`sleap_nn.tracking.mot.ocsort`)."""

    name = "deepocsort"
    default_kalman = "xywh"
    uses_appearance = True
