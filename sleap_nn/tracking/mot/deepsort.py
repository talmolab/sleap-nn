"""DeepSORT (Wojke et al., 2017) and StrongSORT (Du et al., 2023).

DeepSORT adds appearance to SORT: a matching cascade over confirmed and lost
tracks by time since update, scored by the minimum cosine distance to each
track's gallery of re-ID vectors and gated by Kalman (Mahalanobis) distance,
followed by a geometric stage for tentative and just-lost tracks. StrongSORT
replaces the gallery with an EMA feature, the cascade with one global
assignment, blends motion into the cost, and scales the measurement noise by
detection confidence (NSA Kalman). Original implementations from the published
algorithms.
"""

from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.detections import FrameDetections


class DeepSORTTracker(BaseMOTTracker):
    """DeepSORT tracker (see :mod:`sleap_nn.tracking.mot.deepsort`)."""

    name = "deepsort"
    default_kalman = "xyah"
    uses_appearance = True
    requires_vectors = True
    tentative_dies_on_miss = True
    hits_start = 1

    def _associate(self, dets: FrameDetections) -> AssociationResult:
        raise NotImplementedError


class StrongSORTTracker(DeepSORTTracker):
    """StrongSORT tracker (see :mod:`sleap_nn.tracking.mot.deepsort`)."""

    name = "strongsort"
