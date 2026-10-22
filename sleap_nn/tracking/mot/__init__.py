"""Modern tracking-by-detection MOT trackers for animal pose and mask detections.

Native implementations (numpy/scipy) of SORT, DeepSORT, StrongSORT, ByteTrack,
BoT-SORT, OC-SORT and Deep OC-SORT, adapted to sleap-io pose instances and
segmentation masks, plus per-video self-calibration of appearance (re-ID)
thresholds and offline identity linking (long-term object memory, closed-set
re-identification). See the tracking guide for usage.
"""

from typing import Dict, Optional, Type

from sleap_nn.tracking.mot.base import AssociationResult, BaseMOTTracker
from sleap_nn.tracking.mot.bytetrack import BoTSORTTracker, ByteTracker
from sleap_nn.tracking.mot.deepsort import DeepSORTTracker, StrongSORTTracker
from sleap_nn.tracking.mot.ocsort import DeepOCSORTTracker, OCSORTTracker
from sleap_nn.tracking.mot.params import MOT_TRACKER_NAMES, MOTParams
from sleap_nn.tracking.mot.sort import SORTTracker

# Registry: tracker name -> class.
MOT_TRACKERS: Dict[str, Type[BaseMOTTracker]] = {
    cls.name: cls
    for cls in (
        SORTTracker,
        DeepSORTTracker,
        StrongSORTTracker,
        ByteTracker,
        BoTSORTTracker,
        OCSORTTracker,
        DeepOCSORTTracker,
    )
}
assert tuple(MOT_TRACKERS) == MOT_TRACKER_NAMES

# Trackers whose association uses re-ID vectors.
APPEARANCE_TRACKERS = tuple(n for n, c in MOT_TRACKERS.items() if c.uses_appearance)
# Trackers for which appearance is the primary cue (no vectors -> error).
VECTOR_REQUIRED_TRACKERS = tuple(
    n for n, c in MOT_TRACKERS.items() if c.requires_vectors
)


def build_mot_tracker(
    params: MOTParams, carrier: str = "pose", n_nodes: Optional[int] = None
) -> BaseMOTTracker:
    """Instantiate the tracker named by ``params.tracker``.

    Args:
        params: Resolved parameters (see :meth:`MOTParams.for_tracker`).
        carrier: ``"pose"`` or ``"mask"``.
        n_nodes: Skeleton size for pose detections (``None`` if unknown).

    Returns:
        A fresh tracker for one video.
    """
    return MOT_TRACKERS[params.tracker](params, carrier=carrier, n_nodes=n_nodes)


__all__ = [
    "APPEARANCE_TRACKERS",
    "AssociationResult",
    "BaseMOTTracker",
    "BoTSORTTracker",
    "ByteTracker",
    "DeepOCSORTTracker",
    "DeepSORTTracker",
    "MOTParams",
    "MOT_TRACKERS",
    "MOT_TRACKER_NAMES",
    "OCSORTTracker",
    "SORTTracker",
    "StrongSORTTracker",
    "VECTOR_REQUIRED_TRACKERS",
    "build_mot_tracker",
]
