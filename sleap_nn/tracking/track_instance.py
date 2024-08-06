"""TrackInstance Data structure for Tracker queue."""

from typing import Optional
import attrs
import numpy as np
import sleap_io as sio


@attrs.define
class TrackInstance:
    """Class to have a new structure for instances in tracker queue."""

    src_instance: sio.PredictedInstance
    feature: np.array
    instance_score: float = None
    track_id: Optional[int] = None
    tracking_score: Optional[float] = None
