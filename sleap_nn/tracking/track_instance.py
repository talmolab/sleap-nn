"""TrackInstance Data structure for Tracker queue."""

from typing import List, Optional, Union
import attrs
import numpy as np
import sleap_io as sio


@attrs.define
class TrackInstances:
    """Data structure for instances in tracker queue for fixed window method."""

    src_instances: List[sio.PredictedInstance]
    features: List[np.array]
    instance_scores: List[float] = None
    track_ids: Optional[List[int]] = None
    tracking_scores: Optional[List[float]] = None
    frame_idx: Optional[float] = None
    image: Optional[np.array] = None


@attrs.define
class TrackInstanceLocalQueue:
    """Data structure for instances in tracker queue for Local Queue method."""

    src_instance: sio.PredictedInstance
    src_instance_idx: int
    feature: np.array
    instance_score: float = None
    track_id: Optional[int] = None
    tracking_score: Optional[float] = None
    frame_idx: Optional[float] = None
    image: Optional[np.array] = None


@attrs.define
class ShiftedInstance:
    """Data structure for `FlowShiftTracker`.

    Note: This data structure is only used to get the shifted points for the instances
        in the tracker queue (has an assigned track ID).
    """

    src_track_instance: Union[TrackInstances, TrackInstanceLocalQueue]
    shifted_pts: np.array
    src_instance_idx: int
    frame_idx: int
    shift_score: float
    track_id: int
