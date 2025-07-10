"""TrackInstance Data structure for Tracker queue."""

from typing import List, Optional
import attrs
import numpy as np
import sleap_io as sio


@attrs.define
class TrackInstances:
    """Data structure for instances in tracker queue for fixed window method."""

    src_instances: List[sio.PredictedInstance]
    features: List[np.array]
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
    track_id: Optional[int] = None
    tracking_score: Optional[float] = None
    frame_idx: Optional[float] = None
    image: Optional[np.array] = None


@attrs.define
class TrackedInstanceFeature:
    """Data structure for tracked instances.

    This data structure is used for updating the previous tracked instances and get the
    features of the tracked instances. `shifted_keypoints` is used only for the `FlowShiftTracker`
    to store the optical flow shifted instances.
    """

    feature: np.ndarray
    src_predicted_instance: sio.PredictedInstance
    frame_idx: int
    tracking_score: float
    shifted_keypoints: np.ndarray = None
