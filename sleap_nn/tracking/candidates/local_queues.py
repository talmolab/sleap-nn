"""Module to generate Tracking local queue candidates."""

from typing import Dict, Optional, List, Deque, DefaultDict, Union
import numpy as np
import sleap_io as sio
from sleap_nn.tracking.track_instance import (
    TrackInstanceLocalQueue,
    TrackedInstanceFeature,
)
from collections import defaultdict, deque
from loguru import logger


class LocalQueueCandidates:
    """Track local queues method for candidate generation.

    This module handles `tracker_queue` using the local queues method, where track assignments
    are determined based on the last `window_size` instances for each track.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
            Default: 5.
        max_tracks: Maximum number of new tracks that can be created. Default: None.
        min_new_track_points: We won't spawn a new track for an instance with
            fewer than this many points. Default: 0.
        tracker_queue: Dictionary that stores the past frames of all the tracks identified
            so far as `deque`.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_tracks: Optional[int] = None,
        min_new_track_points: int = 0,
    ):
        """Initialize class variables."""
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.min_new_track_points = min_new_track_points
        self.tracker_queue = defaultdict(Deque)
        self.current_tracks = []

    def get_track_instances(
        self,
        feature_list: List[Union[np.array]],
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.array,
    ) -> List[TrackInstanceLocalQueue]:
        """Return a list of `TrackInstanceLocalQueue` instances for the `untracked_instances`."""
        track_instances = []
        for ind, (feat, instance) in enumerate(zip(feature_list, untracked_instances)):
            track_instance = TrackInstanceLocalQueue(
                src_instance=instance,
                src_instance_idx=ind,
                track_id=None,
                feature=feat,
                frame_idx=frame_idx,
                image=image,
            )
            track_instances.append(track_instance)
        return track_instances

    def get_features_from_track_id(
        self, track_id: int, candidates_list: Optional[DefaultDict[int, Deque]] = None
    ) -> List[TrackedInstanceFeature]:
        """Return list of `TrackedInstanceFeature` objects for instances in tracker queue with the given `track_id`.

        Note: If `candidates_list` is `None`, then features of all the instances in the
            tracker queue are returned by default. Else, only the features from the given
            candidates_list are returned.
        """
        tracked_instances = (
            candidates_list if candidates_list is not None else self.tracker_queue
        )
        output = []
        for t in tracked_instances[track_id]:
            tracked_instance_feature = TrackedInstanceFeature(
                feature=t.feature,
                src_predicted_instance=t.src_instance,
                frame_idx=t.frame_idx,
                tracking_score=t.tracking_score,
                shifted_keypoints=None,
            )
            output.append(tracked_instance_feature)
        return output

    def get_new_track_id(self) -> int:
        """Return a new track_id."""
        if not self.current_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.current_tracks) + 1
            if self.max_tracks is not None and new_track_id >= self.max_tracks:
                return None
        self.tracker_queue[new_track_id] = deque(maxlen=self.window_size)
        return new_track_id

    def add_new_tracks(
        self, current_instances: List[TrackInstanceLocalQueue]
    ) -> List[TrackInstanceLocalQueue]:
        """Add new track IDs to the `TrackInstanceLocalQueue` objects and to the tracker queue."""
        track_instances = []
        for t in current_instances:
            # Spawning a new track only if num visbile points is more than the threshold
            num_visible_keypoints = (
                ~np.isnan(t.src_instance.numpy()).any(axis=1)
            ).sum()
            if num_visible_keypoints > self.min_new_track_points:
                new_track_id = self.get_new_track_id()
                if new_track_id is not None:
                    t.track_id = new_track_id
                    t.tracking_score = 1.0
                    self.current_tracks.append(new_track_id)
                    self.tracker_queue[new_track_id].append(t)
                else:
                    continue
                # if new_track_id = `None`, max tracks is reached and we skip this instance
            track_instances.append(t)

        return track_instances

    def update_tracks(
        self,
        current_instances: List[TrackInstanceLocalQueue],
        row_inds: np.array,
        col_inds: np.array,
        tracking_scores: List[float],
    ) -> List[TrackInstanceLocalQueue]:
        """Assign tracks to `TrackInstanceLocalQueue` objects based on the output of track matching algorithm.

        Args:
            current_instances: List of TrackInstanceLocalQueue objects with features and unassigned tracks.
            row_inds: List of indices for the  `current_instances` object that has an assigned
                track.
            col_inds: List of track IDs that have been assigned a new instance.
            tracking_scores: List of tracking scores from the cost matrix.

        """
        res = []
        if row_inds is not None and col_inds is not None:
            for idx, (row, col) in enumerate(zip(row_inds, col_inds)):
                current_instances[row].track_id = col
                current_instances[row].tracking_score = tracking_scores[idx]
                res.append(current_instances[row])

            for track_instance in current_instances:
                if track_instance.track_id is not None:
                    self.tracker_queue[track_instance.track_id].append(track_instance)

            # Create new tracks for instances with unassigned tracks from track matching
            new_current_instances_inds = [
                x for x in range(len(current_instances)) if x not in row_inds
            ]
            if new_current_instances_inds:
                for ind in new_current_instances_inds:
                    res.extend(self.add_new_tracks([current_instances[ind]]))

        return [x for x in res if x.track_id is not None]

    def get_instances_groupby_frame_idx(
        self, candidates_list: Optional[DefaultDict[int, Deque]]
    ) -> Dict[int, List[TrackInstanceLocalQueue]]:
        """Return dictionary with list of `TrackInstanceLocalQueue` objects grouped by frame index."""
        instances_dict = defaultdict(list)
        tracked_instances = (
            candidates_list if candidates_list is not None else self.tracker_queue
        )
        for _, instances in tracked_instances.items():
            for instance in instances:
                instances_dict[instance.frame_idx].append(instance)
        return instances_dict
