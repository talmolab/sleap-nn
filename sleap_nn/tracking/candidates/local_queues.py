"""Module to generate Tracking local queue candidates."""

from typing import Dict, Optional, List, Deque, Union
import numpy as np
import sleap_io as sio
from sleap_nn.tracking.track_instance import TrackInstanceLocalQueue
from collections import defaultdict, deque


class LocalQueueCandidates:
    """Track local queues method for candidate generation.

    This module handles `tracker_queue` using the local queues method, where track assignments
    are determined based on the last `window_size` instances for each track.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
            Default: 5.
        max_tracks: Maximum number of new tracks that can be created. Default: None.
        instance_score_threshold: Instance score threshold for creating new tracks.
            Default: 0.0.
        tracker_queue: Dictionary that stores the past frames of all the tracks identified
            so far as `deque`.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_tracks: Optional[int] = None,
        instance_score_threshold: float = 0.0,
    ):
        """Initialize class variables."""
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.instance_score_threshold = instance_score_threshold
        self.tracker_queue = defaultdict(Deque)
        self.current_tracks = []

    def get_track_instances(
        self,
        feature_list: List[Union[np.array]],
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.array,
    ) -> List[TrackInstanceLocalQueue]:
        """Return a list of `TrackInstanceLocalQueue` instances from `untracked_instances`."""
        track_instances = []
        for ind, (feat, instance) in enumerate(zip(feature_list, untracked_instances)):
            track_instance = TrackInstanceLocalQueue(
                src_instance=instance,
                src_instance_idx=ind,
                track_id=None,
                feature=feat,
                instance_score=instance.score,
                frame_idx=frame_idx,
                image=image,
            )
            track_instances.append(track_instance)
        return track_instances

    def get_features_from_track_id(self, track_id: int) -> List[np.array]:
        """Return list of features for instances in queue with the given `track_id`."""
        return [t.feature for t in self.tracker_queue[track_id]]

    def get_new_track_id(self) -> int:
        """Return a new track_id."""
        if not self.current_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.current_tracks) + 1
            if new_track_id > self.max_tracks:  # TODO
                raise Exception("Exceeding max tracks")
        self.tracker_queue[new_track_id] = deque(maxlen=self.window_size)
        return new_track_id

    def add_new_tracks(
        self, new_track_instances: List[TrackInstanceLocalQueue]
    ) -> List[TrackInstanceLocalQueue]:
        """Add new track IDs to the `TrackInstanceLocalQueue` objects and to the tracker queue."""
        track_instances = []
        for t in new_track_instances:
            if t.instance_score > self.instance_score_threshold:
                new_track_id = self.get_new_track_id()
                t.track_id = new_track_id
                self.current_tracks.append(new_track_id)
                self.tracker_queue[new_track_id].append(t)
            track_instances.append(t)

        return track_instances

    def update_tracks(
        self,
        track_instances: List[TrackInstanceLocalQueue],
        row_inds: np.array,
        col_inds: np.array,
        tracking_scores: List[float],
    ) -> List[TrackInstanceLocalQueue]:
        """Assign tracks to `TrackInstanceLocalQueue` objects based on the output of track matching algorithm.

        Args:
            track_instances: List of TrackInstanceLocalQueue objects with features.
            row_inds: List of indices for the  `track_instances` object that has an assigned
                track.
            col_inds: List of track IDs that have been assigned a new instance.
            tracking_scores: List of tracking scores from the cost matrix.

        """
        if np.any(row_inds) and np.any(col_inds):
            for idx, (row, col) in enumerate(zip(row_inds, col_inds)):
                track_instances[row].track_id = col
                track_instances[row].tracking_score = tracking_scores[idx]

            for track_instance in track_instances:
                if track_instance.track_id is not None:
                    self.tracker_queue[track_instance.track_id].append(track_instance)

            # Create new tracks for instances with unassigned tracks from track matching
            new_track_instances_inds = [
                x for x in range(len(track_instances)) if x not in row_inds
            ]
            if new_track_instances_inds:
                for ind in new_track_instances_inds:
                    if (
                        track_instances[ind].instance_score
                        > self.instance_score_threshold
                    ):
                        self.add_new_tracks(track_instances[ind])

        return track_instances

    def get_instances_groupby_frame_idx(
        self,
    ) -> Dict[int, List[TrackInstanceLocalQueue]]:
        """Return dictionary with list of `TrackInstanceLocalQueue` objects grouped by frame index."""
        instances_dict = defaultdict(list)
        for track_id, instances in self.tracker_queue.items():
            for instance in instances:
                instances_dict[track_id].append(instance)
        return instances_dict
