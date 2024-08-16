"""Module to generate Fixed window candidates."""

from typing import Optional, List, Deque, Union
from sleap_nn.tracking.track_instance import TrackInstances
import sleap_io as sio
from collections import defaultdict, deque
import attrs
import numpy as np


class FixedWindowCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` using the fixed window method, where track assignments
    are determined based on the last `window_size` frames.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
            Default: 5.
        instance_score_threshold: Instance score threshold for creating new tracks.
            Default: 0.0.
        tracker_queue: Deque object that stores the past `window_size` tracked instances.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(self, window_size: int = 5, instance_score_threshold: float = 0.0):
        """Initialize class variables."""
        self.window_size = window_size
        self.instance_score_threshold = instance_score_threshold
        self.tracker_queue = deque(maxlen=self.window_size)
        self.current_tracks = []

    def get_track_instances(
        self,
        feature_list: List[Union[np.array]],
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.array,
    ) -> TrackInstances:
        """Return an instance of `TrackInstances` object for the `untracked_instances`."""
        track_instance = TrackInstances(
            src_instances=untracked_instances,
            track_ids=[None] * len(untracked_instances),
            tracking_scores=[None] * len(untracked_instances),
            features=feature_list,
            instance_scores=[instance.score for instance in untracked_instances],
            frame_idx=frame_idx,
            image=image,
        )
        return track_instance

    def get_features_from_track_id(self, track_id: int) -> List[np.array]:
        """Return list of features for instances in queue with the given `track_id`."""
        output = []
        for t in self.tracker_queue:
            if track_id in t.track_ids:
                output.append(t.features[t.track_ids.index(track_id)])
        return output

    def get_new_track_id(self) -> int:
        """Return a new track_id."""
        if not self.current_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.current_tracks) + 1
        return new_track_id

    def add_new_tracks(
        self, current_instances: TrackInstances, add_to_queue: bool = True
    ) -> TrackInstances:
        """Add new track IDs to the `TrackInstances` object and to the tracker queue."""
        is_new_track = False
        for i, score in enumerate(current_instances.instance_scores):
            if (
                score > self.instance_score_threshold
                and current_instances.track_ids[i] is None
            ):
                is_new_track = True
                new_tracks_id = self.get_new_track_id()
                current_instances.track_ids[i] = new_tracks_id
                current_instances.tracking_scores[i] = 1.0
                self.current_tracks.append(new_tracks_id)

        if add_to_queue and is_new_track:
            self.tracker_queue.append(current_instances)

        return current_instances

    def update_tracks(
        self,
        current_instances: TrackInstances,
        row_inds: np.array,
        col_inds: np.array,
        tracking_scores: List[float],
    ) -> TrackInstances:
        """Assign tracks to `TrackInstances` based on the output of track matching algorithm.

        Args:
            current_instances: `TrackInstances` instance with features and unassigned tracks.
            row_inds: List of indices for the  `current_instances` object that has an assigned
                track.
            col_inds: List of track IDs that have been assigned a new instance.
            tracking_scores: List of tracking scores from the cost matrix.

        """
        add_to_queue = True
        if np.any(row_inds) and np.any(col_inds):

            for idx, (row, col) in enumerate(zip(row_inds, col_inds)):
                current_instances.track_ids[row] = col
                current_instances.tracking_scores[row] = tracking_scores[idx]

            # update tracks to queue
            self.tracker_queue.append(current_instances)
            add_to_queue = False

            # Create new tracks for instances with unassigned tracks from track matching
            new_current_instances_inds = [
                x for x in range(len(current_instances.features)) if x not in row_inds
            ]
            if new_current_instances_inds:
                current_instances = self.add_new_tracks(
                    current_instances, add_to_queue=add_to_queue
                )
        return current_instances
