"""Module to generate Tracking local queue candidates."""

from typing import Optional, List, Deque
import numpy as np
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque


class LocalQueueCandidates:
    """Track local queues method for candidate generation.

    This module handles `tracker_queue` using the local queues method, where track assignments
    are determined based on the last `window_instances` instances for each track.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
        max_tracks: Maximum number of new tracks that can be created.
        tracker_queue: Dictionary that stores the past frames of all the tracks identified
            so far as `deque`.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(self, window_size: int, max_tracks: int):
        """Initialize class variables."""
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.tracker_queue = defaultdict(Deque)
        self.current_tracks = []

    def get_new_track_id(self):
        """Return list of `TrackInstance` objects with the given `track_id`."""
        if not self.current_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.current_tracks) + 1
            if new_track_id > self.max_tracks:
                raise Exception("Exceeding max tracks")
        self.current_tracks.append(new_track_id)
        self.tracker_queue[new_track_id] = deque(maxlen=self.window_size)
        return new_track_id

    def get_instances_from_track_id(self, track_id: int):
        """Return list of `TrackInstance` objects with the given `track_id`."""
        return self.tracker_queue[track_id]

    def update_candidates(
        self,
        new_instances: List[TrackInstance],
    ):
        """Update new instances with assigned tracks to the tracker_queue.

        Args:
            new_instances: List of `TrackInstance` objects with assigned track IDs.
                The instances is not updated to the `tracker_queue` when `track_id` is
                `None`.
        """
        tracks_ids = []
        for track_instance in new_instances:
            if track_instance.track_id is not None:
                self.tracker_queue[track_instance.track_id].append(track_instance)
                tracks_ids.append(track_instance.track_id)

        # Append NaNs for the tracks that don't have an instance at the current iteration.
        unassigned_existing_tracks = [
            x for x in self.current_tracks if x not in tracks_ids
        ]
        if unassigned_existing_tracks:
            instance_shape = self.tracker_queue[self.current_tracks[0]][0].feature.shape
            for t in unassigned_existing_tracks:
                self.tracker_queue[t].append(
                    TrackInstance(
                        src_instance=None, feature=np.full(instance_shape, np.NaN)
                    )
                )
