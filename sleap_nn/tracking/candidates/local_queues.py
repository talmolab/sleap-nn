"""Module to generate Tracking local queue candidates."""

from typing import Optional, List
import numpy as np
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque


class LocalQueueCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` using the fixed window method, where track assignments
    are determined based on the most recent `window_size` frames.

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

    def _add_new_tracks(self, new_track_ids: List):
        """Add new tracks to the `tracker_queue` and initialize the Deque."""
        for track in new_track_ids:
            self.tracker_queue[track] = deque(maxlen=self.window_size)
            self.current_tracks.append(track)

    def update_candidates(
        self,
        new_instances: List[TrackInstance],
        unassigned_existing_tracks: List = None,
        new_track_ids: List = None,
    ):
        """Update new instances with assigned tracks to the tracker_queue.

        Args:
            new_instances: List of `TrackInstance` objects with assigned track IDs.
                The instances is not updated to the `tracker_queue` when `track_id` is
                `None`.
            unassigned_existing_tracks: List of track IDs that is not associated with
                any of the new tracked instances.
            new_track_ids: List of new track IDs to be created.

        """
        if not self.tracker_queue:  # if tracker_queue is empty, create new tracks
            self._add_new_tracks(new_track_ids)
            for track_instance in new_instances:
                if track_instance.track_id is not None:
                    self.tracker_queue[track_instance.track_id].append(track_instance)

        else:
            if new_track_ids:
                self._add_new_tracks(new_track_ids)

            for track_instance in new_instances:
                self.tracker_queue[track_instance.track_id].append(track_instance)

            # Append NaNs for the tracks that don't have an instance at the current iteration.
            if unassigned_existing_tracks:
                instance_shape = (
                    self.tracker_queue[self.current_tracks[0]][0]
                    .src_instance.numpy()
                    .shape
                )
                for t in unassigned_existing_tracks:
                    self.tracker_queue[t].append(np.fill(instance_shape), np.NaN)
