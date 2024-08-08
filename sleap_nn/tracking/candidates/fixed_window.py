"""Module to generate Fixed window candidates."""

from typing import Optional, List, Deque
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque
import attrs
import numpy as np


class FixedWindowCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` using the fixed window method, where track assignments
    are determined based on the last `window_size` frames.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
        max_tracks: Maximum number of new tracks that can be created.
        tracker_queue: Deque object that stores the past `window_size` tracked instances.
        current_tracks: List of track IDs that are being tracked.
    """

    def __init__(self, window_size: int, max_tracks: int):
        """Initialize class variables."""
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.tracker_queue = deque(maxlen=self.window_size)
        self.current_tracks = []

    def get_new_track_id(self):
        """Return a new track_id and add it to the current_tracks."""
        if not self.current_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.current_tracks) + 1
            if new_track_id > self.max_tracks:
                raise Exception("Exceeding max tracks")
        self.current_tracks.append(new_track_id)
        return new_track_id

    def get_instances_from_track_id(self, track_id: int):
        """Return list of `TrackInstance` objects with the given `track_id`."""
        output = []
        for t in self.tracker_queue:
            if t.track_id == track_id:
                output.append(t)
        return output

    def update_candidates(self, new_instances: List[TrackInstance]):
        """Update new instances with assigned tracks to the tracker_queue.

        Args:
            new_instances: List of `TrackInstance` objects with assigned track IDs.
                The instances is not updated to the `tracker_queue` when `track_id` is
                `None`.

        """
        for track_instance in new_instances:
            if track_instance.track_id is not None:
                # if track_instance.track_id not in self.current_tracks: Do we need this?
                #     self.current_tracks.append(track_instance.track_id)
                self.tracker_queue.append(track_instance)
