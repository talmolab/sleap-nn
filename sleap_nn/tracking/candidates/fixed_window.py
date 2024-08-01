"""Module to generate Fixed window candidates."""

from typing import Optional, List, Deque
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque
import attrs
import numpy as np


class FixedWindowCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` using the fixed window method, where track assignments
    are determined based on the most recent `window_size` frames.

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

    def _add_new_tracks(self, new_track_ids: List):
        """Add new tracks to the `tracker_queue`."""
        for track in new_track_ids:
            if track not in self.current_tracks:
                self.current_tracks.append(track)

    def get_instances_from_track_id(self, track_id: int):
        """Return list of `TrackInstance` objects with the given `track_id`."""
        output = []
        for t in self.tracker_queue:
            if t.track_id == track_id:
                output.append(t)
        return output

    def update_candidates(
        self, new_instances: List[TrackInstance], new_track_ids: List = None
    ):
        """Update new instances with assigned tracks to the tracker_queue.

        Args:
            new_instances: List of `TrackInstance` objects with assigned track IDs.
                The instances is not updated to the `tracker_queue` when `track_id` is
                `None`.
            new_track_ids: List of new track IDs to be created.

        """
        if new_track_ids:
            self._add_new_tracks(new_track_ids)

        for track_instance in new_instances:
            if track_instance.track_id is not None:
                self.tracker_queue.append(track_instance)
