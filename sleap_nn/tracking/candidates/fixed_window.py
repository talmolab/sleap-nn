"""Module to generate Fixed window candidates."""

from typing import Optional, List, Deque
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque
import attrs


class FixedWindowCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` by creating/ managing tracks. In fixed window
    approach, a window size is set for storing the number of frames. TODO

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

    def _check_queue_size(self):
        pass

    def update_candidates(self, new_instances: List[TrackInstance]):
        """Update instances to the tracker_queue with new instances with assigned tracks."""
        if not self.tracker_queue:  # if tracker_queue is empty, create new tracks
            track_id = 0
            for track_instance in new_instances:
                track_instance.track_id = track_id
                self.tracker_queue[track_id] = deque(maxlen=self.window_size)
                self.tracker_queue[track_id].append(track_instance)
                self.current_tracks.append(track_id)
                track_id += 1

        else:
            # update NaNs for empty tracks to a certain track ID
            for track_instance in new_instances:
                if track_instance.track_id not in self.current_tracks:
                    self.current_tracks.append(track_instance.track_id)
                self.tracker_queue[track_instance.track_id].append(track_instance)
