"""Module to generate Tracking local queue candidates."""

from typing import Optional, List
import numpy as np
from sleap_nn.tracking.track_instance import TrackInstance
from collections import defaultdict, deque


class LocalQueueCandidates:
    """ """

    def __init__(self, window_size: int):
        self.tracker_queue = defaultdict(deque(maxlen=window_size))
        self.current_tracks = []

    def update_candidates(self, new_instances: List[TrackInstance]):
        pass
