"""Module to generate Fixed window candidates."""

from typing import Optional, List, Deque, Union
from sleap_nn.tracking.track_instance import TrackInstances, TrackedInstanceFeature
import sleap_io as sio
from collections import deque
import numpy as np


class FixedWindowCandidates:
    """Fixed-window method for candidate generation.

    This module handles `tracker_queue` using the fixed window method, where track assignments
    are determined based on the last `window_size` frames.

    Attributes:
        window_size: Number of previous frames to compare the current predicted instance with.
            Default: 5.
        min_new_track_points: We won't spawn a new track for an instance with
            fewer than this many points. Default: 0.
        tracker_queue: Deque object that stores the past `window_size` tracked instances.
        all_tracks: List of track IDs that are created.
    """

    def __init__(self, window_size: int = 5, min_new_track_points: int = 0):
        """Initialize class variables."""
        self.window_size = window_size
        self.min_new_track_points = min_new_track_points
        self.tracker_queue = deque(maxlen=self.window_size)
        self.all_tracks = []

    @property
    def current_tracks(self):
        """Get track IDs of items currently in tracker queue."""
        if not len(self.tracker_queue):
            return []
        else:
            curr_tracks = set()
            for item in self.tracker_queue:
                curr_tracks.update(item.track_ids)
            return list(curr_tracks)

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
            frame_idx=frame_idx,
            image=image,
        )
        return track_instance

    def get_features_from_track_id(
        self, track_id: int, candidates_list: Optional[Deque] = None
    ) -> List[TrackedInstanceFeature]:
        """Return list of `TrackedInstanceFeature` objects for instances in tracker queue with the given `track_id`.

        Note: If `candidates_list` is `None`, then features of all the instances in the
            tracker queue are returned by default. Else, only the features from the given
            candidates_list are returned.
        """
        output = []
        tracked_candidates = (
            candidates_list if candidates_list is not None else self.tracker_queue
        )
        for t in tracked_candidates:
            if track_id in t.track_ids:
                track_idx = t.track_ids.index(track_id)
                tracked_instance_feature = TrackedInstanceFeature(
                    feature=t.features[track_idx],
                    src_predicted_instance=t.src_instances[track_idx],
                    frame_idx=t.frame_idx,
                    tracking_score=t.tracking_scores[track_idx],
                    shifted_keypoints=None,
                )
                output.append(tracked_instance_feature)
        return output

    def get_new_track_id(self) -> int:
        """Return a new track_id."""
        if not self.all_tracks:
            new_track_id = 0
        else:
            new_track_id = max(self.all_tracks) + 1
        return new_track_id

    def add_new_tracks(
        self, current_instances: TrackInstances, add_to_queue: bool = True
    ) -> TrackInstances:
        """Add new track IDs to the `TrackInstances` object and to the tracker queue."""
        is_new_track = False
        for i, src_instance in enumerate(current_instances.src_instances):
            # Spawning a new track only if num visbile points is more than the threshold
            num_visible_keypoints = (~np.isnan(src_instance.numpy()).any(axis=1)).sum()
            if (
                num_visible_keypoints > self.min_new_track_points
                and current_instances.track_ids[i] is None
            ):
                is_new_track = True
                new_tracks_id = self.get_new_track_id()
                current_instances.track_ids[i] = new_tracks_id
                current_instances.tracking_scores[i] = 1.0
                self.all_tracks.append(new_tracks_id)

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
        if row_inds is not None and col_inds is not None:
            for idx, (row, col) in enumerate(zip(row_inds, col_inds)):
                current_instances.track_ids[row] = self.current_tracks[col]
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
