"""Module for tracking."""

from typing import List, Optional, Union
import attrs
import numpy as np
from scipy.optimize import linear_sum_assignment

import sleap_io as sio
from sleap_nn.evaluation import compute_oks
from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates
from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates
from sleap_nn.tracking.track_instance import TrackInstance


@attrs.define
class Tracker:
    """Pose Tracker.

    This module handles tracking instances across frames by creating new track IDs (or)
    assigning track IDs to each instance when the `.track()` is called. This class is
    initialized in the `Predictor` classes.

    Attributes:
        instance_score_threshold: Instance score threshold for creating new tracks.
            Default: 0.5.
        candidates: Either `FixedWindowCandidates` or `LocalQueueCandidates` object.
        use_visual_features: Boolean to indicate whether visual features should be used
            for comparing instances. Default: False.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`].

    """

    candidates: Union[FixedWindowCandidates, LocalQueueCandidates]
    instance_score_threshold: float = 0.0
    use_visual_features: bool = False
    scoring_method: str = "oks"

    @classmethod
    def from_config(
        cls,
        max_tracks: int = 30,
        window_size: int = 10,
        instance_score_threshold: float = 0.5,
        candidates_method: str = "fixed_window",
        use_visual_features: bool = False,
        scoring_method: str = "oks",
    ):
        """Create `Tracker` from config."""
        if candidates_method == "fixed_window":
            candidates = FixedWindowCandidates(
                window_size=window_size, max_tracks=max_tracks
            )
        elif candidates_method == "local_queues":
            candidates = LocalQueueCandidates
        else:
            raise ValueError(
                f"{candidates_method} is not a valid method. Please choose one of [`fixed_window`, `local_queues`]"
            )

        tracker = cls(
            instance_score_threshold=instance_score_threshold,
            candidates=candidates,
            use_visual_features=use_visual_features,
            scoring_method=scoring_method,
        )
        return tracker

    def track(self, untracked_instances: List[sio.PredictedInstance]):
        """Assign track IDs to the untracked list of `sio.PredictedInstance` objects.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.

        Returns:
            List of `sio.PredictedInstance` objects that have an assigned track ID.
        """
        track_instances = self._get_features(untracked_instances)  # get features

        if self.candidates.tracker_queue:

            # scoring function
            scores = self._get_scores(track_instances)

            # track assignment
            cost_matrix = self._scores_to_cost_matrix(scores)
            track_instances, new_track_ids = self._assign_tracks(
                track_instances, cost_matrix
            )

        else:
            # Assign new tracks for instances.
            track_id = 0
            new_track_ids = []
            for t in track_instances:
                if t.instance_score > self.instance_score_threshold:
                    t.track_id = track_id
                    new_track_ids.append(track_id)
                    track_id = track_id + 1
            if not new_track_ids:
                new_track_ids = None

        # update the candidates with the newly tracked instances.
        self.candidates.update_candidates(track_instances, new_track_ids)

        # convert the track_instances back to `List[sio.PredictedInstance]` objects.
        new_pred_instances = []
        for instance in track_instances:
            if instance.track_id is not None:
                instance.src_instance.track = sio.Track(instance.track_id)
                instance.src_instance.tracking_score = None
            new_pred_instances.append(instance.src_instance)

        return new_pred_instances

    def _scores_to_cost_matrix(self, scores):
        """Converts `scores` matrix to cost matrix for track assignments."""
        cost_matrix = -scores
        cost_matrix[np.isnan(cost_matrix)] = np.inf
        return cost_matrix

    def _get_features(self, untracked_instances):
        """Get features for the current untracked instances.

        The feature can either be an embedding of cropped image around each instance (visual feature),
        the bounding box coordinates, or centroids, or the poses as a feature.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.

        Returns:
            List of `TrackInstance` objects with the features assigned to each object and
            track_id set as `None`.
        """
        track_instances = []
        if not self.use_visual_features:
            # Pose (keypoints) as feature
            for instance in untracked_instances:
                pts = instance.numpy()
                track_instance = TrackInstance(
                    src_instance=instance,
                    track_id=None,
                    feature=pts,
                    instance_score=instance.score,
                )
                track_instances.append(track_instance)

        return track_instances

    def _get_scores(self, track_instances):
        """Compute association score between untracked and tracked instances.

        For visual feature vectors, this can be `cosine_sim`, for bounding boxes
        it could be `iou`, for centroids it could be `euclidean_dist`, and for poses it
        could be `oks`.

        Args:
            track_instances: List of `TrackInstance` objects with computed features.

        Returns:
            scores: Score matrix of shape (len(track_instances), num_existing_tracks)
        """
        scores = np.zeros((len(track_instances), len(self.candidates.current_tracks)))

        if self.scoring_method == "oks":
            scoring_method = compute_oks

        for instance_idx, instance in enumerate(track_instances):
            for track in self.candidates.current_tracks:
                oks = [
                    scoring_method(x.feature, instance.feature)
                    for x in self.candidates.get_instances_from_track_id(track)
                ]
                oks = np.nansum(oks) / len(oks)  # scoring reduction - Average
                scores[instance_idx][track] = oks

        return scores

    def _assign_tracks(self, track_instances, cost_matrix):
        """Assign track IDs using Hungarian method.

        Args:
            track_instances: List of `TrackInstance` objects with computed features.
            cost_matrix: Cost matrix of shape (len(track_instances), num_existing_tracks).

        Returns:
            Tuple (`track_instances`, `new_track_ids`) where the first is
            the list of `TrackInstance` objects with track IDs assigned and the latter is
            a list of new track IDs to be created.
        """
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_inds, col_inds):
            track_instances[row].track_id = col

        # Create new tracks for instances with unassigned tracks from Hungarian matching
        new_track_ids = None
        new_track_instances_inds = [
            x for x in range(len(track_instances)) if x not in row_inds
        ]
        if new_track_instances_inds:
            new_track_ids = []
            new_track_id = max(self.candidates.current_tracks) + 1
            for ind in new_track_instances_inds:
                if track_instances[ind].instance_score > self.instance_score_threshold:
                    track_instances[ind].track_id = new_track_id
                    new_track_ids.append(new_track_id)
                    new_track_id += 1

        return track_instances, new_track_ids
