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
from sleap_nn.tracking.utils import (
    compute_euclidean_distance,
    compute_iou,
    compute_cosine_sim,
)


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
        features: One of [`keypoints`, `centroids`, `bboxes`, `image`].
            Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `weighted`]. Default: `mean`.

    """

    candidates: Union[FixedWindowCandidates, LocalQueueCandidates]
    instance_score_threshold: float = 0.0
    features: str = "keypoints"
    scoring_method: str = "oks"
    scoring_reduction: str = "mean"

    @classmethod
    def from_config(
        cls,
        max_tracks: int = 30,
        window_size: int = 10,
        instance_score_threshold: float = 0.5,
        candidates_method: str = "fixed_window",
        features: str = "keypoints",
        scoring_method: str = "oks",
        scoring_reduction: str = "mean",
    ):
        """Create `Tracker` from config."""
        if candidates_method == "fixed_window":
            candidates = FixedWindowCandidates(
                window_size=window_size, max_tracks=max_tracks
            )
        elif candidates_method == "local_queues":
            candidates = LocalQueueCandidates(
                window_size=window_size, max_tracks=max_tracks
            )
        else:
            raise ValueError(
                f"{candidates_method} is not a valid method. Please choose one of [`fixed_window`, `local_queues`]"
            )

        tracker = cls(
            instance_score_threshold=instance_score_threshold,
            candidates=candidates,
            features=features,
            scoring_method=scoring_method,
            scoring_reduction=scoring_reduction,
        )
        return tracker

    def track(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.array = None,
    ):
        """Assign track IDs to the untracked list of `sio.PredictedInstance` objects.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.
            frame_idx: Frame index of the Instances.
            image: Source image if visual features are to be used.

        Returns:
            List of `sio.PredictedInstance` objects that have an assigned track ID.
        """
        track_instances = self._get_features(
            untracked_instances, frame_idx
        )  # get features

        if self.candidates.tracker_queue:

            # scoring function
            scores = self._get_scores(track_instances)

            # track assignment
            cost_matrix = self._scores_to_cost_matrix(scores)
            track_instances = self._assign_tracks(track_instances, cost_matrix)

        else:  # Initialization of tracker queue
            for t in track_instances:
                if t.instance_score > self.instance_score_threshold:
                    new_tracks_id = self.candidates.get_new_track_id()
                    t.track_id = new_tracks_id

        # update the candidates tracker queue with the newly tracked instances.
        self.candidates.update_candidates(track_instances)

        # convert the track_instances back to `List[sio.PredictedInstance]` objects.
        new_pred_instances = []
        for instance in track_instances:
            if instance.track_id is not None:
                instance.src_instance.track = sio.Track(instance.track_id)
                instance.src_instance.tracking_score = None
            new_pred_instances.append(instance.src_instance)

        return new_pred_instances

    def _scores_to_cost_matrix(self, scores: np.array):
        """Converts `scores` matrix to cost matrix for track assignments."""
        cost_matrix = -scores
        cost_matrix[np.isnan(cost_matrix)] = np.inf
        return cost_matrix

    def _get_features(
        self, untracked_instances: List[sio.PredictedInstance], frame_idx: int
    ):
        """Get features for the current untracked instances.

        The feature can either be an embedding of cropped image around each instance (visual feature),
        the bounding box coordinates, or centroids, or the poses as a feature.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.
            frame_idx: Frame index of the Instances.

        Returns:
            List of `TrackInstance` objects with the features assigned to each object and
            track_id set as `None`.
        """
        feature_list = []
        if self.features == "keypoints":
            for instance in untracked_instances:
                pts = instance.numpy()
                feature_list.append(pts)

        elif self.features == "centroids":
            for instance in untracked_instances:
                centroid = np.nanmedian(instance.numpy(), axis=0)
                feature_list.append(centroid)

        elif self.features == "bboxes":
            for instance in untracked_instances:
                points = instance.numpy()
                bbox = np.concatenate(
                    [
                        np.nanmin(points, axis=0),
                        np.nanmax(points, axis=0),
                    ]  # [xmin, ymin, xmax, ymax]
                )
                feature_list.append(bbox)

        # TODO: image embedding

        else:
            raise ValueError(
                "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes` and `image`"
            )

        track_instances = []
        for instance, feat in zip(untracked_instances, feature_list):
            track_instance = TrackInstance(
                src_instance=instance,
                track_id=None,
                feature=feat,
                instance_score=instance.score,
                frame_idx=frame_idx,
            )
            track_instances.append(track_instance)

        return track_instances

    def _get_scores(self, track_instances: List[TrackInstance]):
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

        elif self.scoring_method == "euclidean_dist":
            scoring_method = compute_euclidean_distance

        elif self.scoring_method == "iou":
            scoring_method = compute_iou

        elif self.scoring_method == "cosine_sim":
            scoring_method = compute_cosine_sim

        else:
            raise ValueError(
                "Invalid `scoring_method` argument. Please provide one of `oks`, `cosine_sim`, `iou`, and `euclidean_dist`."
            )

        for instance_idx, instance in enumerate(track_instances):
            for track in self.candidates.current_tracks:
                oks = [
                    scoring_method(x.feature, instance.feature)
                    for x in self.candidates.get_instances_from_track_id(track)
                ]

                if self.scoring_reduction == "mean":
                    scoring_reduction = np.nanmean

                elif self.scoring_reduction == "max":
                    scoring_reduction = np.nanmax

                elif self.scoring_reduction == "weighted":
                    pass

                else:
                    raise ValueError(
                        "Invalid `scoring_reduction` argument. Please provide one of `mean`, `max`, and `weighted`."
                    )

                oks = scoring_reduction(oks)  # scoring reduction - Average
                scores[instance_idx][track] = oks

        return scores

    def _assign_tracks(
        self, track_instances: List[TrackInstance], cost_matrix: np.array
    ):
        """Assign track IDs using Hungarian method.

        Args:
            track_instances: List of `TrackInstance` objects with computed features.
            cost_matrix: Cost matrix of shape (len(track_instances), num_existing_tracks).

        Returns:
            `track_instances` which is a list of `TrackInstance` objects with track IDs assigned.
        """
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_inds, col_inds):
            track_instances[row].track_id = col

        # Create new tracks for instances with unassigned tracks from Hungarian matching
        new_track_instances_inds = [
            x for x in range(len(track_instances)) if x not in row_inds
        ]
        if new_track_instances_inds:
            for ind in new_track_instances_inds:
                if track_instances[ind].instance_score > self.instance_score_threshold:
                    new_track_id = self.candidates.get_new_track_id()
                    track_instances[ind].track_id = new_track_id

        return track_instances
