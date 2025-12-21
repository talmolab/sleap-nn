"""Module for tracking."""

from typing import Any, Dict, List, Union, Deque, DefaultDict, Optional
from collections import defaultdict
import attrs
import cv2
import numpy as np
from time import time
from datetime import datetime
from loguru import logger
import functools
import rich
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

import sleap_io as sio
from sleap_nn.evaluation import compute_oks
from sleap_nn.tracking.candidates.fixed_window import FixedWindowCandidates
from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates
from sleap_nn.tracking.track_instance import (
    TrackedInstanceFeature,
    TrackInstances,
    TrackInstanceLocalQueue,
)
from sleap_nn.tracking.utils import (
    hungarian_matching,
    greedy_matching,
    get_bbox,
    get_centroid,
    get_keypoints,
    compute_euclidean_distance,
    compute_iou,
    compute_cosine_sim,
    cull_instances,
    cull_frame_instances,
)


@attrs.define
class Tracker:
    """Simple Pose Tracker.

    This is the base class for all Trackers. This module handles tracking instances
    across frames by creating new track IDs (or) assigning track IDs to each predicted
    instance when the `.track()` is called. This class is initialized in the `Predictor`
    classes.

    Attributes:
        candidate: Instance of either `FixedWindowCandidates` or `LocalQueueCandidates`.
        min_match_points: Minimum non-NaN points for match candidates. Default: 0.
        features: Feature representation for the candidates to update current detections.
            One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `robust_quantile`]. Default: `mean`.
        track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
            Default: `hungarian`.
        robust_best_instance: If the value is between 0 and 1
            (excluded), use a robust quantile similarity score for the
            track. If the value is 1, use the max similarity (non-robust).
            For selecting a robust score, 0.95 is a good value.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow shifts. Default: `False`.
        is_local_queue: `True` if `LocalQueueCandidates` is used else `False`.

    """

    candidate: Union[FixedWindowCandidates, LocalQueueCandidates] = (
        FixedWindowCandidates()
    )
    min_match_points: int = 0
    features: str = "keypoints"
    scoring_method: str = "oks"
    scoring_reduction: str = "mean"
    track_matching_method: str = "hungarian"
    robust_best_instance: float = 1.0
    use_flow: bool = False
    is_local_queue: bool = False
    tracking_target_instance_count: Optional[int] = None
    tracking_pre_cull_to_target: int = 0
    tracking_pre_cull_iou_threshold: float = 0
    _scoring_functions: Dict[str, Any] = {
        "oks": compute_oks,
        "iou": compute_iou,
        "cosine_sim": compute_cosine_sim,
        "euclidean_dist": compute_euclidean_distance,
    }
    _quantile_method = functools.partial(np.quantile, q=robust_best_instance)
    _scoring_reduction_methods: Dict[str, Any] = {
        "mean": np.nanmean,
        "max": np.nanmax,
        "robust_quantile": _quantile_method,
    }
    _feature_methods: Dict[str, Any] = {
        "keypoints": get_keypoints,
        "centroids": get_centroid,
        "bboxes": get_bbox,
    }
    _track_matching_methods: Dict[str, Any] = {
        "hungarian": hungarian_matching,
        "greedy": greedy_matching,
    }
    _track_objects: Dict[int, sio.Track] = {}

    @classmethod
    def from_config(
        cls,
        window_size: int = 5,
        min_new_track_points: int = 0,
        candidates_method: str = "fixed_window",
        min_match_points: int = 0,
        features: str = "keypoints",
        scoring_method: str = "oks",
        scoring_reduction: str = "mean",
        robust_best_instance: float = 1.0,
        track_matching_method: str = "hungarian",
        max_tracks: Optional[int] = None,
        use_flow: bool = False,
        of_img_scale: float = 1.0,
        of_window_size: int = 21,
        of_max_levels: int = 3,
        tracking_target_instance_count: Optional[int] = None,
        tracking_pre_cull_to_target: int = 0,
        tracking_pre_cull_iou_threshold: float = 0,
    ):
        """Create `Tracker` from config.

        Args:
            window_size: Number of frames to look for in the candidate instances to match
                with the current detections. Default: 5.
            min_new_track_points: We won't spawn a new track for an instance with
                fewer than this many non-nan points. Default: 0.
            candidates_method: Either of `fixed_window` or `local_queues`. In fixed window
                method, candidates from the last `window_size` frames. In local queues,
                last `window_size` instances for each track ID is considered for matching
                against the current detection. Default: `fixed_window`.
            min_match_points: Minimum non-NaN points for match candidates. Default: 0.
            features: Feature representation for the candidates to update current detections.
                One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
            scoring_method: Method to compute association score between features from the
                current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
                `euclidean_dist`]. Default: `oks`.
            scoring_reduction: Method to aggregate and reduce multiple scores if there are
                several detections associated with the same track. One of [`mean`, `max`,
                `robust_quantile`]. Default: `mean`.
            robust_best_instance: If the value is between 0 and 1
                (excluded), use a robust quantile similarity score for the
                track. If the value is 1, use the max similarity (non-robust).
                For selecting a robust score, 0.95 is a good value.
            track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
                Default: `hungarian`.
            max_tracks: Meaximum number of new tracks to be created to avoid redundant tracks.
                (only for local queues candidate) Default: None.
            use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow shifts. Default: `False`.
            of_img_scale: Factor to scale the images by when computing optical flow. Decrease
                this to increase performance at the cost of finer accuracy. Sometimes
                decreasing the image scale can improve performance with fast movements.
                Default: 1.0. (only if `use_flow` is True)
            of_window_size: Optical flow window size to consider at each pyramid scale
                level. Default: 21. (only if `use_flow` is True)
            of_max_levels: Number of pyramid scale levels to consider. This is different
                from the scale parameter, which determines the initial image scaling.
                Default: 3. (only if `use_flow` is True)
            tracking_target_instance_count: Target number of instances to track per frame. (default: None)
            tracking_pre_cull_to_target: If non-zero and target_instance_count is also non-zero, then cull instances over target count per frame *before* tracking. (default: 0)
            tracking_pre_cull_iou_threshold: If non-zero and pre_cull_to_target also set, then use IOU threshold to remove overlapping instances over count *before* tracking. (default: 0)

        """
        if candidates_method == "fixed_window":
            candidate = FixedWindowCandidates(
                window_size=window_size,
                min_new_track_points=min_new_track_points,
            )
            is_local_queue = False

        elif candidates_method == "local_queues":
            candidate = LocalQueueCandidates(
                window_size=window_size,
                max_tracks=max_tracks,
                min_new_track_points=min_new_track_points,
            )
            is_local_queue = True

        else:
            message = f"{candidates_method} is not a valid method. Please choose one of [`fixed_window`, `local_queues`]"
            logger.error(message)
            raise ValueError(message)

        if use_flow:
            return FlowShiftTracker(
                candidate=candidate,
                min_match_points=min_match_points,
                features=features,
                scoring_method=scoring_method,
                scoring_reduction=scoring_reduction,
                robust_best_instance=robust_best_instance,
                track_matching_method=track_matching_method,
                img_scale=of_img_scale,
                of_window_size=of_window_size,
                of_max_levels=of_max_levels,
                is_local_queue=is_local_queue,
                tracking_target_instance_count=tracking_target_instance_count,
                tracking_pre_cull_to_target=tracking_pre_cull_to_target,
                tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
            )

        tracker = cls(
            candidate=candidate,
            min_match_points=min_match_points,
            features=features,
            scoring_method=scoring_method,
            scoring_reduction=scoring_reduction,
            robust_best_instance=robust_best_instance,
            track_matching_method=track_matching_method,
            use_flow=use_flow,
            is_local_queue=is_local_queue,
            tracking_target_instance_count=tracking_target_instance_count,
            tracking_pre_cull_to_target=tracking_pre_cull_to_target,
            tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
        )
        return tracker

    def track(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
    ) -> List[sio.PredictedInstance]:
        """Assign track IDs to the untracked list of `sio.PredictedInstance` objects.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.
            frame_idx: Frame index of the predicted instances.
            image: Source image if visual features are to be used (also when using flow).

        Returns:
            List of `sio.PredictedInstance` objects, each having an assigned track.
        """
        if (
            self.tracking_target_instance_count is not None
            and self.tracking_target_instance_count
            and self.tracking_pre_cull_to_target
        ):
            untracked_instances = cull_frame_instances(
                untracked_instances,
                self.tracking_target_instance_count,
                self.tracking_pre_cull_iou_threshold,
            )
        # get features for the untracked instances.
        current_instances = self.get_features(untracked_instances, frame_idx, image)

        candidates_list = (
            self.generate_candidates()
        )  # either Deque/ DefaultDict for FixedWindow/ LocalQueue candidate.

        if candidates_list:
            # if track queue is not empty

            # update candidates if needed and get the features from previous tracked instances.
            candidates_feature_dict = self.update_candidates(candidates_list, image)

            # scoring function
            scores = self.get_scores(current_instances, candidates_feature_dict)
            cost_matrix = self.scores_to_cost_matrix(scores)

            # track assignment
            current_tracked_instances = self.assign_tracks(
                current_instances, cost_matrix
            )

        else:
            # Initialize the tracker queue if empty.
            current_tracked_instances = self.candidate.add_new_tracks(current_instances)

        # convert the `current_instances` back to `List[sio.PredictedInstance]` objects.
        if self.is_local_queue:
            new_pred_instances = []
            for instance in current_tracked_instances:
                if instance.track_id is not None:
                    if instance.track_id not in self._track_objects:
                        self._track_objects[instance.track_id] = sio.Track(
                            f"track_{instance.track_id}"
                        )
                    instance.src_instance.track = self._track_objects[instance.track_id]
                    instance.src_instance.tracking_score = instance.tracking_score
                new_pred_instances.append(instance.src_instance)

        else:
            new_pred_instances = []
            for idx, inst in enumerate(current_tracked_instances.src_instances):
                track_id = current_tracked_instances.track_ids[idx]
                if track_id is not None:
                    if track_id not in self._track_objects:
                        self._track_objects[track_id] = sio.Track(f"track_{track_id}")
                    inst.track = self._track_objects[track_id]
                    inst.tracking_score = current_tracked_instances.tracking_scores[idx]
                    new_pred_instances.append(inst)

        return new_pred_instances

    def get_features(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
    ) -> Union[TrackInstances, List[TrackInstanceLocalQueue]]:
        """Get features for the current untracked instances.

        The feature can either be an embedding of cropped image around each instance (visual feature),
        the bounding box coordinates, or centroids, or the poses as a feature.

        Args:
            untracked_instances: List of untracked `sio.PredictedInstance` objects.
            frame_idx: Frame index of the current untracked instances.
            image: Image of the current frame if visual features are to be used.

        Returns:
            `TrackInstances` object or `List[TrackInstanceLocalQueue]` with the features
            assigned for the untracked instances and track_id set as `None`.
        """
        if self.features not in self._feature_methods:
            message = "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes` and `image`"
            logger.error(message)
            raise ValueError(message)

        feature_method = self._feature_methods[self.features]
        feature_list = []
        for pred_instance in untracked_instances:
            feature_list.append(feature_method(pred_instance))

        current_instances = self.candidate.get_track_instances(
            feature_list, untracked_instances, frame_idx=frame_idx, image=image
        )

        return current_instances

    def generate_candidates(self):
        """Get the tracked instances from tracker queue."""
        return self.candidate.tracker_queue

    def update_candidates(
        self, candidates_list: Union[Deque, DefaultDict[int, Deque]], image: np.ndarray
    ) -> Dict[int, TrackedInstanceFeature]:
        """Return dictionary with the features of tracked instances.

        Args:
            candidates_list: List of tracked instances from tracker queue to consider.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        candidates_feature_dict = defaultdict(list)
        for track_id in self.candidate.current_tracks:
            candidates_feature_dict[track_id].extend(
                self.candidate.get_features_from_track_id(track_id, candidates_list)
            )
        return candidates_feature_dict

    def get_scores(
        self,
        current_instances: Union[TrackInstances, List[TrackInstanceLocalQueue]],
        candidates_feature_dict: Dict[int, TrackedInstanceFeature],
    ):
        """Compute association score between untracked and tracked instances.

        For visual feature vectors, this can be `cosine_sim`, for bounding boxes
        it could be `iou`, for centroids it could be `euclidean_dist`, and for poses it
        could be `oks`.

        Args:
            current_instances: `TrackInstances` object or `List[TrackInstanceLocalQueue]`
                with features and unassigned tracks.
            candidates_feature_dict: Dictionary with keys as track IDs and values as the
                list of `TrackedInstanceFeature`.

        Returns:
            scores: Score matrix of shape (num_new_instances, num_existing_tracks)
        """
        if self.scoring_method not in self._scoring_functions:
            message = "Invalid `scoring_method` argument. Please provide one of `oks`, `cosine_sim`, `iou`, and `euclidean_dist`."
            logger.error(message)
            raise ValueError(message)

        if self.scoring_reduction not in self._scoring_reduction_methods:
            message = "Invalid `scoring_reduction` argument. Please provide one of `mean`, `max`, and `robust_quantile`."
            logger.error(message)
            raise ValueError(message)

        scoring_method = self._scoring_functions[self.scoring_method]
        scoring_reduction = self._scoring_reduction_methods[self.scoring_reduction]

        # Get list of features for the `current_instances`.
        if self.is_local_queue:
            current_instances_features = [x.feature for x in current_instances]
        else:
            current_instances_features = [x for x in current_instances.features]

        scores = np.zeros(
            (len(current_instances_features), len(self.candidate.current_tracks))
        )

        for f_idx, f in enumerate(current_instances_features):
            for t_idx, track_id in enumerate(self.candidate.current_tracks):
                scores_trackid = [
                    scoring_method(f, x.feature)
                    for x in candidates_feature_dict[track_id]
                    if (~np.isnan(x.src_predicted_instance.numpy()).any(axis=1)).sum()
                    > self.min_match_points  # only if the candidates have min non-nan points
                ]
                score_trackid = scoring_reduction(scores_trackid)  # scoring reduction
                scores[f_idx][t_idx] = score_trackid

        return scores

    def scores_to_cost_matrix(self, scores: np.ndarray):
        """Converts `scores` matrix to cost matrix for track assignments."""
        cost_matrix = -scores
        cost_matrix[np.isnan(cost_matrix)] = np.inf
        return cost_matrix

    def assign_tracks(
        self,
        current_instances: Union[TrackInstances, List[TrackInstanceLocalQueue]],
        cost_matrix: np.ndarray,
    ) -> Union[TrackInstances, List[TrackInstanceLocalQueue]]:
        """Assign track IDs using Hungarian method.

        Args:
            current_instances: `TrackInstances` object or `List[TrackInstanceLocalQueue]`
                with features and unassigned tracks.
            cost_matrix: Cost matrix of shape (num_new_instances, num_existing_tracks).

        Returns:
            `TrackInstances` object or `List[TrackInstanceLocalQueue]`objects with
                track IDs assigned.
        """
        if self.track_matching_method not in self._track_matching_methods:
            message = "Invalid `track_matching_method` argument. Please provide one of `hungarian`, and `greedy`."
            logger.error(message)
            raise ValueError(message)

        matching_method = self._track_matching_methods[self.track_matching_method]

        row_inds, col_inds = matching_method(cost_matrix)
        tracking_scores = [
            -cost_matrix[row, col] for row, col in zip(row_inds, col_inds)
        ]

        # update the candidates tracker queue with the newly tracked instances and assign
        # track IDs to `current_instances`.
        current_tracked_instances = self.candidate.update_tracks(
            current_instances, row_inds, col_inds, tracking_scores
        )

        return current_tracked_instances


@attrs.define
class FlowShiftTracker(Tracker):
    """Module for tracking using optical flow shift matching.

    This module handles tracking instances across frames by creating new track IDs (or)
    assigning track IDs to each instance when the `.track()` is called using optical flow
    based track matching. This is a sub-class of the `Tracker` module, which configures
    the `update_candidates()` method specific to optical flow shift matching. This class is
    initialized in the `Tracker.from_config()` method.

    Attributes:
        candidates: Either `FixedWindowCandidates` or `LocalQueueCandidates` object.
        min_match_points: Minimum non-NaN points for match candidates. Default: 0.
        features: One of [`keypoints`, `centroids`, `bboxes`, `image`].
            Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `robust_quantile`]. Default: `mean`.
        robust_best_instance: If the value is between 0 and 1
                (excluded), use a robust quantile similarity score for the
                track. If the value is 1, use the max similarity (non-robust).
                For selecting a robust score, 0.95 is a good value.
        track_matching_method: track matching algorithm. One of `hungarian`, `greedy.
                Default: `hungarian`.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
            optical flow. Default: `False`.
        is_local_queue: `True` if `LocalQueueCandidates` is used else `False`.
        img_scale: Factor to scale the images by when computing optical flow. Decrease
            this to increase performance at the cost of finer accuracy. Sometimes
            decreasing the image scale can improve performance with fast movements.
            Default: 1.0.
        of_window_size: Optical flow window size to consider at each pyramid scale
            level. Default: 21.
        of_max_levels: Number of pyramid scale levels to consider. This is different
            from the scale parameter, which determines the initial image scaling.
            Default: 3
        tracking_target_instance_count: Target number of instances to track per frame. (default: None)
        tracking_pre_cull_to_target: If non-zero and target_instance_count is also non-zero, then cull instances over target count per frame *before* tracking. (default: 0)
        tracking_pre_cull_iou_threshold: If non-zero and pre_cull_to_target also set, then use IOU threshold to remove overlapping instances over count *before* tracking. (default: 0)

    """

    img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3

    def _compute_optical_flow(
        self, ref_pts: np.ndarray, ref_img: np.ndarray, new_img: np.ndarray
    ):
        """Compute instances on new frame using optical flow displacements."""
        ref_img, new_img = self._preprocess_imgs(ref_img, new_img)
        shifted_pts, status, errs = cv2.calcOpticalFlowPyrLK(
            ref_img,
            new_img,
            (np.concatenate(ref_pts, axis=0)).astype("float32") * self.img_scale,
            None,
            winSize=(self.of_window_size, self.of_window_size),
            maxLevel=self.of_max_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        shifted_pts /= self.img_scale
        return shifted_pts, status, errs

    def _preprocess_imgs(self, ref_img: np.ndarray, new_img: np.ndarray):
        """Pre-process images for optical flow."""
        # Convert to uint8 for cv2.calcOpticalFlowPyrLK
        if np.issubdtype(ref_img.dtype, np.floating):
            ref_img = ref_img.astype("uint8")
        if np.issubdtype(new_img.dtype, np.floating):
            new_img = new_img.astype("uint8")

        # Ensure images are rank 2 in case there is a singleton channel dimension.
        if ref_img.ndim > 3:
            ref_img = np.squeeze(ref_img)
            new_img = np.squeeze(new_img)

        # Convert RGB to grayscale.
        if ref_img.ndim > 2 and ref_img.shape[0] == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        # Input image scaling.
        if self.img_scale != 1:
            ref_img = cv2.resize(ref_img, None, None, self.img_scale, self.img_scale)
            new_img = cv2.resize(new_img, None, None, self.img_scale, self.img_scale)

        return ref_img, new_img

    def get_shifted_instances_from_prv_frames(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        new_img: np.ndarray,
        feature_method,
    ) -> Dict[int, List[TrackedInstanceFeature]]:
        """Generate shifted instances onto the new frame by applying optical flow."""
        shifted_instances_prv_frames = defaultdict(list)

        if self.is_local_queue:
            # for local queue
            ref_candidates = self.candidate.get_instances_groupby_frame_idx(
                candidates_list
            )
            for fidx, ref_candidate_list in ref_candidates.items():
                ref_pts = [x.src_instance.numpy() for x in ref_candidate_list]
                shifted_pts, status, errs = self._compute_optical_flow(
                    ref_pts=ref_pts,
                    ref_img=ref_candidate_list[0].image,
                    new_img=new_img,
                )

                sections = np.cumsum([len(x) for x in ref_pts])[:-1]
                shifted_pts = np.split(shifted_pts, sections, axis=0)
                status = np.split(status, sections, axis=0)
                errs = np.split(errs, sections, axis=0)

                # Create shifted instances.
                for idx, (ref_candidate, pts, found) in enumerate(
                    zip(ref_candidate_list, shifted_pts, status)
                ):
                    # Exclude points that weren't found by optical flow.
                    found = found.squeeze().astype(bool)
                    pts[~found] = np.nan

                    # Create a shifted instance.
                    shifted_instances_prv_frames[ref_candidate.track_id].append(
                        TrackedInstanceFeature(
                            feature=feature_method(pts),
                            src_predicted_instance=ref_candidate.src_instance,
                            frame_idx=fidx,
                            tracking_score=ref_candidate.tracking_score,
                            shifted_keypoints=pts,
                        )
                    )

        else:
            # for fixed window
            candidates_list = (
                candidates_list
                if candidates_list is not None
                else self.candidate.tracker_queue
            )
            for ref_candidate in candidates_list:
                ref_pts = [x.numpy() for x in ref_candidate.src_instances]
                shifted_pts, status, errs = self._compute_optical_flow(
                    ref_pts=ref_pts, ref_img=ref_candidate.image, new_img=new_img
                )

                sections = np.cumsum([len(x) for x in ref_pts])[:-1]
                shifted_pts = np.split(shifted_pts, sections, axis=0)
                status = np.split(status, sections, axis=0)
                errs = np.split(errs, sections, axis=0)

                # Create shifted instances.
                for idx, (pts, found) in enumerate(zip(shifted_pts, status)):
                    # Exclude points that weren't found by optical flow.
                    found = found.squeeze().astype(bool)
                    pts[~found] = np.nan

                    # Create a shifted instance.
                    shifted_instances_prv_frames[ref_candidate.track_ids[idx]].append(
                        TrackedInstanceFeature(
                            feature=feature_method(pts),
                            src_predicted_instance=ref_candidate.src_instances[idx],
                            frame_idx=ref_candidate.frame_idx,
                            tracking_score=ref_candidate.tracking_scores[idx],
                            shifted_keypoints=pts,
                        )
                    )

        return shifted_instances_prv_frames

    def update_candidates(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        image: np.ndarray,
    ) -> Dict[int, TrackedInstanceFeature]:
        """Return dictionary with the features of tracked instances.

        In this method, the tracked instances in the tracker queue are shifted on to the
        current frame using optical flow. The features are then computed from the shifted
        instances.

        Args:
            candidates_list: Tracker queue from the candidate class.
            image: Image of the current untracked frame. (used for flow shift tracker)

        Returns:
            Dictionary with keys as track IDs and values as the list of `TrackedInstanceFeature`.
        """
        # get feature method for the shifted instances
        if self.features not in self._feature_methods:
            message = "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes` and `image`"
            logger.error(message)
            raise ValueError(message)
        feature_method = self._feature_methods[self.features]

        # get shifted instances from optical flow
        shifted_instances_prv_frames = self.get_shifted_instances_from_prv_frames(
            candidates_list=candidates_list,
            new_img=image,
            feature_method=feature_method,
        )

        return shifted_instances_prv_frames


def connect_single_breaks(
    lfs: List[sio.LabeledFrame], max_instances: int
) -> List[sio.LabeledFrame]:
    """Merge single-frame breaks in tracks by connecting single lost track with single new track.

    Args:
        lfs: List of `LabeledFrame` objects with predicted instances.
        max_instances: The maximum number of instances we want per frame.

    Returns:
        Updated list of labeled frames with modified track IDs.
    """
    if not lfs:
        return lfs

    # Move instances in new tracks into tracks that disappeared on previous frame
    fix_track_map = dict()
    last_good_frame_tracks = {inst.track for inst in lfs[0].instances}
    for lf in lfs:
        frame_tracks = {inst.track for inst in lf.instances}

        tracks_fixed_before = frame_tracks.intersection(set(fix_track_map.keys()))
        if tracks_fixed_before:
            for inst in lf.instances:
                if (
                    inst.track in fix_track_map
                    and fix_track_map[inst.track] not in frame_tracks
                ):
                    inst.track = fix_track_map[inst.track]
                    frame_tracks = {inst.track for inst in lf.instances}

        extra_tracks = frame_tracks - last_good_frame_tracks
        missing_tracks = last_good_frame_tracks - frame_tracks

        if len(extra_tracks) == 1 and len(missing_tracks) == 1:
            for inst in lf.instances:
                if inst.track in extra_tracks:
                    old_track = inst.track
                    new_track = missing_tracks.pop()
                    fix_track_map[old_track] = new_track
                    inst.track = new_track

                    break
        else:
            if len(frame_tracks) == max_instances:
                last_good_frame_tracks = frame_tracks

    return lfs


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.1f} frames/s", style="progress.data.speed")


def run_tracker(
    untracked_frames: List[sio.LabeledFrame],
    window_size: int = 5,
    min_new_track_points: int = 0,
    candidates_method: str = "fixed_window",
    min_match_points: int = 0,
    features: str = "keypoints",
    scoring_method: str = "oks",
    scoring_reduction: str = "mean",
    robust_best_instance: float = 1.0,
    track_matching_method: str = "hungarian",
    max_tracks: Optional[int] = None,
    use_flow: bool = False,
    of_img_scale: float = 1.0,
    of_window_size: int = 21,
    of_max_levels: int = 3,
    post_connect_single_breaks: bool = False,
    tracking_target_instance_count: Optional[int] = None,
    tracking_pre_cull_to_target: int = 0,
    tracking_pre_cull_iou_threshold: float = 0,
    tracking_clean_instance_count: int = 0,
    tracking_clean_iou_threshold: float = 0,
) -> List[sio.LabeledFrame]:
    """Run tracking on a given set of frames.

    Args:
        untracked_frames: List of labeled frames with predicted instances to be tracked.
        window_size: Number of frames to look for in the candidate instances to match
                with the current detections. Default: 5.
        min_new_track_points: We won't spawn a new track for an instance with
            fewer than this many points. Default: 0.
        candidates_method: Either of `fixed_window` or `local_queues`. In fixed window
            method, candidates from the last `window_size` frames. In local queues,
            last `window_size` instances for each track ID is considered for matching
            against the current detection. Default: `fixed_window`.
        min_match_points: Minimum non-NaN points for match candidates. Default: 0.
        features: Feature representation for the candidates to update current detections.
            One of [`keypoints`, `centroids`, `bboxes`, `image`]. Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `euclidean_dist`]. Default: `oks`.
        scoring_reduction: Method to aggregate and reduce multiple scores if there are
            several detections associated with the same track. One of [`mean`, `max`,
            `robust_quantile`]. Default: `mean`.
        robust_best_instance: If the value is between 0 and 1
            (excluded), use a robust quantile similarity score for the
            track. If the value is 1, use the max similarity (non-robust).
            For selecting a robust score, 0.95 is a good value.
        track_matching_method: Track matching algorithm. One of `hungarian`, `greedy.
            Default: `hungarian`.
        max_tracks: Meaximum number of new tracks to be created to avoid redundant tracks.
            (only for local queues candidate) Default: None.
        use_flow: If True, `FlowShiftTracker` is used, where the poses are matched using
        optical flow shifts. Default: `False`.
        of_img_scale: Factor to scale the images by when computing optical flow. Decrease
            this to increase performance at the cost of finer accuracy. Sometimes
            decreasing the image scale can improve performance with fast movements.
            Default: 1.0. (only if `use_flow` is True)
        of_window_size: Optical flow window size to consider at each pyramid scale
            level. Default: 21. (only if `use_flow` is True)
        of_max_levels: Number of pyramid scale levels to consider. This is different
            from the scale parameter, which determines the initial image scaling.
                Default: 3. (only if `use_flow` is True).
        post_connect_single_breaks: If True and `max_tracks` is not None with local queues candidate method,
            connects track breaks when exactly one track is lost and exactly one new track is spawned in the frame.
        tracking_target_instance_count: Target number of instances to track per frame. (default: None)
        tracking_pre_cull_to_target: If non-zero and target_instance_count is also non-zero, then cull instances over target count per frame *before* tracking. (default: 0)
        tracking_pre_cull_iou_threshold: If non-zero and pre_cull_to_target also set, then use IOU threshold to remove overlapping instances over count *before* tracking. (default: 0)
        tracking_clean_instance_count: Target number of instances to clean *after* tracking. (default: 0)
        tracking_clean_iou_threshold: IOU to use when culling instances *after* tracking. (default: 0)

    Returns:
        `sio.Labels` object with tracked instances.

    """
    tracker = Tracker.from_config(
        window_size=window_size,
        min_new_track_points=min_new_track_points,
        candidates_method=candidates_method,
        min_match_points=min_match_points,
        features=features,
        scoring_method=scoring_method,
        scoring_reduction=scoring_reduction,
        robust_best_instance=robust_best_instance,
        track_matching_method=track_matching_method,
        max_tracks=max_tracks,
        use_flow=use_flow,
        of_img_scale=of_img_scale,
        of_window_size=of_window_size,
        of_max_levels=of_max_levels,
        tracking_target_instance_count=tracking_target_instance_count,
        tracking_pre_cull_to_target=tracking_pre_cull_to_target,
        tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
    )

    try:
        with Progress(
            "{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            "ETA:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            RateColumn(),
            auto_refresh=False,
            refresh_per_second=4,
            speed_estimate_period=5,
        ) as progress:
            task = progress.add_task("Tracking...", total=len(untracked_frames))
            last_report = time()

            tracked_lfs = []
            for lf in untracked_frames:
                # prefer user instances over predicted instance
                instances = []
                if lf.has_user_instances:
                    instances_to_track = lf.user_instances
                    if lf.has_predicted_instances:
                        instances = lf.predicted_instances
                else:
                    instances_to_track = lf.predicted_instances

                instances.extend(
                    tracker.track(
                        untracked_instances=instances_to_track,
                        frame_idx=lf.frame_idx,
                        image=lf.image,
                    )
                )
                tracked_lfs.append(
                    sio.LabeledFrame(
                        video=lf.video, frame_idx=lf.frame_idx, instances=instances
                    )
                )

                progress.update(task, advance=1)

                if time() - last_report > 0.25:
                    progress.refresh()
                    last_report = time()

    except KeyboardInterrupt:
        logger.info("Tracking interrupted by user")
        raise KeyboardInterrupt

    if tracking_clean_instance_count > 0:
        logger.info("Post-processing: Culling instances...")
        tracked_lfs = cull_instances(
            tracked_lfs, tracking_clean_instance_count, tracking_clean_iou_threshold
        )
        if not post_connect_single_breaks:
            logger.info("Post-processing: Connecting single breaks...")
            tracked_lfs = connect_single_breaks(
                tracked_lfs, tracking_clean_instance_count
            )

    if post_connect_single_breaks:
        if (
            tracking_target_instance_count is None
            or tracking_target_instance_count == 0
        ):
            if max_tracks is not None:
                suggestion = f"Add --tracking_target_instance_count {max_tracks} to your command (using your --max_tracks value)."
            else:
                suggestion = "Add --tracking_target_instance_count N where N is the expected number of instances per frame."
            message = (
                f"--post_connect_single_breaks requires --tracking_target_instance_count to be set. "
                f"{suggestion}"
            )
            logger.error(message)
            raise ValueError(message)
        start_final_pass_time = time()
        start_fp_timestamp = str(datetime.now())
        logger.info(
            f"Started final-pass (connecting single breaks) at: {start_fp_timestamp}"
        )
        tracked_lfs = connect_single_breaks(
            tracked_lfs, max_instances=tracking_target_instance_count
        )
        finish_fp_timestamp = str(datetime.now())
        total_fp_elapsed = time() - start_final_pass_time
        logger.info(
            f"Finished final-pass (connecting single breaks) at: {finish_fp_timestamp}"
        )
        logger.info(f"Total runtime: {total_fp_elapsed} secs")

    return tracked_lfs
