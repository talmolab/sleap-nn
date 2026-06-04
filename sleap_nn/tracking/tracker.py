"""Module for tracking."""

from typing import Any, Dict, List, Union, Deque, DefaultDict, Optional
from collections import defaultdict
import warnings
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
    get_mask,
    count_valid_points,
    is_segmentation_mask,
    compute_euclidean_distance,
    compute_iou,
    compute_mask_iou,
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
        min_match_points: Minimum support for match candidates: non-NaN keypoints,
            or foreground area (px) for `features="masks"`. Default: 0.
        features: Feature representation for the candidates to update current detections.
            One of [`keypoints`, `centroids`, `bboxes`, `masks`, `image`]. `masks`
            tracks bottom-up segmentation `PredictedSegmentationMask` objects.
            Default: `keypoints`.
        scoring_method: Method to compute association score between features from the
            current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
            `mask_iou`, `euclidean_dist`]. `mask_iou` is the pixel IoU between two
            segmentation masks (pair with `features="masks"`). Default: `oks`.
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
    oks_stddev: float = 0.025
    use_flow: bool = False
    is_local_queue: bool = False
    tracking_target_instance_count: Optional[int] = None
    tracking_pre_cull_to_target: int = 0
    tracking_pre_cull_iou_threshold: float = 0
    _scoring_functions: Dict[str, Any] = {
        "oks": compute_oks,
        "iou": compute_iou,
        "mask_iou": compute_mask_iou,
        "cosine_sim": compute_cosine_sim,
        "euclidean_dist": compute_euclidean_distance,
    }
    _scoring_reduction_methods: Dict[str, Any] = {
        "mean": np.nanmean,
        "max": np.nanmax,
        # `robust_quantile` is resolved per-instance in `get_scores` so it honors
        # `self.robust_best_instance`; this entry only registers the valid key
        # (a class-level functools.partial would freeze `q` at the class default).
        "robust_quantile": np.nanmax,
    }
    _feature_methods: Dict[str, Any] = {
        "keypoints": get_keypoints,
        "centroids": get_centroid,
        "bboxes": get_bbox,
        "masks": get_mask,
    }
    _track_matching_methods: Dict[str, Any] = {
        "hungarian": hungarian_matching,
        "greedy": greedy_matching,
    }
    _track_objects: Dict[int, sio.Track] = attrs.field(factory=dict)

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
        oks_stddev: Optional[float] = None,
        track_matching_method: str = "hungarian",
        max_tracks: Optional[int] = None,
        use_flow: bool = False,
        of_img_scale: float = 1.0,
        of_window_size: int = 21,
        of_max_levels: int = 3,
        use_kalman: bool = False,
        kf_track_features: str = "centroid",
        kf_init_frame_count: int = 10,
        kf_node_indices: Optional[List[int]] = None,
        kf_reset_gap_size: int = 5,
        kf_prediction_blend: float = 0.5,
        kf_gate_step_mult: float = 8.0,
        kf_min_gate_px: float = 40.0,
        kf_velocity_cap_mult: float = 3.0,
        kf_min_velocity_cap_px: float = 15.0,
        tracking_target_instance_count: Optional[int] = None,
        tracking_pre_cull_to_target: int = 0,
        tracking_pre_cull_iou_threshold: float = 0,
    ):
        """Create `Tracker` from config.

        Args:
            window_size: Number of frames to look for in the candidate instances to match
                with the current detections. Default: 5.
            min_new_track_points: We won't spawn a new track for an instance with
                fewer than this many non-nan points (for `features="masks"`, this
                is read as a foreground-area floor in px). Default: 0.
            candidates_method: Either of `fixed_window` or `local_queues`. In fixed window
                method, candidates from the last `window_size` frames. In local queues,
                last `window_size` instances for each track ID is considered for matching
                against the current detection. Default: `fixed_window`.
            min_match_points: Minimum support for match candidates: non-NaN
                keypoints, or foreground area (px) for `features="masks"`. Default: 0.
            features: Feature representation for the candidates to update current detections.
                One of [`keypoints`, `centroids`, `bboxes`, `masks`, `image`].
                Default: `keypoints`.
            scoring_method: Method to compute association score between features from the
                current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,
                `mask_iou`, `euclidean_dist`]. Default: `oks`.
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
            oks_stddev: Keypoint-spread normalization constant for `oks` scoring;
                larger is more tolerant of localization error. `None` (default)
                auto-resolves to 0.1 for `kf_track_features="keypoints"` (whose per-node
                prediction is noisier) and 0.025 otherwise.
            use_kalman: If True, `KalmanShiftTracker` is used, where poses are predicted
                with a per-track constant-velocity Kalman filter. Requires
                `tracking_target_instance_count` (or `max_tracks`) and is mutually
                exclusive with `use_flow`. Default: `False`.
            kf_track_features: What the Kalman motion model tracks: `centroid` (default;
                rigid translation of the last pose) or `keypoints` (per-node poses;
                noisier, pair with a larger `oks_stddev` or `features="bboxes"` +
                `scoring_method="iou"`). (only if `use_kalman` is True)
            kf_init_frame_count: Number of warm-up frames tracked with the base path
                before the Kalman filters are fit via EM. Default: 10.
                (only if `use_kalman` is True)
            kf_node_indices: Skeleton node (row) indices to track with the motion model.
                `None` uses all nodes. Default: None. (only if `use_kalman` is True)
            kf_reset_gap_size: Number of consecutive missed frames after which a stale
                track's filter is reset. Default: 5. (only if `use_kalman` is True)
            kf_prediction_blend: Weight of the motion prediction when blending with the
                last observation to form the scoring candidate. Default: 0.5.
                (only if `use_kalman` is True)
            kf_gate_step_mult: Measurement gate as a multiple of the track's median
                step. Default: 8.0. (only if `use_kalman` is True)
            kf_min_gate_px: Floor (px) for the measurement gate. Default: 40.0.
                (only if `use_kalman` is True)
            kf_velocity_cap_mult: Cap on learned velocity as a multiple of the track's
                median step. Default: 3.0. (only if `use_kalman` is True)
            kf_min_velocity_cap_px: Floor (px/frame) for the velocity cap. Default: 15.0.
                (only if `use_kalman` is True)
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

        if use_kalman and use_flow:
            message = (
                "`use_kalman` and `use_flow` are mutually exclusive; choose one "
                "tracker (Kalman tracking does not use optical flow)."
            )
            logger.error(message)
            raise ValueError(message)

        if use_kalman and tracking_target_instance_count is None and max_tracks is None:
            message = (
                "Kalman tracking requires a known target identity count: pass "
                "`tracking_target_instance_count` (or `max_tracks` / `--max_instances`)."
            )
            logger.error(message)
            raise ValueError(message)

        if use_kalman and kf_track_features not in ("centroid", "keypoints"):
            message = (
                f"Invalid kf_track_features={kf_track_features!r}; choose 'centroid' "
                "(default) or 'keypoints'."
            )
            logger.error(message)
            raise ValueError(message)

        # Resolve the OKS tolerance: the per-node 'keypoints' prediction is noisier, so
        # the strict default stddev (0.025) collapses its similarity scores; default it
        # to 0.1 (validated on synthetic + real data). Centroid/base keep 0.025. An
        # explicit oks_stddev always wins.
        if oks_stddev is None:
            oks_stddev = (
                0.1 if (use_kalman and kf_track_features == "keypoints") else 0.025
            )

        if use_kalman:
            return KalmanShiftTracker(
                candidate=candidate,
                min_match_points=min_match_points,
                features=features,
                scoring_method=scoring_method,
                scoring_reduction=scoring_reduction,
                robust_best_instance=robust_best_instance,
                oks_stddev=oks_stddev,
                track_matching_method=track_matching_method,
                kf_track_features=kf_track_features,
                kf_init_frame_count=kf_init_frame_count,
                kf_node_indices=kf_node_indices,
                kf_reset_gap_size=kf_reset_gap_size,
                kf_prediction_blend=kf_prediction_blend,
                kf_gate_step_mult=kf_gate_step_mult,
                kf_min_gate_px=kf_min_gate_px,
                kf_velocity_cap_mult=kf_velocity_cap_mult,
                kf_min_velocity_cap_px=kf_min_velocity_cap_px,
                is_local_queue=is_local_queue,
                tracking_target_instance_count=tracking_target_instance_count,
                tracking_pre_cull_to_target=tracking_pre_cull_to_target,
                tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
            )

        if use_flow:
            return FlowShiftTracker(
                candidate=candidate,
                min_match_points=min_match_points,
                features=features,
                scoring_method=scoring_method,
                scoring_reduction=scoring_reduction,
                robust_best_instance=robust_best_instance,
                oks_stddev=oks_stddev,
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
            oks_stddev=oks_stddev,
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
        # Pre-cull is pose-only (cull_frame_instances uses same_pose_as / bbox);
        # segmentation masks are scoped out of cull for the MVP (apply_tracking
        # rejects the pre-cull flags in mask mode, so this is belt-and-braces).
        masks_input = bool(untracked_instances) and is_segmentation_mask(
            untracked_instances[0]
        )
        if (
            not masks_input
            and self.tracking_target_instance_count is not None
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
            message = "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes`, `masks` and `image`"
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
            message = "Invalid `scoring_method` argument. Please provide one of `oks`, `cosine_sim`, `iou`, `mask_iou`, and `euclidean_dist`."
            logger.error(message)
            raise ValueError(message)

        if self.scoring_reduction not in self._scoring_reduction_methods:
            message = "Invalid `scoring_reduction` argument. Please provide one of `mean`, `max`, and `robust_quantile`."
            logger.error(message)
            raise ValueError(message)

        scoring_method = self._scoring_functions[self.scoring_method]
        if self.scoring_method == "oks":
            # OKS tolerance is configurable: a larger stddev is more forgiving of
            # localization error, which matters for the noisier per-keypoint Kalman
            # prediction (`kf_track_features="keypoints"`).
            scoring_method = functools.partial(compute_oks, stddev=self.oks_stddev)
        scoring_reduction = self._scoring_reduction_methods[self.scoring_reduction]
        if self.scoring_reduction == "robust_quantile":
            # Resolve at runtime so the per-instance `robust_best_instance` is
            # honored (a class-level partial freezes `q` at the class default).
            # `nanquantile` matches the NaN-handling of `nanmean`/`nanmax`.
            scoring_reduction = functools.partial(
                np.nanquantile, q=self.robust_best_instance
            )

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
                    if count_valid_points(x.src_predicted_instance)
                    > self.min_match_points  # candidates with min support (non-NaN
                    # keypoints, or mask area px for segmentation masks)
                ]
                # An empty candidate list (all filtered by `min_match_points`)
                # reduces to NaN (-> inf cost in `scores_to_cost_matrix`); guard
                # explicitly because `np.nanmax([])` raises (`np.nanmean([])` /
                # `np.nanquantile([])` return NaN, but `max` must not crash).
                score_trackid = (
                    np.nan if not scores_trackid else scoring_reduction(scores_trackid)
                )
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
                if not ref_pts:
                    continue
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
                if not ref_pts:
                    continue
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
            message = "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes`, `masks` and `image`"
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


def _get_kalman_filter_cls():
    """Import and return `pykalman.KalmanFilter`, with a clear error if missing.

    `pykalman` is imported lazily so that an install/compat problem only affects
    Kalman tracking rather than breaking the whole tracking/inference module at
    import time.
    """
    try:
        from pykalman import KalmanFilter
    except ImportError as e:  # pragma: no cover - exercised only without pykalman
        message = (
            "Kalman-filter tracking (`use_kalman=True`) requires the `pykalman` "
            "package, which could not be imported. Install it with "
            "`pip install pykalman` (it normally ships as a core sleap-nn dependency)."
        )
        logger.error(message)
        raise ImportError(message) from e
    return KalmanFilter


@attrs.define
class KalmanShiftTracker(Tracker):
    """Tracker that predicts candidate poses with per-track Kalman filters.

    `KalmanShiftTracker` mirrors `FlowShiftTracker`: it subclasses `Tracker` and
    overrides only `update_candidates()` (plus a thin `track()` that records the
    current frame index). Instead of shifting previous-frame keypoints with optical
    flow, it advances one constant-velocity `pykalman.KalmanFilter` per track to
    predict where each tracked instance should be in the current frame. The shared
    scoring/matching path (`get_scores` -> `scores_to_cost_matrix` -> `assign_tracks`)
    is reused unchanged.

    The tracker runs in two phases:

    1. **Warm-up.** For the first `kf_init_frame_count` frames, `update_candidates`
       delegates to the base keypoint-feature path (i.e. behaves like a plain
       fixed-window / local-queue tracker) while accumulating a per-track keypoint
       history. Because the candidate queue is bounded to `window_size`, the history
       is kept in a separate buffer (`_obs_history`) so warm-up can span more frames
       than the queue holds.
    2. **Motion model.** Once `kf_init_frame_count` frames have been seen, one
       constant-velocity Kalman filter is fit per track over the warm-up window — on the
       per-track CENTROID (state ``[cx, vcx, cy, vcy]``), not every keypoint
       independently (a per-keypoint fit overfits noise into non-physical poses). Each
       frame thereafter, `update_candidates`: (a) resets tracks unseen beyond
       `kf_reset_gap_size` frames; (b) corrects each matched filter with its newly
       observed centroid subject to a distance gate (rejecting false positives /
       mismatches), coasting across multi-frame gaps so elapsed motion is not dumped into
       velocity; (c) lazily (re)fits filters for tracks that lack one (entrants /
       post-reset, from a contiguous fresh window); and (d) projects the centroid forward
       and builds the candidate by RIGIDLY translating the last observed pose by a
       fraction (`kf_prediction_blend`) of the predicted centroid displacement —
       translating the real body keeps the candidate geometrically valid so the
       similarity score stays meaningful.

    Robustness knobs (`kf_prediction_blend`, the measurement-gate and velocity-cap
    parameters; tuned defaults, overridable via `Tracker.from_config(...)`) make the
    motion model net-beneficial where association is ambiguous — crossing / converging /
    fast-smooth motion — and neutral on clean, false-positive, and occluded scenes.
    Under heavy detection noise with frequent missed detections it can slightly reduce
    IDF1 vs the memoryless base tracker (lower `kf_prediction_blend`, e.g. 0.25, to
    favor the last observation there).

    Kalman tracking requires a known target identity count
    (`tracking_target_instance_count`, or one derived from `max_tracks`/`max_instances`)
    and is mutually exclusive with `use_flow`; both are validated in
    `Tracker.from_config`.

    Attributes:
        kf_init_frame_count: Number of warm-up frames tracked with the base path
            before the per-track Kalman filters are fit via EM. Default: 10.
        kf_node_indices: Skeleton node (row) indices to track with the motion model.
            `None` (default) uses all nodes.
        kf_reset_gap_size: Number of consecutive missed frames after which a stale
            track's filter is reset (and later re-fit). Default: 5.
        kf_prediction_blend: Weight of the motion prediction when blending it with the
            last observation to form the scoring candidate (`w*pred + (1-w)*last_obs`).
            0 = pure last-observation (no motion model at scoring), 1 = pure prediction.
            Scales toward pure prediction during gaps. Default: 0.5.
        kf_gate_step_mult: Measurement gate as a multiple of the track's median step;
            an observation farther than `max(kf_min_gate_px, kf_gate_step_mult*step)`
            from the prediction is rejected (treated as a miss). Default: 8.0.
        kf_min_gate_px: Floor (px) for the measurement gate. Default: 40.0.
        kf_velocity_cap_mult: Cap on learned per-coordinate velocity as a multiple of
            the track's median step. Default: 3.0.
        kf_min_velocity_cap_px: Floor (px/frame) for the velocity cap. Default: 15.0.
    """

    kf_track_features: str = "centroid"
    kf_init_frame_count: int = 10
    kf_node_indices: Optional[List[int]] = None
    kf_reset_gap_size: int = 5
    kf_prediction_blend: float = 0.5
    kf_gate_step_mult: float = 8.0
    kf_min_gate_px: float = 40.0
    kf_velocity_cap_mult: float = 3.0
    kf_min_velocity_cap_px: float = 15.0

    # Per-instance Kalman state (never passed to the constructor).
    _kalman_filters: Dict[int, Any] = attrs.field(init=False, factory=dict)
    _last_results: Dict[int, Dict[str, Any]] = attrs.field(init=False, factory=dict)
    _last_frame_for_track: Dict[int, int] = attrs.field(init=False, factory=dict)
    _last_corrected_frame: Dict[int, int] = attrs.field(init=False, factory=dict)
    _obs_history: Dict[int, List[Dict[str, Any]]] = attrs.field(
        init=False, factory=dict
    )
    _resolved_node_indices: Optional[List[int]] = attrs.field(init=False, default=None)
    _n_nodes: Optional[int] = attrs.field(init=False, default=None)
    _frames_seen: int = attrs.field(init=False, default=0)
    _initialized: bool = attrs.field(init=False, default=False)
    _current_frame_idx: int = attrs.field(init=False, default=0)
    # Per-track robust inter-frame centroid step, used for the measurement gate and
    # the velocity cap (set at filter init).
    _median_step: Dict[int, float] = attrs.field(init=False, factory=dict)
    # Frame index at which a track was last reset; (re)fit windows only use
    # observations strictly after this, so a refit never straddles an occlusion gap.
    _reset_frame: Dict[int, int] = attrs.field(init=False, factory=dict)

    def track(
        self,
        untracked_instances: List[sio.PredictedInstance],
        frame_idx: int,
        image: np.ndarray = None,
    ) -> List[sio.PredictedInstance]:
        """Record the frame index, run base tracking, then ingest the assignment.

        Observations are recorded into `_obs_history` AFTER `super().track()` (i.e.
        after the current frame's track assignment is finalized) so each track id is
        associated with the instance it was actually matched to this frame, not a
        pre-assignment queue snapshot.
        """
        self._current_frame_idx = int(frame_idx)
        result = super().track(untracked_instances, frame_idx, image)
        self._ingest_observations(self.candidate.tracker_queue)
        return result

    def update_candidates(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        image: np.ndarray,
    ) -> Dict[int, List[TrackedInstanceFeature]]:
        """Return Kalman-predicted candidate features for the current frame.

        During warm-up this delegates to the base keypoint-feature path. Once the
        filters are initialized, each track's filter is corrected (with measurement
        gating) by its newly observed keypoints, stale tracks are reset, filters are
        lazily (re)fit for tracks that lack one, and a motion-predicted pose blended
        with the last observation is returned as the candidate feature.

        Args:
            candidates_list: Tracker queue from the candidate class.
            image: Image of the current untracked frame (unused; Kalman tracking does
                not use image features).

        Returns:
            Dictionary with keys as track IDs and values as lists of
            `TrackedInstanceFeature`.
        """
        if self.features not in self._feature_methods:
            message = "Invalid `features` argument. Please provide one of `keypoints`, `centroids`, `bboxes`, `masks` and `image`"
            logger.error(message)
            raise ValueError(message)
        feature_method = self._feature_methods[self.features]

        if not self._initialized:
            self._frames_seen += 1
            if self._frames_seen >= self.kf_init_frame_count:
                self._init_filters()
            if not self._initialized:
                # Still warming up: behave exactly like the base tracker.
                return super().update_candidates(candidates_list, image)

        # Reset tracks that have gone stale (no *accepted* observation within
        # `kf_reset_gap_size` frames) BEFORE correcting, so a track that is only
        # receiving gated-out observations is dropped to the base path rather than
        # eventually corrupted by a stale-extrapolation match. Then correct matched
        # filters with gated observations, lazily (re)fit filters for tracks that lack
        # one, and predict for scoring.
        self._reset_stale_tracks(self._current_frame_idx)
        self._correct_filters()
        self._init_missing_filters()
        return self._predict_candidates(candidates_list, feature_method)

    def _ingest_observations(
        self, candidates_list: Union[Deque, DefaultDict[int, Deque]]
    ) -> None:
        """Record the most recent observation per track into `_obs_history`.

        Uses the candidate class's own `get_features_from_track_id` accessor so this
        works for both fixed-window and local-queue tracker queues.
        """
        for track_id in self.candidate.current_tracks:
            feats = self.candidate.get_features_from_track_id(track_id, candidates_list)
            if not feats:
                continue
            newest = max(
                feats,
                key=lambda tf: (tf.frame_idx if tf.frame_idx is not None else -1),
            )
            frame_idx = (
                int(newest.frame_idx)
                if newest.frame_idx is not None
                else self._current_frame_idx
            )
            history = self._obs_history.setdefault(track_id, [])
            if history and history[-1]["frame_idx"] >= frame_idx:
                continue  # already recorded this (or a newer) observation
            keypoints = newest.src_predicted_instance.numpy()
            history.append(
                {
                    "frame_idx": frame_idx,
                    "keypoints": keypoints,
                    "src": newest.src_predicted_instance,
                    "score": newest.tracking_score,
                }
            )
            if self._n_nodes is None:
                self._n_nodes = keypoints.shape[0]

    def _resolve_node_indices(self) -> List[int]:
        """Resolve `kf_node_indices` to a concrete list of node-row indices.

        These nodes define the per-track *centroid* the motion model tracks; the
        predicted centroid displacement is applied rigidly to the whole body.
        """
        if self.kf_node_indices is not None:
            return [i for i in self.kf_node_indices if i < (self._n_nodes or 0)]
        return list(range(self._n_nodes)) if self._n_nodes else []

    def _num_track_points(self) -> int:
        """Number of points the motion model tracks per instance.

        1 for ``kf_track_features="centroid"`` (the per-track centroid); one per
        selected node for ``kf_track_features="keypoints"``.
        """
        if self.kf_track_features == "keypoints":
            return max(1, len(self._resolved_node_indices))
        return 1

    def _tracked_points(self, keypoints: np.ndarray) -> np.ndarray:
        """The points the motion model tracks for an instance, shape (P, 2).

        Centroid mode: the single (visibility-aware) centroid. Keypoints mode: the
        selected node coordinates as-is (NaN where a node is missing).
        """
        if self.kf_track_features == "keypoints":
            return np.asarray(keypoints, dtype=float)[self._resolved_node_indices, :]
        return self._centroid(keypoints).reshape(1, 2)

    def _build_matrices(self):
        """Build constant-velocity transition/observation matrices for P points.

        State is ``[x0, vx0, y0, vy0, x1, vx1, y1, vy1, ...]`` (4*P dims); the
        observation is the P point positions ``[x0, y0, x1, y1, ...]`` (2*P dims).
        For ``kf_track_features="centroid"`` P=1 (a single stable centroid filter,
        whose predicted displacement is applied rigidly to the whole body); for
        ``"keypoints"`` P is the number of tracked nodes (each node gets its own
        constant-velocity filter — noisier, but uses the pose directly).
        """
        n_points = self._num_track_points()
        state_dim = 4 * n_points
        obs_dim = 2 * n_points
        transition = [[0.0] * state_dim for _ in range(state_dim)]
        observation = [[0.0] * state_dim for _ in range(obs_dim)]
        for p in range(n_points):
            b = 4 * p
            transition[b][b] = 1.0  # x' = x + vx
            transition[b][b + 1] = 1.0
            transition[b + 1][b + 1] = 1.0  # vx' = vx
            transition[b + 2][b + 2] = 1.0  # y' = y + vy
            transition[b + 2][b + 3] = 1.0
            transition[b + 3][b + 3] = 1.0  # vy' = vy
            observation[2 * p][b] = 1.0  # observe x
            observation[2 * p + 1][b + 2] = 1.0  # observe y
        return transition, observation

    def _centroid(self, keypoints: np.ndarray) -> np.ndarray:
        """NaN-ignoring centroid of the tracked nodes, shape (2,).

        Returns NaN when fewer than half the tracked nodes are visible: a centroid
        built from a small, changing subset of nodes is biased (it shifts as different
        nodes drop), which would otherwise feed the filter a spurious displacement.
        A NaN centroid is treated as a missing observation (the filter coasts) rather
        than a corrupting one.
        """
        import warnings

        pts = np.asarray(keypoints)[self._resolved_node_indices, :]
        visible = int((~np.isnan(pts).any(axis=1)).sum())
        if visible == 0 or visible * 2 < pts.shape[0]:
            return np.array([np.nan, np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(pts, axis=0)

    def _obs_vector(self, keypoints: np.ndarray) -> np.ndarray:
        """Masked observation vector of the tracked points, shape (2*P,)."""
        return np.ma.masked_invalid(
            np.ma.asarray(self._tracked_points(keypoints).flatten(), dtype=float)
        )

    @staticmethod
    def _predicted_points(mean: np.ndarray) -> np.ndarray:
        """Extract predicted point positions ``[[x0,y0],...]`` from a state mean."""
        return np.asarray(mean)[::2].reshape(-1, 2)

    def _predicted_centroid(self, mean: np.ndarray) -> np.ndarray:
        """Centroid of the predicted tracked points, shape (2,) (used for gating)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(self._predicted_points(mean), axis=0)

    @staticmethod
    def _cap_velocity(mean: np.ndarray, cap: float) -> np.ndarray:
        """Clip the per-axis velocity entries (vcx, vcy) of a state mean to +/- cap."""
        mean = np.asarray(mean, dtype=float).copy()
        mean[1::2] = np.clip(mean[1::2], -cap, cap)
        return mean

    def _window_median_step(self, window: List[Dict[str, Any]]) -> float:
        """Noise-robust estimate of the per-frame centroid step over a window.

        Uses the endpoint displacement divided by the number of elapsed FRAMES
        between the first and last valid centroids (not the count of valid intervals):
        dividing by interval count would overestimate the per-frame step by up to the
        gap length when centroids drop out mid-window, which would loosen the velocity
        cap and gate in exactly the noisy regime they protect. The endpoint baseline
        averages per-frame measurement noise out for a roughly constant-velocity track.
        """
        valid = [
            (h["frame_idx"], self._centroid(h["keypoints"]))
            for h in window
            if not np.isnan(self._centroid(h["keypoints"])).any()
        ]
        if len(valid) < 2:
            return 0.0
        span = valid[-1][0] - valid[0][0]
        if span <= 0:
            return 0.0
        baseline = float(np.linalg.norm(valid[-1][1] - valid[0][1])) / span
        return baseline

    def _velocity_cap(self, track_id: int) -> float:
        med = self._median_step.get(track_id, 0.0)
        return max(self.kf_min_velocity_cap_px, self.kf_velocity_cap_mult * med)

    def _gate_distance(self, track_id: int) -> float:
        med = self._median_step.get(track_id, 0.0)
        return max(self.kf_min_gate_px, self.kf_gate_step_mult * med)

    def _contiguous_fresh_window(self, track_id: int) -> List[Dict[str, Any]]:
        """Longest suffix of a track's history that is contiguous and post-reset.

        Observations from before the track's last reset are excluded, and the window
        is broken at any frame-index gap > 1, so a (re)fit never straddles an
        occlusion and the median-step / velocity-cap estimates stay physical.
        """
        history = self._obs_history.get(track_id, [])
        reset_frame = self._reset_frame.get(track_id, -1)
        fresh = [h for h in history if h["frame_idx"] > reset_frame]
        if not fresh:
            return []
        window = [fresh[-1]]
        for h in reversed(fresh[:-1]):
            if window[0]["frame_idx"] - h["frame_idx"] == 1:
                window.insert(0, h)
            else:
                break
        return window

    def _fit_track_filter(self, track_id: int) -> bool:
        """Fit a centroid Kalman filter for a track from a contiguous fresh window.

        Returns True on success. Seeds the initial state from the first *finite*
        centroid (and a capped finite-difference velocity), keeps the initial mean
        fixed during EM, and caps the learned velocity so a short/noisy window cannot
        produce a runaway state.
        """
        window = self._contiguous_fresh_window(track_id)
        if len(window) < 3:
            return False  # need a few contiguous frames for a stable velocity fit
        window = window[-self.kf_init_frame_count :]
        n_points = self._num_track_points()
        obs_dim = 2 * n_points
        rows = np.asarray(
            [self._tracked_points(h["keypoints"]).flatten() for h in window],
            dtype=float,
        )  # (T, 2P), NaN where a tracked point is missing
        obs = np.ma.masked_invalid(rows)

        median_step = self._window_median_step(window)
        velocity_cap = max(
            self.kf_min_velocity_cap_px, self.kf_velocity_cap_mult * median_step
        )

        # Need at least two frames with any usable observation.
        if int(np.sum(~np.isnan(rows).all(axis=1))) < 2:
            return False

        # Per-coordinate seed: position from the first finite value; velocity from the
        # first consecutive finite pair (capped) so a dropout does not mislabel a
        # multi-frame step as a one-frame velocity. Coordinates never seen in the
        # window are filled with the same-axis mean (never the image origin).
        first = np.full(obs_dim, np.nan)
        seed_vel = np.zeros(obs_dim)
        for c in range(obs_dim):
            finite_t = np.where(~np.isnan(rows[:, c]))[0]
            if len(finite_t) == 0:
                continue
            first[c] = rows[finite_t[0], c]
            for t in finite_t:
                if t + 1 < len(rows) and not np.isnan(rows[t + 1, c]):
                    seed_vel[c] = np.clip(
                        rows[t + 1, c] - rows[t, c], -velocity_cap, velocity_cap
                    )
                    break
        if np.isnan(first).all():
            return False
        if np.isnan(first).any():
            fx = np.nanmean(first[0::2]) if not np.isnan(first[0::2]).all() else 0.0
            fy = np.nanmean(first[1::2]) if not np.isnan(first[1::2]).all() else 0.0
            first[0::2] = np.where(np.isnan(first[0::2]), fx, first[0::2])
            first[1::2] = np.where(np.isnan(first[1::2]), fy, first[1::2])
        initial_state_mean = [0.0] * (4 * n_points)
        for p in range(n_points):
            initial_state_mean[4 * p] = float(first[2 * p])
            initial_state_mean[4 * p + 1] = float(seed_vel[2 * p])
            initial_state_mean[4 * p + 2] = float(first[2 * p + 1])
            initial_state_mean[4 * p + 3] = float(seed_vel[2 * p + 1])

        transition, observation = self._build_matrices()
        kalman_filter_cls = _get_kalman_filter_cls()
        try:
            kf = kalman_filter_cls(
                transition_matrices=transition,
                observation_matrices=observation,
                initial_state_mean=initial_state_mean,
            )
            # Learn only the noise covariances; keep the structural matrices and the
            # initial state mean fixed.
            kf = kf.em(
                obs,
                n_iter=20,
                em_vars=[
                    "transition_covariance",
                    "observation_covariance",
                    "initial_state_covariance",
                ],
            )
            means, covariances = kf.filter(obs)
        except Exception as e:  # pragma: no cover - numerical edge cases
            logger.warning(
                f"Kalman filter initialization failed for track {track_id}: {e}"
            )
            return False

        self._kalman_filters[track_id] = kf
        self._last_results[track_id] = {
            "means": self._cap_velocity(means[-1], velocity_cap),
            "covariances": covariances[-1],
        }
        self._last_corrected_frame[track_id] = window[-1]["frame_idx"]
        self._last_frame_for_track[track_id] = window[-1]["frame_idx"]
        self._median_step[track_id] = median_step
        return True

    def _init_filters(self) -> None:
        """Fit a centroid Kalman filter per track at the end of warm-up."""
        self._resolved_node_indices = self._resolve_node_indices()
        if len(self._resolved_node_indices) == 0:
            # Nothing to track with a motion model; fall back to the base path.
            self._initialized = True
            return
        for track_id in list(self._obs_history.keys()):
            self._fit_track_filter(track_id)
        self._initialized = True

    def _init_missing_filters(self) -> None:
        """Lazily (re)fit filters for active tracks that lack one.

        Covers identities that spawn after warm-up and tracks whose filter was reset.
        A filter is fit only once `kf_init_frame_count` CONTIGUOUS fresh (post-reset)
        observations have accumulated, so a just-reset track is not immediately re-fit
        (no thrashing) and the fit window never straddles the occlusion gap.
        """
        if not self._resolved_node_indices:
            return
        for track_id in self.candidate.current_tracks:
            if track_id in self._kalman_filters:
                continue
            window = self._contiguous_fresh_window(track_id)
            if len(window) >= self.kf_init_frame_count:
                self._fit_track_filter(track_id)

    def _correct_filters(self) -> None:
        """Advance each matched filter with gated centroid observations.

        Coasts the filter across multi-frame gaps (one masked predict per missed
        frame) before applying the reappearance observation, so the elapsed motion is
        not dumped into the velocity state. Rejects observations whose centroid is
        beyond the measurement gate from the prediction (e.g. false positives),
        treating them as a miss.
        """
        for track_id, kf in list(self._kalman_filters.items()):
            history = self._obs_history.get(track_id, [])
            last_corrected = self._last_corrected_frame.get(track_id, -1)
            new_observations = [h for h in history if h["frame_idx"] > last_corrected]
            velocity_cap = self._velocity_cap(track_id)
            gate = self._gate_distance(track_id)
            for h in new_observations:
                prior = self._last_results[track_id]
                mean = prior["means"]
                covariance = prior["covariances"]
                gap = h["frame_idx"] - self._last_corrected_frame.get(track_id, -1)
                try:
                    # Coast across missed frames so the single reappearance update
                    # does not absorb the whole gap displacement as velocity.
                    for _ in range(max(0, gap - 1)):
                        mean, covariance = kf.filter_update(
                            mean, covariance, observation=np.ma.masked
                        )
                        mean = self._cap_velocity(mean, velocity_cap)
                    # Predict the observation frame to gate the measurement.
                    pred_mean, pred_cov = kf.filter_update(
                        mean, covariance, observation=np.ma.masked
                    )
                    pred_centroid = self._predicted_centroid(pred_mean)
                    obs_centroid = self._centroid(h["keypoints"])
                    gated_out = (
                        not np.isnan(pred_centroid).any()
                        and not np.isnan(obs_centroid).any()
                        and float(np.linalg.norm(pred_centroid - obs_centroid)) > gate
                    )
                    if gated_out:
                        # Reject the observation (likely a false positive / mismatch);
                        # carry the predict-only state forward as a miss.
                        mean, covariance = pred_mean, pred_cov
                    else:
                        mean, covariance = kf.filter_update(
                            mean,
                            covariance,
                            observation=self._obs_vector(h["keypoints"]),
                        )
                except Exception as e:  # pragma: no cover - numerical edge cases
                    logger.warning(
                        f"Kalman filter update failed for track {track_id}: {e}"
                    )
                    break
                self._last_results[track_id] = {
                    "means": self._cap_velocity(mean, velocity_cap),
                    "covariances": covariance,
                }
                self._last_corrected_frame[track_id] = h["frame_idx"]
                if not gated_out:
                    self._last_frame_for_track[track_id] = h["frame_idx"]

    def _reset_stale_tracks(self, frame_idx: int) -> None:
        """Reset filters for any track unseen for more than `kf_reset_gap_size` frames.

        A reset track drops its (now-unreliable) filter and falls back to the base
        feature path; `_init_missing_filters` re-fits it once it re-accumulates enough
        fresh contiguous history. Unlike the legacy `tracks_with_gap` rule, a single
        long occlusion is reset too, so a stale extrapolation cannot mis-associate the
        reappearing animal. `_reset_frame` is stamped so the next fit window starts
        fresh.
        """
        stale = [
            track_id
            for track_id, last in self._last_frame_for_track.items()
            if frame_idx - last > self.kf_reset_gap_size
        ]
        for track_id in stale:
            self._kalman_filters.pop(track_id, None)
            self._last_results.pop(track_id, None)
            self._last_frame_for_track.pop(track_id, None)
            self._last_corrected_frame.pop(track_id, None)
            self._median_step.pop(track_id, None)
            self._reset_frame[track_id] = frame_idx

    def _predict_candidates(
        self,
        candidates_list: Union[Deque, DefaultDict[int, Deque]],
        feature_method,
    ) -> Dict[int, List[TrackedInstanceFeature]]:
        """Build candidate features by rigidly translating the last observed pose.

        The centroid filter is projected forward from the last corrected state to the
        current frame (coasting across any gap), and the last observed body is
        translated by (a fraction of) the predicted centroid displacement. Translating
        the *real* last pose keeps the candidate rigid and geometrically valid, so the
        OKS / similarity score stays meaningful (predicting each keypoint
        independently produced non-physical poses that scored ~0 and randomized the
        assignment). Tracks without an active filter fall back to the base feature
        path so they remain trackable.
        """
        predicted = defaultdict(list)
        for track_id in self.candidate.current_tracks:
            kf = self._kalman_filters.get(track_id)
            prior = self._last_results.get(track_id)
            history = self._obs_history.get(track_id)
            if kf is None or prior is None or not history:
                predicted[track_id].extend(
                    self.candidate.get_features_from_track_id(track_id, candidates_list)
                )
                continue

            steps = max(
                1,
                self._current_frame_idx
                - self._last_corrected_frame.get(track_id, self._current_frame_idx - 1),
            )
            velocity_cap = self._velocity_cap(track_id)
            try:
                mean = prior["means"]
                covariance = prior["covariances"]
                for _ in range(steps):
                    mean, covariance = kf.filter_update(
                        mean, covariance, observation=np.ma.masked
                    )
                    mean = self._cap_velocity(mean, velocity_cap)
            except Exception as e:  # pragma: no cover - numerical edge cases
                logger.warning(f"Kalman prediction failed for track {track_id}: {e}")
                predicted[track_id].extend(
                    self.candidate.get_features_from_track_id(track_id, candidates_list)
                )
                continue

            ref = history[-1]
            last_keypoints = np.asarray(ref["keypoints"], dtype=float)
            blend = self.kf_prediction_blend
            pred_centroid = self._predicted_centroid(mean)
            last_centroid = self._centroid(last_keypoints)

            if np.isnan(pred_centroid).any() or np.isnan(last_centroid).any():
                candidate_keypoints = last_keypoints  # hold last (no valid prediction)
            elif self.kf_track_features == "keypoints":
                # Per-node mode: blend each tracked node's predicted position with its
                # last observed position; non-tracked nodes are translated rigidly by
                # the mean tracked displacement. Uses the pose directly (noisier than
                # the rigid centroid candidate), with a tolerant similarity score
                # (`oks_stddev`) or bbox/iou recommended.
                idx = self._resolved_node_indices
                pred_points = self._predicted_points(mean)  # (K, 2)
                last_tracked = last_keypoints[idx]
                disp = pred_points - last_tracked
                blended = last_tracked + blend * disp
                blended = np.where(np.isnan(blended), pred_points, blended)
                candidate_keypoints = last_keypoints.copy()
                candidate_keypoints[idx] = blended
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_disp = np.nanmean(disp, axis=0)
                if not np.isnan(mean_disp).any():
                    mask = np.ones(self._n_nodes, dtype=bool)
                    mask[idx] = False
                    candidate_keypoints[mask] = last_keypoints[mask] + blend * mean_disp
            else:
                # Centroid mode: rigidly translate the last observed pose by a fraction
                # of the predicted centroid displacement. The weight is constant (NOT
                # scaled up with staleness): a coasting prediction is *less* reliable,
                # so amplifying it during gaps just injects swaps under noise.
                displacement = blend * (pred_centroid - last_centroid)
                candidate_keypoints = last_keypoints + displacement

            predicted[track_id].append(
                TrackedInstanceFeature(
                    feature=feature_method(candidate_keypoints),
                    src_predicted_instance=ref["src"],
                    frame_idx=ref["frame_idx"],
                    tracking_score=ref["score"] if ref["score"] is not None else 1.0,
                    shifted_keypoints=candidate_keypoints,
                )
            )
        return predicted


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
            # Update last_good_frame_tracks when we have at least as many instances
            # as before. This prevents stale reference when max_instances doesn't
            # match actual count or when first frame has fewer instances.
            if len(frame_tracks) >= len(last_good_frame_tracks):
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
    oks_stddev: Optional[float] = None,
    track_matching_method: str = "hungarian",
    max_tracks: Optional[int] = None,
    use_flow: bool = False,
    of_img_scale: float = 1.0,
    of_window_size: int = 21,
    of_max_levels: int = 3,
    use_kalman: bool = False,
    kf_track_features: str = "centroid",
    kf_init_frame_count: int = 10,
    kf_node_indices: Optional[List[int]] = None,
    kf_reset_gap_size: int = 5,
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
        oks_stddev: Keypoint-spread normalization constant for `oks` scoring; larger is
            more tolerant of localization error. `None` (default) auto-resolves to 0.1
            for `kf_track_features="keypoints"` and 0.025 otherwise.
        use_kalman: If True, `KalmanShiftTracker` is used, where poses are predicted with
            a per-track constant-velocity Kalman filter. Requires
            `tracking_target_instance_count` (or `max_tracks`) and is mutually exclusive
            with `use_flow`. Default: `False`.
        kf_track_features: What the Kalman motion model tracks: `centroid` (default) or
            `keypoints` (per-node poses; noisier). (only if `use_kalman` is True)
        kf_init_frame_count: Number of warm-up frames tracked with the base path before
            the Kalman filters are fit via EM. Default: 10. (only if `use_kalman` is True)
        kf_node_indices: Skeleton node (row) indices to track with the motion model.
            `None` uses all nodes. Default: None. (only if `use_kalman` is True)
        kf_reset_gap_size: Number of consecutive missed frames after which a stale track's
            filter is reset. Default: 5. (only if `use_kalman` is True)
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
        oks_stddev=oks_stddev,
        track_matching_method=track_matching_method,
        max_tracks=max_tracks,
        use_flow=use_flow,
        of_img_scale=of_img_scale,
        of_window_size=of_window_size,
        of_max_levels=of_max_levels,
        use_kalman=use_kalman,
        kf_track_features=kf_track_features,
        kf_init_frame_count=kf_init_frame_count,
        kf_node_indices=kf_node_indices,
        kf_reset_gap_size=kf_reset_gap_size,
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
