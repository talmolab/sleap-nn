"""Entry point for running inference."""

from loguru import logger
from typing import Optional, List, Union
import click
from sleap_nn.inference.predictors import (
    Predictor,
    BottomUpPredictor,
    BottomUpMultiClassPredictor,
    TopDownMultiClassPredictor,
)
from sleap_nn.tracking.tracker import (
    Tracker,
    run_tracker,
    connect_single_breaks,
    cull_instances,
)
from sleap_nn.system_info import get_startup_info_string
from sleap_nn.inference.provenance import (
    build_inference_provenance,
    build_tracking_only_provenance,
)
from omegaconf import OmegaConf
import sleap_io as sio
from pathlib import Path
from datetime import datetime
from time import time
import torch


def frame_list(frame_str: str) -> Optional[List[int]]:
    """Converts 'n-m' string to list of ints.

    Args:
        frame_str: string representing range

    Returns:
        List of ints, or None if string does not represent valid range.
    """
    # Handle ranges of frames. Must be of the form "1-200" (or "1,-200")
    if "-" in frame_str:
        min_max = frame_str.split("-")
        min_frame = int(min_max[0].rstrip(","))
        max_frame = int(min_max[1])
        return list(range(min_frame, max_frame + 1))

    return [int(x) for x in frame_str.split(",")] if len(frame_str) else None


def run_inference(
    data_path: Optional[str] = None,
    input_labels: Optional[sio.Labels] = None,
    input_video: Optional[sio.Video] = None,
    model_paths: Optional[List[str]] = None,
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    max_instances: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    ensure_rgb: Optional[bool] = None,
    input_scale: Optional[float] = None,
    ensure_grayscale: Optional[bool] = None,
    anchor_part: Optional[str] = None,
    only_labeled_frames: bool = False,
    only_suggested_frames: bool = False,
    exclude_user_labeled: bool = False,
    only_predicted_frames: bool = False,
    no_empty_frames: bool = False,
    batch_size: int = 4,
    queue_maxsize: int = 8,
    video_index: Optional[int] = None,
    video_dataset: Optional[str] = None,
    video_input_format: str = "channels_last",
    frames: Optional[list] = None,
    crop_size: Optional[int] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
    filter_overlapping: bool = False,
    filter_overlapping_method: str = "iou",
    filter_overlapping_threshold: float = 0.8,
    integral_refinement: Optional[str] = "integral",
    integral_patch_size: int = 5,
    return_confmaps: bool = False,
    return_pafs: bool = False,
    return_paf_graph: bool = False,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: Union[int, float] = 0,
    min_line_scores: float = 0.25,
    return_class_maps: bool = False,
    return_class_vectors: bool = False,
    make_labels: bool = True,
    output_path: Optional[str] = None,
    device: str = "auto",
    tracking: bool = False,
    tracking_window_size: int = 5,
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
    gui: bool = False,
):
    """Entry point to run inference on trained SLEAP-NN models.

    Args:
        data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
        input_labels: (sio.Labels) Labels object to run inference on. This is an alternative to specifying the data_path.
        input_video: (sio.Video) Video to run inference on. This is an alternative to specifying the data_path. If both input_labels and input_video are provided, input_labels are used.
        model_paths: (List[str]) List of paths to the directory where the best.ckpt
                and training_config.yaml are saved.
        backbone_ckpt_path: (str) To run inference on any `.ckpt` other than `best.ckpt`
                from the `model_paths` dir, the path to the `.ckpt` file should be passed here.
        head_ckpt_path: (str) Path to `.ckpt` file if a different set of head layer weights
                are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt
                from `backbone_ckpt_path` if provided.)
        max_instances: (int) Max number of instances to consider from the predictions.
        max_width: (int) Maximum width the image should be padded to. If not provided, the
                values from the training config are used. Default: None.
        max_height: (int) Maximum height the image should be padded to. If not provided, the
                values from the training config are used. Default: None.
        input_scale: (float) Scale factor to apply to the input image. If not provided, the
                values from the training config are used. Default: None.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
                channel when this is set to `True`, then the images from single-channel
                is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. If not provided, the
                values from the training config are used. Default: `None`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
                is set to True, then we convert the image to grayscale (single-channel)
                image. If the source image has only one channel and this is set to False, then we retain the single channel input. If not provided, the
                values from the training config are used. Default: `None`.
        anchor_part: (str) The node name to use as the anchor for the centroid. If not
                provided, the anchor part in the `training_config.yaml` is used. Default: `None`.
        only_labeled_frames: (bool) `True` if inference should be run only on user-labeled frames. Default: `False`.
        only_suggested_frames: (bool) `True` if inference should be run only on unlabeled suggested frames. Default: `False`.
        exclude_user_labeled: (bool) `True` to skip frames that have user-labeled instances. Default: `False`.
        only_predicted_frames: (bool) `True` to run inference only on frames that already have predictions. Default: `False`.
        no_empty_frames: (bool) `True` if empty frames that did not have predictions should be cleared before saving to output. Default: `False`.
        batch_size: (int) Number of samples per batch. Default: 4.
        queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
        video_index: (int) Integer index of video in .slp file to predict on. To be used with
                an .slp path as an alternative to specifying the video path.
        video_dataset: (str) The dataset for HDF5 videos.
        video_input_format: (str) The input_format for HDF5 videos.
        frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
        crop_size: (int) Crop size. If not provided, the crop size from training_config.yaml is used.
                If `input_scale` is provided, then the cropped image will be resized according to `input_scale`. Default: None.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
        filter_overlapping: (bool) If True, removes overlapping instances after
                inference using greedy NMS. Applied independently of tracking.
                Default: False.
        filter_overlapping_method: (str) Similarity metric for filtering overlapping
                instances. One of "iou" (bounding box) or "oks" (keypoint similarity).
                Default: "iou".
        filter_overlapping_threshold: (float) Similarity threshold for filtering.
                Instances with similarity > threshold are removed (keeping higher-scoring).
                Typical values: 0.3 (aggressive) to 0.8 (permissive). Default: 0.8.
        integral_refinement: (str) If `None`, returns the grid-aligned peaks with no refinement.
                If `"integral"`, peaks will be refined with integral regression.
                Default: `"integral"`.
        integral_patch_size: (int) Size of patches to crop around each rough peak as an
                integer scalar. Default: 5.
        return_confmaps: (bool) If `True`, predicted confidence maps will be returned
                along with the predicted peak values and points. Default: False.
        return_pafs: (bool) If `True`, the part affinity fields will be returned together with
                the predicted instances. This will result in slower inference times since
                the data must be copied off of the GPU, but is useful for visualizing the
                raw output of the model. Default: False.
        return_class_vectors: If `True`, the classification probabilities will be
                returned together with the predicted peaks. This will not line up with the
                grouped instances, for which the associtated class probabilities will always
                be returned in `"instance_scores"`.
        return_paf_graph: (bool) If `True`, the part affinity field graph will be returned
                together with the predicted instances. The graph is obtained by parsing the
                part affinity fields with the `paf_scorer` instance and is an intermediate
                representation used during instance grouping. Default: False.
        max_edge_length_ratio: (float) The maximum expected length of a connected pair of points
                as a fraction of the image size. Candidate connections longer than this
                length will be penalized during matching. Default: 0.25.
        dist_penalty_weight: (float) A coefficient to scale weight of the distance penalty as
                a scalar float. Set to values greater than 1.0 to enforce the distance
                penalty more strictly.Default: 1.0.
        n_points: (int) Number of points to sample along the line integral. Default: 10.
        min_instance_peaks: Union[int, float] Minimum number of peaks the instance should
                have to be considered a real instance. Instances with fewer peaks than
                this will be discarded (useful for filtering spurious detections).
                Default: 0.
        min_line_scores: (float) Minimum line score (between -1 and 1) required to form a match
                between candidate point pairs. Useful for rejecting spurious detections when
                there are no better ones. Default: 0.25.
        return_class_maps: If `True`, the class maps will be returned together with
            the predicted instances. This will result in slower inference times since
            the data must be copied off of the GPU, but is useful for visualizing the
            raw output of the model.
        make_labels: (bool) If `True` (the default), returns a `sio.Labels` instance with
                `sio.PredictedInstance`s. If `False`, just return a list of
                dictionaries containing the raw arrays returned by the inference model.
                Default: True.
        output_path: (str) Path to save the labels file if `make_labels` is True.
                Default is current working directory.
        device: (str) Device on which torch.Tensor will be allocated. One of the
                ('cpu', 'cuda', 'mps', 'auto').
                Default: "auto" (based on available backend either cuda, mps or cpu is chosen). If `cuda` is available, you could also use `cuda:0` to specify the device.
        tracking: (bool) If True, runs tracking on the predicted instances.
        tracking_window_size: Number of frames to look for in the candidate instances to match
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
        gui: (bool) If True, outputs JSON progress lines for GUI integration instead
                of Rich progress bars. Default: False.

    Returns:
        Returns `sio.Labels` object if `make_labels` is True. Else this function returns
            a list of Dictionaries with the predictions.

    """
    preprocess_config = {  # if not given, then use from training config
        "ensure_rgb": ensure_rgb,
        "ensure_grayscale": ensure_grayscale,
        "crop_size": crop_size,
        "max_width": max_width,
        "max_height": max_height,
        "scale": input_scale,
    }

    # Validate mutually exclusive frame filter flags
    if only_labeled_frames and exclude_user_labeled:
        message = (
            "--only_labeled_frames and --exclude_user_labeled are mutually exclusive "
            "(would result in zero frames)"
        )
        logger.error(message)
        raise ValueError(message)

    if (
        only_predicted_frames
        and data_path is not None
        and not data_path.endswith(".slp")
    ):
        message = (
            "--only_predicted_frames requires a .slp file input "
            "(need Labels to know which frames have predictions)"
        )
        logger.error(message)
        raise ValueError(message)

    if model_paths is None or not len(
        model_paths
    ):  # if model paths is not provided, run tracking-only pipeline.
        if not tracking:
            message = """Neither tracker nor path to trained models specified. Use `model_paths` to specify models to use. To retrack on predictions, set `tracking` to True."""
            logger.error(message)
            raise ValueError(message)

        else:
            if (data_path is not None and not data_path.endswith(".slp")) or (
                input_labels is not None and not isinstance(input_labels, sio.Labels)
            ):
                message = "Data path is not a .slp file. To run track-only pipeline, data path must be an .slp file."
                logger.error(message)
                raise ValueError(message)

            start_inf_time = time()
            start_datetime = datetime.now()
            start_timestamp = str(start_datetime)
            logger.info(f"Started tracking at: {start_timestamp}")

            labels = sio.load_slp(data_path) if input_labels is None else input_labels

            lf_frames = labels.labeled_frames

            # select video if video_index is provided
            if video_index is not None:
                lf_frames = labels.find(video=labels.videos[video_index])

            # sort frames before tracking
            lf_frames = sorted(lf_frames, key=lambda lf: lf.frame_idx)

            if frames is not None:
                filtered_frames = []
                for lf in lf_frames:
                    if lf.frame_idx in frames:
                        filtered_frames.append(lf)
                lf_frames = filtered_frames

            if post_connect_single_breaks:
                if max_tracks is None:
                    max_tracks = max_instances

            logger.info(f"Running tracking on {len(lf_frames)} frames...")

            if post_connect_single_breaks or tracking_pre_cull_to_target:
                if tracking_target_instance_count is None and max_instances is None:
                    features_requested = []
                    if post_connect_single_breaks:
                        features_requested.append("--post_connect_single_breaks")
                    if tracking_pre_cull_to_target:
                        features_requested.append("--tracking_pre_cull_to_target")
                    features_str = " and ".join(features_requested)

                    if max_tracks is not None:
                        suggestion = f"Add --tracking_target_instance_count {max_tracks} to your command (using your --max_tracks value)."
                    else:
                        suggestion = "Add --tracking_target_instance_count N where N is the expected number of instances per frame."

                    message = (
                        f"{features_str} requires --tracking_target_instance_count to be set. "
                        f"{suggestion}"
                    )
                    logger.error(message)
                    raise ValueError(message)
                elif tracking_target_instance_count is None:
                    tracking_target_instance_count = max_instances

            tracked_frames = run_tracker(
                untracked_frames=lf_frames,
                window_size=tracking_window_size,
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
                post_connect_single_breaks=post_connect_single_breaks,
                tracking_target_instance_count=tracking_target_instance_count,
                tracking_pre_cull_to_target=tracking_pre_cull_to_target,
                tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
                tracking_clean_instance_count=tracking_clean_instance_count,
                tracking_clean_iou_threshold=tracking_clean_iou_threshold,
            )

            end_datetime = datetime.now()
            finish_timestamp = str(end_datetime)
            total_elapsed = time() - start_inf_time
            logger.info(f"Finished tracking at: {finish_timestamp}")
            logger.info(f"Total runtime: {total_elapsed} secs")

            # Build tracking-only provenance
            tracking_params = {
                "window_size": tracking_window_size,
                "min_new_track_points": min_new_track_points,
                "candidates_method": candidates_method,
                "min_match_points": min_match_points,
                "features": features,
                "scoring_method": scoring_method,
                "scoring_reduction": scoring_reduction,
                "robust_best_instance": robust_best_instance,
                "track_matching_method": track_matching_method,
                "max_tracks": max_tracks,
                "use_flow": use_flow,
                "post_connect_single_breaks": post_connect_single_breaks,
            }
            provenance = build_tracking_only_provenance(
                input_labels=labels,
                input_path=data_path,
                start_time=start_datetime,
                end_time=end_datetime,
                tracking_params=tracking_params,
                frames_processed=len(tracked_frames),
            )

            output = sio.Labels(
                labeled_frames=tracked_frames,
                videos=labels.videos,
                skeletons=labels.skeletons,
                provenance=provenance,
            )

    else:
        start_inf_time = time()
        start_datetime = datetime.now()
        start_timestamp = str(start_datetime)
        logger.info(f"Started inference at: {start_timestamp}")
        logger.info(get_startup_info_string())

        # Convert device to string if it's a torch.device object
        if hasattr(device, "type"):
            device = str(device)

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        logger.info(f"Using device: {device}")

        # initializes the inference model
        predictor = Predictor.from_model_paths(
            model_paths,
            backbone_ckpt_path=backbone_ckpt_path,
            head_ckpt_path=head_ckpt_path,
            peak_threshold=peak_threshold,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            batch_size=batch_size,
            max_instances=max_instances,
            return_confmaps=return_confmaps,
            device=device,
            preprocess_config=OmegaConf.create(preprocess_config),
            anchor_part=anchor_part,
        )

        # Set GUI mode for progress output
        predictor.gui = gui

        if (
            tracking
            and not isinstance(predictor, BottomUpMultiClassPredictor)
            and not isinstance(predictor, TopDownMultiClassPredictor)
        ):
            if post_connect_single_breaks or tracking_pre_cull_to_target:
                if tracking_target_instance_count is None and max_instances is None:
                    features_requested = []
                    if post_connect_single_breaks:
                        features_requested.append("--post_connect_single_breaks")
                    if tracking_pre_cull_to_target:
                        features_requested.append("--tracking_pre_cull_to_target")
                    features_str = " and ".join(features_requested)

                    if max_tracks is not None:
                        suggestion = f"Add --tracking_target_instance_count {max_tracks} to your command (using your --max_tracks value)."
                    else:
                        suggestion = "Add --tracking_target_instance_count N or --max_instances N where N is the expected number of instances per frame."

                    message = (
                        f"{features_str} requires --tracking_target_instance_count or --max_instances to be set. "
                        f"{suggestion}"
                    )
                    logger.error(message)
                    raise ValueError(message)
                elif tracking_target_instance_count is None:
                    tracking_target_instance_count = max_instances
            predictor.tracker = Tracker.from_config(
                candidates_method=candidates_method,
                min_match_points=min_match_points,
                window_size=tracking_window_size,
                min_new_track_points=min_new_track_points,
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

        if isinstance(predictor, BottomUpPredictor):
            predictor.inference_model.paf_scorer.max_edge_length_ratio = (
                max_edge_length_ratio
            )
            predictor.inference_model.paf_scorer.dist_penalty_weight = (
                dist_penalty_weight
            )
            predictor.inference_model.return_pafs = return_pafs
            predictor.inference_model.return_paf_graph = return_paf_graph
            predictor.inference_model.paf_scorer.max_edge_length_ratio = (
                max_edge_length_ratio
            )
            predictor.inference_model.paf_scorer.min_line_scores = min_line_scores
            predictor.inference_model.paf_scorer.min_instance_peaks = min_instance_peaks
            predictor.inference_model.paf_scorer.n_points = n_points

        if isinstance(predictor, BottomUpMultiClassPredictor):
            predictor.inference_model.return_class_maps = return_class_maps

        if isinstance(predictor, TopDownMultiClassPredictor):
            predictor.inference_model.instance_peaks.return_class_vectors = (
                return_class_vectors
            )

        # initialize make_pipeline function

        predictor.make_pipeline(
            inference_object=(
                input_labels
                if input_labels is not None
                else input_video if input_video is not None else data_path
            ),
            queue_maxsize=queue_maxsize,
            frames=frames,
            only_labeled_frames=only_labeled_frames,
            only_suggested_frames=only_suggested_frames,
            exclude_user_labeled=exclude_user_labeled,
            only_predicted_frames=only_predicted_frames,
            video_index=video_index,
            video_dataset=video_dataset,
            video_input_format=video_input_format,
        )

        # run predict
        output = predictor.predict(
            make_labels=make_labels,
        )

        # Filter overlapping instances (independent of tracking)
        if filter_overlapping and make_labels:
            from sleap_nn.inference.postprocessing import filter_overlapping_instances

            output = filter_overlapping_instances(
                output,
                threshold=filter_overlapping_threshold,
                method=filter_overlapping_method,
            )
            logger.info(
                f"Filtered overlapping instances with {filter_overlapping_method.upper()} "
                f"threshold: {filter_overlapping_threshold}"
            )

        if tracking:
            lfs = [x for x in output]
            if tracking_clean_instance_count > 0:
                lfs = cull_instances(
                    lfs, tracking_clean_instance_count, tracking_clean_iou_threshold
                )
                if not post_connect_single_breaks:
                    corrected_lfs = connect_single_breaks(
                        lfs, tracking_clean_instance_count
                    )
            elif post_connect_single_breaks:
                start_final_pass_time = time()
                start_fp_timestamp = str(datetime.now())
                logger.info(
                    f"Started final-pass (connecting single breaks) at: {start_fp_timestamp}"
                )
                corrected_lfs = connect_single_breaks(
                    lfs, max_instances=tracking_target_instance_count
                )
                finish_fp_timestamp = str(datetime.now())
                total_fp_elapsed = time() - start_final_pass_time
                logger.info(
                    f"Finished final-pass (connecting single breaks) at: {finish_fp_timestamp}"
                )
                logger.info(f"Total runtime: {total_fp_elapsed} secs")
            else:
                corrected_lfs = lfs

            output = sio.Labels(
                labeled_frames=corrected_lfs,
                videos=output.videos,
                skeletons=output.skeletons,
            )

        end_datetime = datetime.now()
        finish_timestamp = str(end_datetime)
        total_elapsed = time() - start_inf_time
        logger.info(f"Finished inference at: {finish_timestamp}")
        logger.info(f"Total runtime: {total_elapsed} secs")

        # Determine input labels for provenance preservation
        input_labels_for_prov = None
        if input_labels is not None:
            input_labels_for_prov = input_labels
        elif data_path is not None and data_path.endswith(".slp"):
            # Load input labels to preserve provenance (if not already loaded)
            try:
                input_labels_for_prov = sio.load_slp(data_path)
            except Exception:
                pass

        # Build inference parameters for provenance
        inference_params = {
            "peak_threshold": peak_threshold,
            "filter_overlapping": filter_overlapping,
            "filter_overlapping_method": filter_overlapping_method,
            "filter_overlapping_threshold": filter_overlapping_threshold,
            "integral_refinement": integral_refinement,
            "integral_patch_size": integral_patch_size,
            "batch_size": batch_size,
            "max_instances": max_instances,
            "crop_size": crop_size,
            "input_scale": input_scale,
            "anchor_part": anchor_part,
        }

        # Build tracking parameters if tracking was enabled
        tracking_params_prov = None
        if tracking:
            tracking_params_prov = {
                "window_size": tracking_window_size,
                "min_new_track_points": min_new_track_points,
                "candidates_method": candidates_method,
                "min_match_points": min_match_points,
                "features": features,
                "scoring_method": scoring_method,
                "scoring_reduction": scoring_reduction,
                "robust_best_instance": robust_best_instance,
                "track_matching_method": track_matching_method,
                "max_tracks": max_tracks,
                "use_flow": use_flow,
                "post_connect_single_breaks": post_connect_single_breaks,
            }

        # Determine frame selection method
        frame_selection_method = "all"
        if only_labeled_frames:
            frame_selection_method = "labeled"
        elif only_suggested_frames:
            frame_selection_method = "suggested"
        elif only_predicted_frames:
            frame_selection_method = "predicted"
        elif frames is not None:
            frame_selection_method = "specified"

        # Determine model type from predictor class
        predictor_type_map = {
            "TopDownPredictor": "top_down",
            "SingleInstancePredictor": "single_instance",
            "BottomUpPredictor": "bottom_up",
            "BottomUpMultiClassPredictor": "bottom_up_multi_class",
            "TopDownMultiClassPredictor": "top_down_multi_class",
        }
        model_type = predictor_type_map.get(type(predictor).__name__)

        # Build and set provenance (only for Labels objects)
        if make_labels and isinstance(output, sio.Labels):
            provenance = build_inference_provenance(
                model_paths=model_paths,
                model_type=model_type,
                start_time=start_datetime,
                end_time=end_datetime,
                input_labels=input_labels_for_prov,
                input_path=data_path,
                frames_processed=(
                    len(output.labeled_frames)
                    if hasattr(output, "labeled_frames")
                    else None
                ),
                frame_selection_method=frame_selection_method,
                inference_params=inference_params,
                tracking_params=tracking_params_prov,
                device=device,
            )
            output.provenance = provenance

    if no_empty_frames:
        output.clean(frames=True, skeletons=False)

    if make_labels:
        if output_path is None:
            base_path = Path(data_path if data_path is not None else "results")

            # If video_index is specified, append video name to output path
            if video_index is not None and len(output.videos) > video_index:
                video = output.videos[video_index]
                # Get video filename and sanitize it for use in path
                video_name = (
                    Path(video.filename).stem
                    if isinstance(video.filename, str)
                    else f"video_{video_index}"
                )
                # Insert video name before .predictions.slp extension
                output_path = (
                    base_path.parent / f"{base_path.stem}.{video_name}.predictions.slp"
                )
            else:
                output_path = base_path.with_suffix(".predictions.slp")
        output.save(Path(output_path).as_posix(), restore_original_videos=False)
    finish_timestamp = str(datetime.now())
    logger.info(f"Predictions output path: {output_path}")
    logger.info(f"Saved file at: {finish_timestamp}")

    return output
