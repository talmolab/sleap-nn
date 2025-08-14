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
from sleap_nn.tracking.tracker import Tracker, run_tracker, connect_single_breaks
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
    data_path: str,
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
    batch_size: int = 4,
    queue_maxsize: int = 8,
    video_index: Optional[int] = None,
    video_dataset: Optional[str] = None,
    video_input_format: str = "channels_last",
    frames: Optional[list] = None,
    crop_size: Optional[int] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
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
):
    """Entry point to run inference on trained SLEAP-NN models.

    Args:
        data_path: (str) Path to `.slp` file or `.mp4` to run inference on.
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
        batch_size: (int) Number of samples per batch. Default: 4.
        queue_maxsize: (int) Maximum size of the frame buffer queue. Default: 8.
        video_index: (int) Integer index of video in .slp file to predict on. To be used with
                an .slp path as an alternative to specifying the video path.
        video_dataset: (str) The dataset for HDF5 videos.
        video_input_format: (str) The input_format for HDF5 videos.
        frames: (list) List of frames indices. If `None`, all frames in the video are used. Default: None.
        crop_size: (int) Crop size. If not provided, the crop size from training_config.yaml is used.
                Default: None.
        peak_threshold: (float) Minimum confidence threshold. Peaks with values below
                this will be ignored. Default: 0.2. This can also be `List[float]` for topdown
                centroid and centered-instance model, where the first element corresponds
                to centroid model peak finding threshold and the second element is for
                centered-instance model peak finding.
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
                ('cpu', 'cuda', 'mps', 'auto', 'opencl', 'ideep', 'hip', 'msnpu').
                Default: "auto" (based on available backend either cuda, mps or cpu is chosen).
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

    Returns:
        Returns `sio.Labels` object if `make_labels` is True. Else this function returns
            a list of Dictionaries with the predictions.

    """
    preprocess_config = {  # if not given, then use from training config
        "ensure_rgb": ensure_rgb,
        "ensure_grayscale": ensure_grayscale,
        "crop_hw": (crop_size, crop_size) if crop_size is not None else None,
        "max_width": max_width,
        "max_height": max_height,
        "scale": input_scale,
    }

    if model_paths is None or not len(
        model_paths
    ):  # if model paths is not provided, run tracking-only pipeline.
        if not tracking:
            message = """Neither tracker nor path to trained models specified. Use `model_paths` to specify models to use. To retrack on predictions, set `tracking` to True."""
            logger.error(message)
            raise ValueError(message)

        else:
            start_inf_time = time()
            start_timestamp = str(datetime.now())
            logger.info(f"Started tracking at: {start_timestamp}")

            labels = sio.load_slp(data_path)
            frames = sorted(labels.labeled_frames, key=lambda lf: lf.frame_idx)

            if post_connect_single_breaks:
                if max_tracks is None:
                    max_tracks = max_instances

            tracked_frames = run_tracker(
                untracked_frames=frames,
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
            )

            finish_timestamp = str(datetime.now())
            total_elapsed = time() - start_inf_time
            logger.info(f"Finished tracking at: {finish_timestamp}")
            logger.info(f"Total runtime: {total_elapsed} secs")

            output = sio.Labels(
                labeled_frames=tracked_frames,
                videos=labels.videos,
                skeletons=labels.skeletons,
            )

    else:
        start_inf_time = time()
        start_timestamp = str(datetime.now())
        logger.info(f"Started inference at: {start_timestamp}")

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        if integral_refinement is not None and device == "mps":  # TODO
            # kornia/geometry/transform/imgwarp.py:382: in get_perspective_transform. NotImplementedError: The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
            logger.info(
                "Integral refinement is not supported with MPS device. Using CPU."
            )
            device = "cpu"  # not supported with mps

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

        if (
            tracking
            and not isinstance(predictor, BottomUpMultiClassPredictor)
            and not isinstance(predictor, TopDownMultiClassPredictor)
        ):
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
            data_path,
            queue_maxsize,
            frames,
            only_labeled_frames,
            only_suggested_frames,
            video_index=video_index,
            video_dataset=video_dataset,
            video_input_format=video_input_format,
        )

        # run predict
        output = predictor.predict(
            make_labels=make_labels,
        )

        if tracking and post_connect_single_breaks:
            if max_tracks is None:
                max_tracks = max_instances

            if max_tracks is None:
                message = "Max_tracks (and max instances) is None. To connect single breaks, max_tracks should be set to an integer."
                logger.error(message)
                raise ValueError(message)

            start_final_pass_time = time()
            start_fp_timestamp = str(datetime.now())
            logger.info(
                f"Started final-pass (connecting single breaks) at: {start_fp_timestamp}"
            )
            corrected_lfs = connect_single_breaks(
                lfs=[x for x in output], max_instances=max_tracks
            )
            finish_fp_timestamp = str(datetime.now())
            total_fp_elapsed = time() - start_final_pass_time
            logger.info(
                f"Finished final-pass (connecting single breaks) at: {finish_fp_timestamp}"
            )
            logger.info(f"Total runtime: {total_fp_elapsed} secs")

            output = sio.Labels(
                labeled_frames=corrected_lfs,
                videos=output.videos,
                skeletons=output.skeletons,
            )

        finish_timestamp = str(datetime.now())
        total_elapsed = time() - start_inf_time
        logger.info(f"Finished inference at: {finish_timestamp}")
        logger.info(
            f"Total runtime: {total_elapsed} secs"
        )  # TODO: add number of predicted frames

    if make_labels:
        if output_path is None:
            output_path = Path(data_path).with_suffix(".predictions.slp")
        output.save(Path(output_path).as_posix(), restore_original_videos=False)
    finish_timestamp = str(datetime.now())
    logger.info(f"Predictions output path: {output_path}")
    logger.info(f"Saved file at: {finish_timestamp}")

    return output


@click.command()
@click.option(
    "--data_path",
    type=str,
    required=True,
    help="Path to data to predict on. This can be a labels (.slp) file or any supported video format.",
)
@click.option(
    "--model_paths",
    multiple=True,
    help="Path to trained model directory (with training_config.json). Multiple models can be specified, each preceded by --model_paths.",
)
@click.option(
    "--backbone_ckpt_path",
    type=str,
    default=None,
    help="To run inference on any `.ckpt` other than `best.ckpt` from the `model_paths` dir, the path to the `.ckpt` file should be passed here.",
)
@click.option(
    "--head_ckpt_path",
    type=str,
    default=None,
    help="Path to `.ckpt` file if a different set of head layer weights are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt from `backbone_ckpt_path` if provided.)",
)
@click.option(
    "-n",
    "--max_instances",
    type=int,
    default=None,
    help="Limit maximum number of instances in multi-instance models. Not available for ID models. Defaults to None.",
)
@click.option(
    "--max_height",
    type=int,
    default=None,
    help="Maximum height the image should be padded to. If not provided, the values from the training config are used. Default: None.",
)
@click.option(
    "--max_width",
    type=int,
    default=None,
    help="Maximum width the image should be padded to. If not provided, the values from the training config are used. Default: None.",
)
@click.option(
    "--input_scale",
    type=float,
    default=None,
    help="Scale factor to apply to the input image. If not provided, the values from the training config are used. Default: None.",
)
@click.option(
    "--ensure_rgb",
    is_flag=True,
    default=False,
    help="True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. If not provided, the values from the training config are used. Default: `None`.",
)
@click.option(
    "--ensure_grayscale",
    is_flag=True,
    default=False,
    help="True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. If not provided, the values from the training config are used. Default: `None`.",
)
@click.option(
    "--anchor_part",
    type=str,
    default=None,
    help="The node name to use as the anchor for the centroid. If not provided, the anchor part in the `training_config.yaml` is used. Default: `None`.",
)
@click.option(
    "--only_labeled_frames",
    is_flag=True,
    default=False,
    help="Only run inference on user labeled frames when running on labels dataset. This is useful for generating predictions to compare against ground truth.",
)
@click.option(
    "--only_suggested_frames",
    is_flag=True,
    default=False,
    help="Only run inference on unlabeled suggested frames when running on labels dataset. This is useful for generating predictions for initialization during labeling.",
)
@click.option(
    "--video_index",
    type=int,
    default=None,
    help="Integer index of video in .slp file to predict on. To be used with an .slp path as an alternative to specifying the video path.",
)
@click.option(
    "--video_dataset", type=str, default=None, help="The dataset for HDF5 videos."
)
@click.option(
    "--video_input_format",
    type=str,
    default="channels_last",
    help="The input_format for HDF5 videos.",
)
@click.option(
    "--frames",
    type=str,
    default="",
    help="List of frames to predict when running on a video. Can be specified as a comma separated list (e.g. 1,2,3) or a range separated by hyphen (e.g., 1-3, for 1,2,3). If not provided, defaults to predicting on the entire video.",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    help="Number of frames to predict at a time. Larger values result in faster inference speeds, but require more memory.",
)
@click.option(
    "--integral_patch_size",
    type=int,
    default=5,
    help="Size of patches to crop around each rough peak as an integer scalar. Default: 5.",
)
@click.option(
    "--return_confmaps",
    is_flag=True,
    default=False,
    help="If True, predicted confidence maps will be returned along with the predicted peak values and points. Default: False.",
)
@click.option(
    "--return_pafs",
    is_flag=True,
    default=False,
    help="If True, the part affinity fields will be returned together with the predicted instances. This will result in slower inference times since the data must be copied off of the GPU, but is useful for visualizing the raw output of the model. Default: False.",
)
@click.option(
    "--return_paf_graph",
    is_flag=True,
    default=False,
    help="If True, the part affinity field graph will be returned together with the predicted instances. The graph is obtained by parsing the part affinity fields with the paf_scorer instance and is an intermediate representation used during instance grouping. Default: False.",
)
@click.option(
    "--max_edge_length_ratio",
    type=float,
    default=0.25,
    help="The maximum expected length of a connected pair of points as a fraction of the image size. Candidate connections longer than this length will be penalized during matching. Default: 0.25.",
)
@click.option(
    "--dist_penalty_weight",
    type=float,
    default=1.0,
    help="A coefficient to scale weight of the distance penalty as a scalar float. Set to values greater than 1.0 to enforce the distance penalty more strictly. Default: 1.0.",
)
@click.option(
    "--n_points",
    type=int,
    default=10,
    help="Number of points to sample along the line integral. Default: 10.",
)
@click.option(
    "--min_instance_peaks",
    type=float,
    default=0,
    help="Minimum number of peaks the instance should have to be considered a real instance. Instances with fewer peaks than this will be discarded (useful for filtering spurious detections). Default: 0.",
)
@click.option(
    "--min_line_scores",
    type=float,
    default=0.25,
    help="Minimum line score (between -1 and 1) required to form a match between candidate point pairs. Useful for rejecting spurious detections when there are no better ones. Default: 0.25.",
)
@click.option(
    "--return_class_maps",
    is_flag=True,
    default=False,
    help="If True, the class maps will be returned together with the predicted instances. This will result in slower inference times since the data must be copied off of the GPU, but is useful for visualizing the raw output of the model.",
)
@click.option(
    "--return_class_vectors",
    is_flag=True,
    default=False,
    help="If True, the classification probabilities will be returned together with the predicted peaks. This will not line up with the grouped instances, for which the associated class probabilities will always be returned in instance_scores.",
)
@click.option(
    "--make_labels",
    is_flag=True,
    default=True,
    help="If True (the default), returns a sio.Labels instance with sio.PredictedInstances. If False, just return a list of dictionaries containing the raw arrays returned by the inference model. Default: True.",
)
@click.option(
    "--queue_maxsize",
    type=int,
    default=8,
    help="Maximum size of the frame buffer queue.",
)
@click.option(
    "--crop_size",
    type=int,
    default=None,
    help="Crop size. If not provided, the crop size from training_config.yaml is used.",
)
@click.option(
    "--peak_threshold",
    type=float,
    default=0.2,
    help="Minimum confidence map value to consider a peak as valid.",
)
@click.option(
    "--integral_refinement",
    type=str,
    default="integral",
    help="If `None`, returns the grid-aligned peaks with no refinement. If `'integral'`, peaks will be refined with integral regression. Default: 'integral'.",
)
@click.option(
    "-o",
    "--output_path",
    type=str,
    default=None,
    help="The output filename to use for the predicted data. If not provided, defaults to '[data_path].slp'.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device on which torch.Tensor will be allocated. One of the ('cpu', 'cuda', 'mps', 'auto', 'opencl', 'ideep', 'hip', 'msnpu'). Default: 'auto' (based on available backend either cuda, mps or cpu is chosen).",
)
@click.option(
    "--tracking",
    is_flag=True,
    default=False,
    help="If True, runs tracking on the predicted instances.",
)
@click.option(
    "--tracking_window_size",
    type=int,
    default=5,
    help="Number of frames to look for in the candidate instances to match with the current detections.",
)
@click.option(
    "--min_new_track_points",
    type=int,
    default=0,
    help="We won't spawn a new track for an instance with fewer than this many points.",
)
@click.option(
    "--candidates_method",
    type=str,
    default="fixed_window",
    help="Either of `fixed_window` or `local_queues`. In fixed window method, candidates from the last `window_size` frames. In local queues, last `window_size` instances for each track ID is considered for matching against the current detection.",
)
@click.option(
    "--min_match_points",
    type=int,
    default=0,
    help="Minimum non-NaN points for match candidates.",
)
@click.option(
    "--features",
    type=str,
    default="keypoints",
    help="Feature representation for the candidates to update current detections. One of [`keypoints`, `centroids`, `bboxes`, `image`].",
)
@click.option(
    "--scoring_method",
    type=str,
    default="oks",
    help="Method to compute association score between features from the current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`, `euclidean_dist`].",
)
@click.option(
    "--scoring_reduction",
    type=str,
    default="mean",
    help="Method to aggregate and reduce multiple scores if there are several detections associated with the same track. One of [`mean`, `max`, `robust_quantile`].",
)
@click.option(
    "--robust_best_instance",
    type=float,
    default=1.0,
    help="If the value is between 0 and 1 (excluded), use a robust quantile similarity score for the track. If the value is 1, use the max similarity (non-robust). For selecting a robust score, 0.95 is a good value.",
)
@click.option(
    "--track_matching_method",
    type=str,
    default="hungarian",
    help="Track matching algorithm. One of `hungarian`, `greedy`.",
)
@click.option(
    "--max_tracks",
    type=int,
    default=None,
    help="Maximum number of new tracks to be created to avoid redundant tracks. (only for local queues candidate)",
)
@click.option(
    "--use_flow",
    is_flag=True,
    default=False,
    help="If True, `FlowShiftTracker` is used, where the poses are matched using optical flow shifts.",
)
@click.option(
    "--of_img_scale",
    type=float,
    default=1.0,
    help="Factor to scale the images by when computing optical flow. Decrease this to increase performance at the cost of finer accuracy. Sometimes decreasing the image scale can improve performance with fast movements.",
)
@click.option(
    "--of_window_size",
    type=int,
    default=21,
    help="Optical flow window size to consider at each pyramid scale level.",
)
@click.option(
    "--of_max_levels",
    type=int,
    default=3,
    help="Number of pyramid scale levels to consider. This is different from the scale parameter, which determines the initial image scaling.",
)
@click.option(
    "--post_connect_single_breaks",
    is_flag=True,
    default=False,
    help="If True and `max_tracks` is not None with local queues candidate method, connects track breaks when exactly one track is lost and exactly one new track is spawned in the frame.",
)
def main(**kwargs):
    """CLI command that handles argument conversion and calls run_inference."""
    # Convert model_paths from tuple to list
    if "model_paths" in kwargs and kwargs["model_paths"]:
        kwargs["model_paths"] = list(kwargs["model_paths"])
    else:
        kwargs["model_paths"] = None

    # Convert frames string to list
    if "frames" in kwargs and kwargs["frames"]:
        kwargs["frames"] = frame_list(kwargs["frames"])
    else:
        kwargs["frames"] = None

    # Call the original function
    return run_inference(**kwargs)


if __name__ == "__main__":
    main()
