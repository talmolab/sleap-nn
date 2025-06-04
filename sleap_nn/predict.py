"""Entry point for running inference."""

import argparse
from typing import Optional, List
from loguru import logger
from sleap_nn.inference.predictors import run_inference


def _make_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        The `argparse.ArgumentParser` that defines the CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help=(
            "Path to data to predict on. This can be a labels (.slp) file or any "
            "supported video format."
        ),
    )
    parser.add_argument(
        "--model_paths",
        dest="model_paths",
        action="append",
        help=(
            "Path to trained model directory (with training_config.json). "
            "Multiple models can be specified, each preceded by --model."
        ),
    )
    parser.add_argument(
        "--backbone_ckpt_path",
        type=str,
        default=None,
        help=(
            "To run inference on any `.ckpt` other than `best.ckpt`"
            "from the `model_paths` dir, the path to the `.ckpt` file should be passed here."
        ),
    )
    parser.add_argument(
        "--head_ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to `.ckpt` file if a different set of head layer weights"
            "are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt"
            "from `backbone_ckpt_path` if provided.)"
        ),
    )
    parser.add_argument(
        "-n",
        "--max_instances",
        type=int,
        default=None,
        help=(
            "Limit maximum number of instances in multi-instance models. "
            "Not available for ID models. Defaults to None."
        ),
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=None,
        help=(
            "Maximum height the image should be padded to. If not provided, the"
            "values in the training config are used."
        ),
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=None,
        help=(
            "Maximum width the image should be padded to. If not provided, the"
            "values in the training config are used."
        ),
    )
    parser.add_argument(
        "--is_rgb",
        action="store_true",
        default=False,
        help=(
            "True if the image has 3 channels (RGB image). If input has only one"
            "channel when this is set to `True`, then the images from single-channel"
            "is replicated along the channel axis. If input has three channels and this"
            "is set to False, then we convert the image to grayscale (single-channel)"
            "image."
        ),
    )
    parser.add_argument(
        "--anchor_part",
        type=str,
        default=None,
        help=(
            "The node name to use as the anchor for the centroid. If not"
            "provided, the anchor part in the `training_config.yaml` is used."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help=(
            "Provider class to read the input sleap files."
            "Either 'LabelsReader' or 'VideoReader'"
        ),
    )
    parser.add_argument(
        "--only_labeled_frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on user labeled frames when running on labels dataset. "
            "This is useful for generating predictions to compare against ground truth."
        ),
    )
    parser.add_argument(
        "--only_suggested_frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on unlabeled suggested frames when running on labels "
            "dataset. This is useful for generating predictions for initialization "
            "during labeling."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Number of frames to predict at a time. Larger values result in faster "
            "inference speeds, but require more memory."
        ),
    )
    parser.add_argument(
        "--queue_maxsize",
        type=int,
        default=8,
        help=("Maximum size of the frame buffer queue."),
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help=(
            "List of frames to predict when running on a video. Can be specified as a "
            "comma separated list (e.g. 1,2,3) or a range separated by hyphen (e.g., "
            "1-3, for 1,2,3). If not provided, defaults to predicting on the entire "
            "video."
        ),
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help=(
            "Crop size. If not provided, the crop size from training_config.yaml is used."
        ),
    )
    parser.add_argument(
        "--peak_threshold",
        type=float,
        default=0.2,
        help="Minimum confidence map value to consider a peak as valid.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help=(
            "The output filename to use for the predicted data. If not provided, "
            "defaults to '[data_path].slp'."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device on which torch.Tensor will be allocated. One of the"
            "('cpu', 'cuda', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 'msnpu')."
        ),
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        default=False,
        help=("If True, runs tracking on the predicted instances."),
    )
    parser.add_argument(
        "--tracking_window_size",
        type=int,
        default=5,
        help=(
            "Number of frames to look for in the candidate instances to match"
            "with the current detections."
        ),
    )
    parser.add_argument(
        "--tracking_instance_score_threshold",
        type=float,
        default=0.0,
        help=("Instance score threshold for creating new tracks."),
    )
    parser.add_argument(
        "--candidates_method",
        type=str,
        default="fixed_window",
        help=(
            "Either of `fixed_window` or `local_queues`. In fixed window"
            "method, candidates from the last `window_size` frames. In local queues,"
            "last `window_size` instances for each track ID is considered for matching"
            "against the current detection."
        ),
    )
    parser.add_argument(
        "--features",
        type=str,
        default="keypoints",
        help=(
            "Feature representation for the candidates to update current detections."
            "One of [`keypoints`, `centroids`, `bboxes`, `image`]."
        ),
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="oks",
        help=(
            "Method to compute association score between features from the"
            "current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`,"
            "`euclidean_dist`]."
        ),
    )
    parser.add_argument(
        "--scoring_reduction",
        type=str,
        default="mean",
        help=(
            "Method to aggregate and reduce multiple scores if there are"
            "several detections associated with the same track. One of [`mean`, `max`,"
            "`weighted`]."
        ),
    )
    parser.add_argument(
        "--track_matching_method",
        type=str,
        default="hungarian",
        help=("Track matching algorithm. One of `hungarian`, `greedy."),
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=None,
        help=(
            "Meaximum number of new tracks to be created to avoid redundant tracks."
            "(only for local queues candidate)"
        ),
    )
    parser.add_argument(
        "--use_flow",
        action="store_true",
        default=False,
        help=(
            "If True, `FlowShiftTracker` is used, where the poses are matched using"
            "optical flow shifts."
        ),
    )
    parser.add_argument(
        "--of_img_scale",
        type=float,
        default=1.0,
        help=(
            "Factor to scale the images by when computing optical flow. Decrease"
            "this to increase performance at the cost of finer accuracy. Sometimes"
            "decreasing the image scale can improve performance with fast movements."
        ),
    )
    parser.add_argument(
        "--of_window_size",
        type=int,
        default=21,
        help=("Optical flow window size to consider at each pyramid scale" "level."),
    )
    parser.add_argument(
        "--of_max_levels",
        type=int,
        default=3,
        help=(
            "Number of pyramid scale levels to consider. This is different"
            "from the scale parameter, which determines the initial image scaling."
        ),
    )

    return parser


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


def main(args: Optional[list] = None):
    """Entrypoint for sleap-nn CLI for running inference.

    Args:
        args: A list of arguments to be passed.
    """
    parser = _make_cli_parser()

    # Parse inputs.
    args, _ = parser.parse_known_args(args)
    logger.info("Args:")
    logger.info(vars(args))

    if args.frames:
        args.frames = frame_list(args.frames)

    _ = run_inference(**vars(args))


if __name__ == "__main__":
    main()
