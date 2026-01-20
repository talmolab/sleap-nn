"""Unified CLI for SLEAP-NN using rich-click for styled output."""

import rich_click as click
from click import Command
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import sleap_io as sio
from sleap_nn.predict import run_inference, frame_list
from sleap_nn.evaluation import run_evaluation
from sleap_nn.export.cli import export as export_command
from sleap_nn.export.cli import predict as predict_command
from sleap_nn.train import run_training
from sleap_nn import __version__
import hydra
import sys

# Rich-click configuration for styled help
click.rich_click.TEXT_MARKUP = "markdown"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_EPILOGUE = (
    "Try 'sleap-nn [COMMAND] --help' for more information."
)


def is_config_path(arg: str) -> bool:
    """Check if an argument looks like a config file path.

    Returns True if the arg ends with .yaml or .yml.
    """
    return arg.endswith(".yaml") or arg.endswith(".yml")


def split_config_path(config_path: str) -> tuple:
    """Split a full config path into (config_dir, config_name).

    Args:
        config_path: Full path to a config file.

    Returns:
        Tuple of (config_dir, config_name) where config_dir is an absolute path.
    """
    path = Path(config_path).resolve()
    return path.parent.as_posix(), path.name


def print_version(ctx, param, value):
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"sleap-nn {__version__}")
    ctx.exit()


class TrainCommand(Command):
    """Custom command class that overrides help behavior for train command."""

    def format_help(self, ctx, formatter):
        """Override the help formatting to show custom training help."""
        show_training_help()


def parse_path_map(ctx, param, value):
    """Parse (old, new) path pairs into a dictionary for path mapping options."""
    if not value:
        return None
    result = {}
    for old_path, new_path in value:
        result[old_path] = Path(new_path).as_posix()
    return result


@click.group()
@click.option(
    "--version",
    "-v",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
def cli():
    """SLEAP-NN: Neural network backend for training and inference for animal pose estimation.

    Use subcommands to run different workflows:

    train    - Run training workflow
    track    - Run inference/ tracking workflow
    eval     - Run evaluation workflow
    system   - Display system information and GPU status
    """
    pass


def show_training_help():
    """Display training help information with rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    help_md = """
## Usage

```
sleap-nn train <config.yaml> [overrides]
sleap-nn train --config <path/to/config.yaml> [overrides]
```

## Common Overrides

| Override | Description |
|----------|-------------|
| `trainer_config.max_epochs=100` | Set maximum training epochs |
| `trainer_config.batch_size=32` | Set batch size |
| `trainer_config.save_ckpt=true` | Enable checkpoint saving |

## Examples

**Start a new training run:**
```bash
sleap-nn train path/to/config.yaml
sleap-nn train --config path/to/config.yaml
```

**With overrides:**
```bash
sleap-nn train config.yaml trainer_config.max_epochs=100
```

**Resume training:**
```bash
sleap-nn train config.yaml trainer_config.resume_ckpt_path=/path/to/ckpt
```

**Legacy usage (still supported):**
```bash
sleap-nn train --config-dir /path/to/dir --config-name myrun
```

## Tips

- Use `-m/--multirun` for sweeps; outputs go under `hydra.sweep.dir`
- For Hydra flags and completion, use `--hydra-help`
- Config documentation: https://nn.sleap.ai/config/
"""
    console.print(
        Panel(
            Markdown(help_md),
            title="[bold cyan]sleap-nn train[/bold cyan]",
            subtitle="Train SLEAP models from a config YAML file",
            border_style="cyan",
        )
    )


@cli.command(cls=TrainCommand)
@click.option(
    "--config",
    type=str,
    help="Path to configuration file (e.g., path/to/config.yaml)",
)
@click.option("--config-name", "-c", type=str, help="Configuration file name (legacy)")
@click.option(
    "--config-dir", "-d", type=str, default=".", help="Configuration directory (legacy)"
)
@click.option(
    "--video-paths",
    "-v",
    multiple=True,
    help="Video paths to replace existing paths in the labels file. "
    "Order must match the order of videos in the labels file. "
    "Can be specified multiple times. "
    "Example: --video-paths /path/to/vid1.mp4 --video-paths /path/to/vid2.mp4",
)
@click.option(
    "--video-path-map",
    nargs=2,
    multiple=True,
    callback=parse_path_map,
    metavar="OLD NEW",
    help="Map old video path to new path. Takes two arguments: old path and new path. "
    "Can be specified multiple times. "
    'Example: --video-path-map "/old/vid.mp`4" "/new/vid.mp4"',
)
@click.option(
    "--prefix-map",
    nargs=2,
    multiple=True,
    callback=parse_path_map,
    metavar="OLD NEW",
    help="Map old path prefix to new prefix. Takes two arguments: old prefix and new prefix. "
    "Updates ALL videos that share the same prefix. Useful when moving data between machines. "
    "Can be specified multiple times. "
    'Example: --prefix-map "/old/server/path" "/new/local/path"',
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train(
    config, config_name, config_dir, video_paths, video_path_map, prefix_map, overrides
):
    """Run training workflow with Hydra config overrides.

    Examples:
        sleap-nn train path/to/config.yaml
        sleap-nn train --config path/to/config.yaml trainer_config.max_epochs=100
        sleap-nn train -c myconfig -d /path/to/config_dir/ trainer_config.max_epochs=100
    """
    # Convert overrides to a mutable list
    overrides = list(overrides)

    # Check if the first positional arg is a config path (not a Hydra override)
    config_from_positional = None
    if overrides and is_config_path(overrides[0]):
        config_from_positional = overrides.pop(0)

    # Resolve config path with priority:
    # 1. Positional config path (e.g., sleap-nn train config.yaml)
    # 2. --config flag (e.g., sleap-nn train --config config.yaml)
    # 3. Legacy --config-dir/--config-name flags
    if config_from_positional:
        config_dir, config_name = split_config_path(config_from_positional)
    elif config:
        config_dir, config_name = split_config_path(config)
    elif config_name:
        config_dir = Path(config_dir).resolve().as_posix()
    else:
        # No config provided - show help
        show_training_help()
        return

    # Initialize Hydra manually (config_dir is already an absolute path)
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose config with overrides
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        # Validate config
        if not hasattr(cfg, "model_config") or not cfg.model_config:
            click.echo(
                "No model config found! Use `sleap-nn train --help` for more information."
            )
            raise click.Abort()

        logger.info("Input config:")
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        # Handle video path replacement options
        train_labels = None
        val_labels = None

        # Check that only one replacement option is used
        # video_paths is a tuple (empty if not used), others are None or dict
        has_video_paths = len(video_paths) > 0
        has_video_path_map = video_path_map is not None
        has_prefix_map = prefix_map is not None
        options_used = sum([has_video_paths, has_video_path_map, has_prefix_map])

        if options_used > 1:
            raise click.UsageError(
                "Cannot use multiple path replacement options. "
                "Choose one of: --video-paths, --video-path-map, or --prefix-map."
            )

        if options_used == 1:
            # Load train labels
            train_labels = [
                sio.load_slp(path) for path in cfg.data_config.train_labels_path
            ]

            # Load val labels if they exist
            if (
                cfg.data_config.val_labels_path is not None
                and len(cfg.data_config.val_labels_path) > 0
            ):
                val_labels = [
                    sio.load_slp(path) for path in cfg.data_config.val_labels_path
                ]

            # Build replacement arguments based on option used
            if has_video_paths:
                # List of paths (order must match videos in labels file)
                replace_kwargs = {
                    "new_filenames": [Path(p).as_posix() for p in video_paths]
                }
            elif has_video_path_map:
                # Dictionary mapping old filenames to new filenames
                replace_kwargs = {"filename_map": video_path_map}
            else:  # has_prefix_map
                # Dictionary mapping old prefixes to new prefixes
                replace_kwargs = {"prefix_map": prefix_map}

            # Apply replacement to train labels
            for labels in train_labels:
                labels.replace_filenames(**replace_kwargs)

            # Apply replacement to val labels if they exist
            if val_labels:
                for labels in val_labels:
                    labels.replace_filenames(**replace_kwargs)

        run_training(config=cfg, train_labels=train_labels, val_labels=val_labels)


@cli.command()
@click.option(
    "--data_path",
    "-i",
    type=str,
    required=True,
    help="Path to data to predict on. This can be a labels (.slp) file or any supported video format.",
)
@click.option(
    "--model_paths",
    "-m",
    multiple=True,
    help="Path to trained model directory (with training_config.json). Multiple models can be specified, each preceded by --model_paths.",
)
@click.option(
    "--output_path",
    "-o",
    type=str,
    default=None,
    help="The output filename to use for the predicted data. If not provided, defaults to '[data_path].slp'.",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="auto",
    help="Device on which torch.Tensor will be allocated. One of the ('cpu', 'cuda', 'mps', 'auto'). Default: 'auto' (based on available backend either cuda, mps or cpu is chosen). If `cuda` is available, you could also use `cuda:0` to specify the device.",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    default=4,
    help="Number of frames to predict at a time. Larger values result in faster inference speeds, but require more memory.",
)
@click.option(
    "--tracking",
    "-t",
    is_flag=True,
    default=False,
    help="If True, runs tracking on the predicted instances.",
)
@click.option(
    "-n",
    "--max_instances",
    type=int,
    default=None,
    help="Limit maximum number of instances in multi-instance models. Not available for ID models. Defaults to None.",
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
    "--exclude_user_labeled",
    is_flag=True,
    default=False,
    help="Skip frames that have user-labeled instances. Useful when predicting on entire video but skipping already-labeled frames.",
)
@click.option(
    "--only_predicted_frames",
    is_flag=True,
    default=False,
    help="Only run inference on frames that already have predictions. Requires .slp input file. Useful for re-predicting with a different model.",
)
@click.option(
    "--no_empty_frames",
    is_flag=True,
    default=False,
    help=("Clear empty frames that did not have predictions before saving to output."),
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
    "--integral_patch_size",
    type=int,
    default=5,
    help="Size of patches to crop around each rough peak as an integer scalar. Default: 5.",
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
    "--queue_maxsize",
    type=int,
    default=8,
    help="Maximum size of the frame buffer queue.",
)
@click.option(
    "--crop_size",
    type=int,
    default=None,
    help="Crop size. If not provided, the crop size from training_config.yaml is used. If `input_scale` is provided, then the cropped image will be resized according to `input_scale`.",
)
@click.option(
    "--peak_threshold",
    type=float,
    default=0.2,
    help="Minimum confidence map value to consider a peak as valid.",
)
@click.option(
    "--filter_overlapping",
    is_flag=True,
    default=False,
    help=(
        "Enable filtering of overlapping instances after inference using greedy NMS. "
        "Applied independently of tracking. (default: False)"
    ),
)
@click.option(
    "--filter_overlapping_method",
    type=click.Choice(["iou", "oks"]),
    default="iou",
    help=(
        "Similarity metric for filtering overlapping instances. "
        "'iou': bounding box intersection-over-union. "
        "'oks': Object Keypoint Similarity (pose-based). (default: iou)"
    ),
)
@click.option(
    "--filter_overlapping_threshold",
    type=float,
    default=0.8,
    help=(
        "Similarity threshold for filtering overlapping instances. "
        "Instances with similarity above this threshold are removed, "
        "keeping the higher-scoring instance. "
        "Typical values: 0.3 (aggressive) to 0.8 (permissive). (default: 0.8)"
    ),
)
@click.option(
    "--integral_refinement",
    type=str,
    default="integral",
    help="If `None`, returns the grid-aligned peaks with no refinement. If `'integral'`, peaks will be refined with integral regression. Default: 'integral'.",
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
@click.option(
    "--tracking_target_instance_count",
    type=int,
    default=None,
    help="Target number of instances to track per frame. (default: 0)",
)
@click.option(
    "--tracking_pre_cull_to_target",
    type=int,
    default=0,
    help=(
        "If non-zero and target_instance_count is also non-zero, then cull instances "
        "over target count per frame *before* tracking. (default: 0)"
    ),
)
@click.option(
    "--tracking_pre_cull_iou_threshold",
    type=float,
    default=0,
    help=(
        "If non-zero and pre_cull_to_target also set, then use IOU threshold to remove "
        "overlapping instances over count *before* tracking. (default: 0)"
    ),
)
@click.option(
    "--tracking_clean_instance_count",
    type=int,
    default=0,
    help="Target number of instances to clean *after* tracking. (default: 0)",
)
@click.option(
    "--tracking_clean_iou_threshold",
    type=float,
    default=0,
    help="IOU to use when culling instances *after* tracking. (default: 0)",
)
@click.option(
    "--gui",
    is_flag=True,
    default=False,
    help="Output JSON progress for GUI integration instead of Rich progress bar.",
)
def track(**kwargs):
    """Run Inference and Tracking workflow."""
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


@cli.command()
@click.option(
    "--ground_truth_path",
    "-g",
    type=str,
    required=True,
    help="Path to ground truth labels file (.slp)",
)
@click.option(
    "--predicted_path",
    "-p",
    type=str,
    required=True,
    help="Path to predicted labels file (.slp)",
)
@click.option("--save_metrics", "-s", type=str, help="Path to save metrics (.npz file)")
@click.option(
    "--oks_stddev",
    type=float,
    default=0.05,
    help="Standard deviation for OKS calculation",
)
@click.option("--oks_scale", type=float, help="Scale factor for OKS calculation")
@click.option(
    "--match_threshold", type=float, default=0.0, help="Threshold for instance matching"
)
@click.option(
    "--user_labels_only", is_flag=True, help="Only evaluate user-labeled frames"
)
def eval(**kwargs):
    """Run evaluation workflow."""
    run_evaluation(**kwargs)


@cli.command()
def system():
    """Display system information and GPU status.

    Shows Python version, platform, PyTorch version, CUDA availability,
    driver version with compatibility check, GPU details, and package versions.
    """
    from sleap_nn.system_info import print_system_info

    print_system_info()


cli.add_command(export_command)
cli.add_command(predict_command)


if __name__ == "__main__":
    cli()
