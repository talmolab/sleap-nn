"""Unified CLI for SLEAP-NN using rich-click for styled output."""

import __main__
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import rich_click as click
from click import Command
from loguru import logger

from sleap_nn import __version__


def _needs_module_respawn() -> bool:
    """Check if we need to re-spawn with `python -m` for proper DDP support.

    On Windows and macOS (Python 3.8+), multiprocessing uses 'spawn' instead of
    'fork'. The 'spawn' method starts child processes fresh and needs to re-import
    the training module. It uses `__main__.__spec__` to determine what module to
    import.

    When running via entry point scripts (e.g., `sleap-nn train` or `sleap train`),
    `__main__.__spec__` is None because entry points don't set it. This causes
    PyTorch Lightning's DDP to fail with: `ValueError: __main__.__spec__ is None`

    When running via `python -m sleap_nn.cli`, `__main__.__spec__` is properly set,
    allowing DDP child processes to re-import the module correctly.

    Returns:
        True if we need to re-spawn with `python -m`, False if already in module context.
    """
    return getattr(__main__, "__spec__", None) is None


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


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


class LazyGroup(click.RichGroup):
    """Click group that lazily loads export subcommands on first use."""

    _export_loaded = False

    def list_commands(self, ctx):
        """List all commands, loading export subcommands on first access."""
        self._ensure_export_loaded()
        return super().list_commands(ctx)

    def get_command(self, ctx, cmd_name):
        """Get a command by name, loading export subcommands on first access."""
        self._ensure_export_loaded()
        return super().get_command(ctx, cmd_name)

    def _ensure_export_loaded(self):
        if not self._export_loaded:
            LazyGroup._export_loaded = True
            _register_export_commands()


@click.group(cls=LazyGroup, context_settings=CONTEXT_SETTINGS)
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

    train    - Run training workflow (auto-handles multi-GPU)
    track    - Run inference/tracking workflow
    eval     - Run evaluation workflow
    system   - Display system information and GPU status
    """
    pass


def _get_num_devices_from_config(cfg) -> int:
    """Determine the number of devices from config.

    User preferences take precedence over auto-detection:
    - trainer_device_indices=[0] → 1 device (user choice)
    - trainer_devices=1 → 1 device (user choice)
    - trainer_devices="auto" or unset → auto-detect available GPUs

    Returns:
        Number of devices to use for training.
    """
    import torch
    from omegaconf import OmegaConf

    # User preference: explicit device indices (highest priority)
    device_indices = OmegaConf.select(
        cfg, "trainer_config.trainer_device_indices", default=None
    )
    if device_indices is not None and len(device_indices) > 0:
        return len(device_indices)

    # User preference: explicit device count
    devices = OmegaConf.select(cfg, "trainer_config.trainer_devices", default="auto")

    if isinstance(devices, int):
        return devices

    # Auto-detect only when user hasn't specified (devices is "auto" or None)
    if devices in ("auto", None, "None"):
        accelerator = OmegaConf.select(
            cfg, "trainer_config.trainer_accelerator", default="auto"
        )

        if accelerator == "cpu":
            return 1
        elif torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 1
        else:
            return 1

    return 1


def _finalize_config(cfg):
    """Finalize configuration by generating run_name if not provided.

    This runs ONCE before subprocess, ensuring all workers get the same run_name.
    """
    import sleap_io as sio
    from omegaconf import OmegaConf
    from sleap_nn.config.utils import get_model_type_from_cfg

    # Resolve ckpt_dir first
    ckpt_dir = OmegaConf.select(cfg, "trainer_config.ckpt_dir", default=None)
    if ckpt_dir is None or ckpt_dir == "" or ckpt_dir == "None":
        cfg.trainer_config.ckpt_dir = "."

    # Generate run_name if not provided
    run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
    if run_name is None or run_name == "" or run_name == "None":
        # Get model type from config
        model_type = get_model_type_from_cfg(cfg)

        # Count frames from labels
        train_paths = cfg.data_config.train_labels_path
        val_paths = OmegaConf.select(cfg, "data_config.val_labels_path", default=None)

        train_count = sum(len(sio.load_slp(p)) for p in train_paths)
        val_count = 0
        if val_paths:
            val_count = sum(len(sio.load_slp(p)) for p in val_paths)

        # Generate full run_name with timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_name = f"{timestamp}.{model_type}.n={train_count + val_count}"
        cfg.trainer_config.run_name = run_name

        logger.info(f"Generated run_name: {run_name}")

    return cfg


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


@cli.command(cls=TrainCommand, context_settings=CONTEXT_SETTINGS)
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
@click.option(
    "--video-config",
    type=str,
    hidden=True,
    help="Path to video replacement config YAML (internal use for multi-GPU).",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train(
    config,
    config_name,
    config_dir,
    video_paths,
    video_path_map,
    prefix_map,
    video_config,
    overrides,
):
    """Run training workflow with Hydra config overrides.

    Automatically detects multi-GPU setups and handles run_name synchronization
    by spawning training in a subprocess with a pre-generated config.

    Examples:
        sleap-nn train path/to/config.yaml
        sleap-nn train --config path/to/config.yaml trainer_config.max_epochs=100
        sleap-nn train config.yaml trainer_config.trainer_devices=4
    """
    import hydra
    import sleap_io as sio
    from omegaconf import OmegaConf
    from sleap_nn.train import run_training

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

    # Check video path options early
    # If --video-config is provided (from subprocess), load from file
    if video_config:
        video_cfg = OmegaConf.load(video_config)
        video_paths = tuple(video_cfg.video_paths) if video_cfg.video_paths else ()
        video_path_map = (
            dict(video_cfg.video_path_map) if video_cfg.video_path_map else None
        )
        prefix_map = dict(video_cfg.prefix_map) if video_cfg.prefix_map else None

    has_video_paths = len(video_paths) > 0
    has_video_path_map = video_path_map is not None
    has_prefix_map = prefix_map is not None
    options_used = sum([has_video_paths, has_video_path_map, has_prefix_map])

    if options_used > 1:
        raise click.UsageError(
            "Cannot use multiple path replacement options. "
            "Choose one of: --video-paths, --video-path-map, or --prefix-map."
        )

    # Load config to detect device count
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        # Validate config
        if not hasattr(cfg, "model_config") or not cfg.model_config:
            click.echo(
                "No model config found! Use `sleap-nn train --help` for more information."
            )
            raise click.Abort()

        num_devices = _get_num_devices_from_config(cfg)

        # Check if run_name is already set (for synchronization across DDP ranks)
        run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
        run_name_is_set = run_name is not None and run_name != "" and run_name != "None"

    # Multi-GPU path: spawn subprocess with finalized config
    # We need to re-spawn if EITHER:
    # 1. Not in module context (__main__.__spec__ is None) - required for DDP on
    #    Windows/macOS where multiprocessing uses 'spawn' and needs to know what
    #    module to re-import. See: https://github.com/talmolab/sleap/issues/2656
    # 2. run_name is not set - required for synchronization so all DDP ranks use
    #    the same run_name (otherwise each rank generates different timestamps)
    needs_respawn = _needs_module_respawn() or not run_name_is_set
    if num_devices > 1 and needs_respawn:
        logger.info(
            f"Detected {num_devices} devices, re-spawning with module context for DDP..."
        )

        # Load and finalize config (generate run_name, apply overrides)
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            cfg = _finalize_config(cfg)

        # Save finalized config to temp file
        temp_dir = tempfile.mkdtemp(prefix="sleap_nn_train_")
        temp_config_path = Path(temp_dir) / "training_config.yaml"
        OmegaConf.save(cfg, temp_config_path)
        logger.info(f"Saved finalized config to: {temp_config_path}")

        # Save video replacement config if needed (so subprocess doesn't need CLI args)
        temp_video_config_path = None
        if options_used == 1:
            video_replacement_config = {
                "video_paths": list(video_paths) if has_video_paths else None,
                "video_path_map": dict(video_path_map) if has_video_path_map else None,
                "prefix_map": dict(prefix_map) if has_prefix_map else None,
            }
            temp_video_config_path = Path(temp_dir) / "video_replacement.yaml"
            OmegaConf.save(
                OmegaConf.create(video_replacement_config), temp_video_config_path
            )
            logger.info(f"Saved video replacement config to: {temp_video_config_path}")

        # Build subprocess command (no video args - they're in the temp file)
        cmd = [sys.executable, "-m", "sleap_nn.cli", "train", str(temp_config_path)]
        if temp_video_config_path:
            cmd.extend(["--video-config", str(temp_video_config_path)])

        logger.info(f"Launching subprocess: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(cmd)
            result = process.wait()
            if result != 0:
                logger.error(f"Training failed with exit code {result}")
                sys.exit(result)
        except KeyboardInterrupt:
            logger.info("Training interrupted, terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            sys.exit(1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up temporary files")

        return

    # Single GPU (or subprocess worker): run directly
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        logger.info("Input config:")
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        # Handle video path replacement options
        train_labels = None
        val_labels = None

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
                replace_kwargs = {
                    "new_filenames": [Path(p).as_posix() for p in video_paths]
                }
            elif has_video_path_map:
                replace_kwargs = {"filename_map": video_path_map}
            else:  # has_prefix_map
                replace_kwargs = {"prefix_map": prefix_map}

            # Apply replacement to train labels
            for labels in train_labels:
                labels.replace_filenames(**replace_kwargs)

            # Apply replacement to val labels if they exist
            if val_labels:
                for labels in val_labels:
                    labels.replace_filenames(**replace_kwargs)

        run_training(config=cfg, train_labels=train_labels, val_labels=val_labels)


@cli.command(context_settings=CONTEXT_SETTINGS)
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
    "--ensure_rgb/--no-ensure_rgb",
    default=None,
    help="If True, input images will have 3 channels (RGB). Single-channel images are replicated along the channel axis. If False, RGB conversion is disabled. If not provided, the values from the training config are used. Default: None.",
)
@click.option(
    "--ensure_grayscale/--no-ensure_grayscale",
    default=None,
    help="If True, input images will be converted to single-channel grayscale. If False, grayscale conversion is disabled. If not provided, the values from the training config are used. Default: None.",
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
    default=32,
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
    "--filter_min_visible_nodes",
    type=int,
    default=0,
    help=(
        "Minimum number of visible (non-NaN) keypoints required. "
        "Instances with fewer visible nodes are removed. (default: 0, no filtering)"
    ),
)
@click.option(
    "--filter_min_visible_node_fraction",
    type=float,
    default=0.0,
    help=(
        "Minimum fraction of skeleton nodes that must be visible. "
        "Value should be in [0, 1]. For example, 0.5 requires at least half "
        "of the skeleton's nodes to be detected. (default: 0.0, no filtering)"
    ),
)
@click.option(
    "--filter_min_mean_node_score",
    type=float,
    default=0.0,
    help=(
        "Minimum mean confidence score across visible nodes. "
        "Instances with lower mean node scores are removed. (default: 0.0, no filtering)"
    ),
)
@click.option(
    "--filter_min_instance_score",
    type=float,
    default=0.0,
    help=(
        "Minimum overall instance confidence score. "
        "Instances with lower scores are removed. (default: 0.0, no filtering)"
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
    """Run Inference and Tracking workflow.

    .. deprecated::
       Use ``sleap-nn infer`` instead. The ``track`` alias will be
       removed in a future release. This is currently equivalent to
       ``sleap-nn infer`` (PR 10 of #508 / #518).
    """
    import warnings

    warnings.warn(
        "`sleap-nn track` is deprecated; use `sleap-nn infer` instead. "
        "Aliases will be removed in v0.3.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _run_inference_impl(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# PR 10 of #508 — `sleap-nn infer` unified inference command (#518)
# ──────────────────────────────────────────────────────────────────────────
#
# `infer`, `predict`, and `track` share an option list and dispatch to one
# implementation. The option list is built programmatically below
# (``_INFERENCE_OPTIONS``) so the three commands stay in lockstep — adding
# a flag in one place updates all three.
#
# The new flags introduced by PR 10:
#   --paf-workers / --cpu-workers (alias)  → wires to Predictor(paf_workers=)
#   --stream-to-file                        → triggers Predictor.predict_to_file
#   --write-interval                        → flush cadence for stream-to-file
#   --peak-conf-threshold (alias)           → alternate name for --peak_threshold
#
# The first two are accepted but warn / error today: the underlying new
# Predictor wiring lands in PR 14 (#519, blocked-by this PR). When that
# PR lands, this implementation flips to call
# ``sleap_nn.inference.predictor.Predictor`` directly without changing
# the user-facing CLI surface.


def _run_inference_impl(**kwargs):
    """Shared implementation for ``infer`` / ``predict`` / ``track``.

    Coerces tuple-shaped multi-options into lists, parses the
    ``--frames`` string into a list of int frame indices, validates the
    new PR 10 flags, and routes to the right backend:

    * ``--stream-to-file`` set → builds a new :class:`Predictor` via
      :func:`sleap_nn.inference.factory.from_model_paths` and writes
      incrementally with :meth:`Predictor.predict_to_file` (PR 12).
    * Otherwise → delegates to the legacy ``run_inference`` flow
      (which still owns tracking, frame filtering, GUI progress, etc.).
    """
    from sleap_nn.predict import frame_list, run_inference

    paf_workers = kwargs.pop("paf_workers", 0) or 0
    cpu_workers = kwargs.pop("cpu_workers", None)
    stream_to_file = kwargs.pop("stream_to_file", None)
    write_interval = kwargs.pop("write_interval", None)
    if cpu_workers is not None:
        import warnings

        warnings.warn(
            "--cpu-workers is deprecated; use --paf-workers.",
            DeprecationWarning,
            stacklevel=2,
        )
        if paf_workers == 0:
            paf_workers = cpu_workers
    if write_interval is not None and stream_to_file is None:
        raise click.UsageError(
            "--write-interval is only meaningful together with --stream-to-file."
        )

    if "model_paths" in kwargs and kwargs["model_paths"]:
        kwargs["model_paths"] = list(kwargs["model_paths"])
    else:
        kwargs["model_paths"] = None

    if "frames" in kwargs and kwargs["frames"]:
        kwargs["frames"] = frame_list(kwargs["frames"])
    else:
        kwargs["frames"] = None

    # ── Stream-to-file path: new Predictor + IncrementalLabelsWriter ───
    if stream_to_file is not None:
        return _run_stream_to_file(
            kwargs,
            stream_to_file=stream_to_file,
            write_interval=write_interval or 500,
            paf_workers=paf_workers,
        )

    # ── In-memory new-flow path (PR 13) ────────────────────────────────
    # When the requested flags don't need anything the legacy ``run_inference``
    # alone can do (tracking, suggested-frame filtering, GUI progress reporter,
    # backbone/head ckpt overrides), route through the new
    # :func:`Predictor.from_model_paths(...).predict(...)` flow. The legacy
    # path stays available as a fallback for everything else.
    if _can_use_new_in_memory_flow(kwargs):
        return _run_in_memory_new_flow(kwargs, paf_workers=paf_workers)

    if paf_workers > 0:
        logger.warning(
            "--paf-workers > 0 currently has no effect on the legacy "
            "predict path; pass --stream-to-file or use a simpler config "
            "(no tracking, no advanced frame filtering) to use the worker pool."
        )

    return run_inference(**kwargs)


def _can_use_new_in_memory_flow(kwargs: dict) -> bool:
    """Return True iff the new factory + Predictor.predict can serve this call.

    The new flow handles: video / labels source, one or two ``.ckpt``
    model dirs, optional ``frames`` list, the standard preprocess
    overrides, and ``--tracking`` (post-inference tracker via
    :class:`TrackerConfig`). It does NOT yet handle suggested-frame /
    predicted-frame filtering, the GUI progress reporter, layer-level
    checkpoint overrides, or the pre-tracking ``--filter_*`` knobs.
    """
    for flag in (
        "only_suggested_frames",
        "exclude_user_labeled",
        "only_predicted_frames",
        "no_empty_frames",
        "gui",
    ):
        if kwargs.get(flag):
            return False
    if kwargs.get("backbone_ckpt_path") or kwargs.get("head_ckpt_path"):
        # Layer-level checkpoint overrides aren't piped through the new
        # factory yet; the legacy loader handles them today.
        return False
    if kwargs.get("tracking") and _has_pre_tracking_filter(kwargs):
        # Pre-tracking ``filter_*`` knobs run as a separate stage in
        # legacy ``run_inference``; they aren't routed through
        # ``FilterPipeline`` yet. Stay on legacy when both are set.
        return False
    if not kwargs.get("model_paths"):
        return False
    return True


def _has_pre_tracking_filter(kwargs: dict) -> bool:
    """``True`` if the user set any post-inference filter knob.

    The legacy ``run_inference`` runs these as a separate stage before
    tracking; the new flow's :class:`FilterPipeline` doesn't yet thread
    CLI flags through, so we keep tracking + filter combinations on
    legacy until that wiring lands.
    """
    if kwargs.get("filter_overlapping"):
        return True
    if kwargs.get("filter_min_visible_nodes") or kwargs.get(
        "filter_min_visible_node_fraction"
    ):
        return True
    if kwargs.get("filter_min_mean_node_score") or kwargs.get(
        "filter_min_instance_score"
    ):
        return True
    return False


def _build_tracker_config(kwargs: dict) -> "object":
    """Build a :class:`TrackerConfig` from the CLI ``--tracking_*`` flags."""
    from sleap_nn.inference.tracking import TrackerConfig

    return TrackerConfig(
        window_size=kwargs.get("tracking_window_size", 5),
        min_new_track_points=kwargs.get("min_new_track_points", 0),
        candidates_method=kwargs.get("candidates_method", "fixed_window"),
        min_match_points=kwargs.get("min_match_points", 0),
        features=kwargs.get("features", "keypoints"),
        scoring_method=kwargs.get("scoring_method", "oks"),
        scoring_reduction=kwargs.get("scoring_reduction", "mean"),
        robust_best_instance=kwargs.get("robust_best_instance", 1.0),
        track_matching_method=kwargs.get("track_matching_method", "hungarian"),
        max_tracks=kwargs.get("max_tracks"),
        use_flow=kwargs.get("use_flow", False),
        of_img_scale=kwargs.get("of_img_scale", 1.0),
        of_window_size=kwargs.get("of_window_size", 21),
        of_max_levels=kwargs.get("of_max_levels", 3),
        tracking_target_instance_count=kwargs.get("tracking_target_instance_count"),
        tracking_pre_cull_to_target=kwargs.get("tracking_pre_cull_to_target", 0),
        tracking_pre_cull_iou_threshold=kwargs.get(
            "tracking_pre_cull_iou_threshold", 0.0
        ),
        tracking_clean_instance_count=kwargs.get("tracking_clean_instance_count", 0),
        tracking_clean_iou_threshold=kwargs.get("tracking_clean_iou_threshold", 0.0),
        post_connect_single_breaks=kwargs.get("post_connect_single_breaks", False),
    )


def _run_in_memory_new_flow(kwargs: dict, paf_workers: int) -> "object":
    """Run the new ``Predictor`` flow synchronously and save the resulting Labels.

    This is the in-memory counterpart to :func:`_run_stream_to_file`:
    same factory, same providers, but ``predict()`` instead of
    ``predict_to_file()`` so the result lives in memory until the final
    ``Labels.save()`` call.
    """
    from pathlib import Path

    import sleap_io as sio

    from sleap_nn.inference.factory import from_model_paths
    from sleap_nn.inference.providers import LabelsProvider, VideoProvider

    factory_kwargs = {
        "device": kwargs.get("device", "auto"),
        "peak_threshold": kwargs.get("peak_threshold", 0.2),
        "integral_refinement": kwargs.get("integral_refinement", "integral"),
        "integral_patch_size": kwargs.get("integral_patch_size", 5),
        "batch_size": kwargs.get("batch_size", 4),
        "max_instances": kwargs.get("max_instances"),
        "anchor_part": kwargs.get("anchor_part"),
        "paf_workers": paf_workers,
    }
    if kwargs.get("tracking"):
        factory_kwargs["tracker_config"] = _build_tracker_config(kwargs)
    predictor = from_model_paths(kwargs["model_paths"], **factory_kwargs)

    src = Path(kwargs["data_path"])
    only_labeled = bool(kwargs.get("only_labeled_frames"))
    if src.suffix == ".slp":
        provider = LabelsProvider(
            labels=str(src),
            batch_size=kwargs.get("batch_size", 4),
            only_labeled_frames=only_labeled,
        )
        loaded = sio.load_slp(str(src))
        skeleton = loaded.skeletons[0]
        videos = list(loaded.videos)
    else:
        provider = VideoProvider(
            video=str(src),
            batch_size=kwargs.get("batch_size", 4),
            frames=kwargs.get("frames"),
            dataset=kwargs.get("video_dataset"),
            input_format=kwargs.get("video_input_format"),
        )
        skeleton = _skeleton_from_predictor(predictor, kwargs["model_paths"][0])
        # ``Labels.save()`` traverses ``video.backend``; pass the loaded
        # ``sio.Video`` so the saved .slp has a real backend reference.
        videos = [sio.load_video(str(src))]

    labels = predictor.predict(
        provider, make_labels=True, skeleton=skeleton, videos=videos
    )

    output_path = kwargs.get("output_path") or f"{src}.slp"
    labels.save(output_path)
    return labels


def _run_stream_to_file(
    kwargs: dict,
    stream_to_file: str,
    write_interval: int,
    paf_workers: int,
) -> str:
    """Build a new ``Predictor`` and stream predictions to ``stream_to_file``.

    Handles the ``--stream-to-file`` CLI path (PR 12). Restricted to the
    cases the new factory + ``Predictor.predict_to_file`` support today:
    inference on a video / labels file, no tracking, default frame
    selection. Anything richer raises ``UsageError`` and points at the
    legacy ``--no-stream-to-file`` path.
    """
    if kwargs.get("tracking"):
        raise click.UsageError(
            "--stream-to-file does not yet support --tracking; tracking "
            "lands in a follow-up PR. Drop --stream-to-file to use the "
            "legacy in-memory path with tracking."
        )
    if not kwargs.get("model_paths"):
        raise click.UsageError("--model_paths is required for --stream-to-file.")
    data_path = kwargs.get("data_path")
    if not data_path:
        raise click.UsageError("--data_path is required for --stream-to-file.")

    from pathlib import Path

    import sleap_io as sio

    from sleap_nn.inference.factory import from_model_paths
    from sleap_nn.inference.providers import LabelsProvider, VideoProvider

    factory_kwargs = {
        "device": kwargs.get("device", "auto"),
        "peak_threshold": kwargs.get("peak_threshold", 0.2),
        "integral_refinement": kwargs.get("integral_refinement", "integral"),
        "integral_patch_size": kwargs.get("integral_patch_size", 5),
        "batch_size": kwargs.get("batch_size", 4),
        "max_instances": kwargs.get("max_instances"),
        "return_confmaps": False,
        "anchor_part": kwargs.get("anchor_part"),
        "paf_workers": paf_workers,
    }
    if kwargs.get("backbone_ckpt_path"):
        factory_kwargs["backbone_ckpt_path"] = kwargs["backbone_ckpt_path"]
    if kwargs.get("head_ckpt_path"):
        factory_kwargs["head_ckpt_path"] = kwargs["head_ckpt_path"]

    predictor = from_model_paths(kwargs["model_paths"], **factory_kwargs)

    src = Path(data_path)
    if src.suffix == ".slp":
        provider = LabelsProvider(
            labels=str(src), batch_size=kwargs.get("batch_size", 4)
        )
        labels = sio.load_slp(str(src))
        skeleton = labels.skeletons[0]
    else:
        provider = VideoProvider(
            video=str(src),
            batch_size=kwargs.get("batch_size", 4),
            frames=kwargs.get("frames"),
            dataset=kwargs.get("video_dataset"),
            input_format=kwargs.get("video_input_format"),
        )
        # Skeleton comes from the model's training_config — pull via the layer.
        skeleton = _skeleton_from_predictor(predictor, kwargs["model_paths"][0])

    return predictor.predict_to_file(
        provider,
        path=str(stream_to_file),
        skeleton=skeleton,
        write_interval=write_interval,
    )


def _skeleton_from_predictor(predictor, model_path: str):
    """Extract a ``sleap_io.Skeleton`` from the model's ``training_config``."""
    from omegaconf import OmegaConf

    from sleap_nn.inference.utils import get_skeleton_from_config

    cfg_path = Path(model_path) / "training_config.yaml"
    cfg = OmegaConf.load(cfg_path.as_posix())
    skeletons = get_skeleton_from_config(cfg.data_config.skeletons)
    return skeletons[0]


def _common_inference_options(f):
    """Apply the shared inference flag list to a click command function.

    Defined as a function-level decorator (not a ``@click.option`` chain)
    so the same option set can be reused across ``infer`` / ``predict``
    / ``track`` without copy-pasting ~70 decorator lines per command.
    """
    decorators = [
        click.option(
            "--data_path",
            "-i",
            type=str,
            required=True,
            help="Path to data to predict on. Labels (.slp) file or any supported video format.",
        ),
        click.option(
            "--model_paths",
            "-m",
            multiple=True,
            help="Path to trained model directory. Multiple models may be passed (each preceded by --model_paths).",
        ),
        click.option(
            "--output_path",
            "-o",
            type=str,
            default=None,
            help="Output filename. Defaults to '[data_path].slp'.",
        ),
        click.option(
            "--device",
            "-d",
            type=str,
            default="auto",
            help="Device on which to allocate tensors. One of (cpu, cuda, mps, auto).",
        ),
        click.option(
            "--batch_size",
            "-b",
            type=int,
            default=4,
            help="Number of frames to predict at a time.",
        ),
        click.option(
            "--tracking",
            "-t",
            is_flag=True,
            default=False,
            help="Run tracking on predicted instances.",
        ),
        click.option(
            "-n",
            "--max_instances",
            type=int,
            default=None,
            help="Cap on instances per frame for multi-instance models.",
        ),
        click.option(
            "--backbone_ckpt_path",
            type=str,
            default=None,
            help="Override path to the backbone .ckpt.",
        ),
        click.option(
            "--head_ckpt_path",
            type=str,
            default=None,
            help="Override path to the head .ckpt.",
        ),
        click.option("--max_height", type=int, default=None),
        click.option("--max_width", type=int, default=None),
        click.option("--input_scale", type=float, default=None),
        click.option(
            "--ensure_rgb/--no-ensure_rgb",
            default=None,
            help="Force RGB conversion of input frames.",
        ),
        click.option(
            "--ensure_grayscale/--no-ensure_grayscale",
            default=None,
            help="Force grayscale conversion of input frames.",
        ),
        click.option("--anchor_part", type=str, default=None),
        click.option("--only_labeled_frames", is_flag=True, default=False),
        click.option("--only_suggested_frames", is_flag=True, default=False),
        click.option("--exclude_user_labeled", is_flag=True, default=False),
        click.option("--only_predicted_frames", is_flag=True, default=False),
        click.option("--no_empty_frames", is_flag=True, default=False),
        click.option("--video_index", type=int, default=None),
        click.option("--video_dataset", type=str, default=None),
        click.option(
            "--video_input_format",
            type=str,
            default="channels_last",
        ),
        click.option(
            "--frames",
            type=str,
            default="",
            help="Frame list as comma-separated or hyphen range (e.g., 1,2,3 or 1-3).",
        ),
        click.option("--integral_patch_size", type=int, default=5),
        click.option("--max_edge_length_ratio", type=float, default=0.25),
        click.option("--dist_penalty_weight", type=float, default=1.0),
        click.option("--n_points", type=int, default=10),
        click.option("--min_instance_peaks", type=float, default=0),
        click.option("--min_line_scores", type=float, default=0.25),
        click.option("--queue_maxsize", type=int, default=32),
        click.option("--crop_size", type=int, default=None),
        click.option(
            "--peak_threshold",
            "--peak-conf-threshold",
            "peak_threshold",
            type=float,
            default=0.2,
            help="Min confmap value for a valid peak. --peak-conf-threshold is an alias.",
        ),
        click.option("--filter_overlapping", is_flag=True, default=False),
        click.option(
            "--filter_overlapping_method",
            type=click.Choice(["iou", "oks"]),
            default="iou",
        ),
        click.option("--filter_overlapping_threshold", type=float, default=0.8),
        click.option("--filter_min_visible_nodes", type=int, default=0),
        click.option("--filter_min_visible_node_fraction", type=float, default=0.0),
        click.option("--filter_min_mean_node_score", type=float, default=0.0),
        click.option("--filter_min_instance_score", type=float, default=0.0),
        click.option("--integral_refinement", type=str, default="integral"),
        click.option("--tracking_window_size", type=int, default=5),
        click.option("--min_new_track_points", type=int, default=0),
        click.option("--candidates_method", type=str, default="fixed_window"),
        click.option("--min_match_points", type=int, default=0),
        click.option("--features", type=str, default="keypoints"),
        click.option("--scoring_method", type=str, default="oks"),
        click.option("--scoring_reduction", type=str, default="mean"),
        click.option("--robust_best_instance", type=float, default=1.0),
        click.option("--track_matching_method", type=str, default="hungarian"),
        click.option("--max_tracks", type=int, default=None),
        click.option("--use_flow", is_flag=True, default=False),
        click.option("--of_img_scale", type=float, default=1.0),
        click.option("--of_window_size", type=int, default=21),
        click.option("--of_max_levels", type=int, default=3),
        click.option("--post_connect_single_breaks", is_flag=True, default=False),
        click.option("--tracking_target_instance_count", type=int, default=None),
        click.option("--tracking_pre_cull_to_target", type=int, default=0),
        click.option("--tracking_pre_cull_iou_threshold", type=float, default=0),
        click.option("--tracking_clean_instance_count", type=int, default=0),
        click.option("--tracking_clean_iou_threshold", type=float, default=0),
        click.option("--gui", is_flag=True, default=False),
        # New PR 10 flags ─────────────────────────────────────────────
        click.option(
            "--paf-workers",
            "paf_workers",
            type=int,
            default=0,
            help=(
                "Number of CPU worker processes for the bottom-up PAF "
                "grouping stage. 0 (default) keeps grouping in-process. "
                "Replaces --cpu-workers (kept as an alias for one cycle)."
            ),
        ),
        click.option(
            "--cpu-workers",
            "cpu_workers",
            type=int,
            default=None,
            help="[DEPRECATED] Use --paf-workers.",
        ),
        click.option(
            "--stream-to-file",
            "stream_to_file",
            type=str,
            default=None,
            help=(
                "Stream predictions incrementally to this .slp path "
                "(memory stays O(write-interval)). Triggers the new "
                "Predictor.predict_to_file flow."
            ),
        ),
        click.option(
            "--write-interval",
            "write_interval",
            type=int,
            default=None,
            help=(
                "Number of LabeledFrames to buffer before flushing to "
                "disk when --stream-to-file is set. Default: 500."
            ),
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


@cli.command(context_settings=CONTEXT_SETTINGS)
@_common_inference_options
def infer(**kwargs):
    """Run inference on videos or labels files.

    Single unified inference entry point. Replaces ``track``, ``predict``,
    and ``export predict`` (which remain as aliases with deprecation
    warnings until v0.3).
    """
    return _run_inference_impl(**kwargs)


# NOTE: The `sleap-nn predict` deprecation alias is deferred — a follow-up
# PR will (a) convert `export` from a single command to a click group,
# (b) re-home the existing top-level `predict` (which today runs inference
# on an exported ONNX/TRT model) under `sleap-nn export predict`, and
# (c) reclaim the top-level `predict` name as an alias for `infer`. PR 10
# ships `track` deprecation and the canonical `infer` only, so users can
# migrate via either of those paths first.


@cli.command(context_settings=CONTEXT_SETTINGS)
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
    default=0.025,
    help="Standard deviation for OKS calculation",
)
@click.option("--oks_scale", type=float, help="Scale factor for OKS calculation")
@click.option(
    "--match_threshold", type=float, default=0.0, help="Threshold for instance matching"
)
@click.option(
    "--user_labels_only/--no-user_labels_only",
    default=True,
    help="Only evaluate user-labeled frames (default: True)",
)
def eval(**kwargs):
    """Run evaluation workflow."""
    from sleap_nn.evaluation import run_evaluation

    run_evaluation(**kwargs)


@cli.command(context_settings=CONTEXT_SETTINGS)
def system():
    """Display system information and GPU status.

    Shows Python version, platform, PyTorch version, CUDA availability,
    driver version with compatibility check, GPU details, and package versions.
    """
    from sleap_nn.system_info import print_system_info

    print_system_info()


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("path", type=str)
def info(path):
    """Display model configuration and evaluation metrics.

    PATH can be a trained model directory or a training config YAML file.

    If a model directory is given, shows config summary, training results,
    evaluation metrics (if available), and files in the directory.

    If a config YAML is given, shows only the config summary.

    Examples:
        sleap-nn info path/to/model_dir

        sleap-nn info path/to/training_config.yaml
    """
    from sleap_nn.model_info import print_model_info

    print_model_info(path)


def _register_export_commands():
    """Lazily import and register export subcommands."""
    from sleap_nn.export.cli import export as export_command
    from sleap_nn.export.cli import predict as predict_command

    cli.add_command(export_command)
    cli.add_command(predict_command)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("slp_path", type=str, required=False, default=None)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output path for the generated config file(s).",
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Auto-generate config without interactive TUI.",
)
@click.option(
    "--pipeline",
    type=click.Choice(
        [
            "single_instance",
            "bottomup",
            "topdown",
            "multi_class_bottomup",
            "multi_class_topdown",
        ]
    ),
    default=None,
    help="Override model pipeline type. 'topdown' generates paired centroid + centered_instance configs.",
)
@click.option(
    "--show-yaml",
    is_flag=True,
    default=False,
    help="Print generated YAML to stdout.",
)
def config(
    slp_path,
    output,
    auto,
    pipeline,
    show_yaml,
):
    """Generate training configuration for a SLEAP file.

    **[Experimental]** This feature is experimental and may change in future releases.

    Launch an interactive TUI to configure training, or use --auto to
    generate a config with smart defaults based on your data.

    Examples:
        # Interactive TUI
        sleap-nn config labels.slp

        # Auto-generate config
        sleap-nn config labels.slp --auto -o config.yaml

        # Auto-generate with overrides
        sleap-nn config labels.slp --auto --pipeline bottomup
    """
    from sleap_nn.config_generator.generator import ConfigGenerator

    # Auto mode (non-interactive)
    if auto:
        if not slp_path:
            click.echo("Error: SLP_PATH is required for --auto mode", err=True)
            raise SystemExit(1)

        gen = ConfigGenerator.from_slp(slp_path)
        gen.auto()

        # Apply overrides
        if pipeline:
            # 'topdown' is a CLI alias for the centroid stage, which triggers
            # paired centroid + centered_instance config generation.
            gen.pipeline("centroid" if pipeline == "topdown" else pipeline)

        if show_yaml:
            click.echo(gen.to_yaml())
        elif output:
            gen.save(output)
            if gen.is_topdown:
                path_obj = Path(output)
                stem = path_obj.stem
                suffix = path_obj.suffix or ".yaml"
                parent = path_obj.parent
                click.echo(
                    f"Saved centroid config to: {parent / f'{stem}_centroid{suffix}'}"
                )
                click.echo(
                    f"Saved instance config to: {parent / f'{stem}_centered_instance{suffix}'}"
                )
            else:
                click.echo(f"Saved config to: {output}")
        else:
            # Default output path
            slp_stem = Path(slp_path).stem
            output_path = Path(slp_path).parent / f"{slp_stem}_config.yaml"
            gen.save(str(output_path))
            if gen.is_topdown:
                click.echo(
                    f"Saved centroid config to: {output_path.parent / f'{slp_stem}_config_centroid.yaml'}"
                )
                click.echo(
                    f"Saved instance config to: {output_path.parent / f'{slp_stem}_config_centered_instance.yaml'}"
                )
            else:
                click.echo(f"Saved config to: {output_path}")
        return

    # Interactive TUI mode
    try:
        from sleap_nn.config_generator.tui.app import launch_tui

        launch_tui(slp_path)
    except ImportError as e:
        click.echo(f"Error: TUI dependencies not available: {e}", err=True)
        click.echo("Install with: pip install textual", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
