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


def _parse_int_list(ctx, param, value):
    """Click callback parsing a comma-separated int list (e.g. ``0,1,2``).

    Returns ``None`` for an unset/empty value so it maps cleanly onto the
    ``Optional[List[int]]`` tracker kwargs (e.g. ``kf_node_indices``).
    """
    if value is None or value == "":
        return None
    try:
        return [int(x) for x in str(value).split(",") if x.strip() != ""]
    except ValueError:
        raise click.BadParameter(
            "must be a comma-separated list of integers, e.g. '0,1,2'"
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
    predict  - Run inference workflow (new pipeline)
    track    - Run inference/tracking workflow (legacy pipeline)
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
    help="Path to a trained model directory, or to its best.ckpt or training_config.yaml/.json file (all resolve to the model directory). Multiple models can be specified, each preceded by --model_paths.",
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
    "--use_kalman",
    is_flag=True,
    default=False,
    help="If True, `KalmanShiftTracker` is used: poses are predicted with a per-track constant-velocity Kalman filter. Requires --tracking_target_instance_count (or --max_tracks/--max_instances) and is mutually exclusive with --use_flow.",
)
@click.option(
    "--kf_track_features",
    type=click.Choice(["centroid", "keypoints"]),
    default="centroid",
    help="What the Kalman motion model tracks: 'centroid' (default; rigid, stable) or 'keypoints' (per-node poses; noisier — pair with --oks_stddev or --features bboxes --scoring_method iou). (only if --use_kalman)",
)
@click.option(
    "--oks_stddev",
    type=float,
    default=None,
    help="OKS keypoint-spread normalization constant for `oks` scoring. Larger is more tolerant of localization error (useful with --kf_track_features keypoints). Default: 0.025.",
)
@click.option(
    "--kf_init_frame_count",
    type=int,
    default=10,
    help="Number of warm-up frames tracked with the base path before the Kalman filters are fit via EM. (only if --use_kalman)",
)
@click.option(
    "--kf_node_indices",
    type=str,
    default=None,
    callback=_parse_int_list,
    help="Comma-separated skeleton node indices to track with the motion model (e.g. '0,1,2'). Empty/unset uses all nodes. (only if --use_kalman)",
)
@click.option(
    "--kf_reset_gap_size",
    type=int,
    default=5,
    help="Number of consecutive missed frames after which a stale track's Kalman filter is reset. (only if --use_kalman)",
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
    """Run Inference and Tracking workflow (legacy pipeline).

    This command uses the legacy ``run_inference`` pipeline. For the new
    inference pipeline, use ``sleap-nn predict``.
    """
    from sleap_nn.predict import frame_list, run_inference

    if "model_paths" in kwargs and kwargs["model_paths"]:
        kwargs["model_paths"] = list(kwargs["model_paths"])
    else:
        kwargs["model_paths"] = None

    if "frames" in kwargs and kwargs["frames"]:
        kwargs["frames"] = frame_list(kwargs["frames"])
    else:
        kwargs["frames"] = None

    # Pop new-pipeline-only flags that track doesn't use. NOTE: `gui` is NOT
    # popped — legacy run_inference accepts+honors it (predict.py sets
    # predictor.gui, switching to JSON-per-line progress); popping it silently
    # dropped `track --gui` GUI integration (#583).
    kwargs.pop("paf_workers", None)
    kwargs.pop("cpu_workers", None)
    kwargs.pop("stream_to_file", None)
    kwargs.pop("write_interval", None)
    # `--runtime` selects the exported-model runtime for `sleap-nn predict`;
    # `track` only runs trained checkpoints, so it is inert here.
    kwargs.pop("runtime", None)
    # New-flow-only flags, inert here. `track` cannot do centroid-only
    # inference; a lone centroid model is rejected downstream by run_inference
    # with a message pointing to `sleap-nn predict`.
    kwargs.pop("centroid_only", None)
    kwargs.pop("centroid_peak_threshold", None)
    kwargs.pop("centroid_output", None)
    kwargs.pop("filter_min_centroid_distance", None)

    return run_inference(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# ``sleap-nn predict`` unified inference command
# ──────────────────────────────────────────────────────────────────────────
#
# ``predict`` and ``track`` share an option list and dispatch to one
# implementation. The option list is built programmatically below
# (``_INFERENCE_OPTIONS``) so the commands stay in lockstep — adding
# a flag in one place updates all of them.
#
# ``infer`` is kept as a hidden backward-compatible alias that emits a
# deprecation warning and delegates to the same implementation.


def _run_inference_impl(**kwargs):
    """Implementation for ``sleap-nn predict`` (new pipeline only).

    Coerces tuple-shaped multi-options into lists, parses the
    ``--frames`` string into a list of int frame indices, and routes to
    the new :class:`Predictor`-based pipeline.
    """
    from sleap_nn.predict import frame_list

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

    # Exported ONNX/TRT model directories run through the in-memory flow only.
    _mps = kwargs["model_paths"]
    if (
        stream_to_file is not None
        and _mps
        and len(_mps) == 1
        and _is_export_dir(_mps[0])
    ):
        raise click.UsageError(
            "--stream-to-file is not supported for exported ONNX/TensorRT models. "
            "Run without --stream-to-file, or use a trained checkpoint directory."
        )

    # ── Stream-to-file path: new Predictor + IncrementalLabelsWriter ───
    if stream_to_file is not None:
        return _run_stream_to_file(
            kwargs,
            stream_to_file=stream_to_file,
            write_interval=write_interval or 500,
            paf_workers=paf_workers,
        )

    return _run_in_memory_new_flow(kwargs, paf_workers=paf_workers)


def _resolve_device(value: object) -> str:
    """Resolve a CLI ``--device`` value to a concrete torch device string.

    The legacy :func:`sleap_nn.predict.run_inference` resolves ``"auto"``
    before any checkpoint loading; the new flow needs the same so
    ``torch.load(map_location="auto")`` doesn't blow up on the legacy
    factory loader.
    """
    if hasattr(value, "type"):
        value = str(value)
    if value in (None, "", "auto"):
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return str(value)


def _build_preprocess_config(kwargs: dict):
    """Build an OmegaConf preprocess override from CLI flags, or ``None``."""
    from omegaconf import OmegaConf

    overrides = {
        "ensure_rgb": kwargs.get("ensure_rgb"),
        "ensure_grayscale": kwargs.get("ensure_grayscale"),
        "max_height": kwargs.get("max_height"),
        "max_width": kwargs.get("max_width"),
        "scale": kwargs.get("input_scale"),
        "crop_size": kwargs.get("crop_size"),
    }
    if any(v is not None for v in overrides.values()):
        return OmegaConf.create(overrides)
    return None


def _build_filter_config(kwargs: dict) -> "object":
    """Build a :class:`FilterConfig` from the CLI ``--filter_*`` flags.

    Returns ``None`` when every knob is at its default — the
    :class:`Predictor`'s default ``FilterConfig()`` is the no-op
    identity, so we save a few attrs constructions in the common case.
    """
    from sleap_nn.inference.filters import FilterConfig

    overlapping = bool(kwargs.get("filter_overlapping"))
    min_visible_nodes = int(kwargs.get("filter_min_visible_nodes") or 0)
    min_visible_node_fraction = float(
        kwargs.get("filter_min_visible_node_fraction") or 0.0
    )
    min_mean_node_score = float(kwargs.get("filter_min_mean_node_score") or 0.0)
    min_instance_score = float(kwargs.get("filter_min_instance_score") or 0.0)
    min_centroid_distance = float(kwargs.get("filter_min_centroid_distance") or 0.0)
    if not (
        overlapping
        or min_visible_nodes
        or min_visible_node_fraction
        or min_mean_node_score
        or min_instance_score
        or min_centroid_distance
    ):
        return None
    return FilterConfig(
        overlapping=overlapping,
        overlapping_method=kwargs.get("filter_overlapping_method", "iou"),
        overlapping_threshold=float(
            kwargs.get("filter_overlapping_threshold", 0.8) or 0.8
        ),
        min_visible_nodes=min_visible_nodes,
        min_visible_node_fraction=min_visible_node_fraction,
        min_mean_node_score=min_mean_node_score,
        min_instance_score=min_instance_score,
        min_centroid_distance=min_centroid_distance,
    )


def _build_tracker_config(kwargs: dict) -> "object":
    """Build a :class:`TrackerConfig` from the CLI ``--tracking_*`` flags.

    Replicates the legacy ``run_inference`` edge-layer defaulting (see
    ``sleap_nn/predict.py`` pre-#530) so the new ``predict`` flow behaves
    identically (#582):

    * ``--max_tracks`` with no ``--candidates_method`` defaults the method to
      ``local_queues`` (``max_tracks`` is silently ignored by ``fixed_window``).
    * ``--post_connect_single_breaks`` / ``--tracking_pre_cull_to_target`` with
      only ``--max_instances`` (no explicit ``--tracking_target_instance_count``)
      derive the target count from ``max_instances`` instead of crashing /
      silently no-op'ing.
    * ``--post_connect_single_breaks`` with no ``--max_tracks`` derives
      ``max_tracks`` from ``max_instances`` (legacy track-only default).
    """
    from sleap_nn.inference.tracking import TrackerConfig

    max_instances = kwargs.get("max_instances")
    target = kwargs.get("tracking_target_instance_count")
    max_tracks = kwargs.get("max_tracks")
    pre_cull = kwargs.get("tracking_pre_cull_to_target", 0)
    pcsb = kwargs.get("post_connect_single_breaks", False)
    use_kalman = kwargs.get("use_kalman", False)

    # Default candidates_method to local_queues when max_tracks is set but the
    # user did not explicitly choose a method (the click default is None).
    candidates_method = kwargs.get("candidates_method")
    if candidates_method is None:
        candidates_method = "local_queues" if max_tracks is not None else "fixed_window"

    # Legacy: post_connect_single_breaks defaults max_tracks from max_instances.
    if pcsb and max_tracks is None:
        max_tracks = max_instances

    # Legacy: post_connect / pre_cull derive the target count from max_instances
    # when not given explicitly (leaves target=None when both are None, so the
    # apply_tracking gate raises exactly as legacy did). Kalman tracking also
    # requires a target identity count, so derive it from max_instances too.
    if (pcsb or pre_cull or use_kalman) and target is None:
        target = max_instances

    # --features / --scoring_method default to None (sentinel for "user left the
    # default"). Record whether they were set so apply_tracking can substitute
    # single-node / centroid defaults; fall back to the historical
    # keypoints/oks defaults for the effective values when unset (#586).
    features = kwargs.get("features")
    scoring_method = kwargs.get("scoring_method")
    features_explicit = features is not None
    scoring_method_explicit = scoring_method is not None
    if features is None:
        features = "keypoints"
    if scoring_method is None:
        scoring_method = "oks"

    return TrackerConfig(
        window_size=kwargs.get("tracking_window_size", 5),
        min_new_track_points=kwargs.get("min_new_track_points", 0),
        candidates_method=candidates_method,
        min_match_points=kwargs.get("min_match_points", 0),
        features=features,
        scoring_method=scoring_method,
        features_explicit=features_explicit,
        scoring_method_explicit=scoring_method_explicit,
        scoring_reduction=kwargs.get("scoring_reduction", "mean"),
        robust_best_instance=kwargs.get("robust_best_instance", 1.0),
        oks_stddev=kwargs.get("oks_stddev", 0.025),
        track_matching_method=kwargs.get("track_matching_method", "hungarian"),
        max_tracks=max_tracks,
        use_flow=kwargs.get("use_flow", False),
        of_img_scale=kwargs.get("of_img_scale", 1.0),
        of_window_size=kwargs.get("of_window_size", 21),
        of_max_levels=kwargs.get("of_max_levels", 3),
        use_kalman=use_kalman,
        kf_track_features=kwargs.get("kf_track_features", "centroid"),
        kf_init_frame_count=kwargs.get("kf_init_frame_count", 10),
        kf_node_indices=kwargs.get("kf_node_indices"),
        kf_reset_gap_size=kwargs.get("kf_reset_gap_size", 5),
        tracking_target_instance_count=target,
        tracking_pre_cull_to_target=pre_cull,
        tracking_pre_cull_iou_threshold=kwargs.get(
            "tracking_pre_cull_iou_threshold", 0.0
        ),
        tracking_clean_instance_count=kwargs.get("tracking_clean_instance_count", 0),
        tracking_clean_iou_threshold=kwargs.get("tracking_clean_iou_threshold", 0.0),
        post_connect_single_breaks=pcsb,
    )


def _scope_labels_to_video(labels, video_index: int, frames=None):
    """Scope a ``Labels`` to one video (re-indexed to slot 0), optionally + frames.

    Returns ``(scoped_labels, target_video)``. Raises ``click.UsageError`` for an
    out-of-range ``video_index``. Carries the video's ``suggestions`` and the
    source ``provenance`` so ``--video_index`` composes with
    ``--only_suggested_frames`` / ``--frames`` and preserves input lineage
    (legacy parity, #583).
    """
    import sleap_io as sio

    if video_index >= len(labels.videos):
        raise click.UsageError(
            f"--video_index {video_index} is out of range: the .slp has "
            f"{len(labels.videos)} video(s)."
        )
    target = labels.videos[video_index]
    wanted = set(frames) if frames else None
    lfs = [
        lf
        for lf in labels.find(video=target)
        if wanted is None or lf.frame_idx in wanted
    ]
    suggestions = [
        s
        for s in (getattr(labels, "suggestions", None) or [])
        if s.video is target and (wanted is None or s.frame_idx in wanted)
    ]
    scoped = sio.Labels(
        videos=[target],
        skeletons=labels.skeletons,
        labeled_frames=lfs,
        suggestions=suggestions,
        provenance=dict(getattr(labels, "provenance", None) or {}),
    )
    return scoped, target


def _is_export_dir(path: str) -> bool:
    """True if ``path`` is a directory holding an exported ONNX/TRT model.

    A single ``--model_paths`` entry that is a directory containing
    ``model.onnx`` or ``model.trt`` is an exported model and routes to the
    ONNX/TensorRT runtime instead of checkpoint loading. The exporter writes
    these artifacts to a dedicated directory (never alongside ``best.ckpt``),
    so this never misfires on a checkpoint directory.
    """
    from pathlib import Path

    p = Path(path)
    return p.is_dir() and ((p / "model.onnx").exists() or (p / "model.trt").exists())


def _run_in_memory_new_flow(kwargs: dict, paf_workers: int) -> "object":
    """Run the new ``predict()`` flow synchronously and save the resulting Labels.

    Routes to :meth:`Predictor.retrack` for the tracking-only retrack
    case (no ``model_paths``, ``--tracking`` set, ``.slp`` data path);
    auto-detects an exported ONNX/TRT model directory in ``model_paths`` and
    routes it through the exported-model runtime; otherwise delegates to
    :func:`sleap_nn.inference.run.predict` for trained checkpoints.
    """
    from pathlib import Path

    from sleap_nn.inference.predictor import Predictor

    # ── Tracking-only retrack: no model_paths, --tracking on a .slp ────
    if not kwargs.get("model_paths") and kwargs.get("tracking"):
        return _run_retrack_only(kwargs, Predictor)

    from sleap_nn.inference.providers import LabelsProvider, VideoProvider
    from sleap_nn.inference.run import predict

    src = Path(kwargs["data_path"])

    # Build source: use a provider when CLI-specific filtering or
    # video kwargs are needed, otherwise pass the raw path.
    has_slp_filters = any(
        kwargs.get(k)
        for k in (
            "only_labeled_frames",
            "only_suggested_frames",
            "exclude_user_labeled",
            "only_predicted_frames",
        )
    )
    # Suffix for the auto-derived output filename when --video_index scopes a
    # single video of a multi-video .slp (legacy parity, so per-video runs don't
    # collide on one output path). Stays None for the non-video_index case. #583.
    scoped_video_name = None
    video_index = kwargs.get("video_index")
    if src.suffix == ".slp" and video_index is not None:
        import sleap_io as sio

        # Scope to the requested video (re-indexed to slot 0 so its frames map to
        # videos[0] and avoid an out-of-range video index in packaging). The
        # frames pass through a pre-built LabelsProvider, so the real Video is not
        # re-attached on output (same as the has_slp_filters path); the output is
        # video-name-suffixed instead. Carries suggestions + the --frames filter.
        scoped, target_video = _scope_labels_to_video(
            sio.load_slp(str(src)), video_index, frames=kwargs.get("frames")
        )
        source = LabelsProvider(
            labels=scoped,
            batch_size=kwargs.get("batch_size", 4),
            only_labeled_frames=bool(kwargs.get("only_labeled_frames")),
            only_suggested_frames=bool(kwargs.get("only_suggested_frames")),
            exclude_user_labeled=bool(kwargs.get("exclude_user_labeled")),
            only_predicted_frames=bool(kwargs.get("only_predicted_frames")),
        )
        _vfn = getattr(target_video, "filename", None)
        scoped_video_name = Path(str(_vfn)).stem if _vfn else f"video_{video_index}"
    elif src.suffix == ".slp" and has_slp_filters:
        source = LabelsProvider(
            labels=str(src),
            batch_size=kwargs.get("batch_size", 4),
            only_labeled_frames=bool(kwargs.get("only_labeled_frames")),
            only_suggested_frames=bool(kwargs.get("only_suggested_frames")),
            exclude_user_labeled=bool(kwargs.get("exclude_user_labeled")),
            only_predicted_frames=bool(kwargs.get("only_predicted_frames")),
        )
    elif src.suffix != ".slp" and (
        kwargs.get("video_dataset")
        or kwargs.get("video_input_format", "channels_last") != "channels_last"
    ):
        source = VideoProvider(
            video=str(src),
            batch_size=kwargs.get("batch_size", 4),
            frames=kwargs.get("frames"),
            dataset=kwargs.get("video_dataset"),
            input_format=kwargs.get("video_input_format"),
        )
    else:
        source = str(src)

    peak_thresh = kwargs.get("peak_threshold", 0.2)
    centroid_thresh = kwargs.get("centroid_peak_threshold") or peak_thresh

    # Auto-detect an exported ONNX/TRT model directory: a single --model_paths
    # entry containing model.onnx/model.trt runs through the exported-model
    # runtime. Multiple paths are always trained checkpoints (top-down).
    _mps = kwargs["model_paths"]
    export_dir = (
        _mps[0] if _mps and len(_mps) == 1 and _is_export_dir(_mps[0]) else None
    )

    predict_kwargs: dict = {
        "model_paths": kwargs["model_paths"],
        "device": _resolve_device(kwargs.get("device")),
        "batch_size": kwargs.get("batch_size", 4),
        "paf_workers": paf_workers,
        "peak_threshold": peak_thresh,
        "centroid_threshold": centroid_thresh,
        "keypoint_threshold": peak_thresh,
        "integral_refinement": kwargs.get("integral_refinement", "integral"),
        "integral_patch_size": kwargs.get("integral_patch_size", 5),
        "max_instances": kwargs.get("max_instances"),
        "anchor_part": kwargs.get("anchor_part"),
        "frames": kwargs.get("frames"),
        "clean_empty_frames": bool(kwargs.get("no_empty_frames")),
        "output_path": (
            kwargs.get("output_path")
            or (
                f"{src.with_suffix('')}.{scoped_video_name}.predictions.slp"
                if scoped_video_name is not None
                else f"{src}.slp"
            )
        ),
        "output_format": kwargs.get("output_format") or "slp",
        # Bottom-up PAF grouping knobs (inert for non-bottom-up models). #583.
        "max_edge_length_ratio": kwargs.get("max_edge_length_ratio", 0.25),
        "dist_penalty_weight": kwargs.get("dist_penalty_weight", 1.0),
        "n_points": kwargs.get("n_points", 10),
        "min_instance_peaks": kwargs.get("min_instance_peaks", 0),
        "min_line_scores": kwargs.get("min_line_scores", 0.25),
        # Bottom-up segmentation knobs (inert for non-segmentation models).
        "fg_threshold": kwargs.get("fg_threshold", 0.5),
        "min_mask_area": kwargs.get("min_mask_area", 0),
        "center_nms_kernel": kwargs.get("center_nms_kernel", 3),
        "mask_cleanup": kwargs.get("mask_cleanup", False),
        "mask_cleanup_radius": kwargs.get("mask_cleanup_radius", 0),
        "full_res_masks": kwargs.get("full_res_masks", False),
        "mask_output": kwargs.get("mask_output", "mask"),
        "polygon_epsilon": kwargs.get("polygon_epsilon", 0.01),
    }
    preprocess_config = _build_preprocess_config(kwargs)
    if preprocess_config is not None:
        predict_kwargs["preprocess_config"] = preprocess_config
    if kwargs.get("backbone_ckpt_path"):
        predict_kwargs["backbone_ckpt_path"] = kwargs["backbone_ckpt_path"]
    if kwargs.get("head_ckpt_path"):
        predict_kwargs["head_ckpt_path"] = kwargs["head_ckpt_path"]
    if kwargs.get("tracking"):
        predict_kwargs["tracker_config"] = _build_tracker_config(kwargs)
    if kwargs.get("centroid_only"):
        predict_kwargs["centroid_only"] = True
    if kwargs.get("centroid_output") and kwargs["centroid_output"] != "instance":
        predict_kwargs["emit_centroid"] = kwargs["centroid_output"]
    filter_config = _build_filter_config(kwargs)
    if filter_config is not None:
        predict_kwargs["filter_config"] = filter_config

    if export_dir is not None:
        # Route to the exported ONNX/TRT runtime. Exported graphs bake peak
        # finding and refinement at export time, so the checkpoint-oriented
        # construction/prediction-time knobs are dropped (the baked values are
        # used, mirroring the old `export predict` command). The bottom-up
        # grouping knobs (min_instance_peaks/min_line_scores), max_instances,
        # emit_centroid, paf_workers, filtering, and tracking still apply and
        # are forwarded by run.predict() to from_export_dir.
        predict_kwargs["model_paths"] = None
        predict_kwargs["export_dir"] = export_dir
        predict_kwargs["runtime"] = (kwargs.get("runtime") or "auto").lower()
        for _k in (
            "peak_threshold",
            "centroid_threshold",
            "keypoint_threshold",
            "integral_refinement",
            "integral_patch_size",
            "anchor_part",
            "max_edge_length_ratio",
            "dist_penalty_weight",
            "n_points",
            "fg_threshold",
            "min_mask_area",
            "mask_cleanup_radius",
            "full_res_masks",
            "mask_output",
            "polygon_epsilon",
            "backbone_ckpt_path",
            "head_ckpt_path",
            "preprocess_config",
            "centroid_only",
        ):
            predict_kwargs.pop(_k, None)

    if kwargs.get("gui"):
        predict_kwargs["progress_callback"] = _gui_progress_callback()
        if kwargs.get("tracking"):
            predict_kwargs["tracking_progress_callback"] = _gui_progress_callback()
        return predict(source, **predict_kwargs)

    # Non-gui: show a Rich progress bar (parity with legacy `track`'s default
    # progress UX; the plumbing was wired but no callback was attached). #583.
    progress = _make_rich_progress()
    predict_kwargs["progress_callback"] = _rich_task_callback(progress, "Predicting...")
    if kwargs.get("tracking"):
        predict_kwargs["tracking_progress_callback"] = _rich_task_callback(
            progress, "Tracking..."
        )
    try:
        return predict(source, **predict_kwargs)
    finally:
        progress.stop()


def _run_retrack_only(kwargs: dict, predictor_cls) -> "object":
    """Pure-tracking retrack of an existing ``.slp`` (no inference)."""
    from pathlib import Path

    import sleap_io as sio

    src = Path(kwargs["data_path"])
    if src.suffix != ".slp":
        raise click.UsageError(
            "Tracking-only mode requires --data_path to be a .slp file. "
            "Pass --model_paths to run inference + tracking."
        )

    labels = sio.load_slp(str(src))

    # Scope by --video_index (raises on out-of-range, like the inference paths)
    # and/or --frames, carrying suggestions + the source provenance. #583.
    video_index = kwargs.get("video_index")
    frames = kwargs.get("frames")
    if video_index is not None:
        labels, _ = _scope_labels_to_video(labels, video_index, frames=frames)
    elif frames:
        wanted = set(frames)
        labels = sio.Labels(
            videos=labels.videos,
            skeletons=labels.skeletons,
            labeled_frames=[
                lf for lf in labels.labeled_frames if lf.frame_idx in wanted
            ],
            suggestions=labels.suggestions,
            provenance=dict(getattr(labels, "provenance", None) or {}),
        )

    import attrs as _attrs

    from sleap_nn.inference.provenance import build_tracking_only_provenance

    tracker_config = _build_tracker_config(kwargs)

    if kwargs.get("gui"):
        tracking_cb = _gui_progress_callback()
    else:
        _progress = _make_rich_progress()
        tracking_cb = _rich_task_callback(_progress, "Tracking...")

    _start = datetime.now()
    try:
        out = predictor_cls.retrack(
            labels,
            tracker_config,
            clean_empty_frames=bool(kwargs.get("no_empty_frames")),
            progress_callback=tracking_cb,
        )
    finally:
        if not kwargs.get("gui"):
            _progress.stop()
    # Attach tracking-only provenance to the retracked .slp (legacy parity —
    # apply_tracking leaves provenance to the caller, and the legacy track path
    # set it; the new retrack flow previously saved with empty provenance). #583.
    out.provenance = build_tracking_only_provenance(
        input_labels=labels,
        input_path=str(src),
        start_time=_start,
        end_time=datetime.now(),
        tracking_params=_attrs.asdict(tracker_config),
        frames_processed=len(out.labeled_frames),
    )
    from sleap_nn.inference.run import save_predictions

    output_path = kwargs.get("output_path") or f"{src}.slp"
    save_predictions(
        out, output_path, output_format=kwargs.get("output_format") or "slp"
    )
    return out


def _gui_progress_callback():
    """Build a JSON-progress emitter compatible with the legacy ``--gui`` mode.

    Emits one JSON line per batch with ``n_processed`` / ``n_total`` /
    ``rate`` / ``eta``, throttled to ~4Hz so a downstream GUI process
    can read progress via stdout. Matches the format used by the legacy
    ``Predictor._predict_generator_gui`` exactly.
    """
    import json
    from time import time as _now

    state = {"start": _now(), "last": 0.0, "processed_at_last": 0}

    def cb(processed: int, total: int) -> None:
        now = _now()
        is_last = total > 0 and processed >= total
        if not is_last and (now - state["last"]) < 0.25:
            return
        elapsed = now - state["start"]
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(total - processed, 0) if total > 0 else 0
        eta = remaining / rate if rate > 0 else 0.0
        print(
            json.dumps(
                {
                    "n_processed": processed,
                    "n_total": total if total > 0 else None,
                    "rate": round(rate, 1),
                    "eta": round(eta, 1),
                }
            ),
            flush=True,
        )
        state["last"] = now

    return cb


def _make_fps_column(window_s: float = 5.0, time_fn=None):
    """Build a Rich ``ProgressColumn`` that shows frames/sec over a window.

    Reports throughput across a trailing ``window_s``-second window rather
    than the whole-run average, so model warmup / the slow first batch don't
    drag the number down — it answers "how fast is it *now*". Renders ``--``
    until two samples spanning a positive interval accrue (#610).

    Samples ``(time, task.completed)`` on each ``render`` and keys the sample
    history by ``task.id``, so one shared column correctly serves both the
    "Predicting..." and "Tracking..." tasks. ``rich`` is imported lazily so
    it doesn't slow ``--help``. ``time_fn`` defaults to ``time.monotonic`` and
    is injectable for deterministic tests.
    """
    from collections import deque
    from time import monotonic

    from rich.progress import ProgressColumn
    from rich.text import Text

    if time_fn is None:
        time_fn = monotonic

    class _FPSColumn(ProgressColumn):
        def __init__(self, window_s: float, time_fn):
            super().__init__()
            self.window_s = window_s
            self._time_fn = time_fn
            self._samples: dict = {}  # task_id -> deque[(t, completed)]

        def render(self, task) -> Text:
            samples = self._samples.setdefault(task.id, deque())
            t = self._time_fn()
            samples.append((t, float(task.completed)))
            while samples and t - samples[0][0] > self.window_s:
                samples.popleft()
            if len(samples) < 2:
                return Text("-- fps", style="progress.percentage")
            (t0, c0), (t1, c1) = samples[0], samples[-1]
            dt = t1 - t0
            if dt <= 0:
                return Text("-- fps", style="progress.percentage")
            return Text(f"{(c1 - c0) / dt:5.1f} fps", style="progress.percentage")

    return _FPSColumn(window_s, time_fn)


def _make_rich_progress():
    """Build a shared Rich ``Progress`` instance for the ``predict`` path.

    Returns the ``Progress`` (already started) — the caller MUST stop it
    in a ``finally`` block. Counts are in frames (#610), so the windowed
    FPS column reads as true frames/sec.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        _make_fps_column(),
        "ETA:",
        TimeRemainingColumn(),
        "Elapsed:",
        TimeElapsedColumn(),
        refresh_per_second=4,
    )
    progress.start()
    return progress


def _rich_task_callback(progress, description: str = "Predicting..."):
    """Build a callback that drives a task on an existing ``Progress``.

    Returns ``callback(processed, total)`` that lazily adds a task with
    *description* to *progress* on the first call and updates it on each
    subsequent call. This allows the task to appear only when the phase
    actually starts (e.g. tracking may not run at all).
    """
    state: dict = {"task": None}

    def cb(processed: int, total: int) -> None:
        if state["task"] is None:
            state["task"] = progress.add_task(description, total=total or None)
        elif total and total > 0:
            progress.update(state["task"], total=total)
        progress.update(state["task"], completed=processed)

    return cb


def _rich_progress_callback():
    """Build a Rich progress bar callback for the non-``--gui`` ``predict`` path.

    Returns ``(callback, progress)`` where ``callback(processed, total)`` has
    the ``(processed_frames, total_frames)`` signature the new pipeline uses.
    The ``Progress`` is started immediately (so "Predicting..." shows right
    away) and the caller MUST stop it in a ``finally`` block so the bar closes
    cleanly on success or error. A ``total <= 0`` (length-less provider)
    renders an indeterminate bar.

    ``rich`` is imported lazily so it doesn't slow ``--help``; the columns
    mirror the legacy ``track`` look-and-feel without importing the heavy
    legacy ``predictors`` module. Progress is counted in frames, so the
    count / percent / ETA / FPS are all batch-size-invariant (#610). #583.
    """
    progress = _make_rich_progress()
    cb = _rich_task_callback(progress, "Predicting...")
    return cb, progress


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
    if kwargs.get("no_empty_frames"):
        raise click.UsageError(
            "--no_empty_frames is incompatible with --stream-to-file: "
            "streaming writes each batch to disk and cannot drop empty "
            "frames after the fact. Drop --stream-to-file to use it."
        )
    if (kwargs.get("output_format") or "slp").lower() != "slp":
        raise click.UsageError(
            "--stream-to-file only supports --output_format slp. Drop "
            "--stream-to-file to write analysis HDF5 via the in-memory path."
        )
    if not kwargs.get("model_paths"):
        raise click.UsageError("--model_paths is required for --stream-to-file.")
    data_path = kwargs.get("data_path")
    if not data_path:
        raise click.UsageError("--data_path is required for --stream-to-file.")

    from pathlib import Path

    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider, VideoProvider

    factory_kwargs = {
        "device": _resolve_device(kwargs.get("device")),
        "peak_threshold": kwargs.get("peak_threshold", 0.2),
        "integral_refinement": kwargs.get("integral_refinement", "integral"),
        "integral_patch_size": kwargs.get("integral_patch_size", 5),
        "batch_size": kwargs.get("batch_size", 4),
        "max_instances": kwargs.get("max_instances"),
        "return_confmaps": False,
        "anchor_part": kwargs.get("anchor_part"),
        "paf_workers": paf_workers,
        # Bottom-up PAF grouping knobs (inert for non-bottom-up models). #583.
        "max_edge_length_ratio": kwargs.get("max_edge_length_ratio", 0.25),
        "dist_penalty_weight": kwargs.get("dist_penalty_weight", 1.0),
        "n_points": kwargs.get("n_points", 10),
        "min_instance_peaks": kwargs.get("min_instance_peaks", 0),
        "min_line_scores": kwargs.get("min_line_scores", 0.25),
        # Bottom-up segmentation knobs (inert for non-segmentation models).
        "fg_threshold": kwargs.get("fg_threshold", 0.5),
        "min_mask_area": kwargs.get("min_mask_area", 0),
        "center_nms_kernel": kwargs.get("center_nms_kernel", 3),
        "mask_cleanup": kwargs.get("mask_cleanup", False),
        "mask_cleanup_radius": kwargs.get("mask_cleanup_radius", 0),
        "full_res_masks": kwargs.get("full_res_masks", False),
        "mask_output": kwargs.get("mask_output", "mask"),
        "polygon_epsilon": kwargs.get("polygon_epsilon", 0.01),
    }
    preprocess_config = _build_preprocess_config(kwargs)
    if preprocess_config is not None:
        factory_kwargs["preprocess_config"] = preprocess_config
    if kwargs.get("backbone_ckpt_path"):
        factory_kwargs["backbone_ckpt_path"] = kwargs["backbone_ckpt_path"]
    if kwargs.get("head_ckpt_path"):
        factory_kwargs["head_ckpt_path"] = kwargs["head_ckpt_path"]
    if kwargs.get("centroid_only"):
        factory_kwargs["centroid_only"] = True
    if kwargs.get("centroid_output") and kwargs["centroid_output"] != "instance":
        factory_kwargs["emit_centroid"] = kwargs["centroid_output"]
    filter_config = _build_filter_config(kwargs)
    if filter_config is not None:
        factory_kwargs["filter_config"] = filter_config

    predictor = Predictor.from_model_paths(kwargs["model_paths"], **factory_kwargs)

    src = Path(data_path)
    video_index = kwargs.get("video_index")
    if src.suffix == ".slp" and video_index is not None:
        # Scope streaming inference to the requested video of a multi-video .slp
        # (re-indexed to videos[0] so frames map correctly). Carries suggestions
        # + the --frames filter. #583.
        import sleap_io as sio

        scoped, _target_video = _scope_labels_to_video(
            sio.load_slp(str(src)), video_index, frames=kwargs.get("frames")
        )
        provider = LabelsProvider(
            labels=scoped,
            batch_size=kwargs.get("batch_size", 4),
            only_labeled_frames=bool(kwargs.get("only_labeled_frames")),
            only_suggested_frames=bool(kwargs.get("only_suggested_frames")),
            exclude_user_labeled=bool(kwargs.get("exclude_user_labeled")),
            only_predicted_frames=bool(kwargs.get("only_predicted_frames")),
        )
    elif src.suffix == ".slp":
        provider = LabelsProvider(
            labels=str(src),
            batch_size=kwargs.get("batch_size", 4),
            only_labeled_frames=bool(kwargs.get("only_labeled_frames")),
            only_suggested_frames=bool(kwargs.get("only_suggested_frames")),
            exclude_user_labeled=bool(kwargs.get("exclude_user_labeled")),
            only_predicted_frames=bool(kwargs.get("only_predicted_frames")),
        )
    else:
        provider = VideoProvider(
            video=str(src),
            batch_size=kwargs.get("batch_size", 4),
            frames=kwargs.get("frames"),
            dataset=kwargs.get("video_dataset"),
            input_format=kwargs.get("video_input_format"),
        )

    if kwargs.get("gui"):
        return predictor.predict_to_file(
            provider,
            path=str(stream_to_file),
            write_interval=write_interval,
            progress_callback=_gui_progress_callback(),
        )

    # Non-gui: Rich progress bar, stopped cleanly in finally (#583).
    cb, progress = _rich_progress_callback()
    try:
        return predictor.predict_to_file(
            provider,
            path=str(stream_to_file),
            write_interval=write_interval,
            progress_callback=cb,
        )
    finally:
        progress.stop()


def _common_inference_options(f):
    """Apply the shared inference flag list to a click command function.

    Defined as a function-level decorator (not a ``@click.option`` chain)
    so the same option set can be reused across ``predict`` / ``track``
    without copy-pasting ~70 decorator lines per command.
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
            help="Path to a trained model directory, or to its best.ckpt or training_config.yaml/.json file (all resolve to the model directory). Multiple models may be passed (each preceded by --model_paths).",
        ),
        click.option(
            "--output_path",
            "-o",
            type=str,
            default=None,
            help="Output filename. Defaults to '[data_path].slp'.",
        ),
        click.option(
            "--output_format",
            type=click.Choice(["slp", "analysis_h5", "both"], case_sensitive=False),
            default="slp",
            help="Output format: 'slp' (SLEAP labels file, the default), 'analysis_h5' (SLEAP Analysis HDF5, one '.analysis.h5' per video), or 'both'.",
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
            "--runtime",
            type=click.Choice(["auto", "onnx", "tensorrt"], case_sensitive=False),
            default="auto",
            help="Runtime for an exported ONNX/TensorRT model directory passed to "
            "--model_paths. 'auto' prefers TensorRT, falls back to ONNX. Ignored "
            "for trained checkpoints.",
        ),
        click.option(
            "--tracking",
            "-t",
            is_flag=True,
            default=False,
            help="Run tracking on predicted instances.",
        ),
        click.option(
            "--centroid_only",
            "--centroid-only",
            "centroid_only",
            is_flag=True,
            default=False,
            help=(
                "Force centroid-only output. Required only when both a "
                "centroid and a centered-instance model are configured but "
                "you want centroid-only predictions; if model_paths contains "
                "a single centroid model, this mode is auto-detected. For a "
                "model trained on a multi-node skeleton, output collapses to a "
                "single-node 'centroid' skeleton; use --centroid-output to "
                "choose the representation."
            ),
        ),
        click.option(
            "--centroid-output",
            "--centroid_output",
            "centroid_output",
            type=click.Choice(["instance", "centroid", "both"]),
            default="instance",
            show_default=True,
            help=(
                "Centroid-only output representation: 'instance' (single-node "
                "PredictedInstance, loadable by the current frontend), "
                "'centroid' (sio.PredictedCentroid in LabeledFrame.centroids), "
                "or 'both'. Only meaningful for centroid-only models."
            ),
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
        click.option(
            "--fg_threshold",
            "--fg-threshold",
            "fg_threshold",
            type=float,
            default=0.5,
            help="Foreground probability threshold for binarizing the "
            "segmentation map (bottom-up segmentation models only).",
        ),
        click.option(
            "--min_mask_area",
            "--min-mask-area",
            "min_mask_area",
            type=int,
            default=0,
            help="Drop predicted masks smaller than this many original-image "
            "pixels to suppress over-segmentation (bottom-up segmentation "
            "models only). 0 disables. Measured at output-stride resolution and "
            "converted from original-pixel units.",
        ),
        click.option(
            "--center_nms_kernel",
            "--center-nms-kernel",
            "center_nms_kernel",
            type=int,
            default=3,
            help="Odd window size for instance-center peak NMS; larger values "
            "merge nearby duplicate centers (bottom-up segmentation models "
            "only).",
        ),
        click.option(
            "--mask_cleanup/--no-mask_cleanup",
            "--mask-cleanup/--no-mask-cleanup",
            "mask_cleanup",
            default=False,
            help="Keep only each predicted mask's largest connected component "
            "and fill interior holes (bottom-up segmentation models only).",
        ),
        click.option(
            "--mask_cleanup_radius",
            "--mask-cleanup-radius",
            "mask_cleanup_radius",
            type=int,
            default=0,
            help="When --mask_cleanup is set, also apply a morphological "
            "open->close with an elliptical kernel of this radius (output-stride "
            "pixels) before keep-largest-CC/fill-holes; despeckles and closes "
            "pinholes. 0 keeps the keep-largest+fill behavior (bottom-up "
            "segmentation models only).",
        ),
        click.option(
            "--full_res_masks/--no-full_res_masks",
            "--full-res-masks/--no-full-res-masks",
            "full_res_masks",
            default=False,
            help="Encode predicted segmentation masks at full ORIGINAL "
            "resolution instead of the model output-stride grid. Default off: "
            "output-stride encoding is ~stride^2 smaller and lossless at model "
            "resolution, carrying the image mapping via sio scale/offset. Enable "
            "only for legacy consumers that read mask.data assuming original "
            "resolution (bottom-up segmentation models only).",
        ),
        click.option(
            "--mask_output",
            "--mask-output",
            "mask_output",
            type=click.Choice(["mask", "polygon", "both"]),
            default="mask",
            help="Predicted-mask output representation: 'mask' (RLE "
            "SegmentationMask in LabeledFrame.masks, default), 'polygon' "
            "(Douglas-Peucker-simplified sio.PredictedROI into LabeledFrame.rois "
            "only), or 'both' (exact mask + simplified ROI for interop). "
            "polygon/both are CPU-heavy on noisy masks (cost scales with RLE-run "
            "count) — pair with --mask_cleanup. Bottom-up segmentation models only.",
        ),
        click.option(
            "--polygon_epsilon",
            "--polygon-epsilon",
            "polygon_epsilon",
            type=float,
            default=0.01,
            help="Douglas-Peucker simplification tolerance for --mask_output "
            "polygon/both, as a fraction of each contour's perimeter. Larger = "
            "coarser polygons. 0 disables simplification (bottom-up segmentation "
            "models only).",
        ),
        click.option(
            "--queue_maxsize",
            type=int,
            default=32,
            hidden=True,
            help="[no-op] Retained for CLI compatibility; ignored by the new "
            "inference pipeline (which has no frame-buffer queue). The legacy "
            "`sleap-nn track` path still honors it.",
        ),
        click.option("--crop_size", type=int, default=None),
        click.option(
            "--peak_threshold",
            "--peak-conf-threshold",
            "peak_threshold",
            type=float,
            default=0.2,
            help="Min confmap value for a valid peak. --peak-conf-threshold is an alias.",
        ),
        click.option(
            "--centroid_peak_threshold",
            type=float,
            default=None,
            help=(
                "Override peak threshold for the centroid stage only (top-down). "
                "Defaults to --peak_threshold when not set."
            ),
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
        click.option(
            "--filter_min_centroid_distance",
            type=float,
            default=0.0,
            help=(
                "Centroid-only de-duplication radius in pixels: greedy NMS drops "
                "any predicted centroid within this distance of a higher-scored "
                "kept centroid. 0 disables. Use this (not --filter_overlapping) "
                "for centroid-only output, since bbox-IoU/OKS are degenerate for "
                "single points."
            ),
        ),
        click.option("--integral_refinement", type=str, default="integral"),
        click.option("--tracking_window_size", type=int, default=5),
        click.option("--min_new_track_points", type=int, default=0),
        # Default None (not "fixed_window") so _build_tracker_config can tell
        # "user didn't choose a method" from an explicit choice, and default it
        # to local_queues when --max_tracks is set (#582).
        click.option("--candidates_method", type=str, default=None),
        click.option("--min_match_points", type=int, default=0),
        # Default None (not "keypoints"/"oks") so _build_tracker_config can tell
        # "user left the default" from an explicit choice, mirroring the
        # --candidates_method default=None pattern. A single-node / centroid
        # model then resolves these to centroids/euclidean_dist in
        # apply_tracking (#586).
        click.option("--features", type=str, default=None),
        click.option(
            "--scoring_method",
            type=str,
            default=None,
            help="Track association scoring method. Single-node / centroid "
            "models resolve to euclidean_dist when left unset.",
        ),
        click.option("--scoring_reduction", type=str, default="mean"),
        click.option("--robust_best_instance", type=float, default=1.0),
        click.option("--track_matching_method", type=str, default="hungarian"),
        click.option("--max_tracks", type=int, default=None),
        click.option("--use_flow", is_flag=True, default=False),
        click.option("--of_img_scale", type=float, default=1.0),
        click.option("--of_window_size", type=int, default=21),
        click.option("--of_max_levels", type=int, default=3),
        click.option("--use_kalman", is_flag=True, default=False),
        click.option(
            "--kf_track_features",
            type=click.Choice(["centroid", "keypoints"]),
            default="centroid",
        ),
        click.option("--oks_stddev", type=float, default=None),
        click.option("--kf_init_frame_count", type=int, default=10),
        click.option(
            "--kf_node_indices", type=str, default=None, callback=_parse_int_list
        ),
        click.option("--kf_reset_gap_size", type=int, default=5),
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
                "Write predictions to this .slp path via the new "
                "Predictor.predict_to_file flow. Heavy intermediate tensors "
                "(confmaps/PAFs) are dropped per batch; the .slp is written "
                "once at the end."
            ),
        ),
        click.option(
            "--write-interval",
            "write_interval",
            type=int,
            default=None,
            help=(
                "How often (in frames) to slim+convert buffered outputs to "
                "LabeledFrames when --stream-to-file is set. Default: 500."
            ),
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


@cli.command(context_settings=CONTEXT_SETTINGS)
@_common_inference_options
def predict(**kwargs):
    """Run inference on videos or labels files.

    Single unified inference entry point.
    """
    return _run_inference_impl(**kwargs)


@cli.command("infer", hidden=True, context_settings=CONTEXT_SETTINGS)
@_common_inference_options
def infer(**kwargs):
    """Deprecated alias for ``predict``. Use ``sleap-nn predict`` instead."""
    import warnings

    warnings.warn(
        "sleap-nn infer is deprecated. Use sleap-nn predict instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _run_inference_impl(**kwargs)


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
@click.option(
    "--match_method",
    type=click.Choice(["oks", "centroid", "auto"]),
    default="auto",
    help=(
        "Matching method: 'oks' (full-skeleton), 'centroid' (single-point "
        "pixel-distance), or 'auto' (centroid when the prediction skeleton is "
        "single-node). Default: auto."
    ),
)
@click.option(
    "--anchor_part",
    type=str,
    default=None,
    help=(
        "GT skeleton node used to compute ground-truth centroids in centroid "
        "mode. Defaults to the mean of visible nodes when absent (#586)."
    ),
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
    """Lazily import and register the export command."""
    from sleap_nn.export.cli import export as export_command

    cli.add_command(export_command)


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
            "centroid",
            "bottomup",
            "topdown",
            "multi_class_bottomup",
            "multi_class_topdown",
        ]
    ),
    default=None,
    help=(
        "Override model pipeline type. 'centroid' generates a single STANDALONE "
        "centroid config (one point per animal); 'topdown' generates paired "
        "centroid + centered_instance configs."
    ),
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
            # 'topdown' is a CLI alias for the centroid STAGE (is_topdown=True),
            # which triggers paired centroid + centered_instance generation.
            # 'centroid' is the STANDALONE single-config centroid model
            # (is_topdown=False), emitted via the "centroid_only" pipeline.
            if pipeline == "topdown":
                gen.pipeline("centroid")
            elif pipeline == "centroid":
                gen.pipeline("centroid_only")
            else:
                gen.pipeline(pipeline)

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
