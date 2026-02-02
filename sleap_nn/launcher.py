"""Launcher for distributed training via torchrun.

This module provides the "driver" that:
1. Loads config with Hydra (including all overrides)
2. Finalizes config (generates run_name, resolves devices, etc.)
3. Saves finalized config to a temp file
4. Launches worker processes via torchrun

The launcher script itself is NOT re-executed by torchrun - only the worker script is.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import hydra
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import sleap_io as sio
import torch

from sleap_nn.config.utils import get_model_type_from_cfg


def launch_training(
    config_dir: str,
    config_name: str,
    overrides: List[str] = None,
    video_paths: Tuple[str, ...] = None,
    video_path_map: Optional[Dict[str, str]] = None,
    prefix_map: Optional[Dict[str, str]] = None,
):
    """Launch distributed training with torchrun.

    Args:
        config_dir: Absolute path to config directory.
        config_name: Config file name.
        overrides: List of Hydra-style overrides (e.g., ["trainer_config.max_epochs=100"]).
        video_paths: Video paths for replacement.
        video_path_map: Dict mapping old video paths to new paths.
        prefix_map: Dict mapping old prefixes to new prefixes.
    """
    logger.info("=" * 60)
    logger.info("LAUNCHER: Initializing distributed training")
    logger.info("=" * 60)

    # 1. Load config with Hydra (applies all overrides)
    logger.info(f"Loading config: {config_dir}/{config_name}")
    if overrides:
        logger.info(f"Overrides: {overrides}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    # 2. Handle video path replacements (same as train command)
    video_replacement_config = None
    has_video_paths = video_paths is not None and len(video_paths) > 0
    has_video_path_map = video_path_map is not None
    has_prefix_map = prefix_map is not None

    if has_video_paths or has_video_path_map or has_prefix_map:
        video_replacement_config = {
            "video_paths": list(video_paths) if has_video_paths else None,
            "video_path_map": dict(video_path_map) if has_video_path_map else None,
            "prefix_map": dict(prefix_map) if has_prefix_map else None,
        }

    # 3. Finalize config
    cfg = finalize_config(cfg)

    # 4. Resolve number of devices
    num_devices = resolve_num_devices(cfg)
    logger.info(f"Launching training on {num_devices} device(s)")

    # 5. Save finalized config to temp file
    temp_dir = tempfile.mkdtemp(prefix="sleap_nn_launch_")
    temp_config_path = Path(temp_dir) / "training_config.yaml"
    OmegaConf.save(cfg, temp_config_path)
    logger.info(f"Saved finalized config to: {temp_config_path}")

    # Save video replacement config if needed
    temp_video_config_path = None
    if video_replacement_config:
        temp_video_config_path = Path(temp_dir) / "video_replacement.yaml"
        OmegaConf.save(OmegaConf.create(video_replacement_config), temp_video_config_path)
        logger.info(f"Saved video replacement config to: {temp_video_config_path}")

    # 6. Build torchrun command
    cmd = build_torchrun_command(
        num_devices=num_devices,
        config_path=temp_config_path,
        video_config_path=temp_video_config_path,
    )

    logger.info(f"Launching: {' '.join(cmd)}")
    logger.info("=" * 60)

    # 7. Execute torchrun
    process = None
    try:
        # Use Popen for better signal handling
        process = subprocess.Popen(cmd)
        result = process.wait()
        if result != 0:
            logger.error(f"Training failed with exit code {result}")
            sys.exit(result)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user, terminating workers...")
        if process is not None:
            process.terminate()
            try:
                # Give workers a moment to clean up
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Workers did not terminate gracefully, killing...")
                process.kill()
                process.wait()
        sys.exit(1)
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files")

    logger.info("Training completed successfully!")


def finalize_config(cfg: DictConfig) -> DictConfig:
    """Finalize configuration before launching workers.

    This runs ONCE in the launcher, not in workers.

    Handles:
    - Generating run_name timestamp if not provided
    - Resolving ckpt_dir defaults
    - Any other pre-training setup
    """
    # Resolve ckpt_dir first
    ckpt_dir = OmegaConf.select(cfg, "trainer_config.ckpt_dir", default=None)
    if ckpt_dir is None or ckpt_dir == "" or ckpt_dir == "None":
        cfg.trainer_config.ckpt_dir = "."

    # Generate run_name if not provided
    run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
    if run_name is None or run_name == "" or run_name == "None":
        # Get model type from config (first head with non-None value)
        model_type = get_model_type_from_cfg(cfg)

        # Count frames from labels
        train_paths = cfg.data_config.train_labels_path
        val_paths = OmegaConf.select(cfg, "data_config.val_labels_path", default=None)

        train_count = sum(len(sio.load_slp(p)) for p in train_paths)
        val_count = 0
        if val_paths:
            val_count = sum(len(sio.load_slp(p)) for p in val_paths)

        # Generate full run_name
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_name = f"{timestamp}.{model_type}.n={train_count + val_count}"
        cfg.trainer_config.run_name = run_name

        logger.info(f"Generated run_name: {run_name}")
    else:
        logger.info(f"Using provided run_name: {run_name}")

    return cfg


def resolve_num_devices(cfg: DictConfig) -> int:
    """Resolve the number of devices to use.

    Handles:
    - Explicit device count (trainer_devices=4)
    - Device indices (trainer_device_indices=[0,1,2,3])
    - Auto detection (trainer_devices="auto" or None)
    """
    # Check for explicit device indices first
    device_indices = OmegaConf.select(
        cfg, "trainer_config.trainer_device_indices", default=None
    )
    if device_indices is not None and len(device_indices) > 0:
        return len(device_indices)

    # Check for explicit device count
    devices = OmegaConf.select(cfg, "trainer_config.trainer_devices", default="auto")

    if isinstance(devices, int):
        return devices

    # Auto-detect based on accelerator
    if devices in ("auto", None, "None"):
        accelerator = OmegaConf.select(
            cfg, "trainer_config.trainer_accelerator", default="auto"
        )

        if accelerator == "cpu":
            return 1
        elif torch.cuda.is_available():
            count = torch.cuda.device_count()
            logger.info(f"Auto-detected {count} CUDA device(s)")
            return count
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto-detected MPS device (single device)")
            return 1
        else:
            logger.info("No GPU detected, using CPU")
            return 1

    return 1


def build_torchrun_command(
    num_devices: int,
    config_path: Path,
    video_config_path: Optional[Path] = None,
) -> List[str]:
    """Build the torchrun command to launch workers."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(num_devices),
        "-m",
        "sleap_nn.train_worker",
        "--config",
        str(config_path),
    ]

    if video_config_path:
        cmd.extend(["--video-config", str(video_config_path)])

    return cmd
