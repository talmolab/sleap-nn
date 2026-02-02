"""Training worker script - executed by torchrun in each GPU process.

This script is launched by the launcher via torchrun. It:
1. Loads the pre-finalized config (run_name already set or timestamp provided)
2. Handles video path replacements if specified
3. Runs training

Since config is already finalized by the launcher, all workers see identical config.

Usage (via launcher):
    sleap-nn launch config.yaml

Direct usage (for debugging):
    python -m sleap_nn.train_worker --config /path/to/config.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Tuple

from loguru import logger
from omegaconf import OmegaConf
import sleap_io as sio
import torch
import torch.distributed as dist

from sleap_nn.train import run_training


def main():
    """Main entry point for training worker."""
    parser = argparse.ArgumentParser(description="SLEAP-NN Training Worker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to finalized training config YAML",
    )
    parser.add_argument(
        "--video-config",
        type=str,
        default=None,
        help="Path to video replacement config YAML",
    )
    args = parser.parse_args()

    # Get distributed rank info from environment (set by torchrun)
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(f"[Worker {rank}/{world_size}] Starting...")

    # Initialize distributed process group early so barriers work during dataset creation
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        logger.info(f"[Worker {rank}] Initialized distributed process group")

    try:
        logger.info(f"[Worker {rank}] Loading config from: {args.config}")

        # Load pre-finalized config
        cfg = OmegaConf.load(args.config)

        # Handle video replacements if config provided
        train_labels = None
        val_labels = None

        if args.video_config:
            video_cfg = OmegaConf.load(args.video_config)
            train_labels, val_labels = apply_video_replacements(cfg, video_cfg)

        # Log the run_name to verify all workers have the same value
        run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
        if run_name:
            logger.info(f"[Worker {rank}] run_name: {run_name}")

        # Run training
        run_training(config=cfg, train_labels=train_labels, val_labels=val_labels)

    except KeyboardInterrupt:
        logger.info(f"[Worker {rank}] Training interrupted by user")

    finally:
        # Clean up distributed process group to avoid resource leak warning
        if dist.is_initialized():
            try:
                # Use a barrier with timeout to avoid hanging on cleanup
                # If other workers are dead, this will timeout
                dist.destroy_process_group()
                logger.info(f"[Worker {rank}] Destroyed distributed process group")
            except Exception as e:
                logger.warning(f"[Worker {rank}] Failed to destroy process group: {e}")


def apply_video_replacements(
    cfg: OmegaConf, video_cfg: OmegaConf
) -> Tuple[Optional[List[sio.Labels]], Optional[List[sio.Labels]]]:
    """Apply video path replacements from config.

    Args:
        cfg: Main training config.
        video_cfg: Video replacement config with video_paths, video_path_map, or prefix_map.

    Returns:
        Tuple of (train_labels, val_labels) with replacements applied.
    """
    # Load train labels
    train_labels_paths = cfg.data_config.train_labels_path
    if train_labels_paths is None:
        return None, None

    train_labels = [sio.load_slp(path) for path in train_labels_paths]

    # Load val labels if they exist
    val_labels = None
    val_labels_paths = OmegaConf.select(cfg, "data_config.val_labels_path", default=None)
    if val_labels_paths is not None and len(val_labels_paths) > 0:
        val_labels = [sio.load_slp(path) for path in val_labels_paths]

    # Determine replacement method
    video_paths = OmegaConf.select(video_cfg, "video_paths", default=None)
    video_path_map = OmegaConf.select(video_cfg, "video_path_map", default=None)
    prefix_map = OmegaConf.select(video_cfg, "prefix_map", default=None)

    if video_paths:
        replace_kwargs = {"new_filenames": [Path(p).as_posix() for p in video_paths]}
    elif video_path_map:
        replace_kwargs = {"filename_map": dict(video_path_map)}
    elif prefix_map:
        replace_kwargs = {"prefix_map": dict(prefix_map)}
    else:
        # No replacements needed
        return train_labels, val_labels

    # Apply replacements to train labels
    for labels in train_labels:
        labels.replace_filenames(**replace_kwargs)

    # Apply replacements to val labels if they exist
    if val_labels:
        for labels in val_labels:
            labels.replace_filenames(**replace_kwargs)

    return train_labels, val_labels


if __name__ == "__main__":
    main()
