"""Provenance metadata utilities for inference outputs.

This module provides utilities for building and managing provenance metadata
that is stored in SLP files produced during inference. Provenance metadata
helps track where predictions came from and how they were generated.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import sleap_io as sio

import sleap_nn
from sleap_nn.system_info import get_system_info_dict


def build_inference_provenance(
    model_paths: Optional[list[str]] = None,
    model_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    input_labels: Optional[sio.Labels] = None,
    input_path: Optional[Union[str, Path]] = None,
    frames_processed: Optional[int] = None,
    frames_total: Optional[int] = None,
    frame_selection_method: Optional[str] = None,
    inference_params: Optional[dict[str, Any]] = None,
    tracking_params: Optional[dict[str, Any]] = None,
    device: Optional[str] = None,
    cli_args: Optional[dict[str, Any]] = None,
    include_system_info: bool = True,
) -> dict[str, Any]:
    """Build provenance metadata dictionary for inference output.

    This function creates a comprehensive provenance dictionary that captures
    all relevant metadata about an inference run, enabling reproducibility
    and tracking of prediction origins.

    Args:
        model_paths: List of paths to model checkpoints used for inference.
        model_type: Type of model used (e.g., "top_down", "bottom_up",
            "single_instance").
        start_time: Datetime when inference started.
        end_time: Datetime when inference finished.
        input_labels: Input Labels object if inference was run on an SLP file.
            The provenance from this object will be preserved.
        input_path: Path to input file (SLP or video).
        frames_processed: Number of frames that were processed.
        frames_total: Total number of frames in the input.
        frame_selection_method: Method used to select frames (e.g., "all",
            "labeled", "suggested", "range").
        inference_params: Dictionary of inference parameters (peak_threshold,
            integral_refinement, batch_size, etc.).
        tracking_params: Dictionary of tracking parameters if tracking was run.
        device: Device used for inference (e.g., "cuda:0", "cpu", "mps").
        cli_args: Command-line arguments if available.
        include_system_info: If True, include detailed system information.
            Set to False for lighter-weight provenance.

    Returns:
        Dictionary containing provenance metadata suitable for storing in
        Labels.provenance.

    Example:
        >>> from datetime import datetime
        >>> provenance = build_inference_provenance(
        ...     model_paths=["/path/to/model.ckpt"],
        ...     model_type="top_down",
        ...     start_time=datetime.now(),
        ...     end_time=datetime.now(),
        ...     device="cuda:0",
        ... )
        >>> labels.provenance = provenance
        >>> labels.save("predictions.slp")
    """
    provenance: dict[str, Any] = {}

    # Timestamps
    if start_time is not None:
        provenance["inference_start_timestamp"] = start_time.isoformat()
    if end_time is not None:
        provenance["inference_end_timestamp"] = end_time.isoformat()
    if start_time is not None and end_time is not None:
        runtime_seconds = (end_time - start_time).total_seconds()
        provenance["inference_runtime_seconds"] = runtime_seconds

    # Version information
    provenance["sleap_nn_version"] = sleap_nn.__version__
    provenance["sleap_io_version"] = sio.__version__

    # Model information
    if model_paths is not None:
        # Store as absolute POSIX paths for cross-platform compatibility
        provenance["model_paths"] = [
            Path(p).resolve().as_posix() if isinstance(p, (str, Path)) else str(p)
            for p in model_paths
        ]
    if model_type is not None:
        provenance["model_type"] = model_type

    # Input data lineage
    if input_path is not None:
        provenance["source_file"] = (
            Path(input_path).resolve().as_posix()
            if isinstance(input_path, (str, Path))
            else str(input_path)
        )

    # Preserve input provenance if available
    if input_labels is not None and hasattr(input_labels, "provenance"):
        input_prov = dict(input_labels.provenance)
        if input_prov:
            provenance["input_provenance"] = input_prov
            # Also set source_labels for compatibility with sleap-io conventions
            if "filename" in input_prov:
                provenance["source_labels"] = input_prov["filename"]

    # Frame selection information
    if frames_processed is not None or frames_total is not None:
        frame_info: dict[str, Any] = {}
        if frame_selection_method is not None:
            frame_info["method"] = frame_selection_method
        if frames_processed is not None:
            frame_info["frames_processed"] = frames_processed
        if frames_total is not None:
            frame_info["frames_total"] = frames_total
        if frame_info:
            provenance["frame_selection"] = frame_info

    # Inference parameters
    if inference_params is not None:
        # Filter out None values and convert paths
        clean_params = {}
        for key, value in inference_params.items():
            if value is not None:
                if isinstance(value, Path):
                    clean_params[key] = value.as_posix()
                else:
                    clean_params[key] = value
        if clean_params:
            provenance["inference_config"] = clean_params

    # Tracking parameters
    if tracking_params is not None:
        clean_tracking = {k: v for k, v in tracking_params.items() if v is not None}
        if clean_tracking:
            provenance["tracking_config"] = clean_tracking

    # Device information
    if device is not None:
        provenance["device"] = device

    # CLI arguments
    if cli_args is not None:
        # Filter out None values
        clean_cli = {k: v for k, v in cli_args.items() if v is not None}
        if clean_cli:
            provenance["cli_args"] = clean_cli

    # System information (can be disabled for lighter provenance)
    if include_system_info:
        try:
            system_info = get_system_info_dict()
            # Extract key fields for provenance (avoid excessive nesting)
            provenance["system_info"] = {
                "python_version": system_info.get("python_version"),
                "platform": system_info.get("platform"),
                "pytorch_version": system_info.get("pytorch_version"),
                "cuda_version": system_info.get("cuda_version"),
                "accelerator": system_info.get("accelerator"),
                "gpu_count": system_info.get("gpu_count"),
            }
            # Include GPU names if available
            if system_info.get("gpus"):
                provenance["system_info"]["gpus"] = [
                    gpu.get("name") for gpu in system_info["gpus"]
                ]
        except Exception:
            # Don't fail inference if system info collection fails
            pass

    return provenance


def build_tracking_only_provenance(
    input_labels: Optional[sio.Labels] = None,
    input_path: Optional[Union[str, Path]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    tracking_params: Optional[dict[str, Any]] = None,
    frames_processed: Optional[int] = None,
    include_system_info: bool = True,
) -> dict[str, Any]:
    """Build provenance metadata for tracking-only pipeline.

    This is a simplified version of build_inference_provenance for when
    only tracking is run without model inference.

    Args:
        input_labels: Input Labels object with existing predictions.
        input_path: Path to input SLP file.
        start_time: Datetime when tracking started.
        end_time: Datetime when tracking finished.
        tracking_params: Dictionary of tracking parameters.
        frames_processed: Number of frames that were tracked.
        include_system_info: If True, include system information.

    Returns:
        Dictionary containing provenance metadata.
    """
    provenance: dict[str, Any] = {}

    # Timestamps
    if start_time is not None:
        provenance["tracking_start_timestamp"] = start_time.isoformat()
    if end_time is not None:
        provenance["tracking_end_timestamp"] = end_time.isoformat()
    if start_time is not None and end_time is not None:
        runtime_seconds = (end_time - start_time).total_seconds()
        provenance["tracking_runtime_seconds"] = runtime_seconds

    # Version information
    provenance["sleap_nn_version"] = sleap_nn.__version__
    provenance["sleap_io_version"] = sio.__version__

    # Note that this is tracking-only
    provenance["pipeline_type"] = "tracking_only"

    # Input data lineage
    if input_path is not None:
        provenance["source_file"] = (
            Path(input_path).resolve().as_posix()
            if isinstance(input_path, (str, Path))
            else str(input_path)
        )

    # Preserve input provenance if available
    if input_labels is not None and hasattr(input_labels, "provenance"):
        input_prov = dict(input_labels.provenance)
        if input_prov:
            provenance["input_provenance"] = input_prov
            if "filename" in input_prov:
                provenance["source_labels"] = input_prov["filename"]

    # Frame information
    if frames_processed is not None:
        provenance["frames_processed"] = frames_processed

    # Tracking parameters
    if tracking_params is not None:
        clean_tracking = {k: v for k, v in tracking_params.items() if v is not None}
        if clean_tracking:
            provenance["tracking_config"] = clean_tracking

    # System information
    if include_system_info:
        try:
            system_info = get_system_info_dict()
            provenance["system_info"] = {
                "python_version": system_info.get("python_version"),
                "platform": system_info.get("platform"),
                "pytorch_version": system_info.get("pytorch_version"),
                "accelerator": system_info.get("accelerator"),
            }
        except Exception:
            pass

    return provenance


def merge_provenance(
    base_provenance: dict[str, Any],
    additional: dict[str, Any],
    overwrite: bool = True,
) -> dict[str, Any]:
    """Merge additional provenance fields into base provenance.

    Args:
        base_provenance: Base provenance dictionary.
        additional: Additional fields to merge.
        overwrite: If True, additional fields overwrite base fields.
            If False, base fields take precedence.

    Returns:
        Merged provenance dictionary.
    """
    result = dict(base_provenance)
    for key, value in additional.items():
        if key not in result or overwrite:
            result[key] = value
    return result
