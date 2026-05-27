"""Top-level ``predict`` — one-call inference from model paths to Labels.

This is the "I just want predictions" entry point. It builds a
:class:`Predictor`, runs inference, and returns ``sio.Labels``. For
more control (streaming, raw ``Outputs``, custom filtering), use
:class:`Predictor` directly.

Usage::

    from sleap_nn.inference import predict

    # Simplest call — returns sio.Labels
    labels = predict("video.mp4", model_paths=["/path/to/model"])

    # With prediction-time overrides
    labels = predict(
        "video.mp4",
        model_paths=["/path/to/centroid", "/path/to/centered_instance"],
        peak_threshold=0.3,
        centroid_threshold=0.5,
        keypoint_threshold=0.1,
    )

    # Save to disk
    labels = predict("video.mp4", model_paths=[...], output_path="preds.slp")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import sleap_io as sio


def predict(
    source: Any,
    *,
    model_paths: Optional[List[str]] = None,
    export_dir: Optional[str] = None,
    # Construction-time (model/device)
    device: str = "auto",
    batch_size: int = 4,
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    preprocess_config: Optional[Any] = None,
    anchor_part: Optional[str] = None,
    # Prediction-time (can vary per call)
    frames: Optional[List[int]] = None,
    peak_threshold: Optional[float] = None,
    centroid_threshold: Optional[float] = None,
    keypoint_threshold: Optional[float] = None,
    max_instances: Optional[int] = None,
    integral_refinement: Optional[str] = None,
    integral_patch_size: Optional[int] = None,
    return_confmaps: bool = False,
    return_crops: bool = False,
    # Filtering
    filter_config: Optional[Any] = None,
    # Tracking
    tracker_config: Optional[Any] = None,
    # Output
    output_path: Optional[str] = None,
    clean_empty_frames: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> sio.Labels:
    """Build a predictor, run inference, return Labels.

    Exactly one of ``model_paths`` or ``export_dir`` must be provided.

    Args:
        source: Video path, ``sio.Video``, ``sio.Labels``, or a Provider.
        model_paths: Checkpoint directories (one for single-instance /
            bottom-up, two for top-down).
        export_dir: Path to an exported ONNX/TRT directory (alternative
            to ``model_paths``).
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, etc.
        batch_size: Frames per batch.
        backbone_ckpt_path: Optional backbone weight override.
        head_ckpt_path: Optional head weight override.
        preprocess_config: Optional OmegaConf preprocessing overrides.
        anchor_part: Override centroid anchor node name.
        frames: Frame indices to predict. ``None`` = all.
        peak_threshold: Override peak threshold for all stages.
        centroid_threshold: Override centroid-stage threshold (top-down).
        keypoint_threshold: Override centered-instance threshold (top-down).
        max_instances: Cap on instances per frame.
        integral_refinement: ``"integral"`` or ``"none"``.
        integral_patch_size: Refinement patch size.
        return_confmaps: Keep confidence maps on Outputs.
        return_crops: Keep per-instance crops on Outputs (top-down).
        filter_config: Post-inference :class:`FilterConfig`.
        tracker_config: :class:`TrackerConfig` for tracking.
        output_path: If set, save the Labels to this ``.slp`` path.
        clean_empty_frames: Drop frames with no instances.
        progress_callback: ``(processed, total)`` callback per batch.

    Returns:
        ``sio.Labels`` with predicted instances.

    Raises:
        ValueError: If neither ``model_paths`` nor ``export_dir`` is given,
            or if both are given.
    """
    import torch

    from sleap_nn.inference.predictor import Predictor

    if model_paths and export_dir:
        raise ValueError("Provide model_paths or export_dir, not both.")
    if not model_paths and not export_dir:
        raise ValueError("Either model_paths or export_dir is required.")

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    # Build predictor
    build_kwargs: dict = {"device": device, "batch_size": batch_size}
    if filter_config is not None:
        build_kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        build_kwargs["tracker_config"] = tracker_config

    if model_paths:
        if backbone_ckpt_path is not None:
            build_kwargs["backbone_ckpt_path"] = backbone_ckpt_path
        if head_ckpt_path is not None:
            build_kwargs["head_ckpt_path"] = head_ckpt_path
        if preprocess_config is not None:
            build_kwargs["preprocess_config"] = preprocess_config
        if anchor_part is not None:
            build_kwargs["anchor_part"] = anchor_part
        predictor = Predictor.from_model_paths(model_paths, **build_kwargs)
    else:
        predictor = Predictor.from_export_dir(export_dir, **build_kwargs)

    # Run inference with prediction-time overrides
    labels = predictor.predict(
        source,
        frames=frames,
        make_labels=True,
        clean_empty_frames=clean_empty_frames,
        progress_callback=progress_callback,
        peak_threshold=peak_threshold,
        centroid_threshold=centroid_threshold,
        keypoint_threshold=keypoint_threshold,
        max_instances=max_instances,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        return_crops=return_crops,
    )

    if output_path is not None:
        labels.save(Path(output_path).as_posix())

    return labels
