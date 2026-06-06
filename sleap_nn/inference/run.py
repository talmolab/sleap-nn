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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import sleap_io as sio
from loguru import logger

if TYPE_CHECKING:
    from sleap_nn.inference.filters import FilterConfig
    from sleap_nn.inference.tracking import TrackerConfig


def save_analysis_h5_files(
    labels: sio.Labels,
    slp_output_path: Union[str, Path],
    video_index: Optional[int] = None,
) -> List[Path]:
    """Write SLEAP Analysis HDF5 file(s) from a predicted ``Labels`` object.

    Analysis HDF5 files store a single video each, so one file is written per
    video. The video name is embedded in the filename when more than one video
    is exported (mirroring the multi-video ``.slp`` naming). Videos with no
    predicted frames are skipped.

    Args:
        labels: Predicted ``sio.Labels`` to export.
        slp_output_path: Path to the canonical ``.slp`` predictions file. The
            HDF5 path(s) are derived from it by replacing the trailing
            ``.predictions.slp`` (or ``.slp``) suffix with ``.analysis.h5``.
        video_index: If not ``None``, only this video is exported. Otherwise all
            videos with at least one predicted frame are exported.

    Returns:
        List of ``Path``s that were written.
    """
    slp_output_path = Path(slp_output_path)

    # Derive the base name by stripping the predictions/slp suffix.
    name = slp_output_path.name
    for suffix in (".predictions.slp", ".slp"):
        if name.endswith(suffix):
            base_stem = name[: -len(suffix)]
            break
    else:
        base_stem = slp_output_path.stem
    base = slp_output_path.parent / base_stem

    # Count predicted frames per video (using identity to avoid relying on
    # Video equality semantics).
    frames_per_video = [0] * len(labels.videos)
    for lf in labels.labeled_frames:
        for i, video in enumerate(labels.videos):
            if lf.video is video:
                frames_per_video[i] += 1
                break

    # Determine which videos to export.
    if video_index is not None:
        candidate_indices = (
            [video_index] if 0 <= video_index < len(labels.videos) else []
        )
    else:
        candidate_indices = list(range(len(labels.videos)))

    target_indices = [i for i in candidate_indices if frames_per_video[i] > 0]
    skipped_indices = [i for i in candidate_indices if frames_per_video[i] == 0]
    if skipped_indices:
        logger.warning(
            f"Skipping Analysis HDF5 export for {len(skipped_indices)} video(s) "
            f"with no predicted frames: {skipped_indices}."
        )

    # Build the video name embedded in each filename, disambiguating any videos
    # that share a filename stem by appending the video index.
    def _video_name(i):
        filename = labels.videos[i].filename
        return Path(filename).stem if isinstance(filename, str) else f"video_{i}"

    video_names = {i: _video_name(i) for i in target_indices}
    name_counts = {}
    for vname in video_names.values():
        name_counts[vname] = name_counts.get(vname, 0) + 1
    video_names = {
        i: (f"{vname}_{i}" if name_counts[vname] > 1 else vname)
        for i, vname in video_names.items()
    }

    written_paths = []
    embed_video_name = len(target_indices) > 1
    for i in target_indices:
        if embed_video_name:
            h5_path = base.parent / f"{base.name}.{video_names[i]}.analysis.h5"
        else:
            h5_path = base.parent / f"{base.name}.analysis.h5"
        sio.save_analysis_h5(
            labels,
            h5_path.as_posix(),
            video=i,
            labels_path=slp_output_path.as_posix(),
        )
        written_paths.append(h5_path)
        logger.info(f"Analysis HDF5 output path: {h5_path}")
    return written_paths


def save_predictions(
    labels: sio.Labels,
    output_path: Union[str, Path],
    output_format: str = "slp",
    video_index: Optional[int] = None,
) -> List[Path]:
    """Save predicted ``Labels`` to disk in the requested format(s).

    Args:
        labels: Predicted ``sio.Labels`` to save.
        output_path: Canonical ``.slp`` output path. Analysis HDF5 paths are
            derived from it.
        output_format: One of ``"slp"`` (the default), ``"analysis_h5"``, or
            ``"both"``.
        video_index: Restrict the analysis HDF5 export to a single video index;
            ``None`` exports every video with predicted frames.

    Returns:
        The list of analysis HDF5 paths written (empty when
        ``output_format == "slp"``).

    Raises:
        ValueError: If ``output_format`` is not one of the valid options.
    """
    output_format = str(output_format).lower()
    valid_output_formats = ("slp", "analysis_h5", "both")
    if output_format not in valid_output_formats:
        raise ValueError(
            f"Invalid output_format: {output_format!r}. Must be one of "
            f"{valid_output_formats}."
        )

    if output_format in ("slp", "both"):
        labels.save(Path(output_path).as_posix())

    h5_paths: List[Path] = []
    if output_format in ("analysis_h5", "both"):
        h5_paths = save_analysis_h5_files(labels, output_path, video_index=video_index)
    return h5_paths


def predict(
    source: Any,
    *,
    model_paths: Optional[List[str]] = None,
    export_dir: Optional[str] = None,
    # Construction-time (model/device)
    device: str = "auto",
    batch_size: int = 4,
    runtime: str = "auto",
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    preprocess_config: Optional[Any] = None,
    anchor_part: Optional[str] = None,
    paf_workers: int = 0,
    centroid_only: bool = False,
    emit_centroid: str = "instance",
    # Bottom-up PAF grouping knobs (construction-time; plain bottom-up only)
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: float = 0,
    min_line_scores: float = 0.25,
    # Bottom-up segmentation knobs (construction-time; segmentation only)
    fg_threshold: float = 0.5,
    min_mask_area: int = 0,
    center_nms_kernel: int = 3,
    mask_cleanup: bool = False,
    mask_cleanup_radius: int = 0,
    full_res_masks: bool = False,
    mask_output: str = "mask",
    polygon_epsilon: float = 0.01,
    # SAM prompted-mask producer (explicit, no default; PLAN L2). When set, masks
    # are produced from the existing poses in ``source`` (no trained seg model).
    mask_backend: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    sam3_model_id: str = "facebook/sam3",
    sam_prompt_mode: str = "pose",
    sam_anchor_ind: Optional[int] = None,
    sam_disjointify_masks: bool = False,
    overlay_path: Optional[str] = None,
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
    return_pafs: bool = False,
    return_paf_graph: bool = False,
    return_class_maps: bool = False,
    return_class_vectors: bool = False,
    # Filtering
    filter_config: Optional["FilterConfig"] = None,
    # Tracking
    tracker_config: Optional["TrackerConfig"] = None,
    # Output
    output_path: Optional[str] = None,
    output_format: str = "slp",
    clean_empty_frames: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tracking_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> sio.Labels:
    """Build a predictor, run inference, return Labels.

    Exactly one of ``model_paths`` or ``export_dir`` must be provided.

    Args:
        source: Video path, ``sio.Video``, ``sio.Labels``, or a Provider.
        model_paths: Trained model directories — or a path to a model's
            ``best.ckpt`` or ``training_config.{yaml,json}`` file, which resolves
            to its directory (#575). One for single-instance / bottom-up, two for
            top-down.
        export_dir: Path to an exported ONNX/TRT directory (alternative
            to ``model_paths``).
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, etc.
        batch_size: Frames per batch.
        runtime: Runtime for an exported model: ``"auto"`` (prefer TensorRT,
            fall back to ONNX), ``"onnx"``, or ``"tensorrt"``. Ignored when
            ``model_paths`` is given (checkpoints).
        backbone_ckpt_path: Optional backbone weight override.
        head_ckpt_path: Optional head weight override.
        preprocess_config: Optional OmegaConf preprocessing overrides.
        anchor_part: Override centroid anchor node name.
        paf_workers: CPU worker processes for bottom-up PAF grouping.
        centroid_only: Force centroid-only output even when a
            centered-instance model is among ``model_paths``.
        emit_centroid: Centroid-only output representation: ``"instance"``
            (default; single-node ``PredictedInstance``, frontend-compatible),
            ``"centroid"`` (``sio.PredictedCentroid``), or ``"both"``.
        max_edge_length_ratio: Bottom-up PAF max edge length ratio.
        dist_penalty_weight: Bottom-up PAF distance penalty weight.
        n_points: Bottom-up PAF line integration sample count.
        min_instance_peaks: Bottom-up min peaks for a valid instance.
        min_line_scores: Bottom-up per-edge match threshold. (These five
            apply only to plain bottom-up models.)
        fg_threshold: Foreground probability threshold for binarizing the
            segmentation map (bottom-up segmentation only).
        min_mask_area: Minimum predicted-mask area in original-image pixels;
            smaller masks are dropped to suppress over-segmentation. ``0``
            disables it (bottom-up segmentation only).
        center_nms_kernel: Odd window size for center-peak NMS; larger merges
            nearby duplicate centers (bottom-up segmentation only).
        mask_cleanup: Keep-largest-CC + hole-fill per mask (bottom-up
            segmentation only).
        mask_cleanup_radius: Morphological open->close radius (output-stride
            pixels) applied during ``mask_cleanup``; ``0`` keeps keep-largest +
            fill only (bottom-up segmentation only).
        full_res_masks: Encode masks at full original resolution instead of the
            output-stride grid (default ``False``: stride encoding is ~stride^2
            smaller and lossless at model resolution; bottom-up segmentation only).
        mask_output: Mask output representation — ``"mask"`` (default),
            ``"polygon"`` (``sio.PredictedROI`` only), or ``"both"`` (bottom-up
            segmentation only).
        polygon_epsilon: Douglas-Peucker tolerance (fraction of perimeter) for
            ``mask_output`` polygon/both (bottom-up segmentation only).
        mask_backend: **Explicit** SAM mask backend (PLAN L2): ``"sam"`` (SAM1) /
            ``"sam3"`` (PR-B). When set, ``source`` is treated as a pose ``.slp``
            and masks are predicted from its existing instances (no trained seg
            model, so ``model_paths`` / ``export_dir`` are not required). ``None``
            (the default) leaves the model-driven path untouched.
        sam_checkpoint: SAM1 checkpoint path (required for ``mask_backend="sam"``).
        sam_model_type: SAM1 model registry key.
        sam3_model_id: Hugging Face model id for the gated SAM3 path
            (``mask_backend="sam3"``); defaults to ``"facebook/sam3"``.
        sam_prompt_mode: ``"pose"`` / ``"centroid"`` / ``"box"`` (PLAN §2.2).
        sam_anchor_ind: Centroid anchor node index for ``sam_prompt_mode="centroid"``.
        sam_disjointify_masks: Make per-frame masks disjoint when >=2 instances.
        overlay_path: Optional review-overlay PNG path (PLAN L4; SAM path only).
        frames: Frame indices to predict. ``None`` = all.
        peak_threshold: Override peak threshold for all stages.
        centroid_threshold: Override centroid-stage threshold (top-down).
        keypoint_threshold: Override centered-instance threshold (top-down).
        max_instances: Cap on instances per frame.
        integral_refinement: ``"integral"`` or ``"none"``.
        integral_patch_size: Refinement patch size.
        return_confmaps: Keep confidence maps on Outputs.
        return_crops: Keep per-instance crops on Outputs (top-down).
        return_pafs: Keep part-affinity fields on Outputs (bottom-up).
        return_paf_graph: Keep the PAF graph on Outputs (bottom-up).
        return_class_maps: Keep class maps on Outputs (multi-class bottom-up).
        return_class_vectors: Keep class vectors on Outputs (multi-class top-down).
        filter_config: Post-inference :class:`FilterConfig`.
        tracker_config: :class:`TrackerConfig` for tracking.
        output_path: If set, save the Labels to this path.
        output_format: Format to save the Labels in when ``output_path`` is set.
            One of ``"slp"`` (the default), ``"analysis_h5"`` (a SLEAP Analysis
            HDF5 file, one ``.analysis.h5`` per video), or ``"both"``. Analysis
            HDF5 paths are derived from ``output_path``.
        clean_empty_frames: Drop frames with no instances.
        progress_callback: ``(processed_frames, total_frames)`` callback
            invoked after each batch (counts are in frames).
        tracking_progress_callback: ``(processed_frames, total_frames)``
            callback per frame during tracking.

    Returns:
        ``sio.Labels`` with predicted instances.

    Raises:
        ValueError: If neither ``model_paths`` nor ``export_dir`` is given,
            or if both are given.
    """
    import torch

    from sleap_nn.inference.predictor import Predictor

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    # SAM prompted-mask producer (PLAN L2/L8): masks come from the existing poses
    # in ``source`` — there is no trained seg model, so this short-circuits the
    # model-driven path entirely. ``mask_backend`` is explicit / required to opt
    # in; ``None`` (the default) leaves everything below unchanged.
    if mask_backend is not None:
        if model_paths or export_dir:
            raise ValueError(
                "mask_backend produces masks from the poses in `source` and does "
                "not use a trained seg model; do not also pass model_paths / "
                "export_dir."
            )
        from sleap_nn.inference.sam import run_sam_segmentation

        # Segmentation masks only serialize to ``.slp``: the SLEAP Analysis HDF5
        # format stores poses/tracks, not ``PredictedSegmentationMask``, so it
        # would silently drop the masks (the actual output). Reject it up front
        # rather than write a mask-less ``.h5``.
        if output_path is not None and str(output_format).lower() != "slp":
            raise ValueError(
                f"mask_backend output only supports output_format='slp' (got "
                f"{output_format!r}); the SLEAP Analysis HDF5 format stores "
                "poses/tracks, not segmentation masks."
            )
        # Save handling lives in ``run_sam_segmentation``, which mirrors the
        # regular prediction path: it backreferences the source media via
        # provenance and never re-embeds images (small output; see its docs).
        labels = run_sam_segmentation(
            source,
            mask_backend,
            prompt_mode=sam_prompt_mode,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            sam3_model_id=sam3_model_id,
            device=device,
            anchor_ind=sam_anchor_ind,
            disjointify_masks=sam_disjointify_masks,
            output_path=output_path,
            overlay_path=overlay_path,
            frames=frames,
        )
        return labels

    if model_paths and export_dir:
        raise ValueError("Provide model_paths or export_dir, not both.")
    if not model_paths and not export_dir:
        raise ValueError("Either model_paths or export_dir is required.")

    if tracker_config is not None and emit_centroid != "instance":
        raise ValueError(
            "Tracking is incompatible with emit_centroid="
            f"{emit_centroid!r}: tracking operates on sio.PredictedInstance "
            "objects, but this mode emits sio.PredictedCentroid objects. Use "
            "emit_centroid='instance' (the default) for tracking."
        )

    # Build predictor
    build_kwargs: dict = {
        "device": device,
        "batch_size": batch_size,
        "paf_workers": paf_workers,
    }
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
        if centroid_only:
            build_kwargs["centroid_only"] = True
        if emit_centroid != "instance":
            build_kwargs["emit_centroid"] = emit_centroid
        # Bottom-up PAF knobs configure the scorer at load time (#583); inert
        # for non-bottom-up models (load_model_assets forwards them only there).
        build_kwargs["max_edge_length_ratio"] = max_edge_length_ratio
        build_kwargs["dist_penalty_weight"] = dist_penalty_weight
        build_kwargs["n_points"] = n_points
        build_kwargs["min_instance_peaks"] = min_instance_peaks
        build_kwargs["min_line_scores"] = min_line_scores
        # Segmentation knobs configure the SegmentationLayer at load time; inert
        # for non-segmentation models (load_model_assets forwards them only there).
        build_kwargs["fg_threshold"] = fg_threshold
        build_kwargs["min_mask_area"] = min_mask_area
        build_kwargs["center_nms_kernel"] = center_nms_kernel
        build_kwargs["mask_cleanup"] = mask_cleanup
        build_kwargs["mask_cleanup_radius"] = mask_cleanup_radius
        build_kwargs["full_res_masks"] = full_res_masks
        build_kwargs["mask_output"] = mask_output
        build_kwargs["polygon_epsilon"] = polygon_epsilon
        predictor = Predictor.from_model_paths(model_paths, **build_kwargs)
    else:
        # Exported ONNX/TRT models bake most post-processing into the graph at
        # export time; only these construction-time knobs still apply.
        build_kwargs["runtime"] = runtime
        build_kwargs["min_instance_peaks"] = min_instance_peaks
        build_kwargs["min_line_scores"] = min_line_scores
        build_kwargs["emit_centroid"] = emit_centroid
        if max_instances is not None:
            build_kwargs["max_instances"] = max_instances
        predictor = Predictor.from_export_dir(export_dir, **build_kwargs)

    # Run inference with prediction-time overrides
    labels = predictor.predict(
        source,
        frames=frames,
        make_labels=True,
        clean_empty_frames=clean_empty_frames,
        progress_callback=progress_callback,
        tracking_progress_callback=tracking_progress_callback,
        peak_threshold=peak_threshold,
        centroid_threshold=centroid_threshold,
        keypoint_threshold=keypoint_threshold,
        max_instances=max_instances,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        return_crops=return_crops,
        return_pafs=return_pafs,
        return_paf_graph=return_paf_graph,
        return_class_maps=return_class_maps,
        return_class_vectors=return_class_vectors,
    )

    if output_path is not None:
        save_predictions(labels, output_path, output_format=output_format)

    return labels
