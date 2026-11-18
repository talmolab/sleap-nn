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

import inspect
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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


def _video_has_embedded_images(video) -> bool:
    """Best-effort: does this loaded video carry embedded image frames?

    A video loaded from a ``.pkg.slp`` preserves its original media as
    ``source_video`` provenance — the same signal sleap-io's
    ``restore_original_videos`` keys off. Falls back to the backend's
    ``has_embedded_images`` flag for embedded videos saved without source
    provenance; backend access is guarded defensively (an unopened video
    currently just returns ``backend=None``, but a future/custom ``Video`` could
    make ``.backend`` a lazy property) so detection never raises.
    """
    if getattr(video, "source_video", None) is not None:
        return True
    try:
        backend = video.backend
    except Exception:
        return False
    return bool(getattr(backend, "has_embedded_images", False))


def _refuse_overwriting_frame_source(labels, output_path) -> None:
    """Refuse to write ``output_path`` over a file a video reads its frames from.

    ``-o`` naming the input ``.pkg.slp`` destroyed its stored frames: the output
    either references itself as its own video source (``embed=False`` ->
    ``FileNotFoundError ... (dataset: video0/video)`` on every reload) or, when
    embedding, is truncated while its frames are still being read from it. Both
    are unrecoverable, so fail before anything is written.

    Raises:
        ValueError: If ``output_path`` is the same file as any video's source.
    """
    import os
    from urllib.parse import urlparse

    out = str(output_path)
    for i, video in enumerate(getattr(labels, "videos", None) or []):
        names = video.filename
        for name in names if isinstance(names, (list, tuple)) else [names]:
            # Remote sources cannot be the local output file.
            if not name or len(urlparse(str(name)).scheme) > 1:
                continue
            try:
                same = (
                    os.path.samefile(name, out)
                    if os.path.exists(name) and os.path.exists(out)
                    else Path(name).resolve() == Path(out).resolve()
                )
            except (OSError, ValueError):
                same = False
            if same:
                raise ValueError(
                    f"Refusing to write {out}: video {i} reads its frames from that "
                    f"file ({name}), and overwriting it would destroy them. Write to "
                    "a different output path."
                )


def _source_videos(source) -> list:
    """The videos ``predict(source)`` reads frames from, resolved without decoding.

    Lets the frame-source overwrite check run BEFORE inference: a ``.slp`` path is
    read for its video records only (``open_backend=False``, no frames or
    annotations), and an in-memory ``sio.Labels`` / ``sio.Video`` / provider
    exposes its videos directly. Best effort -- anything unrecognized (arrays,
    remote URLs, an unreadable file) returns ``[]``, and ``save_predictions``
    re-checks the predicted labels' videos before writing anyway.
    """
    from types import SimpleNamespace
    from urllib.parse import urlparse

    if isinstance(source, (str, Path)):
        name = str(source)
        if len(urlparse(name).scheme) > 1:  # remote (a Windows drive is 1 char)
            return []
        if Path(name).suffix.lower() != ".slp":
            return [SimpleNamespace(filename=name)]
        try:
            from sleap_io.io.slp import read_videos

            return read_videos(name, open_backend=False)
        except Exception:  # noqa: BLE001 - loading reports a bad file properly
            return []
    if isinstance(source, sio.Video):
        return [source]
    try:
        return list(getattr(source, "videos", None) or [])
    except Exception:  # noqa: BLE001 - best effort, the save-time check remains
        return []


def _refuse_overwriting_source_before_inference(
    source, output_path, output_format: Union[str, Sequence[str]] = "slp"
) -> None:
    """Fail fast if the ``.slp`` output would overwrite the input's frame source.

    The same refusal ``save_predictions`` makes, moved in front of the compute:
    without it, ``-o`` naming the input ``.pkg.slp`` was refused only after the
    whole inference pass had run. A no-op when no ``.slp`` is written.

    Raises:
        ValueError: If ``output_path`` is a file a source video reads frames from.
    """
    from types import SimpleNamespace

    if output_path is None or "slp" not in _normalize_output_formats(output_format):
        return
    _refuse_overwriting_frame_source(
        SimpleNamespace(videos=_source_videos(source)), output_path
    )


def _resolve_embed(embed, labels) -> bool:
    """Resolve the ``embed`` control (``"auto"``/``"true"``/``"false"`` or bool) to bool.

    ``"auto"`` -> ``True`` iff any video in ``labels`` carries embedded images.
    """
    if isinstance(embed, bool):
        return embed
    value = str(embed).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "auto":
        return any(
            _video_has_embedded_images(v)
            for v in (getattr(labels, "videos", None) or [])
        )
    raise ValueError(f"Invalid embed={embed!r}; expected 'auto', 'true', or 'false'.")


def _normalize_output_formats(
    output_format: Union[str, Sequence[str]],
) -> List[str]:
    """Normalize ``output_format`` (a string or sequence) to a de-duped list.

    Accepts a single format (``"slp"``) or several
    (``["slp", "analysis_h5"]``, as produced by repeating ``--output_format``).
    Returns an order-preserving, de-duplicated list drawn from
    ``{"slp", "analysis_h5"}``; an empty input defaults to ``["slp"]``.

    Raises:
        ValueError: If any requested format is not ``"slp"`` or ``"analysis_h5"``.
    """
    requested = (
        [output_format] if isinstance(output_format, str) else list(output_format)
    )
    formats: List[str] = []
    for fmt in requested:
        fmt = str(fmt).lower()
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        formats = ["slp"]
    valid = ("slp", "analysis_h5")
    invalid = [f for f in formats if f not in valid]
    if invalid:
        raise ValueError(
            f"Invalid output_format: {invalid!r}. Each must be one of {valid} "
            "(pass --output_format more than once to write several)."
        )
    return formats


def save_predictions(
    labels: sio.Labels,
    output_path: Union[str, Path],
    output_format: Union[str, Sequence[str]] = "slp",
    video_index: Optional[int] = None,
    embed: Union[str, bool] = "false",
    restore_source_videos: bool = False,
    save_embedding_vectors: Optional[bool] = None,
) -> List[Path]:
    """Save predicted ``Labels`` to disk in the requested format(s).

    Args:
        labels: Predicted ``sio.Labels`` to save.
        output_path: Canonical ``.slp`` output path. Analysis HDF5 paths are
            derived from it.
        output_format: One format or several — ``"slp"`` (the default),
            ``"analysis_h5"``, or a sequence like ``["slp", "analysis_h5"]`` to
            write both. (The CLI exposes this as a repeatable ``--output_format``.)
        video_index: Restrict the analysis HDF5 export to a single video index;
            ``None`` exports every video with predicted frames.
        embed: Image-embedding policy for the ``.slp`` output, one of
            ``"false"`` (the default; never embed, backreference source media —
            today's behavior), ``"true"`` (embed images into a self-contained
            ``.pkg.slp``-style file), or ``"auto"`` (embed iff the input was
            itself an embedded ``.pkg.slp``). A bool passes through unchanged.
            Only applies to ``.slp`` output.
        restore_source_videos: On a non-embedding ``.slp`` save, ``False`` (the
            default) keeps references to the input ``.pkg.slp`` file(s) — the
            pixels are already there, and the pre-embedding source video is
            often unavailable. ``True`` instead restores references to the
            original pre-embedding source video files, when recorded. Maps to
            sleap-io's ``restore_original_videos`` and is ignored when
            embedding.
        save_embedding_vectors: Whether to persist appearance (re-ID) vectors
            attached to the detections. ``None`` (the default) PRESERVES them iff
            any detection carries one — sleap-io's own default is ``False``, which
            silently dropped every vector when retracking an embedded ``.slp``
            (``predict -i embedded.slp -t --features embeddings`` produced an
            output that could not be retracked again). ``True``/``False`` force it.

    Returns:
        The list of analysis HDF5 paths written (empty unless ``"analysis_h5"``
        was requested).

    Raises:
        ValueError: If any requested format is not ``"slp"`` or ``"analysis_h5"``,
            or if the ``.slp`` output would overwrite a video's frame source (e.g.
            ``output_path`` is the input ``.pkg.slp``).
    """
    formats = _normalize_output_formats(output_format)

    if "slp" in formats:
        _refuse_overwriting_frame_source(labels, output_path)
        if save_embedding_vectors is None:
            # Preserve appearance vectors iff the labels carry any: sleap-io
            # defaults this to False, so a tracked/re-saved .slp silently lost the
            # embeddings it was tracked on.
            from sleap_nn.inference.tracking import _labels_have_embeddings

            save_embedding_vectors = _labels_have_embeddings(labels)
        labels.save(
            Path(output_path).as_posix(),
            embed=_resolve_embed(embed, labels),
            restore_original_videos=restore_source_videos,
            save_embedding_vectors=save_embedding_vectors,
        )
        logger.info(f"Predictions output path: {output_path}")
        logger.info(f"Saved file at: {datetime.now()}")

    h5_paths: List[Path] = []
    if "analysis_h5" in formats:
        h5_paths = save_analysis_h5_files(labels, output_path, video_index=video_index)
    return h5_paths


def _attach_box_source(
    source: Any, boxes: str, score_threshold: float = 0.0
) -> Tuple[Any, str]:
    """Resolve ``predict(boxes=...)``: ``(source, "gt" | "given")``.

    For a detection file or boxes ``.slp``, the boxes are attached to the source
    labels as ``PredictedBoundingBox``es (a ``.slp`` path is loaded; a
    ``sio.Labels`` is copied; a ``LabelsProvider``'s labels are used in place),
    and that labels source is returned.
    """
    from sleap_nn.evaluation_coco import (
        attach_coco_detections,
        transfer_predicted_boxes,
    )
    from sleap_nn.inference.providers import LabelsProvider

    if boxes == "gt":
        return source, "gt"
    suffix = Path(str(boxes)).suffix.lower()
    if suffix not in (".json", ".slp"):
        raise ValueError(
            f"boxes must be 'gt', a COCO detection .json or a .slp; got {boxes!r}."
        )
    if isinstance(source, (str, Path)) and str(source).endswith(".slp"):
        source_path = Path(str(source))
        source = sio.load_slp(str(source))
    elif isinstance(source, sio.Labels):
        source_path = None
        source = source.copy()
    elif isinstance(source, LabelsProvider):
        source_path = (
            Path(source.labels) if isinstance(source.labels, (str, Path)) else None
        )
    else:
        raise ValueError(
            "boxes from a file need a .slp / sio.Labels source (the boxes are "
            f"attached to its frames); got {type(source).__name__}."
        )
    labels = source._sio_labels if isinstance(source, LabelsProvider) else source
    if suffix == ".json":
        counts = attach_coco_detections(labels, boxes, score_threshold=score_threshold)
        logger.info(
            f"Attached {counts['n_boxes']} detections from {boxes} to "
            f"{counts['n_frames']} frames ({counts['n_unmatched']} on images not "
            "in the source)."
        )
    elif source_path is None or Path(boxes).resolve() != source_path.resolve():
        counts = transfer_predicted_boxes(
            labels, sio.load_slp(str(boxes)), score_threshold=score_threshold
        )
        logger.info(
            f"Copied {counts['n_boxes']} boxes from {boxes} to {counts['n_frames']} "
            f"frames ({counts['n_new_frames']} added; {counts['n_unmatched']} on "
            "unmatched videos)."
        )
    return source, "given"


def _build_predictor(
    *,
    device,
    batch_size,
    paf_workers,
    precision,
    export_dir,
    filter_config,
    tracker_config,
    model_paths,
    backbone_ckpt_path,
    head_ckpt_path,
    preprocess_config,
    anchor_part,
    peak_threshold,
    max_instances,
    centroid_only,
    emit_centroid,
    max_edge_length_ratio,
    dist_penalty_weight,
    n_points,
    min_instance_peaks,
    min_line_scores,
    fg_threshold,
    min_mask_area,
    center_nms_kernel,
    mask_cleanup,
    mask_cleanup_radius,
    distance_gate_alpha,
    bbox_nms_iou,
    merge_fragments,
    merge_method,
    merge_thresholds,
    merge_w_valley,
    merge_w_offset,
    merge_dilate,
    full_res_masks,
    mask_output,
    polygon_epsilon,
    flip_test,
    frames_on_gpu,
    runtime,
    box_mode,
    box_score_threshold,
    node_filter,
    strip_prefix,
    part_model_paths,
    part_streaming,
    part_crop_rule,
    part_node_filter,
    part_strip_prefix,
    part_peak_threshold,
    part_prefix,
):
    """Build the predictor (and the streaming part stage) the way :func:`predict` does.

    Shared by :func:`predict` and :func:`open_stream`, so both run the same pipeline.
    Returns ``(predictor, part_stream)``; ``part_stream`` is ``None`` unless the part
    stage streams.
    """
    from sleap_nn.inference.predictor import Predictor

    build_kwargs: dict = {
        "device": device,
        "batch_size": batch_size,
        "paf_workers": paf_workers,
    }
    if precision != "32":
        if export_dir:
            raise ValueError(
                "precision is not available for exported (ONNX/TensorRT) models; "
                "export them at the precision you want instead."
            )
        build_kwargs["precision"] = precision
    if filter_config is not None:
        build_kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        build_kwargs["tracker_config"] = tracker_config

    if model_paths:
        # `embedding` (re-ID) models emit appearance vectors, not poses, so this
        # pose-packaging path cannot consume them (a lone embedding model would emit
        # empty Labels, and a composed centroid+embedding model would crash on the
        # skeleton-less centroid packaging). Route the user to the dedicated stream.
        from sleap_nn.config.utils import get_model_type_from_cfg, resolve_model_dir
        from sleap_nn.inference.loaders import _load_training_config

        _model_types = []
        for _mp in model_paths:
            try:
                _cfg, _ = _load_training_config(resolve_model_dir(_mp))
                _model_types.append(get_model_type_from_cfg(config=_cfg))
            except Exception:  # noqa: BLE001 - only used to detect the embedding case
                _model_types.append(None)
        if "embedding" in _model_types:
            raise ValueError(
                "Embedding (re-ID) models emit appearance vectors, not poses, and are "
                "not supported by `predict` (which packages pose Labels). Use the "
                "dedicated embedding path instead:\n"
                "  sleap-nn predict --data_path <detections.slp> --model_paths "
                "<embedding_dir> --save_embeddings slp\n"
                "(a lone embedding model EMBEDS EXISTING detections, so its input is "
                "a .slp; pass a detection model alongside it to run on a video)\n"
                "or, from Python:\n"
                "  from sleap_nn.inference.embedding import predict_embeddings_to_slp\n"
                "  predict_embeddings_to_slp(model_paths=[embedding_dir], "
                "data_path=src, output_path='out.slp')"
            )
        if backbone_ckpt_path is not None:
            build_kwargs["backbone_ckpt_path"] = backbone_ckpt_path
        if head_ckpt_path is not None:
            build_kwargs["head_ckpt_path"] = head_ckpt_path
        if preprocess_config is not None:
            build_kwargs["preprocess_config"] = preprocess_config
        if anchor_part is not None:
            build_kwargs["anchor_part"] = anchor_part
        if peak_threshold is not None:
            build_kwargs["peak_threshold"] = peak_threshold
        if max_instances is not None:
            build_kwargs["max_instances"] = max_instances
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
        build_kwargs["distance_gate_alpha"] = distance_gate_alpha
        build_kwargs["bbox_nms_iou"] = bbox_nms_iou
        build_kwargs["merge_fragments"] = merge_fragments
        build_kwargs["merge_method"] = merge_method
        build_kwargs["merge_thresholds"] = merge_thresholds
        build_kwargs["merge_w_valley"] = merge_w_valley
        build_kwargs["merge_w_offset"] = merge_w_offset
        build_kwargs["merge_dilate"] = merge_dilate
        build_kwargs["full_res_masks"] = full_res_masks
        build_kwargs["mask_output"] = mask_output
        build_kwargs["polygon_epsilon"] = polygon_epsilon
        build_kwargs["flip_test"] = flip_test
        if frames_on_gpu:
            build_kwargs["frames_on_gpu"] = True
        if runtime != "auto":
            # Network-only export directories among model_paths (`--network-only`
            # or box detector exports) run on this runtime.
            build_kwargs["export_runtime"] = runtime
        if box_mode is not None:
            build_kwargs["boxes"] = box_mode
            build_kwargs["box_score_threshold"] = box_score_threshold
        if node_filter is not None:
            build_kwargs["node_filter"] = node_filter
            build_kwargs["strip_prefix"] = strip_prefix
        predictor = Predictor.from_model_paths(model_paths, **build_kwargs)
    else:
        if node_filter is not None:
            raise ValueError(
                "node_filter is not available for exported (ONNX/TensorRT) models; "
                "export the filtered model (`sleap-nn export --nodes/--node_prefix`) "
                "or run the checkpoint directory."
            )
        if flip_test:
            raise ValueError(
                "flip_test is not available for exported (ONNX/TensorRT) models, "
                "whose peak finding is baked into the graph; run the checkpoint "
                "directory instead."
            )
        if box_mode is not None:
            raise ValueError(
                "boxes is not supported for exported models; use the checkpoint."
            )
        if frames_on_gpu:
            raise ValueError(
                "frames_on_gpu is not available for exported model directories "
                "(export_dir); pass network-only exports as model paths instead."
            )
        # Exported ONNX/TRT models bake most post-processing into the graph at
        # export time; only these construction-time knobs still apply.
        build_kwargs["runtime"] = runtime
        build_kwargs["min_instance_peaks"] = min_instance_peaks
        build_kwargs["min_line_scores"] = min_line_scores
        build_kwargs["emit_centroid"] = emit_centroid
        if max_instances is not None:
            build_kwargs["max_instances"] = max_instances
        predictor = Predictor.from_export_dir(export_dir, **build_kwargs)

    part_stream = None
    if part_model_paths and part_streaming:
        from sleap_nn.inference.keypoint_crops import build_streaming_part_stage

        part_stream, merged = build_streaming_part_stage(
            predictor.skeleton,
            part_model_paths,
            part_crop_rule,
            device=device,
            batch_size=batch_size,
            precision=precision,
            node_filter=part_node_filter,
            strip_prefix=part_strip_prefix,
            peak_threshold=part_peak_threshold,
            part_prefix=part_prefix,
            export_runtime=runtime,
        )
        predictor.part_stage = part_stream
        predictor.skeleton = merged
    return predictor, part_stream


def predict(
    source: Any,
    *,
    model_paths: Optional[List[str]] = None,
    export_dir: Optional[str] = None,
    # Construction-time (model/device)
    device: str = "auto",
    batch_size: int = 4,
    precision: str = "32",
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
    distance_gate_alpha: Optional[float] = None,
    merge_fragments: bool = False,
    merge_method: str = "greedy",
    merge_thresholds: tuple = (0.85, 0.6, 0.4),
    merge_w_valley: float = 1.0,
    merge_w_offset: float = 0.25,
    merge_dilate: int = 1,
    full_res_masks: bool = False,
    mask_output: str = "mask",
    polygon_epsilon: float = 0.01,
    # Bounding-box detector knob (construction-time; bbox_* models only)
    bbox_nms_iou: Optional[float] = None,
    # Flip-test (construction-time; centered_instance top-down models only)
    flip_test: bool = False,
    frames_on_gpu: bool = False,
    # Top-down stage 1 from given boxes (construction-time; crop models only)
    boxes: Optional[str] = None,
    box_score_threshold: float = 0.0,
    # Node filter (construction-time; single_instance / centered_instance only)
    node_filter: Optional[Union[str, List[str]]] = None,
    strip_prefix: Optional[Union[bool, str]] = None,
    # Part stage: crops from the predicted keypoints (off unless part_model_paths)
    part_streaming: bool = False,
    part_model_paths: Optional[List[str]] = None,
    part_crop_rule: Optional[Any] = None,
    part_node_filter: Optional[Union[str, List[str]]] = None,
    part_strip_prefix: Optional[Union[bool, str]] = None,
    part_peak_threshold: Optional[float] = None,
    part_prefix: Optional[str] = None,
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
    output_format: Union[str, Sequence[str]] = "slp",
    embed: Union[str, bool] = "false",
    restore_source_videos: bool = False,
    save_embedding_vectors: Optional[bool] = None,
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
        precision: ``"32"`` (strict fp32, the default), ``"bf16"`` or ``"fp16"``:
            run the model forward under ``torch.autocast`` with that dtype
            (CUDA only). Checkpoint models only; exported models keep the
            precision they were exported with.
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
        distance_gate_alpha: Adaptive distance-gate strength; ``None`` (default)
            keeps the byte-for-byte argmin grouping. When set, foreground pixels
            whose offset-predicted center exceeds ``alpha*sqrt(area/pi)`` from
            their assigned center are dropped (bottom-up segmentation only).
        merge_fragments: Enable the RAG fragment-merge that re-fuses
            over-segmented animal halves while keeping touching distinct animals
            apart; ``False`` (default) is byte-for-byte today (bottom-up
            segmentation only).
        merge_method: ``"greedy"`` (default) or ``"multicut"`` agglomeration;
            inert when ``merge_fragments=False`` (bottom-up segmentation only).
        merge_thresholds: Greedy-merge decreasing affinity thresholds (default
            ``(0.85, 0.6, 0.4)``); inert when off (bottom-up segmentation only).
        merge_w_valley: Center-valley merge-term weight (default ``1.0``); inert
            when off (bottom-up segmentation only).
        merge_w_offset: Offset-agreement merge-term weight (default ``0.25``);
            inert when off (bottom-up segmentation only).
        merge_dilate: Merge contact-test dilation iterations (default ``1``);
            inert when off (bottom-up segmentation only).
        full_res_masks: Encode masks at full original resolution instead of the
            output-stride grid (default ``False``: stride encoding is ~stride^2
            smaller and lossless at model resolution; bottom-up segmentation only).
        mask_output: Mask output representation — ``"mask"`` (default),
            ``"polygon"`` (``sio.PredictedROI`` only), or ``"both"`` (bottom-up
            segmentation only).
        polygon_epsilon: Douglas-Peucker tolerance (fraction of perimeter) for
            ``mask_output`` polygon/both (bottom-up segmentation only).
        bbox_nms_iou: Class-wise IoU-NMS threshold for bounding-box detectors
            (``None``: the style default, 0.45 for SSD / off for CenterNet; ``1.0``
            disables it). ``peak_threshold`` is their score threshold and
            ``max_instances`` their per-frame detection cap.
        flip_test: Top-down ``centered_instance`` models only: also run the
            centered-instance model on the mirrored crops and average the
            confmaps (symmetric nodes swapped) before peak finding. Not
            available for exported models (``export_dir``).
        frames_on_gpu: Top-down pipelines only: move each batch of frames to the
            GPU before stage 1, so the stage-1 resize / letterbox and the crops'
            frame-size match run there (off by default). Results are not
            bit-identical to the CPU resize: a uint8 resize on the GPU rounds some
            pixels one grey level differently, which can move a box near the score
            threshold. Without a CUDA model the frames stay on the CPU, with a
            warning. Not available for ``export_dir``.
        boxes: Seed a top-down crop model's stage 1 from boxes instead of a
            centroid model or detector (pass only the crop model). ``"gt"``: the
            GT boxes of the ``.slp`` / ``sio.Labels`` source (score 1). A path to
            a COCO detection results ``.json`` (``[{"image_id", "bbox", "score",
            "category_id"}]``, matched to frames by image id; see
            :func:`~sleap_nn.evaluation_coco.attach_coco_detections`) or to a
            ``.slp`` whose frames carry ``PredictedBoundingBox``es (e.g. a
            ``bbox_*`` detector's output, or the source itself): one crop per box,
            scored by the box. A ``sio.Labels`` source is copied first, so the
            boxes are not attached to the caller's object.
        box_score_threshold: Drop given boxes scoring below this.
        node_filter: ``single_instance`` / ``centered_instance`` checkpoints only:
            predict only these nodes, a list of names or a string prefix. The
            confmap head is sliced to their channels (see
            :mod:`sleap_nn.inference.node_filter`). Not available for exported
            models (export them filtered instead).
        strip_prefix: Rename the kept nodes: ``True`` strips the ``node_filter``
            prefix, a string strips that prefix.
        part_streaming: Run the part stage on each batch right after the pose
            stage, on the frames already in memory, with a running crop floor
            instead of the whole-video floor
            (:class:`~sleap_nn.inference.keypoint_crops.StreamingPartStage`). The
            running floor is the median over the boxes so far *including the rest
            of the current batch* (so results depend on ``batch_size``); the first
            ``session_min_boxes`` boxes of the first video get no floor; and each
            call (each ``frames`` chunk) starts a new floor. Tracking (with
            ``tracker_config``) uses the pose nodes only, as without streaming.
            Needs ``part_model_paths``. ``False`` (default): the part stage runs
            after the whole video.
        part_model_paths: A part model (a centered-instance crop model) to run
            after the pose model, on square crops built from each predicted
            instance's keypoints by ``part_crop_rule``
            (:mod:`sleap_nn.inference.keypoint_crops`); its points are appended
            to the instance they came from. ``None`` (default): no part stage.
        part_crop_rule: A :class:`~sleap_nn.inference.keypoint_crops.KeypointCropRule`,
            a dict of its fields, or a ``.json`` / ``.yaml`` file of them.
        part_node_filter: Node filter of the part model's head.
        part_strip_prefix: ``strip_prefix`` for ``part_node_filter``.
        part_peak_threshold: Peak threshold of the part model (default: its own).
        part_prefix: Prefix for the part nodes in the merged skeleton.
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
        integral_refinement: Override the sub-pixel peak refinement:
            ``"auto"`` (DARK for the top-down keypoint stage, integral regression
            elsewhere; the construction-time default), ``"dark"``,
            ``"integral"``, ``"local"`` or ``"none"``. ``None`` keeps the
            default. Exported models do no sub-pixel refinement.
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
        embed: Image-embedding policy for a ``.slp`` output, one of ``"false"``
            (the default; never embed, backreference source media — today's
            behavior), ``"true"`` (embed images into a self-contained
            ``.pkg.slp``-style file), or ``"auto"`` (embed iff the input was
            itself an embedded ``.pkg.slp``). Only applies to ``.slp`` output.
        restore_source_videos: On a non-embedding ``.slp`` save, ``False`` (the
            default) keeps references to the input ``.pkg.slp`` file(s) — the
            pixels are already there, and the pre-embedding source video is
            often unavailable. ``True`` instead restores references to the
            original pre-embedding source video files, when recorded.
            Ignored when embedding.
        save_embedding_vectors: Whether to persist appearance (re-ID) vectors on
            the saved detections. ``None`` (the default) preserves them iff any
            detection carries one; see :func:`save_predictions`.
        clean_empty_frames: Drop frames with no instances.
        progress_callback: ``(processed_frames, total_frames)`` callback
            invoked after each batch (counts are in frames).
        tracking_progress_callback: ``(processed_frames, total_frames)``
            callback per frame during tracking.

    Returns:
        ``sio.Labels`` with predicted instances.

    Raises:
        ValueError: If neither ``model_paths`` nor ``export_dir`` is given,
            or if both are given, or if the ``.slp`` output would overwrite a file
            the input reads its frames from (e.g. ``output_path`` is the input
            ``.pkg.slp``) -- checked before any model is loaded.
    """
    import torch

    # Before any model load or inference (SAM included): `save_predictions` makes
    # the same check, but only once the whole pass has run.
    _refuse_overwriting_source_before_inference(source, output_path, output_format)

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
        if output_path is not None and any(
            f != "slp" for f in _normalize_output_formats(output_format)
        ):
            raise ValueError(
                f"mask_backend output only supports output_format='slp' (got "
                f"{output_format!r}); the SLEAP Analysis HDF5 format stores "
                "poses/tracks, not segmentation masks."
            )
        # Save handling lives in ``run_sam_segmentation``, which mirrors the
        # regular prediction path: by default it backreferences the source media
        # via provenance and does not re-embed images (small output; see its
        # docs). The ``embed`` / ``restore_source_videos`` controls are forwarded.
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
            clean_empty_frames=clean_empty_frames,
            embed=embed,
            restore_source_videos=restore_source_videos,
        )
        return labels

    if model_paths and export_dir:
        raise ValueError("Provide model_paths or export_dir, not both.")
    if part_model_paths and (export_dir or part_crop_rule is None):
        raise ValueError(
            "part_model_paths needs checkpoint (or network-only exported) "
            "model_paths for the pose stage and a part_crop_rule."
        )
    if part_streaming and not part_model_paths:
        raise ValueError("part_streaming needs part_model_paths.")
    if not model_paths and not export_dir:
        raise ValueError("Either model_paths or export_dir is required.")

    box_mode = None
    if boxes is not None:
        source, box_mode = _attach_box_source(source, boxes, box_score_threshold)

    if tracker_config is not None and emit_centroid != "instance":
        raise ValueError(
            "Tracking is incompatible with emit_centroid="
            f"{emit_centroid!r}: tracking operates on sio.PredictedInstance "
            "objects, but this mode emits sio.PredictedCentroid objects. Use "
            "emit_centroid='instance' (the default) for tracking."
        )

    # Build predictor
    predictor, part_stream = _build_predictor(
        device=device,
        batch_size=batch_size,
        paf_workers=paf_workers,
        precision=precision,
        export_dir=export_dir,
        filter_config=filter_config,
        tracker_config=tracker_config,
        model_paths=model_paths,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        preprocess_config=preprocess_config,
        anchor_part=anchor_part,
        peak_threshold=peak_threshold,
        max_instances=max_instances,
        centroid_only=centroid_only,
        emit_centroid=emit_centroid,
        max_edge_length_ratio=max_edge_length_ratio,
        dist_penalty_weight=dist_penalty_weight,
        n_points=n_points,
        min_instance_peaks=min_instance_peaks,
        min_line_scores=min_line_scores,
        fg_threshold=fg_threshold,
        min_mask_area=min_mask_area,
        center_nms_kernel=center_nms_kernel,
        mask_cleanup=mask_cleanup,
        mask_cleanup_radius=mask_cleanup_radius,
        distance_gate_alpha=distance_gate_alpha,
        bbox_nms_iou=bbox_nms_iou,
        merge_fragments=merge_fragments,
        merge_method=merge_method,
        merge_thresholds=merge_thresholds,
        merge_w_valley=merge_w_valley,
        merge_w_offset=merge_w_offset,
        merge_dilate=merge_dilate,
        full_res_masks=full_res_masks,
        mask_output=mask_output,
        polygon_epsilon=polygon_epsilon,
        flip_test=flip_test,
        frames_on_gpu=frames_on_gpu,
        runtime=runtime,
        box_mode=box_mode,
        box_score_threshold=box_score_threshold,
        node_filter=node_filter,
        strip_prefix=strip_prefix,
        part_model_paths=part_model_paths,
        part_streaming=part_streaming,
        part_crop_rule=part_crop_rule,
        part_node_filter=part_node_filter,
        part_strip_prefix=part_strip_prefix,
        part_peak_threshold=part_peak_threshold,
        part_prefix=part_prefix,
    )

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

    if part_stream is not None:
        # The keys two passes write (run_part_stage), plus streaming=True; the floors
        # are the running floors at the end of each video (no key without a floor).
        labels.provenance["part_stage"] = dict(
            rule=part_stream.rule.to_dict(),
            part_prefix=part_prefix,
            part_provenance=part_stream.part_predictor._build_inference_provenance(
                source=source,
                start_time=None,
                end_time=None,
                n_frames=len(labels.labeled_frames),
                inference_params={
                    "peak_threshold": part_peak_threshold,
                    "batch_size": batch_size,
                    **part_stream.part_predictor._preprocess_provenance_params(),
                },
            ),
            streaming=True,
        )
        if part_stream.rule.session_floor is not None:  # no key without, as two passes
            labels.provenance["part_stage"]["session_floors"] = {
                int(k): round(float(v), 3) for k, v in part_stream.last_floors.items()
            }
    elif part_model_paths:
        from sleap_nn.inference.keypoint_crops import run_part_stage

        labels = run_part_stage(
            labels,
            part_model_paths,
            part_crop_rule,
            device=device,
            batch_size=batch_size,
            precision=precision,
            node_filter=part_node_filter,
            strip_prefix=part_strip_prefix,
            peak_threshold=part_peak_threshold,
            part_prefix=part_prefix,
            export_runtime=runtime,
        )

    if output_path is not None:
        save_predictions(
            labels,
            output_path,
            output_format=output_format,
            embed=embed,
            restore_source_videos=restore_source_videos,
            save_embedding_vectors=save_embedding_vectors,
        )

    return labels


def _predict_defaults() -> Dict[str, Any]:
    """``predict``'s keyword defaults (the one place they are written)."""
    return {
        k: p.default
        for k, p in inspect.signature(predict).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def open_stream(
    model_paths: List[str],
    *,
    device: str = "auto",
    precision: str = "32",
    runtime: str = "auto",
    preprocess_config: Optional[Any] = None,
    anchor_part: Optional[str] = None,
    bbox_nms_iou: Optional[float] = None,
    flip_test: bool = False,
    frames_on_gpu: bool = False,
    node_filter: Optional[Union[str, List[str]]] = None,
    strip_prefix: Optional[Union[bool, str]] = None,
    filter_config: Optional[Any] = None,
    part_model_paths: Optional[List[str]] = None,
    part_crop_rule: Optional[Any] = None,
    part_node_filter: Optional[Union[str, List[str]]] = None,
    part_strip_prefix: Optional[Union[bool, str]] = None,
    part_peak_threshold: Optional[float] = None,
    part_prefix: Optional[str] = None,
    peak_threshold: Optional[float] = None,
    centroid_threshold: Optional[float] = None,
    keypoint_threshold: Optional[float] = None,
    max_instances: Optional[int] = None,
    integral_refinement: Optional[str] = None,
    integral_patch_size: Optional[int] = None,
    frame_shape: Optional[Tuple[int, ...]] = None,
    channel_order: str = "rgb",
    keep: bool = False,
):
    """Open a frame-in, pose-out stream (:class:`~sleap_nn.inference.stream.PoseStream`).

    Builds the same pipeline as :func:`predict` with the same model arguments
    (checkpoints, or network-only export folders as ``model_paths`` with
    ``runtime``), at batch size 1, with the part stage streaming
    (``part_streaming=True``). Arguments not listed here take :func:`predict`'s
    defaults. Per-frame options (``peak_threshold``, ``centroid_threshold``,
    ``keypoint_threshold``, ``max_instances``, the refinement, ``filter_config``) mean
    what they mean for :func:`predict`. The models must end in keypoints
    (single-instance, bottom-up, or a top-down pair); others raise ``ValueError``.

    Args:
        model_paths: Model directories (checkpoints, or network-only export folders
            run with ``runtime``), as for :func:`predict`.
        device: As for :func:`predict` (``"auto"``: CUDA, then MPS, then CPU).
        precision: As for :func:`predict` (``"32"``, ``"bf16"``, ``"fp16"``).
        runtime: Runtime of network-only export folders, as for :func:`predict`
            (the part model's too).
        preprocess_config: As for :func:`predict`.
        anchor_part: As for :func:`predict`.
        bbox_nms_iou: As for :func:`predict` (box detectors).
        flip_test: As for :func:`predict`.
        frames_on_gpu: As for :func:`predict`: frames go to the GPU before stage 1.
        node_filter: As for :func:`predict`.
        strip_prefix: As for :func:`predict`.
        filter_config: As for :func:`predict` (applied to every frame).
        part_model_paths: The part model, as for :func:`predict`; its stage always
            streams (running crop floor kept by the stream).
        part_crop_rule: As for :func:`predict` (needed with ``part_model_paths``).
        part_node_filter: As for :func:`predict`.
        part_strip_prefix: As for :func:`predict`.
        part_peak_threshold: As for :func:`predict`.
        part_prefix: As for :func:`predict`.
        peak_threshold: As for :func:`predict` (per frame).
        centroid_threshold: As for :func:`predict` (per frame).
        keypoint_threshold: As for :func:`predict` (per frame).
        max_instances: As for :func:`predict`: the cap on instances per frame.
        integral_refinement: As for :func:`predict`.
        integral_patch_size: As for :func:`predict`.
        frame_shape: ``(H, W)`` or ``(H, W, C)`` of the frames to come: when given,
            a warm-up runs before the stream is returned (every network on dummy data). Without
            it there is no warm-up, and the first frames pay the networks'
            first-call costs.
        channel_order: ``"rgb"`` (default, as sleap-io readers give) or ``"bgr"``
            (as OpenCV gives): colour frames in BGR are swapped to RGB.
        keep: Keep every frame's outputs (on the CPU) for :meth:`PoseStream.labels`
            / :meth:`PoseStream.save`; memory grows with every frame pushed.

    Returns:
        A :class:`~sleap_nn.inference.stream.PoseStream`.

    Example::

        from sleap_nn.inference.run import open_stream

        with open_stream(["models/detector", "models/pose"], channel_order="bgr",
                         frame_shape=(1024, 1280, 3)) as stream:
            for frame in camera:                      # uint8 H x W x C frames
                poses = stream.push(frame)            # this frame's instances
                print(len(poses), poses.points.shape, stream.last_latency_ms)
    """
    import torch

    from sleap_nn.inference.stream import PoseStream

    if part_model_paths and part_crop_rule is None:
        raise ValueError("part_model_paths needs a part_crop_rule.")
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    defaults = _predict_defaults()
    build = {
        name: defaults[name]
        for name in inspect.signature(_build_predictor).parameters
        if name in defaults
    }
    build.update(
        device=device,
        batch_size=1,
        paf_workers=0,
        precision=precision,
        export_dir=None,
        filter_config=filter_config,
        tracker_config=None,
        model_paths=list(model_paths),
        preprocess_config=preprocess_config,
        anchor_part=anchor_part,
        peak_threshold=peak_threshold,
        max_instances=max_instances,
        bbox_nms_iou=bbox_nms_iou,
        flip_test=flip_test,
        frames_on_gpu=frames_on_gpu,
        runtime=runtime,
        box_mode=None,
        node_filter=node_filter,
        strip_prefix=strip_prefix,
        part_model_paths=part_model_paths,
        part_streaming=True,
        part_crop_rule=part_crop_rule,
        part_node_filter=part_node_filter,
        part_strip_prefix=part_strip_prefix,
        part_peak_threshold=part_peak_threshold,
        part_prefix=part_prefix,
    )
    predictor, _ = _build_predictor(**build)
    layer = predictor._scoped_postprocess_layer(
        peak_threshold=peak_threshold,
        centroid_threshold=centroid_threshold,
        keypoint_threshold=keypoint_threshold,
        max_instances=max_instances,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
    )
    stream = PoseStream(predictor, layer, channel_order=channel_order, keep=keep)
    if frame_shape is not None:
        stream._warmup(frame_shape)
    return stream


def _is_box_detector_export(export_dir) -> bool:
    """Whether ``export_dir`` holds an exported box detector (decoded on the host)."""
    import json
    from pathlib import Path

    from sleap_nn.inference.exported_network import _NETWORK_ONLY_TYPES

    meta = Path(export_dir) / "export_metadata.json"
    try:
        return json.loads(meta.read_text()).get("model_type") in _NETWORK_ONLY_TYPES
    except (OSError, ValueError):
        return False
