"""Deprecated Python entry point for running inference.

:func:`run_inference` is retained as a thin backwards-compatibility shim over
the new modular inference pipeline (:mod:`sleap_nn.inference.run` /
:class:`sleap_nn.inference.predictor.Predictor`). The legacy
``sleap_nn.inference.predictors`` ``*Predictor`` classes have been removed; new
code should call the factory directly:

    from sleap_nn.inference import Predictor
    labels = Predictor.from_model_paths([model_dir]).predict(source)

or, for disk-streaming / convenience, :func:`sleap_nn.inference.run.predict`.
"""

from typing import Optional, List, Union

import sleap_io as sio
from pathlib import Path


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


def run_inference(
    data_path: Optional[str] = None,
    input_labels: Optional[sio.Labels] = None,
    input_video: Optional[sio.Video] = None,
    model_paths: Optional[List[str]] = None,
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    max_instances: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    ensure_rgb: Optional[bool] = None,
    input_scale: Optional[float] = None,
    ensure_grayscale: Optional[bool] = None,
    anchor_part: Optional[str] = None,
    only_labeled_frames: bool = False,
    only_suggested_frames: bool = False,
    exclude_user_labeled: bool = False,
    only_predicted_frames: bool = False,
    no_empty_frames: bool = False,
    batch_size: int = 4,
    queue_maxsize: int = 32,
    video_index: Optional[int] = None,
    video_dataset: Optional[str] = None,
    video_input_format: str = "channels_last",
    frames: Optional[list] = None,
    crop_size: Optional[int] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
    filter_overlapping: bool = False,
    filter_overlapping_method: str = "iou",
    filter_overlapping_threshold: float = 0.8,
    filter_min_visible_nodes: int = 0,
    filter_min_visible_node_fraction: float = 0.0,
    filter_min_mean_node_score: float = 0.0,
    filter_min_instance_score: float = 0.0,
    integral_refinement: Optional[str] = "integral",
    integral_patch_size: int = 5,
    return_confmaps: bool = False,
    return_pafs: bool = False,
    return_paf_graph: bool = False,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: Union[int, float] = 0,
    min_line_scores: float = 0.25,
    return_class_maps: bool = False,
    return_class_vectors: bool = False,
    make_labels: bool = True,
    output_path: Optional[str] = None,
    device: str = "auto",
    tracking: bool = False,
    tracking_window_size: int = 5,
    min_new_track_points: int = 0,
    candidates_method: str = "fixed_window",
    min_match_points: int = 0,
    features: str = "keypoints",
    scoring_method: str = "oks",
    scoring_reduction: str = "mean",
    robust_best_instance: float = 1.0,
    oks_stddev: Optional[float] = None,
    track_matching_method: str = "hungarian",
    max_tracks: Optional[int] = None,
    use_flow: bool = False,
    of_img_scale: float = 1.0,
    of_window_size: int = 21,
    of_max_levels: int = 3,
    use_kalman: bool = False,
    kf_track_features: str = "centroid",
    kf_init_frame_count: int = 10,
    kf_node_indices: Optional[List[int]] = None,
    kf_reset_gap_size: int = 5,
    post_connect_single_breaks: bool = False,
    tracking_target_instance_count: Optional[int] = None,
    tracking_pre_cull_to_target: int = 0,
    tracking_pre_cull_iou_threshold: float = 0,
    tracking_clean_instance_count: int = 0,
    tracking_clean_iou_threshold: float = 0,
    gui: bool = False,
):
    """Deprecated entry point to run inference on trained SLEAP-NN models.

    .. deprecated::
        ``run_inference`` is a thin shim over the new pipeline and will be
        removed in a future release. Use
        :meth:`sleap_nn.inference.predictor.Predictor.from_model_paths`
        (``.predict(...)`` / ``.predict_to_file(...)``),
        :meth:`Predictor.retrack` for pure tracking, or the convenience
        :func:`sleap_nn.inference.run.predict`.

    The full keyword surface of the historical ``run_inference`` is preserved
    for backwards compatibility; each argument is translated to the equivalent
    new-pipeline construct (``preprocess_config`` / ``FilterConfig`` /
    ``TrackerConfig`` / provider).

    Returns:
        ``sio.Labels`` when ``make_labels=True`` (the default); otherwise the
        list of raw per-batch ``Outputs``.
    """
    import warnings

    warnings.warn(
        "sleap_nn.predict.run_inference() is deprecated and will be removed in a "
        "future release. Use the new pipeline: "
        "sleap_nn.inference.Predictor.from_model_paths([model_dir]).predict(...), "
        "Predictor.retrack(labels, tracker_config) for pure tracking, or "
        "sleap_nn.inference.run.predict(...).",
        DeprecationWarning,
        stacklevel=2,
    )

    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider, VideoProvider
    from sleap_nn.inference.run import predict as _predict
    from sleap_nn.cli import (
        _build_preprocess_config,
        _build_filter_config,
        _build_tracker_config,
        _resolve_device,
        _scope_labels_to_video,
    )

    # Lone-centroid guard: centroid-only inference is a new-pipeline feature; the
    # deprecated run_inference()/track entries redirect such users to `infer`.
    if model_paths:
        from sleap_nn.config.utils import get_model_type_from_cfg, resolve_model_dir
        from sleap_nn.inference.loaders import _load_training_config

        detected_types = []
        try:
            for mp in model_paths:
                cfg, _ = _load_training_config(resolve_model_dir(str(mp)))
                detected_types.append(get_model_type_from_cfg(cfg))
        except Exception:  # noqa: BLE001 — fall through to the normal flow on error
            detected_types = []
        if detected_types and all(t == "centroid" for t in detected_types):
            raise ValueError(
                "Centroid-only inference is not supported via the deprecated "
                "run_inference()/track pipeline. Use the `infer` command (or "
                "Predictor.from_model_paths([...]).predict(...)) for centroid-only "
                "models."
            )

    # Validate mutually exclusive frame filter flags (legacy behavior).
    if only_labeled_frames and exclude_user_labeled:
        raise ValueError(
            "only_labeled_frames and exclude_user_labeled are mutually exclusive "
            "(would result in zero frames)"
        )
    if isinstance(frames, str):
        frames = frame_list(frames)

    # The new pipeline takes a single scalar peak threshold (with separate
    # centroid/keypoint overrides); collapse a legacy per-stage list to its
    # first element.
    pk = (
        float(peak_threshold[0])
        if isinstance(peak_threshold, (list, tuple))
        else float(peak_threshold)
    )

    helper_kwargs = dict(
        ensure_rgb=ensure_rgb,
        ensure_grayscale=ensure_grayscale,
        max_height=max_height,
        max_width=max_width,
        input_scale=input_scale,
        crop_size=crop_size,
        filter_overlapping=filter_overlapping,
        filter_overlapping_method=filter_overlapping_method,
        filter_overlapping_threshold=filter_overlapping_threshold,
        filter_min_visible_nodes=filter_min_visible_nodes,
        filter_min_visible_node_fraction=filter_min_visible_node_fraction,
        filter_min_mean_node_score=filter_min_mean_node_score,
        filter_min_instance_score=filter_min_instance_score,
        max_instances=max_instances,
        tracking_window_size=tracking_window_size,
        min_new_track_points=min_new_track_points,
        candidates_method=candidates_method,
        min_match_points=min_match_points,
        features=features,
        scoring_method=scoring_method,
        scoring_reduction=scoring_reduction,
        robust_best_instance=robust_best_instance,
        oks_stddev=oks_stddev,
        track_matching_method=track_matching_method,
        max_tracks=max_tracks,
        use_flow=use_flow,
        of_img_scale=of_img_scale,
        of_window_size=of_window_size,
        of_max_levels=of_max_levels,
        use_kalman=use_kalman,
        kf_track_features=kf_track_features,
        kf_init_frame_count=kf_init_frame_count,
        kf_node_indices=kf_node_indices,
        kf_reset_gap_size=kf_reset_gap_size,
        post_connect_single_breaks=post_connect_single_breaks,
        tracking_target_instance_count=tracking_target_instance_count,
        tracking_pre_cull_to_target=tracking_pre_cull_to_target,
        tracking_pre_cull_iou_threshold=tracking_pre_cull_iou_threshold,
        tracking_clean_instance_count=tracking_clean_instance_count,
        tracking_clean_iou_threshold=tracking_clean_iou_threshold,
    )
    preprocess_config = _build_preprocess_config(helper_kwargs)
    filter_config = _build_filter_config(helper_kwargs)
    tracker_config = _build_tracker_config(helper_kwargs) if tracking else None

    has_slp_filters = bool(
        only_labeled_frames
        or only_suggested_frames
        or exclude_user_labeled
        or only_predicted_frames
    )

    def _labels_provider(labels_like):
        return LabelsProvider(
            labels=labels_like,
            batch_size=batch_size,
            only_labeled_frames=bool(only_labeled_frames),
            only_suggested_frames=bool(only_suggested_frames),
            exclude_user_labeled=bool(exclude_user_labeled),
            only_predicted_frames=bool(only_predicted_frames),
        )

    # ── Pure-tracking retrack: no models + tracking on an existing Labels ──
    if not model_paths:
        if not tracking:
            raise ValueError(
                "Neither tracker nor path to trained models specified. Use "
                "`model_paths` to specify models, or set `tracking=True` to "
                "retrack existing predictions."
            )
        if input_labels is not None:
            labels_in = input_labels
        elif data_path is not None and str(data_path).endswith(".slp"):
            labels_in = sio.load_slp(str(data_path))
        else:
            raise ValueError(
                "Track-only pipeline requires a .slp file (data_path) or "
                "input_labels."
            )
        # Honor a `frames` filter on retrack (legacy scoped the input).
        if frames is not None:
            fset = set(int(f) for f in frames)
            kept = [lf for lf in labels_in if int(lf.frame_idx) in fset]
            labels_in = sio.Labels(
                videos=list(labels_in.videos),
                skeletons=list(labels_in.skeletons),
                labeled_frames=kept,
            )
        out = Predictor.retrack(
            labels_in, tracker_config, clean_empty_frames=no_empty_frames
        )
        if output_path is not None:
            out.save(Path(output_path).as_posix())
        return out

    # ── Resolve the inference source (provider when filtering/scoping) ──
    # When video_index scopes a single video, pass the scoped ``sio.Labels``
    # object (not a pre-built provider) so the new pipeline re-attaches the real
    # source video to the output — legacy parity, and required for embedded
    # ``.pkg.slp`` save. ``scoped_video_name`` drives the auto output filename.
    scoped_video_name = None
    if input_labels is not None:
        if video_index is not None:
            source, _tv = _scope_labels_to_video(
                input_labels, video_index, frames=frames
            )
            _vfn = getattr(_tv, "filename", None)
            scoped_video_name = Path(str(_vfn)).stem if _vfn else f"video_{video_index}"
        else:
            source = _labels_provider(input_labels) if has_slp_filters else input_labels
    elif input_video is not None:
        source = input_video
    else:
        src = str(data_path)
        if src.endswith(".slp") and video_index is not None:
            source, _tv = _scope_labels_to_video(
                sio.load_slp(src), video_index, frames=frames
            )
            _vfn = getattr(_tv, "filename", None)
            scoped_video_name = Path(str(_vfn)).stem if _vfn else f"video_{video_index}"
        elif src.endswith(".slp") and has_slp_filters:
            source = _labels_provider(src)
        elif (not src.endswith(".slp")) and (
            video_dataset or video_input_format != "channels_last"
        ):
            source = VideoProvider(
                video=src,
                batch_size=batch_size,
                frames=frames,
                dataset=video_dataset,
                input_format=video_input_format,
            )
        else:
            source = src

    # Auto-derive a per-video output filename when video_index scopes a video
    # and no explicit output path was given (legacy parity).
    if (
        make_labels
        and output_path is None
        and scoped_video_name is not None
        and data_path is not None
    ):
        output_path = (
            f"{Path(str(data_path)).with_suffix('')}."
            f"{scoped_video_name}.predictions.slp"
        )

    bottomup_knobs = dict(
        max_edge_length_ratio=max_edge_length_ratio,
        dist_penalty_weight=dist_penalty_weight,
        n_points=n_points,
        min_instance_peaks=min_instance_peaks,
        min_line_scores=min_line_scores,
    )

    if make_labels:
        kw = dict(
            model_paths=model_paths,
            device=device,
            batch_size=batch_size,
            peak_threshold=pk,
            max_instances=max_instances,
            anchor_part=anchor_part,
            frames=frames,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_pafs=return_pafs,
            return_paf_graph=return_paf_graph,
            return_class_maps=return_class_maps,
            return_class_vectors=return_class_vectors,
            clean_empty_frames=no_empty_frames,
            output_path=output_path,
            **bottomup_knobs,
        )
        if preprocess_config is not None:
            kw["preprocess_config"] = preprocess_config
        if filter_config is not None:
            kw["filter_config"] = filter_config
        if tracker_config is not None:
            kw["tracker_config"] = tracker_config
        if backbone_ckpt_path is not None:
            kw["backbone_ckpt_path"] = backbone_ckpt_path
        if head_ckpt_path is not None:
            kw["head_ckpt_path"] = head_ckpt_path
        return _predict(source, **kw)

    # make_labels=False: return raw per-batch Outputs (no tracking/output file).
    build_kwargs = dict(device=_resolve_device(device), batch_size=batch_size)
    if preprocess_config is not None:
        build_kwargs["preprocess_config"] = preprocess_config
    if filter_config is not None:
        build_kwargs["filter_config"] = filter_config
    if anchor_part is not None:
        build_kwargs["anchor_part"] = anchor_part
    if backbone_ckpt_path is not None:
        build_kwargs["backbone_ckpt_path"] = backbone_ckpt_path
    if head_ckpt_path is not None:
        build_kwargs["head_ckpt_path"] = head_ckpt_path
    build_kwargs.update(bottomup_knobs)
    predictor = Predictor.from_model_paths(model_paths, **build_kwargs)
    return predictor.predict(
        source,
        make_labels=False,
        peak_threshold=pk,
        max_instances=max_instances,
        frames=frames,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        return_confmaps=return_confmaps,
        return_pafs=return_pafs,
        return_paf_graph=return_paf_graph,
        return_class_maps=return_class_maps,
        return_class_vectors=return_class_vectors,
    )
