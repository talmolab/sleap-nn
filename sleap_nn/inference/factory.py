"""Build a :class:`Predictor` from model checkpoint paths or export directories.

The factory detects model types from ``training_config.{yaml,json}``, loads
Lightning checkpoints + inference models via :func:`loaders.load_model_assets`,
and wraps them with the appropriate ``InferenceLayer`` subclasses.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from sleap_nn.inference.filters import FilterConfig
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.tracking import TrackerConfig

if TYPE_CHECKING:
    from sleap_nn.export.metadata import ExportMetadata
from sleap_nn.inference.layers.bottomup import BottomUpLayer
from sleap_nn.inference.layers.bottomup_multiclass import BottomUpMultiClassLayer
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.layers.topdown_multiclass import (
    CenteredInstanceMultiClassLayer,
    TopDownMultiClassLayer,
)

# ─────────────────────────────────────────────────────────────────────────
# Layer builders — one per model type, given a LoadedAssets instance
# ─────────────────────────────────────────────────────────────────────────


def _pp_field(assets: Any, name: str, default: Any = None) -> Any:
    """Read a field from the loaded assets' resolved ``preprocess_config``.

    The loader resolves ``None`` inputs from the training config before
    constructing the assets object, so by the time the factory is called
    every field in ``preprocess_config`` is concrete. Supports both
    ``OmegaConf.DictConfig`` and plain dict.
    """
    cfg = getattr(assets, "preprocess_config", None)
    if cfg is None:
        return default
    try:
        val = cfg[name] if not hasattr(cfg, name) else getattr(cfg, name)
    except (KeyError, AttributeError):
        return default
    return val if val is not None else default


def _build_single_instance_layer(predictor: Any, device: str) -> SingleInstanceLayer:
    """Wrap a ``SingleInstanceInferenceModel`` in an ``InferenceLayer``."""
    inf = predictor.inference_model
    return SingleInstanceLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        output_stride=inf.output_stride,
        max_stride=getattr(predictor, "max_stride", 1),
        preprocess_config=PreprocessConfig(
            scale=inf.input_scale,
            max_height=_pp_field(predictor, "max_height"),
            max_width=_pp_field(predictor, "max_width"),
            ensure_rgb=_pp_field(predictor, "ensure_rgb"),
            ensure_grayscale=_pp_field(predictor, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
            return_confmaps=getattr(inf, "return_confmaps", False),
        ),
    )


def _build_bottomup_layer(predictor: Any, device: str) -> BottomUpLayer:
    """Wrap a ``BottomUpInferenceModel`` in an ``InferenceLayer``."""
    inf = predictor.inference_model
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    return BottomUpLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        paf_scorer=inf.paf_scorer,
        cms_output_stride=inf.cms_output_stride,
        pafs_output_stride=inf.pafs_output_stride,
        max_instances=getattr(inf, "max_instances", None),
        max_stride=max_stride,
        max_peaks_per_node=inf.max_peaks_per_node,
        preprocess_config=PreprocessConfig(
            scale=inf.input_scale,
            max_height=_pp_field(predictor, "max_height"),
            max_width=_pp_field(predictor, "max_width"),
            ensure_rgb=_pp_field(predictor, "ensure_rgb"),
            ensure_grayscale=_pp_field(predictor, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
            return_confmaps=getattr(inf, "return_confmaps", False),
            return_pafs=getattr(inf, "return_pafs", False),
            return_paf_graph=getattr(inf, "return_paf_graph", False),
        ),
    )


def _build_bottomup_multiclass_layer(
    predictor: Any, device: str
) -> BottomUpMultiClassLayer:
    """Wrap a ``BottomUpMultiClassInferenceModel`` in an ``InferenceLayer``."""
    inf = predictor.inference_model
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    return BottomUpMultiClassLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        cms_output_stride=inf.cms_output_stride,
        class_maps_output_stride=inf.class_maps_output_stride,
        max_stride=max_stride,
        preprocess_config=PreprocessConfig(
            scale=inf.input_scale,
            max_height=_pp_field(predictor, "max_height"),
            max_width=_pp_field(predictor, "max_width"),
            ensure_rgb=_pp_field(predictor, "ensure_rgb"),
            ensure_grayscale=_pp_field(predictor, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
            return_confmaps=getattr(inf, "return_confmaps", False),
        ),
    )


def _build_centroid_layer(
    centroid_model: Any,
    device: str,
    assets: Optional[Any] = None,
) -> CentroidLayer:
    """Wrap a ``CentroidCrop`` model in a ``CentroidLayer``.

    ``assets``: optional :class:`LoadedAssets` whose resolved
    ``preprocess_config`` carries ``max_height/max_width/ensure_rgb/
    ensure_grayscale``. The centroid layer applies the size-matcher /
    channel-coercion chain matching ``_make_pipeline_inputs``.
    """
    return CentroidLayer(
        backend=TorchBackend(model=centroid_model.torch_model, device=device),
        output_stride=centroid_model.output_stride,
        max_instances=centroid_model.max_instances,
        max_stride=centroid_model.max_stride,
        anchor_ind=centroid_model.anchor_ind,
        use_gt_centroids=False,
        preprocess_config=PreprocessConfig(
            scale=centroid_model.input_scale,
            max_height=_pp_field(assets, "max_height"),
            max_width=_pp_field(assets, "max_width"),
            ensure_rgb=_pp_field(assets, "ensure_rgb"),
            ensure_grayscale=_pp_field(assets, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(
            peak_threshold=centroid_model.peak_threshold,
            refinement=centroid_model.refinement or "none",
            integral_patch_size=centroid_model.integral_patch_size,
            max_instances=centroid_model.max_instances,
        ),
    )


def _build_centered_instance_layer(
    instance_model: Any, device: str
) -> CenteredInstanceLayer:
    """Wrap a ``FindInstancePeaks`` model in a ``CenteredInstanceLayer``.

    No ``max_height/max_width`` forwarded: this layer receives per-instance
    crops (at the model's training ``crop_hw``) in the top-down composition,
    so the size-matcher must be a no-op. Channel-coercion (``ensure_*``) is
    also redundant here because the centroid stage already coerced the
    parent frame's channels.
    """
    return CenteredInstanceLayer(
        backend=TorchBackend(model=instance_model.torch_model, device=device),
        output_stride=instance_model.output_stride,
        max_stride=instance_model.max_stride,
        preprocess_config=PreprocessConfig(scale=instance_model.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=instance_model.peak_threshold,
            refinement=instance_model.refinement or "none",
            integral_patch_size=instance_model.integral_patch_size,
            return_confmaps=getattr(instance_model, "return_confmaps", False),
        ),
    )


def _build_centroid_layer_gt_only(assets: Any, backend: Any) -> CentroidLayer:
    """Build a ``CentroidLayer`` that reads centroids from GT (no model forward).

    Used when ``model_paths`` contains only a centered-instance model.
    The layer reads centroids from the input batch's ``instances`` field
    (populated by :class:`LabelsProvider` from a ``.slp`` source).

    Args:
        assets: :class:`LoadedAssets` whose ``inference_model.centroid_crop``
            provides the ``anchor_ind``.
        backend: :class:`CentroidLayer` validates the backend against the
            :class:`ModelBackend` protocol at construction, but the
            ``use_gt_centroids=True`` branch never invokes it. Reuse the
            centered-instance layer's backend so the device matches and no
            extra GPU memory is allocated.
    """
    centroid_model = assets.inference_model.centroid_crop
    return CentroidLayer(
        backend=backend,
        # Stride / max-instances / max-stride are model-side knobs the GT
        # path bypasses; pass benign defaults.
        output_stride=1,
        max_instances=None,
        max_stride=1,
        anchor_ind=getattr(centroid_model, "anchor_ind", None),
        use_gt_centroids=True,
        # No preprocess on the GT path — centroids come from the batch.
        preprocess_config=PreprocessConfig(scale=1.0),
        postprocess_config=PostprocessConfig(),
    )


def _build_centered_instance_multiclass_layer(
    instance_model: Any, device: str
) -> CenteredInstanceMultiClassLayer:
    """Wrap a ``TopDownMultiClassFindInstancePeaks`` model in a layer.

    See :func:`_build_centered_instance_layer` for why size-matcher
    fields are intentionally omitted.
    """
    return CenteredInstanceMultiClassLayer(
        backend=TorchBackend(model=instance_model.torch_model, device=device),
        output_stride=instance_model.output_stride,
        max_stride=instance_model.max_stride,
        preprocess_config=PreprocessConfig(scale=instance_model.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=instance_model.peak_threshold,
            refinement=instance_model.refinement or "none",
            integral_patch_size=instance_model.integral_patch_size,
            return_confmaps=getattr(instance_model, "return_confmaps", False),
        ),
    )


def _build_topdown_layer(predictor: Any, device: str) -> TopDownLayer:
    """Compose ``CentroidLayer`` + ``CenteredInstanceLayer`` into a ``TopDownLayer``."""
    inf = predictor.inference_model
    centroid_layer = _build_centroid_layer(inf.centroid_crop, device, assets=predictor)
    inst_layer = _build_centered_instance_layer(inf.instance_peaks, device)
    crop_h, crop_w = inf.centroid_crop.crop_hw
    return TopDownLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(crop_h, crop_w),
    )


def _build_topdown_multiclass_layer(
    predictor: Any, device: str
) -> TopDownMultiClassLayer:
    """Compose centroid + multi-class centered-instance into a multiclass topdown."""
    inf = predictor.inference_model
    centroid_layer = _build_centroid_layer(inf.centroid_crop, device, assets=predictor)
    inst_layer = _build_centered_instance_multiclass_layer(inf.instance_peaks, device)
    crop_h, crop_w = inf.centroid_crop.crop_hw
    return TopDownMultiClassLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(crop_h, crop_w),
    )


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


def get_predictor_from_model_paths(
    model_paths: List[str],
    *,
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    peak_threshold: Union[float, List[float]] = 0.2,
    integral_refinement: str = "integral",
    integral_patch_size: int = 5,
    batch_size: int = 4,
    max_instances: Optional[int] = None,
    return_confmaps: bool = False,
    device: str = "cpu",
    preprocess_config: Optional[Any] = None,
    anchor_part: Optional[str] = None,
    filter_config: Optional[FilterConfig] = None,
    paf_workers: int = 0,
    tracker_config: Optional[TrackerConfig] = None,
    centroid_only: bool = False,
):
    """Build a :class:`Predictor` from one or more checkpoint paths.

    Args:
        model_paths: Directories with ``training_config.{yaml,json}`` +
            ``best.ckpt``. For top-down inference, pass two paths
            (centroid + centered-instance) in either order; the factory
            detects each via its ``training_config``.
        backbone_ckpt_path: Override the backbone weights with this
            ``.ckpt`` (legacy ``run_inference`` knob).
        head_ckpt_path: Override the head weights.
        peak_threshold: Min confmap value for valid peaks. ``List[float]``
            for top-down (centroid threshold + centered-instance threshold).
        integral_refinement: ``"integral"`` for sub-pixel refinement,
            ``"none"`` (or ``None``) for grid-aligned peaks.
        integral_patch_size: Refinement patch size.
        batch_size: Default batch size stored on the ``Predictor``. Used
            when ``predict()`` auto-constructs a provider from an
            ``sio.Video`` or ``sio.Labels``.
        max_instances: Cap on instances per frame.
        return_confmaps: Echo confmaps into ``Outputs.pred_confmaps``.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"cuda:N"``.
        preprocess_config: ``OmegaConf`` overrides for the data-config
            ``preprocessing`` block. ``None`` means "use the training
            config as-is".
        anchor_part: Override the centroid anchor part name (top-down
            only).
        filter_config: Optional post-inference :class:`FilterConfig`.
            ``None`` builds one from the legacy ``filter_*`` kwargs of
            ``run_inference`` if any are non-default.
        paf_workers: Number of CPU worker processes for the bottom-up
            PAF grouping stage. Forwarded to :class:`Predictor`.
        tracker_config: Optional :class:`TrackerConfig`. Forwarded to
            :class:`Predictor`; when set, ``predict()`` will track
            instances post-inference.
        centroid_only: When ``True``, force the centroid-only dispatch
            even if a centered-instance model is also among
            ``model_paths``. Use to get centroid-only output from a
            two-model top-down setup without re-exporting. Raises
            ``ValueError`` if no centroid model is present.

    Returns:
        A :class:`sleap_nn.inference.predictor.Predictor` wrapping the
        appropriate layer composition for the given model types.

    Raises:
        ValueError: If ``model_paths`` doesn't contain a recognized
            combination of model types (e.g., two centroid models).
    """
    from sleap_nn.inference.loaders import load_model_assets
    from sleap_nn.inference.predictor import Predictor

    loaded, model_types = load_model_assets(
        model_paths,
        device=device,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        peak_threshold=peak_threshold,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        max_instances=max_instances,
        return_confmaps=return_confmaps,
        preprocess_config=preprocess_config,
        anchor_part=anchor_part,
    )

    if centroid_only:
        if "centroid" not in model_types:
            raise ValueError(
                "centroid_only=True requires a centroid model in model_paths; "
                f"detected types: {model_types}."
            )
        layer = _build_centroid_layer(
            loaded.inference_model.centroid_crop,
            device,
            assets=loaded,
        )
    else:
        layer = _select_layer(loaded, model_types, device)
    skeleton = loaded.skeletons[0] if loaded.skeletons else None
    kwargs: dict = {
        "layer": layer,
        "skeleton": skeleton,
        "batch_size": batch_size,
        "paf_workers": paf_workers,
    }
    if filter_config is not None:
        kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        kwargs["tracker_config"] = tracker_config
    return Predictor(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# get_predictor_from_export_dir — build a Predictor from an exported ONNX/TRT directory
# ─────────────────────────────────────────────────────────────────────────


def get_predictor_from_export_dir(
    export_dir: Union[str, Path],
    *,
    runtime: str = "auto",
    device: str = "auto",
    batch_size: int = 4,
    return_confmaps: bool = False,
    filter_config: Optional[FilterConfig] = None,
    paf_workers: int = 0,
    tracker_config: Optional[TrackerConfig] = None,
    max_instances: Optional[int] = None,
    min_instance_peaks: float = 0,
    min_line_scores: float = 0.25,
):
    """Build a new :class:`Predictor` from an exported model directory.

    The directory is expected to contain ``export_metadata.json`` plus
    one of ``model.onnx`` / ``model.trt``. Pulls model-type, output stride,
    input scale, and peak-threshold from the metadata; constructs the
    appropriate :class:`InferenceLayer` subclass on the
    ``does_baked_postproc=True`` path so peak finding stays inside the
    exported graph.

    Args:
        export_dir: Directory written by ``sleap_nn export`` (or any
            equivalent exporter that emits the same metadata schema).
        runtime: ``"auto"`` (prefer TRT when present, else ONNX),
            ``"onnx"``, or ``"tensorrt"``.
        device: Device string forwarded to the backend.
        batch_size: Default batch size stored on the ``Predictor``.
        return_confmaps: Echo confmaps onto the resulting ``Outputs``
            when the wrapper exports a ``confmaps`` output. Layers gate
            on this flag.
        filter_config: Optional :class:`FilterConfig` (post-inference).
        max_instances: Optional cap on instances per frame. Forwarded
            to bottom-up's CPU grouping stage; ignored for other model
            types.
        min_instance_peaks: Bottom-up only. Drop assembled instances
            with fewer peaks than this.
        min_line_scores: Bottom-up only. Per-edge match threshold
            (between -1 and 1) for the PAF grouping step.
        paf_workers: Forwarded to :class:`Predictor`. Only meaningful
            for bottom-up exports — irrelevant for single-instance.
        tracker_config: Optional :class:`TrackerConfig` (post-inference
            tracker).

    Returns:
        A configured :class:`sleap_nn.inference.predictor.Predictor`.

    Raises:
        FileNotFoundError: ``export_metadata.json`` or the model file
            isn't present at the expected path.
        ValueError: ``runtime`` isn't recognized.

    Notes:
        The skeleton is resolved automatically from the export's
        ``training_config.yaml`` (if present) or from the export
        metadata's ``node_names``.
    """
    from sleap_nn.export.metadata import ExportMetadata
    from sleap_nn.inference.predictor import Predictor

    export_dir = Path(export_dir)

    metadata_path = export_dir / "export_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"export_metadata.json not found at {metadata_path}. "
            f"Pass a directory written by `sleap_nn export`."
        )
    metadata = ExportMetadata.load(metadata_path)

    runtime, model_path = _resolve_export_runtime(export_dir, runtime)
    backend = _build_export_backend(runtime, model_path, device)

    layer = _select_export_layer(
        metadata=metadata,
        backend=backend,
        return_confmaps=return_confmaps,
        max_instances=max_instances,
        min_instance_peaks=min_instance_peaks,
        min_line_scores=min_line_scores,
    )

    skeleton = _skeleton_from_export(export_dir, metadata)
    kwargs: dict = {
        "layer": layer,
        "skeleton": skeleton,
        "batch_size": batch_size,
        "paf_workers": paf_workers,
    }
    if filter_config is not None:
        kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        kwargs["tracker_config"] = tracker_config
    return Predictor(**kwargs)


def _skeleton_from_export(export_dir: Path, metadata: "ExportMetadata") -> Any:
    """Best-effort skeleton from an export directory.

    Tries the embedded training config first (full skeleton with edges);
    falls back to a bare ``sio.Skeleton`` built from ``metadata.node_names``.
    """
    import sleap_io as sio

    training_cfg_path = export_dir / "training_config.yaml"
    if training_cfg_path.exists():
        try:
            from omegaconf import OmegaConf
            from sleap_nn.inference.utils import get_skeleton_from_config

            cfg = OmegaConf.load(str(training_cfg_path))
            skels = get_skeleton_from_config(cfg.data_config.skeletons)
            if skels:
                return skels[0]
        except (KeyError, AttributeError, TypeError, ValueError, FileNotFoundError):
            pass
    if metadata.node_names:
        return sio.Skeleton(nodes=[sio.Node(name=n) for n in metadata.node_names])
    return None


def _resolve_export_runtime(export_dir: Path, runtime: str) -> tuple[str, Path]:
    """Pick the runtime + model file for an export directory.

    Returns ``(runtime, model_path)`` where ``runtime`` is one of
    ``"onnx"`` or ``"tensorrt"``.
    """
    onnx_path = export_dir / "model.onnx"
    trt_path = export_dir / "model.trt"

    if runtime == "auto":
        if trt_path.exists():
            return "tensorrt", trt_path
        if onnx_path.exists():
            return "onnx", onnx_path
        raise FileNotFoundError(
            f"No model file found in {export_dir}. "
            f"Expected model.onnx or model.trt."
        )
    if runtime == "onnx":
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        return "onnx", onnx_path
    if runtime == "tensorrt":
        if not trt_path.exists():
            raise FileNotFoundError(f"TensorRT model not found: {trt_path}")
        return "tensorrt", trt_path
    raise ValueError(
        f"Unknown runtime: {runtime!r}. Expected 'auto', 'onnx', or 'tensorrt'."
    )


def _build_export_backend(runtime: str, model_path: Path, device: str):
    """Construct the right ``ModelBackend`` for an exported model file."""
    if runtime == "onnx":
        from sleap_nn.inference.layers.backends import ONNXBackend

        return ONNXBackend(model_path=str(model_path), device=device)
    if runtime == "tensorrt":
        from sleap_nn.inference.layers.backends import TensorRTBackend

        return TensorRTBackend(engine_path=str(model_path), device=device)
    raise ValueError(f"Unknown runtime: {runtime!r}")


def _select_export_layer(
    metadata: Any,
    backend: Any,
    return_confmaps: bool,
    max_instances: Optional[int] = None,
    min_instance_peaks: float = 0,
    min_line_scores: float = 0.25,
):
    """Dispatch on ``metadata.model_type`` → build the right export adapter.

    Export adapters live in :mod:`sleap_nn.inference.layers.exported` —
    thin translators that consume the wrapper's already-postprocessed
    output and produce a structured :class:`Outputs`. They intentionally
    bypass the standard layer's coord ladder so transforms aren't
    double-applied.

    Supports all model types: ``single_instance``, ``centroid``,
    ``centered_instance``, ``topdown``, ``bottomup``,
    ``multi_class_topdown``, ``multi_class_bottomup``.
    """
    from sleap_nn.inference.layers.exported import (
        ExportedBottomUpLayer,
        ExportedBottomUpMultiClassLayer,
        ExportedCenteredInstanceLayer,
        ExportedCentroidLayer,
        ExportedSingleInstanceLayer,
        ExportedTopDownLayer,
        ExportedTopDownMultiClassLayer,
    )

    model_type = metadata.model_type

    if model_type == "single_instance":
        return ExportedSingleInstanceLayer(
            backend=backend, return_confmaps=return_confmaps
        )
    if model_type == "centered_instance":
        return ExportedCenteredInstanceLayer(
            backend=backend, return_confmaps=return_confmaps
        )
    if model_type == "centroid":
        return ExportedCentroidLayer(backend=backend)
    if model_type == "topdown":
        return ExportedTopDownLayer(backend=backend)
    if model_type == "bottomup":
        if metadata.max_peaks_per_node is None:
            raise ValueError(
                "Bottom-up export metadata is missing `max_peaks_per_node`. "
                "Re-export the model with the latest exporter."
            )
        return ExportedBottomUpLayer(
            backend=backend,
            node_names=list(metadata.node_names),
            edge_inds=[(int(s), int(d)) for s, d in metadata.edge_inds],
            max_peaks_per_node=int(metadata.max_peaks_per_node),
            input_scale=float(metadata.input_scale),
            max_instances=max_instances,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
        )
    if model_type == "multi_class_topdown":
        if metadata.n_classes is None:
            raise ValueError(
                "multi_class_topdown export metadata is missing `n_classes`."
            )
        return ExportedTopDownMultiClassLayer(
            backend=backend,
            n_classes=int(metadata.n_classes),
        )
    if model_type == "multi_class_bottomup":
        if metadata.n_classes is None:
            raise ValueError(
                "multi_class_bottomup export metadata is missing `n_classes`."
            )
        return ExportedBottomUpMultiClassLayer(
            backend=backend,
            n_nodes=int(metadata.n_nodes),
            n_classes=int(metadata.n_classes),
            input_scale=float(metadata.input_scale),
        )

    raise ValueError(f"Unrecognized model_type {model_type!r} in export_metadata.json.")


def _select_layer(assets: Any, model_types: List[str], device: str):
    """Dispatch on detected model types and build the appropriate layer composition."""
    if "single_instance" in model_types:
        return _build_single_instance_layer(assets, device)
    if "bottomup" in model_types:
        return _build_bottomup_layer(assets, device)
    if "multi_class_bottomup" in model_types:
        return _build_bottomup_multiclass_layer(assets, device)
    has_centroid = "centroid" in model_types
    has_centered = "centered_instance" in model_types
    has_multi_centered = "multi_class_topdown" in model_types
    if has_centroid and has_centered:
        return _build_topdown_layer(assets, device)
    if has_centroid and has_multi_centered:
        return _build_topdown_multiclass_layer(assets, device)
    if has_centroid:
        # Centroid-only inference (no stage-2 model). Returns a bare
        # ``CentroidLayer`` so ``Predictor.to_labels`` packages the output
        # with NaN-padded skeleton + centroid at the anchor node slot.
        return _build_centroid_layer(
            assets.inference_model.centroid_crop,
            device,
            assets=assets,
        )
    if has_centered:
        # Standalone centered-instance inference: no centroid model is
        # provided, so the centroid stage reads GT centroids from the
        # batch's ``instances`` field (the ``LabelsProvider`` path).
        #
        # Required input source: a ``.slp`` file with labeled centroids
        # (``LabelsProvider``). Using ``VideoProvider`` will fail at
        # ``CentroidLayer.predict`` with "use_gt_centroids=True requires
        # `instances` to be passed" -- by design.
        inst_layer = _build_centered_instance_layer(
            assets.inference_model.instance_peaks, device
        )
        centroid_layer = _build_centroid_layer_gt_only(assets, inst_layer.backend)
        crop_h, crop_w = assets.inference_model.centroid_crop.crop_hw
        return TopDownLayer(
            centroid_layer=centroid_layer,
            centered_instance_layer=inst_layer,
            crop_size=(crop_h, crop_w),
        )
    raise ValueError(
        f"Unsupported model_paths combination: detected types {model_types}. "
        f"get_predictor_from_model_paths supports: single_instance, "
        f"bottomup, multi_class_bottomup, top-down (centroid + centered_instance), "
        f"top-down multiclass (centroid + multi_class_topdown), centroid-only, "
        f"or centered-instance-only (requires a .slp source for GT centroids)."
    )
