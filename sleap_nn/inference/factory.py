"""Build a new :class:`Predictor` directly from model checkpoint paths.

PR 11 of #508 (#519). The legacy ``sleap_nn.inference.predictors.Predictor``
already knows how to:

* Resolve ``training_config.{yaml,json}`` (incl. SLEAP <=1.4 legacy)
* Reconstruct the right Lightning module per model type with all its
  optimizer / scheduler / hard-mining hyperparams (Lightning's
  ``load_from_checkpoint`` requires those even when only weights matter)
* Apply ``backbone_ckpt_path`` / ``head_ckpt_path`` overrides
* Hydrate the skeleton + place the model on the requested device

That work is non-trivial and a perfect candidate for *reuse*. This
factory delegates it to the legacy predictor, then re-wraps the loaded
torch module(s) and PAF scorer with the new ``InferenceLayer``
subclasses. The result is a brand-new :class:`Predictor` that accepts
the existing ``run_inference`` kwargs without forking the model-loader
logic.

Why not delete the legacy loader entirely? It's tightly coupled to
``LightningModule.load_from_checkpoint`` and a SLEAP <=1.4 legacy
converter — both stable code paths. Eventually (post-#519) the legacy
``inference_model`` and ``make_pipeline`` go away, and the factory
keeps the loader. Until then this stays a thin adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from omegaconf import OmegaConf

from sleap_nn.inference.filters import FilterConfig
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.tracking import TrackerConfig
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
# Layer builders — one per model type, given a loaded legacy inference_model
# ─────────────────────────────────────────────────────────────────────────


def _neutral_preprocess() -> Any:
    """OmegaConf preprocess overrides that mean 'use the training config'."""
    return OmegaConf.create(
        {
            "ensure_rgb": None,
            "ensure_grayscale": None,
            "crop_size": None,
            "max_width": None,
            "max_height": None,
            "scale": None,
        }
    )


def _build_single_instance_layer(predictor: Any, device: str) -> SingleInstanceLayer:
    """Wrap legacy ``SingleInstanceInferenceModel`` with the new layer."""
    inf = predictor.inference_model
    return SingleInstanceLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        output_stride=inf.output_stride,
        preprocess_config=PreprocessConfig(scale=inf.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
            return_confmaps=getattr(inf, "return_confmaps", False),
        ),
    )


def _build_bottomup_layer(predictor: Any, device: str) -> BottomUpLayer:
    """Wrap legacy ``BottomUpInferenceModel`` with the new layer."""
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
        preprocess_config=PreprocessConfig(scale=inf.input_scale),
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
    """Wrap legacy ``BottomUpMultiClassInferenceModel`` with the new layer."""
    inf = predictor.inference_model
    max_stride = predictor.bottomup_config.model_config.backbone_config[
        predictor.backbone_type
    ]["max_stride"]
    return BottomUpMultiClassLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        cms_output_stride=inf.cms_output_stride,
        class_maps_output_stride=inf.class_maps_output_stride,
        max_stride=max_stride,
        preprocess_config=PreprocessConfig(scale=inf.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=inf.peak_threshold,
            refinement=inf.refinement or "none",
            integral_patch_size=inf.integral_patch_size,
            return_confmaps=getattr(inf, "return_confmaps", False),
        ),
    )


def _build_centroid_layer(legacy_centroid: Any, device: str) -> CentroidLayer:
    """Wrap legacy ``CentroidCrop`` with the new ``CentroidLayer``."""
    return CentroidLayer(
        backend=TorchBackend(model=legacy_centroid.torch_model, device=device),
        output_stride=legacy_centroid.output_stride,
        max_instances=legacy_centroid.max_instances,
        max_stride=legacy_centroid.max_stride,
        anchor_ind=legacy_centroid.anchor_ind,
        use_gt_centroids=False,
        preprocess_config=PreprocessConfig(scale=legacy_centroid.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy_centroid.peak_threshold,
            refinement=legacy_centroid.refinement or "none",
            integral_patch_size=legacy_centroid.integral_patch_size,
            max_instances=legacy_centroid.max_instances,
        ),
    )


def _build_centered_instance_layer(
    legacy_inst: Any, device: str
) -> CenteredInstanceLayer:
    """Wrap legacy ``FindInstancePeaks`` with the new layer."""
    return CenteredInstanceLayer(
        backend=TorchBackend(model=legacy_inst.torch_model, device=device),
        output_stride=legacy_inst.output_stride,
        max_stride=legacy_inst.max_stride,
        preprocess_config=PreprocessConfig(scale=legacy_inst.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy_inst.peak_threshold,
            refinement=legacy_inst.refinement or "none",
            integral_patch_size=legacy_inst.integral_patch_size,
            return_confmaps=getattr(legacy_inst, "return_confmaps", False),
        ),
    )


def _build_centered_instance_multiclass_layer(
    legacy_inst: Any, device: str
) -> CenteredInstanceMultiClassLayer:
    """Wrap legacy ``TopDownMultiClassFindInstancePeaks`` with the new layer."""
    return CenteredInstanceMultiClassLayer(
        backend=TorchBackend(model=legacy_inst.torch_model, device=device),
        output_stride=legacy_inst.output_stride,
        max_stride=legacy_inst.max_stride,
        preprocess_config=PreprocessConfig(scale=legacy_inst.input_scale),
        postprocess_config=PostprocessConfig(
            peak_threshold=legacy_inst.peak_threshold,
            refinement=legacy_inst.refinement or "none",
            integral_patch_size=legacy_inst.integral_patch_size,
            return_confmaps=getattr(legacy_inst, "return_confmaps", False),
        ),
    )


def _build_topdown_layer(predictor: Any, device: str) -> TopDownLayer:
    """Compose ``CentroidLayer`` + ``CenteredInstanceLayer`` into a ``TopDownLayer``."""
    inf = predictor.inference_model
    centroid_layer = _build_centroid_layer(inf.centroid_crop, device)
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
    centroid_layer = _build_centroid_layer(inf.centroid_crop, device)
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


def from_model_paths(
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
):
    """Build a new :class:`Predictor` (PR 8) from one or more checkpoint paths.

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
        batch_size: Currently unused — :class:`Provider` controls batch
            size. Kept in the signature for ``run_inference`` compatibility.
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

    Returns:
        A :class:`sleap_nn.inference.predictor.Predictor` wrapping the
        appropriate layer composition for the given model types.

    Raises:
        ValueError: If ``model_paths`` doesn't contain a recognized
            combination of model types (e.g., two centroid models).
    """
    # Local imports avoid circulars (predictor → factory → predictor).
    from sleap_nn.config.utils import get_model_type_from_cfg
    from sleap_nn.inference.predictor import Predictor as NewPredictor
    from sleap_nn.inference.predictors import Predictor as LegacyPredictor

    if preprocess_config is None:
        preprocess_config = _neutral_preprocess()

    legacy_predictor = LegacyPredictor.from_model_paths(
        model_paths=model_paths,
        backbone_ckpt_path=backbone_ckpt_path,
        head_ckpt_path=head_ckpt_path,
        peak_threshold=peak_threshold,
        integral_refinement=integral_refinement,
        integral_patch_size=integral_patch_size,
        batch_size=batch_size,
        max_instances=max_instances,
        return_confmaps=return_confmaps,
        device=device,
        preprocess_config=preprocess_config,
        anchor_part=anchor_part,
    )
    legacy_predictor._initialize_inference_model()

    # Detect model types across the supplied paths.
    model_types: list[str] = []
    for model_path in model_paths:
        path = Path(model_path)
        if (path / "training_config.yaml").exists():
            cfg = OmegaConf.load((path / "training_config.yaml").as_posix())
        elif (path / "training_config.json").exists():
            from sleap_nn.config.training_job_config import TrainingJobConfig

            cfg = TrainingJobConfig.load_sleap_config(
                (path / "training_config.json").as_posix()
            )
        else:  # pragma: no cover — guarded by legacy loader above
            raise ValueError(f"no training_config in {model_path}")
        model_types.append(get_model_type_from_cfg(config=cfg))

    layer = _select_layer(legacy_predictor, model_types, device)
    kwargs: dict = {"layer": layer, "paf_workers": paf_workers}
    if filter_config is not None:
        kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        kwargs["tracker_config"] = tracker_config
    return NewPredictor(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# from_export_dir — build a Predictor from an exported ONNX/TRT directory
# ─────────────────────────────────────────────────────────────────────────


def from_export_dir(
    export_dir: Union[str, Path],
    *,
    runtime: str = "auto",
    device: str = "auto",
    return_confmaps: bool = False,
    filter_config: Optional[FilterConfig] = None,
    paf_workers: int = 0,
    tracker_config: Optional[TrackerConfig] = None,
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
        return_confmaps: Echo confmaps onto the resulting ``Outputs``
            when the wrapper exports a ``confmaps`` output. Layers gate
            on this flag.
        filter_config: Optional :class:`FilterConfig` (post-inference).
        paf_workers: Forwarded to :class:`Predictor`. Only meaningful
            for bottom-up exports — irrelevant for single-instance.
        tracker_config: Optional :class:`TrackerConfig` (post-inference
            tracker).

    Returns:
        A configured :class:`sleap_nn.inference.predictor.Predictor`.

    Raises:
        FileNotFoundError: ``export_metadata.json`` or the model file
            isn't present at the expected path.
        NotImplementedError: ``model_type`` is recognized but its export
            adapter hasn't landed yet. As of PR 18 only
            ``"single_instance"`` is supported; ``centroid`` /
            ``centered_instance`` / top-down combined / bottom-up /
            multiclass land in follow-up PRs.
        ValueError: ``runtime`` isn't recognized.

    Notes:
        Skeleton hydration is *not* done here — call
        :func:`sleap_nn.inference.utils.get_skeleton_from_config` on the
        export's ``training_config.yaml`` separately if you need a
        skeleton for ``Predictor.predict(make_labels=True, ...)``.
    """
    from sleap_nn.export.metadata import ExportMetadata
    from sleap_nn.inference.predictor import Predictor as NewPredictor

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
    )

    kwargs: dict = {"layer": layer, "paf_workers": paf_workers}
    if filter_config is not None:
        kwargs["filter_config"] = filter_config
    if tracker_config is not None:
        kwargs["tracker_config"] = tracker_config
    return NewPredictor(**kwargs)


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
):
    """Dispatch on ``metadata.model_type`` → build the right export layer.

    As of PR 18 only ``single_instance`` is supported. Other types
    raise ``NotImplementedError`` and continue using the legacy
    ``sleap_nn.export.inference.predict`` driver in the meantime.
    """
    model_type = metadata.model_type

    if model_type == "single_instance":
        return SingleInstanceLayer(
            backend=backend,
            output_stride=metadata.output_stride,
            preprocess_config=PreprocessConfig(scale=metadata.input_scale),
            postprocess_config=PostprocessConfig(
                peak_threshold=(
                    metadata.peak_threshold
                    if metadata.peak_threshold is not None
                    else 0.2
                ),
                refinement="none",  # peak-finding is baked into the export graph
                return_confmaps=return_confmaps,
            ),
        )

    if model_type in {
        "centroid",
        "centered_instance",
        "topdown",
        "bottomup",
        "multi_class_bottomup",
        "multi_class_topdown",
    }:
        raise NotImplementedError(
            f"from_export_dir: model_type={model_type!r} adapter not yet "
            f"implemented. Currently supported: 'single_instance'. Use the "
            f"legacy `sleap_nn.export.inference.predict(...)` for other "
            f"model types until follow-up PRs land."
        )

    raise ValueError(f"Unrecognized model_type {model_type!r} in export_metadata.json.")


def _select_layer(legacy_predictor: Any, model_types: List[str], device: str):
    """Dispatch on detected model types → build the new layer composition."""
    if "single_instance" in model_types:
        return _build_single_instance_layer(legacy_predictor, device)
    if "bottomup" in model_types:
        return _build_bottomup_layer(legacy_predictor, device)
    if "multi_class_bottomup" in model_types:
        return _build_bottomup_multiclass_layer(legacy_predictor, device)
    has_centroid = "centroid" in model_types
    has_centered = "centered_instance" in model_types
    has_multi_centered = "multi_class_topdown" in model_types
    if has_centroid and has_centered:
        return _build_topdown_layer(legacy_predictor, device)
    if has_centroid and has_multi_centered:
        return _build_topdown_multiclass_layer(legacy_predictor, device)
    raise ValueError(
        f"Unsupported model_paths combination: detected types {model_types}. "
        f"The new Predictor.from_model_paths supports: single_instance, "
        f"bottomup, multi_class_bottomup, top-down (centroid + centered_instance), "
        f"or top-down multiclass (centroid + multi_class_topdown)."
    )
