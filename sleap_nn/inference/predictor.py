"""``Predictor`` — high-level orchestrator for the inference stack.

Composes an :class:`InferenceLayer` (or composed layer like
:class:`TopDownLayer`) with a :class:`Provider` source and a
:class:`FilterPipeline` post-processor.

Three usage tiers:

* :meth:`predict` — synchronous, returns ``sio.Labels`` (or a list of
  ``Outputs`` if ``make_labels=False``). Loads everything into memory;
  use for short videos / interactive sessions.
* :meth:`predict_streaming` — yields one ``Outputs`` per batch as a
  generator; nothing is retained across batches, so memory stays
  O(batch) (with ``paf_workers>0``, O(in-flight window)). Tracking is
  not supported here (it needs the full frame list); use :meth:`predict`.
* :meth:`predict_to_file` — buffered write of a ``.slp`` via
  :class:`IncrementalLabelsWriter`. Heavy tensors are dropped per batch;
  the (slimmed) ``LabeledFrame``s accumulate until finalize (see that
  class for the memory note).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
)

import attrs
import numpy as np
import torch
from loguru import logger

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.bottomup import BottomUpLayer
from sleap_nn.inference.layers.bottomup_multiclass import BottomUpMultiClassLayer
from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.segmentation import SegmentationLayer
from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.layers.topdown_multiclass import (
    CenteredInstanceMultiClassLayer,
    TopDownMultiClassLayer,
)
from sleap_nn.inference.layers.topdown_segmentation import (
    CenteredInstanceMaskLayer,
    TopDownSegmentationLayer,
)
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.providers import Provider
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking


class _CentroidPackaging(NamedTuple):
    """Single-source centroid output-packaging decision.

    Resolved once per :class:`Predictor` (see ``_resolve_centroid_packaging``)
    and threaded identically into in-memory ``to_labels`` and the streaming
    writer so all output paths agree.
    """

    collapse_skeleton: Optional[Any]  # 1-node sio.Skeleton, or None (no collapse)
    anchor_ind: Optional[int]
    emit_centroid: str  # "instance" | "centroid" | "both"
    source: str  # sio.Centroid.source method tag


if TYPE_CHECKING:
    import sleap_io as sio

    from sleap_nn.export.metadata import ExportMetadata


def _safe_len(provider: Any) -> int:
    """Return ``len(provider)`` or ``-1`` if the provider doesn't expose ``__len__``."""
    try:
        return len(provider)
    except TypeError:
        return -1


def _safe_num_frames(provider: Any) -> int:
    """Return ``provider.num_frames()`` or ``-1`` if unavailable/unknown.

    Used to drive frame-based (batch-size-invariant) progress reporting.
    Falls back to ``-1`` for providers that don't implement ``num_frames``.
    """
    fn = getattr(provider, "num_frames", None)
    if fn is None:
        return -1
    try:
        return fn()
    except (TypeError, AttributeError):
        return -1


# ─────────────────────────────────────────────────────────────────────────
# Layer builders — one per model type, given a LoadedAssets instance
# ─────────────────────────────────────────────────────────────────────────


def _pp_field(assets: Any, name: str, default: Any = None) -> Any:
    """Read a field from the loaded assets' resolved ``preprocess_config``."""
    cfg = getattr(assets, "preprocess_config", None)
    if cfg is None:
        return default
    try:
        val = cfg[name] if not hasattr(cfg, name) else getattr(cfg, name)
    except (KeyError, AttributeError):
        return default
    return val if val is not None else default


def _multiclass_class_names(assets: Any, head_type: str) -> Optional[List[str]]:
    """Ordered class names for a multi-class head from the training config.

    Mirrors legacy ``predictors.py`` track construction:

    * ``multi_class_topdown`` →
      ``confmap_config.model_config.head_configs.multi_class_topdown.class_vectors.classes``
      (TopDownMultiClass, predictors.py:3808-3811).
    * ``multi_class_bottomup`` →
      ``bottomup_config.model_config.head_configs.multi_class_bottomup.class_maps.classes``
      (BottomUpMultiClass, predictors.py:2966-2969).

    Returns ``None`` when the config or class list is unavailable.
    """
    if head_type == "multi_class_topdown":
        cfg = getattr(assets, "confmap_config", None)
        sub_key = "class_vectors"
    elif head_type == "multi_class_bottomup":
        cfg = getattr(assets, "bottomup_config", None)
        sub_key = "class_maps"
    else:
        return None
    if cfg is None:
        return None
    try:
        classes = cfg.model_config.head_configs[head_type][sub_key]["classes"]
    except (KeyError, AttributeError, TypeError):
        return None
    if classes is None:
        return None
    return [str(c) for c in classes]


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
        # ``max_instances`` is carried on the LoadedAssets (the legacy
        # ``BottomUpInferenceModel`` has no such field), so read it from the
        # assets, not the inference model — otherwise the cap is silently
        # dropped for the from_model_paths path (#582).
        max_instances=getattr(predictor, "max_instances", None),
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
        max_instances=getattr(predictor, "max_instances", None),
        class_names=_multiclass_class_names(predictor, "multi_class_bottomup"),
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


def _build_bottomup_segmentation_layer(
    predictor: Any, device: str
) -> SegmentationLayer:
    """Wrap a ``BottomUpSegmentationInferenceModel`` in a ``SegmentationLayer``."""
    inf = predictor.inference_model
    max_stride = getattr(predictor, "max_stride", 1) or 1
    return SegmentationLayer(
        backend=TorchBackend(model=inf.torch_model, device=device),
        output_stride=inf.output_stride,
        max_stride=max_stride,
        fg_threshold=inf.fg_threshold,
        min_mask_area=getattr(inf, "min_mask_area", 0),
        max_instances=getattr(inf, "max_instances", None),
        center_nms_kernel=getattr(inf, "center_nms_kernel", 3),
        mask_cleanup=getattr(inf, "mask_cleanup", False),
        mask_cleanup_radius=getattr(inf, "mask_cleanup_radius", 0),
        full_res_masks=getattr(inf, "full_res_masks", False),
        mask_output=getattr(inf, "mask_output", "mask"),
        polygon_epsilon=getattr(inf, "polygon_epsilon", 0.01),
        preprocess_config=PreprocessConfig(
            scale=inf.input_scale,
            max_height=_pp_field(predictor, "max_height"),
            max_width=_pp_field(predictor, "max_width"),
            ensure_rgb=_pp_field(predictor, "ensure_rgb"),
            ensure_grayscale=_pp_field(predictor, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(peak_threshold=inf.peak_threshold),
    )


def _build_centroid_layer(
    centroid_model: Any,
    device: str,
    assets: Optional[Any] = None,
) -> CentroidLayer:
    """Wrap a ``CentroidCrop`` model in a ``CentroidLayer``."""
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
    """Wrap a ``FindInstancePeaks`` model in a ``CenteredInstanceLayer``."""
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

    Populates the sizematcher fields (``max_height`` / ``max_width`` /
    ``ensure_rgb`` / ``ensure_grayscale``) from the resolved preprocess config so
    the centered-instance-only (GT-centroid) path sizematches each frame the same
    way the model path does. Dropping them diverged from legacy whenever the
    centered-instance config set max_height/max_width (#582). ``scale`` stays 1.0:
    the GT path only re-applies max_height/max_width (not input_scale), and the
    centered-instance layer applies its own input_scale to the crops.
    """
    centroid_model = assets.inference_model.centroid_crop
    return CentroidLayer(
        backend=backend,
        output_stride=1,
        max_instances=None,
        max_stride=1,
        anchor_ind=getattr(centroid_model, "anchor_ind", None),
        use_gt_centroids=True,
        preprocess_config=PreprocessConfig(
            scale=1.0,
            max_height=_pp_field(assets, "max_height"),
            max_width=_pp_field(assets, "max_width"),
            ensure_rgb=_pp_field(assets, "ensure_rgb"),
            ensure_grayscale=_pp_field(assets, "ensure_grayscale"),
        ),
        postprocess_config=PostprocessConfig(),
    )


def _build_centered_instance_multiclass_layer(
    instance_model: Any, device: str, class_names: Optional[List[str]] = None
) -> CenteredInstanceMultiClassLayer:
    """Wrap a ``TopDownMultiClassFindInstancePeaks`` model in a layer."""
    return CenteredInstanceMultiClassLayer(
        backend=TorchBackend(model=instance_model.torch_model, device=device),
        output_stride=instance_model.output_stride,
        max_stride=instance_model.max_stride,
        class_names=class_names,
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
    inst_layer = _build_centered_instance_multiclass_layer(
        inf.instance_peaks,
        device,
        class_names=_multiclass_class_names(predictor, "multi_class_topdown"),
    )
    crop_h, crop_w = inf.centroid_crop.crop_hw
    return TopDownMultiClassLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=inst_layer,
        crop_size=(crop_h, crop_w),
    )


def _build_centered_instance_mask_layer(
    mask_model: Any, device: str
) -> CenteredInstanceMaskLayer:
    """Wrap a ``CenteredInstanceMaskInferenceModel`` in a mask layer (stage 2)."""
    return CenteredInstanceMaskLayer(
        backend=TorchBackend(model=mask_model.torch_model, device=device),
        output_stride=mask_model.output_stride,
        max_stride=mask_model.max_stride,
        fg_threshold=getattr(mask_model, "fg_threshold", 0.5),
        preprocess_config=PreprocessConfig(scale=mask_model.input_scale),
        postprocess_config=PostprocessConfig(),
    )


def _build_topdown_segmentation_layer(
    predictor: Any, device: str
) -> TopDownSegmentationLayer:
    """Compose centroid + per-crop-mask stages into a ``TopDownSegmentationLayer``."""
    inf = predictor.inference_model
    centroid_model = inf.centroid_crop
    mask_model = inf.instance_peaks
    # GT-centroid fallback (seg dir only, no centroid model): the CentroidCrop was
    # built with ``use_gt_centroids=True`` and carries no torch_model, so build a
    # GT-only centroid layer (mirrors the centered-instance-only branch below).
    if getattr(centroid_model, "use_gt_centroids", False):
        mask_layer = _build_centered_instance_mask_layer(mask_model, device)
        centroid_layer = _build_centroid_layer_gt_only(predictor, mask_layer.backend)
    else:
        centroid_layer = _build_centroid_layer(centroid_model, device, assets=predictor)
        mask_layer = _build_centered_instance_mask_layer(mask_model, device)
    crop_h, crop_w = centroid_model.crop_hw
    return TopDownSegmentationLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=mask_layer,
        crop_size=(crop_h, crop_w),
        mask_output=getattr(mask_model, "mask_output", "mask"),
        polygon_epsilon=getattr(mask_model, "polygon_epsilon", 0.01),
    )


def _build_sam_segmentation_layer(
    predictor: Any,
    device: str,
    mask_backend: str,
    *,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    prompt_mode: str = "crop_center",
) -> TopDownSegmentationLayer:
    """Compose a centroid stage + a SAM crop-center stage-2 (the top-down seam).

    Reuses :class:`TopDownSegmentationLayer` verbatim — only the stage-2 layer is
    swapped from the trained ``CenteredInstanceMaskLayer`` to
    :class:`~sleap_nn.inference.sam.mask_layer.FindInstanceMaskSAM`, which prompts
    each crop at its center pixel (PLAN §2.1, P2). The whole offset/scale/
    ``to_masks``/``save`` path is inherited. ``mask_backend`` is **explicit /
    required** (PLAN L2); the SAM import is lazy (inside the backend).

    Args:
        predictor: The :class:`Predictor` carrying the inference model (a centroid
            stage; the seg/instance stage is replaced by SAM and may be absent
            via the GT-centroid fallback).
        device: Torch device for the SAM backend.
        mask_backend: Explicit backend name (``"sam"`` / ``"sam3"``).
        sam_checkpoint: SAM1 checkpoint path (required for ``"sam"``).
        sam_model_type: SAM1 model registry key.
        prompt_mode: The per-crop prompt mode; ``"crop_center"`` (the seam) is the
            default and only mode wired here in PR-A.

    Returns:
        A composed :class:`TopDownSegmentationLayer` with a SAM stage-2.
    """
    from sleap_nn.inference.sam import get_mask_backend
    from sleap_nn.inference.sam.mask_layer import FindInstanceMaskSAM

    inf = predictor.inference_model
    centroid_model = inf.centroid_crop
    backend = get_mask_backend(
        mask_backend,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
    )
    mask_layer = FindInstanceMaskSAM(backend)
    if getattr(centroid_model, "use_gt_centroids", False):
        centroid_layer = _build_centroid_layer_gt_only(predictor, None)
    else:
        centroid_layer = _build_centroid_layer(centroid_model, device, assets=predictor)
    crop_h, crop_w = centroid_model.crop_hw
    return TopDownSegmentationLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=mask_layer,
        crop_size=(crop_h, crop_w),
        mask_output="mask",
    )


def _select_layer(assets: Any, model_types: List[str], device: str):
    """Dispatch on detected model types and build the appropriate layer composition."""
    if "single_instance" in model_types:
        return _build_single_instance_layer(assets, device)
    if "bottomup" in model_types:
        return _build_bottomup_layer(assets, device)
    if "multi_class_bottomup" in model_types:
        return _build_bottomup_multiclass_layer(assets, device)
    if "bottomup_segmentation" in model_types:
        return _build_bottomup_segmentation_layer(assets, device)
    # Top-down (crop-centered) segmentation: a centroid + centered_instance_segmentation
    # pair, OR a seg dir alone (GT-centroid fallback). Checked BEFORE the bare
    # ``has_centroid`` branch so a centroid+seg pair isn't routed to centroid-only.
    if "centered_instance_segmentation" in model_types:
        return _build_topdown_segmentation_layer(assets, device)
    has_centroid = "centroid" in model_types
    has_centered = "centered_instance" in model_types
    has_multi_centered = "multi_class_topdown" in model_types
    if has_centroid and has_centered:
        return _build_topdown_layer(assets, device)
    if has_centroid and has_multi_centered:
        return _build_topdown_multiclass_layer(assets, device)
    if has_centroid:
        return _build_centroid_layer(
            assets.inference_model.centroid_crop,
            device,
            assets=assets,
        )
    if has_centered:
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
        f"Predictor.from_model_paths supports: single_instance, "
        f"bottomup, multi_class_bottomup, top-down (centroid + centered_instance), "
        f"top-down multiclass (centroid + multi_class_topdown), centroid-only, "
        f"or centered-instance-only (requires a .slp source for GT centroids)."
    )


# ─────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────


def _skeleton_from_export(export_dir: Path, metadata: "ExportMetadata") -> Any:
    """Best-effort skeleton from an export directory."""
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
    """Pick the runtime + model file for an export directory."""
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
    peak_conf_threshold: Optional[float] = None,
):
    """Dispatch on ``metadata.model_type`` to build the right export adapter."""
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
        # Resolve the configured anchor node so the centroid lands on it at
        # packaging time, mirroring legacy export inference (#582). Old exports
        # without `anchor_part` keep anchor_ind=None (node-0 fallback).
        anchor_part = getattr(metadata, "anchor_part", None)
        anchor_ind = None
        if anchor_part is not None:
            node_names = list(getattr(metadata, "node_names", []) or [])
            if anchor_part in node_names:
                anchor_ind = node_names.index(anchor_part)
            else:
                raise ValueError(
                    f"Anchor part {anchor_part!r} not found in export node_names: "
                    f"{node_names}."
                )
        return ExportedCentroidLayer(backend=backend, anchor_ind=anchor_ind)
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
            peak_conf_threshold=peak_conf_threshold,
        )
    if model_type in ("multi_class_topdown", "multi_class_topdown_combined"):
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
            peak_conf_threshold=peak_conf_threshold,
        )

    raise ValueError(f"Unrecognized model_type {model_type!r} in export_metadata.json.")


@attrs.define
class Predictor:
    """High-level orchestrator: layer + source dispatch + filter pipeline.

    Args:
        layer: Any object exposing ``predict(image) -> Outputs``. Includes
            every :class:`InferenceLayer` subclass plus composed layers
            like :class:`TopDownLayer`.
        skeleton: Optional ``sio.Skeleton`` resolved from the training
            config. Populated automatically by
            :meth:`from_model_paths`.
            Used as the default for ``predict(make_labels=True)`` and
            ``predict_to_file()`` when no explicit ``skeleton`` kwarg is
            passed.
        batch_size: Default batch size for auto-constructed providers when
            ``predict`` / ``predict_streaming`` receive an ``sio.Video``
            or ``sio.Labels`` instead of a pre-built ``Provider``.
        filter_config: Optional post-inference filter config. Default is
            the no-op identity.
        paf_workers: Number of CPU worker processes for the bottom-up
            PAF grouping stage. ``0`` (default) runs grouping inline in
            the main process — the parity path. ``>0`` is only honored
            when ``layer`` is a :class:`BottomUpLayer`; for any other
            layer type the value is ignored.
        tracker_config: Optional :class:`TrackerConfig`. When set,
            :meth:`predict` runs the tracker on the resulting
            ``sio.Labels`` (requires ``make_labels=True``) before
            returning.

    Notes:
        Keeps no state across calls — same predictor can be reused on
        multiple sources safely.
    """

    layer: Any  # TODO: unify layer types under a common Protocol
    skeleton: Optional["sio.Skeleton"] = None
    batch_size: int = 4
    filter_config: FilterConfig = attrs.Factory(FilterConfig)
    paf_workers: int = 0
    tracker_config: Optional[TrackerConfig] = None
    # Provenance metadata (populated by the factories; attached to the saved /
    # returned Labels by predict() so .slp files carry inference lineage —
    # #530 gap: the new flow previously wrote no provenance).
    model_paths: Optional[List[str]] = None
    device: Optional[str] = None
    # Centroid-only output representation: "instance" (default; single-node
    # PredictedInstance), "centroid" (sio.PredictedCentroid), or "both".
    emit_centroid: str = "instance"

    @property
    def filter_pipeline(self) -> FilterPipeline:
        """Build a fresh ``FilterPipeline`` from the config (cheap)."""
        return FilterPipeline(self.filter_config)

    # ──────────────────────────────────────────────────────────────────
    # Factory classmethods
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_model_paths(
        cls,
        model_paths: List[str],
        *,
        device: str = "cpu",
        batch_size: int = 4,
        backbone_ckpt_path: Optional[str] = None,
        head_ckpt_path: Optional[str] = None,
        peak_threshold: Union[float, List[float]] = 0.2,
        integral_refinement: str = "integral",
        integral_patch_size: int = 5,
        max_instances: Optional[int] = None,
        return_confmaps: bool = False,
        preprocess_config: Optional[Any] = None,
        anchor_part: Optional[str] = None,
        filter_config: Optional["FilterConfig"] = None,
        paf_workers: int = 0,
        tracker_config: Optional["TrackerConfig"] = None,
        centroid_only: bool = False,
        emit_centroid: str = "instance",
        max_edge_length_ratio: float = 0.25,
        dist_penalty_weight: float = 1.0,
        n_points: int = 10,
        min_instance_peaks: Union[int, float] = 0,
        min_line_scores: float = 0.25,
        fg_threshold: float = 0.5,
        min_mask_area: int = 0,
        center_nms_kernel: int = 3,
        mask_cleanup: bool = False,
        mask_cleanup_radius: int = 0,
        full_res_masks: bool = False,
        mask_output: str = "mask",
        polygon_epsilon: float = 0.01,
    ) -> "Predictor":
        """Build a :class:`Predictor` from one or more checkpoint paths.

        Args:
            model_paths: Trained model directories containing
                ``training_config.{yaml,json}`` + ``best.ckpt``. Each entry may
                alternatively be a path to that ``best.ckpt`` or
                ``training_config.{yaml,json}`` file; all forms resolve to the
                model directory and load ``best.ckpt`` (#575). For top-down, pass
                two paths (centroid + centered-instance) in either order.
            device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"cuda:N"``.
            batch_size: Default batch size for auto-constructed providers.
            backbone_ckpt_path: Override backbone weights with this ``.ckpt``.
            head_ckpt_path: Override head weights.
            peak_threshold: Default peak threshold. ``List[float]`` for
                top-down (``[centroid_thresh, keypoint_thresh]``). Can be
                overridden per-call via ``predict(peak_threshold=...)``.
            integral_refinement: ``"integral"`` or ``"none"``.
            integral_patch_size: Refinement patch size.
            max_instances: Cap on instances per frame.
            return_confmaps: Return confidence maps on Outputs.
            preprocess_config: OmegaConf overrides for preprocessing.
            anchor_part: Override centroid anchor node name.
            filter_config: Post-inference :class:`FilterConfig`.
            paf_workers: CPU workers for bottom-up PAF grouping.
            tracker_config: :class:`TrackerConfig` for tracking.
            centroid_only: Force centroid-only output even when a
                centered-instance model is among ``model_paths``.
            emit_centroid: Centroid-only output representation: ``"instance"``
                (default; single-node ``PredictedInstance``), ``"centroid"``
                (``sio.PredictedCentroid``), or ``"both"``. Honored only for
                centroid-only layers.
            max_edge_length_ratio: Bottom-up PAF max edge length ratio.
            dist_penalty_weight: Bottom-up PAF distance penalty weight.
            n_points: Bottom-up PAF line integration sample count.
            min_instance_peaks: Bottom-up min peaks for a valid instance.
            min_line_scores: Bottom-up per-edge match threshold. (These five
                are applied only to plain bottom-up models.)
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
                pixels) applied during ``mask_cleanup``; ``0`` keeps keep-largest
                + fill only (bottom-up segmentation only).
            full_res_masks: Encode masks at full original resolution instead of
                the model output-stride grid (default ``False``: stride encoding
                is ~stride^2 smaller and lossless at model resolution; bottom-up
                segmentation only).
            mask_output: Mask output representation: ``"mask"`` (default),
                ``"polygon"`` (Douglas-Peucker ``sio.PredictedROI`` only), or
                ``"both"`` (bottom-up segmentation only).
            polygon_epsilon: Douglas-Peucker tolerance (fraction of perimeter)
                for ``mask_output`` polygon/both (bottom-up segmentation only).
        """
        from sleap_nn.inference.loaders import load_model_assets

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
            full_res_masks=full_res_masks,
            mask_output=mask_output,
            polygon_epsilon=polygon_epsilon,
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
            "model_paths": [str(p) for p in model_paths],
            "device": device,
            "emit_centroid": emit_centroid,
        }
        if filter_config is not None:
            kwargs["filter_config"] = filter_config
        if tracker_config is not None:
            kwargs["tracker_config"] = tracker_config

        # Spin-up log: a one-line record of *what* model is running on *what*,
        # so a run starts with a legible header instead of silence (#610).
        n_nodes = len(skeleton.nodes) if skeleton is not None else None
        spec = [
            f"type={'+'.join(model_types)}",
            f"backbone={loaded.backbone_type}",
            f"nodes={n_nodes}",
            f"device={device}",
            f"batch_size={batch_size}",
            f"peak_threshold={peak_threshold}",
            f"max_instances={max_instances}",
            f"integral_refinement={integral_refinement}",
            f"paf_workers={paf_workers}",
        ]
        if "bottomup_segmentation" in model_types:
            spec.append(f"fg_threshold={fg_threshold}")
            spec.append(f"min_mask_area={min_mask_area}")
            spec.append(f"full_res_masks={full_res_masks}")
            spec.append(f"mask_output={mask_output}")
        if "centered_instance_segmentation" in model_types:
            spec.append(f"fg_threshold={fg_threshold}")
            spec.append(f"mask_output={mask_output}")
        logger.info("Loaded inference model | " + " | ".join(spec))

        return cls(**kwargs)

    @classmethod
    def from_export_dir(
        cls,
        export_dir: Union[str, Any],
        *,
        runtime: str = "auto",
        device: str = "auto",
        batch_size: int = 4,
        return_confmaps: bool = False,
        filter_config: Optional["FilterConfig"] = None,
        paf_workers: int = 0,
        tracker_config: Optional["TrackerConfig"] = None,
        max_instances: Optional[int] = None,
        min_instance_peaks: float = 0,
        min_line_scores: float = 0.25,
        peak_conf_threshold: Optional[float] = None,
        emit_centroid: str = "instance",
    ) -> "Predictor":
        """Build a :class:`Predictor` from an exported ONNX/TensorRT directory.

        Args:
            export_dir: Directory containing ``export_metadata.json`` +
                ``model.onnx`` or ``model.trt``.
            runtime: ``"auto"`` (prefer TRT), ``"onnx"``, or ``"tensorrt"``.
            device: Device string.
            batch_size: Default batch size.
            return_confmaps: Return confidence maps on Outputs.
            filter_config: Post-inference :class:`FilterConfig`.
            paf_workers: CPU workers for bottom-up PAF grouping.
            tracker_config: :class:`TrackerConfig` for tracking.
            max_instances: Cap on instances per frame (bottom-up).
            min_instance_peaks: Min peaks for a valid instance (bottom-up).
            min_line_scores: Per-edge match threshold (bottom-up).
            peak_conf_threshold: Runtime peak-confidence threshold for the
                exported bottom-up path. Gates PAF candidate connections by the
                src/dst peak confidence (legacy parity). Defaults to the
                threshold baked at export time (``metadata.peak_threshold``).
                Note the wrapper already bakes a peak threshold during peak
                finding, so this can only *tighten* beyond the baked value.
            emit_centroid: Centroid-only output representation for an exported
                standalone centroid model: ``"instance"`` (default; single-node
                ``PredictedInstance``, frontend-compatible), ``"centroid"``
                (``sio.PredictedCentroid``), or ``"both"``. Honored only for
                ``ExportedCentroidLayer``; mirrors ``from_model_paths`` so the
                exported runtime matches the checkpoint path.
        """
        from sleap_nn.export.metadata import ExportMetadata

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

        # Default the runtime peak-confidence threshold to the value baked at
        # export time, falling back to legacy's 0.2 when the metadata carries no
        # baked threshold (matches legacy export inference, #582).
        if peak_conf_threshold is not None:
            resolved_peak_conf = peak_conf_threshold
        else:
            meta_thr = getattr(metadata, "peak_threshold", None)
            resolved_peak_conf = meta_thr if meta_thr is not None else 0.2
        layer = _select_export_layer(
            metadata=metadata,
            backend=backend,
            return_confmaps=return_confmaps,
            max_instances=max_instances,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
            peak_conf_threshold=resolved_peak_conf,
        )

        skeleton = _skeleton_from_export(export_dir, metadata)
        kwargs: dict = {
            "layer": layer,
            "skeleton": skeleton,
            "batch_size": batch_size,
            "paf_workers": paf_workers,
            "model_paths": [str(export_dir)],
            "device": device,
            "emit_centroid": emit_centroid,
        }
        if filter_config is not None:
            kwargs["filter_config"] = filter_config
        if tracker_config is not None:
            kwargs["tracker_config"] = tracker_config
        return cls(**kwargs)

    # ──────────────────────────────────────────────────────────────────
    # Source dispatch: sio.Video / sio.Labels / str / Provider
    # ──────────────────────────────────────────────────────────────────

    def _needs_gt_instances(self) -> bool:
        """Whether the configured layer consumes ground-truth instances.

        GT-fallback layers (``CentroidLayer(use_gt_centroids=True)`` /
        ``CenteredInstanceLayer(use_gt_peaks=True)``) require frames that carry
        user instances, so a ``.slp`` source must be restricted to labeled
        frames. The normal (real-model) path predicts ALL frames, matching the
        legacy ``LabelsReader`` default (#530 audit: new flow wrongly defaulted
        to labeled-only).
        """
        layer = self.layer
        candidates = [layer]
        for sub in ("centroid_layer", "centered_instance_layer"):
            inner = getattr(layer, sub, None)
            if inner is not None:
                candidates.append(inner)
        return any(
            getattr(c, "use_gt_centroids", False) or getattr(c, "use_gt_peaks", False)
            for c in candidates
        )

    def _build_inference_provenance(
        self,
        *,
        source: Any,
        start_time: Any,
        end_time: Any,
        n_frames: int,
        inference_params: dict,
    ) -> dict:
        """Provenance dict for the saved/returned Labels (#530 gap fix)."""
        from sleap_nn.inference.provenance import build_inference_provenance

        tracking_params = (
            attrs.asdict(self.tracker_config)
            if self.tracker_config is not None
            else None
        )
        return build_inference_provenance(
            model_paths=self.model_paths,
            model_type=type(self.layer).__name__,
            device=self.device,
            start_time=start_time,
            end_time=end_time,
            input_path=source if isinstance(source, str) else None,
            frames_processed=n_frames,
            inference_params=inference_params,
            tracking_params=tracking_params,
        )

    @staticmethod
    def _describe_source(source: Any) -> str:
        """Best-effort human label for a prediction source (#610)."""
        if isinstance(source, str):
            return source
        filename = getattr(source, "filename", None)
        if filename:
            return str(filename)
        return type(source).__name__

    def _log_inference_start(
        self, source: Any, provider: "Provider", videos: Optional[list]
    ) -> None:
        """Log a one-line spin-up record of the source being processed (#610)."""
        n_frames = _safe_num_frames(provider)
        parts = [
            f"source={self._describe_source(source)}",
            f"frames={n_frames if n_frames >= 0 else '?'}",
            f"videos={len(videos) if videos else 1}",
        ]
        sio_video = getattr(provider, "_sio_video", None)
        if sio_video is not None:
            try:
                shape = tuple(sio_video.shape)  # (N, H, W, C)
                if len(shape) == 4:
                    parts.append(f"shape={shape[1]}x{shape[2]}x{shape[3]}")
            except Exception:  # pragma: no cover — metadata best-effort
                pass
            fps = getattr(sio_video, "fps", None)
            if fps:
                parts.append(f"fps={fps}")
        parts.append(f"tracking={self.tracker_config is not None}")
        logger.info("Starting inference | " + " | ".join(parts))

    def _log_inference_summary(
        self,
        *,
        n_frames: int,
        elapsed_s: float,
        output: Optional[str] = None,
        n_objects: Optional[int] = None,
        object_label: str = "instances",
    ) -> None:
        """Log a one-line post-run summary (#610).

        ``n_objects`` (instances or masks) is optional — the streaming path
        drops per-frame objects, so it reports frames/throughput only.
        """
        fps = n_frames / elapsed_s if elapsed_s > 0 else 0.0
        parts = [f"frames={n_frames}"]
        if n_objects is not None:
            mean = n_objects / n_frames if n_frames > 0 else 0.0
            parts.append(f"{object_label}={n_objects} ({mean:.2f}/frame)")
        parts += [
            f"elapsed={elapsed_s:.1f}s",
            f"throughput={fps:.1f} fps",
            f"tracking={self.tracker_config is not None}",
        ]
        if output:
            parts.append(f"output={output}")
        logger.info("Inference complete | " + " | ".join(parts))

    def _make_provider(
        self,
        source: Any,
        frames: Optional[List[int]] = None,
        **provider_kwargs: Any,
    ) -> tuple["Provider", Optional[List["sio.Video"]]]:
        """Wrap a source into a ``Provider`` + extract videos for label packaging.

        Returns ``(provider, videos)`` where ``videos`` is a list of
        ``sio.Video`` when derivable from the source, else ``None``.
        """
        import sleap_io as sio

        from sleap_nn.inference.providers import (
            LabelsProvider,
            VideoProvider,
        )

        if isinstance(source, (str, np.ndarray)):
            if isinstance(source, str) and source.endswith(".slp"):
                # Load once so we can both build the provider AND attach the
                # real videos to the output Labels — legacy parity: predicted
                # frames must reference the source video, not be dropped
                # (#530 audit: .slp path returned videos=None).
                labels = sio.load_slp(source)
                provider_kwargs.setdefault(
                    "only_labeled_frames", self._needs_gt_instances()
                )
                provider = LabelsProvider(
                    labels=labels,
                    batch_size=self.batch_size,
                    **provider_kwargs,
                )
                return provider, (list(labels.videos) if labels.videos else None)
            video = sio.Video(source) if isinstance(source, str) else None
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            videos = [video] if video is not None else None
            return provider, videos

        if isinstance(source, sio.Video):
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            return provider, [source]

        if isinstance(source, sio.Labels):
            provider_kwargs.setdefault(
                "only_labeled_frames", self._needs_gt_instances()
            )
            provider = LabelsProvider(
                labels=source,
                batch_size=self.batch_size,
                **provider_kwargs,
            )
            videos = list(source.videos) if source.videos else None
            return provider, videos

        if isinstance(source, (list, tuple)):
            # Multi-source input: predict([v1, v2, v3]) -> one merged Labels with
            # monotonically increasing per-frame video_indices (#582). Recurse to
            # reuse per-type dispatch, then concatenate. Per-source frame
            # selection is not expressible for a flat list, so `frames` is not
            # applied here (build providers explicitly if you need it).
            from sleap_nn.inference.providers import MultiVideoProvider

            if len(source) == 0:
                raise ValueError("predict() received an empty list of sources.")
            sub_providers: list = []
            all_videos: list = []
            video_offsets: list = []
            for sub in source:
                sub_provider, sub_videos = self._make_provider(sub, **provider_kwargs)
                sub_providers.append(sub_provider)
                # Each source starts at the current end of the merged video
                # list; its own (possibly multi-video) indices are offset by
                # this. Substitute a placeholder when a sub-source has no
                # derivable video so frames never reference a None video
                # (unserializable) — matches the writer's placeholder.
                video_offsets.append(len(all_videos))
                if sub_videos:
                    all_videos.extend(sub_videos)
                else:
                    all_videos.append(sio.Video(filename="unknown", backend=None))
            return (
                MultiVideoProvider(
                    providers=sub_providers, video_offsets=video_offsets
                ),
                all_videos,
            )

        if hasattr(source, "__iter__"):
            return source, None

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            f"Pass an sio.Video, sio.Labels, file path string, or a Provider."
        )

    @staticmethod
    def retrack(
        labels: "sio.Labels",
        tracker_config: TrackerConfig,
        clean_empty_frames: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "sio.Labels":
        """Retrack an existing ``sio.Labels`` without running inference.

        Pure tracking -- useful when you already have predicted instances
        in a ``.slp`` and just want to (re)apply a tracker. The tracker
        runs once over the full LabeledFrame list; post-tracking cleanup
        (cull / connect-single-breaks) is applied per ``tracker_config``.

        Args:
            labels: A ``sio.Labels`` whose ``predicted_instances`` are
                tracked in-place semantics — this returns a new
                ``Labels`` with tracked instances.
            tracker_config: :class:`TrackerConfig` to drive the tracker.
            clean_empty_frames: When ``True``, drop empty frames from
                the result (matches ``--no_empty_frames``).
            progress_callback: Optional ``(processed_frames, total_frames)``
                callback invoked after each frame is tracked.

        Returns:
            New ``sio.Labels`` with tracks attached.
        """
        out = apply_tracking(labels, tracker_config, progress_callback)
        if clean_empty_frames:
            out.clean(frames=True, skeletons=False)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Synchronous: returns Outputs list or sio.Labels
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Any,
        *,
        make_labels: bool = True,
        frames: Optional[List[int]] = None,
        skeleton: Optional["sio.Skeleton"] = None,
        videos: Optional[List["sio.Video"]] = None,
        clean_empty_frames: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        tracking_progress_callback: Optional[Callable[[int, int], None]] = None,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
        return_pafs: Optional[bool] = None,
        return_paf_graph: Optional[bool] = None,
        return_class_maps: Optional[bool] = None,
        return_class_vectors: Optional[bool] = None,
    ) -> Union[List[Outputs], "sio.Labels"]:
        """Run inference on a source.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`. When a non-Provider source
                is given, a provider is auto-constructed using
                ``self.batch_size``.
            make_labels: When ``True`` (the default), return a
                ``sio.Labels``. Set to ``False`` for a raw
                ``List[Outputs]``.
            frames: Frame indices to predict on. Only used when ``source``
                is an ``sio.Video`` or video path.
            skeleton: ``sio.Skeleton`` for label conversion. Falls back to
                ``self.skeleton`` when ``None``.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for label conversion. Auto-derived from
                the source when possible.
            clean_empty_frames: When ``True`` and ``make_labels=True``,
                drop ``LabeledFrame``s with no instances from the
                returned ``sio.Labels``.
            progress_callback: Optional ``(processed_frames, total_frames)``
                callback invoked after each batch. Counts are in frames
                (batch-size-invariant); ``total_frames`` is ``-1`` when the
                provider can't report its length up front.
            tracking_progress_callback: Optional
                ``(processed_frames, total_frames)`` callback invoked
                after each frame during tracking.
            peak_threshold: Override peak threshold for all stages. For
                per-stage control on top-down models, use
                ``centroid_threshold`` / ``keypoint_threshold`` instead.
            centroid_threshold: Override peak threshold for the centroid
                stage only (top-down models).
            keypoint_threshold: Override peak threshold for the centered-
                instance stage only (top-down models).
            max_instances: Override max instances per frame.
            integral_refinement: ``"integral"`` or ``"none"``.
            integral_patch_size: Override integral refinement patch size.
            return_confmaps: Override whether to return confidence maps.
            return_crops: Override whether to return per-instance crops
                (top-down only).
            return_pafs: Override whether to return part-affinity fields
                (bottom-up).
            return_paf_graph: Override whether to return the PAF graph
                (bottom-up).
            return_class_maps: Override whether to return class maps
                (multi-class bottom-up).
            return_class_vectors: Override whether to return class vectors
                (multi-class top-down).

        Returns:
            ``sio.Labels`` (default) or ``List[Outputs]`` (when
            ``make_labels=False``).
        """
        from datetime import datetime

        # Tracking operates on sio.PredictedInstance objects; centroid emission
        # (emit_centroid != 'instance') puts predictions in LabeledFrame.centroids
        # as sio.PredictedCentroid, which the tracker would silently drop. Fail
        # fast rather than lose data.
        if self.tracker_config is not None and self.emit_centroid != "instance":
            raise ValueError(
                "Tracking is incompatible with emit_centroid="
                f"{self.emit_centroid!r}: the tracker operates on "
                "sio.PredictedInstance objects, but this mode emits "
                "sio.PredictedCentroid objects (in LabeledFrame.centroids) that "
                "would be dropped. Use emit_centroid='instance' (the default) "
                "for tracking."
            )

        # Segmentation emits sio.PredictedSegmentationMask objects to
        # LabeledFrame.masks; apply_tracking auto-detects the mask-only labels
        # and tracks them by pixel mask-IoU (#619), so tracking is supported.

        provider, auto_videos = self._make_provider(source, frames=frames)
        if videos is None:
            videos = auto_videos

        self._log_inference_start(source, provider, videos)
        _prov_start = datetime.now()
        with self._postprocess_overrides(
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
        ):
            outputs_list = list(self._batch_iter(provider, progress_callback))
        _prov_end = datetime.now()

        if not make_labels:
            if self.tracker_config is not None:
                raise ValueError(
                    "tracker_config requires make_labels=True; the tracker "
                    "operates on sio.PredictedInstance objects."
                )
            return outputs_list
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None and not self._is_segmentation_layer():
            raise ValueError(
                "make_labels=True requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via Predictor.from_model_paths() "
                "which sets it automatically from the training config."
            )
        labels = self.to_labels(outputs_list, videos=videos)
        if self.tracker_config is not None:
            labels = apply_tracking(
                labels, self.tracker_config, tracking_progress_callback
            )
        if clean_empty_frames:
            labels.clean(frames=True, skeletons=False)
        labels.provenance = self._build_inference_provenance(
            source=source,
            start_time=_prov_start,
            end_time=_prov_end,
            n_frames=len(labels.labeled_frames),
            inference_params={
                "peak_threshold": peak_threshold,
                "centroid_threshold": centroid_threshold,
                "keypoint_threshold": keypoint_threshold,
                "max_instances": max_instances,
                "integral_refinement": integral_refinement,
                "integral_patch_size": integral_patch_size,
                "batch_size": self.batch_size,
            },
        )

        # Post-run summary (#610). Segmentation emits masks (LabeledFrame.masks);
        # everything else emits instances (LabeledFrame.instances).
        n_lf = len(labels.labeled_frames)
        if self._is_segmentation_layer():
            n_objects = sum(
                len(getattr(lf, "masks", None) or []) for lf in labels.labeled_frames
            )
            object_label = "masks"
        else:
            n_objects = sum(len(lf.instances) for lf in labels.labeled_frames)
            object_label = "instances"
        self._log_inference_summary(
            n_frames=n_lf,
            elapsed_s=(_prov_end - _prov_start).total_seconds(),
            n_objects=n_objects,
            object_label=object_label,
        )
        return labels

    # ──────────────────────────────────────────────────────────────────
    # Streaming: yields one Outputs at a time
    # ──────────────────────────────────────────────────────────────────

    def predict_streaming(
        self,
        source: Any,
        *,
        frames: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
        return_pafs: Optional[bool] = None,
        return_paf_graph: Optional[bool] = None,
        return_class_maps: Optional[bool] = None,
        return_class_vectors: Optional[bool] = None,
    ) -> Iterator[Outputs]:
        """Yield one ``Outputs`` per batch from ``source``.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`.
            frames: Frame indices (only for video sources).
            progress_callback: Optional ``(processed_frames, total_frames)``
                callback.
            peak_threshold: Override peak threshold for all stages.
            centroid_threshold: Override centroid stage threshold (top-down).
            keypoint_threshold: Override centered-instance threshold (top-down).
            max_instances: Override max instances per frame.
            integral_refinement: ``"integral"`` or ``"none"``.
            integral_patch_size: Override integral refinement patch size.
            return_confmaps: Override whether to return confidence maps.
            return_crops: Override whether to return per-instance crops
                (top-down only).
            return_pafs: Override whether to return part-affinity fields
                (bottom-up).
            return_paf_graph: Override whether to return the PAF graph
                (bottom-up).
            return_class_maps: Override whether to return class maps
                (multi-class bottom-up).
            return_class_vectors: Override whether to return class vectors
                (multi-class top-down).
        """
        if self.tracker_config is not None:
            raise ValueError(
                "tracker_config is not supported on predict_streaming / "
                "predict_to_file. End-of-stream tracker cleanup needs the "
                "full LabeledFrame list; use predict() instead."
            )
        provider, _ = self._make_provider(source, frames=frames)
        with self._postprocess_overrides(
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
        ):
            if self.paf_workers > 0 and self._can_pipeline():
                yield from self._predict_streaming_pipelined(
                    provider, progress_callback
                )
                return
            yield from self._batch_iter(provider, progress_callback)

    # ──────────────────────────────────────────────────────────────────
    # Disk-streaming: write to a .slp incrementally
    # ──────────────────────────────────────────────────────────────────

    def predict_to_file(
        self,
        source: Any,
        path: str,
        *,
        frames: Optional[List[int]] = None,
        skeleton: Optional["sio.Skeleton"] = None,
        videos: Optional[List["sio.Video"]] = None,
        write_interval: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Run inference and write results to a ``.slp`` file.

        Each batch's ``Outputs`` is slimmed and converted to LabeledFrames
        immediately, so the heavy intermediate tensors (confmaps, PAFs) are
        dropped right away. The slimmed LabeledFrames accumulate until the
        file is finalized at close (see :class:`IncrementalLabelsWriter` for
        the memory note).

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`.
            path: Destination ``.slp`` path.
            frames: Frame indices (only for video sources).
            skeleton: ``sio.Skeleton`` for instance conversion. Falls back
                to ``self.skeleton`` when ``None``.
            videos: Optional list of ``sio.Video`` indexed by
                ``video_indices`` for the saved labels.
            write_interval: Number of LabeledFrames to buffer before
                a disk flush.
            progress_callback: Optional ``(processed_frames, total_frames)``
                callback invoked after each batch.

        Returns:
            The (resolved) destination path string.
        """
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None and not self._is_segmentation_layer():
            raise ValueError(
                "predict_to_file requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via Predictor.from_model_paths() "
                "which sets it automatically from the training config."
            )
        from datetime import datetime

        from sleap_nn.inference.writer import IncrementalLabelsWriter

        provider, derived = self._make_provider(source, frames=frames)
        # Preserve the real source video(s) on the streamed .slp instead of the
        # 'unknown' placeholder. Mirrors the in-memory predict() path; for a
        # pre-built Provider source `derived` is None so the caller-supplied
        # `videos` (possibly None) is used (#582).
        if videos is None:
            videos = derived
        self._log_inference_start(source, provider, derived)
        pkg = self._resolve_centroid_packaging()
        writer = IncrementalLabelsWriter(
            path=path,
            skeleton=self.skeleton,
            videos=videos,
            write_interval=write_interval,
            anchor_ind=pkg.anchor_ind,
            collapse_skeleton=pkg.collapse_skeleton,
            emit_centroid=pkg.emit_centroid,
            source=pkg.source,
            mask_output=getattr(self.layer, "mask_output", "mask"),
            polygon_epsilon=getattr(self.layer, "polygon_epsilon", 0.01),
        )
        _prov_start = datetime.now()
        with writer:
            for outputs in self.predict_streaming(
                provider, progress_callback=progress_callback
            ):
                writer.write(outputs)
            # Attach provenance before the context exit finalizes the .slp, so
            # the streamed output carries lineage like the in-memory path (#583).
            _prov_end = datetime.now()
            writer.provenance = self._build_inference_provenance(
                source=source,
                start_time=_prov_start,
                end_time=_prov_end,
                n_frames=writer.frame_count,
                inference_params={"batch_size": self.batch_size},
            )
        # Post-run summary (#610). The streaming path drops per-frame objects to
        # keep memory O(window), so report frames / throughput only.
        self._log_inference_summary(
            n_frames=writer.frame_count,
            elapsed_s=(_prov_end - _prov_start).total_seconds(),
            output=path,
        )
        return path

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _batch_iter(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Run ``layer.predict`` + ``FilterPipeline`` per provider batch."""
        import inspect

        try:
            sig = inspect.signature(self.layer.predict)
            layer_accepts_instances = "instances" in sig.parameters
        except (TypeError, ValueError):  # pragma: no cover — non-introspectable
            layer_accepts_instances = False

        pipeline = self.filter_pipeline
        total = _safe_num_frames(provider)
        frames_done = 0
        for batch in provider:
            kwargs: dict = {}
            if batch.instances is not None and layer_accepts_instances:
                kwargs["instances"] = (
                    batch.instances
                    if isinstance(batch.instances, torch.Tensor)
                    else torch.from_numpy(batch.instances)
                )
            outputs = self.layer.predict(batch.images, **kwargs)
            outputs = pipeline(outputs)
            outputs = self._stamp_metadata(outputs, batch)
            yield outputs
            if progress_callback is not None:
                frames_done += int(batch.images.shape[0])
                progress_callback(frames_done, total)

    # ──────────────────────────────────────────────────────────────────
    # Pipelined bottom-up: GPU stage in main proc, CPU grouping in pool
    # ──────────────────────────────────────────────────────────────────

    def _can_pipeline(self) -> bool:
        """``True`` iff ``layer`` is a :class:`BottomUpLayer` (not multiclass)."""
        from sleap_nn.inference.layers.bottomup import BottomUpLayer

        return isinstance(self.layer, BottomUpLayer)

    def _predict_streaming_pipelined(
        self,
        provider: Provider,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Outputs]:
        """Stream ``Outputs`` with the CPU grouping stage in a worker pool."""
        from sleap_nn.inference.streaming import PafGroupingPool

        pipeline = self.filter_pipeline
        layer = self.layer
        params = layer.grouping_params()
        total = _safe_num_frames(provider)

        # Bound the number of in-flight batches so memory stays O(window),
        # not O(whole video). Submitting every batch before draining (the old
        # behavior) kept all ScoredBatch payloads + cached batches resident at
        # once (#583). Keep a small multiple of n_workers in flight so workers
        # stay fed; drain the OLDEST completed result (FIFO) before submitting
        # more, preserving submission-order output.
        max_in_flight = max(2 * self.paf_workers, self.paf_workers + 1)
        meta: dict[int, Any] = {}
        frames_done = 0
        with PafGroupingPool(
            n_workers=self.paf_workers, grouping_params=params
        ) as pool:
            for ordinal, batch in enumerate(provider):
                x, info = layer.preprocess(batch.images)
                raw = layer.backend(x)
                scored = layer._score_pafs_on_gpu(raw, info)
                pool.submit(ordinal, scored)
                meta[ordinal] = batch
                while len(pool) >= max_in_flight:
                    done_ordinal, outputs = pool.drain_one()
                    done_batch = meta.pop(done_ordinal)
                    outputs = pipeline(outputs)
                    outputs = self._stamp_metadata(outputs, done_batch)
                    yield outputs
                    if progress_callback is not None:
                        frames_done += int(done_batch.images.shape[0])
                        progress_callback(frames_done, total)
            # Drain the remaining in-flight batches.
            while True:
                result = pool.drain_one()
                if result is None:
                    break
                done_ordinal, outputs = result
                done_batch = meta.pop(done_ordinal)
                outputs = pipeline(outputs)
                outputs = self._stamp_metadata(outputs, done_batch)
                yield outputs
                if progress_callback is not None:
                    frames_done += int(done_batch.images.shape[0])
                    progress_callback(frames_done, total)

    @staticmethod
    def _stamp_metadata(outputs: Outputs, batch: Any) -> Outputs:
        """Attach ``frame_indices`` / ``video_indices`` from the batch."""
        kwargs: dict = {}
        if batch.frame_indices is not None and outputs.frame_indices is None:
            kwargs["frame_indices"] = (
                batch.frame_indices
                if isinstance(batch.frame_indices, torch.Tensor)
                else torch.from_numpy(np.asarray(batch.frame_indices))
            )
        if batch.video_indices is not None and outputs.video_indices is None:
            kwargs["video_indices"] = (
                batch.video_indices
                if isinstance(batch.video_indices, torch.Tensor)
                else torch.from_numpy(np.asarray(batch.video_indices))
            )
        if not kwargs:
            return outputs
        return attrs.evolve(outputs, **kwargs)

    def to_labels(
        self,
        outputs_list: List[Outputs],
        videos: Optional[List["sio.Video"]] = None,
    ) -> "sio.Labels":
        """Concatenate per-batch ``Outputs`` into a single ``sio.Labels``."""
        import sleap_io as sio

        skeleton = self.skeleton
        pkg = self._resolve_centroid_packaging()
        tracks = self._multiclass_tracks()
        videos = list(videos) if videos else [None]
        # When a standalone centroid model collapses to a 1-node 'centroid'
        # skeleton, that is the skeleton attached to emitted instances and to
        # Labels.skeletons; otherwise the original training skeleton is kept.
        out_skeleton = (
            pkg.collapse_skeleton if pkg.collapse_skeleton is not None else skeleton
        )
        # Segmentation mask output representation (read off the layer; identity
        # defaults for non-segmentation layers).
        mask_output = getattr(self.layer, "mask_output", "mask")
        polygon_epsilon = getattr(self.layer, "polygon_epsilon", 0.01)
        all_lf: list = []
        used_tracks: list = []
        seen_track_ids: set = set()
        for outputs in outputs_list:
            sub = outputs.to_labels(
                skeleton=skeleton,
                videos=videos,
                anchor_ind=pkg.anchor_ind,
                tracks=tracks,
                collapse_skeleton=pkg.collapse_skeleton,
                emit_centroid=pkg.emit_centroid,
                source=pkg.source,
                mask_output=mask_output,
                polygon_epsilon=polygon_epsilon,
            )
            all_lf.extend(sub.labeled_frames)
            for trk in sub.tracks:
                if id(trk) not in seen_track_ids:
                    seen_track_ids.add(id(trk))
                    used_tracks.append(trk)
        valid_videos = [v for v in videos if v is not None]
        labels = sio.Labels(
            labeled_frames=all_lf,
            videos=valid_videos,
            skeletons=[out_skeleton],
        )
        if used_tracks:
            labels.tracks = used_tracks
        return labels

    def _multiclass_tracks(self) -> Optional[list["sio.Track"]]:
        """Build the ``sio.Track`` registry for multi-class identity packaging.

        Reads ``class_names`` off the (possibly composed) multi-class layer —
        populated at build time from the training config — and constructs one
        ``sio.Track`` per class, ordered by class index. Returns ``None`` for
        non-multiclass layers. Matches legacy ``predictors.py`` track
        construction (TopDownMultiClass:3808-3811, BottomUpMultiClass:2966).
        """
        import sleap_io as sio

        class_names = getattr(self.layer, "class_names", None)
        if not class_names:
            return None
        return [sio.Track(name=str(name)) for name in class_names]

    def _packaging_anchor_ind(self) -> Optional[int]:
        """Anchor-node slot for centroid-only output packaging."""
        from sleap_nn.inference.layers.exported import ExportedCentroidLayer

        if isinstance(self.layer, (CentroidLayer, ExportedCentroidLayer)):
            return self.layer.anchor_ind
        return None

    def _is_centroid_only_layer(self) -> bool:
        """``True`` iff ``layer`` is a standalone centroid layer."""
        from sleap_nn.inference.layers.exported import ExportedCentroidLayer

        return isinstance(self.layer, (CentroidLayer, ExportedCentroidLayer))

    def _is_segmentation_layer(self) -> bool:
        """``True`` iff ``layer`` is a mask-producing instance-segmentation layer.

        Covers both bottom-up (:class:`SegmentationLayer`) and top-down
        (:class:`TopDownSegmentationLayer`). Gates tracking/no-skeleton/mask-count
        behavior — a top-down seg layer emits ``pred_masks`` just like bottom-up.
        """
        return isinstance(self.layer, (SegmentationLayer, TopDownSegmentationLayer))

    def _resolve_centroid_packaging(self) -> _CentroidPackaging:
        """Resolve the single-source centroid output-packaging decision.

        For a centroid-only layer trained on a MULTI-node skeleton, the output
        collapses to a 1-node 'centroid' skeleton (``sio.get_centroid_skeleton()``);
        a genuinely 1-node model is left as-is (``collapse_skeleton=None``). The
        ``source`` tag mirrors the #586 anchor-fallback semantics. For
        non-centroid layers, emission is forced to ``"instance"`` and no
        collapse occurs.
        """
        from sleap_nn.inference.centroid_convert import centroid_source_for_anchor

        if not self._is_centroid_only_layer():
            return _CentroidPackaging(
                collapse_skeleton=None,
                anchor_ind=None,
                emit_centroid="instance",
                source=centroid_source_for_anchor(None),
            )
        anchor_ind = self._packaging_anchor_ind()
        node_names = (
            list(self.skeleton.node_names) if self.skeleton is not None else None
        )
        source = centroid_source_for_anchor(anchor_ind, node_names)
        collapse_skeleton = None
        if self.skeleton is not None and len(self.skeleton.nodes) > 1:
            import sleap_io as sio

            collapse_skeleton = sio.get_centroid_skeleton()
        return _CentroidPackaging(
            collapse_skeleton=collapse_skeleton,
            anchor_ind=anchor_ind,
            emit_centroid=self.emit_centroid,
            source=source,
        )

    # ──────────────────────────────────────────────────────────────────
    # Prediction-time postprocess overrides
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_postprocess_targets(layer: Any) -> list:
        """Return all sub-layers that own a ``postprocess_config``."""
        from sleap_nn.inference.layers.topdown import TopDownLayer

        if isinstance(layer, TopDownLayer):
            targets = [layer.centroid_layer, layer.centered_instance_layer]
        elif hasattr(layer, "postprocess_config"):
            targets = [layer]
        else:
            targets = []
        return targets

    @contextmanager
    def _postprocess_overrides(
        self,
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
        return_pafs: Optional[bool] = None,
        return_paf_graph: Optional[bool] = None,
        return_class_maps: Optional[bool] = None,
        return_class_vectors: Optional[bool] = None,
    ):
        """Context manager that temporarily overrides postprocess configs.

        For top-down layers, ``centroid_threshold`` applies to the centroid
        stage and ``keypoint_threshold`` to the centered-instance stage.
        ``peak_threshold`` sets both when the per-stage kwargs aren't given.
        """
        from sleap_nn.inference.layers.topdown import TopDownLayer

        has_any = any(
            v is not None
            for v in (
                peak_threshold,
                centroid_threshold,
                keypoint_threshold,
                max_instances,
                integral_refinement,
                integral_patch_size,
                return_confmaps,
                return_crops,
                return_pafs,
                return_paf_graph,
                return_class_maps,
                return_class_vectors,
            )
        )
        if not has_any:
            yield
            return

        saved: list[tuple[Any, PostprocessConfig]] = []
        saved_return_crops: Optional[bool] = None

        try:
            targets = self._collect_postprocess_targets(self.layer)

            for target in targets:
                old_cfg = target.postprocess_config
                saved.append((target, old_cfg))

                overrides: dict = {}

                # Threshold routing for top-down. Use explicit None checks so an
                # explicit 0.0 override ("accept all peaks") is honored rather
                # than swallowed by a falsy `or` (#584).
                if isinstance(self.layer, TopDownLayer):
                    is_centroid = target is self.layer.centroid_layer
                    if is_centroid:
                        t = (
                            centroid_threshold
                            if centroid_threshold is not None
                            else peak_threshold
                        )
                    else:
                        t = (
                            keypoint_threshold
                            if keypoint_threshold is not None
                            else peak_threshold
                        )
                else:
                    t = peak_threshold

                if t is not None:
                    overrides["peak_threshold"] = t
                if max_instances is not None and hasattr(old_cfg, "max_instances"):
                    overrides["max_instances"] = max_instances
                if integral_refinement is not None:
                    overrides["refinement"] = integral_refinement
                if integral_patch_size is not None:
                    overrides["integral_patch_size"] = integral_patch_size
                if return_confmaps is not None:
                    overrides["return_confmaps"] = return_confmaps
                # The remaining intermediate-tensor toggles all live on
                # PostprocessConfig; guard with hasattr defensively (#583).
                for _name, _val in (
                    ("return_pafs", return_pafs),
                    ("return_paf_graph", return_paf_graph),
                    ("return_class_maps", return_class_maps),
                    ("return_class_vectors", return_class_vectors),
                ):
                    if _val is not None and hasattr(old_cfg, _name):
                        overrides[_name] = _val

                if overrides:
                    target.postprocess_config = attrs.evolve(old_cfg, **overrides)

            # return_crops lives on TopDownLayer, not on postprocess_config
            if return_crops is not None and isinstance(self.layer, TopDownLayer):
                saved_return_crops = self.layer.return_crops
                self.layer.return_crops = return_crops

            yield
        finally:
            for target, old_cfg in saved:
                target.postprocess_config = old_cfg
            if saved_return_crops is not None:
                self.layer.return_crops = saved_return_crops
