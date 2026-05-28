"""``Predictor`` — high-level orchestrator for the inference stack.

Composes an :class:`InferenceLayer` (or composed layer like
:class:`TopDownLayer`) with a :class:`Provider` source and a
:class:`FilterPipeline` post-processor.

Three usage tiers:

* :meth:`predict` — synchronous, returns ``sio.Labels`` (or a list of
  ``Outputs`` if ``make_labels=False``). Loads everything into memory;
  use for short videos / interactive sessions.
* :meth:`predict_streaming` — yields one ``Outputs`` per batch as a
  generator. Memory stays O(tracker_window).
* :meth:`predict_to_file` — disk-streaming write of a ``.slp`` via
  :class:`IncrementalLabelsWriter`. Memory stays O(write_interval).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.filters import FilterConfig, FilterPipeline
from sleap_nn.inference.layers.backends import TorchBackend
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
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.providers import Provider
from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

if TYPE_CHECKING:
    import sleap_io as sio

    from sleap_nn.export.metadata import ExportMetadata


def _safe_len(provider: Any) -> int:
    """Return ``len(provider)`` or ``-1`` if the provider doesn't expose ``__len__``."""
    try:
        return len(provider)
    except TypeError:
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
    """Build a ``CentroidLayer`` that reads centroids from GT (no model forward)."""
    centroid_model = assets.inference_model.centroid_crop
    return CentroidLayer(
        backend=backend,
        output_stride=1,
        max_instances=None,
        max_stride=1,
        anchor_ind=getattr(centroid_model, "anchor_ind", None),
        use_gt_centroids=True,
        preprocess_config=PreprocessConfig(scale=1.0),
        postprocess_config=PostprocessConfig(),
    )


def _build_centered_instance_multiclass_layer(
    instance_model: Any, device: str
) -> CenteredInstanceMultiClassLayer:
    """Wrap a ``TopDownMultiClassFindInstancePeaks`` model in a layer."""
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
    ) -> "Predictor":
        """Build a :class:`Predictor` from one or more checkpoint paths.

        Args:
            model_paths: Directories containing ``training_config.{yaml,json}``
                + ``best.ckpt``. For top-down, pass two paths (centroid +
                centered-instance) in either order.
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
        return cls(**kwargs)

    # ──────────────────────────────────────────────────────────────────
    # Source dispatch: sio.Video / sio.Labels / str / Provider
    # ──────────────────────────────────────────────────────────────────

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
                provider = LabelsProvider(
                    labels=source,
                    batch_size=self.batch_size,
                    **provider_kwargs,
                )
                return provider, None
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            return provider, None

        if isinstance(source, sio.Video):
            provider = VideoProvider(
                video=source,
                batch_size=self.batch_size,
                frames=frames,
                **provider_kwargs,
            )
            return provider, [source]

        if isinstance(source, sio.Labels):
            provider = LabelsProvider(
                labels=source,
                batch_size=self.batch_size,
                **provider_kwargs,
            )
            videos = list(source.videos) if source.videos else None
            return provider, videos

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

        Returns:
            New ``sio.Labels`` with tracks attached.
        """
        out = apply_tracking(labels, tracker_config)
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
        peak_threshold: Optional[float] = None,
        centroid_threshold: Optional[float] = None,
        keypoint_threshold: Optional[float] = None,
        max_instances: Optional[int] = None,
        integral_refinement: Optional[str] = None,
        integral_patch_size: Optional[int] = None,
        return_confmaps: Optional[bool] = None,
        return_crops: Optional[bool] = None,
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
            progress_callback: Optional ``(processed_batches, total_batches)``
                callback invoked after each batch.
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

        Returns:
            ``sio.Labels`` (default) or ``List[Outputs]`` (when
            ``make_labels=False``).
        """
        provider, auto_videos = self._make_provider(source, frames=frames)
        if videos is None:
            videos = auto_videos

        with self._postprocess_overrides(
            peak_threshold=peak_threshold,
            centroid_threshold=centroid_threshold,
            keypoint_threshold=keypoint_threshold,
            max_instances=max_instances,
            integral_refinement=integral_refinement,
            integral_patch_size=integral_patch_size,
            return_confmaps=return_confmaps,
            return_crops=return_crops,
        ):
            outputs_list = list(self._batch_iter(provider, progress_callback))

        if not make_labels:
            if self.tracker_config is not None:
                raise ValueError(
                    "tracker_config requires make_labels=True; the tracker "
                    "operates on sio.PredictedInstance objects."
                )
            return outputs_list
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None:
            raise ValueError(
                "make_labels=True requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via Predictor.from_model_paths() "
                "which sets it automatically from the training config."
            )
        labels = self.to_labels(outputs_list, videos=videos)
        if self.tracker_config is not None:
            labels = apply_tracking(labels, self.tracker_config)
        if clean_empty_frames:
            labels.clean(frames=True, skeletons=False)
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
    ) -> Iterator[Outputs]:
        """Yield one ``Outputs`` per batch from ``source``.

        Args:
            source: ``sio.Video``, ``sio.Labels``, video path string, or
                a pre-built :class:`Provider`.
            frames: Frame indices (only for video sources).
            progress_callback: Optional ``(processed_batches, total_batches)``
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
        """Run inference and stream results to a ``.slp`` file.

        Memory stays O(``write_interval``) — outputs are slimmed and
        converted to LabeledFrames per batch; heavy tensors are dropped
        immediately.

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
            progress_callback: Optional callback per batch.

        Returns:
            The (resolved) destination path string.
        """
        if skeleton is not None:
            self.skeleton = skeleton
        if self.skeleton is None:
            raise ValueError(
                "predict_to_file requires a skeleton. Either pass "
                "`skeleton=...` or build the Predictor via Predictor.from_model_paths() "
                "which sets it automatically from the training config."
            )
        from sleap_nn.inference.writer import IncrementalLabelsWriter

        provider, _ = self._make_provider(source, frames=frames)
        with IncrementalLabelsWriter(
            path=path,
            skeleton=self.skeleton,
            videos=videos,
            write_interval=write_interval,
        ) as writer:
            for outputs in self.predict_streaming(
                provider, progress_callback=progress_callback
            ):
                writer.write(outputs)
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
        total = _safe_len(provider)
        for i, batch in enumerate(provider):
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
                progress_callback(i + 1, total)

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
        total = _safe_len(provider)

        meta: dict[int, Any] = {}
        with PafGroupingPool(
            n_workers=self.paf_workers, grouping_params=params
        ) as pool:
            for ordinal, batch in enumerate(provider):
                x, info = layer.preprocess(batch.images)
                raw = layer.backend(x)
                scored = layer._score_pafs_on_gpu(raw, info)
                pool.submit(ordinal, scored)
                meta[ordinal] = batch
            completed = 0
            for ordinal, outputs in pool.iter_completed():
                outputs = pipeline(outputs)
                outputs = self._stamp_metadata(outputs, meta.pop(ordinal))
                yield outputs
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

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
        anchor_ind = self._packaging_anchor_ind()
        videos = list(videos) if videos else [None]
        all_lf: list = []
        for outputs in outputs_list:
            sub = outputs.to_labels(
                skeleton=skeleton, videos=videos, anchor_ind=anchor_ind
            )
            all_lf.extend(sub.labeled_frames)
        valid_videos = [v for v in videos if v is not None]
        return sio.Labels(
            labeled_frames=all_lf,
            videos=valid_videos,
            skeletons=[skeleton],
        )

    def _packaging_anchor_ind(self) -> Optional[int]:
        """Anchor-node slot for centroid-only output packaging."""
        from sleap_nn.inference.layers.centroid import CentroidLayer

        if isinstance(self.layer, CentroidLayer):
            return self.layer.anchor_ind
        return None

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

                # Threshold routing for top-down
                if isinstance(self.layer, TopDownLayer):
                    is_centroid = target is self.layer.centroid_layer
                    if is_centroid:
                        t = centroid_threshold or peak_threshold
                    else:
                        t = keypoint_threshold or peak_threshold
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
