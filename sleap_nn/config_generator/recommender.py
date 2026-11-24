"""Pipeline and parameter recommendation logic.

This module provides intelligent recommendations for training configuration
based on dataset statistics extracted from SLP files.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType

# Type aliases for configuration options
PipelineType = Literal[
    "single_instance",
    "centroid",
    "centroid_only",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
    "bbox_centernet",
    "bbox_ssd",
    "bbox_detr",
    "topdown_bbox_centernet",
    "topdown_bbox_ssd",
    "topdown_bbox_detr",
]

#: The bounding-box detector pipelines (single model, boxes only).
BBOX_DETECTORS = ("bbox_centernet", "bbox_ssd", "bbox_detr")

# Top-down pipelines whose first stage is a bounding-box detector instead of a
# centroid model, mapped to that detector's pipeline. The second stage is a
# centered-instance model with box-scaled crops (``crop_mode: bbox``).
TOPDOWN_BBOX_DETECTORS = {
    "topdown_bbox_centernet": "bbox_centernet",
    "topdown_bbox_ssd": "bbox_ssd",
    "topdown_bbox_detr": "bbox_detr",
}

#: Touching-frame rate (GT boxes with IoU > 0) above which ``bbox_detr`` is
#: worth suggesting: it predicts a set of boxes with no NMS, so touching animals
#: are not suppressed as duplicates.
DETR_TOUCH_FREQUENCY = 0.05
#: Share of the GPU a recommended ``bbox_detr`` training run may use.
DETR_GPU_HEADROOM = 0.8

BackboneType = Literal[
    "unet_medium_rf",
    "unet_large_rf",
    "convnext_tiny",
    "convnext_small",
    "swint_tiny",
    "swint_small",
]


@dataclass
class PipelineRecommendation:
    """Recommendation for which pipeline to use.

    Attributes:
        recommended: The recommended pipeline type.
        reason: Human-readable explanation for the recommendation.
        alternatives: List of alternative pipeline types.
        warnings: List of warning messages.
        requires_second_model: Whether top-down requires a second model.
        second_model_type: The type of the second model (if required).
    """

    recommended: PipelineType
    reason: str
    alternatives: List[PipelineType] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_second_model: bool = False
    second_model_type: Optional[PipelineType] = None


@dataclass
class ConfigRecommendation:
    """Complete configuration recommendation.

    Attributes:
        pipeline: Pipeline recommendation.
        backbone: Recommended backbone architecture.
        backbone_reason: Explanation for backbone choice.
        sigma: Recommended sigma value for confidence maps.
        sigma_reason: Explanation for sigma choice.
        input_scale: Recommended input scaling factor.
        scale_reason: Explanation for scale choice.
        batch_size: Recommended batch size.
        batch_reason: Explanation for batch size choice.
        rotation_range: Recommended rotation augmentation range (min, max).
        rotation_reason: Explanation for rotation choice.
        crop_size: Recommended crop size for centered_instance models.
        anchor_part: Recommended anchor part for top-down models.
    """

    pipeline: PipelineRecommendation
    backbone: BackboneType
    backbone_reason: str
    sigma: float
    sigma_reason: str
    input_scale: float
    scale_reason: str
    batch_size: int
    batch_reason: str
    rotation_range: Tuple[float, float]
    rotation_reason: str
    crop_size: Optional[int] = None
    anchor_part: Optional[str] = None


def detect_gpu_memory_gb() -> Optional[float]:
    """Total memory of the first visible CUDA GPU in GB, or ``None`` without one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return None


def detr_fits_gpu(stats: DatasetStats, gpu_memory_gb: Optional[float]) -> bool:
    """Whether the default ``bbox_detr`` (DINOv3 ViT-S, batch 4) trains on the GPU.

    Sized at the token-width resolution of the dataset
    (``recommend_detr_resolution``) with ``DETR_GPU_HEADROOM`` of ``gpu_memory_gb``.
    Also needs ``transformers`` (``pip install sleap-nn[backbones]``) for the
    DINOv3 weights.

    Args:
        stats: DatasetStats from analyze_slp().
        gpu_memory_gb: GPU memory in GB; ``None`` (unknown) never fits.

    Returns:
        True when the run is expected to fit.
    """
    import importlib.util

    from sleap_nn.config_generator.memory import estimate_detr_memory

    if gpu_memory_gb is None or importlib.util.find_spec("transformers") is None:
        return False
    res = detr_resolution_for(stats)
    mem = estimate_detr_memory(
        stats, "dinov3_s", 4, res.max_height, res.max_width, num_queries=100
    )
    return mem.total_gpu_gb <= DETR_GPU_HEADROOM * gpu_memory_gb


def detr_bbox_padding(stats: DatasetStats) -> float:
    """Padding (px) of ``bbox_detr`` training boxes derived from poses.

    Keypoints sit inside the animal's outline, so the tight box of the visible
    nodes is a little small; pad it by 5% of the median animal size on every side.
    """
    median = stats.median_bbox_size or stats.avg_bbox_size
    return float(round(0.05 * median))


def detr_resolution_for(stats: DatasetStats, bbox_padding: float = 0.0):
    """The token-width input size (``recommend_detr_resolution``) of a dataset.

    Args:
        stats: DatasetStats from analyze_slp().
        bbox_padding: Pixels added on every side of the training boxes.

    Returns:
        A ``DetrResolution``.
    """
    from sleap_nn.config_generator.architecture_estimates import (
        recommend_detr_resolution,
    )

    small = stats.small_bbox_size or stats.avg_bbox_size
    median = stats.median_bbox_size or stats.avg_bbox_size
    return recommend_detr_resolution(
        stats.max_height,
        stats.max_width,
        small + 2 * bbox_padding,
        median + 2 * bbox_padding,
    )


def _topdown_recommendation(
    stats: DatasetStats,
    reason: str,
    alternatives: List[PipelineType],
    warnings: List[str],
    gpu_memory_gb: Optional[float],
) -> PipelineRecommendation:
    """A multi-animal top-down recommendation: centroid, or ``bbox_detr`` stage 1.

    ``topdown_bbox_detr`` is recommended when it fits ``gpu_memory_gb``; it is the
    better first stage when animals touch, since it has no NMS to merge them.
    Otherwise the centroid pair is recommended and ``topdown_bbox_detr`` is listed
    as an alternative.
    """
    touching = stats.touch_frequency
    touch_note = (
        f"animals touch in {touching:.0%} of frames and bbox_detr has no NMS to "
        "suppress one of them"
    )
    if detr_fits_gpu(stats, gpu_memory_gb):
        why = "Multiple animals and the GPU fits a DETR box detector"
        if touching >= DETR_TOUCH_FREQUENCY:
            why += f"; {touch_note}"
        return PipelineRecommendation(
            recommended="topdown_bbox_detr",
            reason=f"{why} - top-down with a bbox_detr first stage",
            alternatives=["centroid"] + alternatives,
            warnings=warnings
            + ["You'll need to train a centered_instance model as well"],
            requires_second_model=True,
            second_model_type="centered_instance",
        )
    if touching >= DETR_TOUCH_FREQUENCY:
        warnings = warnings + [
            f"{touch_note[0].upper()}{touch_note[1:]}: consider "
            "topdown_bbox_detr if your GPU has the memory"
        ]
    return PipelineRecommendation(
        recommended="centroid",
        reason=reason,
        alternatives=alternatives + ["topdown_bbox_detr"],
        warnings=warnings,
        requires_second_model=True,
        second_model_type="centered_instance",
    )


def recommend_pipeline(
    stats: DatasetStats, gpu_memory_gb: Optional[float] = None
) -> PipelineRecommendation:
    """Recommend the best pipeline based on dataset statistics.

    Decision tree:
    1. Single animal per frame -> single_instance
    2. Multiple small animals (<20% frame area) -> top-down (centroid, or a
       bbox_detr first stage when it fits ``gpu_memory_gb``)
    3. Multiple large animals with edges -> bottomup
    4. Multiple large animals without edges -> top-down (as in 2)

    Note: Multi-class models are available as alternatives when tracks exist,
    but are not recommended by default.

    Args:
        stats: DatasetStats from analyze_slp().
        gpu_memory_gb: GPU memory in GB (``detect_gpu_memory_gb``). ``None``
            (default) never recommends ``bbox_detr``, so the recommendation does
            not depend on the machine.

    Returns:
        PipelineRecommendation with suggested pipeline and reasoning.

    Example:
        >>> stats = analyze_slp("labels.slp")
        >>> rec = recommend_pipeline(stats)
        >>> print(f"Use {rec.recommended}: {rec.reason}")
    """
    warnings: List[str] = []

    # Single instance case
    if stats.is_single_instance:
        return PipelineRecommendation(
            recommended="single_instance",
            reason="Only one animal detected per frame",
            alternatives=["centered_instance"],
            warnings=[],
            requires_second_model=False,
        )

    # Multi-instance with a single-node skeleton: there are no keypoints to
    # crop-and-refine, so the only meaningful detection target is the centroid
    # itself. Recommend a STANDALONE centroid model (one config, no second
    # centered-instance stage) rather than the paired top-down bundle.
    if stats.num_nodes == 1:
        single_node_alternatives: List[PipelineType] = ["bottomup"]
        if stats.has_identity:
            single_node_alternatives.append("multi_class_bottomup")
        return PipelineRecommendation(
            recommended="centroid_only",
            reason="Single-node skeleton - a standalone centroid model detects "
            "the one point per animal directly (no second model needed)",
            alternatives=single_node_alternatives,
            warnings=[],
            requires_second_model=False,
        )

    # Multi-instance: small animals -> top-down
    if stats.animal_to_frame_ratio < 0.20:
        # Small animals - recommend top-down
        alternatives: List[PipelineType] = ["bottomup"]
        if stats.has_identity:
            alternatives.append("multi_class_topdown")
        return _topdown_recommendation(
            stats,
            reason=f"Animals are small relative to frame (~{int(stats.animal_to_frame_ratio * 100)}% area) - "
            "top-down approach recommended",
            alternatives=alternatives,
            warnings=["You'll need to train a centered_instance model as well"],
            gpu_memory_gb=gpu_memory_gb,
        )
    else:
        # Large animals
        if stats.num_edges == 0:
            # No edges - can't use bottom-up
            warnings.append("No edges in skeleton - bottom-up requires edges for PAFs")
            alternatives = []
            if stats.has_identity:
                alternatives.append("multi_class_topdown")
            return _topdown_recommendation(
                stats,
                reason="No skeleton edges available for bottom-up",
                alternatives=alternatives,
                warnings=warnings,
                gpu_memory_gb=gpu_memory_gb,
            )
        # Has edges - recommend bottom-up
        alternatives = ["centroid", "topdown_bbox_detr"]
        if stats.has_identity:
            alternatives.append("multi_class_bottomup")
        return PipelineRecommendation(
            recommended="bottomup",
            reason=f"Larger animals (~{int(stats.animal_to_frame_ratio * 100)}% of frame) - "
            "bottom-up handles occlusions well",
            alternatives=alternatives,
            warnings=[],
            requires_second_model=False,
        )


def _recommend_backbone(stats: DatasetStats) -> Tuple[BackboneType, str]:
    """Recommend backbone architecture based on dataset statistics.

    Args:
        stats: DatasetStats from analyze_slp().

    Returns:
        Tuple of (backbone_type, reason).
    """
    if stats.max_bbox_size > 200 or stats.max_dimension > 1024:
        return (
            "unet_large_rf",
            "Large animals/images need larger receptive field (max_stride=32)",
        )
    else:
        return (
            "unet_medium_rf",
            "Standard receptive field sufficient for this data (max_stride=16)",
        )


def _recommend_sigma(stats: DatasetStats, pipeline: PipelineType) -> Tuple[float, str]:
    """Recommend sigma value based on dataset and pipeline.

    Args:
        stats: DatasetStats from analyze_slp().
        pipeline: The pipeline type being used.

    Returns:
        Tuple of (sigma, reason).
    """
    if pipeline == "centroid_only":
        return (2.5, "Tighter sigma for precise standalone centroid localization")
    elif pipeline == "bottomup":
        return (2.5, "Tighter sigma for multi-instance disambiguation")
    elif stats.max_bbox_size < 50:
        return (2.5, "Small animals need precise localization")
    elif stats.max_bbox_size < 150:
        return (5.0, "Default sigma for medium-sized animals")
    else:
        return (7.5, "Larger sigma for large animals (easier to learn)")


def _recommend_scale(stats: DatasetStats) -> Tuple[float, str]:
    """Recommend input scaling factor based on image size.

    Args:
        stats: DatasetStats from analyze_slp().

    Returns:
        Tuple of (scale, reason).
    """
    if stats.max_dimension > 2048:
        return (0.25, "Very large images - scaling required for memory")
    elif stats.max_dimension > 1024:
        return (0.5, "Large images - scaling recommended")
    else:
        return (1.0, "Image size suitable for full resolution")


def _recommend_batch_size(
    stats: DatasetStats, backbone: BackboneType
) -> Tuple[int, str]:
    """Recommend batch size based on estimated memory usage.

    Args:
        stats: DatasetStats from analyze_slp().
        backbone: The backbone architecture.

    Returns:
        Tuple of (batch_size, reason).
    """
    # Rough memory estimation
    pixels = stats.max_height * stats.max_width

    # ConvNeXt and SwinT use more memory
    if "convnext" in backbone or "swint" in backbone:
        if pixels > 1_000_000:
            return (2, "Large images with transformer backbone - reduced batch size")
        else:
            return (4, "Standard batch size for transformer backbone")

    # UNet is more memory efficient
    if pixels > 2_000_000:
        return (2, "Very large images - reduced batch size for memory")
    elif pixels > 1_000_000:
        return (4, "Large images - moderate batch size")
    else:
        return (8, "Moderate image size allows larger batch")


def _recommend_rotation(view_type: ViewType) -> Tuple[Tuple[float, float], str]:
    """Recommend rotation augmentation based on camera view.

    Args:
        view_type: The camera view orientation.

    Returns:
        Tuple of ((min_rotation, max_rotation), reason).
    """
    if view_type == ViewType.TOP:
        return ((-180.0, 180.0), "Top-view: all orientations are valid")
    elif view_type == ViewType.SIDE:
        return ((-15.0, 15.0), "Side-view: limited rotation (upside-down unnatural)")
    else:
        return (
            (-15.0, 15.0),
            "Default conservative rotation (specify view_type for better defaults)",
        )


def recommend_config(
    stats: DatasetStats,
    view_type: ViewType = ViewType.UNKNOWN,
    gpu_memory_gb: Optional[float] = None,
) -> ConfigRecommendation:
    """Generate complete configuration recommendation.

    Args:
        stats: DatasetStats from analyze_slp().
        view_type: Camera view (side, top, or unknown).
        gpu_memory_gb: GPU memory in GB, for the ``bbox_detr`` recommendation
            (see ``recommend_pipeline``).

    Returns:
        ConfigRecommendation with all parameter suggestions.

    Example:
        >>> stats = analyze_slp("labels.slp")
        >>> rec = recommend_config(stats, ViewType.TOP)
        >>> print(f"Pipeline: {rec.pipeline.recommended}")
        >>> print(f"Backbone: {rec.backbone}")
    """
    pipeline = recommend_pipeline(stats, gpu_memory_gb)
    backbone, backbone_reason = _recommend_backbone(stats)
    sigma, sigma_reason = _recommend_sigma(stats, pipeline.recommended)
    input_scale, scale_reason = _recommend_scale(stats)
    batch_size, batch_reason = _recommend_batch_size(stats, backbone)
    rotation_range, rotation_reason = _recommend_rotation(view_type)

    # Crop size for centered instance
    crop_size = None
    if pipeline.recommended in ["centered_instance", "multi_class_topdown"] or (
        pipeline.requires_second_model
        and pipeline.second_model_type == "centered_instance"
    ):
        # Rough estimate: 1.5x max bbox, rounded up to stride
        raw_crop = int(stats.max_bbox_size * 1.5)
        max_stride = 32 if "large_rf" in backbone else 16
        crop_size = ((raw_crop + max_stride - 1) // max_stride) * max_stride
        crop_size = max(crop_size, 100)  # Minimum crop size

    return ConfigRecommendation(
        pipeline=pipeline,
        backbone=backbone,
        backbone_reason=backbone_reason,
        sigma=sigma,
        sigma_reason=sigma_reason,
        input_scale=input_scale,
        scale_reason=scale_reason,
        batch_size=batch_size,
        batch_reason=batch_reason,
        rotation_range=rotation_range,
        rotation_reason=rotation_reason,
        crop_size=crop_size,
        anchor_part=None,  # User should specify based on skeleton
    )
