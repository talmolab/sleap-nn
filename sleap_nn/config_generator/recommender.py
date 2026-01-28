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
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
]

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


def recommend_pipeline(stats: DatasetStats) -> PipelineRecommendation:
    """Recommend the best pipeline based on dataset statistics.

    Decision tree:
    1. Single animal per frame -> single_instance
    2. Multiple animals with tracks -> multi_class variants
    3. Multiple small animals (<15% frame area) -> top-down (centroid)
    4. Multiple large animals with edges -> bottomup
    5. Multiple large animals without edges -> top-down (centroid)

    Args:
        stats: DatasetStats from analyze_slp().

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

    # Multi-instance with identity
    if stats.has_identity:
        if stats.animal_to_frame_ratio < 0.15:
            return PipelineRecommendation(
                recommended="multi_class_topdown",
                reason="Multiple small animals with track IDs",
                alternatives=["multi_class_bottomup"],
                warnings=[
                    "Top-down requires training two models (centroid + centered_instance)"
                ],
                requires_second_model=True,
                second_model_type="centered_instance",
            )
        else:
            if stats.num_edges == 0:
                warnings.append("No edges in skeleton - cannot use bottom-up")
                return PipelineRecommendation(
                    recommended="multi_class_topdown",
                    reason="No skeleton edges available for bottom-up PAFs",
                    alternatives=[],
                    warnings=warnings,
                    requires_second_model=True,
                    second_model_type="centered_instance",
                )
            return PipelineRecommendation(
                recommended="multi_class_bottomup",
                reason="Multiple larger animals with track IDs",
                alternatives=["multi_class_topdown"],
                warnings=[],
                requires_second_model=False,
            )

    # Multi-instance without identity
    if stats.animal_to_frame_ratio < 0.15:
        return PipelineRecommendation(
            recommended="centroid",
            reason="Animals are small relative to frame (<15% area) - "
            "top-down approach recommended",
            alternatives=["bottomup"],
            warnings=["You'll need to train a centered_instance model as well"],
            requires_second_model=True,
            second_model_type="centered_instance",
        )
    else:
        if stats.num_edges == 0:
            warnings.append("No edges in skeleton - bottom-up requires edges for PAFs")
            return PipelineRecommendation(
                recommended="centroid",
                reason="No skeleton edges available for bottom-up",
                alternatives=[],
                warnings=warnings,
                requires_second_model=True,
                second_model_type="centered_instance",
            )
        return PipelineRecommendation(
            recommended="bottomup",
            reason="Multiple larger animals - bottom-up handles occlusions well",
            alternatives=["centroid"],
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
    if pipeline == "bottomup":
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
    stats: DatasetStats, view_type: ViewType = ViewType.UNKNOWN
) -> ConfigRecommendation:
    """Generate complete configuration recommendation.

    Args:
        stats: DatasetStats from analyze_slp().
        view_type: Camera view (side, top, or unknown).

    Returns:
        ConfigRecommendation with all parameter suggestions.

    Example:
        >>> stats = analyze_slp("labels.slp")
        >>> rec = recommend_config(stats, ViewType.TOP)
        >>> print(f"Pipeline: {rec.pipeline.recommended}")
        >>> print(f"Backbone: {rec.backbone}")
    """
    pipeline = recommend_pipeline(stats)
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
