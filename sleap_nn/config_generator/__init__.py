"""Config generator for SLEAP-NN training configurations.

This module provides tools for automatically generating training configurations
from SLP label files with sensible defaults based on data analysis.

Quick Start
-----------
One-liner to generate config::

    from sleap_nn.config_generator import generate_config
    generate_config("labels.slp", "config.yaml")

With customization::

    from sleap_nn.config_generator import ConfigGenerator
    config = (
        ConfigGenerator.from_slp("labels.slp")
        .auto(view="top")
        .batch_size(8)
        .save("config.yaml")
    )

Just analyze the data::

    from sleap_nn.config_generator import analyze_slp
    stats = analyze_slp("labels.slp")
    print(stats)

Interactive TUI::

    sleap-nn config labels.slp --interactive
"""

from omegaconf import DictConfig

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.generator import ConfigGenerator
from sleap_nn.config_generator.memory import MemoryEstimate, estimate_memory
from sleap_nn.config_generator.recommender import (
    BackboneType,
    ConfigRecommendation,
    PipelineRecommendation,
    PipelineType,
    recommend_config,
    recommend_pipeline,
)

__all__ = [
    # Core classes
    "ConfigGenerator",
    "DatasetStats",
    "MemoryEstimate",
    "PipelineRecommendation",
    "ConfigRecommendation",
    # Type aliases
    "PipelineType",
    "BackboneType",
    "ViewType",
    # Functions
    "analyze_slp",
    "recommend_pipeline",
    "recommend_config",
    "estimate_memory",
    "generate_config",
]


def generate_config(
    slp_path: str,
    output_path: str = None,
    *,
    view: str = None,
    pipeline: str = None,
    backbone: str = None,
    batch_size: int = None,
    **kwargs,
) -> DictConfig:
    """Generate a training configuration from an SLP file.

    This is a convenience function for quick config generation.
    For more control, use the ConfigGenerator class directly.

    Args:
        slp_path: Path to the .slp label file.
        output_path: Optional path to save YAML config.
        view: Camera view type ("side" or "top") for augmentation defaults.
        pipeline: Override auto-detected pipeline type.
        backbone: Override auto-detected backbone.
        batch_size: Override auto-detected batch size.
        **kwargs: Additional parameters to override.

    Returns:
        Training configuration as OmegaConf DictConfig.

    Examples:
        Auto-generate and save::

            generate_config("labels.slp", "config.yaml")

        Auto-generate with view hint::

            config = generate_config("labels.slp", view="top")

        With overrides::

            config = generate_config(
                "labels.slp",
                "config.yaml",
                pipeline="bottomup",
                batch_size=8
            )
    """
    gen = ConfigGenerator.from_slp(slp_path).auto(view=view)

    # Apply overrides
    if pipeline:
        gen.pipeline(pipeline)
    if backbone:
        gen.backbone(backbone)
    if batch_size:
        gen.batch_size(batch_size)

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if value is not None and hasattr(gen, key):
            getattr(gen, key)(value)

    config = gen.build()

    if output_path:
        gen.save(output_path)

    return config
