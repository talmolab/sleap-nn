"""Main ConfigGenerator class for creating training configurations.

This module provides a fluent API for generating sleap-nn training
configurations from SLP files with sensible defaults.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.memory import MemoryEstimate, estimate_memory
from sleap_nn.config_generator.recommender import (
    BackboneType,
    ConfigRecommendation,
    PipelineType,
    recommend_config,
    recommend_pipeline,
)

if TYPE_CHECKING:
    import sleap_io as sio


class ConfigGenerator:
    """Generate sleap-nn training configurations from SLP files.

    Provides a fluent API for generating configurations with sensible
    defaults based on data analysis.

    Examples:
        Quick auto-config (recommended for most users)::

            config = ConfigGenerator.from_slp("labels.slp").auto().build()
            config.save("config.yaml")

        Auto-config with view type hint::

            config = ConfigGenerator.from_slp("labels.slp").auto(view="top").build()

        Customized config::

            config = (
                ConfigGenerator.from_slp("labels.slp")
                .auto()
                .pipeline("bottomup")
                .batch_size(8)
                .sigma(3.0)
                .build()
            )

        Manual config (no auto-fill)::

            config = (
                ConfigGenerator.from_slp("labels.slp")
                .pipeline("single_instance")
                .backbone("unet_medium_rf")
                .batch_size(4)
                .build()
            )

        Get recommendations without building::

            gen = ConfigGenerator.from_slp("labels.slp")
            print(gen.stats)  # Dataset statistics
            print(gen.recommend())  # Full recommendations

        Save directly::

            ConfigGenerator.from_slp("labels.slp").auto().save("config.yaml")
    """

    def __init__(self, slp_path: str):
        """Initialize with path to SLP file.

        Args:
            slp_path: Path to .slp or .pkg.slp file.

        Raises:
            FileNotFoundError: If the SLP file does not exist.
        """
        self.slp_path = Path(slp_path).resolve()
        if not self.slp_path.exists():
            raise FileNotFoundError(f"SLP file not found: {self.slp_path}")

        # Lazily computed
        self._stats: Optional[DatasetStats] = None
        self._recommendation: Optional[ConfigRecommendation] = None

        # Configuration state with defaults
        self._pipeline: Optional[PipelineType] = None
        self._backbone: BackboneType = "unet_medium_rf"
        self._batch_size: int = 4
        self._max_epochs: int = 200
        self._learning_rate: float = 1e-4
        self._input_scale: float = 1.0
        self._sigma: float = 5.0
        self._output_stride: int = 1
        self._max_stride: int = 16
        self._filters: int = 32
        self._filters_rate: float = 2.0
        self._use_augmentations: bool = True
        self._rotation_range: Tuple[float, float] = (-15.0, 15.0)
        self._scale_range: Tuple[float, float] = (0.9, 1.1)
        self._early_stopping: bool = True
        self._early_stopping_patience: int = 10
        self._validation_fraction: float = 0.1
        self._anchor_part: Optional[str] = None
        self._crop_size: Optional[int] = None
        self._view_type: ViewType = ViewType.UNKNOWN
        self._ensure_rgb: bool = False
        self._ensure_grayscale: bool = False

    @classmethod
    def from_slp(cls, path: str) -> "ConfigGenerator":
        """Create a ConfigGenerator from an SLP file path.

        Args:
            path: Path to the .slp file.

        Returns:
            ConfigGenerator instance.

        Example:
            >>> gen = ConfigGenerator.from_slp("labels.slp")
        """
        return cls(path)

    @classmethod
    def from_labels(cls, labels: "sio.Labels") -> "ConfigGenerator":
        """Create a ConfigGenerator from a sleap_io.Labels object.

        Args:
            labels: sleap_io.Labels object.

        Returns:
            ConfigGenerator instance.
        """
        import tempfile

        import sleap_io as sio

        with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
            sio.save_slp(labels, f.name)
            gen = cls(f.name)
            gen._temp_file = f.name
            return gen

    @property
    def stats(self) -> DatasetStats:
        """Get dataset statistics (lazily computed).

        Returns:
            DatasetStats object with extracted statistics.
        """
        if self._stats is None:
            self._stats = analyze_slp(str(self.slp_path))
        return self._stats

    def recommend(self, view: Optional[str] = None) -> ConfigRecommendation:
        """Get configuration recommendations based on data analysis.

        Args:
            view: Camera view type ("side", "top", or None for auto).

        Returns:
            ConfigRecommendation with all parameter suggestions.
        """
        view_type = ViewType(view) if view else self._view_type
        return recommend_config(self.stats, view_type)

    def auto(self, view: Optional[str] = None) -> "ConfigGenerator":
        """Automatically configure all parameters based on data analysis.

        This is the recommended way to get started. It analyzes your data
        and sets sensible defaults for all parameters.

        Args:
            view: Camera view type ("side" or "top"). Affects rotation
                augmentation. If None, uses conservative defaults.

        Returns:
            self for method chaining.

        Example:
            >>> config = ConfigGenerator.from_slp("labels.slp").auto(view="top").build()
        """
        if view:
            self._view_type = ViewType(view)

        rec = self.recommend(view)

        self._pipeline = rec.pipeline.recommended
        self._backbone = rec.backbone
        self._sigma = rec.sigma
        self._input_scale = rec.input_scale
        self._batch_size = rec.batch_size
        self._rotation_range = rec.rotation_range

        if rec.crop_size:
            self._crop_size = rec.crop_size

        # Set backbone-specific parameters
        if "large_rf" in self._backbone:
            self._max_stride = 32
            self._filters = 24
            self._filters_rate = 1.5
        else:
            self._max_stride = 16
            self._filters = 32
            self._filters_rate = 2.0

        # Set channel configuration
        self._ensure_rgb = self.stats.is_rgb
        self._ensure_grayscale = self.stats.is_grayscale

        self._recommendation = rec
        return self

    # Fluent setters for all parameters

    def pipeline(self, pipeline: PipelineType) -> "ConfigGenerator":
        """Set the pipeline type.

        Args:
            pipeline: One of "single_instance", "centroid", "centered_instance",
                "bottomup", "multi_class_bottomup", "multi_class_topdown".

        Returns:
            self for method chaining.
        """
        self._pipeline = pipeline
        return self

    def backbone(self, backbone: BackboneType) -> "ConfigGenerator":
        """Set the backbone architecture.

        Args:
            backbone: One of "unet_medium_rf", "unet_large_rf",
                "convnext_tiny", "convnext_small", "swint_tiny", "swint_small".

        Returns:
            self for method chaining.
        """
        self._backbone = backbone
        # Update related parameters
        if "large_rf" in backbone:
            self._max_stride = 32
            self._filters = 24
            self._filters_rate = 1.5
        elif "unet" in backbone:
            self._max_stride = 16
            self._filters = 32
            self._filters_rate = 2.0
        else:
            self._max_stride = 32  # ConvNeXt/SwinT
        return self

    def batch_size(self, size: int) -> "ConfigGenerator":
        """Set the batch size.

        Args:
            size: Batch size for training.

        Returns:
            self for method chaining.
        """
        self._batch_size = size
        return self

    def max_epochs(self, epochs: int) -> "ConfigGenerator":
        """Set maximum training epochs.

        Args:
            epochs: Maximum number of training epochs.

        Returns:
            self for method chaining.
        """
        self._max_epochs = epochs
        return self

    def learning_rate(self, lr: float) -> "ConfigGenerator":
        """Set the learning rate.

        Args:
            lr: Learning rate for optimizer.

        Returns:
            self for method chaining.
        """
        self._learning_rate = lr
        return self

    def input_scale(self, scale: float) -> "ConfigGenerator":
        """Set input image scaling factor (0.0-1.0).

        Args:
            scale: Input scaling factor.

        Returns:
            self for method chaining.
        """
        self._input_scale = scale
        return self

    def sigma(self, sigma: float) -> "ConfigGenerator":
        """Set confidence map sigma (Gaussian spread in pixels).

        Args:
            sigma: Sigma value for confidence maps.

        Returns:
            self for method chaining.
        """
        self._sigma = sigma
        return self

    def output_stride(self, stride: int) -> "ConfigGenerator":
        """Set output stride (1, 2, 4, or 8).

        Args:
            stride: Output stride for confidence maps.

        Returns:
            self for method chaining.
        """
        self._output_stride = stride
        return self

    def rotation(self, min_deg: float, max_deg: float) -> "ConfigGenerator":
        """Set rotation augmentation range in degrees.

        Args:
            min_deg: Minimum rotation angle.
            max_deg: Maximum rotation angle.

        Returns:
            self for method chaining.
        """
        self._rotation_range = (min_deg, max_deg)
        return self

    def scale_augmentation(
        self, min_scale: float, max_scale: float
    ) -> "ConfigGenerator":
        """Set scale augmentation range.

        Args:
            min_scale: Minimum scale factor.
            max_scale: Maximum scale factor.

        Returns:
            self for method chaining.
        """
        self._scale_range = (min_scale, max_scale)
        return self

    def augmentation(self, enabled: bool) -> "ConfigGenerator":
        """Enable or disable data augmentation.

        Args:
            enabled: Whether to enable augmentation.

        Returns:
            self for method chaining.
        """
        self._use_augmentations = enabled
        return self

    def early_stopping(
        self, enabled: bool = True, patience: int = 10
    ) -> "ConfigGenerator":
        """Configure early stopping.

        Args:
            enabled: Whether to enable early stopping.
            patience: Number of epochs without improvement before stopping.

        Returns:
            self for method chaining.
        """
        self._early_stopping = enabled
        self._early_stopping_patience = patience
        return self

    def validation_fraction(self, fraction: float) -> "ConfigGenerator":
        """Set validation split fraction (0.0-1.0).

        Args:
            fraction: Fraction of data to use for validation.

        Returns:
            self for method chaining.
        """
        self._validation_fraction = fraction
        return self

    def anchor_part(self, part_name: str) -> "ConfigGenerator":
        """Set anchor part for centroid/centered_instance models.

        Args:
            part_name: Name of the anchor body part.

        Returns:
            self for method chaining.
        """
        self._anchor_part = part_name
        return self

    def crop_size(self, size: int) -> "ConfigGenerator":
        """Set crop size for centered_instance models.

        Args:
            size: Crop size in pixels.

        Returns:
            self for method chaining.
        """
        self._crop_size = size
        return self

    def build(self) -> DictConfig:
        """Build the configuration as an OmegaConf DictConfig.

        Returns:
            Complete training configuration ready for sleap-nn.

        Raises:
            ValueError: If required parameters are not set.
        """
        if self._pipeline is None:
            raise ValueError("Pipeline not set. Call .auto() or .pipeline() first.")

        # Build the configuration dict
        config = {
            "data_config": self._build_data_config(),
            "model_config": self._build_model_config(),
            "trainer_config": self._build_trainer_config(),
        }

        return OmegaConf.create(config)

    def _build_data_config(self) -> dict:
        """Build data configuration section."""
        return {
            "train_labels_path": [str(self.slp_path)],
            "val_labels_path": [],
            "validation_fraction": self._validation_fraction,
            "user_instances_only": True,
            "data_pipeline_fw": "torch_dataset",
            "preprocessing": {
                "ensure_rgb": self._ensure_rgb,
                "ensure_grayscale": self._ensure_grayscale,
                "scale": self._input_scale,
                "crop_size": self._crop_size,
            },
            "use_augmentations_train": self._use_augmentations,
            "augmentation_config": {
                "geometric": {
                    "rotation_min": self._rotation_range[0],
                    "rotation_max": self._rotation_range[1],
                    "scale_min": self._scale_range[0],
                    "scale_max": self._scale_range[1],
                    "affine_p": 1.0 if self._use_augmentations else 0.0,
                },
                "intensity": {
                    "contrast_p": 0.0,
                    "brightness_p": 0.0,
                },
            },
        }

    def _build_model_config(self) -> dict:
        """Build model configuration section."""
        # Determine input channels
        in_channels = 3 if self._ensure_rgb else 1

        # Backbone config
        if "unet" in self._backbone:
            backbone_config = {
                "unet": {
                    "in_channels": in_channels,
                    "filters": self._filters,
                    "filters_rate": self._filters_rate,
                    "max_stride": self._max_stride,
                    "output_stride": self._output_stride,
                }
            }
        elif "convnext" in self._backbone:
            model_type = "tiny" if "tiny" in self._backbone else "small"
            backbone_config = {
                "convnext": {
                    "in_channels": in_channels,
                    "model_type": model_type,
                    "output_stride": self._output_stride,
                }
            }
        else:  # swint
            model_type = "tiny" if "tiny" in self._backbone else "small"
            backbone_config = {
                "swint": {
                    "in_channels": in_channels,
                    "model_type": model_type,
                    "output_stride": self._output_stride,
                }
            }

        # Head config
        head_configs = self._build_head_config()

        return {
            "init_weights": "default",
            "backbone_config": backbone_config,
            "head_configs": head_configs,
        }

    def _build_head_config(self) -> dict:
        """Build head configuration based on pipeline type.

        Returns config with all head types, setting the active one's config
        and null for all others (matches expected sleap-nn config format).
        """
        base_confmap = {
            "sigma": self._sigma,
            "output_stride": self._output_stride,
        }

        # Initialize all head types to null
        head_configs = {
            "single_instance": None,
            "centroid": None,
            "centered_instance": None,
            "bottomup": None,
            "multi_class_bottomup": None,
            "multi_class_topdown": None,
        }

        if self._pipeline == "single_instance":
            head_configs["single_instance"] = {"confmaps": base_confmap}

        elif self._pipeline == "centroid":
            cfg = {"confmaps": {**base_confmap, "anchor_part": self._anchor_part}}
            head_configs["centroid"] = cfg

        elif self._pipeline == "centered_instance":
            cfg = {"confmaps": {**base_confmap, "anchor_part": self._anchor_part}}
            head_configs["centered_instance"] = cfg

        elif self._pipeline == "bottomup":
            head_configs["bottomup"] = {
                "confmaps": {
                    **base_confmap,
                    "loss_weight": 1.0,
                },
                "pafs": {
                    "sigma": 15.0,  # PAFs use larger sigma
                    "output_stride": max(self._output_stride, 4),
                    "loss_weight": 1.0,
                },
            }

        elif self._pipeline == "multi_class_bottomup":
            head_configs["multi_class_bottomup"] = {
                "confmaps": {**base_confmap, "loss_weight": 1.0},
                "pafs": {
                    "sigma": 15.0,
                    "output_stride": max(self._output_stride, 4),
                    "loss_weight": 1.0,
                },
                "class_vectors": {
                    "num_fc_layers": 1,
                    "num_fc_units": 64,
                },
            }

        elif self._pipeline == "multi_class_topdown":
            cfg = {
                "confmaps": {**base_confmap, "anchor_part": self._anchor_part},
                "class_vectors": {
                    "num_fc_layers": 1,
                    "num_fc_units": 64,
                },
            }
            head_configs["multi_class_topdown"] = cfg

        return head_configs

    def _build_trainer_config(self) -> dict:
        """Build trainer configuration section."""
        return {
            "train_data_loader": {
                "batch_size": self._batch_size,
                "shuffle": True,
                "num_workers": 0,
            },
            "val_data_loader": {
                "batch_size": self._batch_size,
                "shuffle": False,
                "num_workers": 0,
            },
            "max_epochs": self._max_epochs,
            "trainer_accelerator": "auto",
            "trainer_devices": "auto",
            "optimizer_name": "Adam",
            "optimizer": {
                "lr": self._learning_rate,
            },
            "early_stopping": {
                "stop_training_on_plateau": self._early_stopping,
                "patience": self._early_stopping_patience,
                "min_delta": 1e-8,
            },
            "save_ckpt": True,
        }

    def save(self, path: str) -> "ConfigGenerator":
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file.

        Returns:
            self for method chaining.
        """
        config = self.build()
        OmegaConf.save(config, path)
        return self

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns:
            YAML string representation.
        """
        return OmegaConf.to_yaml(self.build())

    def memory_estimate(self) -> MemoryEstimate:
        """Get memory estimate for current configuration.

        Returns:
            MemoryEstimate with breakdown and recommendations.
        """
        return estimate_memory(
            self.stats,
            self._backbone,
            self._batch_size,
            self._input_scale,
            self._output_stride,
        )

    def summary(self) -> str:
        """Get a human-readable summary of the configuration.

        Returns:
            Multi-line summary string.
        """
        mem = self.memory_estimate()
        rec = self._recommendation or self.recommend()

        lines = [
            "=" * 60,
            "SLEAP-NN Configuration Summary",
            "=" * 60,
            "",
            "Dataset:",
            f"  File: {self.slp_path.name}",
            f"  Labeled frames: {self.stats.num_labeled_frames}",
            f"  Image size: {self.stats.max_width}x{self.stats.max_height}",
            f"  Channels: {self.stats.num_channels} "
            f"({'grayscale' if self.stats.is_grayscale else 'RGB'})",
            f"  Max instances/frame: {self.stats.max_instances_per_frame}",
            f"  Skeleton: {self.stats.num_nodes} nodes, {self.stats.num_edges} edges",
            "",
            "Recommendation:",
            f"  Pipeline: {rec.pipeline.recommended}",
            f"  Reason: {rec.pipeline.reason}",
            "",
            "Configuration:",
            f"  Pipeline: {self._pipeline}",
            f"  Backbone: {self._backbone}",
            f"  Input scale: {self._input_scale}",
            f"  Sigma: {self._sigma}",
            f"  Batch size: {self._batch_size}",
            f"  Max epochs: {self._max_epochs}",
            f"  Learning rate: {self._learning_rate}",
            f"  Rotation: {self._rotation_range[0]}deg to {self._rotation_range[1]}deg",
            "",
            "Memory Estimate:",
            f"  GPU: {mem.total_gpu_gb:.1f} GB ({mem.gpu_status}) - {mem.gpu_message}",
            f"  CPU cache: {mem.cache_memory_gb:.1f} GB - {mem.cpu_message}",
            "",
            "=" * 60,
        ]

        if rec.pipeline.warnings:
            # Insert warnings before the final separator
            lines.insert(-1, "Warnings:")
            for w in rec.pipeline.warnings:
                lines.insert(-1, f"  * {w}")
            lines.insert(-1, "")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return repr string."""
        return f"ConfigGenerator(slp_path='{self.slp_path}')"
