"""Centralized reactive state management for the TUI.

This module provides a ConfigState class that wraps ConfigGenerator
with reactive properties for real-time UI updates.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.generator import ConfigGenerator
from sleap_nn.config_generator.memory import MemoryEstimate, estimate_memory
from sleap_nn.config_generator.recommender import (
    BackboneType,
    ConfigRecommendation,
    PipelineType,
    recommend_config,
)


class SchedulerType(str, Enum):
    """Available LR scheduler types."""

    NONE = "none"
    REDUCE_ON_PLATEAU = "ReduceLROnPlateau"
    STEP_LR = "StepLR"
    COSINE_ANNEALING_WARMUP = "CosineAnnealingWarmup"
    LINEAR_WARMUP_LINEAR_DECAY = "LinearWarmupLinearDecay"


class DataPipelineType(str, Enum):
    """Available data pipeline types."""

    TORCH_DATASET = "torch_dataset"
    MEMORY_CACHE = "litdata"
    DISK_CACHE = "litdata_disk"


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU  # Web app default
    # ReduceLROnPlateau params
    factor: float = 0.5
    plateau_patience: int = 5
    min_lr: float = 1e-8
    cooldown: int = 3  # Web app default
    # StepLR params
    step_size: int = 10  # Web app default
    gamma: float = 0.1
    # CosineAnnealingWarmup params
    warmup_epochs: int = 5  # Web app default
    warmup_start_lr: float = 0.0  # Web app default
    eta_min: float = 0.0  # Web app default
    # LinearWarmupLinearDecay params
    linear_warmup_epochs: int = 5  # Web app default
    linear_warmup_start_lr: float = 0.0  # Web app default
    end_lr: float = 0.0  # Web app default


@dataclass
class OHKMConfig:
    """Online Hard Keypoint Mining configuration."""

    enabled: bool = False
    hard_to_easy_ratio: float = 2.0
    loss_scale: float = 5.0  # Web app default
    min_hard_keypoints: int = 2
    max_hard_keypoints: Optional[int] = None  # Web app default is null


@dataclass
class WandBConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    entity: str = ""
    project: str = "sleap-training"  # Web app default
    name: str = ""
    api_key: str = ""
    mode: str = "online"  # online, offline, disabled
    viz_enabled: bool = True  # Log visualizations to wandb
    save_viz_imgs: bool = False  # Upload local viz images to wandb


@dataclass
class EvaluationConfig:
    """OKS evaluation during training configuration."""

    enabled: bool = False
    frequency: int = 1
    oks_stddev: float = 0.025


@dataclass
class CacheConfig:
    """Data caching configuration for disk/memory caching pipelines."""

    cache_img_path: str = ""  # Path to cache directory (for disk caching)
    use_existing_imgs: bool = False  # Reuse existing cached images
    delete_cache_after_training: bool = True  # Delete cache after training
    parallel_caching: bool = True  # Enable parallel caching
    cache_workers: int = 0  # Number of workers for caching (0 = main process)


@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""

    enabled: bool = True  # Master toggle (save_ckpt)
    run_name: str = ""
    checkpoint_dir: str = ""
    save_top_k: int = 1
    save_last: bool = False  # Web app default is null/false
    resume_from: str = ""


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    enabled: bool = True
    # Geometric - rotation
    rotation_enabled: bool = True
    rotation_min: float = -15.0
    rotation_max: float = 15.0
    # Geometric - scale
    scale_enabled: bool = True
    scale_min: float = 0.9
    scale_max: float = 1.1
    # Geometric - translate
    translate_enabled: bool = False
    translate: float = 0.0  # Percentage (0-50), applied to both x and y
    # Intensity - brightness
    brightness_enabled: bool = False
    brightness_limit: float = 0.2
    # Intensity - contrast
    contrast_enabled: bool = False
    contrast_limit: float = 0.2


@dataclass
class PAFConfig:
    """Part Affinity Field configuration for bottom-up models."""

    sigma: float = 15.0
    output_stride: int = 4  # Web app default (usually 2× confmaps stride)
    loss_weight: float = 1.0
    confmaps_loss_weight: float = (
        1.0  # Web app default - loss weight for confmaps in bottom-up
    )


@dataclass
class ClassVectorConfig:
    """Class vector configuration for multi-class models."""

    num_fc_layers: int = 1
    num_fc_units: int = 64
    loss_weight: float = 1.0


class ConfigState:
    """Centralized state management for the config generator TUI.

    This class wraps ConfigGenerator and provides:
    - Reactive state with observer pattern for UI updates
    - Additional configuration options not in the base generator
    - Computed properties for UI display
    - State serialization for dual config generation
    """

    def __init__(self, slp_path: str):
        """Initialize state from an SLP file.

        Args:
            slp_path: Path to the .slp file.
        """
        self.slp_path = Path(slp_path)
        self._generator = ConfigGenerator.from_slp(str(slp_path))

        # Load stats immediately
        self._stats: Optional[DatasetStats] = None
        self._recommendation: Optional[ConfigRecommendation] = None

        # Extended configuration state
        self._view_type: ViewType = ViewType.UNKNOWN

        # Data config
        self._input_scale: float = 1.0
        self._max_height: Optional[int] = None
        self._max_width: Optional[int] = None
        self._ensure_rgb: bool = False
        self._ensure_grayscale: bool = True
        self._data_pipeline: DataPipelineType = DataPipelineType.TORCH_DATASET
        self._num_workers: int = 0
        self._validation_fraction: float = 0.1
        self._user_instances_only: bool = True  # Web app default

        # Cache config (for disk/memory caching)
        self._cache_config: CacheConfig = CacheConfig()

        # Augmentation config
        self._augmentation: AugmentationConfig = AugmentationConfig()
        # For top-down, separate augmentation for centered instance
        self._ci_augmentation: AugmentationConfig = AugmentationConfig()

        # Model config
        self._pipeline: Optional[PipelineType] = None
        self._backbone: BackboneType = "unet_medium_rf"
        self._max_stride: int = 16
        self._filters: int = 32
        self._filters_rate: float = 1.5  # Web app default
        self._sigma: float = 2.5  # Web app default
        self._output_stride: int = 1
        self._anchor_part: Optional[str] = None
        self._crop_size: Optional[int] = None
        self._pretrained_backbone: str = ""
        self._pretrained_head: str = ""
        self._use_imagenet_pretrained: bool = (
            True  # Web app default - for ConvNeXt/SwinT
        )

        # For top-down, separate model config for centered instance
        self._ci_backbone: BackboneType = "unet_medium_rf"
        self._ci_max_stride: int = 16
        self._ci_filters: int = 32
        self._ci_filters_rate: float = 1.5
        self._ci_sigma: float = 2.5
        self._ci_output_stride: int = 2
        self._ci_pretrained_backbone: str = ""
        self._ci_pretrained_head: str = ""
        self._ci_input_scale: float = 1.0
        self._ci_min_crop_size: int = 100
        self._ci_crop_padding: Optional[int] = None

        # PAF config (for bottom-up)
        self._paf_config: PAFConfig = PAFConfig()

        # Class vector config (for multi-class)
        self._class_vector_config: ClassVectorConfig = ClassVectorConfig()

        # Training config
        self._batch_size: int = 4
        self._max_epochs: int = 200
        self._learning_rate: float = 1e-4
        self._optimizer: str = "Adam"
        self._accelerator: str = "auto"
        self._devices: str = "auto"  # Number of GPUs: "auto", "1", "2", etc.
        self._min_steps_per_epoch: int = 200  # Web app default
        self._random_seed: Optional[int] = None
        self._enable_progress_bar: bool = True  # Web app default
        self._visualize_preds: bool = False  # Web app default
        self._keep_viz: bool = False  # Web app default

        # For top-down, separate training config for centered instance
        self._ci_batch_size: int = 4
        self._ci_max_epochs: int = 200
        self._ci_learning_rate: float = 1e-4
        self._ci_optimizer: str = "Adam"

        # Early stopping
        self._early_stopping: bool = True
        self._early_stopping_patience: int = 10
        self._early_stopping_min_delta: float = 1e-8

        # For top-down
        self._ci_early_stopping: bool = True
        self._ci_early_stopping_patience: int = 5
        self._ci_early_stopping_min_delta: float = 1e-6

        # Scheduler config
        self._scheduler: SchedulerConfig = SchedulerConfig()
        self._ci_scheduler: SchedulerConfig = SchedulerConfig()

        # Checkpoint config
        self._checkpoint: CheckpointConfig = CheckpointConfig()
        # For top-down, separate checkpoint config for centered instance
        self._ci_checkpoint_dir: str = ""
        self._ci_run_name: str = ""

        # OHKM config
        self._ohkm: OHKMConfig = OHKMConfig()

        # W&B config
        self._wandb: WandBConfig = WandBConfig()

        # Evaluation config
        self._evaluation: EvaluationConfig = EvaluationConfig()

        # Observers for reactive updates
        self._observers: List[Callable[[], None]] = []

    @property
    def stats(self) -> DatasetStats:
        """Get dataset statistics (lazily computed)."""
        if self._stats is None:
            self._stats = analyze_slp(str(self.slp_path))
        return self._stats

    @property
    def recommendation(self) -> ConfigRecommendation:
        """Get configuration recommendation."""
        if self._recommendation is None:
            self._recommendation = recommend_config(self.stats, self._view_type)
        return self._recommendation

    @property
    def skeleton_nodes(self) -> List[str]:
        """Get list of skeleton node names."""
        return self.stats.node_names

    @property
    def is_topdown(self) -> bool:
        """Check if current pipeline is top-down (requires dual config)."""
        return self._pipeline in [
            "centroid",
            "centered_instance",
            "multi_class_topdown",
        ]

    @property
    def is_bottomup(self) -> bool:
        """Check if current pipeline is bottom-up (requires PAF config)."""
        return self._pipeline in ["bottomup", "multi_class_bottomup"]

    @property
    def is_multiclass(self) -> bool:
        """Check if current pipeline is multi-class (requires class vector config)."""
        return self._pipeline in ["multi_class_bottomup", "multi_class_topdown"]

    @property
    def effective_height(self) -> int:
        """Calculate effective image height after preprocessing."""
        h = self.stats.max_height
        if self._max_height:
            h = min(h, self._max_height)
        return int(h * self._input_scale)

    @property
    def effective_width(self) -> int:
        """Calculate effective image width after preprocessing."""
        w = self.stats.max_width
        if self._max_width:
            w = min(w, self._max_width)
        return int(w * self._input_scale)

    @property
    def output_height(self) -> int:
        """Calculate output confidence map height."""
        return self.effective_height // self._output_stride

    @property
    def output_width(self) -> int:
        """Calculate output confidence map width."""
        return self.effective_width // self._output_stride

    @property
    def model_params_estimate(self) -> int:
        """Estimate total model parameters based on architecture."""
        return self._estimate_params(
            self._backbone, self._filters, self._filters_rate, self._max_stride
        )

    # RF map: max_stride -> receptive field size (for UNet)
    RF_MAP = {8: 36, 16: 76, 32: 156, 64: 316}

    @property
    def receptive_field(self) -> int:
        """Estimate receptive field based on max_stride."""
        return self.RF_MAP.get(self._max_stride, 76)

    @property
    def encoder_blocks(self) -> int:
        """Number of encoder blocks based on max_stride."""
        import math

        return int(math.log2(self._max_stride))

    def _compute_max_stride_for_animal_size(self, animal_size: float) -> int:
        """Compute the minimum max_stride needed for RF to cover animal size.

        Args:
            animal_size: Maximum animal bounding box size in pixels.

        Returns:
            The smallest max_stride where RF >= animal_size.
        """
        # Find smallest stride where RF >= animal_size
        for stride in [8, 16, 32, 64]:
            if self.RF_MAP[stride] >= animal_size:
                return stride
        # If animal is very large, use max stride
        return 64

    def _estimate_params(
        self, backbone: str, filters: int, filters_rate: float, max_stride: int
    ) -> int:
        """Estimate model parameters."""
        import math

        if "convnext" in backbone or "swint" in backbone:
            # Pretrained models have fixed sizes
            if "tiny" in backbone:
                return 28_000_000
            return 50_000_000

        # UNet parameter estimation
        num_blocks = int(math.log2(max_stride))
        total = 0

        # Encoder
        in_ch = 3 if self._ensure_rgb else 1
        for i in range(num_blocks):
            out_ch = int(filters * (filters_rate**i))
            # Conv blocks (2 per level)
            total += in_ch * out_ch * 9 + out_ch  # 3x3 conv + bias
            total += out_ch * out_ch * 9 + out_ch
            in_ch = out_ch

        # Decoder (similar structure)
        for i in range(num_blocks - 1, -1, -1):
            out_ch = int(filters * (filters_rate**i))
            skip_ch = out_ch  # Skip connection
            total += (in_ch + skip_ch) * out_ch * 9 + out_ch
            total += out_ch * out_ch * 9 + out_ch
            in_ch = out_ch

        # Head
        total += in_ch * self.stats.num_nodes * 1 + self.stats.num_nodes

        return total

    def add_observer(self, callback: Callable[[], None]) -> None:
        """Add an observer callback for state changes."""
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[], None]) -> None:
        """Remove an observer callback."""
        if callback in self._observers:
            self._observers.remove(callback)

    def notify_observers(self) -> None:
        """Notify all observers of state change."""
        for callback in self._observers:
            callback()

    def auto_configure(self, view: Optional[str] = None) -> None:
        """Auto-configure all parameters based on data analysis.

        If pipeline is already set, it will not be overwritten.
        """
        if view:
            self._view_type = ViewType(view)

        rec = self.recommendation

        # Only set pipeline if not already set (allows user to override)
        if self._pipeline is None:
            self._pipeline = rec.pipeline.recommended

        self._backbone = rec.backbone
        self._sigma = rec.sigma
        self._input_scale = rec.input_scale
        self._batch_size = rec.batch_size
        self._augmentation.rotation_min = rec.rotation_range[0]
        self._augmentation.rotation_max = rec.rotation_range[1]

        if rec.crop_size:
            self._crop_size = rec.crop_size

        # Set backbone-specific parameters
        if "large_rf" in self._backbone:
            base_max_stride = 32
            self._filters = 24
            self._filters_rate = 1.5
        else:
            base_max_stride = 16
            self._filters = 32
            self._filters_rate = 2.0

        # Adjust max_stride if RF < animal size (scaled by input_scale)
        scaled_animal_size = self.stats.max_bbox_size * self._input_scale
        required_stride = self._compute_max_stride_for_animal_size(scaled_animal_size)
        self._max_stride = max(base_max_stride, required_stride)

        # Set channel configuration
        self._ensure_rgb = self.stats.is_rgb
        self._ensure_grayscale = self.stats.is_grayscale

        # Set defaults for top-down models
        if self.is_topdown:
            # Centroid model defaults - lower scale is OK for detecting centers
            self._input_scale = 0.5
            self._sigma = 5.0
            self._output_stride = 2

            # Adjust centroid max_stride if RF < scaled animal size
            scaled_animal_size = self.stats.max_bbox_size * self._input_scale
            centroid_required_stride = self._compute_max_stride_for_animal_size(
                scaled_animal_size
            )
            self._max_stride = max(16, centroid_required_stride)

            # Centered instance model defaults
            # Always use max_stride=16 for instance - crops are sized appropriately
            self._ci_backbone = "unet_medium_rf"
            self._ci_max_stride = 16
            self._ci_filters = 32
            self._ci_filters_rate = 1.5
            self._ci_sigma = 2.5
            self._ci_output_stride = 2
            self._ci_input_scale = 1.0  # Full resolution for keypoint detection

            self._ci_augmentation = AugmentationConfig(
                rotation_min=rec.rotation_range[0],
                rotation_max=rec.rotation_range[1],
            )

        self.notify_observers()

    def memory_estimate(self) -> MemoryEstimate:
        """Get memory estimate for current configuration."""
        return estimate_memory(
            self.stats,
            self._backbone,
            self._batch_size,
            self._input_scale,
            self._output_stride,
            filters=self._filters,
            filters_rate=self._filters_rate,
            max_stride=self._max_stride,
            num_keypoints=len(self.skeleton_nodes),
        )

    def build_config(self) -> Dict[str, Any]:
        """Build the complete configuration dictionary."""
        if self._pipeline is None:
            raise ValueError("Pipeline not set. Call auto_configure() first.")

        return {
            "data_config": self._build_data_config(),
            "model_config": self._build_model_config(),
            "trainer_config": self._build_trainer_config(),
        }

    def build_centroid_config(self) -> Dict[str, Any]:
        """Build centroid model config for top-down pipeline."""
        if not self.is_topdown:
            raise ValueError("Centroid config only for top-down pipelines")

        # Temporarily set pipeline to centroid
        orig_pipeline = self._pipeline
        self._pipeline = "centroid"
        config = self.build_config()
        self._pipeline = orig_pipeline
        return config

    def build_centered_instance_config(self) -> Dict[str, Any]:
        """Build centered instance model config for top-down pipeline."""
        if not self.is_topdown:
            raise ValueError("Centered instance config only for top-down pipelines")

        # Use CI-specific settings
        return {
            "data_config": self._build_ci_data_config(),
            "model_config": self._build_ci_model_config(),
            "trainer_config": self._build_ci_trainer_config(),
        }

    def _build_data_config(self) -> Dict[str, Any]:
        """Build data configuration section."""
        config = {
            "train_labels_path": [str(self.slp_path)],
            "val_labels_path": [],
            "validation_fraction": self._validation_fraction,
            "user_instances_only": self._user_instances_only,
            "data_pipeline_fw": self._data_pipeline.value,
            "preprocessing": {
                "ensure_rgb": self._ensure_rgb,
                "ensure_grayscale": self._ensure_grayscale,
                "scale": self._input_scale,
            },
            "use_augmentations_train": self._augmentation.enabled,
            "augmentation_config": self._build_augmentation_config(self._augmentation),
        }

        if self._max_height:
            config["preprocessing"]["max_height"] = self._max_height
        if self._max_width:
            config["preprocessing"]["max_width"] = self._max_width
        if self._crop_size:
            config["preprocessing"]["crop_size"] = self._crop_size

        # Add caching config when not using default torch_dataset
        if self._data_pipeline != DataPipelineType.TORCH_DATASET:
            if self._data_pipeline == DataPipelineType.DISK_CACHE:
                # Disk caching options
                if self._cache_config.cache_img_path:
                    config["cache_img_path"] = self._cache_config.cache_img_path
                config["use_existing_imgs"] = self._cache_config.use_existing_imgs
                config["delete_cache_imgs_after_training"] = (
                    self._cache_config.delete_cache_after_training
                )
            config["parallel_caching"] = self._cache_config.parallel_caching
            config["cache_workers"] = self._cache_config.cache_workers

        return config

    def _build_ci_data_config(self) -> Dict[str, Any]:
        """Build data config for centered instance model."""
        config = {
            "train_labels_path": [str(self.slp_path)],
            "val_labels_path": [],
            "validation_fraction": self._validation_fraction,
            "user_instances_only": self._user_instances_only,
            "data_pipeline_fw": self._data_pipeline.value,
            "preprocessing": {
                "ensure_rgb": self._ensure_rgb,
                "ensure_grayscale": self._ensure_grayscale,
                "scale": self._ci_input_scale,
                "crop_size": self._crop_size or self._compute_auto_crop_size(),
                "min_crop_size": self._ci_min_crop_size,
            },
            "use_augmentations_train": self._ci_augmentation.enabled,
            "augmentation_config": self._build_augmentation_config(
                self._ci_augmentation
            ),
        }

        if self._ci_crop_padding is not None:
            config["preprocessing"]["crop_padding"] = self._ci_crop_padding

        # Add caching config when not using default torch_dataset
        if self._data_pipeline != DataPipelineType.TORCH_DATASET:
            if self._data_pipeline == DataPipelineType.DISK_CACHE:
                if self._cache_config.cache_img_path:
                    config["cache_img_path"] = self._cache_config.cache_img_path
                config["use_existing_imgs"] = self._cache_config.use_existing_imgs
                config["delete_cache_imgs_after_training"] = (
                    self._cache_config.delete_cache_after_training
                )
            config["parallel_caching"] = self._cache_config.parallel_caching
            config["cache_workers"] = self._cache_config.cache_workers

        return config

    def _compute_auto_crop_size(self) -> int:
        """Compute automatic crop size based on instance bounding boxes."""
        # Use 1.5x the max bbox dimension, rounded to max_stride
        max_dim = self.stats.max_bbox_size
        crop = int(max_dim * 1.5)
        # Round up to multiple of max_stride
        crop = (
            (crop + self._ci_max_stride - 1) // self._ci_max_stride
        ) * self._ci_max_stride
        return max(crop, self._ci_min_crop_size)

    def _build_augmentation_config(self, aug: AugmentationConfig) -> Dict[str, Any]:
        """Build augmentation configuration section."""
        # Only include enabled augmentations
        rotation_min = aug.rotation_min if aug.rotation_enabled else 0.0
        rotation_max = aug.rotation_max if aug.rotation_enabled else 0.0
        scale_min = aug.scale_min if aug.scale_enabled else 1.0
        scale_max = aug.scale_max if aug.scale_enabled else 1.0
        translate = (
            aug.translate / 100.0 if aug.translate_enabled else 0.0
        )  # Convert % to fraction

        # Geometric augmentations are applied if any geometric aug is enabled
        any_geometric = (
            aug.rotation_enabled or aug.scale_enabled or aug.translate_enabled
        )
        affine_p = 1.0 if (aug.enabled and any_geometric) else 0.0

        return {
            "geometric": {
                "rotation_min": rotation_min,
                "rotation_max": rotation_max,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "translate_width": translate,
                "translate_height": translate,
                "affine_p": affine_p,
            },
            "intensity": {
                "brightness_limit": (
                    aug.brightness_limit if aug.brightness_enabled else 0.0
                ),
                "brightness_p": 0.5 if aug.brightness_enabled else 0.0,
                "contrast_limit": aug.contrast_limit if aug.contrast_enabled else 0.0,
                "contrast_p": 0.5 if aug.contrast_enabled else 0.0,
            },
        }

    def _build_model_config(self) -> Dict[str, Any]:
        """Build model configuration section."""
        in_channels = 3 if self._ensure_rgb else 1

        # Backbone config
        backbone_config = self._build_backbone_config(
            self._backbone,
            in_channels,
            self._filters,
            self._filters_rate,
            self._max_stride,
            self._output_stride,
        )

        # Head config
        head_configs = self._build_head_config()

        config = {
            "init_weights": "default",
            "backbone_config": backbone_config,
            "head_configs": head_configs,
        }

        # Add pretrained weights if specified
        if self._pretrained_backbone:
            config["pretrained_backbone_path"] = self._pretrained_backbone
        if self._pretrained_head:
            config["pretrained_head_path"] = self._pretrained_head

        return config

    def _build_ci_model_config(self) -> Dict[str, Any]:
        """Build model config for centered instance model."""
        in_channels = 3 if self._ensure_rgb else 1

        backbone_config = self._build_backbone_config(
            self._ci_backbone,
            in_channels,
            self._ci_filters,
            self._ci_filters_rate,
            self._ci_max_stride,
            self._ci_output_stride,
        )

        head_configs = {
            "single_instance": None,
            "centroid": None,
            "centered_instance": {
                "confmaps": {
                    "sigma": self._ci_sigma,
                    "output_stride": self._ci_output_stride,
                    "anchor_part": self._anchor_part,
                }
            },
            "bottomup": None,
            "multi_class_bottomup": None,
            "multi_class_topdown": None,
        }

        config = {
            "init_weights": "default",
            "backbone_config": backbone_config,
            "head_configs": head_configs,
        }

        if self._ci_pretrained_backbone:
            config["pretrained_backbone_path"] = self._ci_pretrained_backbone
        if self._ci_pretrained_head:
            config["pretrained_head_path"] = self._ci_pretrained_head

        return config

    def _build_backbone_config(
        self,
        backbone: str,
        in_channels: int,
        filters: int,
        filters_rate: float,
        max_stride: int,
        output_stride: int,
    ) -> Dict[str, Any]:
        """Build backbone configuration."""
        if "unet" in backbone:
            return {
                "unet": {
                    "in_channels": in_channels,
                    "filters": filters,
                    "filters_rate": filters_rate,
                    "max_stride": max_stride,
                    "output_stride": output_stride,
                }
            }
        elif "convnext" in backbone:
            model_type = "tiny" if "tiny" in backbone else "small"
            return {
                "convnext": {
                    "in_channels": in_channels,
                    "model_type": model_type,
                    "output_stride": output_stride,
                }
            }
        else:  # swint
            model_type = "tiny" if "tiny" in backbone else "small"
            return {
                "swint": {
                    "in_channels": in_channels,
                    "model_type": model_type,
                    "output_stride": output_stride,
                }
            }

    def _build_head_config(self) -> Dict[str, Any]:
        """Build head configuration based on pipeline type."""
        base_confmap = {
            "sigma": self._sigma,
            "output_stride": self._output_stride,
        }

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
            head_configs["centroid"] = {
                "confmaps": {**base_confmap, "anchor_part": self._anchor_part}
            }

        elif self._pipeline == "centered_instance":
            head_configs["centered_instance"] = {
                "confmaps": {**base_confmap, "anchor_part": self._anchor_part}
            }

        elif self._pipeline == "bottomup":
            head_configs["bottomup"] = {
                "confmaps": {
                    **base_confmap,
                    "loss_weight": self._paf_config.confmaps_loss_weight,
                },
                "pafs": {
                    "sigma": self._paf_config.sigma,
                    "output_stride": self._paf_config.output_stride,
                    "loss_weight": self._paf_config.loss_weight,
                },
            }

        elif self._pipeline == "multi_class_bottomup":
            head_configs["multi_class_bottomup"] = {
                "confmaps": {
                    **base_confmap,
                    "loss_weight": self._paf_config.confmaps_loss_weight,
                },
                "pafs": {
                    "sigma": self._paf_config.sigma,
                    "output_stride": self._paf_config.output_stride,
                    "loss_weight": self._paf_config.loss_weight,
                },
                "class_vectors": {
                    "num_fc_layers": self._class_vector_config.num_fc_layers,
                    "num_fc_units": self._class_vector_config.num_fc_units,
                },
            }

        elif self._pipeline == "multi_class_topdown":
            head_configs["multi_class_topdown"] = {
                "confmaps": {**base_confmap, "anchor_part": self._anchor_part},
                "class_vectors": {
                    "num_fc_layers": self._class_vector_config.num_fc_layers,
                    "num_fc_units": self._class_vector_config.num_fc_units,
                },
            }

        return head_configs

    def _build_trainer_config(self) -> Dict[str, Any]:
        """Build trainer configuration section."""
        config = {
            "train_data_loader": {
                "batch_size": self._batch_size,
                "shuffle": True,
                "num_workers": self._num_workers,
            },
            "val_data_loader": {
                "batch_size": self._batch_size,
                "shuffle": False,
                "num_workers": self._num_workers,
            },
            "max_epochs": self._max_epochs,
            "trainer_accelerator": self._accelerator,
            "trainer_devices": self._devices,
            "enable_progress_bar": self._enable_progress_bar,
            "visualize_preds_during_training": self._visualize_preds,
            "keep_viz": self._keep_viz,
            "optimizer_name": self._optimizer,
            "optimizer": {
                "lr": self._learning_rate,
                "amsgrad": False,
            },
            "early_stopping": {
                "stop_training_on_plateau": self._early_stopping,
                "patience": self._early_stopping_patience,
                "min_delta": self._early_stopping_min_delta,
            },
            "save_ckpt": self._checkpoint.enabled,
            "min_steps_per_epoch": self._min_steps_per_epoch,
        }

        if self._random_seed is not None:
            config["random_seed"] = self._random_seed

        # Add scheduler config
        if self._scheduler.type != SchedulerType.NONE:
            config["lr_scheduler"] = self._build_scheduler_config(self._scheduler)

        # Add checkpoint config
        if self._checkpoint.run_name:
            config["run_name"] = self._checkpoint.run_name
        if self._checkpoint.checkpoint_dir:
            config["ckpt_dir"] = self._checkpoint.checkpoint_dir
        if self._checkpoint.resume_from:
            config["resume_ckpt_path"] = self._checkpoint.resume_from
        config["save_top_k"] = self._checkpoint.save_top_k
        config["save_last"] = self._checkpoint.save_last

        # Add OHKM config
        if self._ohkm.enabled:
            config["ohkm"] = {
                "enabled": True,
                "hard_to_easy_ratio": self._ohkm.hard_to_easy_ratio,
                "loss_scale": self._ohkm.loss_scale,
                "min_hard_keypoints": self._ohkm.min_hard_keypoints,
                "max_hard_keypoints": self._ohkm.max_hard_keypoints,
            }

        # Add W&B config
        if self._wandb.enabled:
            config["wandb"] = {
                "enabled": True,
                "entity": self._wandb.entity,
                "project": self._wandb.project,
                "name": self._wandb.name,
                "mode": self._wandb.mode,
                "viz_enabled": self._wandb.viz_enabled,
                "save_viz_imgs_wandb": self._wandb.save_viz_imgs,
            }
            if self._wandb.api_key:
                config["wandb"]["api_key"] = self._wandb.api_key

        # Add evaluation config
        if self._evaluation.enabled:
            config["evaluation"] = {
                "enabled": True,
                "frequency": self._evaluation.frequency,
                "oks_stddev": self._evaluation.oks_stddev,
            }

        return config

    def _build_ci_trainer_config(self) -> Dict[str, Any]:
        """Build trainer config for centered instance model."""
        config = {
            "train_data_loader": {
                "batch_size": self._ci_batch_size,
                "shuffle": True,
                "num_workers": self._num_workers,
            },
            "val_data_loader": {
                "batch_size": self._ci_batch_size,
                "shuffle": False,
                "num_workers": self._num_workers,
            },
            "max_epochs": self._ci_max_epochs,
            "trainer_accelerator": self._accelerator,
            "trainer_devices": self._devices,
            "optimizer_name": self._ci_optimizer,
            "optimizer": {
                "lr": self._ci_learning_rate,
            },
            "early_stopping": {
                "stop_training_on_plateau": self._ci_early_stopping,
                "patience": self._ci_early_stopping_patience,
                "min_delta": self._ci_early_stopping_min_delta,
            },
            "save_ckpt": self._checkpoint.enabled,
            "min_steps_per_epoch": self._min_steps_per_epoch,
        }

        if self._random_seed is not None:
            config["random_seed"] = self._random_seed

        # Add scheduler config
        if self._ci_scheduler.type != SchedulerType.NONE:
            config["lr_scheduler"] = self._build_scheduler_config(self._ci_scheduler)

        return config

    def _build_scheduler_config(self, scheduler: SchedulerConfig) -> Dict[str, Any]:
        """Build scheduler configuration."""
        config = {"type": scheduler.type.value}

        if scheduler.type == SchedulerType.REDUCE_ON_PLATEAU:
            config.update(
                {
                    "factor": scheduler.factor,
                    "patience": scheduler.plateau_patience,
                    "min_lr": scheduler.min_lr,
                    "cooldown": scheduler.cooldown,
                }
            )
        elif scheduler.type == SchedulerType.STEP_LR:
            config.update(
                {
                    "step_size": scheduler.step_size,
                    "gamma": scheduler.gamma,
                }
            )
        elif scheduler.type == SchedulerType.COSINE_ANNEALING_WARMUP:
            config.update(
                {
                    "warmup_epochs": scheduler.warmup_epochs,
                    "warmup_start_lr": scheduler.warmup_start_lr,
                    "eta_min": scheduler.eta_min,
                }
            )
        elif scheduler.type == SchedulerType.LINEAR_WARMUP_LINEAR_DECAY:
            config.update(
                {
                    "warmup_epochs": scheduler.linear_warmup_epochs,
                    "warmup_start_lr": scheduler.linear_warmup_start_lr,
                    "end_lr": scheduler.end_lr,
                }
            )

        return config

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        from omegaconf import OmegaConf

        config = self.build_config()
        return OmegaConf.to_yaml(OmegaConf.create(config))

    def to_centroid_yaml(self) -> str:
        """Convert centroid config to YAML string."""
        from omegaconf import OmegaConf

        config = self.build_centroid_config()
        return OmegaConf.to_yaml(OmegaConf.create(config))

    def to_centered_instance_yaml(self) -> str:
        """Convert centered instance config to YAML string."""
        from omegaconf import OmegaConf

        config = self.build_centered_instance_config()
        return OmegaConf.to_yaml(OmegaConf.create(config))

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        from omegaconf import OmegaConf

        config = self.build_config()
        OmegaConf.save(OmegaConf.create(config), path)

    def save_dual(self, base_path: str) -> Tuple[str, str]:
        """Save dual configs for top-down pipeline.

        Args:
            base_path: Base path for output files (without extension).

        Returns:
            Tuple of (centroid_path, centered_instance_path).
        """
        from omegaconf import OmegaConf

        centroid_path = f"{base_path}_centroid.yaml"
        ci_path = f"{base_path}_centered_instance.yaml"

        OmegaConf.save(OmegaConf.create(self.build_centroid_config()), centroid_path)
        OmegaConf.save(OmegaConf.create(self.build_centered_instance_config()), ci_path)

        return centroid_path, ci_path
