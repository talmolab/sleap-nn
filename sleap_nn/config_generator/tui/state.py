"""Centralized reactive state management for the TUI.

This module provides a ConfigState class that wraps ConfigGenerator
with reactive properties for real-time UI updates.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.architecture_estimates import (
    compute_max_stride_for_animal_size,
    compute_receptive_field,
    compute_suggested_crop_size,
    encoder_blocks as _encoder_blocks,
    estimate_unet_params,
    recommend_default_max_stride,
)
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
    checkpoint_dir: str = "./models"  # Web app default
    save_top_k: int = 1
    save_last: bool = True  # Web app default
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
        self._ensure_grayscale: bool = False
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
        self._visualize_preds: bool = True  # Web app default (matches checkbox)
        self._keep_viz: bool = False  # Web app default

        # For top-down, separate training config for centered instance
        self._ci_batch_size: int = 4
        self._ci_max_epochs: int = 200
        self._ci_learning_rate: float = 1e-4
        self._ci_optimizer: str = "Adam"

        # Early stopping
        self._early_stopping: bool = True
        self._early_stopping_patience: int = 5  # Web-app HTML default
        self._early_stopping_min_delta: float = 1e-6  # Web-app HTML default

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

    @property
    def receptive_field(self) -> int:
        """Receptive field of the deepest encoder layer (UNet)."""
        return compute_receptive_field(self._max_stride)

    @property
    def encoder_blocks(self) -> int:
        """Number of encoder blocks based on max_stride."""
        return _encoder_blocks(self._max_stride)

    def _compute_max_stride_for_animal_size(self, animal_size: float) -> int:
        """Smallest max_stride whose RF covers the animal."""
        return compute_max_stride_for_animal_size(animal_size)

    def _compute_auto_crop_size(self) -> int:
        """Auto-suggest a crop size for the centered-instance model.

        Uses the canonical formula
        :py:func:`compute_suggested_crop_size` (web-app parity), then floors
        to ``_ci_min_crop_size``.
        """
        crop = compute_suggested_crop_size(
            self.stats.max_bbox_size,
            max_stride=self._ci_max_stride,
            use_augmentation=self._ci_augmentation.enabled,
            user_padding=self._ci_crop_padding,
            rotation_max=(
                self._ci_augmentation.rotation_max
                if self._ci_augmentation.rotation_enabled
                else 0.0
            ),
            scale_max=(
                self._ci_augmentation.scale_max
                if self._ci_augmentation.scale_enabled
                else 1.0
            ),
        )
        return max(crop, self._ci_min_crop_size)

    def _estimate_params(
        self, backbone: str, filters: int, filters_rate: float, max_stride: int
    ) -> int:
        """Estimate model parameter count."""
        if "convnext" in backbone or "swint" in backbone:
            # Pretrained backbones have ~fixed parameter counts.
            if "tiny" in backbone:
                return 28_000_000
            return 50_000_000

        in_channels = 3 if self._ensure_rgb else 1
        num_keypoints = self.stats.num_nodes if self.stats else 1
        return estimate_unet_params(
            filters=filters,
            max_stride=max_stride,
            output_stride=self._output_stride,
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            filters_rate=filters_rate,
        )

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
            self._filters_rate = 1.5

        # Default max_stride from web-app bucket logic (avg-bbox-diagonal * scale).
        # Mirrors ``setDefaultParameters`` in app.html (which uses
        # ``slpData.avgAnimalSize`` = avg of bbox diagonals) so TUI and web app
        # produce the same recommendation for the same SLP.
        bucket_stride = recommend_default_max_stride(
            self.stats.avg_bbox_diagonal, self._input_scale
        )
        # Floor: ensure RF still covers the largest bbox (scaled).
        scaled_max_animal_size = self.stats.max_bbox_size * self._input_scale
        coverage_stride = self._compute_max_stride_for_animal_size(
            scaled_max_animal_size
        )
        self._max_stride = max(base_max_stride, bucket_stride, coverage_stride)

        # Channel conversion: only request a conversion when the original
        # channel count differs from what's needed (RGB needed for pretrained
        # backbones, otherwise default to whatever the SLP provides).
        is_pretrained = "convnext" in self._backbone or "swint" in self._backbone
        self._ensure_rgb = bool(is_pretrained and self.stats.num_channels == 1)
        self._ensure_grayscale = False

        # Set defaults for top-down models
        if self.is_topdown:
            # Centroid model defaults - lower scale is OK for detecting centers
            self._input_scale = 0.5
            self._sigma = 5.0
            self._output_stride = 2

            # The web app does NOT recompute max_stride after switching to
            # centroid (it stays as picked at scale=1.0). Match that behavior:
            # leave max_stride alone here. The pre-existing value from the
            # bucket above (computed at scale=1.0) is what the web app shows.
            # We still floor by RF coverage at the new scale to be safe.
            scaled_max_animal_size = self.stats.max_bbox_size * self._input_scale
            coverage_stride = self._compute_max_stride_for_animal_size(
                scaled_max_animal_size
            )
            self._max_stride = max(self._max_stride, coverage_stride)

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
        """Build the complete configuration dictionary.

        Delegates to ``ConfigGenerator`` so the TUI emits the canonical
        schema (matches ``docs/configuration/config-picker/app.html``).
        """
        if self._pipeline is None:
            raise ValueError("Pipeline not set. Call auto_configure() first.")

        self._apply_to_generator(self._generator, ci_mode=False)
        cfg = self._generator.build()
        return OmegaConf.to_container(cfg, resolve=True)

    def build_centroid_config(self) -> Dict[str, Any]:
        """Build the centroid-stage config for a top-down pipeline.

        Always emits a ``centroid`` head — regardless of whether the user
        selected ``centroid`` or ``multi_class_topdown`` as the pipeline,
        the first stage of top-down is always centroid detection.
        """
        if not self.is_topdown:
            raise ValueError("Centroid config only for top-down pipelines")

        self._apply_to_generator(self._generator, ci_mode=False)
        cfg = self._generator.build_centroid()
        return OmegaConf.to_container(cfg, resolve=True)

    def build_centered_instance_config(self) -> Dict[str, Any]:
        """Build the centered-instance head config for a top-down pipeline."""
        if not self.is_topdown:
            raise ValueError("Centered instance config only for top-down pipelines")

        self._apply_to_generator(self._generator, ci_mode=True)
        # Use multi_class_topdown if originally selected; else centered_instance.
        ci_pipeline = (
            "multi_class_topdown"
            if self._pipeline == "multi_class_topdown"
            else "centered_instance"
        )
        self._generator._pipeline = ci_pipeline
        cfg = self._generator.build()
        return OmegaConf.to_container(cfg, resolve=True)

    def _apply_to_generator(self, gen: "ConfigGenerator", *, ci_mode: bool) -> None:
        """Push state values onto a ``ConfigGenerator`` so its build matches.

        ``ci_mode`` selects the centered-instance state attrs (``_ci_*``) for
        top-down dual-config generation; otherwise uses the main attrs.
        """
        # Pipeline + skeleton-derived flags
        gen._pipeline = self._pipeline
        gen._anchor_part = self._anchor_part
        gen._view_type = self._view_type
        gen._ensure_rgb = self._ensure_rgb
        gen._ensure_grayscale = self._ensure_grayscale

        # Model
        gen._backbone = self._ci_backbone if ci_mode else self._backbone
        gen._max_stride = self._ci_max_stride if ci_mode else self._max_stride
        gen._filters = self._ci_filters if ci_mode else self._filters
        gen._filters_rate = self._ci_filters_rate if ci_mode else self._filters_rate
        gen._sigma = self._ci_sigma if ci_mode else self._sigma
        gen._output_stride = self._ci_output_stride if ci_mode else self._output_stride
        gen._use_imagenet_pretrained = self._use_imagenet_pretrained
        gen._pretrained_backbone_weights = (
            self._ci_pretrained_backbone if ci_mode else self._pretrained_backbone
        ) or None
        gen._pretrained_head_weights = (
            self._ci_pretrained_head if ci_mode else self._pretrained_head
        ) or None

        # Data / preprocessing
        gen._input_scale = self._ci_input_scale if ci_mode else self._input_scale
        gen._validation_fraction = self._validation_fraction
        gen._max_height = self._max_height
        gen._max_width = self._max_width
        gen._crop_size = self._crop_size
        gen._min_crop_size = self._ci_min_crop_size if ci_mode else 100
        gen._crop_padding = self._ci_crop_padding if ci_mode else None

        # Augmentation
        aug = self._ci_augmentation if ci_mode else self._augmentation
        gen._use_augmentations = aug.enabled
        rot_min = aug.rotation_min if aug.rotation_enabled else 0.0
        rot_max = aug.rotation_max if aug.rotation_enabled else 0.0
        gen._rotation_range = (rot_min, rot_max)
        scale_min = aug.scale_min if aug.scale_enabled else 1.0
        scale_max = aug.scale_max if aug.scale_enabled else 1.0
        gen._scale_range = (scale_min, scale_max)
        # translate is stored as a percentage (0-50) in TUI; canonical is fraction.
        gen._translate = (aug.translate / 100.0) if aug.translate_enabled else 0.0
        gen._brightness = aug.brightness_limit if aug.brightness_enabled else 0.0
        gen._contrast = aug.contrast_limit if aug.contrast_enabled else 0.0

        # PAF / multi-class head settings (bottom-up only meaningful)
        gen._paf_sigma = self._paf_config.sigma
        gen._paf_output_stride = self._paf_config.output_stride
        gen._paf_loss_weight = self._paf_config.loss_weight
        gen._confmaps_loss_weight = self._paf_config.confmaps_loss_weight
        gen._class_fc_layers = self._class_vector_config.num_fc_layers
        gen._class_fc_units = self._class_vector_config.num_fc_units
        gen._class_loss_weight = self._class_vector_config.loss_weight
        gen._mc_confmaps_loss_weight = self._paf_config.confmaps_loss_weight

        # Trainer
        gen._batch_size = self._ci_batch_size if ci_mode else self._batch_size
        gen._max_epochs = self._ci_max_epochs if ci_mode else self._max_epochs
        gen._learning_rate = self._ci_learning_rate if ci_mode else self._learning_rate
        gen._optimizer_name = self._ci_optimizer if ci_mode else self._optimizer
        gen._trainer_accelerator = self._accelerator
        gen._trainer_devices = self._devices
        gen._enable_progress_bar = self._enable_progress_bar
        gen._visualize_preds_during_training = self._visualize_preds
        gen._keep_viz = self._keep_viz
        gen._min_train_steps_per_epoch = self._min_steps_per_epoch
        gen._seed = self._random_seed
        gen._num_workers = self._num_workers

        # Early stopping
        if ci_mode:
            gen._early_stopping = self._ci_early_stopping
            gen._early_stopping_patience = self._ci_early_stopping_patience
            gen._early_stopping_min_delta = self._ci_early_stopping_min_delta
        else:
            gen._early_stopping = self._early_stopping
            gen._early_stopping_patience = self._early_stopping_patience
            gen._early_stopping_min_delta = self._early_stopping_min_delta

        # LR scheduler
        sched = self._ci_scheduler if ci_mode else self._scheduler
        scheduler_map = {
            SchedulerType.NONE: "none",
            SchedulerType.REDUCE_ON_PLATEAU: "reduce_lr_on_plateau",
            SchedulerType.STEP_LR: "step_lr",
            SchedulerType.COSINE_ANNEALING_WARMUP: "cosine_annealing_warmup",
            SchedulerType.LINEAR_WARMUP_LINEAR_DECAY: "linear_warmup_linear_decay",
        }
        gen._lr_scheduler = scheduler_map.get(sched.type, "reduce_lr_on_plateau")
        gen._reduce_lr_factor = sched.factor
        gen._reduce_lr_patience = sched.plateau_patience
        gen._reduce_lr_min = sched.min_lr
        gen._reduce_lr_cooldown = sched.cooldown
        gen._step_lr_step_size = sched.step_size
        gen._step_lr_gamma = sched.gamma
        gen._cosine_warmup_epochs = sched.warmup_epochs
        gen._cosine_warmup_start_lr = sched.warmup_start_lr
        gen._cosine_eta_min = sched.eta_min
        gen._linear_warmup_epochs = sched.linear_warmup_epochs
        gen._linear_warmup_start_lr = sched.linear_warmup_start_lr
        gen._linear_end_lr = sched.end_lr

        # Checkpoint
        gen._save_ckpt = self._checkpoint.enabled
        gen._save_top_k = self._checkpoint.save_top_k
        gen._save_last = self._checkpoint.save_last
        if ci_mode:
            gen._ckpt_dir = (
                self._ci_checkpoint_dir or self._checkpoint.checkpoint_dir or "./models"
            )
            gen._run_name = self._ci_run_name or None
        else:
            gen._ckpt_dir = self._checkpoint.checkpoint_dir or "./models"
            gen._run_name = self._checkpoint.run_name or None
        gen._resume_ckpt_path = self._checkpoint.resume_from or None

        # OHKM
        gen._enable_ohkm = self._ohkm.enabled
        gen._ohkm_ratio = self._ohkm.hard_to_easy_ratio
        gen._ohkm_min_hard = self._ohkm.min_hard_keypoints
        gen._ohkm_max_hard = self._ohkm.max_hard_keypoints
        gen._ohkm_loss_scale = self._ohkm.loss_scale

        # WandB
        gen._enable_wandb = self._wandb.enabled
        gen._wandb_entity = self._wandb.entity or None
        gen._wandb_project = self._wandb.project or "sleap-training"
        gen._wandb_name = self._wandb.name or None
        gen._wandb_api_key = self._wandb.api_key or None
        gen._wandb_mode = self._wandb.mode if self._wandb.mode != "online" else None
        gen._wandb_viz_enabled = self._wandb.viz_enabled
        gen._wandb_save_viz = self._wandb.save_viz_imgs

        # Eval
        gen._enable_eval = self._evaluation.enabled
        gen._eval_frequency = self._evaluation.frequency
        gen._eval_oks_stddev = self._evaluation.oks_stddev

        # Data pipeline / caching
        gen._data_pipeline_fw = self._data_pipeline.value
        gen._cache_img_path = self._cache_config.cache_img_path or None
        gen._use_existing_imgs = self._cache_config.use_existing_imgs
        gen._delete_cache_imgs_after_training = (
            self._cache_config.delete_cache_after_training
        )
        gen._parallel_caching = self._cache_config.parallel_caching
        gen._cache_workers = self._cache_config.cache_workers

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
