"""Main ConfigGenerator class for creating training configurations.

This module provides a fluent API for generating sleap-nn training
configurations from SLP files with sensible defaults.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from sleap_nn.config_generator.analyzer import DatasetStats, ViewType, analyze_slp
from sleap_nn.config_generator.architecture_estimates import (
    compute_max_stride_for_animal_size,
    compute_suggested_crop_size,
    recommend_default_max_stride,
)
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
        self._filters_rate: float = 1.5
        self._use_augmentations: bool = True
        self._rotation_range: Tuple[float, float] = (-15.0, 15.0)
        self._scale_range: Tuple[float, float] = (0.9, 1.1)
        self._translate: float = 0.0  # 0.0..1.0 fraction of image width/height
        self._brightness: float = 0.0  # 0..1 fraction; 0 disables
        self._contrast: float = 0.0  # 0..1 fraction; 0 disables
        self._early_stopping: bool = True
        self._early_stopping_patience: int = 5  # Web-app HTML default
        self._early_stopping_min_delta: float = 1e-6  # Web-app HTML default
        self._validation_fraction: float = 0.1
        self._anchor_part: Optional[str] = None
        self._crop_size: Optional[int] = None
        self._min_crop_size: int = 100
        self._crop_padding: Optional[int] = None
        self._max_height: Optional[int] = None  # override; defaults to stats max
        self._max_width: Optional[int] = None
        self._view_type: ViewType = ViewType.UNKNOWN
        self._ensure_rgb: bool = False
        self._ensure_grayscale: bool = False
        self._input_channels: Optional[int] = None  # overrides stats.num_channels

        # Head settings
        self._paf_sigma: float = 15.0
        self._paf_output_stride: int = 4
        self._paf_loss_weight: float = 1.0
        self._confmaps_loss_weight: float = 1.0
        # multi-class
        self._class_fc_layers: int = 1
        self._class_fc_units: int = 64
        self._class_loss_weight: float = 1.0
        self._mc_confmaps_loss_weight: float = 1.0

        # Pretrained weights
        self._pretrained_backbone_weights: Optional[str] = None
        self._pretrained_head_weights: Optional[str] = None
        self._use_imagenet_pretrained: bool = True

        # Trainer
        self._optimizer_name: str = "Adam"
        self._amsgrad: bool = False
        self._trainer_accelerator: str = "auto"
        self._trainer_devices: Any = "auto"
        self._save_ckpt: bool = True
        self._save_top_k: int = 1
        self._save_last: Optional[bool] = True  # Web app default
        self._ckpt_dir: str = "./models"  # Web app default
        self._run_name: Optional[str] = None
        self._resume_ckpt_path: Optional[str] = None
        self._seed: Optional[int] = None
        self._min_train_steps_per_epoch: int = 200
        self._num_workers: int = 0
        self._enable_progress_bar: bool = True
        self._visualize_preds_during_training: bool = True
        self._keep_viz: bool = False

        # LR scheduler: name + per-branch params
        self._lr_scheduler: str = (
            "reduce_lr_on_plateau"  # or step_lr/cosine_annealing_warmup/linear_warmup_linear_decay/none
        )
        self._reduce_lr_threshold: float = 1e-6
        self._reduce_lr_threshold_mode: str = "abs"
        self._reduce_lr_cooldown: int = 3
        self._reduce_lr_patience: int = 5
        self._reduce_lr_factor: float = 0.5
        self._reduce_lr_min: float = 1e-8
        self._step_lr_step_size: int = 10
        self._step_lr_gamma: float = 0.1
        self._cosine_warmup_epochs: int = 5
        self._cosine_warmup_start_lr: float = 0.0
        self._cosine_eta_min: float = 0.0
        self._linear_warmup_epochs: int = 5
        self._linear_warmup_start_lr: float = 0.0
        self._linear_end_lr: float = 0.0

        # Online hard keypoint mining
        self._enable_ohkm: bool = False
        self._ohkm_ratio: float = 2.0
        self._ohkm_min_hard: int = 2
        self._ohkm_max_hard: Optional[int] = None
        self._ohkm_loss_scale: float = 5.0

        # WandB
        self._enable_wandb: bool = False
        self._wandb_entity: Optional[str] = None
        self._wandb_project: str = "sleap-training"
        self._wandb_name: Optional[str] = None
        self._wandb_api_key: Optional[str] = None
        self._wandb_mode: Optional[str] = None
        self._wandb_viz_enabled: bool = True
        self._wandb_save_viz: bool = False

        # Eval
        self._enable_eval: bool = False
        self._eval_frequency: int = 1
        self._eval_oks_stddev: float = 0.025
        self._eval_match_threshold: float = 50.0

        # Data pipeline / caching
        self._data_pipeline_fw: str = "torch_dataset"
        self._cache_img_path: Optional[str] = None
        self._use_existing_imgs: bool = False
        self._delete_cache_imgs_after_training: bool = True
        self._parallel_caching: bool = True
        self._cache_workers: int = 0

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
            base_max_stride = 32
            self._filters = 24
            self._filters_rate = 1.5
        else:
            base_max_stride = 16
            self._filters = 32
            self._filters_rate = 1.5  # Web app default

        # Channel configuration: only emit ``ensure_rgb``/``ensure_grayscale``
        # when an actual conversion is required. Web app behavior:
        #   ensure_rgb        = True iff (user wants 3ch OR pretrained) AND original is 1ch
        #   ensure_grayscale  = True iff user wants 1ch AND original is 3ch
        # On auto-config the user keeps the original channels, so both are
        # False unless the user later switches to a pretrained backbone.
        is_pretrained = "convnext" in self._backbone or "swint" in self._backbone
        self._ensure_rgb = bool(is_pretrained and self.stats.num_channels == 1)
        self._ensure_grayscale = False

        # Default max_stride uses the web-app bucket on avg bbox diagonal
        # at the recommender's input scale (``setDefaultParameters`` in
        # app.html runs at scale=1.0). Compute BEFORE applying the centroid
        # 0.5 scale so we match the web app's default for top-down models.
        bucket_stride = recommend_default_max_stride(
            self.stats.avg_bbox_diagonal, self._input_scale
        )

        # Top-down centroid scale override (web app applies this in
        # selectModelType, after setDefaultParameters has already picked
        # max_stride at scale=1.0). Don't recompute the bucket here.
        if self._pipeline in ("centroid", "multi_class_topdown"):
            self._input_scale = 0.5
            self._sigma = 5.0
            self._output_stride = 2

        # Floor by RF coverage of the largest instance at the (possibly
        # lowered) scale so we never under-provision.
        scaled_max_animal_size = self.stats.max_bbox_size * self._input_scale
        coverage_stride = compute_max_stride_for_animal_size(scaled_max_animal_size)
        self._max_stride = max(base_max_stride, bucket_stride, coverage_stride)

        # Override the recommender's 1.5x crop size with the augmentation-aware
        # canonical formula for top-down centered-instance pipelines.
        if self._pipeline in ("centered_instance", "multi_class_topdown") or (
            rec.pipeline.requires_second_model
            and rec.pipeline.second_model_type == "centered_instance"
        ):
            ci_max_stride = 16  # CI default; instance crops don't need deeper RF
            rot_min, rot_max = self._rotation_range
            scale_min, scale_max_aug = self._scale_range
            self._crop_size = compute_suggested_crop_size(
                self.stats.max_bbox_size,
                max_stride=ci_max_stride,
                use_augmentation=self._use_augmentations,
                rotation_max=max(abs(rot_min), abs(rot_max)),
                scale_max=max(scale_min, scale_max_aug),
            )

        self._recommendation = rec
        return self

    # Fluent setters for all parameters

    def pipeline(self, pipeline: PipelineType) -> "ConfigGenerator":
        """Set the pipeline type.

        Pipeline-specific defaults match the web app:

        - ``centroid``: centroid stage of top-down — main preprocessing,
          scale=0.5, sigma=5.0, output_stride=2.
        - ``centered_instance``, ``multi_class_topdown``: CI stage of top-down —
          CI preprocessing (crop_size), scale=1.0, sigma=2.5, output_stride=2.
        - ``single_instance``, ``bottomup``, ``multi_class_bottomup``: main
          preprocessing, scale=1.0, sigma=5.0, output_stride=2.

        Args:
            pipeline: One of the six canonical pipeline types.

        Returns:
            self for method chaining.
        """
        self._pipeline = pipeline

        if pipeline == "centroid":
            self._input_scale = 0.5
            self._sigma = 5.0
            self._output_stride = 2
            self._recalculate_max_stride()
        elif pipeline in ("centered_instance", "multi_class_topdown"):
            self._input_scale = 1.0
            self._sigma = 2.5
            self._output_stride = 2
            # CI stages operate on cropped instances; max_stride=16 is the
            # web-app default ("ci-max-stride" HTML).
            self._max_stride = 16
        else:  # single_instance, bottomup, multi_class_bottomup
            self._input_scale = 1.0
            self._sigma = 5.0
            self._output_stride = 2
            self._recalculate_max_stride()

        return self

    def _recalculate_max_stride(self) -> None:
        """Refloor max_stride by RF coverage at the current scale.

        Web-app behavior: ``setDefaultParameters`` picks max_stride from the
        bucket at scale=1.0. ``selectModelType`` may then change the scale
        but does NOT rebucket. So here we only floor by RF coverage so that
        a deep enough network is still chosen if the user picks a pipeline
        with a smaller scale; we do not downgrade ``_max_stride``.
        """
        base_max_stride = 32 if "large_rf" in self._backbone else 16
        scaled_max_animal_size = self.stats.max_bbox_size * self._input_scale
        coverage_stride = compute_max_stride_for_animal_size(scaled_max_animal_size)
        self._max_stride = max(base_max_stride, coverage_stride, self._max_stride)

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

    @property
    def is_topdown(self) -> bool:
        """Check if current pipeline is top-down (requires 2 models)."""
        return self._pipeline in ("centroid", "multi_class_topdown")

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
        """Build data configuration section.

        Mirrors the web app's ``generateConfigYaml`` data_config block
        (``app.html``) and matches the canonical ``DataConfig`` /
        ``PreprocessingConfig`` / ``AugmentationConfig`` schemas in
        ``sleap_nn/config/data_config.py``.
        """
        is_centered_instance = self._pipeline in (
            "centered_instance",
            "multi_class_topdown",
        )

        # Preprocessing block depends on whether this is a CI model.
        preprocessing: dict = {
            "ensure_rgb": self._ensure_rgb,
            "ensure_grayscale": self._ensure_grayscale,
        }
        if is_centered_instance:
            preprocessing["crop_size"] = self._crop_size
            preprocessing["min_crop_size"] = self._min_crop_size
            preprocessing["crop_padding"] = self._crop_padding
            preprocessing["scale"] = 1.0
        else:
            preprocessing["max_height"] = (
                self._max_height
                if self._max_height is not None
                else getattr(self.stats, "max_height", None)
            )
            preprocessing["max_width"] = (
                self._max_width
                if self._max_width is not None
                else getattr(self.stats, "max_width", None)
            )
            preprocessing["scale"] = self._input_scale
            preprocessing["crop_size"] = None

        config = {
            "train_labels_path": [str(self.slp_path)],
            "val_labels_path": None,
            "validation_fraction": self._validation_fraction,
            "test_file_path": None,
            "provider": "LabelsReader",
            "user_instances_only": True,
            "data_pipeline_fw": self._data_pipeline_fw,
            "cache_img_path": self._cache_img_path,
            "use_existing_imgs": self._use_existing_imgs,
            "delete_cache_imgs_after_training": self._delete_cache_imgs_after_training,
            "parallel_caching": self._parallel_caching,
            "cache_workers": self._cache_workers,
            "preprocessing": preprocessing,
            "use_augmentations_train": self._use_augmentations,
            "augmentation_config": self._build_augmentation_config(),
        }
        return config

    def _build_augmentation_config(self) -> Optional[dict]:
        """Build the structured augmentation_config block.

        Returns ``None`` if augmentations are disabled or all sliders are zero.
        Otherwise emits ``intensity`` and/or ``geometric`` subblocks with the
        canonical field names from ``sleap_nn/config/data_config.py``
        (``IntensityConfig``, ``GeometricConfig``).
        """
        if not self._use_augmentations:
            return None

        rot_min, rot_max = self._rotation_range
        scale_min, scale_max = self._scale_range
        has_rotation = (rot_min, rot_max) != (0.0, 0.0)
        has_scale = (scale_min, scale_max) != (1.0, 1.0)
        has_translate = self._translate > 0
        has_brightness = self._brightness > 0
        has_contrast = self._contrast > 0

        block: dict = {}

        if has_brightness or has_contrast:
            intensity: dict = {}
            if has_contrast:
                intensity["contrast_min"] = round(1 - self._contrast, 2)
                intensity["contrast_max"] = round(1 + self._contrast, 2)
                intensity["contrast_p"] = 1.0
            if has_brightness:
                intensity["brightness_min"] = round(1 - self._brightness, 2)
                intensity["brightness_max"] = round(1 + self._brightness, 2)
                intensity["brightness_p"] = 1.0
            block["intensity"] = intensity

        if has_rotation or has_scale or has_translate:
            geometric: dict = {}
            if has_rotation:
                geometric["rotation_min"] = float(rot_min)
                geometric["rotation_max"] = float(rot_max)
                geometric["rotation_p"] = 1.0
            if has_scale:
                geometric["scale_min"] = float(scale_min)
                geometric["scale_max"] = float(scale_max)
                geometric["scale_p"] = 1.0
            if has_translate:
                geometric["translate_width"] = float(self._translate)
                geometric["translate_height"] = float(self._translate)
                geometric["translate_p"] = 1.0
            block["geometric"] = geometric

        return block or None

    def _build_model_config(self) -> dict:
        """Build model configuration section.

        Emits all three backbone keys (unet, convnext, swint) with exactly one
        non-null value, mirroring the web app and matching the canonical
        ``BackboneConfig`` ``oneof`` schema in
        ``sleap_nn/config/model_config.py``.
        """
        is_pretrained = self._backbone.startswith(
            "convnext"
        ) or self._backbone.startswith("swint")
        # Pretrained backbones force RGB.
        if is_pretrained:
            in_channels = 3
        else:
            in_channels = 3 if self._ensure_rgb else 1

        backbone_config: dict = {"unet": None, "convnext": None, "swint": None}

        if self._backbone.startswith("unet"):
            backbone_config["unet"] = {
                "in_channels": in_channels,
                "kernel_size": 3,
                "filters": self._filters,
                "filters_rate": self._filters_rate,
                "max_stride": self._max_stride,
                "output_stride": self._output_stride,
            }
        elif self._backbone.startswith("convnext"):
            model_type = self._backbone.replace("convnext_", "") or "tiny"
            weights_map = {
                "tiny": "ConvNeXt_Tiny_Weights",
                "small": "ConvNeXt_Small_Weights",
                "base": "ConvNeXt_Base_Weights",
                "large": "ConvNeXt_Large_Weights",
            }
            backbone_config["convnext"] = {
                "model_type": model_type,
                "pre_trained_weights": (
                    weights_map.get(model_type, "ConvNeXt_Tiny_Weights")
                    if self._use_imagenet_pretrained
                    else None
                ),
                "in_channels": in_channels,
                "max_stride": 32,
                "output_stride": self._output_stride,
            }
        elif self._backbone.startswith("swint"):
            model_type = self._backbone.replace("swint_", "") or "tiny"
            weights_map = {
                "tiny": "Swin_T_Weights",
                "small": "Swin_S_Weights",
                "base": "Swin_B_Weights",
            }
            backbone_config["swint"] = {
                "model_type": model_type,
                "pre_trained_weights": (
                    weights_map.get(model_type, "Swin_T_Weights")
                    if self._use_imagenet_pretrained
                    else None
                ),
                "in_channels": in_channels,
                "max_stride": 32,
                "output_stride": self._output_stride,
            }

        return {
            "init_weights": "default",
            "pretrained_backbone_weights": self._pretrained_backbone_weights,
            "pretrained_head_weights": self._pretrained_head_weights,
            "backbone_config": backbone_config,
            "head_configs": self._build_head_config(),
        }

    def _build_head_config(self) -> dict:
        """Build head configuration based on pipeline type.

        All six head keys are emitted with exactly one non-null value, matching
        the canonical ``HeadConfig`` ``oneof`` schema and the web app's
        ``generateConfigYaml`` head section.
        """
        part_names = list(self.stats.node_names) if self.stats else []
        edges = list(getattr(self.stats, "edges", []) or [])

        head_configs = {
            "single_instance": None,
            "centroid": None,
            "centered_instance": None,
            "bottomup": None,
            "multi_class_bottomup": None,
            "multi_class_topdown": None,
        }

        if self._pipeline == "single_instance":
            head_configs["single_instance"] = {
                "confmaps": {
                    "part_names": part_names,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                }
            }

        elif self._pipeline == "centroid":
            head_configs["centroid"] = {
                "confmaps": {
                    "anchor_part": self._anchor_part,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                }
            }

        elif self._pipeline == "centered_instance":
            head_configs["centered_instance"] = {
                "confmaps": {
                    "part_names": part_names,
                    "anchor_part": self._anchor_part,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                    "loss_weight": 1.0,
                }
            }

        elif self._pipeline == "bottomup":
            head_configs["bottomup"] = {
                "confmaps": {
                    "part_names": part_names,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                    "loss_weight": self._confmaps_loss_weight,
                },
                "pafs": {
                    "edges": edges,
                    "sigma": self._paf_sigma,
                    "output_stride": self._paf_output_stride,
                    "loss_weight": self._paf_loss_weight,
                },
            }

        elif self._pipeline == "multi_class_bottomup":
            head_configs["multi_class_bottomup"] = {
                "confmaps": {
                    "part_names": part_names,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                    "loss_weight": self._mc_confmaps_loss_weight,
                },
                "class_maps": {
                    "classes": None,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                    "loss_weight": self._class_loss_weight,
                },
            }

        elif self._pipeline == "multi_class_topdown":
            head_configs["multi_class_topdown"] = {
                "confmaps": {
                    "part_names": part_names,
                    "anchor_part": self._anchor_part,
                    "sigma": self._sigma,
                    "output_stride": self._output_stride,
                    "loss_weight": self._mc_confmaps_loss_weight,
                },
                "class_vectors": {
                    "classes": None,
                    "num_fc_layers": self._class_fc_layers,
                    "num_fc_units": self._class_fc_units,
                    "global_pool": True,
                    "output_stride": self._max_stride,
                    "loss_weight": self._class_loss_weight,
                },
            }

        return head_configs

    def _build_trainer_config(self) -> dict:
        """Build trainer configuration section.

        Matches the canonical ``TrainerConfig`` in
        ``sleap_nn/config/trainer_config.py`` and the web app's
        ``generateConfigYaml`` trainer block.
        """
        config: dict = {
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
            "model_ckpt": {
                "save_top_k": self._save_top_k,
                "save_last": self._save_last,
            },
            "trainer_devices": self._trainer_devices,
            "trainer_accelerator": self._trainer_accelerator,
            "enable_progress_bar": self._enable_progress_bar,
            "min_train_steps_per_epoch": self._min_train_steps_per_epoch,
            "visualize_preds_during_training": self._visualize_preds_during_training,
            "keep_viz": self._keep_viz,
            "max_epochs": self._max_epochs,
            "seed": self._seed,
            "use_wandb": self._enable_wandb,
            "save_ckpt": self._save_ckpt,
            "ckpt_dir": self._ckpt_dir,
            "run_name": self._run_name,
            "resume_ckpt_path": self._resume_ckpt_path,
            "optimizer_name": self._optimizer_name,
            "optimizer": {
                "lr": self._learning_rate,
                "amsgrad": self._amsgrad,
            },
            "lr_scheduler": self._build_lr_scheduler_config(),
            "early_stopping": {
                "stop_training_on_plateau": self._early_stopping,
                "min_delta": self._early_stopping_min_delta,
                "patience": self._early_stopping_patience,
            },
            "online_hard_keypoint_mining": {
                "online_mining": self._enable_ohkm,
                "hard_to_easy_ratio": self._ohkm_ratio,
                "min_hard_keypoints": self._ohkm_min_hard,
                "max_hard_keypoints": self._ohkm_max_hard,
                "loss_scale": self._ohkm_loss_scale,
            },
        }

        if self._enable_wandb:
            config["wandb"] = {
                "entity": self._wandb_entity,
                "project": self._wandb_project,
                "name": self._wandb_name,
                "api_key": self._wandb_api_key,
                "wandb_mode": self._wandb_mode,
                "viz_enabled": self._wandb_viz_enabled,
                "save_viz_imgs_wandb": self._wandb_save_viz,
            }

        if self._enable_eval:
            config["eval"] = {
                "enabled": True,
                "frequency": self._eval_frequency,
                "oks_stddev": self._eval_oks_stddev,
                "match_threshold": self._eval_match_threshold,
            }

        return config

    def _build_lr_scheduler_config(self) -> dict:
        """Build the lr_scheduler block with one branch active, others null.

        Matches ``LRSchedulerConfig`` (``sleap_nn/config/trainer_config.py``).
        """
        block: dict = {
            "step_lr": None,
            "reduce_lr_on_plateau": None,
            "cosine_annealing_warmup": None,
            "linear_warmup_linear_decay": None,
        }
        if self._lr_scheduler == "step_lr":
            block["step_lr"] = {
                "step_size": self._step_lr_step_size,
                "gamma": self._step_lr_gamma,
            }
        elif self._lr_scheduler == "cosine_annealing_warmup":
            block["cosine_annealing_warmup"] = {
                "warmup_epochs": self._cosine_warmup_epochs,
                "warmup_start_lr": self._cosine_warmup_start_lr,
                "eta_min": self._cosine_eta_min,
            }
        elif self._lr_scheduler == "linear_warmup_linear_decay":
            block["linear_warmup_linear_decay"] = {
                "warmup_epochs": self._linear_warmup_epochs,
                "warmup_start_lr": self._linear_warmup_start_lr,
                "end_lr": self._linear_end_lr,
            }
        elif self._lr_scheduler == "none":
            pass  # all four branches null
        else:
            # Default: reduce_lr_on_plateau
            block["reduce_lr_on_plateau"] = {
                "threshold": self._reduce_lr_threshold,
                "threshold_mode": self._reduce_lr_threshold_mode,
                "cooldown": self._reduce_lr_cooldown,
                "patience": self._reduce_lr_patience,
                "factor": self._reduce_lr_factor,
                "min_lr": self._reduce_lr_min,
            }
        return block

    def build_centroid(self) -> DictConfig:
        """Build the centroid-stage config for a top-down dual emit.

        Always emits a ``centroid`` head with main preprocessing, scale=0.5,
        sigma=5.0, output_stride=2 — matches the web app's centroid stage
        for both ``topdown`` and ``multi_class_topdown`` model types.

        ``max_stride`` is taken from the web-app bucket on the SLP's avg
        bbox diagonal at scale=1.0, then floored by RF coverage at scale=0.5.
        This way the centroid stage's max_stride doesn't depend on prior
        ``pipeline()`` calls — the bucket runs from the dataset stats.

        Returns:
            Centroid-stage configuration.
        """
        orig_pipeline = self._pipeline
        orig_scale = self._input_scale
        orig_sigma = self._sigma
        orig_output_stride = self._output_stride
        orig_max_stride = self._max_stride

        self._pipeline = "centroid"
        self._input_scale = 0.5
        self._sigma = 5.0
        self._output_stride = 2
        # Web-app behavior: max_stride bucket runs at scale=1.0, then the
        # centroid scale (0.5) is applied.
        bucket_stride = recommend_default_max_stride(self.stats.avg_bbox_diagonal, 1.0)
        scaled_max_animal_size = self.stats.max_bbox_size * self._input_scale
        coverage_stride = compute_max_stride_for_animal_size(scaled_max_animal_size)
        base_max_stride = 32 if "large_rf" in self._backbone else 16
        self._max_stride = max(base_max_stride, bucket_stride, coverage_stride)

        config = self.build()

        self._pipeline = orig_pipeline
        self._input_scale = orig_scale
        self._sigma = orig_sigma
        self._output_stride = orig_output_stride
        self._max_stride = orig_max_stride

        return config

    def build_centered_instance(self) -> DictConfig:
        """Build the centered-instance / multi-class-topdown CI config.

        Used as the second config in a top-down dual emit. The head type is
        chosen from the current pipeline:

        - ``multi_class_topdown`` → emits a ``multi_class_topdown`` head
          (with class_vectors), matching the web app's CI tab when the user
          picks the multi-class top-down model type.
        - anything else (including the centroid stage) → emits a
          ``centered_instance`` head.

        Returns:
            Centered-instance / multi-class-topdown CI configuration.
        """
        # Save current state
        orig_pipeline = self._pipeline
        orig_scale = self._input_scale
        orig_sigma = self._sigma
        orig_output_stride = self._output_stride
        orig_max_stride = self._max_stride

        # Switch to CI stage settings (web app's CI tab)
        if orig_pipeline == "multi_class_topdown":
            self._pipeline = "multi_class_topdown"
        else:
            self._pipeline = "centered_instance"
        self._input_scale = 1.0  # Full resolution for instance
        self._sigma = 2.5  # Tighter sigma for instance
        self._output_stride = 2
        self._max_stride = 16  # CI default; crops are size-limited

        # Build the config
        config = self.build()

        # Restore original state
        self._pipeline = orig_pipeline
        self._input_scale = orig_scale
        self._sigma = orig_sigma
        self._output_stride = orig_output_stride
        self._max_stride = orig_max_stride

        return config

    def save(self, path: str) -> "ConfigGenerator":
        """Save configuration to YAML file(s).

        For top-down models (centroid, multi_class_topdown), saves TWO files:
        - {path}_centroid.yaml
        - {path}_centered_instance.yaml

        For other models, saves a single file.

        Args:
            path: Output path for YAML file (extension will be adjusted for top-down).

        Returns:
            self for method chaining.
        """
        path_obj = Path(path)
        stem = path_obj.stem
        suffix = path_obj.suffix or ".yaml"
        parent = path_obj.parent

        if self.is_topdown:
            # Save centroid stage (always a ``centroid`` head)
            centroid_path = parent / f"{stem}_centroid{suffix}"
            centroid_config = self.build_centroid()
            OmegaConf.save(centroid_config, centroid_path)

            # Save CI stage (centered_instance or multi_class_topdown head,
            # depending on the original pipeline).
            instance_path = parent / f"{stem}_centered_instance{suffix}"
            instance_config = self.build_centered_instance()
            OmegaConf.save(instance_config, instance_path)
        else:
            config = self.build()
            OmegaConf.save(config, path)

        return self

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        For top-down models, returns both centroid and centered_instance configs.

        Returns:
            YAML string representation.
        """
        if self.is_topdown:
            centroid_yaml = OmegaConf.to_yaml(self.build())
            instance_yaml = OmegaConf.to_yaml(self.build_centered_instance())
            return f"# === CENTROID CONFIG ===\n{centroid_yaml}\n\n# === CENTERED INSTANCE CONFIG ===\n{instance_yaml}"
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
