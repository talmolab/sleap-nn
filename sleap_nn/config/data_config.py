"""Serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

from attrs import define, field, validators
from omegaconf import MISSING
from typing import Optional, Tuple, Any, List
from loguru import logger
import sleap_io as sio
import yaml
from sleap_io.io.skeleton import SkeletonDecoder, SkeletonYAMLEncoder


@define
class PreprocessingConfig:
    """Configuration of Preprocessing.

    Attributes:
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
        max_height: (int) Maximum height the image should be padded to. If not provided, the original image size will be retained. Default: None.
        max_width: (int) Maximum width the image should be padded to. If not provided, the original image size will be retained. Default: None.
        scale: (float or List[float]) Factor to resize the image dimensions by, specified as either a float scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both dimensions are resized by the same factor.
        crop_hw: (Tuple[int]) Crop height and width of each instance (h, w) for centered-instance model. If None, this would be automatically computed based on the largest instance in the sio.Labels file.
        min_crop_size: (int) Minimum crop size to be used if crop_hw is None.
    """

    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = field(
        default=1.0, validator=lambda instance, attr, value: instance.validate_scale()
    )
    crop_hw: Optional[Tuple[int, int]] = None
    min_crop_size: Optional[int] = 100  # to help app work in case of error

    def validate_scale(self):
        """Scale Validation.

        Ensures PreprocessingConfig's scale is a float>=0 or list of floats>=0
        """
        if isinstance(self.scale, float) and self.scale >= 0:
            return
        if isinstance(self.scale, list) and all(
            isinstance(x, float) and x >= 0 for x in self.scale
        ):
            return
        message = "PreprocessingConfig's scale must be a float or a list of floats."
        logger.error(message)
        raise ValueError(message)


def validate_proportion(instance, attribute, value):
    """General Proportion Validation.

    Ensures all proportions are a 0<=float<=1.0
    """
    if not (0.0 <= value <= 1.0):
        message = f"{attribute.name} must be between 0.0 and 1.0, got {value}"
        logger.error(message)
        raise ValueError(message)


@define
class IntensityConfig:
    """Configuration of Intensity (Optional).

    Attributes:
        uniform_noise_min: (float) Minimum value for uniform noise (uniform_noise_min >=0).
        uniform_noise_max: (float) Maximum value for uniform noise (uniform_noise_max <>=1).
        uniform_noise_p: (float) Probability of applying random uniform noise. Default=0.0
        gaussian_noise_mean: (float) The mean of the gaussian noise distribution.
        gaussian_noise_std: (float) The standard deviation of the gaussian noise distribution.
        gaussian_noise_p: (float) Probability of applying random gaussian noise. Default=0.0
        contrast_min: (float) Minimum contrast factor to apply. Default: 0.5.
        contrast_max: (float) Maximum contrast factor to apply. Default: 2.0.
        contrast_p: (float) Probability of applying random contrast. Default=0.0
        brightness: (float) The brightness factor to apply. Default: (1.0, 1.0).
        brightness_p: (float) Probability of applying random brightness. Default=0.0
    """

    uniform_noise_min: float = field(default=0.0, validator=validators.ge(0))
    uniform_noise_max: float = field(default=1.0, validator=validators.le(1))
    uniform_noise_p: float = field(default=0.0, validator=validate_proportion)
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 1.0
    gaussian_noise_p: float = field(default=0.0, validator=validate_proportion)
    contrast_min: float = field(default=0.5, validator=validators.ge(0))
    contrast_max: float = field(default=2.0, validator=validators.ge(0))
    contrast_p: float = field(default=0.0, validator=validate_proportion)
    brightness: Tuple[float, float] = (1.0, 1.0)
    brightness_p: float = field(default=0.0, validator=validate_proportion)


@define
class GeometricConfig:
    """Configuration of Geometric (Optional).

    Attributes:
        rotation: (float) Angles in degrees as a scalar float of the amount of rotation. A random angle in (-rotation, rotation) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation.
        scale: (List[float]) scaling factor interval. If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b. If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d Default: None.
        translate_width: (float) Maximum absolute fraction for horizontal translation. For example, if translate_width=a, then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a. Will not translate by default.
        translate_height: (float) Maximum absolute fraction for vertical translation. For example, if translate_height=a, then vertical shift is randomly sampled in the range -img_height * a < dy < img_height * a. Will not translate by default.
        affine_p: (float) Probability of applying random affine transformations. Default=0.0
        erase_scale_min: (float) Minimum value of range of proportion of erased area against input image. Default: 0.0001.
        erase_scale_max: (float) Maximum value of range of proportion of erased area against input image. Default: 0.01.
        erase_ratio_min: (float) Minimum value of range of aspect ratio of erased area. Default: 1.
        erase_ratio_max: (float) Maximum value of range of aspect ratio of erased area. Default: 1.
        erase_p: (float) Probability of applying random erase. Default=0.0
        mixup_lambda: (list) min-max value of mixup strength. Default is [0.01, 0.05]. Default: None.
        mixup_p: (float) Probability of applying random mixup v2. Default=0.0
    """

    rotation: float = 15.0
    scale: Optional[List[float]] = (0.9, 1.1)
    translate_width: float = 0.0
    translate_height: float = 0.0
    affine_p: float = field(default=0.0, validator=validate_proportion)
    erase_scale_min: float = 0.0001
    erase_scale_max: float = 0.01
    erase_ratio_min: float = 1.0
    erase_ratio_max: float = 1.0
    erase_p: float = field(default=0.0, validator=validate_proportion)
    mixup_lambda: List[float] = [0.01, 0.05]
    mixup_p: float = field(default=0.0, validator=validate_proportion)


@define
class AugmentationConfig:
    """Configuration of Augmentation.

    Attributes:
        intensity: Configuration options for intensity-based augmentations like brightness, contrast, etc. If None, no intensity augmentations will be applied.
        geometric: Configuration options for geometric augmentations like rotation, scaling, translation etc. If None, no geometric augmentations will be applied.
    """

    intensity: Optional[IntensityConfig] = None
    geometric: Optional[GeometricConfig] = None


@define
class DataConfig:
    """Data configuration.

    train_labels_path: (List[str]) List of paths to training data (.slp file)
    val_labels_path: (List[str]) List of paths to validation data (.slp file)
    validation_fraction: Float between 0 and 1 specifying the fraction of the
        training set to sample for generating the validation set. The remaining
        labeled frames will be left in the training set. If the `validation_labels`
        are already specified, this has no effect. Default: 0.1.
    test_file_path: (str) Path to test dataset (`.slp` file or `.mp4` file). *Note*: This is used only
        with CLI to get evaluation on test set after training is completed.
    provider: (str) Provider class to read the input sleap files. Only "LabelsReader"
        supported for the training pipeline.
    user_instances_only: (bool) True if only user labeled instances should be used for
        training. If False, both user labeled and predicted instances would be used. Default: True.
    data_pipeline_fw: Framework to create the data loaders. One of [`torch_dataset`,
            `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. Default: "torch_dataset".
    cache_img_path: Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline
        framework. If `None`, the path provided in `trainer_config.save_ckpt` is used (else working dir is used). The `train_imgs` and `val_imgs` dirs are created inside this path. Default: None.
    use_existing_imgs: Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` framework. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs.
        Default: False.
    delete_cache_imgs_after_training: If `False`, the images (torch_dataset_cache_img_disk) are
        retained after training. Else, the files are deleted. Default: True.
    preprocessing: Configuration options related to data preprocessing.
    use_augmentations_train: (bool) True if the data augmentation should be applied to the training data, else False.
    augmentation_config: Configurations related to augmentation.
        # Your list specifies "(only if use_augmentations_train is True)"
    skeletons: skeleton configuration for the `.slp` file. This will be pulled from the
        train dataset and saved to the `training_config.yaml`
    """

    train_labels_path: List[str] = []
    val_labels_path: Optional[List[str]] = None  # TODO : revisit MISSING!
    validation_fraction: float = 0.1
    test_file_path: Optional[str] = None
    provider: str = "LabelsReader"
    user_instances_only: bool = True
    data_pipeline_fw: str = "torch_dataset"
    cache_img_path: Optional[str] = None
    use_existing_imgs: bool = False
    delete_cache_imgs_after_training: bool = True
    preprocessing: PreprocessingConfig = field(factory=PreprocessingConfig)
    use_augmentations_train: bool = False
    augmentation_config: Optional[AugmentationConfig] = None
    skeletons: Optional[dict] = None


def data_mapper(legacy_config: dict) -> DataConfig:
    """Maps the legacy data configuration to the new data configuration.

    Args:
        legacy_config: A dictionary containing the legacy data configuration.

    Returns:
        An instance of `DataConfig` with the mapped configuration.
    """
    legacy_config_data = legacy_config.get("data", {})
    legacy_config_optimization = legacy_config.get("optimization", {})
    train_labels_path = legacy_config_data.get("labels", {}).get(
        "training_labels", None
    )
    val_labels_path = legacy_config_data.get("labels", {}).get(
        "validation_labels", None
    )

    # get skeleton(s)
    json_skeletons = legacy_config_data.get("labels", {}).get("skeletons", None)
    skeletons_dict = None
    if json_skeletons is not None:
        skeletons = SkeletonDecoder().decode(json_skeletons)
        skeletons_dict = yaml.safe_load(SkeletonYAMLEncoder().encode(skeletons))

    return DataConfig(
        train_labels_path=[train_labels_path] if train_labels_path is not None else [],
        val_labels_path=[val_labels_path] if val_labels_path is not None else [],
        validation_fraction=legacy_config_data.get("labels", {}).get(
            "validation_fraction", None
        ),
        test_file_path=legacy_config_data.get("labels", {}).get("test_labels", None),
        preprocessing=PreprocessingConfig(
            ensure_rgb=legacy_config_data.get("preprocessing", {}).get(
                "ensure_rgb", False
            ),
            ensure_grayscale=legacy_config_data.get("preprocessing", {}).get(
                "ensure_grayscale", False
            ),
            max_height=legacy_config_data.get("preprocessing", {}).get(
                "target_height", None
            ),
            max_width=legacy_config_data.get("preprocessing", {}).get(
                "target_width", None
            ),
            scale=legacy_config_data.get("preprocessing", {}).get("input_scaling", 1.0),
            crop_hw=(
                (
                    legacy_config_data.get("instance_cropping", {}).get(
                        "crop_size", None
                    ),
                    legacy_config_data.get("instance_cropping", {}).get(
                        "crop_size", None
                    ),
                )
                if legacy_config_data.get("instance_cropping", {}).get(
                    "crop_size", None
                )
                is not None
                else None
            ),
        ),
        augmentation_config=(
            AugmentationConfig(
                intensity=IntensityConfig(
                    uniform_noise_min=legacy_config_optimization.get(
                        "augmentation_config", {}
                    ).get("uniform_noise_min_val", 0.0),
                    uniform_noise_max=min(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "uniform_noise_max_val", 1.0
                        ),
                        1.0,
                    ),
                    uniform_noise_p=float(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "uniform_noise", 1.0
                        )
                    ),
                    gaussian_noise_mean=legacy_config_optimization.get(
                        "augmentation_config", {}
                    ).get("gaussian_noise_mean", 0.0),
                    gaussian_noise_std=legacy_config_optimization.get(
                        "augmentation_config", {}
                    ).get("gaussian_noise_stddev", 1.0),
                    gaussian_noise_p=float(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "gaussian_noise", 1.0
                        )
                    ),
                    contrast_min=legacy_config_optimization.get(
                        "augmentation_config", {}
                    ).get("contrast_min_gamma", 0.5),
                    contrast_max=legacy_config_optimization.get(
                        "augmentation_config", {}
                    ).get("contrast_max_gamma", 2.0),
                    contrast_p=float(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "contrast", 1.0
                        )
                    ),
                    brightness=(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "brightness_min_val", 1.0
                        ),
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "brightness_max_val", 1.0
                        ),
                    ),
                    brightness_p=float(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "brightness", 1.0
                        )
                    ),
                ),
                geometric=GeometricConfig(
                    rotation=(
                        legacy_config_optimization.get("augmentation_config", {}).get(
                            "rotation_max_angle", 15.0
                        )
                        if legacy_config_optimization.get(
                            "augmentation_config", {}
                        ).get("rotate", True)
                        else 0
                    ),
                    scale=(
                        (
                            legacy_config_optimization.get(
                                "augmentation_config", {}
                            ).get("scale_min", 0.9),
                            legacy_config_optimization.get(
                                "augmentation_config", {}
                            ).get("scale_max", 1.1),
                        )
                        if legacy_config_optimization.get(
                            "augmentation_config", {}
                        ).get("scale", False)
                        else (1.0, 1.0)
                    ),
                    affine_p=(
                        1.0
                        if any(
                            [
                                legacy_config_optimization.get(
                                    "augmentation_config", {}
                                ).get("rotate", True),
                                legacy_config_optimization.get(
                                    "augmentation_config", {}
                                ).get("scale", False),
                            ]
                        )
                        else 0.0
                    ),
                ),
            )
        ),
        use_augmentations_train=True,
        skeletons=skeletons_dict,
    )
