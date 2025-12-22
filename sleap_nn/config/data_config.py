"""Serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

from attrs import define, field, validators
from omegaconf import MISSING
from typing import Optional, Tuple, Any, List, Union
from loguru import logger
import sleap_io as sio
import yaml
from sleap_io.io.skeleton import SkeletonDecoder, SkeletonYAMLEncoder


@define
class PreprocessingConfig:
    """Configuration of Preprocessing.

    Attributes:
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. *Default*: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. *Default*: `False`.
        max_height: (int) Maximum height the original image should be resized and padded to. If not provided, the original image size will be retained. *Default*: `None`.
        max_width: (int) Maximum width the original image should be resized and padded to. If not provided, the original image size will be retained. *Default*: `None`.
        scale: (float) Factor to resize the image dimensions by, specified as a float. *Default*: `1.0`.
        crop_size: (int) Crop size of each instance for centered-instance model. If `None`, this would be automatically computed based on the largest instance in the `sio.Labels` file.
            If `scale` is provided, then the cropped image will be resized according to `scale`.*Default*: `None`.
        min_crop_size: (int) Minimum crop size to be used if `crop_size` is `None`. *Default*: `100`.
        crop_padding: (int) Padding in pixels to add around the instance bounding box when computing crop size.
            If `None`, padding is auto-computed based on augmentation settings (rotation/scale).
            Only used when `crop_size` is `None`. *Default*: `None`.
    """

    ensure_rgb: bool = False
    ensure_grayscale: bool = False
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = field(
        default=1.0, validator=lambda instance, attr, value: instance.validate_scale()
    )
    crop_size: Optional[int] = None
    min_crop_size: Optional[int] = 100  # to help app work in case of error
    crop_padding: Optional[int] = None

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
        uniform_noise_min: (float) Minimum value for uniform noise (uniform_noise_min >=0). *Default*: `0.0`.
        uniform_noise_max: (float) Maximum value for uniform noise (uniform_noise_max <>=1). *Default*: `1.0`.
        uniform_noise_p: (float) Probability of applying random uniform noise. *Default*: `0.0`.
        gaussian_noise_mean: (float) The mean of the gaussian noise distribution. *Default*: `0.0`.
        gaussian_noise_std: (float) The standard deviation of the gaussian noise distribution. *Default*: `1.0`.
        gaussian_noise_p: (float) Probability of applying random gaussian noise. *Default*: `0.0`.
        contrast_min: (float) Minimum contrast factor to apply. *Default*: `0.9`.
        contrast_max: (float) Maximum contrast factor to apply. *Default*: `1.1`.
        contrast_p: (float) Probability of applying random contrast. *Default*: `0.0`.
        brightness_min: (float) Minimum brightness factor to apply. *Default*: `1.0`.
        brightness_max: (float) Maximum brightness factor to apply. *Default*: `1.0`.
        brightness_p: (float) Probability of applying random brightness. *Default*: `0.0`.
    """

    uniform_noise_min: float = field(default=0.0, validator=validators.ge(0))
    uniform_noise_max: float = field(default=1.0, validator=validators.le(1))
    uniform_noise_p: float = field(default=0.0, validator=validate_proportion)
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 1.0
    gaussian_noise_p: float = field(default=0.0, validator=validate_proportion)
    contrast_min: float = field(default=0.9, validator=validators.ge(0))
    contrast_max: float = field(default=1.1, validator=validators.ge(0))
    contrast_p: float = field(default=0.0, validator=validate_proportion)
    brightness_min: float = field(default=1.0, validator=validators.ge(0))
    brightness_max: float = field(default=1.0, validator=validators.le(2))
    brightness_p: float = field(default=0.0, validator=validate_proportion)


@define
class GeometricConfig:
    """Configuration of Geometric (Optional).

    Attributes:
        rotation_min: (float) Minimum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. *Default*: `-15.0`.
        rotation_max: (float) Maximum rotation angle in degrees. A random angle in (rotation_min, rotation_max) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation. *Default*: `15.0`.
        rotation_p: (float, optional) Probability of applying random rotation independently. If set, rotation is applied separately from scale/translate. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `None`.
        scale_min: (float) Minimum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. *Default*: `0.9`.
        scale_max: (float) Maximum scaling factor. If scale_min and scale_max are provided, the scale is randomly sampled from the range scale_min <= scale <= scale_max for isotropic scaling. *Default*: `1.1`.
        scale_p: (float, optional) Probability of applying random scaling independently. If set, scaling is applied separately from rotation/translate. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `None`.
        translate_width: (float) Maximum absolute fraction for horizontal translation. For example, if translate_width=a, then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a. Will not translate by default. *Default*: `0.0`.
        translate_height: (float) Maximum absolute fraction for vertical translation. For example, if translate_height=a, then vertical shift is randomly sampled in the range -img_height * a < dy < img_height * a. Will not translate by default. *Default*: `0.0`.
        translate_p: (float, optional) Probability of applying random translation independently. If set, translation is applied separately from rotation/scale. If `None`, falls back to `affine_p` for bundled behavior. *Default*: `None`.
        affine_p: (float) Probability of applying random affine transformations (rotation, scale, translate bundled together). Used for backwards compatibility when individual `*_p` params are not set. *Default*: `0.0`.
        erase_scale_min: (float) Minimum value of range of proportion of erased area against input image. *Default*: `0.0001`.
        erase_scale_max: (float) Maximum value of range of proportion of erased area against input image. *Default*: `0.01`.
        erase_ratio_min: (float) Minimum value of range of aspect ratio of erased area. *Default*: `1.0`.
        erase_ratio_max: (float) Maximum value of range of aspect ratio of erased area. *Default*: `1.0`.
        erase_p: (float) Probability of applying random erase. *Default*: `1.0`.
        mixup_lambda_min: (float) Minimum mixup strength value. *Default*: `0.01`.
        mixup_lambda_max: (float) Maximum mixup strength value. *Default*: `0.05`.
        mixup_p: (float) Probability of applying random mixup v2. *Default*: `0.0`.
    """

    rotation_min: float = field(default=-15.0, validator=validators.ge(-180))
    rotation_max: float = field(default=15.0, validator=validators.le(180))
    rotation_p: Optional[float] = field(default=None)
    scale_min: float = field(default=0.9, validator=validators.ge(0))
    scale_max: float = field(default=1.1, validator=validators.ge(0))
    scale_p: Optional[float] = field(default=None)
    translate_width: float = 0.0
    translate_height: float = 0.0
    translate_p: Optional[float] = field(default=None)
    affine_p: float = field(default=0.0, validator=validate_proportion)
    erase_scale_min: float = 0.0001
    erase_scale_max: float = 0.01
    erase_ratio_min: float = 1.0
    erase_ratio_max: float = 1.0
    erase_p: float = field(default=0.0, validator=validate_proportion)
    mixup_lambda_min: float = field(default=0.01, validator=validators.ge(0))
    mixup_lambda_max: float = field(default=0.05, validator=validators.le(1))
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


def validate_test_file_path(instance, attribute, value):
    """Validate test_file_path to accept str or List[str].

    Args:
        instance: The instance being validated.
        attribute: The attribute being validated.
        value: The value to validate.

    Raises:
        ValueError: If value is not None, str, or list of strings.
    """
    if value is None:
        return
    if isinstance(value, str):
        return
    if isinstance(value, (list, tuple)) and all(isinstance(p, str) for p in value):
        return
    message = f"{attribute.name} must be a string or list of strings, got {type(value).__name__}"
    logger.error(message)
    raise ValueError(message)


@define
class DataConfig:
    """Data configuration.

    Attributes:
        train_labels_path: (List[str]) List of paths to training data (`.slp` file(s)). *Default*: `None`.
        val_labels_path: (List[str]) List of paths to validation data (`.slp` file(s)). *Default*: `None`.
        validation_fraction: (float) Float between 0 and 1 specifying the fraction of the training set to sample for generating the validation set. The remaining labeled frames will be left in the training set. If the `validation_labels` are already specified, this has no effect. *Default*: `0.1`.
        use_same_data_for_val: (bool) If `True`, use the same data for both training and validation (train = val). Useful for intentional overfitting on small datasets. When enabled, `val_labels_path` and `validation_fraction` are ignored. *Default*: `False`.
        test_file_path: (str or List[str]) Path or list of paths to test dataset(s) (`.slp` file(s) or `.mp4` file(s)). *Note*: This is used only with CLI to get evaluation on test set after training is completed. *Default*: `None`.
        provider: (str) Provider class to read the input sleap files. Only "LabelsReader" is currently supported for the training pipeline. *Default*: `"LabelsReader"`.
        user_instances_only: (bool) `True` if only user labeled instances should be used for training. If `False`, both user labeled and predicted instances would be used. *Default*: `True`.
        data_pipeline_fw: (str) Framework to create the data loaders. One of [`torch_dataset`, `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. *Default*: `"torch_dataset"`. (Note: When using `torch_dataset`, `num_workers` in `trainer_config` should be set to 0 as multiprocessing doesn't work with pickling video backends.)
        cache_img_path: (str) Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline framework. If `None`, the path provided in `trainer_config.save_ckpt` is used. The `train_imgs` and `val_imgs` dirs are created inside this path. *Default*: `None`.
        use_existing_imgs: (bool) Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` frameworks. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs. *Default*: `False`.
        delete_cache_imgs_after_training: (bool) If `False`, the images (torch_dataset_cache_img_disk) are retained after training. Else, the files are deleted. *Default*: `True`.
        preprocessing: Configuration options related to data preprocessing.
        use_augmentations_train: (bool) True if the data augmentation should be applied to the training data, else False. *Default*: `True`.
        augmentation_config: Configurations related to augmentation. (only if `use_augmentations_train` is `True`)
        skeletons: skeleton configuration for the `.slp` file. This will be pulled from the train dataset and saved to the `training_config.yaml`
    """

    train_labels_path: Optional[List[str]] = None
    val_labels_path: Optional[List[str]] = None  # TODO : revisit MISSING!
    validation_fraction: float = 0.1
    use_same_data_for_val: bool = False
    test_file_path: Optional[Any] = field(
        default=None, validator=validate_test_file_path
    )
    provider: str = "LabelsReader"
    user_instances_only: bool = True
    data_pipeline_fw: str = "torch_dataset"
    cache_img_path: Optional[str] = None
    use_existing_imgs: bool = False
    delete_cache_imgs_after_training: bool = True
    preprocessing: PreprocessingConfig = field(factory=PreprocessingConfig)
    use_augmentations_train: bool = True
    augmentation_config: Optional[AugmentationConfig] = None
    skeletons: Optional[list] = None


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
    skeletons_list = None
    if json_skeletons is not None:
        skeletons_list = []
        skeletons = SkeletonDecoder().decode(json_skeletons)
        skeletons = yaml.safe_load(SkeletonYAMLEncoder().encode(skeletons))
        for skl_name in skeletons.keys():
            skl = skeletons[skl_name]
            skl["name"] = skl_name
            skeletons_list.append(skl)

    data_cfg_args = {}
    preprocessing_args = {}
    intensity_args = {}
    geometric_args = {}

    if train_labels_path is not None:
        data_cfg_args["train_labels_path"] = [train_labels_path]
    if val_labels_path is not None:
        data_cfg_args["val_labels_path"] = [val_labels_path]
    if (
        legacy_config_data.get("labels", {}).get("validation_fraction", None)
        is not None
    ):
        data_cfg_args["validation_fraction"] = legacy_config_data["labels"][
            "validation_fraction"
        ]
    if legacy_config_data.get("labels", {}).get("test_labels", None) is not None:
        data_cfg_args["test_file_path"] = legacy_config_data["labels"]["test_labels"]

    # preprocessing
    if legacy_config_data.get("preprocessing", {}).get("ensure_rgb", None) is not None:
        preprocessing_args["ensure_rgb"] = legacy_config_data["preprocessing"][
            "ensure_rgb"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("ensure_grayscale", None)
        is not None
    ):
        preprocessing_args["ensure_grayscale"] = legacy_config_data["preprocessing"][
            "ensure_grayscale"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("target_height", None)
        is not None
    ):
        preprocessing_args["max_height"] = legacy_config_data["preprocessing"][
            "target_height"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("target_width", None)
        is not None
    ):
        preprocessing_args["max_width"] = legacy_config_data["preprocessing"][
            "target_width"
        ]
    if (
        legacy_config_data.get("preprocessing", {}).get("input_scaling", None)
        is not None
    ):
        preprocessing_args["scale"] = legacy_config_data["preprocessing"][
            "input_scaling"
        ]
    if (
        legacy_config_data.get("instance_cropping", {}).get("crop_size", None)
        is not None
    ):
        size = legacy_config_data["instance_cropping"]["crop_size"]
        preprocessing_args["crop_size"] = size

    # augmentation
    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise_min_val", None
        )
        is not None
    ):
        intensity_args["uniform_noise_min"] = legacy_config_optimization[
            "augmentation_config"
        ]["uniform_noise_min_val"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise_max_val", None
        )
        is not None
    ):
        intensity_args["uniform_noise_max"] = min(
            legacy_config_optimization["augmentation_config"]["uniform_noise_max_val"],
            1.0,
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "uniform_noise", None
        )
        is not None
    ):
        intensity_args["uniform_noise_p"] = float(
            legacy_config_optimization["augmentation_config"]["uniform_noise"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise_mean", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_mean"] = legacy_config_optimization[
            "augmentation_config"
        ]["gaussian_noise_mean"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise_stddev", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_std"] = legacy_config_optimization[
            "augmentation_config"
        ]["gaussian_noise_stddev"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "gaussian_noise", None
        )
        is not None
    ):
        intensity_args["gaussian_noise_p"] = float(
            legacy_config_optimization["augmentation_config"]["gaussian_noise"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "contrast_min_gamma", None
        )
        is not None
    ):
        intensity_args["contrast_min"] = legacy_config_optimization[
            "augmentation_config"
        ]["contrast_min_gamma"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "contrast_max_gamma", None
        )
        is not None
    ):
        intensity_args["contrast_max"] = legacy_config_optimization[
            "augmentation_config"
        ]["contrast_max_gamma"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("contrast", None)
        is not None
    ):
        intensity_args["contrast_p"] = float(
            legacy_config_optimization["augmentation_config"]["contrast"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness_min_val", None
        )
        is not None
    ):
        intensity_args["brightness_min"] = min(
            legacy_config_optimization["augmentation_config"]["brightness_min_val"], 2.0
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness_max_val", None
        )
        is not None
    ):
        intensity_args["brightness_max"] = min(
            legacy_config_optimization["augmentation_config"]["brightness_max_val"], 2.0
        )  # kornia brightness_max can only be 2.0

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "brightness", None
        )
        is not None
    ):
        intensity_args["brightness_p"] = float(
            legacy_config_optimization["augmentation_config"]["brightness"]
        )

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "rotation_min_angle", None
        )
        is not None
    ):
        geometric_args["rotation_min"] = legacy_config_optimization[
            "augmentation_config"
        ]["rotation_min_angle"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get(
            "rotation_max_angle", None
        )
        is not None
    ):
        geometric_args["rotation_max"] = legacy_config_optimization[
            "augmentation_config"
        ]["rotation_max_angle"]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("scale_min", None)
        is not None
    ):
        geometric_args["scale_min"] = legacy_config_optimization["augmentation_config"][
            "scale_min"
        ]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("scale_max", None)
        is not None
    ):
        geometric_args["scale_max"] = legacy_config_optimization["augmentation_config"][
            "scale_max"
        ]

    if (
        legacy_config_optimization.get("augmentation_config", {}).get("scale", None)
        is not None
    ):
        geometric_args["scale_min"] = legacy_config_optimization["augmentation_config"][
            "scale_min"
        ]
        geometric_args["scale_max"] = legacy_config_optimization["augmentation_config"][
            "scale_max"
        ]

    geometric_args["affine_p"] = (
        1.0
        if any(
            [
                legacy_config_optimization.get("augmentation_config", {}).get(
                    "rotate", False
                ),
                legacy_config_optimization.get("augmentation_config", {}).get(
                    "scale", False
                ),
            ]
        )
        else 0.0
    )

    data_cfg_args["preprocessing"] = PreprocessingConfig(**preprocessing_args)
    data_cfg_args["augmentation_config"] = AugmentationConfig(
        intensity=IntensityConfig(**intensity_args),
        geometric=GeometricConfig(**geometric_args),
    )

    data_cfg_args["skeletons"] = (
        skeletons_list
        if skeletons_list is not None and len(skeletons_list) > 0
        else None
    )

    return DataConfig(**data_cfg_args)
