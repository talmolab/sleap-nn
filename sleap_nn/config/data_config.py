"""Serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all
the parameters required to initialize the data config.
"""

from attrs import define, field, validators
from omegaconf import MISSING
from typing import Optional, Tuple, Any, Union, List


@define
class PreprocessingConfig:
    """Configuration of Preprocessing.

    Attributes:
        is_rgb: (bool) True if the image has 3 channels (RGB image). If input has only one channel when this is set to True, then the images from single-channel is replicated along the channel axis. If input has three channels and this is set to False, then we convert the image to grayscale (single-channel) image.
        max_height: (int) Maximum height the image should be padded to. If not provided, the original image size will be retained. Default: None.
        max_width: (int) Maximum width the image should be padded to. If not provided, the original image size will be retained. Default: None.
        scale: (float or List[float]) Factor to resize the image dimensions by, specified as either a float scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both dimensions are resized by the same factor.
        crop_hw: (Tuple[int]) Crop height and width of each instance (h, w) for centered-instance model. If None, this would be automatically computed based on the largest instance in the sio.Labels file.
        min_crop_size: (int) Minimum crop size to be used if crop_hw is None.
    """

    is_rgb: bool = False
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: float = field(
        default=1.0, validator=lambda instance, attr, value: instance.validate_scale()
    )
    crop_hw: Optional[Tuple[int, int]] = None
    min_crop_size: Optional[int] = 100  # to help app work incase of error

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
        raise ValueError(
            "PreprocessingConfig's scale must be a float or a list of floats."
        )


def validate_proportion(instance, attribute, value):
    """General Proportion Validation.

    Ensures all proportions are a 0<=float<=1.0
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{attribute.name} must be between 0.0 and 1.0, got {value}")


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
        mixup_lambda: (float) min-max value of mixup strength. Default is 0-1. Default: None.
        mixup_p: (float) Probability of applying random mixup v2. Default=0.0
    """

    rotation: float = 15.0
    scale: Optional[List[float]] = (0.9, 1.1)
    translate_width: float = 0.2
    translate_height: float = 0.2
    affine_p: float = field(default=0.0, validator=validate_proportion)
    erase_scale_min: float = 0.0001
    erase_scale_max: float = 0.01
    erase_ratio_min: float = 1.0
    erase_ratio_max: float = 1.0
    erase_p: float = field(default=0.0, validator=validate_proportion)
    mixup_lambda: float = 0.1
    mixup_p: float = field(default=0.0, validator=validate_proportion)


@define
class AugmentationConfig:
    """Configuration of Augmentation.

    Attributes:
        intensity: Configuration options for intensity-based augmentations like brightness, contrast, etc. If None, no intensity augmentations will be applied.
        geometric: Configuration options for geometric augmentations like rotation, scaling, translation etc. If None, no geometric augmentations will be applied.
    """

    intensity: Optional[IntensityConfig] = field(factory=IntensityConfig)
    geometric: Optional[GeometricConfig] = field(factory=GeometricConfig)


@define
class DataConfig:
    """Data configuration.

    train_labels_path: (str) Path to training data (.slp file)
    val_labels_path: (str) Path to validation data (.slp file)
    test_labels_path: (str) Path to test dataset (`.slp` file or `.mp4` file). *Note*: This is used only
        with CLI to get evaluation on test set after training is completed.
    provider: (str) Provider class to read the input sleap files. Only "LabelsReader"
        supported for the training pipeline.
    user_instances_only: (bool) True if only user labeled instances should be used for
        training. If False, both user labeled and predicted instances would be used. Default: True.
    data_pipeline_fw: (str) Framework to create the data loaders.
        One of [`litdata`, `torch_dataset`, `torch_dataset_np_chunks`].
    np_chunks_path: (str) Path to save `.npz` chunks created with `torch_dataset_np_chunks` data pipeline framework.
        If `None`, the path provided in `trainer_config.save_ckpt` is used (else working dir is used).
        The `train_chunks` and `val_chunks` dirs are created inside this path.
    litdata_chunks_path: (str) Path to save `.bin` files created with `litdata` data pipeline framework.
        If `None`, the path provided in `trainer_config.save_ckpt` is used (else working dir is used).
        The `train_chunks` and `val_chunks` dirs are created inside this path.
    use_existing_chunks: (bool) Use existing train and val chunks in the `np_chunks_path`
        or `chunks_path` for `torch_dataset_np_chunks` or `litdata` frameworks.
        If `True`, the `np_chunks_path` (or `chunks_path`) should have `train_chunks` and `val_chunks` dirs.
    chunk_size: (int) Size of each chunk (in MB). Default: 100.  # Your list shows "100" in quotes.
    delete_chunks_after_training: (bool) If `False`, the chunks (numpy or litdata chunks)
        are retained after training. Else, the chunks are deleted.
    preprocessing: Configuration options related to data preprocessing.
    use_augmentations_train: (bool) True if the data augmentation should be applied to the training data, else False.
    augmentation_config: Configurations related to augmentation.
        # Your list specifies "(only if use_augmentations_train is True)"
    skeletons: skeleton configuration for the `.slp` file. This will be pulled from the
        train dataset and saved to the `training_config.yaml`
    """

    train_labels_path: str = MISSING
    val_labels_path: str = MISSING
    test_file_path: Optional[str] = None
    provider: str = "LabelsReader"
    user_instances_only: bool = True
    data_pipeline_fw: str = "torch_dataset"
    np_chunks_path: Optional[str] = None
    litdata_chunks_path: Optional[str] = None
    use_existing_chunks: bool = False
    chunk_size: int = 100
    delete_chunks_after_training: bool = True
    preprocessing: PreprocessingConfig = field(factory=PreprocessingConfig)
    use_augmentations_train: bool = False
    augmentation_config: Optional[AugmentationConfig] = None
    skeletons: Optional[dict] = None
