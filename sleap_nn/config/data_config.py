import attrs
from omegaconf import MISSING
from typing import Optional, Tuple, List, Dict


"""Serializable configuration classes for specifying all data configuration parameters.

These configuration classes are intended to specify all 
the parameters required to initialize the data config.
"""

@attrs.define
class DataConfig:
    """Data configuration.

    provider: (str) Provider class to read the input sleap files. Only "LabelsReader" supported for the training pipeline.
    train_labels_path: (str) Path to training data (.slp file)
    val_labels_path: (str) Path to validation data (.slp file)
    user_instances_only: (bool) True if only user labeled instances should be used for training. If False, both user labeled and predicted instances would be used. Default: True.
    chunk_size: (int) Size of each chunk (in MB). Default: "100". #TODO: change in inference ckpts
    preprocessing: Configuration options related to data preprocessing.
    use_augmentations_train: (bool) True if the data augmentation should be applied to the training data, else False.
    augmentation_config: Configurations related to augmentation
    instance_cropping: Configuration options related to instance cropping for centroid
        and topdown models.
    """

    provider: str="LabelsReader"
    train_labels_path: str=MISSING
    val_labels_path: str=MISSING
    user_instances_only: bool = True
    chunk_size: int = 100
    preprocessing: PreprocessingConfig = attrs.field(factory=PreprocessingConfig)
    use_augmentations_train: bool=False
    augmentation_config: Optional[AugmentationConfig] = None


@attrs.define
class PreprocessingConfig:
    """ Configuration of Preprocessing.

    Attributes:
        is_rgb: (bool) True if the image has 3 channels (RGB image). If input has only one channel when this is set to True, then the images from single-channel is replicated along the channel axis. If input has three channels and this is set to False, then we convert the image to grayscale (single-channel) image.
        max_height: (int) Maximum height the image should be padded to. If not provided, the original image size will be retained. Default: None.
        max_width: (int) Maximum width the image should be padded to. If not provided, the original image size will be retained. Default: None.
        scale: (float or List[float]) Factor to resize the image dimensions by, specified as either a float scalar or as a 2-tuple of [scale_x, scale_y]. If a scalar is provided, both dimensions are resized by the same factor.
        crop_hw: (Tuple[int]) Crop height and width of each instance (h, w) for centered-instance model. If None, this would be automatically computed based on the largest instance in the sio.Labels file.
        min_crop_size: (int) Minimum crop size to be used if crop_hw is None.
    """

    is_rgb: bool = True
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: Union[float, Tuple[float, float]] = 1.0
    crop_hw: Optional[Tuple[int, int]] = None
    min_crop_size: int = 32                  #to help app work incase of error

@attrs.define
class AugmentationConfig:
    """ Configuration of Augmentation

    Attributes:
        random crop: (Optional) (Dict[float]) {"random_crop_p": None, "crop_height": None. "crop_width": None}, where random_crop_p is the probability of applying random crop and crop_height and crop_width are the desired output size (out_h, out_w) of the crop.
        intensity: (Optional)
        geometric: (Optional)
    """

    random_crop: Optional[Dict[str, Optional[float]]] = None
    intensity: Optional[IntensityConfig] = attrs.field(default=None)
    geometric: Optional[GeometricConfig] = attrs.field(default=None)

@attrs.define
class IntensityConfig:
    """ Configuration of Intensity (Optional):

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

    uniform_noise_min: float = attrs.field(default=0.0, validator=attrs.validators.ge(0))
    uniform_noise_max: float = attrs.field(default=1.0, validator=attrs.validators.le(1))
    uniform_noise_p: float = 0.0
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 1.0
    gaussian_noise_p: float = 0.0
    contrast_min: float = attrs.field(default=0.5, validator=attrs.validators.ge(0))
    contrast_max: float = attrs.field(default=2.0, validator=attrs.validators.ge(0))
    contrast_p: float = 0.0
    brightness: Tuple[float, float] = (1.0, 1.0)
    brightness_p: float = 0.0

@attrs.define
class GeometricConfig:
    """
    Configuration of Geometric (Optional)

    Attributes:
        rotation: (float) Angles in degrees as a scalar float of the amount of rotation. A random angle in (-rotation, rotation) will be sampled and applied to both images and keypoints. Set to 0 to disable rotation augmentation.
        scale: (float) scaling factor interval. If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b. If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d Default: None.
        translate_width: (float) Maximum absolute fraction for horizontal translation. For example, if translate_width=a, then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a. Will not translate by default.
        translate_height: (float) Maximum absolute fraction for vertical translation. For example, if translate_height=a, then vertical shift is randomly sampled in the range -img_height * a < dy < img_height * a. Will not translate by default.
        affine_p: (float) Probability of applying random affine transformations. Default=0.0
        erase_scale_min: (float) Minimum value of range of proportion of erased area against input image. Default: 0.0001.
        erase_scale_max: (float) Maximum value of range of proportion of erased area against input image. Default: 0.01.
        erase_ration_min: (float) Minimum value of range of aspect ratio of erased area. Default: 1.
        erase_ratio_max: (float) Maximum value of range of aspect ratio of erased area. Default: 1.
        erase_p: (float) Probability of applying random erase. Default=0.0
        mixup_lambda: (float) min-max value of mixup strength. Default is 0-1. Default: None.
        mixup_p: (float) Probability of applying random mixup v2. Default=0.0
        input_key: (str) Can be image or instance. The input_key instance expects the KorniaAugmenter to follow the InstanceCropper else image otherwise for default.
    """

    rotation: float = 0.0
    scale: Optional[Tuple[float, float, float, float]] = None
    translate_width: float = 0.0
    translate_height: float = 0.0
    affine_p: float = 0.0
    erase_scale_min: float = 0.0001
    erase_scale_max: float = 0.01
    erase_ratio_min: float = 1.0
    erase_ratio_max: float = 1.0
    erase_p: float = 0.0
    mixup_lambda: Optional[float] = None
    mixup_p: float = 0.0
    input_key: str = "image"
