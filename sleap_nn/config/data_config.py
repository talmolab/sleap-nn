import attr
from omegaconf import MISSING
from typing import Optional, Tuple, List, Dict

@attr.s(auto_attribs=True)
class DataConfig:
    """Data configuration.

    labels: Configuration options related to user labels for training or testing.
    preprocessing: Configuration options related to data preprocessing.
    instance_cropping: Configuration options related to instance cropping for centroid
        and topdown models.
    """

    provider: str="LabelsReader"
    train_labels_path: str=MISSING
    val_labels_path: str=MISSING
    preprocessing: PreprocessingConfig = attr.ib(factory=PreprocessingConfig)
    use_augmentations_train: bool=False
    augmentation_config: Optional[AugmentationConfig] = None


@attr.s(auto_attribs=True)
class PreprocessingConfig:
    is_rgb: bool = True
    max_height: Optional[int] = None
    max_width: Optional[int] = None
    scale: Union[float, Tuple[float, float]] = 1.0
    crop_hw: Optional[Tuple[int, int]] = None
    min_crop_size: int = 32                  #to help app work incase of error

@attr.s(auto_attribs=True)
class AugmentationConfig:
    random_crop: Optional[Dict[str, Optional[float]]] = None
    intensity: Optional[IntensityConfig] = attr.ib(default=None)
    geometric: Optional[GeometricConfig] = attr.ib(default=None)

@attr.s(auto_attribs=True)
class IntensityConfig:
    uniform_noise_min: float = 0.0
    uniform_noise_max: float = 1.0
    uniform_noise_p: float = 0.0
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 1.0
    gaussian_noise_p: float = 0.0
    contrast_min: float = 0.5
    contrast_max: float = 2.0
    contrast_p: float = 0.0
    brightness: Tuple[float, float] = (1.0, 1.0)
    brightness_p: float = 0.0

    # validate parameters
    @uniform_noise_min.validator
    def check_uniform_noise_min(self, attribute, value):
        if value < 0:
            raise ValueError(f"{attribute.name} must be >= 0.")

    @uniform_noise_max.validator
    def check_uniform_noise_max(self, attribute, value):
        if value <= 0:
            raise ValueError(f"{attribute.name} must be <= 1.")

@attr.s(auto_attribs=True)
class GeometricConfig:
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
