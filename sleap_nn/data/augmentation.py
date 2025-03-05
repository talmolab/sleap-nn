"""This module implements data pipeline blocks for augmentation operations."""

from typing import Any, Dict, Optional, Tuple, Union, Iterator
import kornia as K
import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.container import AugmentationSequential
from kornia.augmentation.utils.param_validation import _range_bound
from kornia.core import Tensor
from torch.utils.data.datapipes.datapipe import IterDataPipe


def apply_intensity_augmentation(
    image: torch.Tensor,
    instances: torch.Tensor,
    uniform_noise_min: Optional[float] = 0.0,
    uniform_noise_max: Optional[float] = 0.04,
    uniform_noise_p: float = 0.0,
    gaussian_noise_mean: Optional[float] = 0.02,
    gaussian_noise_std: Optional[float] = 0.004,
    gaussian_noise_p: float = 0.0,
    contrast_min: Optional[float] = 0.5,
    contrast_max: Optional[float] = 2.0,
    contrast_p: float = 0.0,
    brightness: Optional[float] = 0.0,
    brightness_p: float = 0.0,
) -> Tuple[torch.Tensor]:
    """Apply kornia intensity augmentation on image and instances.

    Args:
        image: Input image. Shape: (n_samples, C, H, W)
        instances: Input keypoints. (n_samples, n_instances, n_nodes, 2) or (n_samples, n_nodes, 2)
        uniform_noise_min: Minimum value for uniform noise (uniform_noise_min >=0).
        uniform_noise_max: Maximum value for uniform noise (uniform_noise_max <=1).
        uniform_noise_p: Probability of applying random uniform noise.
        gaussian_noise_mean: The mean of the gaussian distribution.
        gaussian_noise_std: The standard deviation of the gaussian distribution.
        gaussian_noise_p: Probability of applying random gaussian noise.
        contrast_min: Minimum contrast factor to apply. Default: 0.5.
        contrast_max: Maximum contrast factor to apply. Default: 2.0.
        contrast_p: Probability of applying random contrast.
        brightness: The brightness factor to apply Default: 0.0.
        brightness_p: Probability of applying random brightness.

    Returns:
        Returns tuple: (image, instances) with augmentation applied.
    """
    aug_stack = []
    if uniform_noise_p > 0:
        aug_stack.append(
            RandomUniformNoise(
                noise=(uniform_noise_min, uniform_noise_max),
                p=uniform_noise_p,
                keepdim=True,
                same_on_batch=True,
            )
        )
    if gaussian_noise_p > 0:
        aug_stack.append(
            K.augmentation.RandomGaussianNoise(
                mean=gaussian_noise_mean,
                std=gaussian_noise_std,
                p=gaussian_noise_p,
                keepdim=True,
                same_on_batch=True,
            )
        )
    if contrast_p > 0:
        aug_stack.append(
            K.augmentation.RandomContrast(
                contrast=(contrast_min, contrast_max),
                p=contrast_p,
                keepdim=True,
                same_on_batch=True,
            )
        )
    if brightness_p > 0:
        aug_stack.append(
            K.augmentation.RandomBrightness(
                brightness=brightness,
                p=brightness_p,
                keepdim=True,
                same_on_batch=True,
            )
        )

    augmenter = AugmentationSequential(
        *aug_stack,
        data_keys=["input", "keypoints"],
        keepdim=True,
        same_on_batch=True,
    )

    inst_shape = instances.shape
    # Before (full image): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
    # or
    # Before (cropped image): (B=1, C, crop_H, crop_W), (n_samples, n_nodes, 2)
    instances = instances.reshape(inst_shape[0], -1, 2)
    # (n_samples, C, H, W), (n_samples, n_instances * n_nodes, 2) OR (n_samples, n_nodes, 2)

    aug_image, aug_instances = augmenter(image, instances)

    # After (full image): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
    # or
    # After (cropped image): (n_samples, C, crop_H, crop_W), (n_samples, n_nodes, 2)
    return aug_image, aug_instances.reshape(*inst_shape)


def apply_geometric_augmentation(
    image: torch.Tensor,
    instances: torch.Tensor,
    rotation: Optional[float] = 15.0,
    scale: Union[Tuple[float, float], Tuple[float, float, float, float]] = None,
    translate_width: Optional[float] = 0.02,
    translate_height: Optional[float] = 0.02,
    affine_p: float = 0.0,
    erase_scale_min: Optional[float] = 0.0001,
    erase_scale_max: Optional[float] = 0.01,
    erase_ratio_min: Optional[float] = 1,
    erase_ratio_max: Optional[float] = 1,
    erase_p: float = 0.0,
    mixup_lambda: Union[Optional[float], Tuple[float, float], None] = None,
    mixup_p: float = 0.0,
) -> Tuple[torch.Tensor]:
    """Apply kornia geometric augmentation on image and instances.

    Args:
        image: Input image. Shape: (n_samples, C, H, W)
        instances: Input keypoints. (n_samples, n_instances, n_nodes, 2) or (n_samples, n_nodes, 2)
        rotation: Angles in degrees as a scalar float of the amount of rotation. A
            random angle in `(-rotation, rotation)` will be sampled and applied to both
            images and keypoints. Set to 0 to disable rotation augmentation.
        scale: scaling factor interval. If (a, b) represents isotropic scaling, the scale
            is randomly sampled from the range a <= scale <= b. If (a, b, c, d), the scale
            is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Default: None.
        translate_width: Maximum absolute fraction for horizontal translation. For example,
            if translate_width=a, then horizontal shift is randomly sampled in the range
            -img_width * a < dx < img_width * a. Will not translate by default.
        translate_height: Maximum absolute fraction for vertical translation. For example,
            if translate_height=a, then vertical shift is randomly sampled in the range
            -img_height * a < dy < img_height * a. Will not translate by default.
        affine_p: Probability of applying random affine transformations.
        erase_scale_min: Minimum value of range of proportion of erased area against input image. Default: 0.0001.
        erase_scale_max: Maximum value of range of proportion of erased area against input image. Default: 0.01.
        erase_ratio_min: Minimum value of range of aspect ratio of erased area. Default: 1.
        erase_ratio_max: Maximum value of range of aspect ratio of erased area. Default: 1.
        erase_p: Probability of applying random erase.
        mixup_lambda: min-max value of mixup strength. Default is 0-1. Default: `None`.
        mixup_p: Probability of applying random mixup v2.


    Returns:
        Returns tuple: (image, instances) with augmentation applied.
    """
    aug_stack = []
    if affine_p > 0:
        aug_stack.append(
            K.augmentation.RandomAffine(
                degrees=rotation,
                translate=(translate_width, translate_height),
                scale=scale,
                p=affine_p,
                keepdim=True,
                same_on_batch=True,
            )
        )

    if erase_p > 0:
        aug_stack.append(
            K.augmentation.RandomErasing(
                scale=(erase_scale_min, erase_scale_max),
                ratio=(erase_ratio_min, erase_ratio_max),
                p=erase_p,
                keepdim=True,
                same_on_batch=True,
            )
        )
    if mixup_p > 0:
        aug_stack.append(
            K.augmentation.RandomMixUpV2(
                lambda_val=mixup_lambda,
                p=mixup_p,
                keepdim=True,
                same_on_batch=True,
            )
        )

    augmenter = AugmentationSequential(
        *aug_stack,
        data_keys=["input", "keypoints"],
        keepdim=True,
        same_on_batch=True,
    )

    inst_shape = instances.shape
    # Before (full image): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
    # or
    # Before (cropped image): (B=1, C, crop_H, crop_W), (n_samples, n_nodes, 2)
    instances = instances.reshape(inst_shape[0], -1, 2)
    # (n_samples, C, H, W), (n_samples, n_instances * n_nodes, 2) OR (n_samples, n_nodes, 2)

    aug_image, aug_instances = augmenter(image, instances)

    # After (full image): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
    # or
    # After (cropped image): (n_samples, C, crop_H, crop_W), (n_samples, n_nodes, 2)
    return aug_image, aug_instances.reshape(*inst_shape)


class RandomUniformNoise(IntensityAugmentationBase2D):
    """Data transformer for applying random uniform noise to input images.

    This is a custom Kornia augmentation inheriting from `IntensityAugmentationBase2D`.
    Uniform noise within (min_val, max_val) is applied to the entire input image.

    Note: Inverse transform is not implemented and re-applying the same transformation
    in the example below does not work when included in an AugmentationSequential class.

    Args:
        noise: 2-tuple (min_val, max_val); 0.0 <= min_val <= max_val <= 1.0.
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input `True` or broadcast it
          to the batch form `False`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.rand(1, 1, 2, 2)
        >>> RandomUniformNoise(min_val=0., max_val=0.1, p=1.)(img)
        tensor([[[[0.9607, 0.5865],
                  [0.2705, 0.5920]]]])

    To apply the exact augmentation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomUniformNoise(min_val=0., max_val=0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    Ref: `kornia.augmentation._2d.intensity.gaussian_noise
    <https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/_2d/intensity/gaussian_noise.html#RandomGaussianNoise>`_.
    """

    def __init__(
        self,
        noise: Tuple[float, float],
        p: float = 0.5,
        p_batch: float = 1.0,
        clip_output: bool = True,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize the class."""
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self.flags = {
            "uniform_noise": _range_bound(noise, "uniform_noise", bounds=(0.0, 1.0))
        }
        self.clip_output = clip_output

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the uniform noise, add, and clamp output."""
        if "uniform_noise" in params:
            uniform_noise = params["uniform_noise"]
        else:
            uniform_noise = (
                torch.FloatTensor(input.shape)
                .uniform_(flags["uniform_noise"][0], flags["uniform_noise"][1])
                .to(input.device)
            )
            self._params["uniform_noise"] = uniform_noise
        if self.clip_output:
            return torch.clamp(
                input + uniform_noise, 0.0, 1.0
            )  # RandomGaussianNoise doesn't clamp.
        return input + uniform_noise


class KorniaAugmenter(IterDataPipe):
    """IterDataPipe for applying rotation and scaling augmentations using Kornia.

    This IterDataPipe will apply augmentations to images and instances in examples from the
    input pipeline.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"instances"` and
            `"image"` keys.
        rotation: Angles in degrees as a scalar float of the amount of rotation. A
            random angle in `(-rotation, rotation)` will be sampled and applied to both
            images and keypoints. Set to 0 to disable rotation augmentation.
        scale: scaling factor interval. If (a, b) represents isotropic scaling, the scale
            is randomly sampled from the range a <= scale <= b. If (a, b, c, d), the scale
            is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Default: None.
        translate_width: Maximum absolute fraction for horizontal translation. For example,
            if translate_width=a, then horizontal shift is randomly sampled in the range
            -img_width * a < dx < img_width * a. Will not translate by default.
        translate_height: Maximum absolute fraction for vertical translation. For example,
            if translate_height=a, then vertical shift is randomly sampled in the range
            -img_height * a < dy < img_height * a. Will not translate by default.
        affine_p: Probability of applying random affine transformations.
        uniform_noise_min: Minimum value for uniform noise (uniform_noise_min >=0).
        uniform_noise_max: Maximum value for uniform noise (uniform_noise_max <=1).
        uniform_noise_p: Probability of applying random uniform noise.
        gaussian_noise_mean: The mean of the gaussian distribution.
        gaussian_noise_std: The standard deviation of the gaussian distribution.
        gaussian_noise_p: Probability of applying random gaussian noise.
        contrast_min: Minimum contrast factor to apply. Default: 0.5.
        contrast_max: Maximum contrast factor to apply. Default: 2.0.
        contrast_p: Probability of applying random contrast.
        brightness: The brightness factor to apply Default: 0.0.
        brightness_p: Probability of applying random brightness.
        erase_scale_min: Minimum value of range of proportion of erased area against input image. Default: 0.0001.
        erase_scale_max: Maximum value of range of proportion of erased area against input image. Default: 0.01.
        erase_ratio_min: Minimum value of range of aspect ratio of erased area. Default: 1.
        erase_ratio_max: Maximum value of range of aspect ratio of erased area. Default: 1.
        erase_p: Probability of applying random erase.
        mixup_lambda: min-max value of mixup strength. Default is 0-1. Default: `None`.
        mixup_p: Probability of applying random mixup v2.
        input_key: Can be `image` or `instance`. The input_key `instance` expects the
            the KorniaAugmenter to follow the InstanceCropper else `image` otherwise
            for default.

    Notes:
        This block expects the "image" and "instances" keys to be present in the input
        examples.

        The `"image"` key should contain a torch.Tensor of dtype torch.float32
        and of shape `(..., C, H, W)`, i.e., rank >= 3.

        The `"instances"` key should contain a torch.Tensor of dtype torch.float32 and
        of shape `(..., n_instances, n_nodes, 2)`, i.e., rank >= 3.

        The augmented versions will be returned with the same keys and shapes.
    """

    def __init__(
        self,
        source_dp: IterDataPipe,
        rotation: Optional[float] = 15.0,
        scale: Union[
            Optional[float], Tuple[float, float], Tuple[float, float, float, float]
        ] = None,
        translate_width: Optional[float] = 0.02,
        translate_height: Optional[float] = 0.02,
        affine_p: float = 0.0,
        uniform_noise_min: Optional[float] = 0.0,
        uniform_noise_max: Optional[float] = 0.04,
        uniform_noise_p: float = 0.0,
        gaussian_noise_mean: Optional[float] = 0.02,
        gaussian_noise_std: Optional[float] = 0.004,
        gaussian_noise_p: float = 0.0,
        contrast_min: Optional[float] = 0.5,
        contrast_max: Optional[float] = 2.0,
        contrast_p: float = 0.0,
        brightness: Optional[float] = 0.0,
        brightness_p: float = 0.0,
        erase_scale_min: Optional[float] = 0.0001,
        erase_scale_max: Optional[float] = 0.01,
        erase_ratio_min: Optional[float] = 1,
        erase_ratio_max: Optional[float] = 1,
        erase_p: float = 0.0,
        mixup_lambda: Union[Optional[float], Tuple[float, float], None] = None,
        mixup_p: float = 0.0,
        image_key: str = "image",
        instance_key: str = "instances",
    ) -> None:
        """Initialize the block and the augmentation pipeline."""
        self.source_dp = source_dp
        self.rotation = rotation
        self.scale = scale
        if isinstance(self.scale, float):
            self.scale = (scale, scale)
        self.translate_width = translate_width
        self.translate_height = translate_height
        self.affine_p = affine_p
        self.uniform_noise_min = uniform_noise_min
        self.uniform_noise_max = uniform_noise_max
        self.uniform_noise_p = uniform_noise_p
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_noise_p = gaussian_noise_p
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self.contrast_p = contrast_p
        self.brightness = brightness
        self.brightness_p = brightness_p
        self.erase_scale_min = erase_scale_min
        self.erase_scale_max = erase_scale_max
        self.erase_ratio_min = erase_ratio_min
        self.erase_ratio_max = erase_ratio_max
        self.erase_p = erase_p
        self.mixup_lambda = mixup_lambda
        self.mixup_p = mixup_p
        self.image_key = image_key
        self.instance_key = instance_key

        aug_stack = []
        if self.affine_p > 0:
            aug_stack.append(
                K.augmentation.RandomAffine(
                    degrees=self.rotation,
                    translate=(self.translate_width, self.translate_height),
                    scale=self.scale,
                    p=self.affine_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.uniform_noise_p > 0:
            aug_stack.append(
                RandomUniformNoise(
                    noise=(self.uniform_noise_min, self.uniform_noise_max),
                    p=self.uniform_noise_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.gaussian_noise_p > 0:
            aug_stack.append(
                K.augmentation.RandomGaussianNoise(
                    mean=self.gaussian_noise_mean,
                    std=self.gaussian_noise_std,
                    p=self.gaussian_noise_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.contrast_p > 0:
            aug_stack.append(
                K.augmentation.RandomContrast(
                    contrast=(self.contrast_min, self.contrast_max),
                    p=self.contrast_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.brightness_p > 0:
            aug_stack.append(
                K.augmentation.RandomBrightness(
                    brightness=self.brightness,
                    p=self.brightness_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.erase_p > 0:
            aug_stack.append(
                K.augmentation.RandomErasing(
                    scale=(self.erase_scale_min, self.erase_scale_max),
                    ratio=(self.erase_ratio_min, self.erase_ratio_max),
                    p=self.erase_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.mixup_p > 0:
            aug_stack.append(
                K.augmentation.RandomMixUpV2(
                    lambda_val=self.mixup_lambda,
                    p=self.mixup_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )

        self.augmenter = AugmentationSequential(
            *aug_stack,
            data_keys=["input", "keypoints"],
            keepdim=True,
            same_on_batch=True,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an example dictionary with the augmented image and instances."""
        for ex in self.source_dp:
            inst_shape = ex[self.instance_key].shape
            # Before (self.input_key="image"): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
            # or
            # Before (self.input_key="instance"): (B=1, C, crop_H, crop_W), (n_samples, n_nodes, 2)
            image, instances = ex[self.image_key], ex[self.instance_key].reshape(
                inst_shape[0], -1, 2
            )  # (n_samples, C, H, W), (n_samples, n_instances * n_nodes, 2) OR (n_samples, n_nodes, 2)

            aug_image, aug_instances = self.augmenter(image, instances)
            ex.update(
                {
                    self.image_key: aug_image,
                    self.instance_key: aug_instances.reshape(*inst_shape),
                }
            )
            # After (self.input_key="image"): (n_samples, C, H, W), (n_samples, n_instances, n_nodes, 2)
            # or
            # After (self.input_key="instance"): (n_samples, C, crop_H, crop_W), (n_samples, n_nodes, 2)
            yield ex
