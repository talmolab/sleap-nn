"""This module implements data pipeline blocks for augmentation operations."""

from typing import Any, Dict, Optional, Tuple, Union
import kornia as K
import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.container import AugmentationSequential
from kornia.augmentation.utils.param_validation import _range_bound
from kornia.core import Tensor


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
    brightness_min: Optional[float] = 1.0,
    brightness_max: Optional[float] = 1.0,
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
        brightness_min: Minimum brightness factor to apply. Default: 1.0.
        brightness_max: Maximum brightness factor to apply. Default: 1.0.
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
                brightness=(brightness_min, brightness_max),
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
    rotation_min: Optional[float] = -15.0,
    rotation_max: Optional[float] = 15.0,
    rotation_p: Optional[float] = None,
    scale_min: Optional[float] = 0.9,
    scale_max: Optional[float] = 1.1,
    scale_p: Optional[float] = None,
    translate_width: Optional[float] = 0.02,
    translate_height: Optional[float] = 0.02,
    translate_p: Optional[float] = None,
    affine_p: float = 0.0,
    erase_scale_min: Optional[float] = 0.0001,
    erase_scale_max: Optional[float] = 0.01,
    erase_ratio_min: Optional[float] = 1,
    erase_ratio_max: Optional[float] = 1,
    erase_p: float = 0.0,
    mixup_lambda_min: Optional[float] = 0.01,
    mixup_lambda_max: Optional[float] = 0.05,
    mixup_p: float = 0.0,
) -> Tuple[torch.Tensor]:
    """Apply kornia geometric augmentation on image and instances.

    Args:
        image: Input image. Shape: (n_samples, C, H, W)
        instances: Input keypoints. (n_samples, n_instances, n_nodes, 2) or (n_samples, n_nodes, 2)
        rotation_min: Minimum rotation angle in degrees. Default: -15.0.
        rotation_max: Maximum rotation angle in degrees. Default: 15.0.
        rotation_p: Probability of applying random rotation independently. If None,
            falls back to affine_p for bundled behavior. Default: None.
        scale_min: Minimum scaling factor for isotropic scaling. Default: 0.9.
        scale_max: Maximum scaling factor for isotropic scaling. Default: 1.1.
        scale_p: Probability of applying random scaling independently. If None,
            falls back to affine_p for bundled behavior. Default: None.
        translate_width: Maximum absolute fraction for horizontal translation. Default: 0.02.
        translate_height: Maximum absolute fraction for vertical translation. Default: 0.02.
        translate_p: Probability of applying random translation independently. If None,
            falls back to affine_p for bundled behavior. Default: None.
        affine_p: Probability of applying random affine transformations (rotation, scale,
            translate bundled). Used when individual *_p params are None. Default: 0.0.
        erase_scale_min: Minimum value of range of proportion of erased area against input image. Default: 0.0001.
        erase_scale_max: Maximum value of range of proportion of erased area against input image. Default: 0.01.
        erase_ratio_min: Minimum value of range of aspect ratio of erased area. Default: 1.
        erase_ratio_max: Maximum value of range of aspect ratio of erased area. Default: 1.
        erase_p: Probability of applying random erase. Default: 0.0.
        mixup_lambda_min: Minimum mixup strength value. Default: 0.01.
        mixup_lambda_max: Maximum mixup strength value. Default: 0.05.
        mixup_p: Probability of applying random mixup v2. Default: 0.0.

    Returns:
        Returns tuple: (image, instances) with augmentation applied.
    """
    aug_stack = []

    # Check if any individual probability is set
    use_independent = (
        rotation_p is not None or scale_p is not None or translate_p is not None
    )

    if use_independent:
        # New behavior: Apply augmentations independently with separate probabilities
        if rotation_p is not None and rotation_p > 0:
            aug_stack.append(
                K.augmentation.RandomRotation(
                    degrees=(rotation_min, rotation_max),
                    p=rotation_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )

        if scale_p is not None and scale_p > 0:
            aug_stack.append(
                K.augmentation.RandomAffine(
                    degrees=0,  # No rotation
                    translate=None,  # No translation
                    scale=(scale_min, scale_max),
                    p=scale_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )

        if translate_p is not None and translate_p > 0:
            aug_stack.append(
                K.augmentation.RandomAffine(
                    degrees=0,  # No rotation
                    translate=(translate_width, translate_height),
                    scale=None,  # No scaling
                    p=translate_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
    elif affine_p > 0:
        # Legacy behavior: Bundled affine transformation
        aug_stack.append(
            K.augmentation.RandomAffine(
                degrees=(rotation_min, rotation_max),
                translate=(translate_width, translate_height),
                scale=(scale_min, scale_max),
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
                lambda_val=(mixup_lambda_min, mixup_lambda_max),
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
