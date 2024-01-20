"""This module implements data pipeline blocks for augmentation operations."""
from typing import Any, Dict, Optional, Tuple, Union, Iterator

import kornia as K
import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.container import AugmentationSequential
from kornia.augmentation.utils.param_validation import _range_bound
from kornia.core import Tensor
from torch.utils.data.datapipes.datapipe import IterDataPipe


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
        scale: A scaling factor as a scalar float specifying the amount of scaling. A
            random factor between `(1 - scale, 1 + scale)` will be sampled and applied
            to both images and keypoints. If `None`, no scaling augmentation will be
            applied.
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        affine_p: Probability of applying random affine transformations.
        uniform_noise: tuple of uniform noise `(min_noise, max_noise)`.
            Must satisfy 0. <= min_noise <= max_noise <= 1.
        uniform_noise_p: Probability of applying random uniform noise.
        gaussian_noise_mean: The mean of the gaussian distribution.
        gaussian_noise_std: The standard deviation of the gaussian distribution.
        gaussian_noise_p: Probability of applying random gaussian noise.
        contrast: The contrast factor to apply. Default: `(1.0, 1.0)`.
        contrast_p: Probability of applying random contrast.
        brightness: The brightness factor to apply Default: `(1.0, 1.0)`.
        brightness_p: Probability of applying random brightness.
        erase_scale: Range of proportion of erased area against input image. Default: `(0.0001, 0.01)`.
        erase_ratio: Range of aspect ratio of erased area. Default: `(1, 1)`.
        erase_p: Probability of applying random erase.
        mixup_lambda: min-max value of mixup strength. Default is 0-1. Default: `None`.
        mixup_p: Probability of applying random mixup v2.
        random_crop_hw: Desired output size (out_h, out_w) of the crop. Must be Tuple[int, int],
            then out_h = size[0], out_w = size[1].
        random_crop_p: Probability of applying random crop.
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
        scale: Optional[float] = 0.05,
        translate: Optional[Tuple[float, float]] = (0.02, 0.02),
        affine_p: float = 0.0,
        uniform_noise: Optional[Tuple[float, float]] = (0.0, 0.04),
        uniform_noise_p: float = 0.0,
        gaussian_noise_mean: Optional[float] = 0.02,
        gaussian_noise_std: Optional[float] = 0.004,
        gaussian_noise_p: float = 0.0,
        contrast: Optional[Tuple[float, float]] = (0.5, 2.0),
        contrast_p: float = 0.0,
        brightness: Optional[float] = 0.0,
        brightness_p: float = 0.0,
        erase_scale: Optional[Tuple[float, float]] = (0.0001, 0.01),
        erase_ratio: Optional[Tuple[float, float]] = (1, 1),
        erase_p: float = 0.0,
        mixup_lambda: Union[Optional[float], Tuple[float, float], None] = None,
        mixup_p: float = 0.0,
        random_crop_hw: Tuple[int, int] = (0, 0),
        random_crop_p: float = 0.0,
        image_key: str = "image",
        instance_key: str = "instances",
    ) -> None:
        """Initialize the block and the augmentation pipeline."""
        self.source_dp = source_dp
        self.rotation = rotation
        self.scale = (1 - scale, 1 + scale)
        self.translate = translate
        self.affine_p = affine_p
        self.uniform_noise = uniform_noise
        self.uniform_noise_p = uniform_noise_p
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_noise_p = gaussian_noise_p
        self.contrast = contrast
        self.contrast_p = contrast_p
        self.brightness = brightness
        self.brightness_p = brightness_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase_p = erase_p
        self.mixup_lambda = mixup_lambda
        self.mixup_p = mixup_p
        self.random_crop_hw = random_crop_hw
        self.random_crop_p = random_crop_p
        self.image_key = image_key
        self.instance_key = instance_key

        aug_stack = []
        if self.affine_p > 0:
            aug_stack.append(
                K.augmentation.RandomAffine(
                    degrees=self.rotation,
                    translate=self.translate,
                    scale=self.scale,
                    p=self.affine_p,
                    keepdim=True,
                    same_on_batch=True,
                )
            )
        if self.uniform_noise_p > 0:
            aug_stack.append(
                RandomUniformNoise(
                    noise=self.uniform_noise,
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
                    contrast=self.contrast,
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
                    scale=self.erase_scale,
                    ratio=self.erase_ratio,
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
        if self.random_crop_p > 0:
            if self.random_crop_hw[0] > 0 and self.random_crop_hw[1] > 0:
                aug_stack.append(
                    K.augmentation.RandomCrop(
                        size=self.random_crop_hw,
                        pad_if_needed=True,
                        p=self.random_crop_p,
                        keepdim=True,
                        same_on_batch=True,
                    )
                )
            else:
                raise ValueError(
                    f"random_crop_hw height and width must be greater than 0."
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
            # Before (self.input_key="image"): (B=1, C, H, W), (B=1, num_instances, num_nodes, 2)
            # or
            # Before (self.input_key="instance"): (B=1, C, crop_H, crop_W), (B=1, num_nodes, 2)
            image, instances = ex[self.image_key], ex[self.instance_key].reshape(
                inst_shape[0], -1, 2
            )  # (B=1, C, H, W), (B=1, num_instances * num_nodes, 2) OR (B=1, num_nodes, 2)

            aug_image, aug_instances = self.augmenter(image, instances)
            ex.update(
                {
                    self.image_key: aug_image,
                    self.instance_key: aug_instances.reshape(*inst_shape),
                }
            )
            # After (self.input_key="image"): (B=1, C, H, W), (B=1, num_instances, num_nodes, 2)
            # or
            # After (self.input_key="instance"): (B=1, C, crop_H, crop_W), (B=1, num_nodes, 2)
            yield ex
