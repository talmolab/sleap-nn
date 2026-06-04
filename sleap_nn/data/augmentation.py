"""This module implements data pipeline blocks for augmentation operations.

Uses Skia (skia-python) for ~1.5x faster augmentation compared to Kornia.
"""

from typing import Optional, Sequence, Tuple
import torch

from sleap_nn.data.skia_augmentation import (
    apply_intensity_augmentation_skia,
    apply_geometric_augmentation_skia,
    apply_flip_augmentation_skia,
)


def apply_flip_augmentation(
    image: torch.Tensor,
    instances: torch.Tensor,
    symmetric_inds: Optional[Sequence[Tuple[int, int]]] = None,
    flip_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mirror an image and keypoints left/right, swapping symmetric pairs.

    When an image is mirrored left/right, left/right symmetric body parts physically
    exchange sides, so their node slots must be swapped to keep semantic labels
    correct. Ported from SLEAP v1.4's ``RandomFlipper`` (horizontal flip).

    Args:
        image: Input image. Shape: (n_samples, C, H, W)
        instances: Input keypoints. (n_samples, n_instances, n_nodes, 2) or (n_samples, n_nodes, 2)
        symmetric_inds: Iterable of ``(i, j)`` node-index pairs to swap after mirroring.
            ``None``/empty means no swap (correct only for truly symmetric labeling,
            e.g. centroids).
        flip_p: Probability of applying the flip. ``0`` disables (no-op).

    Returns:
        Returns tuple: (image, instances) with the flip applied (or unchanged inputs
        when not applied). NaN keypoints are preserved.
    """
    return apply_flip_augmentation_skia(
        image=image,
        instances=instances,
        symmetric_inds=symmetric_inds,
        flip_p=flip_p,
    )


def apply_intensity_augmentation(
    image: torch.Tensor,
    instances: torch.Tensor,
    uniform_noise_min: Optional[float] = 0.0,
    uniform_noise_max: Optional[float] = 0.04,
    uniform_noise_p: float = 0.0,
    gaussian_noise_mean: Optional[float] = 0.0,
    gaussian_noise_std: Optional[float] = 0.02,
    gaussian_noise_p: float = 0.0,
    contrast_min: Optional[float] = 0.9,
    contrast_max: Optional[float] = 1.1,
    contrast_p: float = 0.0,
    brightness_min: Optional[float] = 0.9,
    brightness_max: Optional[float] = 1.1,
    brightness_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply intensity augmentation on image and instances.

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
    return apply_intensity_augmentation_skia(
        image=image,
        instances=instances,
        uniform_noise_min=uniform_noise_min,
        uniform_noise_max=uniform_noise_max,
        uniform_noise_p=uniform_noise_p,
        gaussian_noise_mean=gaussian_noise_mean,
        gaussian_noise_std=gaussian_noise_std,
        gaussian_noise_p=gaussian_noise_p,
        contrast_min=contrast_min,
        contrast_max=contrast_max,
        contrast_p=contrast_p,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        brightness_p=brightness_p,
    )


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
    flip_p: float = 0.0,
    symmetric_inds: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply geometric augmentation on image and instances.

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
        flip_p: Probability of mirroring the sample left/right (with symmetric-node
            swap). Default: 0.0 (disabled).
        symmetric_inds: Node-index pairs to swap after mirroring. Passed separately
            from the config scalars because it is runtime skeleton data. None/empty
            means no swap. Default: None.

    Returns:
        Returns tuple: (image, instances) with augmentation applied.
    """
    return apply_geometric_augmentation_skia(
        image=image,
        instances=instances,
        rotation_min=rotation_min,
        rotation_max=rotation_max,
        rotation_p=rotation_p,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_p=scale_p,
        translate_width=translate_width,
        translate_height=translate_height,
        translate_p=translate_p,
        affine_p=affine_p,
        erase_scale_min=erase_scale_min,
        erase_scale_max=erase_scale_max,
        erase_ratio_min=erase_ratio_min,
        erase_ratio_max=erase_ratio_max,
        erase_p=erase_p,
        mixup_lambda_min=mixup_lambda_min,
        mixup_lambda_max=mixup_lambda_max,
        mixup_p=mixup_p,
        flip_p=flip_p,
        symmetric_inds=symmetric_inds,
    )
