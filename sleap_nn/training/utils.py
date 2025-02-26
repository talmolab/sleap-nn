"""Miscellaneous utility functions for training."""

import numpy as np
from torch import nn
from typing import Optional, Tuple

import sleap_io as sio
from sleap_nn.config.data_config import (
    AugmentationConfig,
    IntensityConfig,
    GeometricConfig,
)
from sleap_nn.config.model_config import (
    ConvNextConfig,
    SwinTConfig,
    BackboneConfig,
    HeadConfig,
    UNetConfig,
    UNetMediumRFConfig,
    UNetLargeRFConfig,
    ConvNextSmallConfig,
    ConvNextLargeConfig,
    ConvNextBaseConfig,
    SwinTBaseConfig,
    SwinTSmallConfig,
    SingleInstanceConfig,
    CentroidConfig,
    CenteredInstanceConfig,
    BottomUpConfig,
)
from sleap_nn.data.providers import get_max_instances, get_max_height_width


def get_aug_config(intensity_aug, geometric_aug):
    """Returns `AugmentationConfig` object based on the user-provided parameters."""
    aug_config = AugmentationConfig()
    if isinstance(intensity_aug, str):
        intensity_aug = [intensity_aug]

    for i in intensity_aug:
        if i == "uniform_noise":
            aug_config.intensity.uniform_noise_p = 1.0
        if i == "gaussian_noise":
            aug_config.intensity.gaussian_noise_p = 1.0
        if i == "contrast":
            aug_config.intensity.contrast_p = 1.0
        if i == "brightness":
            aug_config.intensity.brightness_p = 1.0

    if isinstance(intensity_aug, dict):
        aug_config.intensity = IntensityConfig(**intensity_aug)

    if isinstance(geometric_aug, str):
        geometric_aug = [geometric_aug]

    for g in geometric_aug:
        if g == "rotation":
            aug_config.geometric.affine_p = 1.0
            aug_config.geometric.scale = 1.0
            aug_config.geometric.translate_height = 0
            aug_config.geometric.translate_width = 0
        if g == "scale":
            aug_config.geometric.affine_p = 1.0
            aug_config.geometric.rotation = 0
            aug_config.geometric.translate_height = 0
            aug_config.geometric.translate_width = 0
        if g == "translate":
            aug_config.geometric.affine_p = 1.0
            aug_config.geometric.rotation = 0
            aug_config.geometric.scale = 1.0
        if g == "erase_scale":
            aug_config.geometric.erase_p = 1.0
        if g == "mixup":
            aug_config.geometric.mixup_p = 1.0

    if isinstance(geometric_aug, dict):
        aug_config.geometric = GeometricConfig(**geometric_aug)

    return aug_config


def get_backbone_config(backbone_cfg):
    """Returns `BackboneConfig` object based on the user-provided parameters."""
    backbone_config = BackboneConfig()
    unet_config_mapper = {
        "unet": UNetConfig(),
        "unet_medium_rf": UNetMediumRFConfig(),
        "unet_large_rf": UNetLargeRFConfig(),
    }
    convnext_config_mapper = {
        "convnext": ConvNextConfig(),
        "convnext_tiny": ConvNextConfig(),
        "convnext_small": ConvNextSmallConfig(),
        "convnext_base": ConvNextBaseConfig(),
        "convnext_large": ConvNextLargeConfig(),
    }
    swint_config_mapper = {
        "swint": SwinTConfig(),
        "swint_tiny": SwinTConfig(),
        "swint_small": SwinTSmallConfig(),
        "swint_base": SwinTBaseConfig(),
    }
    if isinstance(backbone_cfg, str):
        if backbone_cfg.startswith("unet"):
            backbone_config.unet = unet_config_mapper[backbone_cfg]
        elif backbone_cfg.startswith("convnext"):
            backbone_config.convnext = convnext_config_mapper[backbone_cfg]
        elif backbone_cfg.startswith("swint"):
            backbone_config.swint = swint_config_mapper[backbone_cfg]

    elif isinstance(backbone_cfg, dict):
        backbone_config = BackboneConfig(**backbone_cfg)
    return backbone_config


def get_head_configs(head_cfg):
    """Returns `HeadConfig` object based on the user-provided parameters."""
    head_configs = HeadConfig()
    if isinstance(head_cfg, str):
        if head_cfg == "centered_instance":
            head_configs.centered_instance = CenteredInstanceConfig()
        elif head_cfg == "single_instance":
            head_configs.single_instance = SingleInstanceConfig()
        elif head_cfg == "centroid":
            head_configs.centroid = CentroidConfig()
        elif head_cfg == "bottomup":
            head_configs.bottomup = BottomUpConfig()

    elif isinstance(head_cfg, dict):
        head_configs = HeadConfig(**head_cfg)

    return head_configs


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


def check_memory(
    labels: sio.Labels,
    max_hw: Tuple[int, int],
    model_type: str,
    input_scaling: float,
    crop_size: Optional[int],
):
    """Return memory required for caching the image samples."""
    if model_type == "centered_instance":
        num_samples = len(labels) * get_max_instances(labels)
        img = (labels[0].image / 255.0).astype(np.float32)
        img_mem = (crop_size**2) * img.shape[-1] * img.itemsize * num_samples

        return img_mem

    num_lfs = len(labels)
    img = (labels[0].image / 255.0).astype(np.float32)
    h, w = max_hw[0] * input_scaling, max_hw[1] * input_scaling
    img_mem = h * w * img.shape[-1] * img.itemsize * num_lfs

    return img_mem
