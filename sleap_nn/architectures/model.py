"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `nn.Module` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""

from typing import List

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
import math
from loguru import logger
from sleap_nn.architectures.heads import (
    Head,
    CentroidConfmapsHead,
    SingleInstanceConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    ClassVectorsHead,
    OffsetRefinementHead,
)
from sleap_nn.architectures.unet import UNet
from sleap_nn.architectures.convnext import ConvNextWrapper
from sleap_nn.architectures.swint import SwinTWrapper


def get_backbone(backbone: str, backbone_config: DictConfig) -> nn.Module:
    """Get a backbone model `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given backbone name.

    Args:
        backbone (str): Name of the backbone. Supported values are 'unet'.
        backbone_config (DictConfig): A config for the backbone.

    Returns:
        nn.Module: An instance of the requested backbone model.

    Raises:
        KeyError: If the provided backbone name is not one of the supported values.
    """
    backbones = {"unet": UNet, "convnext": ConvNextWrapper, "swint": SwinTWrapper}

    if backbone not in backbones:
        message = f"Unsupported backbone: {backbone}. Supported backbones are: {', '.join(backbones.keys())}"
        logger.error(message)
        raise KeyError(message)

    backbone = backbones[backbone].from_config(backbone_config)

    return backbone


def get_head(model_type: str, head_config: DictConfig) -> Head:
    """Get a head `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given head name.

    Args:
        model_type (str): Name of the head. Supported values are
            - 'single_instance'
            - 'centroid'
            - 'centered_instance'
            - 'bottomup'
        head_config (DictConfig): A config for the head.

    Returns:
        nn.Module: An instance of the requested head.
    """
    heads = []
    if model_type == "single_instance":
        heads.append(SingleInstanceConfmapsHead(**head_config.confmaps))

    elif model_type == "centered_instance":
        heads.append(CenteredInstanceConfmapsHead(**head_config.confmaps))

    elif model_type == "centroid":
        heads.append(CentroidConfmapsHead(**head_config.confmaps))

    elif model_type == "bottomup":
        heads.append(MultiInstanceConfmapsHead(**head_config.confmaps))
        heads.append(PartAffinityFieldsHead(**head_config.pafs))

    else:
        message = f"{model_type} is not a defined model type. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`."
        logger.error(message)
        raise Exception(message)

    return heads


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        backbone_type: Backbone type. One of `unet`, `convnext` and `swint`.
        backbone_config: An `DictConfig` configuration dictionary for the model backbone.
        head_configs: An `DictConfig` configuration dictionary for the model heads.
        input_expand_channels: Integer representing the number of channels the image
                                should be expanded to.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
    """

    def __init__(
        self,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        input_expand_channels: int,
        model_type: str,
    ) -> None:
        """Initialize the backbone and head based on the backbone_config."""
        super().__init__()
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config
        self.head_configs = head_configs
        self.input_expand_channels = input_expand_channels

        self.heads = get_head(model_type, self.head_configs)

        output_strides = []
        for head_type in head_configs:
            head_config = head_configs[head_type]
            output_strides.append(head_config.output_stride)

        min_output_stride = min(output_strides)

        self.backbone = get_backbone(
            self.backbone_type,
            backbone_config,
        )

        strides = self.backbone.dec.current_strides
        self.head_layers = nn.ModuleList([])
        for head in self.heads:
            in_channels = int(
                round(
                    self.backbone.max_channels
                    / (
                        self.backbone_config.filters_rate
                        ** len(self.backbone.dec.decoder_stack)
                    )
                )
            )
            if head.output_stride != min_output_stride:
                factor = strides.index(min_output_stride) - strides.index(
                    head.output_stride
                )
                in_channels = in_channels * (self.backbone_config.filters_rate**factor)
            self.head_layers.append(head.make_head(x_in=int(in_channels)))

    @classmethod
    def from_config(
        cls,
        backbone_type: str,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        input_expand_channels: int,
        model_type: str,
    ) -> "Model":
        """Create the model from a config dictionary."""
        return cls(
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            input_expand_channels=input_expand_channels,
            model_type=model_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        backbone_outputs = self.backbone(x)

        outputs = {}
        for head, head_layer in zip(self.heads, self.head_layers):
            idx = backbone_outputs["strides"].index(head.output_stride)
            outputs[head.name] = head_layer(backbone_outputs["outputs"][idx])

        return outputs
