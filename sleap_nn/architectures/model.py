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


def get_backbone(
    backbone: str, backbone_config: DictConfig, output_stride: int
) -> nn.Module:
    """Get a backbone model `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given backbone name.

    Args:
        backbone (str): Name of the backbone. Supported values are 'unet'.
        backbone_config (DictConfig): A config for the backbone.
        output_stride (int): Output stride to compute the number of down blocks.

    Returns:
        nn.Module: An instance of the requested backbone model.

    Raises:
        KeyError: If the provided backbone name is not one of the supported values.
    """
    backbones = {"unet": UNet, "convnext": ConvNextWrapper, "swint": SwinTWrapper}

    if backbone not in backbones:
        raise KeyError(
            f"Unsupported backbone: {backbone}. Supported backbones are: {', '.join(backbones.keys())}"
        )

    backbone = backbones[backbone].from_config(
        backbone_config, output_stride=output_stride
    )

    return backbone


def get_head(head: str, head_config: DictConfig) -> Head:
    """Get a head `nn.Module` based on the provided name.

    This function returns an instance of a PyTorch `nn.Module`
    corresponding to the given head name.

    Args:
        head (str): Name of the head. Supported values are
            - 'SingleInstanceConfmapsHead'
            - 'CentroidConfmapsHead'
            - 'CenteredInstanceConfmapsHead'
            - 'MultiInstanceConfmapsHead'
            - 'PartAffinityFieldsHead'
            - 'ClassMapsHead'
            - 'ClassVectorsHead'
            - 'OffsetRefinementHead'
        head_config (DictConfig): A config for the head.

    Returns:
        nn.Module: An instance of the requested head.

    Raises:
        KeyError: If the provided head name is not one of the supported values.
    """
    heads = {
        "SingleInstanceConfmapsHead": SingleInstanceConfmapsHead,
        "CentroidConfmapsHead": CentroidConfmapsHead,
        "CenteredInstanceConfmapsHead": CenteredInstanceConfmapsHead,
        "MultiInstanceConfmapsHead": MultiInstanceConfmapsHead,
        "PartAffinityFieldsHead": PartAffinityFieldsHead,
        "ClassMapsHead": ClassMapsHead,
        "ClassVectorsHead": ClassVectorsHead,
        "OffsetRefinementHead": OffsetRefinementHead,
    }

    if head not in heads:
        raise KeyError(
            f"Unsupported head: {head}. Supported heads are: {', '.join(heads.keys())}"
        )

    head = heads[head](**head_config)

    return head


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        backbone_config: An `DictConfig` configuration dictionary for the model backbone.
        head_configs: An `DictConfig` configuration dictionary for the model heads.
        input_expand_channels: Integer representing the number of channels the image
                                should be expanded to.
    """

    def __init__(
        self,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        input_expand_channels: int,
    ) -> None:
        """Initialize the backbone and head based on the backbone_config."""
        super().__init__()
        self.backbone_config = backbone_config
        self.head_configs = head_configs
        self.input_expand_channels = input_expand_channels

        self.heads = []
        output_strides = []
        for head_type in head_configs:
            head_config = head_configs[head_type]
            head = get_head(head_config.head_type, head_config.head_config)
            self.heads.append(head)
            output_strides.append(head_config.head_config.output_stride)

        min_output_stride = min(output_strides)

        self.backbone = get_backbone(
            backbone_config.backbone_type,
            backbone_config.backbone_config,
            min_output_stride,
        )

        strides = self.backbone.dec.current_strides
        self.head_layers = nn.ModuleList([])
        for head in self.heads:
            in_channels = int(
                self.backbone.max_channels
                / (
                    self.backbone_config.backbone_config.filters_rate
                    ** len(self.backbone.dec.decoder_stack)
                )
            )
            if head.output_stride != min_output_stride:
                factor = strides.index(min_output_stride) - strides.index(
                    head.output_stride
                )
                in_channels = in_channels * (
                    self.backbone_config.backbone_config.filters_rate**factor
                )
            self.head_layers.append(head.make_head(x_in=int(in_channels)))

    @classmethod
    def from_config(
        cls,
        backbone_config: DictConfig,
        head_configs: DictConfig,
        input_expand_channels: int,
    ) -> "Model":
        """Create the model from a config dictionary."""
        return cls(
            backbone_config=backbone_config,
            head_configs=head_configs,
            input_expand_channels=input_expand_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.input_expand_channels != 1:
            input_list = []
            for _ in range(self.input_expand_channels):
                input_list.append(x)
            x = torch.concatenate(input_list, dim=-3)
        backbone_outputs = self.backbone(x)

        outputs = {}
        for head, head_layer in zip(self.heads, self.head_layers):
            idx = backbone_outputs["strides"].index(head.output_stride)
            outputs[head.name] = head_layer(backbone_outputs["outputs"][idx])

        return outputs
