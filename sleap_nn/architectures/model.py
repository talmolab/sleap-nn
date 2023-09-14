"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `nn.Module` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""
from typing import List

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

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
    backbones = {"unet": UNet}

    if backbone not in backbones:
        raise KeyError(
            f"Unsupported backbone: {backbone}. Supported backbones are: {', '.join(backbones.keys())}"
        )

    backbone = backbones[backbone](**backbone_config)

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
        head_config: An `DictConfig` configuration dictionary for the model head.
    """

    def __init__(
        self, backbone_config: DictConfig, head_configs: List[DictConfig]
    ) -> None:
        """Initialize the backbone and head based on the backbone_config."""
        super().__init__()
        self.model_config = backbone_config
        self.head_configs = head_configs

        self.backbone = get_backbone(
            backbone_config.backbone_type, backbone_config.backbone_config
        )

        if backbone_config.backbone_type == "unet":
            in_channels = int(
                backbone_config.backbone_config.filters
                * (
                    backbone_config.backbone_config.filters_rate
                    ** (
                        backbone_config.backbone_config.down_blocks
                        - 1
                        - backbone_config.backbone_config.up_blocks
                        + 1
                    )
                )
            )

        self.heads = nn.ModuleList()
        for head_config in head_configs:
            head = get_head(head_config.head_type, head_config.head_config).make_head(
                x_in=in_channels
            )
            self.heads.append(head)

    @classmethod
    def from_config(
        cls, backbone_config: DictConfig, head_configs: List[DictConfig]
    ) -> "Model":
        """Create the model from a config dictionary."""
        return cls(backbone_config=backbone_config, head_configs=head_configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.backbone(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs
