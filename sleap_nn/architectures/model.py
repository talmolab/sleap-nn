"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `nn.Module` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from sleap_nn.architectures.heads import SingleInstanceConfmapsHead
from sleap_nn.architectures.unet import UNet


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        model_config: An `DictConfig` configuration dictionary for the model backbone.
        head_config: An `DictConfig` configuration dictionary for the model head.
    """

    def __init__(self, model_config: DictConfig, head_config: DictConfig) -> None:
        """Initialize the backbone and head based on the model_config."""
        super().__init__()
        self.model_config = model_config
        self.head_config = head_config

        if model_config.backbone_type == "unet":
            self.backbone = UNet(**dict(model_config.backbone_config))

            in_channels = int(
                model_config.backbone_config.filters
                * (
                    model_config.backbone_config.filters_rate
                    ** (
                        model_config.backbone_config.down_blocks
                        - 1
                        - model_config.backbone_config.up_blocks
                        + 1
                    )
                )
            )
            self.head = SingleInstanceConfmapsHead(**dict(head_config)).make_head(
                x_in=in_channels
            )

    @classmethod
    def from_config(cls, model_config: DictConfig, head_config: DictConfig) -> "Model":
        """Create the model from a config dictionary."""
        return cls(model_config=model_config, head_config=head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.backbone(x)
        x = self.head(x)
        return x
