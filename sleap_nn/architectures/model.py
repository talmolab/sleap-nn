"""This module defines the main SLEAP model class for defining a trainable model.

This is a higher level wrapper around `tf.keras.Model` that holds all the configuration
parameters required to construct the actual model. This allows for easy querying of the
model configuration without actually instantiating the model itself.
"""
import torch
from omegaconf import OmegaConf
from torch import nn

from sleap_nn.architectures.heads import UNetOutputHead
from sleap_nn.architectures.unet import UNet


class Model(nn.Module):
    """Model creates a model consisting of a backbone and head.

    Attributes:
        model_config: An `OmegaConf` configuration dictionary.
    """

    def __init__(self, model_config: OmegaConf) -> None:
        """Initialize the backbone and head based on the model_config."""
        super().__init__()
        self.model_config = model_config

        if model_config.backbone_type == "unet":
            self.backbone = UNet(**dict(model_config.backbone_config))

            self.head = UNetOutputHead(
                in_channels=int(
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
                ),
                **dict(model_config.head_config),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.backbone(x)
        x = self.head(x)
        return x
