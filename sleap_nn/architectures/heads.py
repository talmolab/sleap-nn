"""Model head definitions for defining model output types."""

from typing import Tuple, Union

import torch
from torch import nn


class UNetOutputHead(nn.Module):
    """UNetOutputHead is a module for the confidence map output head in a U-Net architecture.

    Attributes:
        in_channels: Number of input channels; int.
        out_channels: Number of output channels; int.
        kernel_size: Size of the convolutional kernel; int or tuple.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: str,
    ) -> None:
        """Initialize the Conv2d confidence map output head."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conv2d."""
        return self.conv(x)
