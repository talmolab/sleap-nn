"""Common utilities for architecture and model building."""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


class MaxPool2dWithSamePadding(nn.MaxPool2d):
    """A MaxPool2d module with support for same padding.

    This class extends the torch.nn.MaxPool2d module and adds the ability
    to perform 'same' padding, similar to 'same' padding in convolutional
    layers. When 'same' padding is specified, the input tensor is padded
    with zeros to ensure that the output spatial dimensions match the input
    spatial dimensions as closely as possible.

    Args:
        nn.MaxPool2d arguments: Arguments that are passed to the parent
            torch.nn.MaxPool2d class.

    Attributes:
        Inherits all attributes from torch.nn.MaxPool2d.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the MaxPool2dWithSamePadding module.

    Note:
        The 'same' padding is applied only when self.padding is set to "same".

    Example:
        # Create an instance of MaxPool2dWithSamePadding
        maxpool_layer = MaxPool2dWithSamePadding(kernel_size=3, stride=2, padding="same")

        # Perform a forward pass on an input tensor
        input_tensor = torch.rand(1, 3, 32, 32)  # Example input tensor
        output = maxpool_layer(input_tensor)  # Apply the MaxPool2d operation with same padding.
    """

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """Calculate the required padding to achieve 'same' padding.

        Args:
            i (int): Input dimension (height or width).
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.

        Returns:
            int: The calculated padding value.
        """
        return int(
            max(
                (torch.ceil(torch.tensor(i / s)).item() - 1) * s + (k - 1) * d + 1 - i,
                0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MaxPool2dWithSamePadding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the MaxPool2d operation.
        """
        if self.padding == "same":
            ih, iw = x.size()[-2:]

            pad_h = self._calc_same_pad(
                i=ih,
                k=(
                    self.kernel_size
                    if type(self.kernel_size) is int
                    else self.kernel_size[0]
                ),
                s=self.stride if type(self.stride) is int else self.stride[0],
                d=self.dilation if type(self.dilation) is int else self.dilation[0],
            )
            pad_w = self._calc_same_pad(
                i=iw,
                k=(
                    self.kernel_size
                    if type(self.kernel_size) is int
                    else self.kernel_size[1]
                ),
                s=self.stride if type(self.stride) is int else self.stride[1],
                d=self.dilation if type(self.dilation) is int else self.dilation[1],
            )

            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                )
            self.padding = 0

        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
