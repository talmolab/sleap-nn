"""This module provides a generalized implementation of UNet.

See the `UNet` class docstring for more information.
"""

import math
from typing import List, Optional, Text, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.architectures.encoder_decoder import Decoder, Encoder


class UNet(nn.Module):
    """U-Net architecture for pose estimation.

    This class defines the U-Net architecture for pose estimation, combining an
    encoder and a decoder. The encoder extracts features from the input, while the
    decoder generates confidence maps based on the features.

    Args:
        in_channels: Number of input channels. Default is 1.
        kernel_size: Size of the convolutional kernels. Default is 3.
        filters: Number of filters for the initial block. Default is 32.
        filters_rate: Factor to adjust the number of filters per block. Default is 1.5.
        stem_blocks: Number of initial stem blocks. Default is 0.
        down_blocks: Number of downsampling blocks. Default is 4.
        up_blocks: Number of upsampling blocks in the decoder. Default is 3.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        middle_block: Whether to include a middle block in the encoder. Default is True.
        block_contraction: Whether to contract the channels in the decoder blocks. Default is False.

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 3,
        filters: int = 32,
        filters_rate: int = 1.5,
        stem_blocks: int = 0,
        down_blocks: int = 4,
        up_blocks: int = 3,
        convs_per_block: int = 2,
        middle_block: bool = True,
        block_contraction: bool = False,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.enc = Encoder(
            in_channels=in_channels,
            filters=filters,
            down_blocks=down_blocks,
            filters_rate=filters_rate,
            stem_blocks=stem_blocks,
            convs_per_block=convs_per_block,
            kernel_size=kernel_size,
            middle_block=middle_block,
            block_contraction=block_contraction,
        )

        current_stride = int(
            np.prod(
                [
                    block.pooling_stride
                    for block in self.enc.encoder_stack
                    if hasattr(block, "pool") and block.pool
                ]
                + [1]
            )
        )

        x_in_shape = int(filters * (filters_rate ** (down_blocks + stem_blocks)))

        self.dec = Decoder(
            x_in_shape=x_in_shape,
            current_stride=current_stride,
            filters=filters,
            up_blocks=up_blocks,
            down_blocks=down_blocks,
            filters_rate=filters_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net architecture.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the U-Net operations.
        """
        x, features = self.enc(x)
        x = self.dec(x, features)
        return x
