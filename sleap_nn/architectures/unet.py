"""This module provides a generalized implementation of UNet.

See the `UNet` class docstring for more information.
"""

from typing import Tuple, List

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Decoder, Encoder


class UNet(nn.Module):
    """U-Net architecture for pose estimation.

    This class defines the U-Net architecture for pose estimation, combining an
    encoder and a decoder. The encoder extracts features from the input, while the
    decoder generates confidence maps based on the features.

    Args:
        in_channels: Number of input channels. Default is 1.
        output_stride: Minimum of the strides of the output heads. The input confidence map.
        kernel_size: Size of the convolutional kernels. Default is 3.
        stem_kernel_size: Kernle size for the stem blocks.
        filters: Number of filters for the initial block. Default is 32.
        filters_rate: Factor to adjust the number of filters per block. Default is 1.5.
        down_blocks: Number of downsampling blocks. Default is 4.
        up_blocks: Number of upsampling blocks in the decoder. Default is 3.
        stem_blocks: If >0, will create additional "down" blocks for initial
            downsampling. These will be configured identically to the down blocks below.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        middle_block: If True, add an additional block at the end of the encoder.
        up_interpolate: If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales.
        block_contraction: If True, reduces the number of filters at the end of middle
            and decoder blocks. This has the effect of introducing an additional
            bottleneck before each upsampling step. The original implementation does not
            do this, but the CARE implementation does.

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        output_stride: int = 2,
        in_channels: int = 1,
        kernel_size: int = 3,
        stem_kernel_size: int = 7,
        filters: int = 32,
        filters_rate: int = 1.5,
        down_blocks: int = 4,
        up_blocks: int = 3,
        stem_blocks: int = 0,
        convs_per_block: int = 2,
        middle_block: bool = True,
        up_interpolate: bool = True,
        block_contraction: bool = False,
        stacks: int = 1,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.filters_rate = filters_rate
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.stem_blocks = stem_blocks
        self.convs_per_block = convs_per_block
        self.stem_kernel_size = stem_kernel_size
        self.middle_block = middle_block
        self.up_interpolate = up_interpolate
        self.block_contraction = block_contraction
        self.stacks = stacks

        self.enc = Encoder(
            in_channels=in_channels,
            filters=filters,
            down_blocks=down_blocks,
            filters_rate=filters_rate,
            convs_per_block=convs_per_block,
            kernel_size=kernel_size,
            stem_blocks=stem_blocks,
            stem_kernel_size=stem_kernel_size,
            middle_block=self.middle_block,
            block_contraction=self.block_contraction,
        )

        self.current_stride = int(
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
            current_stride=self.current_stride,
            filters=filters,
            up_blocks=up_blocks,
            down_blocks=down_blocks,
            filters_rate=filters_rate,
            stem_blocks=stem_blocks,
            block_contraction=block_contraction,
            output_stride=output_stride,
            kernel_size=kernel_size,
            up_interpolate=up_interpolate,
        )

    @classmethod
    def from_config(cls, config: OmegaConf):
        """Create UNet from a config."""
        stem_blocks = 0
        if config.stem_stride is not None:
            stem_blocks = np.log2(config.stem_stride).astype(int)
        down_blocks = np.log2(config.max_stride).astype(int) - stem_blocks
        up_blocks = np.log2(config.max_stride / config.output_stride).astype(int)
        return cls(
            in_channels=config.in_channels,
            kernel_size=config.kernel_size,
            filters=config.filters,
            filters_rate=config.filters_rate,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            stem_blocks=stem_blocks,
            convs_per_block=config.convs_per_block,
            middle_block=config.middle_block,
            up_interpolate=config.up_interpolate,
            stacks=config.stacks,
            output_stride=config.output_stride,
        )

    @property
    def max_channels(self):
        """Returns the maximum channels of the UNet (last layer of the encoder)."""
        return self.dec.x_in_shape

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the U-Net architecture.

        Args:
            x: Input tensor.

        Returns:
            x: Output a tensor after applying the U-Net operations.
            current_strides: a list of the current strides from the decoder.
        """
        x, features = self.enc(x)
        x = self.dec(x, features)
        return x
