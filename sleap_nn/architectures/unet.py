"""This module provides a generalized implementation of UNet.

See the `UNet` class docstring for more information.
"""

from typing import Tuple, List

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from sleap_nn.architectures.encoder_decoder import Decoder, Encoder, StemBlock


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

        # Create stem block if stem_blocks > 0
        if self.stem_blocks > 0:
            self.stem = StemBlock(
                in_channels=in_channels,
                filters=filters,
                stem_blocks=stem_blocks,
                filters_rate=filters_rate,
                convs_per_block=convs_per_block,
                kernel_size=stem_kernel_size,
                prefix="stem",
            )
        else:
            self.stem = None

        # Initialize lists to store multiple encoders and decoders
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(self.stacks):
            # Create encoder for this stack
            in_channels = (
                int(self.filters * (self.filters_rate ** (self.stem_blocks)))
                if self.stem_blocks > 0
                else in_channels
            )
            encoder = Encoder(
                in_channels=in_channels,
                filters=filters,
                down_blocks=down_blocks,
                filters_rate=filters_rate,
                convs_per_block=convs_per_block,
                kernel_size=kernel_size,
                stem_blocks=stem_blocks,
                prefix=f"stack{i}_enc",
            )

            # Create middle block separately (not part of encoder stack)
            self.middle_blocks = nn.ModuleList()
            # Get the last block filters from encoder
            last_block_filters = int(
                filters * (filters_rate ** (down_blocks + stem_blocks - 1))
            )
            enc_num = len(encoder.encoder_stack)
            if self.middle_block:
                if convs_per_block > 1:
                    # Middle expansion block
                    from sleap_nn.architectures.encoder_decoder import SimpleConvBlock

                    middle_expand = SimpleConvBlock(
                        in_channels=last_block_filters,
                        pool=False,
                        pool_before_convs=False,
                        pooling_stride=2,
                        num_convs=convs_per_block - 1,
                        filters=int(
                            filters * (filters_rate ** (down_blocks + stem_blocks))
                        ),
                        kernel_size=kernel_size,
                        use_bias=True,
                        batch_norm=False,
                        activation="relu",
                        prefix=f"stack{i}_enc{enc_num}_middle_expand",
                    )
                    enc_num += 1
                    self.middle_blocks.append(middle_expand)

                # Middle contraction block
                if self.block_contraction:
                    # Contract the channels with an exponent lower than the last encoder block
                    block_filters = int(last_block_filters)
                else:
                    # Keep the block output filters the same
                    block_filters = int(
                        filters * (filters_rate ** (down_blocks + stem_blocks))
                    )

                middle_contract = SimpleConvBlock(
                    in_channels=int(
                        filters * (filters_rate ** (down_blocks + stem_blocks))
                    ),
                    pool=False,
                    pool_before_convs=False,
                    pooling_stride=2,
                    num_convs=1,
                    filters=block_filters,
                    kernel_size=kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                    prefix=f"stack{i}_enc{enc_num}_middle_contract",
                )
                enc_num += 1
                self.middle_blocks.append(middle_contract)

            self.encoders.append(encoder)

            # Calculate current stride for this encoder
            # Start with stem stride if stem blocks exist
            current_stride = 2**self.stem_blocks if self.stem_blocks > 0 else 1

            # Add encoder strides
            for block in encoder.encoder_stack:
                if hasattr(block, "pool") and block.pool:
                    current_stride *= block.pooling_stride

            current_stride *= (
                2  # for last pool layer MaxPool2dWithSamePadding in encoder
            )

            # Create decoder for this stack
            if self.block_contraction:
                # Contract the channels with an exponent lower than the last encoder block
                x_in_shape = int(
                    filters * (filters_rate ** (down_blocks + stem_blocks - 1))
                )
            else:
                # Keep the block output filters the same
                x_in_shape = int(
                    filters * (filters_rate ** (down_blocks + stem_blocks))
                )
            decoder = Decoder(
                x_in_shape=x_in_shape,
                current_stride=current_stride,
                filters=filters,
                up_blocks=up_blocks,
                down_blocks=down_blocks,
                filters_rate=filters_rate,
                stem_blocks=stem_blocks,
                output_stride=output_stride,
                kernel_size=kernel_size,
                block_contraction=self.block_contraction,
                up_interpolate=up_interpolate,
                prefix=f"stack{i}_dec",
            )
            self.decoders.append(decoder)

        if len(self.decoders) and len(self.decoders[-1].decoder_stack):
            self.final_dec_channels = (
                self.decoders[-1].decoder_stack[-1].refine_convs_filters
            )
        else:
            self.final_dec_channels = (
                last_block_filters if not self.middle_block else block_filters
            )

        self.decoder_stride_to_filters = self.decoders[-1].stride_to_filters

    @classmethod
    def from_config(cls, config: OmegaConf):
        """Create UNet from a config."""
        stem_blocks = 0
        if config.stem_stride is not None:
            stem_blocks = np.log2(config.stem_stride).astype(int)
        down_blocks = np.log2(config.max_stride).astype(int) - stem_blocks
        up_blocks = (
            np.log2(config.max_stride / config.output_stride).astype(int) + stem_blocks
        )
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
        return self.decoders[0].x_in_shape

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the U-Net architecture.

        Args:
            x: Input tensor.

        Returns:
            x: Output a tensor after applying the U-Net operations.
            current_strides: a list of the current strides from the decoder.
        """
        # Process through stem block if it exists
        stem_output = x
        if self.stem is not None:
            stem_output = self.stem(x)

        # Process through all stacks
        outputs = []
        output = stem_output
        for i in range(self.stacks):
            # Get encoder and decoder for this stack
            encoder = self.encoders[i]
            decoder = self.decoders[i]

            # Forward pass through encoder
            encoded, features = encoder(output)

            # Process through middle block if it exists
            middle_output = encoded
            if self.middle_block and hasattr(self, "middle_blocks"):
                for middle_block in self.middle_blocks:
                    middle_output = middle_block(middle_output)

            if self.stem_blocks > 0:
                features.append(stem_output)

            output = decoder(middle_output, features)
            output["middle_output"] = middle_output
            outputs.append(output)

        return outputs[-1]
