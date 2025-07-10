"""Generic encoder-decoder fully convolutional backbones.

This module contains building blocks for creating encoder-decoder architectures of
general form.

The encoder branch of the network forms the initial multi-scale feature extraction via
repeated blocks of convolutions and pooling steps.

The decoder branch is then responsible for upsampling the low resolution feature maps
to achieve the target output stride.

This pattern is generalizable and describes most fully convolutional architectures. For
example:
    - simple convolutions with pooling form the structure in `LEAP CNN
<https://www.nature.com/articles/s41592-018-0234-5>`_;
    - adding skip connections forms `U-Net <https://arxiv.org/pdf/1505.04597.pdf>`_;
    - using residual blocks with skip connections forms the base module in `stacked
    hourglass <https://arxiv.org/pdf/1603.06937.pdf>`_;
    - using dense blocks with skip connections forms `FC-DenseNet
<https://arxiv.org/pdf/1611.09326.pdf>`_.

This module implements blocks used in all of these variants on top of a generic base
classes.

See the `EncoderDecoder` base class for requirements for creating new architectures.
"""

from typing import List, Text, Tuple, Union

import torch
from torch import nn

from sleap_nn.architectures.common import MaxPool2dWithSamePadding
from sleap_nn.architectures.utils import get_act_fn


class SimpleConvBlock(nn.Module):
    """A simple convolutional block module.

    This class defines a convolutional block that consists of convolutional layers,
    optional pooling layers, batch normalization, and activation functions.

    The layers within the SimpleConvBlock are organized as follows:

    1. Optional max pooling (with same padding) layer (before convolutional layers).
    2. Convolutional layers with specified number of filters, kernel size, and activation.
    3. Optional batch normalization layer after each convolutional layer (if batch_norm is True).
    4. Activation function after each convolutional layer (ReLU, Sigmoid, Tanh, etc.).
    5. Optional max pooling (with same padding) layer (after convolutional layers).

    Args:
        in_channels: Number of input channels.
        pool: Whether to include pooling layers. Default is True.
        pooling_stride: Stride for pooling layers. Default is 2.
        pool_before_convs: Whether to apply pooling before convolutional layers. Default is False.
        num_convs: Number of convolutional layers. Default is 2.
        filters: Number of filters for convolutional layers. Default is 32.
        kernel_size: Size of the convolutional kernels. Default is 3.
        use_bias: Whether to use bias in convolutional layers. Default is True.
        batch_norm: Whether to apply batch normalization. Default is False.
        activation: Activation function name. Default is "relu".

    Attributes:
        Inherits all attributes from torch.nn.Module.

    Note:
        The 'same' padding is applied using custom MaxPool2dWithSamePadding layers.
    """

    def __init__(
        self,
        in_channels: int,
        pool: bool = True,
        pooling_stride: int = 2,
        pool_before_convs: bool = False,
        num_convs: int = 2,
        filters: int = 32,
        kernel_size: int = 3,
        use_bias: bool = True,
        batch_norm: bool = False,
        activation: Text = "relu",
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.pool = pool
        self.pooling_stride = pooling_stride
        self.pool_before_convs = pool_before_convs
        self.num_convs = num_convs
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation

        self.blocks = []
        if pool and pool_before_convs:
            self.blocks.append(
                MaxPool2dWithSamePadding(
                    kernel_size=2, stride=pooling_stride, padding="same"
                )
            )

        for i in range(num_convs):
            self.blocks.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    bias=use_bias,
                )
            )

            if batch_norm:
                self.blocks.append(nn.BatchNorm2d(filters))

            self.blocks.append(get_act_fn(activation))

        if pool and not pool_before_convs:
            self.blocks.append(
                MaxPool2dWithSamePadding(
                    kernel_size=2, stride=pooling_stride, padding="same"
                )
            )

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SimpleConvBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional block operations.
        """
        return self.blocks(x)


class Encoder(nn.Module):
    """Encoder module for a neural network architecture.

    This class defines the encoder part of a neural network architecture,
    which consists of a stack of convolutional blocks for feature extraction.

    The Encoder consists of a stack of SimpleConvBlocks designed for feature extraction.

    Args:
        in_channels: Number of input channels. Default is 3.
        filters: Number of filters for the initial block. Default is 64.
        down_blocks: Number of downsampling blocks. Default is 4.
        filters_rate: Factor to increase the number of filters per block. Default is 2.
        current_stride: Initial stride for pooling operations. Default is 2.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        kernel_size: Size of the convolutional kernels. Default is 3.

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        in_channels: int = 3,
        filters: int = 64,
        down_blocks: int = 4,
        filters_rate: Union[float, int] = 2,
        current_stride: int = 2,
        convs_per_block: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stem_blocks: int = 0,
        stem_kernel_size: int = 7,
        middle_block: bool = True,
        block_contraction: bool = False,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.down_blocks = down_blocks
        self.filters_rate = filters_rate
        self.current_stride = current_stride
        self.convs_per_block = convs_per_block
        self.kernel_size = kernel_size
        self.stem_blocks = stem_blocks

        self.encoder_stack = nn.ModuleList([])
        for block in range(stem_blocks):
            prev_block_filters = -1 if block == 0 else block_filters
            block_filters = int(self.filters * (self.filters_rate**block))
            self.encoder_stack.append(
                SimpleConvBlock(
                    in_channels=in_channels if block == 0 else prev_block_filters,
                    pool=(block > 0),
                    pool_before_convs=True,
                    pooling_stride=2,
                    num_convs=convs_per_block,
                    filters=block_filters,
                    kernel_size=stem_kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                )
            )

        for block in range(down_blocks):
            prev_block_filters = -1 if block + self.stem_blocks == 0 else block_filters
            block_filters = int(filters * (filters_rate ** (block + self.stem_blocks)))

            self.encoder_stack.append(
                SimpleConvBlock(
                    in_channels=(
                        in_channels
                        if block + self.stem_blocks == 0
                        else prev_block_filters
                    ),
                    pool=(block + self.stem_blocks > 0),
                    pool_before_convs=True,
                    pooling_stride=2,
                    num_convs=convs_per_block,
                    filters=block_filters,
                    kernel_size=kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                )
            )
        after_block_filters = block_filters

        self.encoder_stack.append(
            MaxPool2dWithSamePadding(kernel_size=2, stride=2, padding="same")
        )

        if middle_block:
            if convs_per_block > 1:
                # First convs are one exponent higher than the last encoder block.

                if block_contraction:
                    # Contract the channels with an exponent lower than the last encoder block.
                    block_filters = int(
                        self.filters
                        * (
                            self.filters_rate
                            ** (self.down_blocks + self.stem_blocks - 1)
                        )
                    )
                else:
                    # Keep the block output filters the same.
                    block_filters = int(
                        self.filters
                        * (self.filters_rate ** (self.down_blocks + self.stem_blocks))
                    )
                self.encoder_stack.append(
                    SimpleConvBlock(
                        in_channels=after_block_filters,
                        pool=False,
                        pool_before_convs=False,
                        pooling_stride=2,
                        num_convs=convs_per_block - 1,
                        filters=block_filters,
                        kernel_size=kernel_size,
                        use_bias=True,
                        batch_norm=False,
                        activation="relu",
                    )
                )

            if block_contraction:
                # Contract the channels with an exponent lower than the last encoder block.
                block_filters = int(
                    self.filters
                    * (self.filters_rate ** (self.down_blocks + self.stem_blocks - 1))
                )
            else:
                # Keep the block output filters the same.
                block_filters = int(
                    self.filters
                    * (self.filters_rate ** (self.down_blocks + self.stem_blocks))
                )

            self.encoder_stack.append(
                SimpleConvBlock(
                    in_channels=block_filters,
                    pool=False,
                    pool_before_convs=False,
                    pooling_stride=2,
                    num_convs=1,
                    filters=block_filters,
                    kernel_size=kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                )
            )

        self.intermediate_features = {}
        for i, block in enumerate(self.encoder_stack):
            if isinstance(block, SimpleConvBlock) and block.pool:
                current_stride *= block.pooling_stride

            if current_stride not in self.intermediate_features.values():
                self.intermediate_features[i] = current_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Encoder module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the encoder operations.
            list: List of intermediate feature tensors from different levels of the encoder.
        """
        features = []
        for i in range(len(self.encoder_stack)):
            x = self.encoder_stack[i](x)

            if i in self.intermediate_features.keys():
                features.append(x)

        return x, features[::-1]


class SimpleUpsamplingBlock(nn.Module):
    """A simple upsampling and refining block module.

    This class defines an upsampling and refining block that consists of upsampling layers,
    convolutional layers for refinement, batch normalization, and activation functions.

    The block includes:
    1. Upsampling layers with adjustable stride and interpolation method.
    2. Refinement convolutional layers with customizable parameters.
    3. BatchNormalization layers (if specified; can be before or after activation function).
    4. Activation functions (default is ReLU) applied before or after BatchNormalization.

    Args:
        x_in_shape: Number of input channels for the feature map.
        current_stride: Current stride value to adjust during upsampling.
        upsampling_stride: Stride for upsampling. Default is 2.
        interp_method: Interpolation method for upsampling. Default is "bilinear".
        refine_convs: Number of convolutional layers for refinement. Default is 2.
        refine_convs_filters: Number of filters for refinement convolutional layers. Default is 64.
        refine_convs_kernel_size: Size of the refinement convolutional kernels. Default is 3.
        refine_convs_use_bias: Whether to use bias in refinement convolutional layers. Default is True.
        refine_convs_batch_norm: Whether to apply batch normalization. Default is True.
        refine_convs_batch_norm_before_activation: Whether to apply batch normalization before activation.
        refine_convs_activation: Activation function name. Default is "relu".
        transpose_convs_filters: Number of filters for Transpose convolutional layers. Default is 64.
        transpose_convs_use_bias: Whether to use bias in Transpose convolutional layers. Default is True.
        transpose_convs_batch_norm: Whether to apply batch normalization for Transpose Conv layers. Default is True.
        transpose_convs_batch_norm_before_activation: Whether to apply batch normalization before activation.
        transpose_convs_activation: Activation function name for Transpose Conv layers. Default is "relu".

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        x_in_shape: int,
        current_stride: int,
        upsampling_stride: int = 2,
        up_interpolate: bool = True,
        interp_method: Text = "bilinear",
        refine_convs: int = 2,
        refine_convs_filters: int = 64,
        refine_convs_kernel_size: int = 3,
        refine_convs_use_bias: bool = True,
        refine_convs_batch_norm: bool = True,
        refine_convs_batch_norm_before_activation: bool = True,
        refine_convs_activation: Text = "relu",
        transpose_convs_filters: int = 64,
        transpose_convs_kernel_size: int = 2,
        transpose_convs_use_bias: bool = True,
        transpose_convs_batch_norm: bool = True,
        transpose_convs_batch_norm_before_activation: bool = True,
        transpose_convs_activation: Text = "relu",
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.x_in_shape = x_in_shape
        self.current_stride = current_stride
        self.upsampling_stride = upsampling_stride
        self.interp_method = interp_method
        self.refine_convs = refine_convs
        self.refine_convs_filters = refine_convs_filters
        self.refine_convs_kernel_size = refine_convs_kernel_size
        self.refine_convs_use_bias = refine_convs_use_bias
        self.refine_convs_batch_norm = refine_convs_batch_norm
        self.refine_convs_batch_norm_before_activation = (
            refine_convs_batch_norm_before_activation
        )
        self.refine_convs_activation = refine_convs_activation
        self.up_interpolate = up_interpolate

        self.blocks = nn.ModuleList([])
        if current_stride is not None:
            # Append the strides to the block prefix.
            new_stride = current_stride // upsampling_stride

        # Upsample via interpolation.
        if self.up_interpolate:
            self.blocks.append(
                nn.Upsample(
                    scale_factor=upsampling_stride,
                    mode=interp_method,
                )
            )
        else:
            # Upsample via strided transposed convolution.
            filters = transpose_convs_filters
            self.blocks.append(
                nn.ConvTranspose2d(
                    in_channels=transpose_convs_filters,
                    out_channels=transpose_convs_filters,
                    kernel_size=transpose_convs_kernel_size,
                    stride=upsampling_stride,
                    padding=0,
                    bias=transpose_convs_use_bias,
                )
            )
            self.norm_act_layers = 1
            if (
                transpose_convs_batch_norm
                and transpose_convs_batch_norm_before_activation
            ):
                self.blocks.append(nn.BatchNorm2d(num_features=filters))
                self.norm_act_layers += 1

            self.blocks.append(get_act_fn(transpose_convs_activation))
            self.norm_act_layers += 1

            if (
                transpose_convs_batch_norm
                and not transpose_convs_batch_norm_before_activation
            ):
                self.blocks.append(nn.BatchNorm2d(num_features=filters))
                self.norm_act_layers += 1

        # Add further convolutions to refine after upsampling and/or skip.
        for i in range(refine_convs):
            filters = refine_convs_filters
            self.blocks.append(
                nn.Conv2d(
                    in_channels=int(x_in_shape) if i == 0 else int(filters),
                    out_channels=int(filters),
                    kernel_size=refine_convs_kernel_size,
                    stride=1,
                    padding="same",
                    bias=refine_convs_use_bias,
                )
            )

            if refine_convs_batch_norm and refine_convs_batch_norm_before_activation:
                self.blocks.append(nn.BatchNorm2d(num_features=filters))

            self.blocks.append(get_act_fn(refine_convs_activation))

            if (
                refine_convs_batch_norm
                and not refine_convs_batch_norm_before_activation
            ):
                self.blocks.append(nn.BatchNorm2d(num_features=filters))

    def forward(self, x: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SimpleUpsamplingBlock module.

        Args:
            x: Input tensor.
            feature: Feature tensor to be concatenated with the upsampled tensor.

        Returns:
            torch.Tensor: Output tensor after applying the upsampling and refining operations.
        """
        for idx, b in enumerate(self.blocks):
            if (
                not self.up_interpolate
                and idx == self.norm_act_layers
                and feature is not None
            ):
                x = torch.concat((x, feature), dim=1)
            elif (
                self.up_interpolate and idx == 1 and feature is not None
            ):  # Right after upsampling or convtranspose2d.
                x = torch.concat((x, feature), dim=1)
            x = b(x)
        return x


class Decoder(nn.Module):
    """Decoder module for the UNet architecture.

    This class defines the decoder part of the UNet,
    which consists of a stack of upsampling and refining blocks for feature reconstruction.

    Args:
        x_in_shape: Number of input channels for the decoder's input.
        output_stride: Minimum of the strides of the output heads. The input confidence map
        tensor is expected to be at the same stride.
        current_stride: Current stride value to adjust during upsampling.
        filters: Number of filters for the initial block. Default is 64.
        up_blocks: Number of upsampling blocks. Default is 4.
        down_blocks: Number of downsampling blocks. Default is 3.
        stem_blocks: If >0, will create additional "down" blocks for initial
            downsampling. These will be configured identically to the down blocks below.
        filters_rate: Factor to adjust the number of filters per block. Default is 2.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        kernel_size: Size of the convolutional kernels. Default is 3.
        block_contraction: If True, reduces the number of filters at the end of middle
            and decoder blocks. This has the effect of introducing an additional
            bottleneck before each upsampling step. The original implementation does not
            do this, but the CARE implementation does.
        up_interpolate: If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales.

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        x_in_shape: int,
        output_stride: int,
        current_stride: int,
        filters: int = 64,
        up_blocks: int = 4,
        down_blocks: int = 3,
        stem_blocks: int = 0,
        filters_rate: int = 2,
        convs_per_block: int = 2,
        kernel_size: int = 3,
        block_contraction: bool = False,
        up_interpolate: bool = True,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.x_in_shape = x_in_shape
        self.current_stride = current_stride
        self.filters = filters
        self.up_blocks = up_blocks
        self.down_blocks = down_blocks
        self.stem_blocks = stem_blocks
        self.filters_rate = filters_rate
        self.convs_per_block = convs_per_block
        self.kernel_size = kernel_size
        self.block_contraction = block_contraction

        self.current_strides = []
        self.residuals = 0

        self.decoder_stack = nn.ModuleList([])
        for block in range(up_blocks):
            prev_block_filters_in = -1 if block == 0 else block_filters_in
            block_filters_in = int(
                filters * (filters_rate ** (down_blocks + self.stem_blocks - 1 - block))
            )
            if self.block_contraction:
                block_filters_out = int(
                    self.filters
                    * (
                        self.filters_rate
                        ** (self.down_blocks + self.stem_blocks - 2 - block)
                    )
                )
            else:
                block_filters_out = block_filters_in

            next_stride = current_stride // 2
            up_sample_in_filter = (
                (x_in_shape + block_filters_in)
                if block == 0
                else (prev_block_filters_in + block_filters_in)
            )

            self.decoder_stack.append(
                SimpleUpsamplingBlock(
                    x_in_shape=up_sample_in_filter,
                    current_stride=current_stride,
                    upsampling_stride=2,
                    interp_method="bilinear",
                    refine_convs=self.convs_per_block,
                    refine_convs_filters=block_filters_out,
                    refine_convs_kernel_size=self.kernel_size,
                    refine_convs_batch_norm=False,
                    up_interpolate=up_interpolate,
                    transpose_convs_filters=(
                        x_in_shape if block == 0 else prev_block_filters_in
                    ),
                    transpose_convs_batch_norm=False,
                )
            )

            self.current_strides.append(current_stride)
            current_stride = next_stride
            self.residuals += 1

        while current_stride >= output_stride:
            next_stride = current_stride // 2
            block_filters_in = int(
                filters * (filters_rate ** (down_blocks + self.stem_blocks - 1 - block))
            )

            block_filters_out = block_filters_in // filters_rate
            self.decoder_stack.append(
                SimpleUpsamplingBlock(
                    x_in_shape=block_filters_in,
                    current_stride=current_stride,
                    upsampling_stride=2,
                    interp_method="bilinear",
                    refine_convs=self.convs_per_block,
                    refine_convs_filters=block_filters_out,
                    refine_convs_kernel_size=self.kernel_size,
                    refine_convs_batch_norm=False,
                    up_interpolate=up_interpolate,
                    transpose_convs_filters=block_filters_in,
                    transpose_convs_batch_norm=False,
                )
            )
            self.current_strides.append(current_stride)
            current_stride = next_stride
            block += 1

    def forward(
        self, x: torch.Tensor, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the Decoder module.

        Args:
            x: Input tensor for the decoder.
            features: List of feature tensors from different encoder levels.

        Returns:
            outputs: List of output tensors after applying the decoder operations.
            current_strides: the current strides from the decoder blocks.
        """
        outputs = {
            "outputs": [],
        }
        outputs["intermediate_feat"] = x
        for i in range(len(self.decoder_stack)):
            if i < self.residuals:
                x = self.decoder_stack[i](x, features[i])
            else:
                x = self.decoder_stack[i](x, None)
            outputs["outputs"].append(x)
        outputs["strides"] = self.current_strides

        return outputs
