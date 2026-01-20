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

from typing import List, Optional, Text, Tuple, Union
from collections import OrderedDict
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
        prefix: Text = "",
        name: Text = "",
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
        self.prefix = prefix
        self.name = name

        self.blocks = OrderedDict()
        if pool and pool_before_convs:
            self.blocks[f"{prefix}_pool"] = MaxPool2dWithSamePadding(
                kernel_size=2, stride=pooling_stride, padding="same"
            )

        for i in range(num_convs):
            self.blocks[f"{prefix}_conv{i}"] = nn.Conv2d(
                in_channels=in_channels if i == 0 else filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=use_bias,
            )

            if batch_norm:
                self.blocks[f"{prefix}_bn{i}"] = nn.BatchNorm2d(filters)

            self.blocks[f"{prefix}_act{i}_{activation}"] = get_act_fn(activation)

        if pool and not pool_before_convs:
            self.blocks[f"{prefix}_pool"] = MaxPool2dWithSamePadding(
                kernel_size=2, stride=pooling_stride, padding="same"
            )

        self.blocks = nn.Sequential(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SimpleConvBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional block operations.
        """
        for block in self.blocks:
            x = block(x)
        return x


class StemBlock(nn.Module):
    """Stem block module for initial feature extraction.

    This class defines a stem block that consists of a stack of convolutional blocks
    for initial feature extraction before the main encoder. The stem blocks are typically
    used for initial downsampling and feature extraction.

    Args:
        in_channels: Number of input channels. Default is 3.
        filters: Number of filters for the initial block. Default is 64.
        stem_blocks: Number of stem blocks. Default is 0.
        filters_rate: Factor to increase the number of filters per block. Default is 2.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        kernel_size: Size of the convolutional kernels. Default is 7.
        prefix: Prefix for layer naming. Default is "stem".

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        in_channels: int = 3,
        filters: int = 64,
        stem_blocks: int = 0,
        filters_rate: Union[float, int] = 2,
        convs_per_block: int = 2,
        kernel_size: int = 7,
        prefix: str = "stem",
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.stem_blocks = stem_blocks
        self.filters_rate = filters_rate
        self.convs_per_block = convs_per_block
        self.kernel_size = kernel_size
        self.prefix = prefix

        self.stem_stack = nn.ModuleList([])

        for block in range(self.stem_blocks):
            prev_block_filters = in_channels if block == 0 else block_filters
            block_filters = int(self.filters * (self.filters_rate**block))

            self.stem_stack.append(
                SimpleConvBlock(
                    in_channels=prev_block_filters,
                    pool=(block > 0),
                    pool_before_convs=True,
                    pooling_stride=2,
                    num_convs=convs_per_block,
                    filters=block_filters,
                    kernel_size=kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                    prefix=f"{prefix}{block}",
                )
            )

        # Always finish with a pooling block to account for pooling before convs.
        final_pool_dict = OrderedDict()
        final_pool_dict[f"{self.prefix}{block + 1}_last_pool"] = (
            MaxPool2dWithSamePadding(kernel_size=2, stride=2, padding="same")
        )
        self.stem_stack.append(nn.Sequential(final_pool_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the StemBlock module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the stem operations.
        """
        for block in self.stem_stack:
            x = block(x)
        return x


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
        prefix: str = "enc",
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
        self.prefix = prefix

        self.encoder_stack = nn.ModuleList([])
        block_filters = int(filters * (filters_rate ** (stem_blocks - 1)))
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
                    kernel_size=self.kernel_size,
                    use_bias=True,
                    batch_norm=False,
                    activation="relu",
                    prefix=f"{self.prefix}{block}",
                    name=f"{self.prefix}{block}",
                )
            )
        after_block_filters = block_filters

        # Add final pooling layer with proper naming
        block += 1
        final_pool_dict = OrderedDict()
        final_pool_dict[f"{self.prefix}{block}_last_pool"] = MaxPool2dWithSamePadding(
            kernel_size=2, stride=2, padding="same"
        )
        self.encoder_stack.append(nn.Sequential(final_pool_dict))

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
        refine_convs_batch_norm: bool = False,
        refine_convs_batch_norm_before_activation: bool = True,
        refine_convs_activation: Text = "relu",
        transpose_convs_filters: int = 64,
        transpose_convs_kernel_size: int = 3,
        transpose_convs_use_bias: bool = True,
        transpose_convs_batch_norm: bool = True,
        transpose_convs_batch_norm_before_activation: bool = True,
        transpose_convs_activation: Text = "relu",
        feat_concat: bool = True,
        prefix: Text = "",
        skip_channels: Optional[int] = None,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        # Determine skip connection channels
        # If skip_channels is provided, use it; otherwise fall back to refine_convs_filters
        # This allows ConvNext/SwinT to specify actual encoder channels
        self.skip_channels = (
            skip_channels if skip_channels is not None else refine_convs_filters
        )

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
        self.feat_concat = feat_concat
        self.prefix = prefix

        self.blocks = OrderedDict()
        if current_stride is not None:
            # Append the strides to the block prefix.
            new_stride = current_stride // upsampling_stride

        # Upsample via interpolation.
        if self.up_interpolate:
            self.blocks[f"{prefix}_interp_{interp_method}"] = nn.Upsample(
                scale_factor=upsampling_stride,
                mode=interp_method,
                align_corners=False,
            )
        else:
            # Upsample via strided transposed convolution.
            # The transpose conv should output the target number of filters
            self.blocks[f"{prefix}_trans_conv"] = nn.ConvTranspose2d(
                in_channels=x_in_shape,  # Input channels from the input tensor
                out_channels=transpose_convs_filters,  # Output channels for the upsampled tensor
                kernel_size=transpose_convs_kernel_size,
                stride=upsampling_stride,
                output_padding=1,
                padding=1,
                bias=transpose_convs_use_bias,
            )
            self.norm_act_layers = 1
            if (
                transpose_convs_batch_norm
                and transpose_convs_batch_norm_before_activation
            ):
                self.blocks[f"{prefix}_trans_conv_bn"] = nn.BatchNorm2d(
                    num_features=transpose_convs_filters
                )
                self.norm_act_layers += 1

            self.blocks[f"{prefix}_trans_conv_act_{transpose_convs_activation}"] = (
                get_act_fn(transpose_convs_activation)
            )
            self.norm_act_layers += 1

            if (
                transpose_convs_batch_norm
                and not transpose_convs_batch_norm_before_activation
            ):
                self.blocks[f"{prefix}_trans_conv_bn_after"] = nn.BatchNorm2d(
                    num_features=transpose_convs_filters
                )
                self.norm_act_layers += 1

        # Add further convolutions to refine after upsampling and/or skip.
        for i in range(refine_convs):
            filters = refine_convs_filters
            # For the first conv, calculate the actual input channels after concatenation
            if i == 0:
                if not self.feat_concat:
                    first_conv_in_channels = refine_convs_filters
                else:
                    if self.up_interpolate:
                        # With interpolation, input is x_in_shape + skip_channels
                        # skip_channels may differ from refine_convs_filters for ConvNext/SwinT
                        first_conv_in_channels = x_in_shape + self.skip_channels
                    else:
                        # With transpose conv, input is transpose_conv_output + skip_channels
                        first_conv_in_channels = (
                            self.skip_channels + transpose_convs_filters
                        )
            else:
                if not self.feat_concat:
                    first_conv_in_channels = refine_convs_filters
                first_conv_in_channels = filters

            self.blocks[f"{prefix}_refine_conv{i}"] = nn.Conv2d(
                in_channels=int(first_conv_in_channels),
                out_channels=int(filters),
                kernel_size=refine_convs_kernel_size,
                stride=1,
                padding="same",
                bias=refine_convs_use_bias,
            )

            if refine_convs_batch_norm and refine_convs_batch_norm_before_activation:
                self.blocks[f"{prefix}_refine_conv{i}_bn"] = nn.BatchNorm2d(
                    num_features=refine_convs_filters
                )

            self.blocks[f"{prefix}_refine_conv{i}_act_{refine_convs_activation}"] = (
                get_act_fn(refine_convs_activation)
            )

            if (
                refine_convs_batch_norm
                and not refine_convs_batch_norm_before_activation
            ):
                self.blocks[f"{prefix}_refine_conv_bn_after{i}"] = nn.BatchNorm2d(
                    num_features=refine_convs_filters
                )

        self.blocks = nn.Sequential(self.blocks)

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
                x = torch.concat((feature, x), dim=1)
            elif (
                self.up_interpolate and idx == 1 and feature is not None
            ):  # Right after upsampling or convtranspose2d.
                x = torch.concat((feature, x), dim=1)
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
        prefix: str = "dec",
        encoder_channels: Optional[List[int]] = None,
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
        self.prefix = prefix
        self.stride_to_filters = {}
        self.encoder_channels = encoder_channels

        self.current_strides = []
        self.residuals = 0

        self.decoder_stack = nn.ModuleList([])

        self.stride_to_filters[current_stride] = x_in_shape

        for block in range(up_blocks):
            prev_block_filters = -1 if block == 0 else block_filters_out
            block_filters_out = int(
                filters
                * (filters_rate ** max(0, down_blocks + self.stem_blocks - 1 - block))
            )

            if self.block_contraction:
                block_filters_out = int(
                    self.filters
                    * (
                        self.filters_rate
                        ** (self.down_blocks + self.stem_blocks - 2 - block)
                    )
                )

            next_stride = current_stride // 2

            # Determine skip channels for this decoder block
            # If encoder_channels provided, use actual encoder channels
            # Otherwise fall back to computed filters (for UNet compatibility)
            skip_channels = None
            if encoder_channels is not None and block < len(encoder_channels):
                skip_channels = encoder_channels[block]

            if self.stem_blocks > 0 and block >= down_blocks + self.stem_blocks:
                # This accounts for the case where we dont have any more down block features to concatenate with.
                # In this case, add a simple upsampling block with a conv layer and with no concatenation
                self.decoder_stack.append(
                    SimpleUpsamplingBlock(
                        x_in_shape=(x_in_shape if block == 0 else prev_block_filters),
                        current_stride=current_stride,
                        upsampling_stride=2,
                        interp_method="bilinear",
                        refine_convs=1,
                        refine_convs_filters=block_filters_out,
                        refine_convs_kernel_size=self.kernel_size,
                        refine_convs_batch_norm=False,
                        up_interpolate=up_interpolate,
                        transpose_convs_filters=block_filters_out,
                        transpose_convs_batch_norm=False,
                        feat_concat=False,
                        prefix=f"{self.prefix}{block}_s{current_stride}_to_s{next_stride}",
                        skip_channels=skip_channels,
                    )
                )
            else:
                self.decoder_stack.append(
                    SimpleUpsamplingBlock(
                        x_in_shape=(x_in_shape if block == 0 else prev_block_filters),
                        current_stride=current_stride,
                        upsampling_stride=2,
                        interp_method="bilinear",
                        refine_convs=self.convs_per_block,
                        refine_convs_filters=block_filters_out,
                        refine_convs_kernel_size=self.kernel_size,
                        refine_convs_batch_norm=False,
                        up_interpolate=up_interpolate,
                        transpose_convs_filters=block_filters_out,
                        transpose_convs_batch_norm=False,
                        prefix=f"{self.prefix}{block}_s{current_stride}_to_s{next_stride}",
                        skip_channels=skip_channels,
                    )
                )

            self.stride_to_filters[next_stride] = block_filters_out

            self.current_strides.append(next_stride)
            current_stride = next_stride
            self.residuals += 1

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
            if i < len(features):
                x = self.decoder_stack[i](x, features[i])
            else:
                x = self.decoder_stack[i](x, None)
            outputs["outputs"].append(x)
        outputs["strides"] = self.current_strides

        return outputs
