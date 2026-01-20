"""This module provides a generalized implementation of ConvNext.

See the `ConvNextWrapper` class docstring for more information.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Dict, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once
from sleap_nn.architectures.encoder_decoder import Decoder, SimpleConvBlock
from torchvision.models.convnext import LayerNorm2d, CNBlock, CNBlockConfig
from omegaconf import OmegaConf
from sleap_nn.architectures.common import MaxPool2dWithSamePadding


class ConvNeXtEncoder(nn.Module):
    """ConvNext backbone for pose estimation.

    This class implements ConvNext from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`
    paper. Source: torchvision.models. This module serves as the backbone/ encoder
    architecture to extract features from the input image.

    Args:
        blocks (dict) : Dictionary of depths and channels. Default is "Tiny architecture"
                        {'depths': [3,3,9,3], 'channels':[96, 192, 384, 768]}
        in_channels (int): Input number of channels. Default: 1.
        stem_kernel (int): Size of the convolutional kernels in the stem layer.
                        Default is 4.
        stem_stride (int): Convolutional stride in the stem layer. Default is 2.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        layer_scale (float): Scale for Layer normalization layer. Default: 1e-6.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(
        self,
        blocks: dict = {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels: int = 1,
        stem_kernel: int = 4,
        stem_stride: int = 2,
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ConvNext Encoder."""
        super().__init__()
        _log_api_usage_once(self)

        depths, channels = blocks["depths"], blocks["channels"]
        block_setting = [0] * len(depths)
        for idx in range(len(depths)):
            if idx == len(depths) - 1:
                last = None
            else:
                last = channels[idx + 1]
            block_setting[idx] = CNBlockConfig(channels[idx], last, depths[idx])
        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=1,  ## 0 -> 1
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(
                            cnf.input_channels,
                            cnf.out_channels,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )
        self.features = nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        features_list = []
        for l in self.features:
            x = l(x)
            features_list.append(x)
        return features_list

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ConvNext encoder.

        Args:
            x: Input tensor.

        Returns:
            Outputs a list of tensors from each stage after applying the ConvNext backbone.
        """
        return self._forward_impl(x)


class ConvNextWrapper(nn.Module):
    """ConvNext architecture for pose estimation.

    This class defines the ConvNext architecture for pose estimation, combining an
    ConvNext as the encoder and a decoder. The encoder extracts features from the input,
    while the decoder generates confidence maps based on the features.

    Args:
        model_type: One of the ConvNext architecture types: ["tiny", "small", "base", "large"].
        output_stride: Minimum of the strides of the output heads. The input confidence map.
        tensor is expected to be at the same stride.
        in_channels: Number of input channels. Default is 1.
        arch: Dictionary of depths and channels. Default is "Tiny architecture"
        {'depths': [3,3,9,3], 'channels':[96, 192, 384, 768]}
        kernel_size: Size of the convolutional kernels. Default is 3.
        stem_patch_kernel: Size of the convolutional kernels in the stem layer. Default is 4.
        stem_patch_stride: Convolutional stride in the stem layer. Default is 2.
        filters_rate: Factor to adjust the number of filters per block. Default is 2.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        up_interpolate: If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales.
        max_stride: Factor by which input image size is reduced through the layers. This is always `16` for all convnext architectures.
        block_contraction: If True, reduces the number of filters at the end of middle
            and decoder blocks. This has the effect of introducing an additional
            bottleneck before each upsampling step.

    Attributes:
        Inherits all attributes from torch.nn.Module.
    """

    def __init__(
        self,
        model_type: str,
        output_stride: int,
        arch: dict = {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels: int = 1,
        kernel_size: int = 3,
        stem_patch_kernel: int = 4,
        stem_patch_stride: int = 2,
        filters_rate: int = 2,
        convs_per_block: int = 2,
        up_interpolate: bool = True,
        max_stride: int = 32,
        block_contraction: bool = False,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.filters_rate = filters_rate
        self.block_contraction = block_contraction
        arch_types = {
            "tiny": {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
            "small": {"depths": [3, 3, 27, 3], "channels": [96, 192, 384, 768]},
            "base": {"depths": [3, 3, 27, 3], "channels": [128, 256, 512, 1024]},
            "large": {"depths": [3, 3, 27, 3], "channels": [192, 384, 768, 1536]},
        }
        if model_type in arch_types:
            self.arch = arch_types[model_type]
        elif arch is not None:
            self.arch = arch
        else:
            self.arch = arch_types["tiny"]

        self.max_stride = (
            stem_patch_stride * (2**3) * 2
        )  # stem_stride * down_blocks_stride * final_max_pool_stride
        self.stem_blocks = 1  # 1 stem block + 3 down blocks in convnext

        self.up_blocks = np.log2(
            self.max_stride / (stem_patch_stride * output_stride)
        ).astype(int) + np.log2(stem_patch_stride).astype(int)
        self.convs_per_block = convs_per_block
        self.stem_patch_kernel = stem_patch_kernel
        self.stem_patch_stride = stem_patch_stride
        self.output_stride = output_stride
        self.up_interpolate = up_interpolate
        self.down_blocks = len(self.arch["channels"]) - 1

        self.enc = ConvNeXtEncoder(
            blocks=self.arch,
            in_channels=in_channels,
            stem_stride=stem_patch_stride,
            stem_kernel=stem_patch_kernel,
        )

        # Add additional pooling layer after encoder
        self.additional_pool = MaxPool2dWithSamePadding(
            kernel_size=2, stride=2, padding="same"
        )

        # Create middle blocks
        self.middle_blocks = nn.ModuleList()
        # Get the last block filters from encoder
        last_block_filters = self.arch["channels"][-1]

        if convs_per_block > 1:
            # Middle expansion block
            middle_expand = SimpleConvBlock(
                in_channels=last_block_filters,
                pool=False,
                pool_before_convs=False,
                pooling_stride=2,
                num_convs=convs_per_block - 1,
                filters=int(last_block_filters * filters_rate),
                kernel_size=kernel_size,
                use_bias=True,
                batch_norm=False,
                activation="relu",
                prefix="convnext_middle_expand",
            )
            self.middle_blocks.append(middle_expand)

        # Middle contraction block
        if self.block_contraction:
            # Contract the channels with an exponent lower than the last encoder block
            block_filters = int(last_block_filters)
        else:
            # Keep the block output filters the same
            block_filters = int(last_block_filters * filters_rate)

        middle_contract = SimpleConvBlock(
            in_channels=int(last_block_filters * filters_rate),
            pool=False,
            pool_before_convs=False,
            pooling_stride=2,
            num_convs=1,
            filters=block_filters,
            kernel_size=kernel_size,
            use_bias=True,
            batch_norm=False,
            activation="relu",
            prefix="convnext_middle_contract",
        )
        self.middle_blocks.append(middle_contract)

        self.current_stride = (
            self.stem_patch_stride * (2**3) * 2
        )  # stem_stride * down_blocks_stride * pool

        # Calculate x_in_shape based on whether we have block contraction
        if self.block_contraction:
            # Contract the channels with an exponent lower than the last encoder block
            x_in_shape = int(self.arch["channels"][-1])
        else:
            # Keep the block output filters the same
            x_in_shape = int(self.arch["channels"][-1] * filters_rate)

        # Encoder channels for skip connections (reversed to match decoder order)
        # The forward pass uses enc_output[::2][::-1] for skip features
        encoder_channels = self.arch["channels"][::-1]

        self.dec = Decoder(
            x_in_shape=x_in_shape,
            current_stride=self.current_stride,
            filters=self.arch["channels"][0],
            up_blocks=self.up_blocks,
            down_blocks=self.down_blocks,
            filters_rate=filters_rate,
            kernel_size=self.kernel_size,
            stem_blocks=1,
            block_contraction=self.block_contraction,
            output_stride=self.output_stride,
            up_interpolate=up_interpolate,
            encoder_channels=encoder_channels,
        )

        if len(self.dec.decoder_stack):
            self.final_dec_channels = self.dec.decoder_stack[-1].refine_convs_filters
        else:
            self.final_dec_channels = block_filters

        self.decoder_stride_to_filters = self.dec.stride_to_filters

    @property
    def max_channels(self):
        """Returns the maximum channels of the ConvNext (last layer of the encoder)."""
        return self.dec.x_in_shape

    @classmethod
    def from_config(cls, config: OmegaConf):
        """Create ConvNextWrapper from a config."""
        return cls(
            in_channels=config.in_channels,
            model_type=config.model_type,
            arch=config.arch,
            kernel_size=config.kernel_size,
            filters_rate=config.filters_rate,
            convs_per_block=config.convs_per_block,
            up_interpolate=config.up_interpolate,
            output_stride=config.output_stride,
            stem_patch_kernel=config.stem_patch_kernel,
            stem_patch_stride=config.stem_patch_stride,
            max_stride=config.max_stride,
            block_contraction=(
                config.block_contraction
                if hasattr(config, "block_contraction")
                else False
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the ConvNeXt architecture.

        Args:
            x: Input tensor (Batch, Channels, Height, Width).

        Returns:
            x: Outputs a dictionary with `outputs` and `strides` containing the output
            at different strides.
        """
        enc_output = self.enc(x)
        x, features = enc_output[-1], enc_output[::2]
        features = features[::-1]

        # Apply additional pooling layer
        x = self.additional_pool(x)

        # Process through middle blocks
        middle_output = x
        for middle_block in self.middle_blocks:
            middle_output = middle_block(middle_output)

        x = self.dec(middle_output, features)
        x["middle_output"] = middle_output
        return x
