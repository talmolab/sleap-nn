"""This module provides a generalized implementation of ConvNext.

See the `ConvNextWrapper` class docstring for more information.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once
from sleap_nn.architectures.encoder_decoder import Decoder
from torchvision.models.convnext import LayerNorm2d, CNBlock, CNBlockConfig


class ConvNeXtEncoder(nn.Module):
    """
    Src: torchvision.models
    Implements ConvNext from the `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        blocks (dict) : Dictionary of depths and channels. Default is "Tiny architecture" {'depths': [3,3,9,3], 'channels':[96, 192, 384, 768]}
        in_channels (int): Input number of channels. Default: 1.
        stem_kernel (int): Size of the convolutional kernels in the stem layer. Default is 4.
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
        super().__init__()
        _log_api_usage_once(self)

        if not blocks:
            raise ValueError("The block_setting should not be empty")

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
        features_list.append(x)
        for l in self.features:
            x = l(x)
            features_list.append(x)
        return features_list

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ConvNextWrapper(nn.Module):
    """ConvNext architecture for pose estimation.

    This class defines the ConvNext architecture for pose estimation, combining an
    ConvNext as the encoder and a decoder. The encoder extracts features from the input, while the
    decoder generates confidence maps based on the features.

    Args:
        in_channels: Number of input channels. Default is 1.
        arch: Dictionary of depths and channels. Default is "Tiny architecture" {'depths': [3,3,9,3], 'channels':[96, 192, 384, 768]}
        kernel_size: Size of the convolutional kernels. Default is 3.
        stem_patch_kernel: Size of the convolutional kernels in the stem layer. Default is 4.
        stem_patch_stride: Convolutional stride in the stem layer. Default is 2.
        filters_rate: Factor to adjust the number of filters per block. Default is 2.
        up_blocks: Number of upsampling blocks in the decoder. Default is 3.
        convs_per_block: Number of convolutional layers per block. Default is 2.

    Attributes:
        Inherits all attributes from torchvision.models.
    """

    def __init__(
        self,
        arch: dict = {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels: int = 1,
        kernel_size: int = 3,
        stem_patch_kernel: int = 4,
        stem_patch_stride: int = 2,
        filters_rate: int = 2,
        up_blocks: int = 3,
        convs_per_block: int = 2,
        **kwargs,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.filters_rate = filters_rate
        self.up_blocks = up_blocks
        self.convs_per_block = convs_per_block
        self.stem_patch_kernel = stem_patch_kernel
        self.stem_patch_stride = stem_patch_stride
        self.arch = arch
        self.down_blocks = len(self.arch["depths"]) - 1
        self.enc = ConvNeXtEncoder(
            blocks=arch,
            in_channels=in_channels,
            stem_stride=stem_patch_stride,
            stem_kernel=stem_patch_kernel,
            kwargs=kwargs,
        )

        current_stride = self.stem_patch_stride * (2 ** (self.down_blocks - 1))
        x_in_shape = self.arch["channels"][-1]

        self.dec = Decoder(
            x_in_shape=x_in_shape,
            current_stride=current_stride,
            filters=self.arch["channels"][0],
            up_blocks=self.up_blocks,
            down_blocks=self.down_blocks,
            filters_rate=filters_rate,
            kernel_size=self.kernel_size,
        )

    @property
    def output_channels(self):
        """Returns the output channels of the ConvNext model."""
        return int(
            self.arch["channels"][0]
            * (self.filters_rate ** (self.down_blocks - 1 - self.up_blocks + 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the ConvNext architecture.

        Args:
            x: Input tensor.

        Returns:
            x: Output a tensor after applying the ConvNext Encoder and Decoder operations.
            current_strides: a list of the current strides from the decoder.
        """
        enc_output = self.enc(x)
        x, features = enc_output[-1], enc_output[::2]
        features = features[:-1][::-1]
        x = self.dec(x, features)
        return x
