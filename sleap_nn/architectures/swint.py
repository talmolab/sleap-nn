"""This module provides a generalized implementation of SwinT.

See the `SwinTWrapper` class docstring for more information.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Dict, Tuple
import numpy as np
import torch
from torch import nn
from sleap_nn.architectures.encoder_decoder import Decoder
from omegaconf import OmegaConf
from torchvision.ops.misc import Permute
from sleap_nn.architectures.encoder_decoder import Decoder, SimpleConvBlock
from sleap_nn.architectures.common import MaxPool2dWithSamePadding

from torchvision.utils import _log_api_usage_once
from torchvision.models.swin_transformer import (
    PatchMerging,
    PatchMergingV2,
    shifted_window_attention,
    ShiftedWindowAttention,
    ShiftedWindowAttentionV2,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)


torch.fx.wrap("_patch_merging_pad")
torch.fx.wrap("_get_relative_position_bias")
torch.fx.wrap("shifted_window_attention")


class SwinTransformerEncoder(nn.Module):
    """SwinT backbone for pose estimation.

    This class implements ConvNext from the `"Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows `<https://arxiv.org/abs/2103.14030>`paper.
    Source: torchvision.models. This module serves as the backbone/ encoder architecture
    to extract features from the input image.

    Args:
        in_channels (int): Number of input channels. Default is 1.
        patch_size (List[int]): Patch size. Default: [4,4]
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (List(int)): Depth of each Swin Transformer layer. Default: [2,2,6,2].
        num_heads (List(int)): Number of attention heads in different layers.
                        Default: [3,6,12,24].
        window_size (List[int]): Window size. Default: [7,7].
        stem_stride (int): Stride for the patch. Default is 2.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: List[int] = [4, 4],
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [7, 7],
        stem_stride: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """Initialize the class."""
        super().__init__()
        _log_api_usage_once(self)

        block = SwinTransformerBlock  # or v2
        downsample_layer = PatchMerging  # or v2

        if not norm_layer:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    stride=(stem_stride, stem_stride),
                    padding=1,
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

    def forward(self, x):
        """Forward pass through the SwinT encoder.

        Args:
            x: Input tensor.

        Returns:
            Outputs a list of tensors from each stage after applying the SwinT backbone.
        """
        features_list = []
        for idx, l in enumerate(self.features):
            x = l(x)
            if idx == len(self.features) - 1:
                x = self.norm(x)
            features_list.append(self.permute(x))
        return features_list


class SwinTWrapper(nn.Module):
    """SwinT architecture for pose estimation.

    This class defines the SwinT architecture for pose estimation, combining an
    SwinT as the encoder and a decoder. The encoder extracts features from the input,
    while the decoder generates confidence maps based on the features.

    Args:
        in_channels: Number of input channels. Default is 1.
        model_type: One of the ConvNext architecture types: ["tiny", "small", "base"].
        output_stride: Minimum of the strides of the output heads. The input confidence map.
        patch_size: Patch size. Default: [4,4]
        arch: Dictionary of embed dimension, depths and number of heads in each layer.
        Default is "Tiny architecture".
        {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}
        window_size: Window size. Default: [7,7].
        stem_patch_stride: Stride for the patch. Default is 2.
        kernel_size: Size of the convolutional kernels. Default is 3.
        filters_rate: Factor to adjust the number of filters per block. Default is 2.
        convs_per_block: Number of convolutional layers per block. Default is 2.
        up_interpolate: If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales.
        max_stride: Factor by which input image size is reduced through the layers.
            This is always `16` for all swint architectures.
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
        in_channels: int = 1,
        patch_size: List[int] = [4, 4],
        arch: dict = {"embed": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
        window_size: List[int] = [7, 7],
        stem_patch_stride: int = 2,
        kernel_size: int = 3,
        filters_rate: int = 2,
        convs_per_block: int = 2,
        up_interpolate: bool = True,
        max_stride: int = 32,
        block_contraction: bool = False,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.filters_rate = filters_rate
        self.block_contraction = block_contraction
        arch_types = {
            "tiny": {"embed": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
            "small": {
                "embed": 96,
                "depths": [2, 2, 18, 2],
                "num_heads": [3, 6, 12, 24],
            },
            "base": {
                "embed": 128,
                "depths": [2, 2, 18, 2],
                "num_heads": [4, 8, 16, 32],
            },
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
        self.stem_blocks = 1  # 1 stem block + 3 down blocks in swint

        self.up_blocks = np.log2(
            self.max_stride / (stem_patch_stride * output_stride)
        ).astype(int) + np.log2(stem_patch_stride).astype(int)
        self.convs_per_block = convs_per_block
        self.stem_patch_stride = stem_patch_stride
        self.down_blocks = len(self.arch["depths"]) - 1
        self.enc = SwinTransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=self.arch["embed"],
            depths=self.arch["depths"],
            num_heads=self.arch["num_heads"],
            window_size=window_size,
            stem_stride=stem_patch_stride,
        )

        self.additional_pool = MaxPool2dWithSamePadding(
            kernel_size=2, stride=2, padding="same"
        )

        # Create middle blocks
        self.middle_blocks = nn.ModuleList()
        # Get the last block filters from encoder
        last_block_filters = self.arch["embed"] * (2 ** (self.down_blocks))

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
        )  # stem_stride * down_blocks_stride * final_max_pool_stride

        # Encoder channels for skip connections (reversed to match decoder order)
        # SwinT channels: embed * 2^i for each stage i, then reversed
        num_stages = len(self.arch["depths"])
        encoder_channels = [
            self.arch["embed"] * (2 ** (num_stages - 1 - i)) for i in range(num_stages)
        ]

        self.dec = Decoder(
            x_in_shape=block_filters,
            current_stride=self.current_stride,
            filters=self.arch["embed"],
            up_blocks=self.up_blocks,
            down_blocks=self.down_blocks,
            filters_rate=filters_rate,
            kernel_size=self.kernel_size,
            stem_blocks=self.stem_blocks,
            block_contraction=self.block_contraction,
            output_stride=output_stride,
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
        """Returns the maximum channels of the SwinT (last layer of the encoder)."""
        return self.dec.x_in_shape

    @classmethod
    def from_config(cls, config: OmegaConf):
        """Create SwinTWrapper from a config."""
        return cls(
            in_channels=config.in_channels,
            model_type=config.model_type,
            arch=config.arch,
            patch_size=(config.patch_size, config.patch_size),
            window_size=(config.window_size, config.window_size),
            kernel_size=config.kernel_size,
            filters_rate=config.filters_rate,
            convs_per_block=config.convs_per_block,
            up_interpolate=config.up_interpolate,
            output_stride=config.output_stride,
            stem_patch_stride=config.stem_patch_stride,
            max_stride=config.max_stride,
            block_contraction=(
                config.block_contraction
                if hasattr(config, "block_contraction")
                else False
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the SwinT architecture.

        Args:
            x: Input tensor.

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
