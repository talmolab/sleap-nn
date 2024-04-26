"""This module provides a generalized implementation of SwinT.

See the `SwinTWrapper` class docstring for more information.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Dict, Tuple

import torch
from torch import nn
from sleap_nn.architectures.encoder_decoder import Decoder

from torchvision.ops.misc import Permute
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
        features_list.append(x)
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
        patch_size (List[int]): Patch size. Default: [4,4]
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (List(int)): Depth of each Swin Transformer layer. Default: [2,2,6,2].
        num_heads (List(int)): Number of attention heads in different layers.
                            Default: [3,6,12,24].
        window_size (List[int]): Window size. Default: [7,7].
        stem_stride (int): Stride for the patch. Default is 2.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        norm_layer: Normalization layer. Default: None.
        kernel_size: Size of the convolutional kernels. Default is 3.
        filters_rate: Factor to adjust the number of filters per block. Default is 2.
        up_blocks: Number of upsampling blocks in the decoder. Default is 3.
        convs_per_block: Number of convolutional layers per block. Default is 2.

    Attributes:
        Inherits all attributes from torch.nn.Module.
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
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = "",
        kernel_size: int = 3,
        filters_rate: int = 2,
        up_blocks: int = 3,
        convs_per_block: int = 2,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.filters_rate = filters_rate
        self.up_blocks = up_blocks
        self.convs_per_block = convs_per_block
        self.embed_dim = embed_dim
        self.stem_stride = stem_stride
        self.down_blocks = len(depths) - 1
        self.enc = SwinTransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stem_stride=stem_stride,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
        )
        current_stride = self.stem_stride * (2 ** (self.down_blocks - 1))

        x_in_shape = embed_dim * (2 ** (self.down_blocks))

        self.dec = Decoder(
            x_in_shape=x_in_shape,
            current_stride=current_stride,
            filters=embed_dim,
            up_blocks=self.up_blocks,
            down_blocks=self.down_blocks,
            filters_rate=filters_rate,
            kernel_size=self.kernel_size,
        )

    @property
    def output_channels(self):
        """Returns the output channels of the SwinT."""
        return int(
            self.embed_dim
            * (self.filters_rate ** (self.down_blocks - 1 - self.up_blocks + 1))
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
        features = features[:-1][::-1]
        x = self.dec(x, features)
        return x
