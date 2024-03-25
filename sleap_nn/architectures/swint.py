from functools import partial
from typing import Any, Callable, List, Optional, Dict, Tuple

import torch
from torch import nn
from sleap_nn.architectures.encoder_decoder import Decoder

from torchvision.ops.misc import Permute
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.swin_transformer import (
    _patch_merging_pad,
    _get_relative_position_bias,
    PatchMerging,
    PatchMergingV2,
    shifted_window_attention,
    ShiftedWindowAttention,
    ShiftedWindowAttentionV2,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)


__all__ = [
    "SwinTransformer",
    "SwinTWrapper",
]


torch.fx.wrap("_patch_merging_pad")
torch.fx.wrap("_get_relative_position_bias")
torch.fx.wrap("shifted_window_attention")


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
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
        in_channels: int,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        stem_stride=4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
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
                # nn.Conv2d(
                #     in_channels, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                # ),
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
        features_list = []
        features_list.append(x)
        for idx, l in enumerate(self.features):
            x = l(x)
            if idx == len(self.features) - 1:
                x = self.norm(x)
            features_list.append(self.permute(x))
        return features_list


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}


class SwinTWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stem_stride=2,
        stochastic_depth_prob=0.2,
        norm_layer="",
        kernel_size: int = 3,
        filters_rate: int = 1.5,
        up_blocks: int = 3,
        convs_per_block: int = 2,
    ) -> None:
        """Initialize the class."""
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.filters_rate = filters_rate
        self.up_blocks = up_blocks
        self.convs_per_block = convs_per_block
        self.embed_dim = embed_dim
        self.stem_stride = stem_stride
        self.down_blocks = len(depths) - 1
        self.enc = SwinTransformer(
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
        )

    @property
    def output_channels(self):
        """Returns the output channels of the UNet."""
        return int(
            self.embed_dim
            * (self.filters_rate ** (self.down_blocks - 1 - self.up_blocks + 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List]:
        """Forward pass through the U-Net architecture.

        Args:
            x: Input tensor.

        Returns:
            x: Output a tensor after applying the U-Net operations.
            current_strides: a list of the current strides from the decoder.
        """
        enc_output = self.enc(x)
        x, features = enc_output[-1], enc_output[::2]
        features = features[:-1][::-1]
        x = self.dec(x, features)
        return x