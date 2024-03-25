from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Dict, Tuple, Union, Text

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from sleap_nn.architectures.encoder_decoder import Decoder
import numpy as np
from sleap_nn.architectures.common import xavier_init_weights

from torchvision.models.convnext import LayerNorm2d, CNBlock, CNBlockConfig

__all__ = [
    "ConvNeXt",
    "ConvNeXt_Tiny_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "ConvNextWrapper",
]


class ConvNeXt(nn.Module):
    def __init__(
        self,
        blocks: Dict,
        in_channels: int = 1,
        stem_kernel: int = 4,
        stem_stride: int = 4,
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


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
    "_docs": """
        These weights improve upon the results of the original paper by using a modified version of TorchVision's
        `new training recipe
        <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    """,
}


class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=236),
        meta={
            **_COMMON_META,
            "num_params": 28589128,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.520,
                    "acc@5": 96.146,
                }
            },
            "_ops": 4.456,
            "_file_size": 109.119,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_small-0c510722.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=230),
        meta={
            **_COMMON_META,
            "num_params": 50223688,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.616,
                    "acc@5": 96.650,
                }
            },
            "_ops": 8.684,
            "_file_size": 191.703,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88591464,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
            "_ops": 15.355,
            "_file_size": 338.064,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 197767336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.414,
                    "acc@5": 96.976,
                }
            },
            "_ops": 34.361,
            "_file_size": 754.537,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNextWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        arch: Dict = None,
        kernel_size: int = 3,
        stem_patch_kernel: int = 4,
        stem_patch_stride: int = 2,
        filters_rate: int = 1.5,
        up_blocks: int = 3,
        convs_per_block: int = 2,
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
        self.enc = ConvNeXt(
            blocks=arch,
            in_channels=in_channels,
            stem_stride=stem_patch_stride,
            stem_kernel=stem_patch_kernel,
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
        )

    @property
    def output_channels(self):
        """Returns the output channels of the UNet."""
        return int(
            self.arch["channels"][0]
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
