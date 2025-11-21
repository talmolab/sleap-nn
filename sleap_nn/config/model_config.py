"""Serializable configuration classes for specifying all model config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the model config.
"""

from attrs import define, field
from sleap_nn.config.utils import oneof
from typing import Optional, List
from loguru import logger


# Define configuration for each backbone type (unet, convnext, swint) configurations
@define
class UNetConfig:
    """UNet config for backbone.

    Attributes:
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters: (int) Base number of filters in the network. *Default*: `32`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `1.5`.
        max_stride: (int) Scalar integer specifying the maximum stride that the image must be divisible by. *Default*: `16`.
        stem_stride: (int) If not None, will create additional "down" blocks for initial downsampling based on the stride. These will be configured identically to the down blocks below. *Default*: `None`.
        middle_block: (bool) If True, add an additional block at the end of the encoder. *Default*: `True`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        stacks: (int) Number of upsampling blocks in the decoder. *Default*: `1`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
    """

    in_channels: int = 1
    kernel_size: int = 3
    filters: int = 32
    filters_rate: float = 1.5
    max_stride: int = 16
    stem_stride: Optional[int] = None
    middle_block: bool = True
    up_interpolate: bool = True
    stacks: int = 1
    convs_per_block: int = 2
    output_stride: int = 1


@define
class UNetLargeRFConfig(UNetConfig):
    """UNet config for backbone with large receptive field.

    Attributes:
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters: (int) Base number of filters in the network. *Default*: `24`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `1.5`.
        max_stride: (int) Scalar integer specifying the maximum stride that the image must be divisible by. *Default*: `32`.
        stem_stride: (int) If not None, will create additional "down" blocks for initial downsampling based on the stride. These will be configured identically to the down blocks below. *Default*: `None`.
        middle_block: (bool) If True, add an additional block at the end of the encoder. *Default*: `True`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        stacks: (int) Number of upsampling blocks in the decoder. *Default*: `1`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
    """

    in_channels: int = 1
    kernel_size: int = 3
    filters: int = 24
    filters_rate: float = 1.5
    max_stride: int = 32
    stem_stride: Optional[int] = None
    middle_block: bool = True
    up_interpolate: bool = True
    stacks: int = 1
    convs_per_block: int = 2
    output_stride: int = 1


@define
class UNetMediumRFConfig(UNetConfig):
    """UNet config for backbone with medium receptive field.

    Attributes:
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters: (int) Base number of filters in the network. *Default*: `32`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `2`.
        max_stride: (int) Scalar integer specifying the maximum stride that the image must be divisible by. *Default*: `16`.
        stem_stride: (int) If not None, will create additional "down" blocks for initial downsampling based on the stride. These will be configured identically to the down blocks below. *Default*: `None`.
        middle_block: (bool) If True, add an additional block at the end of the encoder. *Default*: `True`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        stacks: (int) Number of upsampling blocks in the decoder. *Default*: `1`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
    """

    in_channels: int = 1
    kernel_size: int = 3
    filters: int = 32
    filters_rate: float = 2
    max_stride: int = 16
    stem_stride: Optional[int] = None
    middle_block: bool = True
    up_interpolate: bool = True
    stacks: int = 1
    convs_per_block: int = 2
    output_stride: int = 1


@define
class ConvNextConfig:
    """Convnext configuration for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            ConvNext backbones. For ConvNext, one of ["ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"].
        arch: (Default is Tiny architecture config. No need to provide if model_type is provided)
            depths: (List[int]) Number of layers in each block. *Default*: `[3, 3, 9, 3]`.
            channels: (List[int]) Number of channels in each block. *Default*: `[96, 192, 384, 768]`.
        model_type: (str) One of the ConvNext architecture types: ["tiny", "small", "base", "large"]. *Default*: `"tiny"`.
        stem_patch_kernel: (int) Size of the convolutional kernels in the stem layer. *Default*: `4`.
        stem_patch_stride: (int) Convolutional stride in the stem layer. *Default*: `2`.
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `2`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
        max_stride: (int) Factor by which input image size is reduced through the layers. This is always `32` for all convnext architectures. *Default*: `32`.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = "tiny"  # Options: tiny, small, base, large
    arch: Optional[dict] = None
    stem_patch_kernel: int = 4
    stem_patch_stride: int = 2
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1
    max_stride: int = 32

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        convnext_weights are one of
        (
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        )
        """
        if value is None:
            return

        convnext_weights = [
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        ]

        if value not in convnext_weights:
            message = f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}"
            logger.error(message)
            raise ValueError(message)


@define
class ConvNextSmallConfig(ConvNextConfig):
    """Convnext configuration for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            ConvNext backbones. For ConvNext, one of ["ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"].
        arch: (Default is Tiny architecture config. No need to provide if model_type
            is provided)
            depths: (List(int)) Number of layers in each block. Default: [3, 3, 9, 3].
            channels: (List(int)) Number of channels in each block. Default:
                [96, 192, 384, 768].
        model_type: (str) One of the ConvNext architecture types:
            ["tiny", "small", "base", "large"]. Default: "tiny".
        stem_patch_kernel: (int) Size of the convolutional kernels in the stem layer.
            Default is 4.
        stem_patch_stride: (int) Convolutional stride in the stem layer. Default is 2.
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters_rate: (float) Factor to adjust the number of filters per block.
            Default is 2.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output stride
            of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
        max_stride: Factor by which input image size is reduced through the layers.
            This is always `32` for all convnext architectures.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = "small"  # Options: tiny, small, base, large
    arch: Optional[dict] = None
    stem_patch_kernel: int = 4
    stem_patch_stride: int = 2
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1
    max_stride: int = 32

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        convnext_weights are one of
        (
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        )
        """
        if value is None:
            return

        convnext_weights = [
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        ]

        if value not in convnext_weights:
            message = f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}"
            logger.error(message)
            raise ValueError(message)


@define
class ConvNextBaseConfig(ConvNextConfig):
    """Convnext configuration for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            ConvNext backbones. For ConvNext, one of ["ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"].
        arch: (Default is Tiny architecture config. No need to provide if model_type
            is provided)
            depths: (List(int)) Number of layers in each block. Default: [3, 3, 9, 3].
            channels: (List(int)) Number of channels in each block. Default:
                [96, 192, 384, 768].
        model_type: (str) One of the ConvNext architecture types:
            ["tiny", "small", "base", "large"]. Default: "tiny".
        stem_patch_kernel: (int) Size of the convolutional kernels in the stem layer.
            Default is 4.
        stem_patch_stride: (int) Convolutional stride in the stem layer. Default is 2.
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters_rate: (float) Factor to adjust the number of filters per block.
            Default is 2.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output stride
            of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
        max_stride: Factor by which input image size is reduced through the layers.
            This is always `32` for all convnext architectures.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = "base"  # Options: tiny, small, base, large
    arch: Optional[dict] = None
    stem_patch_kernel: int = 4
    stem_patch_stride: int = 2
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1
    max_stride: int = 32

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        convnext_weights are one of
        (
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        )
        """
        if value is None:
            return

        convnext_weights = [
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        ]

        if value not in convnext_weights:
            message = f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}"
            logger.error(message)
            raise ValueError(message)


@define
class ConvNextLargeConfig(ConvNextConfig):
    """Convnext configuration for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            ConvNext backbones. For ConvNext, one of ["ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"].
        arch: (Default is Tiny architecture config. No need to provide if model_type
            is provided)
            depths: (List(int)) Number of layers in each block. Default: [3, 3, 9, 3].
            channels: (List(int)) Number of channels in each block. Default:
                [96, 192, 384, 768].
        model_type: (str) One of the ConvNext architecture types:
            ["tiny", "small", "base", "large"]. Default: "tiny".
        stem_patch_kernel: (int) Size of the convolutional kernels in the stem layer.
            Default is 4.
        stem_patch_stride: (int) Convolutional stride in the stem layer. Default is 2.
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters_rate: (float) Factor to adjust the number of filters per block.
            Default is 2.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed
            convolutions for upsampling. Interpolation is faster but transposed
            convolutions may be able to learn richer or more complex upsampling to
            recover details from higher scales. Default: True.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output stride
            of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
        max_stride: Factor by which input image size is reduced through the layers.
            This is always `32` for all convnext architectures.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = "large"  # Options: tiny, small, base, large
    arch: Optional[dict] = None
    stem_patch_kernel: int = 4
    stem_patch_stride: int = 2
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1
    max_stride: int = 32

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        convnext_weights are one of
        (
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        )
        """
        if value is None:
            return

        convnext_weights = [
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        ]

        if value not in convnext_weights:
            message = f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}"
            logger.error(message)
            raise ValueError(message)


@define
class SwinTConfig:
    """SwinT configuration (tiny) for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            SwinT backbones. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        model_type: (str) One of the SwinT architecture types: ["tiny", "small", "base"]. *Default*: `"tiny"`.
        arch: Dictionary of embed dimension, depths and number of heads in each layer. Default is "Tiny architecture". {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}. *Default*: `None`.
        max_stride: (int) Factor by which input image size is reduced through the layers. This is always `32` for all swint architectures. *Default*: `32`.
        patch_size: (int) Patch size for the stem layer of SwinT. *Default*: `4`.
        stem_patch_stride: (int) Stride for the patch. *Default*: `2`.
        window_size: (int) Window size. *Default*: `7`.
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `2`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = field(
        default="tiny",
        validator=lambda instance, attr, value: instance.validate_model_type(value),
    )
    arch: Optional[dict] = None
    max_stride: int = 32
    patch_size: int = 4
    stem_patch_stride: int = 2
    window_size: int = 7
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1

    def validate_model_type(self, value):
        """Validate model_type.

        Ensure model_type is one of "tiny", "small", or "base".
        """
        valid_types = ["tiny", "small", "base"]
        if value not in valid_types:
            message = f"Invalid model_type. Must be one of {valid_types}"
            logger.error(message)
            raise ValueError(message)

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        swint_weights are one of
        (
            "Swin_T_Weights",
            "Swin_S_Weights",
            "Swin_B_Weights"
        )
        """
        if value is None:
            return

        swint_weights = ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]

        if value not in swint_weights:
            message = (
                f"Invalid pre-trained weights for SwinT. Must be one of {swint_weights}"
            )
            logger.error(message)
            raise ValueError(message)


@define
class SwinTSmallConfig(SwinTConfig):
    """SwinT configuration (small) for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            SwinT backbones. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        model_type: (str) One of the SwinT architecture types: ["tiny", "small", "base"]. *Default*: `"small"`.
        arch: Dictionary of embed dimension, depths and number of heads in each layer. Default is "Tiny architecture". {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}. *Default*: `None`.
        max_stride: (int) Factor by which input image size is reduced through the layers. This is always `32` for all swint architectures. *Default*: `32`.
        patch_size: (int) Patch size for the stem layer of SwinT. *Default*: `4`.
        stem_patch_stride: (int) Stride for the patch. *Default*: `2`.
        window_size: (int) Window size. *Default*: `7`.
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `2`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.
    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = field(
        default="small",
        validator=lambda instance, attr, value: instance.validate_model_type(value),
    )
    arch: Optional[dict] = None
    max_stride: int = 32
    patch_size: int = 4
    stem_patch_stride: int = 2
    window_size: int = 7
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1

    def validate_model_type(self, value):
        """Validate model_type.

        Ensure model_type is one of "tiny", "small", or "base".
        """
        valid_types = ["tiny", "small", "base"]
        if value not in valid_types:
            message = f"Invalid model_type. Must be one of {valid_types}"
            logger.error(message)
            raise ValueError(message)

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        swint_weights are one of
        (
            "Swin_T_Weights",
            "Swin_S_Weights",
            "Swin_B_Weights"
        )
        """
        if value is None:
            return

        swint_weights = ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]

        if value not in swint_weights:
            message = (
                f"Invalid pre-trained weights for SwinT. Must be one of {swint_weights}"
            )
            logger.error(message)
            raise ValueError(message)


@define
class SwinTBaseConfig(SwinTConfig):
    """SwinT configuration for backbone.

    Attributes:
        pre_trained_weights: (str) Pretrained weights file name supported only for
            SwinT backbones. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        model_type: (str) One of the SwinT architecture types: ["tiny", "small", "base"]. *Default*: `"base"`.
        arch: Dictionary of embed dimension, depths and number of heads in each layer. Default is "Tiny architecture". {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}. *Default*: `None`.
        max_stride: (int) Factor by which input image size is reduced through the layers. This is always `32` for all swint architectures. *Default*: `32`.
        patch_size: (int) Patch size for the stem layer of SwinT. *Default*: `4`.
        stem_patch_stride: (int) Stride for the patch. *Default*: `2`.
        window_size: (int) Window size. *Default*: `7`.
        in_channels: (int) Number of input channels. *Default*: `1`.
        kernel_size: (int) Size of the convolutional kernels. *Default*: `3`.
        filters_rate: (float) Factor to adjust the number of filters per block. *Default*: `2`.
        convs_per_block: (int) Number of convolutional layers per block. *Default*: `2`.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. *Default*: `True`.
        output_stride: (int) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution. *Default*: `1`.

    """

    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    model_type: str = field(
        default="base",
        validator=lambda instance, attr, value: instance.validate_model_type(value),
    )
    arch: Optional[dict] = None
    max_stride: int = 32
    patch_size: int = 4
    stem_patch_stride: int = 2
    window_size: int = 7
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 2
    convs_per_block: int = 2
    up_interpolate: bool = True
    output_stride: int = 1

    def validate_model_type(self, value):
        """Validate model_type.

        Ensure model_type is one of "tiny", "small", or "base".
        """
        valid_types = ["tiny", "small", "base"]
        if value not in valid_types:
            message = f"Invalid model_type. Must be one of {valid_types}"
            logger.error(message)
            raise ValueError(message)

    def validate_pre_trained_weights(self, value):
        """Validate pre_trained_weights.

        Check:
        swint_weights are one of
        (
            "Swin_T_Weights",
            "Swin_S_Weights",
            "Swin_B_Weights"
        )
        """
        if value is None:
            return

        swint_weights = ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]

        if value not in swint_weights:
            message = (
                f"Invalid pre-trained weights for SwinT. Must be one of {swint_weights}"
            )
            logger.error(message)
            raise ValueError(message)


@define
class SingleInstanceConfMapsConfig:
    """Single Instance configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly.
            Else provide text name of the body parts (nodes) that the head will be
            configured to produce. The number of parts determines the number of channels
            in the output. If not specified, all body parts in the skeleton will be used.
            This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a
            scalar float. Smaller values are more precise but may be difficult to learn
            as they have a lower density within the image space. Larger values are
            easier to learn but are less precise with respect to the peak coordinate.
            This spread is in units of pixels of the model input image,
            i.e., the image resolution after any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
    """

    part_names: Optional[List[str]] = None
    sigma: float = 5.0
    output_stride: int = 1


@define
class CentroidConfMapsConfig:
    """Centroid configuration map.

    Attributes:
        anchor_part: (str) Node name to use as the anchor point. If None, the midpoint of the
            bounding box of all visible instance points will be used as the anchor. The bounding
            box midpoint will also be used if the anchor part is specified but not visible in the
            instance. Setting a reliable anchor point can significantly improve topdown model
            accuracy as they benefit from a consistent geometry of the body parts relative to the
            center of the image. Default is None.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a
            scalar float. Smaller values are more precise but may be difficult to learn as
            they have a lower density within the image space. Larger values are easier to
            learn but are less precise with respect to the peak coordinate. This spread is
            in units of pixels of the model input image, i.e., the image resolution after
            any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
    """

    anchor_part: Optional[str] = None
    sigma: float = 5.0
    output_stride: int = 1


@define
class CenteredInstanceConfMapsConfig:
    """Centered Instance configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly.
            Else provide text name of the body parts (nodes) that the head will be
            configured to produce. The number of parts determines the number of channels
            in the output. If not specified, all body parts in the skeleton will be used.
            This config does not apply for 'PartAffinityFieldsHead'.
        anchor_part: (str) Node name to use as the anchor point. If None, the midpoint of the
            bounding box of all visible instance points will be used as the anchor. The bounding
            box midpoint will also be used if the anchor part is specified but not visible in the
            instance. Setting a reliable anchor point can significantly improve topdown model
            accuracy as they benefit from a consistent geometry of the body parts relative to the
            center of the image. Default is None.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a
            scalar float. Smaller values are more precise but may be difficult to learn
            as they have a lower density within the image space. Larger values are
            easier to learn but are less precise with respect to the peak coordinate.
            This spread is in units of pixels of the model input image, i.e., the image
            resolution after any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head
            during training. Increase this to encourage the optimization to focus on
            improving this specific output in multi-head models.
    """

    part_names: Optional[List[str]] = None
    anchor_part: Optional[str] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0


@define
class BottomUpConfMapsConfig:
    """Bottomup configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly.
            Else provide text name of the body parts (nodes) that the head will be
            configured to produce. The number of parts determines the number of channels
            in the output. If not specified, all body parts in the skeleton will be used.
            This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a
            scalar float. Smaller values are more precise but may be difficult to learn
            as they have a lower density within the image space. Larger values are easier
            to learn but are less precise with respect to the peak coordinate. This spread
            is in units of pixels of the model input image, i.e., the image resolution
            after any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to the
            input image. This is the reciprocal of the resolution, e.g., an output stride
            of 2 results in confidence maps that are 0.5x the size of the input.
            Increasing this value can considerably speed up model performance and
            decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head
            during training. Increase this to encourage the optimization to focus on
            improving this specific output in multi-head models.
    """

    part_names: Optional[List[str]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: Optional[float] = None


@define
class PAFConfig:
    """PAF configuration map.

    Attributes:
        edges: (List[str]) None if edges from sio.Labels file can be used directly.
            Note: Only for 'PartAffinityFieldsHead'. List of indices (src, dest) that
            form an edge.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as
            a scalar float. Smaller values are more precise but may be difficult to
            learn as they have a lower density within the image space. Larger values
            are easier to learn but are less precise with respect to the peak
            coordinate. This spread is in units of pixels of the model input image,
            i.e., the image resolution after any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to
            the input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in confidence maps that are 0.5x the size of the
            input. Increasing this value can considerably speed up model performance
            and decrease memory requirements, at the cost of decreased spatial
            resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head
            during training. Increase this to encourage the optimization to focus on
            improving this specific output in multi-head models.
    """

    edges: Optional[List[List[str]]] = None
    sigma: float = 15.0
    output_stride: int = 1
    loss_weight: Optional[float] = None


@define
class ClassMapConfig:
    """Class map head config.

    Attributes:
        classes: (List[str]) List of class (track) names. Default is `None`. When `None`, these are inferred from the track names in the labels file.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as
            a scalar float. Smaller values are more precise but may be difficult to
            learn as they have a lower density within the image space. Larger values
            are easier to learn but are less precise with respect to the peak
            coordinate. This spread is in units of pixels of the model input image,
            i.e., the image resolution after any input scaling is applied.
        output_stride: (int) The stride of the output confidence maps relative to
            the input image. This is the reciprocal of the resolution, e.g., an output
            stride of 2 results in confidence maps that are 0.5x the size of the
            input. Increasing this value can considerably speed up model performance
            and decrease memory requirements, at the cost of decreased spatial
            resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head
            during training. Increase this to encourage the optimization to focus on
            improving this specific output in multi-head models.
    """

    classes: Optional[List[str]] = None
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: Optional[float] = None


@define
class ClassVectorsConfig:
    """Configurations for class vectors heads.

    These heads are used in top-down multi-instance models that classify detected
    points using a fixed set of learned classes (e.g., animal identities).

    Attributes:
        classes: List of string names of the classes that this head will predict.
        num_fc_layers: Number of fully-connected layers before the classification output
            layer. These can help in transforming general image features into
            classification-specific features.
        num_fc_units: Number of units (dimensions) in the fully-connected layers before
            classification. Increasing this can improve the representational capacity in
            the pre-classification layers.
        output_stride: (Ideally this should be same as the backbone's maxstride).
            The stride of the output class maps relative to the input image.
            This is the reciprocal of the resolution, e.g., an output stride of 2
            results in maps that are 0.5x the size of the input. This should be the same
            size as the confidence maps they are associated with.
        loss_weight: Scalar float used to weigh the loss term for this head during
            training. Increase this to encourage the optimization to focus on improving
            this specific output in multi-head models.
    """

    classes: Optional[List[str]] = None
    num_fc_layers: int = 1
    num_fc_units: int = 64
    global_pool: bool = True
    output_stride: int = 1
    loss_weight: float = 1.0


@define
class SingleInstanceConfig:
    """single instance head_config."""

    confmaps: Optional[SingleInstanceConfMapsConfig] = None


@define
class CentroidConfig:
    """centroid head_config."""

    confmaps: Optional[CentroidConfMapsConfig] = None


@define
class CenteredInstanceConfig:
    """centered_instance head_config."""

    confmaps: Optional[CenteredInstanceConfMapsConfig] = None


@define
class BottomUpConfig:
    """bottomup head_config."""

    confmaps: Optional[BottomUpConfMapsConfig] = None
    pafs: Optional[PAFConfig] = None


@define
class BottomUpMultiClassConfig:
    """Head config for BottomUp Id models."""

    confmaps: Optional[BottomUpConfMapsConfig] = None
    class_maps: Optional[ClassMapConfig] = None


@define
class TopDownCenteredInstanceMultiClassConfig:
    """Head config for TopDown centered instance ID models."""

    confmaps: Optional[CenteredInstanceConfMapsConfig] = None
    class_vectors: Optional[ClassVectorsConfig] = None


@oneof
@define
class HeadConfig:
    """Configurations related to the model output head type.

    Only one attribute of this class can be set, which defines the model output type.

    Attributes:
        single_instance: An instance of `SingleInstanceConfmapsHeadConfig`.
        centroid: An instance of `CentroidsHeadConfig`.
        centered_instance: An instance of `CenteredInstanceConfmapsHeadConfig`.
        bottomup: An instance of `BottomUpConfig`.
        multi_class_bottomup: An instance of `BottomUpMultiClassConfig`.
        multi_class_topdown: An instance of `TopDownCenteredInstanceMultiClassConfig`.
    """

    single_instance: Optional[SingleInstanceConfig] = None
    centroid: Optional[CentroidConfig] = None
    centered_instance: Optional[CenteredInstanceConfig] = None
    bottomup: Optional[BottomUpConfig] = None
    multi_class_bottomup: Optional[BottomUpMultiClassConfig] = None
    multi_class_topdown: Optional[TopDownCenteredInstanceMultiClassConfig] = None


@oneof
@define
class BackboneConfig:
    """Configurations related to model backbone configuration.

    Attributes:
        unet: An instance of `UNetConfig`.
        convnext: An instance of `ConvNextConfig`.
        swint: An instance of `SwinTConfig`.
    """

    unet: Optional[UNetConfig] = None
    convnext: Optional[ConvNextConfig] = None
    swint: Optional[SwinTConfig] = None


@define
class ModelConfig:
    """Configurations related to model architecture.

    Attributes:
        init_weights: (str) model weights initialization method. "default" uses kaiming
            uniform initialization and "xavier" uses Xavier initialization method.
        pretrained_backbone_weights: Path of the `ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file with which the backbone
            is initialized. If `None`, random init is used.
        pretrained_head_weights: Path of the `ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file with which the head layers
            are initialized. If `None`, random init is used.
        backbone_config: initialize either UNetConfig, ConvNextConfig, or SwinTConfig
            based on input from backbone_type
        head_configs: (Dict) Dictionary with the following keys having head configs for
            the model to be trained. Note: Configs should be provided only for the model
            to train and others should be None
        total_params: (int) Total number of parameters in the model. This is automatically
            computed when the training starts.
    """

    init_weights: str = "default"
    pretrained_backbone_weights: Optional[str] = None
    pretrained_head_weights: Optional[str] = None
    backbone_config: BackboneConfig = field(factory=BackboneConfig)
    head_configs: HeadConfig = field(factory=HeadConfig)
    total_params: Optional[int] = None


def model_mapper(legacy_config: dict) -> ModelConfig:
    """Map the legacy model configuration to the new model configuration.

    Args:
        legacy_config: A dictionary containing the legacy model configuration.

    Returns:
        An instance of `ModelConfig` with the mapped configuration.
    """
    legacy_config_model = legacy_config.get("model", {})
    backbone_cfg_args = {}
    head_cfg_args = {}
    if legacy_config_model.get("backbone", {}).get("unet", None) is not None:
        backbone_cfg_args["unet"] = UNetConfig(
            filters=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("filters", 32),
            filters_rate=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("filters_rate", 1.5),
            max_stride=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("max_stride", 16),
            stem_stride=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("stem_stride", 16),
            middle_block=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("middle_block", True),
            up_interpolate=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("up_interpolate", True),
            stacks=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("stacks", 1),
            output_stride=legacy_config_model.get("backbone", {})
            .get("unet", {})
            .get("output_stride", 1),
        )

    backbone_cfg = BackboneConfig(**backbone_cfg_args)

    if legacy_config_model.get("heads", {}).get("single_instance", None) is not None:
        head_cfg_args["single_instance"] = SingleInstanceConfig(
            confmaps=SingleInstanceConfMapsConfig(
                part_names=legacy_config_model.get("heads", {})
                .get("single_instance", {})
                .get("part_names", None),
                sigma=legacy_config_model.get("heads", {})
                .get("single_instance", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("single_instance", {})
                .get("output_stride", 1),
            )
        )
    if legacy_config_model.get("heads", {}).get("centroid", None) is not None:
        head_cfg_args["centroid"] = CentroidConfig(
            confmaps=CentroidConfMapsConfig(
                anchor_part=legacy_config_model.get("heads", {})
                .get("centroid", {})
                .get("anchor_part", None),
                sigma=legacy_config_model.get("heads", {})
                .get("centroid", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("centroid", {})
                .get("output_stride", 1),
            )
        )
    if legacy_config_model.get("heads", {}).get("centered_instance", None) is not None:
        head_cfg_args["centered_instance"] = CenteredInstanceConfig(
            confmaps=CenteredInstanceConfMapsConfig(
                anchor_part=legacy_config_model.get("heads", {})
                .get("centered_instance", {})
                .get("anchor_part", None),
                sigma=legacy_config_model.get("heads", {})
                .get("centered_instance", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("centered_instance", {})
                .get("output_stride", 1),
                part_names=legacy_config_model.get("heads", {})
                .get("centered_instance", {})
                .get("part_names", None),
            )
        )
    if legacy_config_model.get("heads", {}).get("multi_instance", None) is not None:
        head_cfg_args["bottomup"] = BottomUpConfig(
            confmaps=BottomUpConfMapsConfig(
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("confmaps", {})
                .get("loss_weight", 1.0),
                sigma=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("confmaps", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("confmaps", {})
                .get("output_stride", 1),
                part_names=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("confmaps", {})
                .get("part_names", None),
            ),
            pafs=PAFConfig(
                edges=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("pafs", {})
                .get("edges", None),
                sigma=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("pafs", {})
                .get("sigma", 15.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("pafs", {})
                .get("output_stride", 1),
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_instance", {})
                .get("pafs", {})
                .get("loss_weight", 1.0),
            ),
        )
    if (
        legacy_config_model.get("heads", {}).get("multi_class_bottomup", None)
        is not None
    ):
        head_cfg_args["multi_class_bottomup"] = BottomUpMultiClassConfig(
            confmaps=BottomUpConfMapsConfig(
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("confmaps", {})
                .get("loss_weight", 1.0),
                sigma=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("confmaps", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("confmaps", {})
                .get("output_stride", 1),
                part_names=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("confmaps", {})
                .get("part_names", None),
            ),
            class_maps=ClassMapConfig(
                sigma=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("class_maps", {})
                .get("sigma", 15.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("class_maps", {})
                .get("output_stride", 1),
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("class_maps", {})
                .get("loss_weight", 1.0),
                classes=legacy_config_model.get("heads", {})
                .get("multi_class_bottomup", {})
                .get("class_maps", {})
                .get("classes", None),
            ),
        )

    if (
        legacy_config_model.get("heads", {}).get("multi_class_topdown", None)
        is not None
    ):
        head_cfg_args["multi_class_topdown"] = TopDownCenteredInstanceMultiClassConfig(
            confmaps=CenteredInstanceConfMapsConfig(
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("confmaps", {})
                .get("loss_weight", 1.0),
                sigma=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("confmaps", {})
                .get("sigma", 5.0),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("confmaps", {})
                .get("output_stride", 1),
                anchor_part=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("confmaps", {})
                .get("anchor_part", None),
                part_names=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("confmaps", {})
                .get("part_names", None),
            ),
            class_vectors=ClassVectorsConfig(
                classes=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("classes", None),
                num_fc_layers=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("num_fc_layers", 2),
                num_fc_units=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("num_fc_units", 1024),
                global_pool=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("global_pool", True),
                output_stride=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("output_stride", 1),
                loss_weight=legacy_config_model.get("heads", {})
                .get("multi_class_topdown", {})
                .get("class_vectors", {})
                .get("loss_weight", 1.0),
            ),
        )

    head_cfg = HeadConfig(**head_cfg_args)

    trained_weights_path = legacy_config_model.get("base_checkpoint", None)

    return ModelConfig(
        backbone_config=backbone_cfg,
        head_configs=head_cfg,
        pretrained_backbone_weights=trained_weights_path,
        pretrained_head_weights=trained_weights_path,
    )
