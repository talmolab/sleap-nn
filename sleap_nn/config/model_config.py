"""Serializable configuration classes for specifying all model config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the model config.
"""

from attrs import define, field
from enum import Enum
from sleap_nn.config.utils import oneof
from typing import Optional, List


# Define configuration for each backbone type (unet, convnext, swint) configurations
@define
class UNetConfig:
    """unet config for backbone.

    Attributes:
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters: (int) Base number of filters in the network. Default is 32
        filters_rate: (float) Factor to adjust the number of filters per block. Default is 1.5.
        max_stride: (int) Scalar integer specifying the maximum stride that the image must be divisible by.
        stem_stride: (int) If not None, will create additional "down" blocks for initial downsampling based on the stride. These will be configured identically to the down blocks below.
        middle_block: (bool) If True, add an additional block at the end of the encoder. default: True
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. Default: True.
        stacks: (int) Number of upsampling blocks in the decoder. Default is 3.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
    """

    in_channels: int = 1
    kernel_size: int = 3
    filters: int = 32
    filters_rate: float = 1.5
    max_stride: Optional[int] = None
    stem_stride: Optional[int] = None
    middle_block: bool = True
    up_interpolate: bool = True
    stacks: int = 3
    convs_per_block: int = 2


@define
class ConvNextConfig:
    """convnext configuration for backbone.

    Attributes:
        arch: (Default is Tiny architecture config. No need to provide if model_type is provided)
            depths: (List(int)) Number of layers in each block. Default: [3, 3, 9, 3].
            channels: (List(int)) Number of channels in each block. Default: [96, 192, 384, 768].
        model_type: (str) One of the ConvNext architecture types: ["tiny", "small", "base", "large"]. Default: "tiny".
        stem_patch_kernel: (int) Size of the convolutional kernels in the stem layer. Default is 4.
        stem_patch_stride: (int) Convolutional stride in the stem layer. Default is 2.
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters_rate: (float) Factor to adjust the number of filters per block. Default is 1.5.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. Default: True.
    """

    model_type: str = "tiny"  # Options: tiny, small, base, large
    arch: dict = field(
        factory=lambda: {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]}
    )
    stem_patch_kernel: int = 4
    stem_patch_stride: int = 2
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 1.5
    convs_per_block: int = 2
    up_interpolate: bool = True


@define
class SwinTConfig:
    """swinT configuration for backbone.

    Attributes:
        model_type: (str) One of the SwinT architecture types: ["tiny", "small", "base"]. Default: "tiny".
        arch: Dictionary of embed dimension, depths and number of heads in each layer. Default is "Tiny architecture". {'embed': 96, 'depths': [2,2,6,2], 'channels':[3, 6, 12, 24]}
        patch_size: (List[int]) Patch size for the stem layer of SwinT. Default: [4,4].
        stem_patch_stride: (int) Stride for the patch. Default is 2.
        window_size: (List[int]) Window size. Default: [7,7].
        in_channels: (int) Number of input channels. Default is 1.
        kernel_size: (int) Size of the convolutional kernels. Default is 3.
        filters_rate: (float) Factor to adjust the number of filters per block. Default is 1.5.
        convs_per_block: (int) Number of convolutional layers per block. Default is 2.
        up_interpolate: (bool) If True, use bilinear interpolation instead of transposed convolutions for upsampling. Interpolation is faster but transposed convolutions may be able to learn richer or more complex upsampling to recover details from higher scales. Default: True.
    """

    model_type: str = "tiny"  # Options: tiny, small, base
    arch: dict = field(
        factory=lambda: {
            "embed": 96,
            "depths": [2, 2, 6, 2],
            "channels": [3, 6, 12, 24],
        }
    )
    patch_size: list = field(factory=lambda: [4, 4])
    stem_patch_stride: int = 2
    window_size: list = field(factory=lambda: [7, 7])
    in_channels: int = 1
    kernel_size: int = 3
    filters_rate: float = 1.5
    convs_per_block: int = 2
    up_interpolate: bool = True


@define
class SingleInstanceConfMapsConfig:
    """Single Instance configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    """

    part_names: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None


@define
class CentroidConfMapsConfig:
    """Centroid configuration map.

    Attributes:
        anchor_part: (int) Note: Only for 'CenteredInstanceConfmapsHead'. Index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    """

    anchor_part: Optional[int] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None


@define
class CenteredInstanceConfMapsConfig:
    """Centered Instance configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        anchor_part: (int) Note: Only for 'CenteredInstanceConfmapsHead'. Index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    """

    part_names: Optional[List[str]] = None
    anchor_part: Optional[int] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None


@define
class BottomUpConfMapsConfig:
    """Bottomup configuration map.

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
    """

    part_names: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None
    loss_weight: Optional[float] = None


@define
class PAFConfig:
    """PAF configuration map.

    Attributes:
        edges: (List[str]) None if edges from sio.Labels file can be used directly. Note: Only for 'PartAffinityFieldsHead'. List of indices (src, dest) that form an edge.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
    """

    edges: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None
    loss_weight: Optional[float] = None


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


@oneof
@define
class HeadConfig:
    """Configurations related to the model output head type.

    Only one attribute of this class can be set, which defines the model output type.

    Attributes:
        single_instance: An instance of `SingleInstanceConfmapsHeadConfig`.
        centroid: An instance of `CentroidsHeadConfig`.
        centered_instance: An instance of `CenteredInstanceConfmapsHeadConfig`.
        multi_instance: An instance of `MultiInstanceConfig`.
        multi_class_bottomup: An instance of `MultiClassBottomUpConfig`.
        multi_class_topdown: An instance of `MultiClassTopDownConfig`.
    """

    single_instance: Optional[SingleInstanceConfig] = None
    centroid: Optional[CentroidConfig] = None
    centered_instance: Optional[CenteredInstanceConfig] = None
    bottomup: Optional[BottomUpConfig] = None


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
        backbone_type: (str) Backbone architecture for the model to be trained. One of "unet", "convnext" or "swint".
        init_weight: (str) model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method.
        pre_trained_weights: (str) Pretrained weights file name supported only for ConvNext and SwinT backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        backbone_config: initialize either UNetConfig, ConvNextConfig, or SwinTConfig based on input from backbone_type
        head_configs: (Dict) Dictionary with the following keys having head configs for the model to be trained. Note: Configs should be provided only for the model to train and others should be None
    """

    backbone_type: str = field(
        default="unet",
        validator=lambda instance, attr, value: instance.validate_backbone_type(value),
    )
    init_weight: str = "default"
    pre_trained_weights: Optional[str] = field(
        default=None,
        validator=lambda instance, attr, value: instance.validate_pre_trained_weights(
            value
        ),
    )
    backbone_config: BackboneConfig = field(factory=BackboneConfig)
    head_configs: HeadConfig = field(factory=HeadConfig)

    def validate_backbone_type(self, value):
        """Validate backbone_type.

        Ensure backbone_type is one of "unet", "convnext", or "swint".
        """
        valid_types = ["unet", "convnext", "swint"]
        if value not in valid_types:
            raise ValueError(f"Invalid backbone_type. Must be one of {valid_types}")

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
        swint_weights are one of
        (
            "Swin_T_Weights",
            "Swin_S_Weights",
            "Swin_B_Weights"
        )
        unet weights is None
        """
        if value is None:
            return

        convnext_weights = [
            "ConvNeXt_Base_Weights",
            "ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights",
            "ConvNeXt_Large_Weights",
        ]
        swint_weights = ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]

        if self.backbone_type == "convnext":
            if value not in convnext_weights:
                raise ValueError(
                    f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}"
                )
        elif self.backbone_type == "swint":
            if value not in swint_weights:
                raise ValueError(
                    f"Invalid pre-trained weights for SwinT. Must be one of {swint_weights}"
                )
        elif self.backbone_type == "unet":
            raise ValueError("UNet does not support pre-trained weights.")
