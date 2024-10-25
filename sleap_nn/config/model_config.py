import attrs
from enum import Enum

@attrs.define
class ModelConfig:
    """Configurations related to model architecture.

    Attributes:
        init_weight: (str) model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method.
        pre_trained_weights: (str) Pretrained weights file name supported only for ConvNext and SwinT backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        backbone_type: (str) Backbone architecture for the model to be trained. One of "unet", "convnext" or "swint".

    """

    init_weight: str = "default"
    pre_trained_weights: str = None
    backbone_type: BackboneType = BackboneType.UNET
    backbone_config: Union[UNetConfig, ConvNextConfig, SwinTConfig] = attrs.field(init=False)   # backbone_config can be any of these 3 configurations. init=False lets you set the parameters later (not in initialization)
    head_configs: HeadConfig = attrs.field(factory=HeadConfig)

    # post-initialization
    def __attrs_post_init__(self):
        self.backbone_config = self.set_backbone_config()
        self.validate_pre_trained_weights()

    # configures back_bone config to one of these types
    def set_backbone_config(self):
        if self.backbone_type == BackboneType.UNET:
            return UNetConfig()
        elif self.backbone_type == BackboneType.CONVNEXT:
            return ConvNextConfig()
        elif self.backbone_type == BackboneType.SWINT:
            return SwinTConfig()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def validate_pre_trained_weights(self):
        convnext_weights = ["ConvNeXt_Base_Weights", "ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]
        swint_weights = ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"]

        if self.backbone_type == BackboneType.CONVNEXT:
            if self.pre_trained_weights not in convnext_weights:
                raise ValueError(f"Invalid pre-trained weights for ConvNext. Must be one of {convnext_weights}")
        elif self.backbone_type == BackboneType.SWINT:
            if self.pre_trained_weights not in swint_weights:
                raise ValueError(f"Invalid pre-trained weights for SwinT. Must be one of {swint_weights}")
        elif self.backbone_type == BackboneType.UNET and self.pre_trained_weights is not None:
            raise ValueError("UNet does not support pre-trained weights.")

    class BackboneType(Enum):
        UNET = "unet"
        CONVNEXT = 'convnext'
        SWINT = 'swint'
    
    # Define configuration for each backbone type
    @attrs.define
    class UNetConfig:
        in_channels: int = 1
        kernel_size: int = 3
        filters: int = 32
        filters_rate: float = 1.5
        max_stride: int = None
        stem_stride: int = None
        middle_block: bool = True
        up_interpolate: bool = True
        stacks: int = 3
        convs_per_block: int = 2

    @attrs.define
    class ConvNextConfig:
        model_type: str = "tiny"  # Options: tiny, small, base, large
        arch: dict = attrs.field(factory=lambda: {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]})
        stem_patch_kernel: int = 4
        stem_patch_stride: int = 2
        in_channels: int = 1
        kernel_size: int = 3
        filters_rate: float = 1.5
        convs_per_block: int = 2
        up_interpolate: bool = True

    @attrs.define
    class SwinTConfig:
        model_type: str = "tiny"  # Options: tiny, small, base
        arch: dict = attrs.field(factory=lambda: {'embed': 96, 'depths': [2, 2, 6, 2], 'channels': [3, 6, 12, 24]})
        patch_size: list = attrs.field(factory=lambda: [4, 4])
        stem_patch_stride: int = 2
        window_size: list = attrs.field(factory=lambda: [7, 7])
        in_channels: int = 1
        kernel_size: int = 3
        filters_rate: float = 1.5
        convs_per_block: int = 2
        up_interpolate: bool = True

@oneof
@attr.s(auto_attribs=True)
class HeadsConfig:
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

# Head_config single instance
@attrs.define
class SingleInstanceConfig:
    confmaps: Optional[SingleInstanceConfMapsConfig] = None

# Head_config centroid
@attrs.define
class CentroidConfig:
    confmaps: Optional[CentroidConfMapsConfig] = None

# Head_config centered_instance
@attrs.define
class CenteredInstanceConfig:
    confmaps: Optional[CenteredInstanceConfMapsConfig] = None

# Head_config bottomup
@attrs.define
class BottomUpConfig:
    confmaps: Optional[BottomUpConfMapsConfig] = None
    pafs: Optional[PAFConfig] = None

@attrs.define
class SingleInstanceConfMapsConfig:
    '''

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    '''
    part_names: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None

@attrs.define
class CentroidConfMapsConfig:
    '''

    Attributes:
        anchor_part: (int) Note: Only for 'CenteredInstanceConfmapsHead'. Index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    '''
    anchor_part: Optional[int] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None

@attrs.define
class CenteredInstanceConfMapsConfig:
    '''

    Attributes:  
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        anchor_part: (int) Note: Only for 'CenteredInstanceConfmapsHead'. Index of the anchor node to use as the anchor point. If None, the midpoint of the bounding box of all visible instance points will be used as the anchor. The bounding box midpoint will also be used if the anchor part is specified but not visible in the instance. Setting a reliable anchor point can significantly improve topdown model accuracy as they benefit from a consistent geometry of the body parts relative to the center of the image.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
    '''
    part_names: Optional[List[str]] = None
    anchor_part: Optional[int] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None

@attrs.define
class BottomUpConfMapsConfig():
    '''

    Attributes:
        part_names: (List[str]) None if nodes from sio.Labels file can be used directly. Else provide text name of the body parts (nodes) that the head will be configured to produce. The number of parts determines the number of channels in the output. If not specified, all body parts in the skeleton will be used. This config does not apply for 'PartAffinityFieldsHead'.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
    '''
    part_names: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None
    loss_weight: Optional[float] = None

@attrs.define
class PAFConfig():
    '''

    Attributes:
        edges: (List[str]) None if edges from sio.Labels file can be used directly. Note: Only for 'PartAffinityFieldsHead'. List of indices (src, dest) that form an edge.
        sigma: (float) Spread of the Gaussian distribution of the confidence maps as a scalar float. Smaller values are more precise but may be difficult to learn as they have a lower density within the image space. Larger values are easier to learn but are less precise with respect to the peak coordinate. This spread is in units of pixels of the model input image, i.e., the image resolution after any input scaling is applied.
        output_stride: (float) The stride of the output confidence maps relative to the input image. This is the reciprocal of the resolution, e.g., an output stride of 2 results in confidence maps that are 0.5x the size of the input. Increasing this value can considerably speed up model performance and decrease memory requirements, at the cost of decreased spatial resolution.
        loss_weight: (float) Scalar float used to weigh the loss term for this head during training. Increase this to encourage the optimization to focus on improving this specific output in multi-head models.
    '''
    edges: Optional[List[str]] = None
    sigma: Optional[float] = None
    output_stride: Optional[float] = None
    loss_weight: Optional[float] = None
    