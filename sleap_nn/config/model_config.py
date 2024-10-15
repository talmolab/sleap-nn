@attr.s(auto_attribs=True)
class ModelConfig:
    """Configurations related to model architecture.

    Attributes:
        init_weight: (str) model weights initialization method. "default" uses kaiming uniform initialization and "xavier" uses Xavier initialization method.
        pre_trained_weights: (str) Pretrained weights file name supported only for ConvNext and SwinT backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights", "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. For SwinT, one of ["Swin_T_Weights", "Swin_S_Weights", "Swin_B_Weights"].
        backbone_type: (str) Backbone architecture for the model to be trained. One of "unet", "convnext" or "swint".

    """

    init_weight: str = "default"
    pre_trained_weights: str = 
    backbone_type: BackboneType = BackboneType.UNET
    backbone_config: 

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
