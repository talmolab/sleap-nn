import pytest
from sleap_nn.config.model_config import (
    ModelConfig,
    BackboneConfig,
    UNetConfig,
    ConvNextConfig,
    SwinTConfig,
    HeadConfig,
    SingleInstanceConfig,
    CentroidConfig,
    CenteredInstanceConfig,
    BottomUpConfig,
    SingleInstanceConfMapsConfig,
    CentroidConfMapsConfig,
    CenteredInstanceConfMapsConfig,
    BottomUpConfMapsConfig,
    PAFConfig
)

@pytest.fixture
def default_config():
    """Fixture for a default ModelConfig instance."""
    return ModelConfig(
                        init_weight="default", 
                        pre_trained_weights=None,
                        backbone_type="unet",
                        backbone_config=BackboneConfig(),
                        head_configs=HeadConfig()
                        )


def test_default_initialization(default_config):
    """Test default initialization of ModelConfig."""
    assert default_config.init_weight == "default"
    assert default_config.pre_trained_weights == None

def test_invalid_pre_trained_weights():
    """Test validation failure with an invalid pre_trained_weights."""
    with pytest.raises(ValueError):
        ModelConfig(pre_trained_weights="here", backbone_type="unet")

def test_invalid_backbonetype():
    """Test validation failure with an invalid pre_trained_weights."""
    with pytest.raises(ValueError, match='backbone_type must be one of "unet", "convnext", "swint"'):
        ModelConfig(backbone_type="net")

# def test_update_config(default_config):
#     """Test updating configuration attributes."""
#     default_config.model_type = "resnet"
#     default_config.input_size = (256, 256)
#     default_config.num_classes = 20

#     assert default_config.model_type == "resnet"
#     assert default_config.input_size == (256, 256)
#     assert default_config.num_classes == 20

# def test_validation_on_update(default_config):
#     """Test validation logic when updating attributes."""
#     with pytest.raises(ValueError, match="num_classes must be a positive integer"):
#         default_config.num_classes = 0

#     with pytest.raises(ValueError, match="input_size must be a tuple of two positive integers"):
#         default_config.input_size = (-1, 224)

# def test_repr(default_config):
#     """Test the string representation of ModelConfig."""
#     expected_repr = (
#         "ModelConfig(model_type='default', input_size=(224, 224), num_classes=10)"
#     )
#     assert repr(default_config) == expected_repr
