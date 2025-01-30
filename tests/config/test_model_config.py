"""Tests for the serializable configuration classes for specifying all model config parameters.

These configuration classes are intended to specify all 
the parameters required to initialize the model config.
"""

import pytest
from omegaconf import OmegaConf
from sleap_nn.config.model_config import (
    ModelConfig,
    BackboneConfig,
    BackboneType,
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
    PAFConfig,
)


@pytest.fixture
def default_config():
    """Fixture for a default ModelConfig instance."""
    return ModelConfig(
        init_weight="default",
        pre_trained_weights=None,
        backbone_type="unet",
        backbone_config=BackboneConfig(),
        head_configs=HeadConfig(),
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
    with pytest.raises(AttributeError):
        ModelConfig(backbone_type=BackboneType.NET)


def test_update_config(default_config):
    """Test updating configuration attributes."""
    config = OmegaConf.structured(
        ModelConfig(
            backbone_type=BackboneType.UNET,
            init_weight="default",
            pre_trained_weights=None,
            backbone_config=BackboneConfig(),
            head_configs=HeadConfig(),
        )
    )

    with pytest.raises(AttributeError):
        config.backbone_type = BackboneType.NET
