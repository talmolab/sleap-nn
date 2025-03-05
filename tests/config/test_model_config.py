"""Tests for the serializable configuration classes for specifying all model config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the model config.
"""

import pytest
from omegaconf import OmegaConf
from loguru import logger

from _pytest.logging import LogCaptureFixture

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
    PAFConfig,
)


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def default_config():
    """Fixture for a default ModelConfig instance."""
    return ModelConfig(
        init_weights="default",
        pre_trained_weights=None,
        backbone_config=BackboneConfig(),
        head_configs=HeadConfig(),
    )


def test_default_initialization(default_config):
    """Test default initialization of ModelConfig."""
    assert default_config.init_weights == "default"
    assert default_config.pre_trained_weights == None


def test_invalid_pre_trained_weights(caplog):
    """Test validation failure with an invalid pre_trained_weights."""
    with pytest.raises(ValueError):
        ModelConfig(pre_trained_weights="here", backbone_config=BackboneConfig(unet=UNetConfig())
    assert "UNet" in caplog.text


def test_update_config(default_config):
    """Test updating configuration attributes."""
    config = OmegaConf.structured(
        ModelConfig(
            init_weights="default",
            pre_trained_weights=None,
            backbone_config=BackboneConfig(unet=UNetConfig()),
            head_configs=HeadConfig(),
        )
    )


def test_valid_model_type():
    """Test valid model_type values."""
    valid_types = ["tiny", "small", "base"]
    for model_type in valid_types:
        config = SwinTConfig(model_type=model_type)


def test_invalid_model_type(caplog):
    """Test validation failure with an invalid model_type."""
    with pytest.raises(ValueError):
        SwinTConfig(model_type="invalid_model_type")
    assert "Invalid model_type." in caplog.text
