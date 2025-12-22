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
    SwinTBaseConfig,
    SwinTSmallConfig,
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
    model_mapper,
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
        backbone_config=BackboneConfig(),
        head_configs=HeadConfig(),
    )


def test_default_initialization(default_config):
    """Test default initialization of ModelConfig."""
    assert default_config.init_weights == "default"


def test_invalid_pre_trained_weights(caplog):
    """Test validation failure with an invalid pre_trained_weights."""
    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                convnext=ConvNextConfig(
                    pre_trained_weights="here",
                )
            ),
        )
    assert "Invalid pre-trained" in caplog.text

    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                swint=SwinTConfig(
                    pre_trained_weights="here",
                )
            ),
        )
    assert "Invalid pre-trained" in caplog.text


def test_update_config(default_config):
    """Test updating configuration attributes."""
    config = OmegaConf.structured(
        ModelConfig(
            init_weights="default",
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
    assert "Invalid model_type" in caplog.text

    with pytest.raises(ValueError):
        SwinTSmallConfig(model_type="invalid_model_type")
    assert "Invalid model_type" in caplog.text

    with pytest.raises(ValueError):
        SwinTBaseConfig(model_type="invalid_model_type")
    assert "Invalid model_type" in caplog.text


def test_model_mapper():
    """Test the model_mapper function with a sample legacy configuration."""
    legacy_config = {
        "model": {
            "backbone": {
                "unet": {
                    "filters": 64,
                    "filters_rate": 2.0,
                    "max_stride": 32,
                    "stem_stride": 8,
                    "middle_block": True,
                    "up_interpolate": False,
                    "stacks": 2,
                    "output_stride": 2,
                }
            },
            "heads": {
                "single_instance": {
                    "part_names": ["head", "thorax", "abdomen"],
                    "sigma": 3.0,
                    "output_stride": 2,
                },
            },
        },
        "backbone_type": "unet",
    }

    config = model_mapper(legacy_config)
    # Test backbone config
    assert config.backbone_config.unet is not None
    assert config.backbone_config.unet.filters == 64
    assert config.backbone_config.unet.filters_rate == 2.0
    assert config.backbone_config.unet.max_stride == 32
    assert config.backbone_config.unet.stem_stride == 8
    assert config.backbone_config.unet.middle_block is True
    assert config.backbone_config.unet.up_interpolate is False
    assert config.backbone_config.unet.stacks == 2
    assert config.backbone_config.unet.output_stride == 2

    # Test head configs
    assert config.head_configs.single_instance is not None
    assert config.head_configs.single_instance.confmaps.part_names == [
        "head",
        "thorax",
        "abdomen",
    ]
    assert config.head_configs.single_instance.confmaps.sigma == 3.0
    assert config.head_configs.single_instance.confmaps.output_stride == 2


def test_model_oneof_failure_model_config(caplog):
    """Test validation failure with oneof fields."""
    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                unet=UNetConfig(),
                swint=SwinTConfig(),
            )
        )
    assert "Only one attribute of this class can be set (not None).\n" in caplog.text


def test_model_oneof_failure_head_config(caplog):
    """Test validation failure with oneof fields."""
    with pytest.raises(ValueError):
        HeadConfig(
            single_instance=SingleInstanceConfig(),
            centroid=CentroidConfig(),
        )
    assert "Only one attribute of this class can be set (not None).\n" in caplog.text
