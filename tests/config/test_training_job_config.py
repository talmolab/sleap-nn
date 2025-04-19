"""Tests for the Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
        human, as well as easily be serialized/deserialized to/from simple dictionaries
        and YAML.

    2. Decoupling from the implementation. This makes it easier to design functional
        modules with attributes/parameters that contain objects that may not be easily
        serializable or may implement additional logic that relies on runtime
        information or other parameters.

In general, classes that implement the actual functionality related to these
configuration classes should provide a classmethod for instantiation from the
configuration class instances. This makes it easier to implement other logic not related
to the high level parameters at creation time.

Conveniently, this format also provides a single location where all user-facing
parameters are aggregated and documented for end users (as opposed to developers).
"""

import pytest
import os
import tempfile
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.trainer_config import TrainerConfig, EarlyStoppingConfig
from sleap_nn.config.data_config import IntensityConfig
from omegaconf import DictConfig, OmegaConf, MissingMandatoryValue
from dataclasses import asdict
from loguru import logger
from _pytest.logging import LogCaptureFixture


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
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "name": "TestConfig",
        "description": "A sample configuration for testing.",
        "data_config": DataConfig(
            train_labels_path="example_train_path",
            val_labels_path="example_val_path",
            provider="default",
        ),
        "model_config": ModelConfig(
            init_weights="default",
            pre_trained_weights=None,
            backbone_config="unet",
        ),
        "trainer_config": TrainerConfig(early_stopping=EarlyStoppingConfig()),
    }


def test_intensity_config_validation_logging(caplog):
    """Test IntensityConfig validation and logging."""
    with pytest.raises(ValueError):
        IntensityConfig(gaussian_noise_p=-0.1)  # This should trigger a validation error

    assert "gaussian_noise_p" in caplog.text


def test_to_sleap_nn_cfg():
    """Test serializing a TrainingJobConfig to YAML."""
    cfg = TrainingJobConfig()
    cfg.data_config.train_labels_path = "test.slp"
    cfg.data_config.val_labels_path = "test.slp"
    omegacfg = cfg.to_sleap_nn_cfg()
    assert isinstance(omegacfg, DictConfig)
    assert omegacfg.data_config.train_labels_path == "test.slp"

    with pytest.raises(MissingMandatoryValue):
        config = TrainingJobConfig().to_sleap_nn_cfg()
