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
from sleap_nn.config.training_job_config import load_sleap_config
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.trainer_config import TrainerConfig, EarlyStoppingConfig
from sleap_nn.config.data_config import IntensityConfig
from tests.assets.fixtures.datasets import sleapnn_data_dir, training_job_config_path
from omegaconf import OmegaConf, MissingMandatoryValue
from dataclasses import asdict
from loguru import logger
from _pytest.logging import LogCaptureFixture
import json


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


def test_from_yaml(sample_config):
    """Test creating a TrainingJobConfig from a valid YAML string."""
    config_dict = {
        "name": sample_config["name"],
        "description": sample_config["description"],
        "data_config": {
            "train_labels_path": sample_config["data_config"].train_labels_path,
            "val_labels_path": sample_config["data_config"].val_labels_path,
            "provider": sample_config["data_config"].provider,
        },
        "model_config": {
            "init_weights": sample_config["model_config"].init_weights,
        },
        "trainer_config": {
            "early_stopping": {
                "patience": sample_config["trainer_config"].early_stopping.patience,
            },
        },
    }
    yaml_data = OmegaConf.to_yaml(config_dict)
    config = TrainingJobConfig.from_yaml(yaml_data)

    assert config.name == sample_config["name"]
    assert config.description == sample_config["description"]
    assert (
        config.data_config.train_labels_path
        == sample_config["data_config"].train_labels_path
    )
    assert (
        config.data_config.val_labels_path
        == sample_config["data_config"].val_labels_path
    )
    assert (
        config.trainer_config.early_stopping.patience
        == sample_config["trainer_config"].early_stopping.patience
    )


def test_to_yaml(sample_config):
    """Test serializing a TrainingJobConfig to YAML."""
    config_dict = {
        "name": sample_config["name"],
        "description": sample_config["description"],
        "data_config": {
            "train_labels_path": sample_config["data_config"].train_labels_path,
            "val_labels_path": sample_config["data_config"].val_labels_path,
            "provider": sample_config["data_config"].provider,
        },
        "model_config": {
            "init_weights": sample_config["model_config"].init_weights,
        },
        "trainer_config": sample_config[
            "trainer_config"
        ],  # Include full trainer config
    }
    yaml_data = OmegaConf.to_yaml(config_dict)
    parsed_yaml = OmegaConf.create(yaml_data)

    assert parsed_yaml.name == sample_config["name"]
    assert parsed_yaml.description == sample_config["description"]
    assert (
        parsed_yaml.data_config.train_labels_path
        == sample_config["data_config"].train_labels_path
    )
    assert (
        parsed_yaml.data_config.val_labels_path
        == sample_config["data_config"].val_labels_path
    )
    assert parsed_yaml.data_config.provider == sample_config["data_config"].provider

    assert (
        parsed_yaml.model_config.init_weights
        == sample_config["model_config"].init_weights
    )
    assert parsed_yaml.trainer_config == sample_config["trainer_config"]


def test_load_yaml(sample_config):
    """Test loading a TrainingJobConfig from a YAML file."""
    # Create proper config objects
    data_config = DataConfig(
        train_labels_path=sample_config["data_config"].train_labels_path,
        val_labels_path=sample_config["data_config"].val_labels_path,
        provider=sample_config["data_config"].provider,
    )

    model_config = ModelConfig(
        init_weights=sample_config["model_config"].init_weights,
    )

    trainer_config = TrainerConfig(
        early_stopping=sample_config["trainer_config"].early_stopping
    )

    config = TrainingJobConfig(
        name=sample_config["name"],
        description=sample_config["description"],
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_config.yaml")

        # Use the to_yaml method to save the file
        config.to_yaml(filename=file_path)

        # Load from file
        loaded_config = TrainingJobConfig.load_yaml(file_path)
        assert loaded_config.name == config.name
        assert loaded_config.description == config.description
        # Use dictionary access for loaded config
        assert (
            loaded_config.data_config.train_labels_path
            == config.data_config.train_labels_path
        )
        assert (
            loaded_config.data_config.val_labels_path
            == config.data_config.val_labels_path
        )
        assert (
            loaded_config.trainer_config.early_stopping.patience
            == config.trainer_config.early_stopping.patience
        )


def test_missing_attributes(sample_config):
    """Test creating a TrainingJobConfig from a valid YAML string."""
    config_dict = {
        "name": sample_config["name"],
        "description": sample_config["description"],
        "data_config": {
            "provider": sample_config["data_config"].provider,
        },
        "model_config": {
            "init_weights": sample_config["model_config"].init_weights,
        },
        "trainer_config": {
            "early_stopping": {
                "patience": sample_config["trainer_config"].early_stopping.patience,
            },
        },
    }
    yaml_data = OmegaConf.to_yaml(config_dict)

    with pytest.raises(MissingMandatoryValue):
        config = TrainingJobConfig.from_yaml(yaml_data)


def test_load_sleap_config_from_file(training_job_config_path):
    """Test the load_sleap_config function with a sample legacy configuration from a JSON file."""
    # Path to the training_config.json file
    json_file_path = training_job_config_path

    # Load the configuration using the load_sleap_config method
    config = load_sleap_config(TrainingJobConfig, json_file_path)

    # Assertions to check if the output matches expected values
    assert config.data_config.train_labels_path is None  # As per the JSON file
    assert config.data_config.val_labels_path is None  # As per the JSON file
    assert config.model_config.backbone_config.unet.filters == 8
    assert config.model_config.backbone_config.unet.max_stride == 16
    assert config.trainer_config.max_epochs == 200
    assert config.trainer_config.optimizer_name == "Adam"
    assert config.trainer_config.optimizer.lr == 0.0001
    assert config.trainer_config.trainer_devices == "auto"  # Default value
    assert config.trainer_config.trainer_accelerator == "auto"  # Default value
    assert config.trainer_config.enable_progress_bar is True  # Default value
    assert config.trainer_config.train_data_loader.batch_size == 4  # From the JSON file
    assert (
        config.trainer_config.lr_scheduler.reduce_lr_on_plateau is not None
    )  # From the JSON file
    assert (
        config.trainer_config.early_stopping.stop_training_on_plateau is True
    )  # From the JSON file
