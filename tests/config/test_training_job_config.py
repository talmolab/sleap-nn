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
from sleap_nn.config.trainer_config import TrainerConfig
from omegaconf import OmegaConf, MissingMandatoryValue
from dataclasses import asdict


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
            backbone_type="unet",
            init_weight="default",
            pre_trained_weights=None,
            backbone_config="unet",
        ),
        "trainer_config": TrainerConfig(),
    }


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
            "backbone_type": sample_config["model_config"].backbone_type,
            "init_weight": sample_config["model_config"].init_weight,
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
        config.model_config.backbone_type == sample_config["model_config"].backbone_type
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
            "backbone_type": sample_config["model_config"].backbone_type,
            "init_weight": sample_config["model_config"].init_weight,
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
        parsed_yaml.model_config.backbone_type.lower()
        == sample_config["model_config"].backbone_type
    )
    assert (
        parsed_yaml.model_config.init_weight
        == sample_config["model_config"].init_weight
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
        backbone_type=sample_config["model_config"].backbone_type,
        init_weight=sample_config["model_config"].init_weight,
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
            loaded_config.model_config.backbone_type.lower()
            == config.model_config.backbone_type.lower()
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
            "backbone_type": sample_config["model_config"].backbone_type,
            "init_weight": sample_config["model_config"].init_weight,
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
