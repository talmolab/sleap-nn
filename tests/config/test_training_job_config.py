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
from sleap_nn.config.model_config import ModelConfig, BackboneType
from sleap_nn.config.data_config import DataConfig
from sleap_nn.config.trainer_config import TrainerConfig
from omegaconf import OmegaConf


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "name": "TestConfig",
        "description": "A sample configuration for testing.",
        "data": DataConfig(
            train_labels_path="example_train_path", val_labels_path="example_val_path"
        ),
        "model": ModelConfig(backbone_type=BackboneType.UNET),
        "trainer": TrainerConfig(),
    }


def test_from_yaml(sample_config):
    """Test creating a TrainingJobConfig from a YAML string."""
    yaml_data = OmegaConf.to_yaml(sample_config)
    config = TrainingJobConfig.from_yaml(yaml_data)

    assert config.name == sample_config["name"]
    assert config.description == sample_config["description"]
    assert isinstance(config.data, dict)  # Updated to check for dict
    assert config.data["provider"] == sample_config["data"].provider
    assert config.data["train_labels_path"] == sample_config["data"].train_labels_path
    assert config.data["val_labels_path"] == sample_config["data"].val_labels_path


def test_to_yaml(sample_config):
    """Test serializing a TrainingJobConfig to YAML."""
    config = TrainingJobConfig(**sample_config)
    yaml_data = config.to_yaml()
    parsed_yaml = OmegaConf.create(yaml_data)

    assert parsed_yaml.name == sample_config["name"]
    assert parsed_yaml.description == sample_config["description"]
    assert parsed_yaml.data == sample_config["data"]
    assert (
        parsed_yaml.model.backbone_type.lower()
        == sample_config["model"].backbone_type.value
    )
    assert parsed_yaml.model.init_weight == sample_config["model"].init_weight
    assert parsed_yaml.trainer == sample_config["trainer"]


def test_save_and_load_yaml(sample_config):
    """Test saving and loading a TrainingJobConfig as a YAML file."""
    config = TrainingJobConfig(**sample_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_config.yaml")

        # Save to file
        config.save_yaml(file_path)
        assert os.path.exists(file_path)

        # Load from file
        loaded_config = TrainingJobConfig.load_yaml(file_path)
        assert loaded_config.name == config.name
        assert loaded_config.description == config.description
        assert (
            loaded_config.data["augmentation_config"] == config.data.augmentation_config
        )
        assert (
            loaded_config.model["backbone_type"].lower()
            == config.model.backbone_type.value
        )
        assert (
            loaded_config.trainer["early_stopping"]["patience"]
            == config.trainer.early_stopping.patience
        )
