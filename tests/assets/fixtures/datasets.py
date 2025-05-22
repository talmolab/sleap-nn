"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleapnn_data_dir(pytestconfig):
    """Dir path to sleap-nn data."""
    return Path(pytestconfig.rootdir) / "tests/assets/sleap_configs"


@pytest.fixture
def bottomup_multiclass_training_config_path(sleapnn_data_dir):
    """Path to bottomup_multiclass_training_config file."""
    return Path(sleapnn_data_dir) / "bottomup_multiclass_training_config.json"


@pytest.fixture
def bottomup_training_config_path(sleapnn_data_dir):
    """Path to bottomup_training_config file."""
    return Path(sleapnn_data_dir) / "bottomup_training_config.json"


@pytest.fixture
def centered_instance_training_config_path(sleapnn_data_dir):
    """Path to centered_instance_training_config file."""
    return Path(sleapnn_data_dir) / "centered_instance_training_config.json"


@pytest.fixture
def centered_instance_with_scaling_training_config_path(sleapnn_data_dir):
    """Path to centered_instance_training_config file."""
    return (
        Path(sleapnn_data_dir) / "centered_instance_with_scaling_training_config.json"
    )


@pytest.fixture
def centroid_training_config_path(sleapnn_data_dir):
    """Path to centroid_training_config file."""
    return Path(sleapnn_data_dir) / "centroid_training_config.json"


@pytest.fixture
def single_instance_training_config_path(sleapnn_data_dir):
    """Path to single_instance_training_config file."""
    return Path(sleapnn_data_dir) / "single_instance_training_config.json"


@pytest.fixture
def topdown_training_config_path(sleapnn_data_dir):
    """Path to topdown_training_config file."""
    return Path(sleapnn_data_dir) / "topdown_training_config.json"
