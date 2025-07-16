"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleap_nn_legacy_json_configs_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets/legacy_sleap_json_configs"


@pytest.fixture
def bottomup_multiclass_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to bottomup_multiclass_training_config file."""
    return (
        Path(sleap_nn_legacy_json_configs_dir)
        / "bottomup_multiclass_training_config.json"
    )


@pytest.fixture
def bottomup_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to bottomup_training_config file."""
    return Path(sleap_nn_legacy_json_configs_dir) / "bottomup_training_config.json"


@pytest.fixture
def centered_instance_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to centered_instance_training_config file."""
    return (
        Path(sleap_nn_legacy_json_configs_dir)
        / "centered_instance_training_config.json"
    )


@pytest.fixture
def centered_instance_with_scaling_training_config_path(
    sleap_nn_legacy_json_configs_dir,
):
    """Path to centered_instance_training_config file."""
    return (
        Path(sleap_nn_legacy_json_configs_dir)
        / "centered_instance_with_scaling_training_config.json"
    )


@pytest.fixture
def centroid_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to centroid_training_config file."""
    return Path(sleap_nn_legacy_json_configs_dir) / "centroid_training_config.json"


@pytest.fixture
def single_instance_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to single_instance_training_config file."""
    return (
        Path(sleap_nn_legacy_json_configs_dir) / "single_instance_training_config.json"
    )


@pytest.fixture
def topdown_training_config_path(sleap_nn_legacy_json_configs_dir):
    """Path to topdown_training_config file."""
    return Path(sleap_nn_legacy_json_configs_dir) / "topdown_training_config.json"
