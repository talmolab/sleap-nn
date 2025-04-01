"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleapnn_data_dir(pytestconfig):
    """Dir path to sleap-nn data."""
    return Path(pytestconfig.rootdir) / "tests/assets"

@pytest.fixture
def training_job_config_path(sleapnn_data_dir):
    """Path to training_job_config file."""
    return Path(sleapnn_data_dir) / "training_config.json"