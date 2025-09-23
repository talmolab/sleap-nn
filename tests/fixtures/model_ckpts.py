"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleap_nn_model_ckpts_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets/model_ckpts"


@pytest.fixture
def minimal_instance_centered_instance_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_centered_instance"


@pytest.fixture
def minimal_instance_single_instance_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_single_instance"


@pytest.fixture
def minimal_instance_centroid_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_centroid"


@pytest.fixture
def minimal_instance_bottomup_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for BottomUP model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_bottomup"


@pytest.fixture
def minimal_instance_multi_class_bottomup_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for BottomUp ID model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_multiclass_bottomup"


@pytest.fixture
def minimal_instance_multi_class_topdown_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for topdown ID model."""
    return (
        Path(sleap_nn_model_ckpts_dir) / "minimal_instance_multiclass_centered_instance"
    )


@pytest.fixture
def single_instance_with_metrics_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for topdown ID model."""
    return Path(sleap_nn_model_ckpts_dir) / "single_instance_with_metrics"
