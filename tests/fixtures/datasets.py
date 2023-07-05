import pytest
from pathlib import Path


@pytest.fixture
def sleap_data_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/data"


@pytest.fixture
def minimal_instance(sleap_data_dir):
    """Sleap single fly .slp and video file paths."""
    return Path(sleap_data_dir) / "minimal_instance.pkg.slp"
