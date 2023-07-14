from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
import pytest


def test_providers(minimal_instance):
    """Test sleap dataset

    Args:
        minimal_instance: minimal_instance testing fixture
    """
    l = LabelsReader.from_filename(minimal_instance)
    first_lf = next(iter(l))
    assert len(first_lf) == 2
    assert "minimal_instance" in first_lf.video.filename
    assert first_lf.video.shape == (1, 384, 384, 1)
