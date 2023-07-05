from sleap_nn.data.sleap_dataset import SleapDataset
import sleap_io as sio
import pytest


def test_sleap_dataset(minimal_instance):
    """Test sleap dataset

    Args:
        minimal_instance: minimal_instance testing fixture
    """
    l = sio.load_slp(minimal_instance)
    dataset = SleapDataset(l)
    first_lf = dataset[0]
    assert len(dataset) == 2
    assert "minimal_instance.pkg.slp" in first_lf[0].video.filename
    assert first_lf[2].shape == (384, 384, 1)
