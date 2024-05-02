import torch

from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.resizing import SizeMatcher
import sleap_io as sio
import numpy as np
import sleap_io as sio
import pytest


def test_resizer(minimal_instance):
    """Test SizeMatcher module."""
    l = LabelsReader.from_filename(minimal_instance)
    pipe = SizeMatcher(l, max_height=500, max_width=400)
    sample = next(iter(pipe))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 500, 400])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    l = LabelsReader.from_filename(minimal_instance)
    pipe = SizeMatcher(l, max_height=100, max_width=500)
    with pytest.raises(
        Exception,
        match=f"Max height {100} should be greater than the current image height: {384}",
    ):
        sample = next(iter(pipe))

    l = LabelsReader.from_filename(minimal_instance)
    pipe = SizeMatcher(l, max_height=500, max_width=100)
    with pytest.raises(
        Exception,
        match=f"Max width {100} should be greater than the current image width: {384}",
    ):
        sample = next(iter(pipe))
