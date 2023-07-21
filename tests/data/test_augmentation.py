"""Module for testing augmentations with Kornia"""

import pytest
from sleap_nn.data.augmentation import KorniaAugmenter
from sleap_nn.data.providers import LabelsReader
import sleap_io as sio
from torch.utils.data import DataLoader
import torch


def test_kornia_augmentation(minimal_instance: sio.Labels):
    """Test the Kornia augmentations."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    org_img = lf.image
    org_img = torch.Tensor(org_img).permute(2, 0, 1)
    org_pts = torch.from_numpy(lf[0].numpy())

    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = KorniaAugmenter(datapipe, rotation=90, probability=1.0, scale=(0.1, 0.3))

    dataloader = DataLoader(datapipe)
    sample = next(iter(dataloader))
    image, instance = sample["image"], sample["instance"]
    img, pts = image[0], instance[0]

    assert torch.is_tensor(img)
    assert torch.is_tensor(pts)
    assert img.shape == org_img.shape
    assert pts.shape == org_pts.shape


if __name__ == "__main__":
    pytest.main([f"{__file__}::test_kornia_augmentation"])
