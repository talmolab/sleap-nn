from sleap_nn.data.providers import LabelsReader
import torch
from sleap_nn.data.instance_centroids import (
    InstanceCentroidFinder,
    find_points_bbox_midpoint,
)


def test_instance_centroids(minimal_instance):
    """Test InstanceCentroidFinder

    Args:
        minimal_instance: minimal_instance testing fixture
    """
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    sample = next(iter(datapipe))
    centroid = sample["centroid"]
    centroid = centroid.int()
    gt = torch.Tensor([122, 180]).int()
    assert torch.equal(centroid, gt)
