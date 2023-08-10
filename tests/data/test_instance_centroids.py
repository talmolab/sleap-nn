import torch

from sleap_nn.data.instance_centroids import (
    InstanceCentroidFinder,
    find_centroids,
)
from sleap_nn.data.providers import LabelsReader


def test_instance_centroids(minimal_instance):
    # Undefined anchor_ind
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    sample = next(iter(datapipe))
    instances = sample["instances"]
    centroids = sample["centroids"]
    centroids = centroids.int()
    gt = torch.Tensor([[[122, 180], [242, 195]]]).int()
    assert torch.equal(centroids, gt)

    # Defined anchor_ind
    centroids = find_centroids(instances, 1).int()
    gt = torch.Tensor([[[152, 158], [278, 203]]])
    assert torch.equal(centroids, gt)

    # Defined anchor_ind, but missing one
    partial_instance = torch.Tensor(
        [
            [
                [[92.6522, 202.7260], [152.3419, 158.4236], [97.2618, 53.5834]],
                [[205.9301, 187.8896], [torch.nan, torch.nan], [201.4264, 75.2373]],
            ]
        ]
    )
    centroids = find_centroids(partial_instance, 1).int()
    gt = torch.Tensor([[[152, 158], [203, 131]]])
    assert torch.equal(centroids, gt)
