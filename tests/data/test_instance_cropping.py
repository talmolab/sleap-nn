import torch

from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper, make_centered_bboxes
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.providers import LabelsReader


def test_make_centered_bboxes():
    # Test bounding box calculation.
    gt = torch.Tensor(
        [
            [72.9970474243164, 131.07481384277344],
            [171.99703979492188, 131.07481384277344],
            [171.99703979492188, 230.07481384277344],
            [72.9970474243164, 230.07481384277344],
        ]
    )

    centroid = torch.Tensor([122.49704742431640625000, 180.57481384277343750000])
    bbox = make_centered_bboxes(centroid, 100, 100)
    assert torch.equal(gt, bbox)


def test_instance_cropper(minimal_instance):
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCropper(datapipe, (100, 100))
    sample = next(iter(datapipe))

    gt_sample_keys = [
        "image",
        "instances",
        "centroids",
        "instance",
        "instance_bbox",
        "instance_image",
    ]

    # Test shapes.
    assert len(sample.keys()) == len(gt_sample_keys)
    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance"].shape == (2, 2)
    assert sample["instance_image"].shape == (1, 100, 100)
    assert sample["instance_bbox"].shape == (4, 2)

    # Test samples.
    gt = torch.Tensor(
        [
            [19.65515899658203, 71.65116882324219],
            [79.34484100341797, 27.348831176757812],
        ]
    )
    centered_instance = sample["instance"]
    assert torch.equal(centered_instance, gt)
