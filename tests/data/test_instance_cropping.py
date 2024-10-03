import sleap_io as sio
import torch

from sleap_nn.data.instance_centroids import InstanceCentroidFinder, generate_centroids
from sleap_nn.data.instance_cropping import (
    InstanceCropper,
    find_instance_crop_size,
    generate_crops,
    make_centered_bboxes,
)
from sleap_nn.data.normalization import Normalizer, apply_normalization
from sleap_nn.data.resizing import SizeMatcher, Resizer, PadToStride
from sleap_nn.data.providers import LabelsReader, process_lf


def test_find_instance_crop_size(minimal_instance):
    """Test `find_instance_crop_size` function."""
    labels = sio.load_slp(minimal_instance)
    crop_size = find_instance_crop_size(labels)
    assert crop_size == 74

    crop_size = find_instance_crop_size(labels, min_crop_size=100)
    assert crop_size == 100

    crop_size = find_instance_crop_size(labels, padding=10)
    assert crop_size == 84


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
    """Test InstanceCropper module."""
    provider = LabelsReader.from_filename(minimal_instance)
    provider.max_instances = 3
    datapipe = Normalizer(provider)
    datapipe = SizeMatcher(datapipe, provider)
    datapipe = Resizer(datapipe, scale=1.0)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = InstanceCropper(datapipe, (100, 100))
    sample = next(iter(datapipe))

    gt_sample_keys = [
        "centroid",
        "instance",
        "instance_bbox",
        "instance_image",
        "video_idx",
        "frame_idx",
        "num_instances",
        "orig_size",
    ]

    # Test shapes.
    assert len(list(iter(datapipe))) == 2
    assert len(sample.keys()) == len(gt_sample_keys)
    for gt_key, key in zip(sorted(gt_sample_keys), sorted(sample.keys())):
        assert gt_key == key
    assert sample["instance"].shape == (1, 2, 2)
    assert sample["centroid"].shape == (1, 2)
    assert sample["instance_image"].shape == (1, 1, 100, 100)
    assert sample["instance_bbox"].shape == (1, 4, 2)
    assert sample["num_instances"] == 2

    # Test samples.
    gt = torch.Tensor(
        [
            [19.65515899658203, 71.65116882324219],
            [79.34484100341797, 27.348831176757812],
        ]
    )
    centered_instance = sample["instance"]
    assert torch.equal(centered_instance, gt.unsqueeze(0))


def test_generate_crops(minimal_instance):
    """Test `generate_crops` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)
    ex["image"] = apply_normalization(ex["image"])

    centroids = generate_centroids(ex["instances"], 0)
    cropped_ex = generate_crops(
        ex["image"], ex["instances"][0, 0], centroids[0, 0], crop_size=(100, 100)
    )

    assert cropped_ex["instance"].shape == (1, 2, 2)
    assert cropped_ex["centroid"].shape == (1, 2)
    assert cropped_ex["instance_image"].shape == (1, 1, 100, 100)
    assert cropped_ex["instance_bbox"].shape == (1, 4, 2)
