from sleap_nn.data.providers import LabelsReader
import torch
from sleap_nn.data.instance_cropping import (
    make_centered_bboxes,
    normalize_bboxes,
    InstanceCropper,
)
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.normalization import Normalizer


def test_normalize_bboxes(minimal_instance):
    bboxes = torch.Tensor(
        [
            [72.4970, 130.5748, 172.4970, 230.5748],
            [3.0000, 5.5748, 100.0000, 220.1235],
        ]
    )

    norm_bboxes = normalize_bboxes(bboxes, image_height=200, image_width=300)

    gt = torch.Tensor(
        [
            [
                0.3643065392971039,
                0.4367050230503082,
                0.8668190836906433,
                0.7711531519889832,
            ],
            [
                0.015075377188622952,
                0.01864481531083584,
                0.5025125741958618,
                0.7361990213394165,
            ],
        ]
    )

    assert torch.equal(norm_bboxes, gt)


def test_instance_cropper(minimal_instance):
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCropper(datapipe, 100, 100)
    sample = next(iter(datapipe))

    # test shapes
    assert sample["instance"].shape == (2, 2)
    assert sample["instance_image"].shape == (1, 1, 100, 100)
    assert sample["bbox"].shape == (1, 4, 2)
    # test bounding box calculation
    gt = torch.Tensor(
        [
            [72.49704742431640625, 130.57481384277343750],
            [172.49703979492187500, 130.57481384277343750],
            [172.49703979492187500, 230.57481384277343750],
            [72.49704742431640625, 230.57481384277343750],
        ]
    )

    centroid = torch.Tensor([122.49704742431640625000, 180.57481384277343750000])
    bbox = make_centered_bboxes(centroid, 100, 100)
    assert torch.equal(gt, bbox)

    # test samples
    gt = torch.Tensor(
        [
            [20.15515899658203125, 72.15116882324218750],
            [79.84484100341796875, 27.84883117675781250],
        ]
    )
    centered_instance = sample["instance"]
    assert torch.equal(centered_instance, gt)
