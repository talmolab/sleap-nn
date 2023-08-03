from sleap_nn.data.providers import LabelsReader
import torch
from sleap_nn.data.instance_cropping import make_centered_bboxes, InstanceCropper
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.normalization import Normalizer


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
