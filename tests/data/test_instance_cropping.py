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

    # test bounding box calculation
    gt = torch.Tensor(
        [
            [72.4970, 130.5748],
            [172.4970, 130.5748],
            [172.4970, 230.5748],
            [72.4970, 230.5748],
        ]
    ).int()
    bbox = make_centered_bboxes(sample["centroids"], 100, 100).int()
    assert torch.equal(gt, bbox)

    # test samples
    gt = torch.Tensor([[92, 202], [152, 158]]).int()
    instance = sample["instances"].int()
    assert torch.equal(instance, gt)
    gt = torch.Tensor([122, 180]).int()
    centroid = sample["centroids"].int()
    assert torch.equal(centroid, gt)
    gt = torch.Tensor([[20, 72], [79, 27]]).int()
    centered_instance = sample["centered_instances"].int()
    assert torch.equal(centered_instance, gt)
