from sleap_nn.data.providers import LabelsReader
import torch
from sleap_nn.data.instance_cropping import make_centered_bboxes, InstanceCropper
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.confidence_maps import ConfidenceMapGenerator


def test_confmaps(minimal_instance):
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCropper(datapipe, 100, 100)
    datapipe1 = ConfidenceMapGenerator(datapipe, sigma=1.5, output_stride=1)
    sample = next(iter(datapipe1))

    assert sample["confidence_maps"].shape == (100, 100, 2)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor(
        [0.989626109600067138671875]
    )

    datapipe2 = ConfidenceMapGenerator(datapipe, sigma=3.0, output_stride=2)
    sample = next(iter(datapipe2))

    assert sample["confidence_maps"].shape == (50, 50, 2)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor(
        [0.99739634990692138671875]
    )
