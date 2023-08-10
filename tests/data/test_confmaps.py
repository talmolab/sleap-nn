import torch

from sleap_nn.data.confidence_maps import ConfidenceMapGenerator, make_confmaps
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper, make_centered_bboxes
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.utils import make_grid_vectors


def test_confmaps(minimal_instance):
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCropper(datapipe, 100, 100)
    datapipe1 = ConfidenceMapGenerator(datapipe, sigma=1.5, output_stride=1)
    sample = next(iter(datapipe1))

    assert sample["confidence_maps"].shape == (2, 100, 100)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor(
        [0.989626109600067138671875]
    )

    datapipe2 = ConfidenceMapGenerator(datapipe, sigma=3.0, output_stride=2)
    sample = next(iter(datapipe2))

    assert sample["confidence_maps"].shape == (2, 50, 50)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor(
        [0.99739634990692138671875]
    )

    xv, yv = make_grid_vectors(2, 2, 1)
    points = torch.Tensor([[1.0, 1.0], [torch.nan, torch.nan]])
    cms = make_confmaps(points, xv, yv, 2.0)
    gt = torch.Tensor(
        [
            [
                [0.7788, 0.8824],
                [0.8824, 1.0000],
            ],
            [
                [0.0000, 0.0000],
                [0.0000, 0.0000],
            ],
        ]
    )

    torch.testing.assert_close(gt, cms, atol=0.001, rtol=0.0)
