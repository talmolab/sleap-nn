import torch

from sleap_nn.data.confidence_maps import ConfidenceMapGenerator, make_confmaps
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.utils import make_grid_vectors
import pytest


def test_confmaps(minimal_instance):
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCropper(datapipe, (100, 100))
    datapipe1 = ConfidenceMapGenerator(
        datapipe,
        sigma=1.5,
        output_stride=1,
        image_key="instance_image",
        instance_key="instance",
    )
    sample = next(iter(datapipe1))

    # assert sample["confidence_maps"].shape == (1, 2, 100, 100)
    assert sample["confidence_maps"].shape == (2, 100, 100)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor([0.9479378461837769])

    datapipe2 = ConfidenceMapGenerator(
        datapipe,
        sigma=3.0,
        output_stride=2,
        image_key="instance_image",
        instance_key="instance",
    )
    sample = next(iter(datapipe2))

    # assert sample["confidence_maps"].shape == (1, 2, 50, 50)
    assert sample["confidence_maps"].shape == (2, 50, 50)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor([0.9867223501205444])

    xv, yv = make_grid_vectors(2, 2, 1)
    points = torch.Tensor([[1.0, 1.0], [torch.nan, torch.nan]])
    # cms = make_confmaps(points.unsqueeze(0), xv, yv, 2.0)
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

    # torch.testing.assert_close(gt.unsqueeze(0), cms, atol=0.001, rtol=0.0)
    torch.testing.assert_close(gt, cms, atol=0.001, rtol=0.0)


if __name__ == "__main__":
    pytest.main([f"{__file__}::test_confmaps"])
