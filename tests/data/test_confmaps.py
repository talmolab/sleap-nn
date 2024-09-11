import torch
import sleap_io as sio
from sleap_nn.data.confidence_maps import (
    ConfidenceMapGenerator,
    MultiConfidenceMapGenerator,
    make_multi_confmaps,
    make_confmaps,
    generate_confmaps,
    generate_multiconfmaps,
)
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import Resizer
from sleap_nn.data.providers import LabelsReader, process_lf
from sleap_nn.data.utils import make_grid_vectors
import numpy as np


def test_confmaps(minimal_instance):
    """Test ConfidenceMapGenerator module."""
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

    assert sample["confidence_maps"].shape == (1, 2, 100, 100)
    assert torch.max(sample["confidence_maps"]) == torch.Tensor([0.9479378461837769])

    datapipe2 = ConfidenceMapGenerator(
        datapipe,
        sigma=3.0,
        output_stride=2,
        image_key="instance_image",
        instance_key="instance",
    )
    sample = next(iter(datapipe2))

    assert sample["confidence_maps"].shape == (1, 2, 50, 50)
    max_confmap = torch.max(sample["confidence_maps"])
    torch.testing.assert_close(
        max_confmap, torch.FloatTensor([0.9967])[0], atol=1e-4, rtol=0.0
    )

    xv, yv = make_grid_vectors(2, 2, 1)
    points = torch.Tensor([[1.0, 1.0], [torch.nan, torch.nan]])
    cms = make_confmaps(points.unsqueeze(0), xv, yv, 2.0)
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

    torch.testing.assert_close(gt.unsqueeze(0), cms, atol=0.001, rtol=0.0)


def test_multi_confmaps(minimal_instance):
    """Test MultiConfidenceMapGenerator module."""
    # centroids = True
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe1 = MultiConfidenceMapGenerator(
        datapipe,
        sigma=1.5,
        output_stride=1,
        centroids=True,
        image_key="image",
        instance_key="instances",
    )
    sample = next(iter(datapipe1))

    assert sample["centroids_confidence_maps"].shape == (1, 1, 384, 384)

    datapipe2 = MultiConfidenceMapGenerator(
        datapipe,
        sigma=3,
        output_stride=2,
        centroids=True,
        image_key="image",
        instance_key="instances",
    )
    sample = next(iter(datapipe2))

    assert sample["centroids_confidence_maps"].shape == (1, 1, 192, 192)
    assert torch.sum(sample["centroids_confidence_maps"] > 0.98) == 2

    xv, yv = make_grid_vectors(2, 2, 1)
    points = torch.Tensor([[[[torch.nan, torch.nan], [torch.nan, torch.nan]]]])
    cms = make_multi_confmaps(points, xv, yv, 1)
    gt = torch.Tensor([[0.0000, 0.0000], [0.0000, 0.0000]])
    torch.testing.assert_close(gt, cms[0][0], atol=0.001, rtol=0.0)

    # centroids = False (for instances)
    datapipe = LabelsReader.from_filename(minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = Resizer(datapipe, scale=2)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe1 = MultiConfidenceMapGenerator(
        datapipe,
        sigma=1.5,
        output_stride=1,
        centroids=False,
        image_key="image",
        instance_key="instances",
    )
    sample = next(iter(datapipe1))

    assert sample["confidence_maps"].shape == (1, 2, 768, 768)
    assert torch.sum(sample["confidence_maps"] > 0.93) == 4


def test_generate_confmaps(minimal_instance):
    """Test `generate_confmaps` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)

    confmaps = generate_confmaps(
        ex["instances"][:, 0].unsqueeze(dim=1), img_hw=(384, 384)
    )
    assert confmaps.shape == (1, 2, 192, 192)


def test_generate_multiconfmaps(minimal_instance):
    """Test `generate_multiconfmaps` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)

    confmaps = generate_multiconfmaps(
        ex["instances"], img_hw=(384, 384), num_instances=ex["num_instances"]
    )
    assert confmaps.shape == (1, 2, 192, 192)

    confmaps = generate_multiconfmaps(
        ex["instances"][:, :, 0, :],
        img_hw=(384, 384),
        num_instances=ex["num_instances"],
        is_centroids=True,
    )
    assert confmaps.shape == (1, 1, 192, 192)
