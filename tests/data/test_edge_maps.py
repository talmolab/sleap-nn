import numpy as np
import torch
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.edge_maps import (
    distance_to_edge,
    make_edge_maps,
    make_pafs,
    make_multi_pafs,
    get_edge_points,
    PartAffinityFieldsGenerator
)


def test_distance_to_edge():
    xv, yv = make_grid_vectors(3, 3, 1)

    edge_source = torch.tensor([[1, 0.5], [0, 0]], dtype=torch.float32)
    edge_destination = torch.tensor([[1, 1.5], [2, 2]], dtype=torch.float32)

    yy, xx = torch.meshgrid(yv, xv, indexing="ij")
    points = torch.stack((xx, yy), dim=-1)

    result = distance_to_edge(
        points=points, edge_source=edge_source, edge_destination=edge_destination
    )

    gt_result = torch.tensor(
        [
            [[1.2500, 0.0000], [0.2500, 0.5000], [1.2500, 2.0000]],
            [[1.0000, 0.5000], [0.0000, 0.0000], [1.0000, 0.5000]],
            [[1.2500, 2.0000], [0.2500, 0.5000], [1.2500, 0.0000]],
        ]
    )

    assert gt_result.numpy().tolist() == result.numpy().tolist()


def test_make_edge_maps():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = torch.tensor([[1, 0.5], [0, 0]], dtype=torch.float32)
    edge_destination = torch.tensor([[1, 1.5], [2, 2]], dtype=torch.float32)
    sigma = 1.0

    edge_confidence_map = make_edge_maps(
        xv=xv,
        yv=yv,
        edge_source=edge_source,
        edge_destination=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        edge_confidence_map,
        [
            [[0.458, 1.000], [0.969, 0.882], [0.458, 0.135]],
            [[0.607, 0.882], [1.000, 1.000], [0.607, 0.882]],
            [[0.458, 0.135], [0.969, 0.882], [0.458, 1.000]],
        ],
        atol=1e-3,
    )


def test_make_pafs():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = torch.tensor([[1, 0.5], [0, 0]], dtype=torch.float32)
    edge_destination = torch.tensor([[1, 1.5], [2, 2]], dtype=torch.float32)
    sigma = 1.0

    pafs = make_pafs(
        xv=xv,
        yv=yv,
        edge_source=edge_source,
        edge_destination=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        pafs,
        [
            [
                [[0.0, 0.458], [0.707, 0.707]],
                [[0.0, 0.969], [0.624, 0.624]],
                [[0.0, 0.458], [0.096, 0.096]],
            ],
            [
                [[0.0, 0.607], [0.624, 0.624]],
                [[0.0, 1.0], [0.707, 0.707]],
                [[0.0, 0.607], [0.624, 0.624]],
            ],
            [
                [[0.0, 0.458], [0.096, 0.096]],
                [[0.0, 0.969], [0.624, 0.624]],
                [[0.0, 0.458], [0.707, 0.707]],
            ],
        ],
        atol=1e-3,
    )


def test_make_multi_pafs():
    xv, yv = make_grid_vectors(image_height=3, image_width=3, output_stride=1)
    edge_source = torch.tensor(
        [
            [[1, 0.5], [0, 0]],
            [[1, 0.5], [0, 0]],
        ],
        dtype=torch.float32,
    )

    edge_destination = torch.tensor(
        [
            [[1, 1.5], [2, 2]],
            [[1, 1.5], [2, 2]],
        ],
        dtype=torch.float32,
    )
    sigma = 1.0

    pafs = make_multi_pafs(
        xv=xv,
        yv=yv,
        edge_sources=edge_source,
        edge_destinations=edge_destination,
        sigma=sigma,
    )

    np.testing.assert_allclose(
        pafs,
        [
            [
                [[0.0, 0.916], [1.414, 1.414]],
                [[0.0, 1.938], [1.248, 1.248]],
                [[0.0, 0.916], [0.191, 0.191]],
            ],
            [
                [[0.0, 1.213], [1.248, 1.248]],
                [[0.0, 2.0], [1.414, 1.414]],
                [[0.0, 1.213], [1.248, 1.248]],
            ],
            [
                [[0.0, 0.916], [0.191, 0.191]],
                [[0.0, 1.938], [1.248, 1.248]],
                [[0.0, 0.916], [1.414, 1.414]],
            ],
        ],
        atol=1e-3,
    )


def test_get_edge_points():
    instances = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    edge_inds = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32)
    edge_sources, edge_destinations = get_edge_points(instances, edge_inds)

    np.testing.assert_array_equal(
        edge_sources,
        [
            [[0, 1], [2, 3], [0, 1]],
            [[6, 7], [8, 9], [6, 7]],
            [[12, 13], [14, 15], [12, 13]],
            [[18, 19], [20, 21], [18, 19]],
        ],
    )
    np.testing.assert_array_equal(
        edge_destinations,
        [
            [[2, 3], [4, 5], [4, 5]],
            [[8, 9], [10, 11], [10, 11]],
            [[14, 15], [16, 17], [16, 17]],
            [[20, 21], [22, 23], [22, 23]],
        ],
    )


def test_part_affinity_fields_generator(minimal_instance):
    provider = LabelsReader.from_filename(minimal_instance)
    paf_generator = PartAffinityFieldsGenerator(
        provider, 
        sigma=8, 
        output_stride=2, 
        edge_inds=torch.tensor(provider.labels.skeletons[0].edge_inds)
    )
    out = next(iter(paf_generator))
    assert out["part_affinity_fields"].shape == (192, 192, 1, 2)
    assert out["part_affinity_fields"].dtype == torch.float32