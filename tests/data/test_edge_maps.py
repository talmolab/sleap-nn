import numpy as np
import torch
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.data.edge_maps import distance_to_edge, make_edge_maps, make_pafs, make_multi_pafs


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
        dtype=torch.float32
    )

    edge_destination = torch.tensor(
        [
            [[1, 1.5], [2, 2]],
            [[1, 1.5], [2, 2]],
        ],
        dtype=torch.float32
    )
    sigma = 1.0

    pafs = make_multi_pafs(
        xv=xv,
        yv=yv,
        edge_sources=edge_source,
        edge_destinations=edge_destination,
        sigma=sigma,
    )