import numpy as np
import torch
import sleap_io as sio
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.data.providers import process_lf
from sleap_nn.data.edge_maps import (
    distance_to_edge,
    make_edge_maps,
    make_pafs,
    make_multi_pafs,
    get_edge_points,
    generate_pafs,
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
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.4578, 0.9692, 0.4578],
                    [0.6065, 1.0000, 0.6065],
                    [0.4578, 0.9692, 0.4578],
                ],
            ],
            [
                [
                    [0.7071, 0.6240, 0.0957],
                    [0.6240, 0.7071, 0.6240],
                    [0.0957, 0.6240, 0.7071],
                ],
                [
                    [0.7071, 0.6240, 0.0957],
                    [0.6240, 0.7071, 0.6240],
                    [0.0957, 0.6240, 0.7071],
                ],
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
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.9157, 1.9385, 0.9157],
                    [1.2131, 2.0000, 1.2131],
                    [0.9157, 1.9385, 0.9157],
                ],
            ],
            [
                [
                    [1.4142, 1.2480, 0.1914],
                    [1.2480, 1.4142, 1.2480],
                    [0.1914, 1.2480, 1.4142],
                ],
                [
                    [1.4142, 1.2480, 0.1914],
                    [1.2480, 1.4142, 1.2480],
                    [0.1914, 1.2480, 1.4142],
                ],
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


def test_generate_pafs(minimal_instance):
    """Test `generate_pafs` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    pafs = generate_pafs(
        ex["instances"],
        img_hw=(384, 384),
        edge_inds=torch.Tensor(labels.skeletons[0].edge_inds),
    )
    assert pafs.shape == (1, 2, 192, 192)
