import torch

from sleap_nn.paf_grouping import (
    get_connection_candidates,
    make_line_subs,
    get_paf_lines,
)


def test_get_connection_candidates():
    peak_channel_inds_sample = torch.tensor([0, 0, 0, 1, 1, 2])
    skeleton_edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    n_nodes = 4

    edge_inds, edge_peak_inds = get_connection_candidates(
        peak_channel_inds_sample, skeleton_edges, n_nodes
    )

    gt_edge_inds = [0, 0, 0, 0, 0, 0, 1, 1]
    gt_edge_peak_inds = [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [4, 5]]

    assert edge_inds.numpy().tolist() == gt_edge_inds
    assert edge_peak_inds.numpy().tolist() == gt_edge_peak_inds


def test_make_line_subs():
    peaks_sample = torch.tensor([[0, 0], [4, 8]], dtype=torch.float32)
    edge_peak_inds = torch.tensor([[0, 1]], dtype=torch.int32)
    edge_inds = torch.tensor([0], dtype=torch.int32)

    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points=3, pafs_stride=2
    )

    assert line_subs.numpy().tolist() == [
        [[[0, 0, 0], [0, 0, 1]], [[2, 1, 0], [2, 1, 1]], [[4, 2, 0], [4, 2, 1]]]
    ]


def test_get_paf_lines():
    pafs_sample = torch.arange(6 * 4 * 2).view(6, 4, 2).float()
    peaks_sample = torch.tensor([[0, 0], [4, 8]], dtype=torch.float32)
    edge_peak_inds = torch.tensor([[0, 1]], dtype=torch.int32)
    edge_inds = torch.tensor([0], dtype=torch.int32)

    paf_lines = get_paf_lines(
        pafs_sample,
        peaks_sample,
        edge_peak_inds,
        edge_inds,
        n_line_points=3,
        pafs_stride=2,
    )

    gt_paf_lines = [[[0, 1], [18, 19], [36, 37]]]

    assert paf_lines.numpy().tolist() == gt_paf_lines
