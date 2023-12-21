import torch

from sleap_nn.paf_grouping import get_connection_candidates


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
