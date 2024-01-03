import torch
from torch.testing import assert_close
from numpy.testing import assert_array_equal

from sleap_nn.paf_grouping import (
    get_connection_candidates,
    make_line_subs,
    get_paf_lines,
    compute_distance_penalty,
    score_paf_lines,
    score_paf_lines_batch,
    match_candidates_sample
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


def test_compute_distance_penalty():
    spatial_vec_lengths_1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    penalties_1 = compute_distance_penalty(spatial_vec_lengths_1, max_edge_length=2)
    assert_close(
        penalties_1, torch.tensor([0, 0, 2 / 3 - 1, 2 / 4 - 1]), atol=1e-6, rtol=1e-6
    )

    spatial_vec_lengths_2 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    penalties_2 = compute_distance_penalty(
        spatial_vec_lengths_2, max_edge_length=2, dist_penalty_weight=2
    )
    assert_close(
        penalties_2, torch.tensor([0, 0, -0.6666666, -1]), atol=1e-6, rtol=1e-6
    )


def test_score_paf_lines():
    pafs_sample_torch = torch.arange(6 * 4 * 2).view(6, 4, 2).float()
    peaks_sample_torch = torch.tensor([[0, 0], [4, 8]], dtype=torch.float32)
    edge_peak_inds_torch = torch.tensor([[0, 1]], dtype=torch.int32)
    edge_inds_torch = torch.tensor([0], dtype=torch.int32)

    paf_lines_torch = get_paf_lines(
        pafs_sample_torch,
        peaks_sample_torch,
        edge_peak_inds_torch,
        edge_inds_torch,
        n_line_points=3,
        pafs_stride=2,
    )

    scores_torch = score_paf_lines(
        paf_lines_torch, peaks_sample_torch, edge_peak_inds_torch, max_edge_length=2
    )

    # Asserting the correctness of the scores
    assert_close(scores_torch, torch.tensor([24.27]), atol=1e-2, rtol=1e-2)


def test_score_paf_lines_batch():
    pafs = torch.arange(6 * 4 * 2, dtype=torch.float32).reshape(1, 6, 4, 2)
    peaks = torch.tensor([[[0, 0], [4, 8]]], dtype=torch.float32)
    peak_channel_inds = torch.tensor([[0, 1]], dtype=torch.int32)
    skeleton_edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)

    n_line_points = 3
    pafs_stride = 2
    max_edge_length_ratio = 2 / 12
    dist_penalty_weight = 1.0
    n_nodes = 4

    edge_inds, edge_peak_inds, line_scores = score_paf_lines_batch(
        pafs,
        peaks,
        peak_channel_inds,
        skeleton_edges,
        n_line_points,
        pafs_stride,
        max_edge_length_ratio,
        dist_penalty_weight,
        n_nodes,
    )

    assert len(edge_inds) == 1
    assert edge_inds[0].numpy().tolist() == [0.]
    assert len(edge_peak_inds) == 1
    assert edge_peak_inds[0].numpy().tolist() == [[0, 1]]
    assert len(line_scores) == 1
    assert_close(line_scores[0], torch.tensor([24.27]), rtol=8e-2, atol=8e-2)


def test_match_candidates_sample():
    edge_inds_sample = torch.tensor([0, 0], dtype=torch.int32)
    edge_peak_inds_sample = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)
    line_scores_sample = torch.tensor([-0.5, 1.0], dtype=torch.float32)
    n_edges = 1

    (match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores) = match_candidates_sample(
        edge_inds_sample,
        edge_peak_inds_sample,
        line_scores_sample,
        n_edges
    )

    src_peak_inds_k = torch.unique(edge_peak_inds_sample[:, 0])
    dst_peak_inds_k = torch.unique(edge_peak_inds_sample[:, 1])

    assert_array_equal(match_edge_inds, [0])
    assert_array_equal(match_src_peak_inds, [1])
    assert_array_equal(match_dst_peak_inds, [0])
    assert_array_equal(match_line_scores, [1.0])
    assert src_peak_inds_k[match_src_peak_inds][0] == 2
    assert dst_peak_inds_k[match_dst_peak_inds][0] == 1