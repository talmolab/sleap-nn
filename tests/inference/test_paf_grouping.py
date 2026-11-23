import numpy as np
import pytest
import torch
from torch.testing import assert_close
from numpy.testing import assert_array_equal, assert_array_almost_equal
from omegaconf import OmegaConf

from sleap_nn.inference.paf_grouping import (
    get_connection_candidates,
    make_line_subs,
    get_paf_lines,
    compute_distance_penalty,
    score_paf_lines,
    score_paf_lines_batch,
    split_peaks_by_sample,
    match_candidates_sample,
    match_candidates_batch,
    EdgeType,
    EdgeConnection,
    PeakID,
    toposort_edges,
    assign_connections_to_instances,
    make_predicted_instances,
    group_instances_sample,
    group_instances_batch,
    PAFScorer,
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
        peaks_sample,
        edge_peak_inds,
        edge_inds,
        n_line_points=3,
        pafs_stride=2,
        pafs_hw=(9, 9),
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
    peaks = [torch.tensor([[0, 0], [4, 8]], dtype=torch.float32)]
    peak_channel_inds = [torch.tensor([0, 1], dtype=torch.int32)]
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
    assert edge_inds[0].numpy().tolist() == [0.0]
    assert len(edge_peak_inds) == 1
    assert edge_peak_inds[0].numpy().tolist() == [[0, 1]]
    assert len(line_scores) == 1
    assert_close(line_scores[0], torch.tensor([24.27]), rtol=8e-2, atol=8e-2)


def test_match_candidates_sample():
    edge_inds_sample = torch.tensor([0, 0], dtype=torch.int32)
    edge_peak_inds_sample = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)
    line_scores_sample = torch.tensor([-0.5, 1.0], dtype=torch.float32)
    n_edges = 1

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = match_candidates_sample(
        edge_inds_sample, edge_peak_inds_sample, line_scores_sample, n_edges
    )

    src_peak_inds_k = torch.unique(edge_peak_inds_sample[:, 0])
    dst_peak_inds_k = torch.unique(edge_peak_inds_sample[:, 1])

    assert_array_equal(match_edge_inds, [0])
    assert_array_equal(match_src_peak_inds, [1])
    assert_array_equal(match_dst_peak_inds, [0])
    assert_array_equal(match_line_scores, [1.0])
    assert src_peak_inds_k[match_src_peak_inds][0] == 2
    assert dst_peak_inds_k[match_dst_peak_inds][0] == 1


def test_match_candidates_batch():
    edge_inds = [torch.tensor([0, 0], dtype=torch.int32)]
    edge_peak_inds = [torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)]
    line_scores = [torch.tensor([-0.5, 1.0], dtype=torch.float32)]
    n_edges = 1

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = match_candidates_batch(edge_inds, edge_peak_inds, line_scores, n_edges)
    assert len(match_edge_inds) == 1
    assert len(match_src_peak_inds) == 1
    assert len(match_dst_peak_inds) == 1
    assert len(match_line_scores) == 1
    assert match_edge_inds[0].numpy().tolist() == [0]
    assert match_src_peak_inds[0].numpy().tolist() == [1]
    assert match_dst_peak_inds[0].numpy().tolist() == [0]
    assert match_line_scores[0].numpy().tolist() == [1.0]


def test_toposort_edges():
    edge_inds = [
        (5, 7),
        (5, 8),
        (5, 9),
        (5, 6),
        (5, 11),
        (5, 12),
        (1, 0),
        (1, 3),
        (1, 2),
        (1, 10),
        (1, 13),
        (1, 14),
        (4, 5),
        (4, 1),
    ]

    edge_types = [EdgeType(src_node, dst_node) for src_node, dst_node in edge_inds]

    sorted_edge_inds = toposort_edges(edge_types)
    assert sorted_edge_inds == (12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    edge_inds = [
        (1, 4),
        (1, 5),
        (6, 8),
        (6, 7),
        (6, 9),
        (9, 10),
        (1, 0),
        (1, 3),
        (1, 2),
        (6, 1),
    ]
    edge_types = [EdgeType(src_node, dst_node) for src_node, dst_node in edge_inds]
    sorted_edge_inds = toposort_edges(edge_types)
    assert sorted_edge_inds == (2, 3, 4, 9, 5, 0, 1, 6, 7, 8)


def test_assign_connections_to_instances():
    connections = {
        EdgeType(src_node_ind=5, dst_node_ind=7): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0465653)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=8): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0607507)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=9): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9563284)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=6): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=1, score=0.5797864)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=11): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9892818)
        ],
        EdgeType(src_node_ind=5, dst_node_ind=12): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.7557168)
        ],
        EdgeType(src_node_ind=1, dst_node_ind=0): [],
        EdgeType(src_node_ind=1, dst_node_ind=3): [],
        EdgeType(src_node_ind=1, dst_node_ind=2): [],
        EdgeType(src_node_ind=1, dst_node_ind=10): [],
        EdgeType(src_node_ind=1, dst_node_ind=13): [],
        EdgeType(src_node_ind=1, dst_node_ind=14): [],
        EdgeType(src_node_ind=4, dst_node_ind=5): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.9735552)
        ],
        EdgeType(src_node_ind=4, dst_node_ind=1): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=0.31536198)
        ],
    }
    instance_assignments = assign_connections_to_instances(
        connections,
        min_instance_peaks=0,
        n_nodes=15,
    )
    assert instance_assignments == {
        PeakID(node_ind=5, peak_ind=0): 0,
        PeakID(node_ind=7, peak_ind=0): 0,
        PeakID(node_ind=8, peak_ind=0): 0,
        PeakID(node_ind=9, peak_ind=0): 0,
        PeakID(node_ind=6, peak_ind=1): 0,
        PeakID(node_ind=11, peak_ind=0): 0,
        PeakID(node_ind=12, peak_ind=0): 0,
        PeakID(node_ind=4, peak_ind=0): 1,
        PeakID(node_ind=1, peak_ind=0): 1,
    }

    # Now in topological order:
    edge_types = list(connections.keys())
    sorted_edge_inds = toposort_edges(edge_types)
    instance_assignments = assign_connections_to_instances(
        {
            edge_types[edge_ind]: connections[edge_types[edge_ind]]
            for edge_ind in sorted_edge_inds
        },
        min_instance_peaks=0,
        n_nodes=15,
    )
    assert all(x == 0 for x in instance_assignments.values())

    connections = {
        EdgeType(src_node_ind=0, dst_node_ind=1): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0),
            EdgeConnection(src_peak_ind=1, dst_peak_ind=1, score=1.0),
        ],
        EdgeType(src_node_ind=1, dst_node_ind=2): [
            EdgeConnection(src_peak_ind=0, dst_peak_ind=0, score=1.0)
        ],
        EdgeType(src_node_ind=2, dst_node_ind=3): [
            EdgeConnection(src_peak_ind=1, dst_peak_ind=1, score=1.0)
        ],
    }

    min_instance_peaks = 0.5
    instance_assignments = assign_connections_to_instances(
        connections, min_instance_peaks=min_instance_peaks, n_nodes=4
    )

    expected_assignments = {
        PeakID(node_ind=0, peak_ind=0): 0,
        PeakID(node_ind=1, peak_ind=0): 0,
        PeakID(node_ind=0, peak_ind=1): 1,
        PeakID(node_ind=1, peak_ind=1): 1,
        PeakID(node_ind=2, peak_ind=0): 0,
        PeakID(node_ind=2, peak_ind=1): 2,
        PeakID(node_ind=3, peak_ind=1): 2,
    }

    assert instance_assignments == expected_assignments


def test_make_predicted_instances():
    peaks = np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])
    peak_scores = np.array([[0.9, 0.8], [0.7, 0.6]])
    connections = {
        EdgeType(0, 1): [EdgeConnection(0, 0, 0.5), EdgeConnection(1, 1, 0.4)]
    }
    instance_assignments = {
        PeakID(0, 0): 0,
        PeakID(0, 1): 1,
        PeakID(1, 0): 0,
        PeakID(1, 1): 1,
    }

    expected_instances = np.array([[[0, 0], [2, 2]], [[1, 1], [3, 3]]])
    expected_peak_scores = np.array([[0.9, 0.7], [0.8, 0.6]])
    expected_instance_scores = np.array([0.5, 0.4])

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = make_predicted_instances(peaks, peak_scores, connections, instance_assignments)

    assert_array_almost_equal(predicted_instances, expected_instances)
    assert_array_almost_equal(predicted_peak_scores, expected_peak_scores)
    assert_array_almost_equal(predicted_instance_scores, expected_instance_scores)


def test_group_instances_sample():
    peaks_sample = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    peak_scores_sample = torch.arange(5, dtype=torch.float32)
    peak_channel_inds_sample = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    match_edge_inds_sample = torch.tensor([0, 1, 0], dtype=torch.int32)
    match_src_peak_inds_sample = torch.tensor([0, 0, 1], dtype=torch.int32)
    match_dst_peak_inds_sample = torch.tensor([0, 0, 1], dtype=torch.int32)
    match_line_scores_sample = torch.ones(3, dtype=torch.float32)

    n_nodes = 3
    sorted_edge_inds = (0, 1)

    edge_types = [EdgeType(0, 1), EdgeType(1, 2)]
    min_instance_peaks = 0

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = group_instances_sample(
        peaks_sample,
        peak_scores_sample,
        peak_channel_inds_sample,
        match_edge_inds_sample,
        match_src_peak_inds_sample,
        match_dst_peak_inds_sample,
        match_line_scores_sample,
        n_nodes,
        sorted_edge_inds,
        edge_types,
        min_instance_peaks,
    )

    assert_array_equal(
        predicted_instances,
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [
                [6.0, 7.0],
                [8.0, 9.0],
                [np.nan, np.nan],
            ],
        ],
    )
    assert_array_equal(predicted_peak_scores, [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])
    assert_array_equal(predicted_instance_scores, [2.0, 1.0])


def test_group_instances_batch():
    gt_predicted_instances = [
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        [[6.0, 7.0], [8.0, 9.0], [np.nan, np.nan]],
    ]

    peaks = [torch.arange(10, dtype=torch.float32).reshape(5, 2)]
    peak_scores = [torch.arange(5, dtype=torch.float32)]
    peak_channel_inds = [torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)]
    match_edge_inds = [torch.tensor([0, 1, 0], dtype=torch.int32)]
    match_src_peak_inds = [torch.tensor([0, 0, 1], dtype=torch.int32)]
    match_dst_peak_inds = [torch.tensor([0, 0, 1], dtype=torch.int32)]
    match_line_scores = [torch.ones(3, dtype=torch.float32)]

    n_nodes = 3
    sorted_edge_inds = (0, 1)
    edge_types = [EdgeType(0, 1), EdgeType(1, 2)]
    min_instance_peaks = 0

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = group_instances_batch(
        peaks,
        peak_scores,
        peak_channel_inds,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
        n_nodes,
        sorted_edge_inds,
        edge_types,
        min_instance_peaks,
    )

    assert isinstance(predicted_instances, list)
    assert isinstance(predicted_peak_scores, list)
    assert isinstance(predicted_instance_scores, list)
    assert len(predicted_instances) == 1
    assert len(predicted_peak_scores) == 1
    assert len(predicted_instance_scores) == 1

    assert_array_equal(predicted_instances[0].numpy(), gt_predicted_instances)
    assert_array_equal(
        predicted_peak_scores[0].numpy(), [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]]
    )
    assert_array_equal(predicted_instance_scores[0].numpy(), [2.0, 1.0])


def test_paf_scorer_from_config():
    config = OmegaConf.create(
        {
            "confmaps": {"part_names": ["a", "b"]},
            "pafs": {"edges": [("a", "b")], "output_stride": 1},
        }
    )
    paf_scorer = PAFScorer.from_config(config=config)
    assert paf_scorer


def test_paf_scorer_score_paf_lines():
    pafs = torch.arange(6 * 4 * 2, dtype=torch.float32).reshape(1, 6, 4, 2)
    peaks = [torch.tensor([[0, 0], [4, 8]], dtype=torch.float32)]
    peak_channel_inds = [torch.tensor([0, 1], dtype=torch.int32)]
    skeleton_edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)

    n_line_points = 3
    pafs_stride = 2
    max_edge_length_ratio = 2 / 12
    dist_penalty_weight = 1.0
    n_nodes = 4

    config = OmegaConf.create(
        {
            "confmaps": {"part_names": ["a", "b"]},
            "pafs": {"edges": [("a", "b")], "output_stride": 1},
        }
    )
    paf_scorer = PAFScorer.from_config(
        config=config,
        max_edge_length_ratio=max_edge_length_ratio,
        dist_penalty_weight=dist_penalty_weight,
        n_points=n_line_points,
    )
    paf_scorer.edge_inds = skeleton_edges
    paf_scorer.pafs_stride = pafs_stride
    paf_scorer.n_nodes = n_nodes

    edge_inds, edge_peak_inds, line_scores = paf_scorer.score_paf_lines(
        pafs, peaks, peak_channel_inds
    )

    assert len(edge_inds) == 1
    assert edge_inds[0].numpy().tolist() == [0.0]
    assert len(edge_peak_inds) == 1
    assert edge_peak_inds[0].numpy().tolist() == [[0, 1]]
    assert len(line_scores) == 1
    assert_close(line_scores[0], torch.tensor([24.27]), rtol=8e-2, atol=8e-2)


def test_paf_scorer_match_candidates():
    edge_inds = [torch.tensor([0, 0], dtype=torch.int32)]
    edge_peak_inds = [torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)]
    line_scores = [torch.tensor([-0.5, 1.0], dtype=torch.float32)]
    n_edges = 1

    config = OmegaConf.create(
        {
            "confmaps": {"part_names": ["a", "b"]},
            "pafs": {"edges": [("a", "b")], "output_stride": 1},
        }
    )
    paf_scorer = PAFScorer.from_config(
        config=config,
    )
    paf_scorer.n_edges = n_edges

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = paf_scorer.match_candidates(edge_inds, edge_peak_inds, line_scores)

    assert len(match_edge_inds) == 1
    assert len(match_src_peak_inds) == 1
    assert len(match_dst_peak_inds) == 1
    assert len(match_line_scores) == 1
    assert match_edge_inds[0].numpy().tolist() == [0]
    assert match_src_peak_inds[0].numpy().tolist() == [1]
    assert match_dst_peak_inds[0].numpy().tolist() == [0]
    assert match_line_scores[0].numpy().tolist() == [1.0]


def test_paf_scorer_group_instances():
    gt_predicted_instances = [
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        [[6.0, 7.0], [8.0, 9.0], [np.nan, np.nan]],
    ]

    peaks = [torch.arange(10, dtype=torch.float32).reshape(5, 2)]
    peak_scores = [torch.arange(5, dtype=torch.float32)]
    peak_channel_inds = [torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)]
    match_edge_inds = [torch.tensor([0, 1, 0], dtype=torch.int32)]
    match_src_peak_inds = [torch.tensor([0, 0, 1], dtype=torch.int32)]
    match_dst_peak_inds = [torch.tensor([0, 0, 1], dtype=torch.int32)]
    match_line_scores = [torch.ones(3, dtype=torch.float32)]

    n_nodes = 3
    sorted_edge_inds = (0, 1)
    edge_types = [EdgeType(0, 1), EdgeType(1, 2)]
    min_instance_peaks = 0

    config = OmegaConf.create(
        {
            "confmaps": {"part_names": ["a", "b"]},
            "pafs": {"edges": [("a", "b")], "output_stride": 1},
        }
    )
    paf_scorer = PAFScorer.from_config(
        config=config,
    )
    paf_scorer.n_nodes = n_nodes
    paf_scorer.sorted_edge_inds = sorted_edge_inds
    paf_scorer.edge_types = edge_types
    paf_scorer.min_instance_peaks = min_instance_peaks

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = paf_scorer.group_instances(
        peaks,
        peak_scores,
        peak_channel_inds,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )

    assert isinstance(predicted_instances, list)
    assert isinstance(predicted_peak_scores, list)
    assert isinstance(predicted_instance_scores, list)
    assert len(predicted_instances) == 1
    assert len(predicted_peak_scores) == 1
    assert len(predicted_instance_scores) == 1

    assert_array_equal(predicted_instances[0].numpy(), gt_predicted_instances)
    assert_array_equal(
        predicted_peak_scores[0].numpy(), [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]]
    )
    assert_array_equal(predicted_instance_scores[0].numpy(), [2.0, 1.0])


# ---------------------------------------------------------------------------
# Batched PAF line scoring vs the per-sample loop it replaced.
# ---------------------------------------------------------------------------

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")

# A 7-node tree with a branch, plus a node (6) that no edge uses.
_EDGES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
_N_NODES = 7


def _loop_score_paf_lines_batch(
    pafs,
    peaks,
    peak_channel_inds,
    skeleton_edges,
    n_line_points,
    pafs_stride,
    max_edge_length_ratio,
    dist_penalty_weight,
    n_nodes,
):
    """The per-sample loop that `score_paf_lines_batch` used before."""
    max_edge_length = (
        max_edge_length_ratio
        * max(pafs.shape[-1], pafs.shape[-2], pafs.shape[-3])
        * pafs_stride
    )
    out = ([], [], [])
    for sample in range(pafs.shape[0]):
        edge_inds_sample, edge_peak_inds_sample = get_connection_candidates(
            peak_channel_inds[sample], skeleton_edges, n_nodes
        )
        paf_lines_sample = get_paf_lines(
            pafs[sample],
            peaks[sample],
            edge_peak_inds_sample,
            edge_inds_sample,
            n_line_points,
            pafs_stride,
        )
        line_scores_sample = score_paf_lines(
            paf_lines_sample,
            peaks[sample],
            edge_peak_inds_sample,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        )
        out[0].append(edge_inds_sample)
        out[1].append(edge_peak_inds_sample)
        out[2].append(line_scores_sample)
    return out


def _random_paf_batch(rng, device, counts, hw=(24, 32), stride=4, dup=False):
    """Random PAFs and peaks; ``counts[b][k]`` peaks of node ``k`` in sample ``b``.

    Peaks of a sample are shuffled across nodes, like `find_local_peaks` gives
    them (ordered by position, not by node). With ``dup``, some peaks repeat
    another peak's position exactly (zero-length candidate lines).
    """
    B = len(counts)
    h, w = hw
    pafs = torch.from_numpy(
        rng.uniform(-1, 1, size=(B, h, w, 2 * len(_EDGES))).astype(np.float32)
    ).to(device)
    peaks, channels = [], []
    for b in range(B):
        ch = np.repeat(np.arange(_N_NODES), counts[b])
        rng.shuffle(ch)
        xy = rng.uniform(-5, [w * stride + 5, h * stride + 5], size=(ch.size, 2))
        if dup and ch.size > 1:
            src = rng.integers(0, ch.size, size=ch.size // 3)
            dst = rng.integers(0, ch.size, size=ch.size // 3)
            xy[dst] = xy[src]
        peaks.append(torch.from_numpy(xy.astype(np.float32)).to(device))
        channels.append(torch.from_numpy(ch.astype(np.int32)).to(device))
    return pafs, peaks, channels


def _sorted_candidates(edge_inds, edge_peak_inds, line_scores):
    """Candidates of one sample on the CPU, ordered by (edge, src, dst)."""
    e = edge_inds.cpu().long()
    p = edge_peak_inds.cpu().long()
    n = int(p.max()) + 1 if p.numel() else 1
    order = torch.argsort((e * n + p[:, 0]) * n + p[:, 1])
    return e[order], p[order], line_scores.cpu()[order]


def _paf_counts(rng, B):
    """Peak counts per (sample, node): empty samples, single peaks, crowds."""
    kind = rng.integers(0, 4)
    if kind == 0:
        return rng.integers(0, 3, size=(B, _N_NODES))
    if kind == 1:
        return rng.integers(0, 7, size=(B, _N_NODES))
    counts = rng.integers(0, 20, size=(B, _N_NODES))
    counts[rng.random(B) < 0.3] = 0  # some samples with no peaks at all
    return counts


@pytest.mark.parametrize("device", DEVICES)
def test_score_paf_lines_batch_matches_loop(device):
    rng = np.random.default_rng(0)
    edges = torch.tensor(_EDGES, dtype=torch.int32)
    for trial in range(150):
        B = int(rng.integers(1, 6))
        counts = _paf_counts(rng, B)
        pafs, peaks, channels = _random_paf_batch(
            rng, device, counts, dup=bool(trial % 3 == 0)
        )
        args = (pafs, peaks, channels, edges, 7, 4, 0.25, 1.0, _N_NODES)
        old = _loop_score_paf_lines_batch(*args)
        new = score_paf_lines_batch(*args)
        new_with_counts = score_paf_lines_batch(*args, peak_counts=counts)
        for res in (new, new_with_counts):
            assert [len(x) for x in res] == [B, B, B]
            for b in range(B):
                # Same dtypes and devices as before.
                for o, n in zip((old[0][b], old[1][b], old[2][b]), (r[b] for r in res)):
                    assert n.dtype == o.dtype and n.device == o.device
                o = _sorted_candidates(old[0][b], old[1][b], old[2][b])
                n = _sorted_candidates(res[0][b], res[1][b], res[2][b])
                assert torch.equal(o[0], n[0])
                assert torch.equal(o[1], n[1])
                # Same arithmetic per candidate: equal bits, NaNs included
                # (zero-length lines from duplicated peaks give NaN).
                assert torch.equal(o[2].isnan(), n[2].isnan())
                assert torch.equal(o[2].nan_to_num(), n[2].nan_to_num())


@pytest.mark.parametrize("device", DEVICES)
def test_score_paf_lines_batch_candidate_order(device):
    """Within a sample: by edge, then source peak, then destination peak."""
    pafs = torch.zeros(2, 8, 8, 2 * 2, device=device)
    peaks = [
        torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.float32),
        torch.zeros((0, 2), dtype=torch.float32),
    ]
    # Node of each peak, out of node order (as `find_local_peaks` gives them).
    channels = [
        torch.tensor([1, 0, 2, 1, 0], dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
    ]
    peaks = [p.to(device) for p in peaks]
    channels = [c.to(device) for c in channels]
    edge_inds, edge_peak_inds, line_scores = score_paf_lines_batch(
        pafs, peaks, channels, [(0, 1), (1, 2)], 3, 2, 1.0, 1.0, 3
    )
    assert edge_inds[0].tolist() == [0, 0, 0, 0, 1, 1]
    assert edge_peak_inds[0].tolist() == [
        [1, 0],
        [1, 3],
        [4, 0],
        [4, 3],
        [0, 2],
        [3, 2],
    ]
    assert edge_inds[1].shape == (0,) and edge_peak_inds[1].shape == (0, 2)
    assert line_scores[1].shape == (0,)


@pytest.mark.parametrize("device", DEVICES)
def test_score_paf_lines_batch_grouping_matches_loop(device):
    """Matching + grouping give the same instances as with the loop's output."""
    rng = np.random.default_rng(1)
    edges = torch.tensor(_EDGES, dtype=torch.int32)
    scorer = PAFScorer(
        part_names=[f"n{i}" for i in range(_N_NODES)],
        edges=[(f"n{s}", f"n{d}") for s, d in _EDGES],
        pafs_stride=4,
    )
    for _ in range(40):
        B = int(rng.integers(1, 5))
        counts = rng.integers(0, 6, size=(B, _N_NODES))
        pafs, peaks, channels = _random_paf_batch(rng, device, counts)
        args = (pafs, peaks, channels, edges, 10, 4, 0.25, 1.0, _N_NODES)
        results = []
        for fn in (_loop_score_paf_lines_batch, score_paf_lines_batch):
            edge_inds, edge_peak_inds, line_scores = fn(*args)
            matched = match_candidates_batch(
                edge_inds, edge_peak_inds, line_scores, len(_EDGES)
            )
            results.append(
                scorer.group_instances(
                    [p.cpu() for p in peaks],
                    [torch.ones(p.shape[0]) for p in peaks],
                    [c.cpu() for c in channels],
                    *matched,
                )
            )
        for o, n in zip(*results):
            for ob, nb in zip(o, n):
                assert torch.equal(ob.isnan(), nb.isnan())
                assert torch.equal(ob.nan_to_num(), nb.nan_to_num())


def _loop_split(peaks, peak_vals, sample_inds, peak_channel_inds, B):
    out = ([], [], [])
    for b in range(B):
        m = sample_inds == b
        out[0].append(peaks[m])
        out[1].append(peak_vals[m].to(torch.float32))
        out[2].append(peak_channel_inds[m])
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shuffle", [False, True])
def test_split_peaks_by_sample_matches_loop(device, shuffle):
    rng = np.random.default_rng(2)
    for trial in range(100):
        B = int(rng.integers(1, 7))
        counts = _paf_counts(rng, B)
        sample_inds = np.repeat(np.arange(B), counts.sum(axis=1))
        channels = np.concatenate(
            [rng.permutation(np.repeat(np.arange(_N_NODES), c)) for c in counts]
        ).astype(np.int64)
        if shuffle:
            order = rng.permutation(sample_inds.size)
            sample_inds, channels = sample_inds[order], channels[order]
        n = sample_inds.size
        peaks = torch.from_numpy(rng.uniform(0, 100, (n, 2)).astype(np.float32))
        dtype = torch.float16 if trial % 2 else torch.float32
        vals = torch.from_numpy(rng.uniform(0, 1, n)).to(dtype)
        args = [
            peaks.to(device),
            vals.to(device),
            torch.from_numpy(sample_inds).to(torch.int32).to(device),
            torch.from_numpy(channels).to(torch.int32).to(device),
        ]
        old = _loop_split(*args, B)
        *new, peak_counts = split_peaks_by_sample(*args, B, _N_NODES)
        assert peak_counts.shape == (B, _N_NODES) and peak_counts.dtype == np.int64
        np.testing.assert_array_equal(peak_counts, counts)
        for o_list, n_list in zip(old, new):
            assert len(n_list) == B
            for o, nt in zip(o_list, n_list):
                assert o.dtype == nt.dtype and o.device == nt.device
                assert torch.equal(o, nt)


@pytest.mark.parametrize("device", DEVICES)
def test_score_paf_lines_batch_no_peaks(device):
    """A batch where no frame has a peak, and a batch of no frames."""
    edges = torch.tensor(_EDGES, dtype=torch.int32)
    pafs = torch.zeros(3, 6, 8, 2 * len(_EDGES), device=device)
    peaks = [torch.zeros((0, 2), device=device) for _ in range(3)]
    channels = [torch.zeros(0, dtype=torch.int32, device=device) for _ in range(3)]
    args = (pafs, peaks, channels, edges, 5, 4, 0.25, 1.0, _N_NODES)
    old = _loop_score_paf_lines_batch(*args)
    new = score_paf_lines_batch(*args)
    for o_list, n_list in zip(old, new):
        assert len(n_list) == 3
        for o, n in zip(o_list, n_list):
            assert n.shape == o.shape and n.dtype == o.dtype and n.device == o.device

    empty = score_paf_lines_batch(pafs[:0], [], [], edges, 5, 4, 0.25, 1.0, _N_NODES)
    assert empty == ([], [], [])

    *lists, counts = split_peaks_by_sample(
        torch.zeros((0, 2), device=device),
        torch.zeros(0, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
        3,
        _N_NODES,
    )
    assert counts.shape == (3, _N_NODES) and counts.sum() == 0
    assert all(len(x) == 3 and all(t.shape[0] == 0 for t in x) for x in lists)
