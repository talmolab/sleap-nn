"""Tests for sleap_nn.export.utils module."""

import pytest
import torch

from sleap_nn.export.utils import build_bottomup_candidate_template


class TestBuildBottomupCandidateTemplate:
    """Tests for build_bottomup_candidate_template function."""

    def test_output_shapes(self):
        """Test that output tensors have correct shapes."""
        n_nodes = 5
        k = 3
        edge_inds = [(0, 1), (1, 2), (2, 3)]
        n_edges = len(edge_inds)

        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # peak_channel_inds: (n_nodes * k,)
        assert peak_ch.shape == (n_nodes * k,)

        # edge_inds: (n_edges * k * k,)
        assert edge_idx.shape == (n_edges * k * k,)

        # edge_peak_inds: (n_edges * k * k, 2)
        assert edge_peaks.shape == (n_edges * k * k, 2)

    def test_peak_channel_indices(self):
        """Test that peak_channel_inds maps flat peak index to correct node."""
        n_nodes = 4
        k = 2

        peak_ch, _, _ = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=[(0, 1)]
        )

        # peak_ch should be [0, 0, 1, 1, 2, 2, 3, 3]
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
        assert torch.equal(peak_ch, expected)

    def test_edge_indices_ordering(self):
        """Test that edge_inds assigns correct edge index for each candidate."""
        n_nodes = 3
        k = 2
        edge_inds = [(0, 1), (1, 2)]

        _, edge_idx, _ = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # For k=2, each edge has k*k=4 candidates
        # Expected: [0,0,0,0, 1,1,1,1]
        expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        assert torch.equal(edge_idx, expected)

    def test_edge_peak_indices_row_major_ordering(self):
        """Test that edge_peak_inds follows row-major (i*k + j) pattern."""
        n_nodes = 3
        k = 2
        edge_inds = [(0, 1)]  # src_node=0, dst_node=1

        _, _, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # For edge (0, 1) with k=2:
        # Position 0: i=0, j=0 -> src=0*2+0=0, dst=1*2+0=2
        # Position 1: i=0, j=1 -> src=0*2+0=0, dst=1*2+1=3
        # Position 2: i=1, j=0 -> src=0*2+1=1, dst=1*2+0=2
        # Position 3: i=1, j=1 -> src=0*2+1=1, dst=1*2+1=3
        expected = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=torch.int32)
        assert torch.equal(edge_peaks, expected)

    def test_multiple_edges(self):
        """Test correct indexing for multiple edges."""
        n_nodes = 4
        k = 2
        edge_inds = [(0, 1), (2, 3)]

        _, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Edge 0: (0, 1) -> src_base=0, dst_base=2
        # Edge 1: (2, 3) -> src_base=4, dst_base=6
        expected_peaks = torch.tensor(
            [
                # Edge 0: src from [0,1], dst from [2,3]
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                # Edge 1: src from [4,5], dst from [6,7]
                [4, 6],
                [4, 7],
                [5, 6],
                [5, 7],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(edge_peaks, expected_peaks)

    def test_realistic_skeleton(self):
        """Test with realistic skeleton dimensions (15 nodes, 14 edges, k=20)."""
        n_nodes = 15
        k = 20
        # Chain skeleton: node 0 -> 1 -> 2 -> ... -> 14
        edge_inds = [(i, i + 1) for i in range(14)]
        n_edges = len(edge_inds)

        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Verify shapes
        assert peak_ch.shape == (n_nodes * k,)  # 15 * 20 = 300
        assert edge_idx.shape == (n_edges * k * k,)  # 14 * 400 = 5600
        assert edge_peaks.shape == (n_edges * k * k, 2)

        # Spot check: first candidate for edge (5, 6)
        # Edge index 5, position 0
        idx = 5 * k * k
        assert edge_idx[idx] == 5
        assert edge_peaks[idx, 0] == 5 * k  # src_base = 5 * 20 = 100
        assert edge_peaks[idx, 1] == 6 * k  # dst_base = 6 * 20 = 120

    def test_empty_edges(self):
        """Test behavior with no edges."""
        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=5, max_peaks_per_node=3, edge_inds=[]
        )

        assert peak_ch.shape == (15,)  # n_nodes * k
        assert edge_idx.shape == (0,)
        assert edge_peaks.shape == (0, 2)

    def test_single_edge(self):
        """Test with single edge."""
        n_nodes = 10
        k = 5
        edge_inds = [(3, 7)]  # Non-adjacent nodes

        _, _, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Verify first and last candidate
        # First: i=0, j=0 -> src=3*5+0=15, dst=7*5+0=35
        assert edge_peaks[0, 0] == 15
        assert edge_peaks[0, 1] == 35

        # Last: i=4, j=4 -> src=3*5+4=19, dst=7*5+4=39
        assert edge_peaks[-1, 0] == 19
        assert edge_peaks[-1, 1] == 39

    def test_dtypes(self):
        """Test that output tensors have correct dtypes."""
        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=3, max_peaks_per_node=2, edge_inds=[(0, 1)]
        )

        assert peak_ch.dtype == torch.int32
        assert edge_idx.dtype == torch.int32
        assert edge_peaks.dtype == torch.int32
