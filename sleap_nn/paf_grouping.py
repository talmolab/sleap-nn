"""This module provides a set of utilities for grouping peaks based on PAFs.

Part affinity fields (PAFs) are a representation used to resolve the peak grouping
problem for multi-instance pose estimation [1].

They are a convenient way to represent directed graphs with support in image space. For
each edge, a PAF can be represented by an image with two channels, corresponding to the
x and y components of a unit vector pointing along the direction of the underlying
directed graph formed by the connections of the landmarks belonging to an instance.

Given a pair of putatively connected landmarks, the agreement between the line segment
that connects them and the PAF vectors found at the coordinates along the same line can
be used as a measure of "connectedness". These scores can then be used to guide the
instance-wise grouping of landmarks.

This image space representation is particularly useful as it is amenable to neural
network-based prediction from unlabeled images.

A high-level API for grouping based on PAFs is provided through the `PAFScorer` class.

References:
    .. [1] Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D
       Pose Estimation using Part Affinity Fields. In _CVPR_, 2017.
"""

import attr
from typing import Tuple
import torch


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PeakID:
    """Indices to uniquely identify a single peak.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        node_ind: Index of the node type (channel) of the peak.
        peak_ind: Index of the peak within its node type.
    """

    node_ind: int
    peak_ind: int


@attr.s(auto_attribs=True, slots=True, frozen=True)
class EdgeType:
    """Indices to uniquely identify a single edge type.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        src_node_ind: Index of the source node type within the skeleton edges.
        dst_node_ind: Index of the destination node type within the skeleton edges.
    """

    src_node_ind: int
    dst_node_ind: int


@attr.s(auto_attribs=True, slots=True)
class EdgeConnection:
    """Indices to specify a matched connection between two peaks.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        src_peak_ind: Index of the source peak within all peaks.
        dst_peak_ind: Index of the destination peak within all peaks.
        score: Score of the match.
    """

    src_peak_ind: int
    dst_peak_ind: int
    score: float


def get_connection_candidates(
    peak_channel_inds_sample: torch.Tensor, skeleton_edges: torch.Tensor, n_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the indices of all the possible connections formed by the detected peaks.

    Args:
        peak_channel_inds_sample: The channel indices of the peaks found in a sample.
            This is a `tf.Tensor` of shape `(n_peaks,)` and dtype `tf.int32` that is
            used to represent a detected peak by its channel/node index in the skeleton.
        skeleton_edges: The indices of the nodes that form the skeleton graph as a
            `tf.Tensor` of shape `(n_edges, 2)` and dtype `tf.int32` where each row
            corresponds to the source and destination node indices.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.

    Returns:
        A tuple of `(edge_inds, edge_peak_inds)`.

        `edge_inds` is a `tf.Tensor` of shape `(n_candidates,)` indicating the indices
        of the edge that each of the candidate connections belongs to.

        `edge_peak_inds` is a `tf.Tensor` of shape `(n_candidates, 2)` with the indices
        of the peaks that form the source and destination of each candidate connection.
        This indexes into the input `peak_channel_inds_sample`.
    """
    peak_inds = torch.argsort(peak_channel_inds_sample)
    node_inds = torch.gather(peak_channel_inds_sample, 0, peak_inds)
    node_grouped_peak_inds = [
        peak_inds[node_inds == k] for k in range(n_nodes)
    ]  # (n_nodes, (n_peaks_k))
    edge_grouped_peak_inds = [
        (node_grouped_peak_inds[src], node_grouped_peak_inds[dst])
        for src, dst in skeleton_edges
    ]  # (n_edges, (n_src_peaks), (n_dst_peaks))

    edge_inds = []  # (n_edges, (n_src * n_dst))
    edge_peak_inds = []  # (n_edges, (n_src * n_dst), 2)
    for k, (src_peaks, dst_peaks) in enumerate(edge_grouped_peak_inds):
        grid_src, grid_dst = torch.meshgrid(src_peaks, dst_peaks, indexing="ij")
        grid_src_dst = torch.stack([grid_src.flatten(), grid_dst.flatten()], dim=1)

        edge_inds.append(torch.full((grid_src_dst.size(0),), k, dtype=torch.int32))
        edge_peak_inds.append(grid_src_dst)

    edge_inds = torch.cat(edge_inds)  # (n_candidates,)
    edge_peak_inds = torch.cat(edge_peak_inds)  # (n_candidates, 2)

    return edge_inds, edge_peak_inds
