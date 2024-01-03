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
from typing import Tuple, List
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

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
            This is a `torch.Tensor` of shape `(n_peaks,)` and dtype `torch.int32` that is
            used to represent a detected peak by its channel/node index in the skeleton.
        skeleton_edges: The indices of the nodes that form the skeleton graph as a
            `torch.Tensor` of shape `(n_edges, 2)` and dtype `torch.int32` where each row
            corresponds to the source and destination node indices.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.

    Returns:
        A tuple of `(edge_inds, edge_peak_inds)`.

        `edge_inds` is a `torch.Tensor` of shape `(n_candidates,)` indicating the indices
        of the edge that each of the candidate connections belongs to.

        `edge_peak_inds` is a `torch.Tensor` of shape `(n_candidates, 2)` with the indices
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


def make_line_subs(
    peaks_sample: torch.Tensor,
    edge_peak_inds: torch.Tensor,
    edge_inds: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> torch.Tensor:
    """Create the lines between candidate connections for evaluating the PAFs.

    Args:
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype `torch.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The line subscripts as a `torch.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `torch.int32`.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        The subscripts are interpolated via nearest neighbor, so multiple fractional
        coordinates may map on to the same pixel if the line is short.

    See also: get_connection_candidates
    """
    src_peaks = torch.index_select(peaks_sample, 0, edge_peak_inds[:, 0])
    dst_peaks = torch.index_select(peaks_sample, 0, edge_peak_inds[:, 1])
    n_candidates = torch.tensor(src_peaks.shape[0])

    linspace_values = torch.linspace(0, 1, n_line_points, dtype=torch.float32)
    linspace_values = linspace_values.repeat(n_candidates, 1).view(
        n_candidates, n_line_points, 1
    )
    XY = (
        src_peaks.view(n_candidates, 1, 2)
        + (dst_peaks - src_peaks).view(n_candidates, 1, 2) * linspace_values
    )
    XY = XY.transpose(1, 2)
    XY = (
        (XY / pafs_stride).round().int()
    )  # (n_candidates, 2, n_line_points)  # dim 1 is [x, y]
    XY = XY[:, [1, 0], :]  # dim 1 is [row, col]

    edge_inds_expanded = edge_inds.view(-1, 1, 1).expand(-1, 1, n_line_points)
    line_subs = torch.cat((XY, edge_inds_expanded), dim=1)
    line_subs = line_subs.permute(
        0, 2, 1
    )  # (n_candidates, n_line_points, 3) -- last dim is [row, col, edge_ind]

    multiplier = torch.tensor([1, 1, 2], dtype=torch.int32).view(1, 1, 3)
    adder = torch.tensor([0, 0, 1], dtype=torch.int32).view(1, 1, 3)

    line_subs_first = line_subs * multiplier
    line_subs_second = line_subs * multiplier + adder
    line_subs = torch.stack(
        (line_subs_first, line_subs_second), dim=2
    )  # (n_candidates, n_line_points, 2, 3)
    # The last dim is [row, col, edge_ind], but for both PAF (x and y) edge channels.

    return line_subs


def get_paf_lines(
    pafs_sample: torch.Tensor,
    peaks_sample: torch.Tensor,
    edge_peak_inds: torch.Tensor,
    edge_inds: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> torch.Tensor:
    """Get the PAF values at the lines formed between all detected peaks in a sample.

    Args:
        pafs_sample: The PAFs for the sample as a `torch.Tensor` of shape
            `(height, width, 2 * n_edges)`.
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype `torch.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The PAF vectors at all of the line points as a `torch.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `torch.int32`.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        If only the subscripts are needed, use `make_line_subs()` to generate the lines
        without retrieving the PAF vector at the line points.

    See also: get_connection_candidates, make_line_subs, score_paf_lines
    """
    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points, pafs_stride
    )
    lines = pafs_sample[line_subs[..., 0], line_subs[..., 1], line_subs[..., 2]]
    return lines


def compute_distance_penalty(
    spatial_vec_lengths: torch.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the distance penalty component of the PAF line integral score.

    Args:
        spatial_vec_lengths: Euclidean distance between candidate source and
            destination points as a `torch.float32` tensor of any shape (typically
            `(n_candidates, 1)`).
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The distance penalty for each candidate as a `torch.float32` tensor of the same
        shape as `spatial_vec_lengths`.

        The penalty will be 0 (when below the threshold) and -1 as the distance
        approaches infinity. This is then scaled by the `dist_penalty_weight`.

    Notes:
        The penalty is computed from the distances scaled by the max length:

        ```
        if distance <= max_edge_length:
            penalty = 0
        else:
            penalty = (max_edge_length / distance) - 1
        ```

        For example, if the max length is 10 and the distance is 20, then the penalty
        will be: `(10 / 20) - 1 == 0.5 - 1 == -0.5`.

    See also: score_paf_lines
    """
    penalty = torch.clamp((max_edge_length / spatial_vec_lengths) - 1, max=0)
    return penalty * dist_penalty_weight


def score_paf_lines(
    paf_lines_sample: torch.Tensor,
    peaks_sample: torch.Tensor,
    edge_peak_inds_sample: torch.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the connectivity score for each PAF line in a sample.

    Args:
        paf_lines_sample: The PAF vectors evaluated at the lines formed between
            candidate conncetions as a `torch.Tensor` of shape
            `(n_candidates, n_line_points, 2, 3)` dtype `torch.int32`. This can be
            generated by `get_paf_lines()`.
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale.
        edge_peak_inds_sample: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype
            `torch.int32` with the indices of the peaks that form the source and
            destination of each candidate connection. This indexes into the input
            `peaks_sample`. Can be generated using `get_connection_candidates()`.
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The line scores as a `torch.Tensor` of shape `(n_candidates,)` and dtype
        `torch.float32`. Each score value is the average dot product between the PAFs and
        the normalized displacement vector between source and destination peaks.

        Scores range from roughly -1.5 to 1.0, where larger values indicate a better
        connectivity score for the candidate. Values can be larger or smaller due to
        prediction error.

    Notes:
        This function operates on a single sample (frame). For batches of multiple
        frames, use `score_paf_lines_batch()`.

    See also: get_paf_lines, score_paf_lines_batch, compute_distance_penalty
    """
    # Pull out points using advanced indexing
    src_peaks = peaks_sample[edge_peak_inds_sample[:, 0]]  # (n_candidates, 2)
    dst_peaks = peaks_sample[edge_peak_inds_sample[:, 1]]  # (n_candidates, 2)

    # Compute normalized spatial displacement vector
    spatial_vecs = dst_peaks - src_peaks
    spatial_vec_lengths = torch.norm(
        spatial_vecs, dim=1, keepdim=True
    )  # (n_candidates, 1)
    spatial_vecs = spatial_vecs / spatial_vec_lengths  # Normalize

    # Compute similarity scores
    spatial_vecs = spatial_vecs.unsqueeze(2)  # Add dimension for matrix multiplication
    line_scores = torch.squeeze(
        paf_lines_sample @ spatial_vecs, dim=-1
    )  # (n_candidates, n_line_points)

    # Compute distance penalties
    dist_penalties = torch.squeeze(
        compute_distance_penalty(
            spatial_vec_lengths,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        ),
        dim=1,
    )  # (n_candidates,)

    # Compute average line scores with distance penalty.
    mean_line_scores = torch.mean(line_scores, dim=1)
    penalized_line_scores = mean_line_scores + dist_penalties  # (n_candidates,)

    return penalized_line_scores


import torch

def get_connection_candidates(
    peak_channel_inds_sample: torch.Tensor, skeleton_edges: torch.Tensor, n_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the indices of all the possible connections formed by the detected peaks.

    Args:
        peak_channel_inds_sample: The channel indices of the peaks found in a sample.
            This is a `torch.Tensor` of shape `(n_peaks,)` and dtype `torch.int32` that is
            used to represent a detected peak by its channel/node index in the skeleton.
        skeleton_edges: The indices of the nodes that form the skeleton graph as a
            `torch.Tensor` of shape `(n_edges, 2)` and dtype `torch.int32` where each row
            corresponds to the source and destination node indices.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.

    Returns:
        A tuple of `(edge_inds, edge_peak_inds)`.

        `edge_inds` is a `torch.Tensor` of shape `(n_candidates,)` indicating the indices
        of the edge that each of the candidate connections belongs to.

        `edge_peak_inds` is a `torch.Tensor` of shape `(n_candidates, 2)` with the indices
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

def make_line_subs(
    peaks_sample: torch.Tensor,
    edge_peak_inds: torch.Tensor,
    edge_inds: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> torch.Tensor:
    """Create the lines between candidate connections for evaluating the PAFs.

    Args:
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype `torch.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The line subscripts as a `torch.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `torch.int32`.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        The subscripts are interpolated via nearest neighbor, so multiple fractional
        coordinates may map on to the same pixel if the line is short.

    See also: get_connection_candidates
    """
    src_peaks = torch.index_select(peaks_sample, 0, edge_peak_inds[:, 0])
    dst_peaks = torch.index_select(peaks_sample, 0, edge_peak_inds[:, 1])
    n_candidates = torch.tensor(src_peaks.shape[0])

    linspace_values = torch.linspace(0, 1, n_line_points, dtype=torch.float32)
    linspace_values = linspace_values.repeat(n_candidates, 1).view(
        n_candidates, n_line_points, 1
    )
    XY = (
        src_peaks.view(n_candidates, 1, 2)
        + (dst_peaks - src_peaks).view(n_candidates, 1, 2) * linspace_values
    )
    XY = XY.transpose(1, 2)
    XY = (
        (XY / pafs_stride).round().int()
    )  # (n_candidates, 2, n_line_points)  # dim 1 is [x, y]
    XY = XY[:, [1, 0], :]  # dim 1 is [row, col]

    edge_inds_expanded = edge_inds.view(-1, 1, 1).expand(-1, 1, n_line_points)
    line_subs = torch.cat((XY, edge_inds_expanded), dim=1)
    line_subs = line_subs.permute(
        0, 2, 1
    )  # (n_candidates, n_line_points, 3) -- last dim is [row, col, edge_ind]

    multiplier = torch.tensor([1, 1, 2], dtype=torch.int32).view(1, 1, 3)
    adder = torch.tensor([0, 0, 1], dtype=torch.int32).view(1, 1, 3)

    line_subs_first = line_subs * multiplier
    line_subs_second = line_subs * multiplier + adder
    line_subs = torch.stack(
        (line_subs_first, line_subs_second), dim=2
    )  # (n_candidates, n_line_points, 2, 3)
    # The last dim is [row, col, edge_ind], but for both PAF (x and y) edge channels.

    return line_subs


def get_paf_lines(
    pafs_sample: torch.Tensor,
    peaks_sample: torch.Tensor,
    edge_peak_inds: torch.Tensor,
    edge_inds: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
) -> torch.Tensor:
    """Get the PAF values at the lines formed between all detected peaks in a sample.

    Args:
        pafs_sample: The PAFs for the sample as a `torch.Tensor` of shape
            `(height, width, 2 * n_edges)`.
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale (they will be scaled by the `pafs_stride`).
        edge_peak_inds: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype `torch.int32`
            with the indices of the peaks that form the source and destination of each
            candidate connection. This indexes into the input `peaks_sample`. Can be
            generated using `get_connection_candidates()`.
        edge_inds: A `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to. Can be generated using `get_connection_candidates()`.
        n_line_points: The number of points to interpolate between source and
            destination peaks in each connection candidate as a scalar integer. Values
            ranging from 5 to 10 are pretty reasonable.
        pafs_stride: The stride (1/scale) of the PAFs that these lines will need to
            index into relative to the image. Coordinates in `peaks_sample` will be
            divided by this value to adjust the indexing into the PAFs tensor.

    Returns:
        The PAF vectors at all of the line points as a `torch.Tensor` of shape
        `(n_candidates, n_line_points, 2, 3)` and dtype `torch.int32`.

        The last dimension of the line subscripts correspond to the full
        `[row, col, channel]` subscripts of each element of the lines. Axis -2 contains
        the same `[row, col]` for each line but `channel` is adjusted to match the
        channels in the PAFs tensor.

    Notes:
        If only the subscripts are needed, use `make_line_subs()` to generate the lines
        without retrieving the PAF vector at the line points.

    See also: get_connection_candidates, make_line_subs, score_paf_lines
    """
    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points, pafs_stride
    )
    lines = pafs_sample[line_subs[..., 0], line_subs[..., 1], line_subs[..., 2]]
    return lines

def compute_distance_penalty(
    spatial_vec_lengths: torch.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the distance penalty component of the PAF line integral score.

    Args:
        spatial_vec_lengths: Euclidean distance between candidate source and
            destination points as a `torch.float32` tensor of any shape (typically
            `(n_candidates, 1)`).
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The distance penalty for each candidate as a `torch.float32` tensor of the same
        shape as `spatial_vec_lengths`.

        The penalty will be 0 (when below the threshold) and -1 as the distance
        approaches infinity. This is then scaled by the `dist_penalty_weight`.

    Notes:
        The penalty is computed from the distances scaled by the max length:

        ```
        if distance <= max_edge_length:
            penalty = 0
        else:
            penalty = (max_edge_length / distance) - 1
        ```

        For example, if the max length is 10 and the distance is 20, then the penalty
        will be: `(10 / 20) - 1 == 0.5 - 1 == -0.5`.

    See also: score_paf_lines
    """
    penalty = torch.clamp((max_edge_length / spatial_vec_lengths) - 1, max=0)
    return penalty * dist_penalty_weight


def score_paf_lines(
    paf_lines_sample: torch.Tensor,
    peaks_sample: torch.Tensor,
    edge_peak_inds_sample: torch.Tensor,
    max_edge_length: float,
    dist_penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the connectivity score for each PAF line in a sample.

    Args:
        paf_lines_sample: The PAF vectors evaluated at the lines formed between
            candidate conncetions as a `torch.Tensor` of shape
            `(n_candidates, n_line_points, 2, 3)` dtype `torch.int32`. This can be
            generated by `get_paf_lines()`.
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale.
        edge_peak_inds_sample: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype
            `torch.int32` with the indices of the peaks that form the source and
            destination of each candidate connection. This indexes into the input
            `peaks_sample`. Can be generated using `get_connection_candidates()`.
        max_edge_length: Maximum length expected for any connection as a scalar `float`
            in units of pixels (corresponding to `peaks_sample`). Scores of lines
            longer than this will be penalized. Useful for ignoring spurious
            connections that are far apart in space.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.

    Returns:
        The line scores as a `torch.Tensor` of shape `(n_candidates,)` and dtype
        `torch.float32`. Each score value is the average dot product between the PAFs and
        the normalized displacement vector between source and destination peaks.

        Scores range from roughly -1.5 to 1.0, where larger values indicate a better
        connectivity score for the candidate. Values can be larger or smaller due to
        prediction error.

    Notes:
        This function operates on a single sample (frame). For batches of multiple
        frames, use `score_paf_lines_batch()`.

    See also: get_paf_lines, score_paf_lines_batch, compute_distance_penalty
    """
    # Pull out points using advanced indexing
    src_peaks = peaks_sample[edge_peak_inds_sample[:, 0]]  # (n_candidates, 2)
    dst_peaks = peaks_sample[edge_peak_inds_sample[:, 1]]  # (n_candidates, 2)

    # Compute normalized spatial displacement vector
    spatial_vecs = dst_peaks - src_peaks
    spatial_vec_lengths = torch.norm(
        spatial_vecs, dim=1, keepdim=True
    )  # (n_candidates, 1)
    spatial_vecs = spatial_vecs / spatial_vec_lengths  # Normalize

    # Compute similarity scores
    spatial_vecs = spatial_vecs.unsqueeze(2)  # Add dimension for matrix multiplication
    line_scores = torch.squeeze(
        paf_lines_sample @ spatial_vecs, dim=-1
    )  # (n_candidates, n_line_points)

    # Compute distance penalties
    dist_penalties = torch.squeeze(
        compute_distance_penalty(
            spatial_vec_lengths,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        ),
        dim=1,
    )  # (n_candidates,)

    # Compute average line scores with distance penalty.
    mean_line_scores = torch.mean(line_scores, dim=1)
    penalized_line_scores = mean_line_scores + dist_penalties  # (n_candidates,)

    return penalized_line_scores

def score_paf_lines_batch(
    pafs: torch.Tensor,
    peaks: torch.Tensor,
    peak_channel_inds: List[torch.Tensor],
    skeleton_edges: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
    max_edge_length_ratio: float,
    dist_penalty_weight: float,
    n_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Processes a batch of images to score the Part Affinity Fields (PAFs) lines 
    formed between connection candidates for each sample.

    This function loops over each sample in the batch and applies the process of 
    getting connection candidates, retrieving PAF vectors for each line, and 
    computing the connectivity score for each candidate based on the PAF lines.

    Args:
        pafs: A tensor of shape `(n_samples, height, width, 2 * n_edges)` 
            containing the part affinity fields for each sample in the batch.
        peaks: A tensor of shape `(n_samples, n_peaks, 2)` containing the 
            (x, y) coordinates of the detected peaks for each sample.
        peak_channel_inds: A list of tensors of shape `(n_samples, (n_peaks))` indicating 
            the channel (node) index that each peak corresponds to.
        skeleton_edges: A tensor of shape `(n_edges, 2)` indicating the indices 
            of the nodes that form each edge of the skeleton.
        n_line_points: The number of points used to interpolate between source 
            and destination peaks in each connection candidate.
        pafs_stride: The stride (1/scale) of the PAFs relative to the image scale.
        max_edge_length_ratio: The maximum expected length of a connected pair 
            of points relative to the image dimensions.
        dist_penalty_weight: A coefficient to scale the weight of the distance 
            penalty applied to the score of each line.
        n_nodes: The total number of nodes in the skeleton.

    Returns:
        A tuple containing three lists for each sample in the batch:
            - A list of tensors of shape `(n_samples, (n_connections,))` indicating the indices 
              of the edges that each connection corresponds to.
            - A list of tensors of shape `(n_samples, (n_connections, 2))` containing the indices 
              of the source and destination peaks forming each connection.
            - A list of tensors of shape `(n_samples, (n_connections,))` containing the scores 
              for each connection based on the PAFs.
    """
    max_edge_length = max_edge_length_ratio * max(pafs.shape[2], pafs.shape[3]) * pafs_stride

    n_samples = pafs.shape[0]
    batch_edge_inds = []
    batch_edge_peak_inds = []
    batch_line_scores = []

    for sample in range(n_samples):
        pafs_sample = pafs[sample]
        peaks_sample = peaks[sample]
        peak_channel_inds_sample = peak_channel_inds[sample]

        edge_inds_sample, edge_peak_inds_sample = get_connection_candidates(
            peak_channel_inds_sample, skeleton_edges, n_nodes
        )
        paf_lines_sample = get_paf_lines(
            pafs_sample,
            peaks_sample,
            edge_peak_inds_sample,
            edge_inds_sample,
            n_line_points,
            pafs_stride,
        )
        line_scores_sample = score_paf_lines(
            paf_lines_sample,
            peaks_sample,
            edge_peak_inds_sample,
            max_edge_length,
            dist_penalty_weight=dist_penalty_weight,
        )

        # Appending as lists to maintain the nested structure
        batch_edge_inds.append(edge_inds_sample)
        batch_edge_peak_inds.append(edge_peak_inds_sample)
        batch_line_scores.append(line_scores_sample)

    return batch_edge_inds, batch_edge_peak_inds, batch_line_scores


def match_candidates_sample(
    edge_inds_sample: torch.Tensor,
    edge_peak_inds_sample: torch.Tensor,
    line_scores_sample: torch.Tensor,
    n_edges: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match candidate connections for a sample based on PAF scores.

    Args:
        edge_inds_sample: A `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.int32`
            indicating the indices of the edge that each of the candidate connections
            belongs to for the sample. Can be generated using
            `get_connection_candidates()`.
        edge_peak_inds_sample: A `torch.Tensor` of shape `(n_candidates, 2)` and dtype
            `torch.int32` with the indices of the peaks that form the source and
            destination of each candidate connection. Can be generated using
            `get_connection_candidates()`.
        line_scores_sample: Scores for each candidate connection in the sample as a
            `torch.Tensor` of shape `(n_candidates,)` and dtype `torch.float32`. Can be
            generated using `score_paf_lines()`.
        n_edges: A scalar `int` denoting the number of edges in the skeleton.

    Returns:
        The connection peaks for each edge matched based on score as tuple of
        `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

        `match_edge_inds`: Indices of the skeleton edge that each connection corresponds
        to as a `torch.Tensor` of shape `(n_connections,)` and dtype `torch.int32`.

        `match_src_peak_inds`: Indices of the source peaks that form each connection
        as a `torch.Tensor` of shape `(n_connections,)` and dtype `torch.int32`. Important:
        These indices correspond to the edge-grouped peaks, not the set of all peaks in
        the sample.

        `match_dst_peak_inds`: Indices of the destination peaks that form each
        connection as a `torch.Tensor` of shape `(n_connections,)` and dtype `torch.int32`.
        Important: These indices correspond to the edge-grouped peaks, not the set of
        all peaks in the sample.

        `match_line_scores`: PAF line scores of the matched connections as a `torch.Tensor`
        of shape `(n_connections,)` and dtype `torch.float32`.

    Notes:
        The matching is performed using the Munkres algorithm implemented in
        `scipy.optimize.linear_sum_assignment()`.

    See also: match_candidates_batch
    """
    match_edge_inds = []
    match_src_peak_inds = []
    match_dst_peak_inds = []
    match_line_scores = []

    for k in range(n_edges):
        is_edge_k = (edge_inds_sample == k).nonzero(as_tuple=True)[0]
        edge_peak_inds_k = edge_peak_inds_sample[is_edge_k]
        line_scores_k = line_scores_sample[is_edge_k]

        # Get the unique peak indices
        src_peak_inds_k = torch.unique(edge_peak_inds_k[:, 0])
        dst_peak_inds_k = torch.unique(edge_peak_inds_k[:, 1])

        n_src = src_peak_inds_k.size(0)
        n_dst = dst_peak_inds_k.size(0)

        # Initialize cost matrix with infinite cost
        cost_matrix = torch.full((n_src, n_dst), np.inf)

        # Update cost matrix with line scores
        for i, src_ind in enumerate(src_peak_inds_k):
            for j, dst_ind in enumerate(dst_peak_inds_k):
                mask = (edge_peak_inds_k[:, 0] == src_ind) & (edge_peak_inds_k[:, 1] == dst_ind)
                if mask.any():
                    cost_matrix[i, j] = -line_scores_k[mask].item()  # Flip sign for maximization

        # Convert cost matrix to numpy for use with scipy's linear_sum_assignment
        cost_matrix_np = cost_matrix.numpy()

        # Match
        match_src_inds, match_dst_inds = linear_sum_assignment(cost_matrix_np)

        # Pull out matched scores from the numpy cost matrix
        match_line_scores_k = -cost_matrix_np[match_src_inds, match_dst_inds]  # Flip sign back

        # Get the peak indices for the matched points (these index into peaks_sample)
        # These index into the edge-grouped peaks
        match_src_peak_inds_k = match_src_inds
        match_dst_peak_inds_k = match_dst_inds

        # Save
        match_edge_inds.append(torch.full((match_src_peak_inds_k.size,), k, dtype=torch.int32))
        match_src_peak_inds.append(torch.tensor(match_src_peak_inds_k, dtype=torch.int32))
        match_dst_peak_inds.append(torch.tensor(match_dst_peak_inds_k, dtype=torch.int32))
        match_line_scores.append(torch.tensor(match_line_scores_k, dtype=torch.float32))

    # Convert lists to tensors
    match_edge_inds = torch.cat(match_edge_inds)
    match_src_peak_inds = torch.cat(match_src_peak_inds)
    match_dst_peak_inds = torch.cat(match_dst_peak_inds)
    match_line_scores = torch.cat(match_line_scores)

    return match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores
