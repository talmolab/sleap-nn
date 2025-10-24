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
import attrs
from typing import Tuple, List, Dict, Union, Text
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from omegaconf import OmegaConf
from sleap_nn.inference.utils import interp1d


@attrs.define(auto_attribs=True, frozen=True)
class PeakID:
    """Indices to uniquely identify a single peak.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        node_ind: Index of the node type (channel) of the peak.
        peak_ind: Index of the peak within its node type.
    """

    node_ind: int
    peak_ind: int


@attrs.define(auto_attribs=True, frozen=True)
class EdgeType:
    """Indices to uniquely identify a single edge type.

    This is a convenience named tuple for use in the matching pipeline.

    Attributes:
        src_node_ind: Index of the source node type within the skeleton edges.
        dst_node_ind: Index of the destination node type within the skeleton edges.
    """

    src_node_ind: int
    dst_node_ind: int


@attrs.define(auto_attribs=True)
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
    pafs_hw: tuple,
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
        pafs_hw: Tuple (height, width) with the dimension of PAFs tensor.

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
    n_candidates = torch.tensor(src_peaks.shape[0], device=peaks_sample.device)

    X = torch.cat(
        (src_peaks[:, 0].unsqueeze(dim=-1), dst_peaks[:, 0].unsqueeze(dim=-1)), dim=-1
    ).to(torch.float32)
    Y = torch.cat(
        (src_peaks[:, 1].unsqueeze(dim=-1), dst_peaks[:, 1].unsqueeze(dim=-1)), dim=-1
    ).to(torch.float32)
    samples = torch.tensor([0, 1], device=X.device).repeat(n_candidates, 1)
    samples_new = torch.linspace(0, 1, steps=n_line_points, device=X.device).repeat(
        n_candidates, 1
    )

    X = interp1d(samples, X, samples_new).unsqueeze(
        dim=1
    )  # (n_candidates, 1, n_line_points)
    Y = interp1d(samples, Y, samples_new).unsqueeze(
        dim=1
    )  # (n_candidates, 1, n_line_points)
    XY = torch.concat([X, Y], dim=1)

    XY = (
        (XY / pafs_stride).round().int()
    )  # (n_candidates, 2, n_line_points)  # dim 1 is [x, y]
    XY = XY[:, [1, 0], :]  # dim 1 is [row, col]

    # clip coordinates for size of pafs tensor.
    height, width = pafs_hw
    XY[:, 0] = torch.clip(XY[:, 0], min=0, max=height - 1)
    XY[:, 1] = torch.clip(XY[:, 1], min=0, max=width - 1)

    edge_inds_expanded = (
        edge_inds.view(-1, 1, 1)
        .expand(-1, 1, n_line_points)
        .to(device=peaks_sample.device)
    )
    line_subs = torch.cat((XY, edge_inds_expanded), dim=1)
    line_subs = line_subs.permute(
        0, 2, 1
    )  # (n_candidates, n_line_points, 3) -- last dim is [row, col, edge_ind]

    multiplier = torch.tensor(
        [1, 1, 2], dtype=torch.int32, device=line_subs.device
    ).view(1, 1, 3)
    adder = torch.tensor([0, 0, 1], dtype=torch.int32, device=line_subs.device).view(
        1, 1, 3
    )

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
    pafs_hw = pafs_sample.shape[:2]
    line_subs = make_line_subs(
        peaks_sample, edge_peak_inds, edge_inds, n_line_points, pafs_stride, pafs_hw
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
            candidate connections as a `torch.Tensor` of shape
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
    peak_channel_inds: torch.Tensor,
    skeleton_edges: torch.Tensor,
    n_line_points: int,
    pafs_stride: int,
    max_edge_length_ratio: float,
    dist_penalty_weight: float,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a batch of images to score the Part Affinity Fields (PAFs) lines formed between connection candidates for each sample.

    This function loops over each sample in the batch and applies the process of
    getting connection candidates, retrieving PAF vectors for each line, and
    computing the connectivity score for each candidate based on the PAF lines.

    Args:
        pafs: A tensor of shape `(n_samples, height, width, 2 * n_edges)`
            containing the part affinity fields for each sample in the batch.
        peaks: A list of tensors (torch nested tensors) of shape `(n_samples, (n_peaks), 2)` containing the
            (x, y) coordinates of the detected peaks for each sample.
        peak_channel_inds: A list of tensors (torch nested tensors) of shape `(n_samples, (n_peaks))` indicating
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
            - A list of tensors (torch nested tensors) of shape `(n_samples, (n_connections,))` indicating the indices
              of the edges that each connection corresponds to.
            - A list of tensors (torch nested tensors) of shape `(n_samples, (n_connections, 2))` containing the indices
              of the source and destination peaks forming each connection.
            - A list of tensors (torch nested tensors) of shape `(n_samples, (n_connections,))` containing the scores
              for each connection based on the PAFs.
    """
    max_edge_length = (
        max_edge_length_ratio
        * max(pafs.shape[-1], pafs.shape[-2], pafs.shape[-3])
        * pafs_stride
    )

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

        # Appending as lists to maintain the nested structure.
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
    # Move tensors to CPU once to avoid repeated device<->host synchronizations
    edge_inds_sample = edge_inds_sample.detach().cpu()
    edge_peak_inds_sample = edge_peak_inds_sample.detach().cpu()
    line_scores_sample = line_scores_sample.detach().cpu()

    match_edge_inds = []
    match_src_peak_inds = []
    match_dst_peak_inds = []
    match_line_scores = []

    for k in range(n_edges):
        is_edge_k = (edge_inds_sample == k).nonzero(as_tuple=True)[0]
        edge_peak_inds_k = edge_peak_inds_sample[is_edge_k]
        line_scores_k = line_scores_sample[is_edge_k]

        # Get the unique peak indices.
        src_peak_inds_k = torch.unique(edge_peak_inds_k[:, 0])
        dst_peak_inds_k = torch.unique(edge_peak_inds_k[:, 1])

        n_src = src_peak_inds_k.size(0)
        n_dst = dst_peak_inds_k.size(0)

        # Initialize cost matrix with infinite cost.
        cost_matrix = torch.full((n_src, n_dst), np.inf)

        # Update cost matrix with line scores.
        for i, src_ind in enumerate(src_peak_inds_k):
            for j, dst_ind in enumerate(dst_peak_inds_k):
                mask = (edge_peak_inds_k[:, 0] == src_ind) & (
                    edge_peak_inds_k[:, 1] == dst_ind
                )
                if mask.any():
                    # `line_scores_k` is already on CPU; `.item()` does not trigger
                    # a device synchronization and matches the original behaviour.
                    cost_matrix[i, j] = -line_scores_k[
                        mask
                    ].item()  # Flip sign for maximization.

        # Convert cost matrix to numpy for use with scipy's linear_sum_assignment.
        cost_matrix_np = cost_matrix.numpy()
        cost_matrix_np[np.isnan(cost_matrix_np)] = np.inf

        # Match.
        match_src_inds, match_dst_inds = linear_sum_assignment(cost_matrix_np)

        # Pull out matched scores from the numpy cost matrix.
        match_line_scores_k = -cost_matrix_np[
            match_src_inds, match_dst_inds
        ]  # Flip sign back.

        # Get the peak indices for the matched points (these index into peaks_sample).
        # These index into the edge-grouped peaks.
        match_src_peak_inds_k = match_src_inds
        match_dst_peak_inds_k = match_dst_inds

        # Save.
        match_edge_inds.append(
            torch.full((match_src_peak_inds_k.size,), k, dtype=torch.int32)
        )
        match_src_peak_inds.append(
            torch.tensor(match_src_peak_inds_k, dtype=torch.int32)
        )
        match_dst_peak_inds.append(
            torch.tensor(match_dst_peak_inds_k, dtype=torch.int32)
        )
        match_line_scores.append(torch.tensor(match_line_scores_k, dtype=torch.float32))

    # Convert lists to tensors.
    match_edge_inds = torch.cat(match_edge_inds)
    match_src_peak_inds = torch.cat(match_src_peak_inds)
    match_dst_peak_inds = torch.cat(match_dst_peak_inds)
    match_line_scores = torch.cat(match_line_scores)

    return match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores


def match_candidates_batch(
    edge_inds: torch.Tensor,
    edge_peak_inds: torch.Tensor,
    line_scores: torch.Tensor,
    n_edges: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match candidate connections for a batch based on PAF scores.

    Args:
        edge_inds: Sample-grouped edge indices as a torch nested `torch.Tensor`s of shape
            `(n_samples, (n_candidates))` and dtype `torch.int32` indicating the indices
            of the edge that each of the candidate connections belongs to. Can be
            generated using `score_paf_lines_batch()`.
        edge_peak_inds: Sample-grouped indices of the peaks that form the source and
            destination of each candidate connection as a torch nested `torch.Tensor`s of shape
            `(n_samples, (n_candidates), 2)` and dtype `torch.int32`. Can be generated
            using `score_paf_lines_batch()`.
        line_scores: Sample-grouped scores for each candidate connection as a
            torch nested `torch.Tensor`s of shape `(n_samples, (n_candidates))` and dtype
            `torch.float32`. Can be generated using `score_paf_lines_batch()`.
        n_edges: A scalar `int` denoting the number of edges in the skeleton.

    Returns:
        The connection peaks for each edge matched based on score as tuple of
        `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

        `match_edge_inds`: Sample-grouped indices of the skeleton edge for each
        connection as a torch nested `torch.Tensor`s of shape `(n_samples, (n_connections))` and
        dtype `torch.int32`.

        `match_src_peak_inds`: Sample-grouped indices of the source peaks that form each
        connection as a torch nested `torch.Tensor`s of shape `(n_samples, (n_connections))` and
        dtype `torch.int32`. Important: These indices correspond to the edge-grouped peaks,
        not the set of all peaks in the sample.

        `match_dst_peak_inds`: Sample-grouped indices of the destination peaks that form
        each connection as a torch nested `torch.Tensor`s of shape `(n_samples, (n_connections))`
        and dtype `torch.int32`. Important: These indices correspond to the edge-grouped
        peaks, not the set of all peaks in the sample.

        `match_line_scores`: Sample-grouped PAF line scores of the matched connections
        as a torch nested `torch.Tensor`s of shape `(n_samples, (n_connections))` and dtype
        `torch.float32`.

    Notes:
        The matching is performed using the Munkres algorithm implemented in
        `scipy.optimize.linear_sum_assignment()`.

    See also: match_candidates_sample, score_paf_lines_batch, group_instances_batch
    """
    match_sample_inds = []
    match_edge_inds = []
    match_src_peak_inds = []
    match_dst_peak_inds = []
    match_line_scores = []

    for sample in range(len(edge_inds)):
        edge_inds_sample = edge_inds[sample]
        edge_peak_inds_sample = edge_peak_inds[sample]
        line_scores_sample = line_scores[sample]

        matched_sample = match_candidates_sample(
            edge_inds_sample, edge_peak_inds_sample, line_scores_sample, n_edges
        )

        (
            match_edge_inds_sample,
            match_src_peak_inds_sample,
            match_dst_peak_inds_sample,
            match_line_scores_sample,
        ) = matched_sample

        match_sample_inds.append(
            torch.full_like(match_edge_inds_sample, sample, dtype=torch.int32)
        )
        match_edge_inds.append(match_edge_inds_sample)
        match_src_peak_inds.append(match_src_peak_inds_sample)
        match_dst_peak_inds.append(match_dst_peak_inds_sample)
        match_line_scores.append(match_line_scores_sample)

    return match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores


def assign_connections_to_instances(
    connections: Dict[EdgeType, List[EdgeConnection]],
    min_instance_peaks: Union[int, float] = 0,
    n_nodes: int = None,
) -> Dict[PeakID, int]:
    """Assign connected edges to instances via greedy graph partitioning.

    Args:
        connections: A dict that maps EdgeType to a list of EdgeConnections found
            through connection scoring. This can be generated by the
            filter_connection_candidates function.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a float in the
            range (0., 1.] is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an integer is provided, this is the
            absolute minimum number of peaks.
        n_nodes: Total node type count. Used to convert min_instance_peaks to an
            absolute number when a fraction is specified. If not provided, the node
            count is inferred from the unique node inds in connections.

    Returns:
        instance_assignments: A dict mapping PeakID to a unique instance ID specified
        as an integer.

        A PeakID is a tuple of (node_type_ind, peak_ind), where the peak_ind is the
        index or identifier specified in a EdgeConnection as a src_peak_ind or
        dst_peak_ind.

    Note:
        Instance IDs are not necessarily consecutive since some instances may be
        filtered out during the partitioning or filtering.

        This function expects connections from a single sample/frame!
    """
    # Grouping table that maps PeakID(node_ind, peak_ind) to an instance_id.
    instance_assignments = dict()

    # Loop through edge types.
    for edge_type, edge_connections in connections.items():
        # Loop through connections for the current edge.
        for connection in edge_connections:
            # Notation: specific peaks are identified by (node_ind, peak_ind).
            src_id = PeakID(edge_type.src_node_ind, connection.src_peak_ind)
            dst_id = PeakID(edge_type.dst_node_ind, connection.dst_peak_ind)

            # Get instance assignments for the connection peaks.
            src_instance = instance_assignments.get(src_id, None)
            dst_instance = instance_assignments.get(dst_id, None)

            if src_instance is None and dst_instance is None:
                # Case 1: Neither peak is assigned to an instance yet. We'll create a
                # new instance to hold both.
                new_instance = max(instance_assignments.values(), default=-1) + 1
                instance_assignments[src_id] = new_instance
                instance_assignments[dst_id] = new_instance

            elif src_instance is not None and dst_instance is None:
                # Case 2: The source peak is assigned already, but not the destination
                # peak. We'll assign the destination peak to the same instance as the
                # source.
                instance_assignments[dst_id] = src_instance

            elif src_instance is not None and dst_instance is not None:
                # Case 3: Both peaks have been assigned. We'll update the destination
                # peak to be a part of the source peak instance.
                instance_assignments[dst_id] = src_instance

                # We'll also check if they form disconnected subgraphs, in which case
                # we'll merge them by assigning all peaks belonging to the destination
                # peak's instance to the source peak's instance.
                src_instance_nodes = set(
                    peak_id.node_ind
                    for peak_id, instance in instance_assignments.items()
                    if instance == src_instance
                )
                dst_instance_nodes = set(
                    peak_id.node_ind
                    for peak_id, instance in instance_assignments.items()
                    if instance == dst_instance
                )

                if len(src_instance_nodes.intersection(dst_instance_nodes)) == 0:
                    for peak_id in instance_assignments:
                        if instance_assignments[peak_id] == dst_instance:
                            instance_assignments[peak_id] = src_instance

    if min_instance_peaks > 0:
        if isinstance(min_instance_peaks, float):
            if n_nodes is None:
                # Infer number of nodes if not specified.
                all_node_types = set()
                for edge_type in connections:
                    all_node_types.add(edge_type.src_node_ind)
                    all_node_types.add(edge_type.dst_node_ind)
                n_nodes = len(all_node_types)

            # Calculate minimum threshold.
            min_instance_peaks = int(min_instance_peaks * n_nodes)

        # Compute instance peak counts.
        instance_ids, instance_peak_counts = np.unique(
            list(instance_assignments.values()), return_counts=True
        )
        instance_peak_counts = {
            instance: peaks_count
            for instance, peaks_count in zip(instance_ids, instance_peak_counts)
        }

        # Filter out small instances.
        instance_assignments = {
            peak_id: instance
            for peak_id, instance in instance_assignments.items()
            if instance_peak_counts[instance] >= min_instance_peaks
        }

    return instance_assignments


def make_predicted_instances(
    peaks: np.array,
    peak_scores: np.array,
    connections: List[EdgeConnection],
    instance_assignments: Dict[PeakID, int],
) -> Tuple[np.array, np.array, np.array]:
    """Group peaks by assignments and accumulate scores.

    Args:
        peaks: Node-grouped peaks
        peak_scores: Node-grouped peak scores
        connections: `EdgeConnection`s grouped by edge type
        instance_assignments: `PeakID` to instance ID mapping

    Returns:
        Tuple of (predicted_instances, predicted_peak_scores, predicted_instance_scores)

        predicted_instances: (n_instances, n_nodes, 2) array
        predicted_peak_scores: (n_instances, n_nodes) array
        predicted_instance_scores: (n_instances,) array
    """
    # Ensure instance IDs are contiguous.
    instance_ids, instance_inds = np.unique(
        list(instance_assignments.values()), return_inverse=True
    )
    for peak_id, instance_ind in zip(instance_assignments.keys(), instance_inds):
        instance_assignments[peak_id] = instance_ind
    n_instances = len(instance_ids)

    # Compute instance scores as the sum of all edge scores.
    predicted_instance_scores = np.full((n_instances,), 0.0, dtype="float32")

    for edge_type, edge_connections in connections.items():
        # Loop over all connections for this edge type.
        for edge_connection in edge_connections:
            # Look up the source peak.
            src_peak_id = PeakID(
                node_ind=edge_type.src_node_ind, peak_ind=edge_connection.src_peak_ind
            )
            if src_peak_id in instance_assignments:
                # Add to the total instance score.
                instance_ind = instance_assignments[src_peak_id]
                predicted_instance_scores[instance_ind] += edge_connection.score

                # Sanity check: both peaks in the edge should have been assigned to the
                # same instance.
                dst_peak_id = PeakID(
                    node_ind=edge_type.dst_node_ind,
                    peak_ind=edge_connection.dst_peak_ind,
                )
                assert instance_ind == instance_assignments[dst_peak_id]

    # Fill out instances and peak scores.
    n_nodes = len(peaks)
    predicted_instances = np.full((n_instances, n_nodes, 2), np.nan, dtype="float32")
    predicted_peak_scores = np.full((n_instances, n_nodes), np.nan, dtype="float32")
    for peak_id, instance_ind in instance_assignments.items():
        predicted_instances[instance_ind, peak_id.node_ind, :] = peaks[
            peak_id.node_ind
        ][peak_id.peak_ind]
        predicted_peak_scores[instance_ind, peak_id.node_ind] = peak_scores[
            peak_id.node_ind
        ][peak_id.peak_ind]

    return predicted_instances, predicted_peak_scores, predicted_instance_scores


def toposort_edges(edge_types: List[EdgeType]) -> Tuple[int]:
    """Find a topological ordering for a list of edge types.

    Args:
        edge_types: A list of `EdgeType` instances describing a skeleton.

    Returns:
        A tuple of indices specifying the topological order that the edge types should
        be accessed in during instance assembly (`assign_connections_to_instances`).

        This is important to ensure that instances are assembled starting at the root
        of the skeleton and moving down.

    See also: assign_connections_to_instances
    """
    edges = [
        (edge_type.src_node_ind, edge_type.dst_node_ind) for edge_type in edge_types
    ]
    dg = nx.DiGraph(edges)
    root_ind = next(nx.topological_sort(dg))
    sorted_edges = nx.bfs_edges(dg, root_ind)
    sorted_edge_inds = tuple([edges.index(edge) for edge in sorted_edges])
    return sorted_edge_inds


def group_instances_sample(
    peaks_sample: torch.Tensor,
    peak_scores_sample: torch.Tensor,
    peak_channel_inds_sample: torch.Tensor,
    match_edge_inds_sample: torch.Tensor,
    match_src_peak_inds_sample: torch.Tensor,
    match_dst_peak_inds_sample: torch.Tensor,
    match_line_scores_sample: torch.Tensor,
    n_nodes: int,
    sorted_edge_inds: Tuple[int],
    edge_types: List[EdgeType],
    min_instance_peaks: int,
    min_line_scores: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group matched connections into full instances for a single sample.

    Args:
        peaks_sample: The detected peaks in a sample as a `torch.Tensor` of shape
            `(n_peaks, 2)` and dtype `torch.float32`. These should be `(x, y)` coordinates
            of each peak in the image scale.
        peak_scores_sample: The scores of the detected peaks in a sample as a
            `torch.Tensor` of shape `(n_peaks,)` and dtype `torch.float32`.
        peak_channel_inds_sample: The indices of the channel (node) that each detected
            peak is associated with as a `torch.Tensor` of shape `(n_peaks,)` and dtype
            `torch.int32`.
        match_edge_inds_sample: Indices of the skeleton edge that each connection
            corresponds to as a `torch.Tensor` of shape `(n_connections,)` and dtype
            `torch.int32`. This can be generated by `match_candidates_sample()`.
        match_src_peak_inds_sample: Indices of the source peaks that form each
            connection as a `torch.Tensor` of shape `(n_connections,)` and dtype
            `torch.int32`. Important: These indices correspond to the edge-grouped peaks,
            not the set of all peaks in the sample. This can be generated by
            `match_candidates_sample()`.
        match_dst_peak_inds_sample: Indices of the destination peaks that form each
            connection as a `torch.Tensor` of shape `(n_connections,)` and dtype
            `torch.int32`. Important: These indices correspond to the edge-grouped peaks,
            not the set of all peaks in the sample. This can be generated by
            `match_candidates_sample()`.
        match_line_scores_sample: PAF line scores of the matched connections as a
            `torch.Tensor` of shape `(n_connections,)` and dtype `torch.float32`. This can be
            generated by `match_candidates_sample()`.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).
        edge_types: A list of `EdgeType`s associated with the skeleton.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a `float` in the
            range `(0., 1.]` is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an `int` is provided, this is the
            absolute minimum number of peaks.
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs.

    Returns:
        A tuple of arrays with the grouped instances:

        `predicted_instances`: The grouped coordinates for each instance as an array of
        shape `(n_instances, n_nodes, 2)` and dtype `float32`. Missing peaks are
        represented by `np.nan`s.

        `predicted_peak_scores`: The confidence map values for each peak as an array of
        `(n_instances, n_nodes)` and dtype `float32`.

        `predicted_instance_scores`: The grouping score for each instance as an array of
        shape `(n_instances,)` and dtype `float32`.
    """
    # Convert PyTorch tensors to NumPy arrays for non-tensor computations
    if isinstance(peaks_sample, torch.Tensor):
        peaks_sample = peaks_sample.cpu().numpy()
        peak_scores_sample = peak_scores_sample.cpu().numpy()
        peak_channel_inds_sample = peak_channel_inds_sample.cpu().numpy()
        match_edge_inds_sample = match_edge_inds_sample.cpu().numpy()
        match_src_peak_inds_sample = match_src_peak_inds_sample.cpu().numpy()
        match_dst_peak_inds_sample = match_dst_peak_inds_sample.cpu().numpy()
        match_line_scores_sample = match_line_scores_sample.cpu().numpy()

    # Filter out low scoring matches.
    is_valid_match = match_line_scores_sample >= min_line_scores
    match_edge_inds_sample = match_edge_inds_sample[is_valid_match]
    match_src_peak_inds_sample = match_src_peak_inds_sample[is_valid_match]
    match_dst_peak_inds_sample = match_dst_peak_inds_sample[is_valid_match]
    match_line_scores_sample = match_line_scores_sample[is_valid_match]

    # Group peaks by channel.
    peaks = []
    peak_scores = []
    for i in range(n_nodes):
        in_channel = peak_channel_inds_sample == i
        peaks.append(peaks_sample[in_channel])
        peak_scores.append(peak_scores_sample[in_channel])

    # Group connection data by edge in sorted order.
    # Note: This step is crucial since the instance assembly depends on the ordering
    # of the edges.
    connections = {}
    for edge_ind in sorted_edge_inds:
        in_edge = match_edge_inds_sample == edge_ind
        edge_type = edge_types[edge_ind]

        src_peak_inds = match_src_peak_inds_sample[in_edge]
        dst_peak_inds = match_dst_peak_inds_sample[in_edge]
        line_scores = match_line_scores_sample[in_edge]

        connections[edge_type] = [
            EdgeConnection(src, dst, score)
            for src, dst, score in zip(src_peak_inds, dst_peak_inds, line_scores)
        ]

    # Bipartite graph partitioning to group connections into instances.
    instance_assignments = assign_connections_to_instances(
        connections,
        min_instance_peaks=min_instance_peaks,
        n_nodes=n_nodes,
    )

    # Gather the data by instance.
    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = make_predicted_instances(peaks, peak_scores, connections, instance_assignments)

    return predicted_instances, predicted_peak_scores, predicted_instance_scores


def group_instances_batch(
    peaks: torch.Tensor,
    peak_vals: torch.Tensor,
    peak_channel_inds: torch.Tensor,
    match_edge_inds: torch.Tensor,
    match_src_peak_inds: torch.Tensor,
    match_dst_peak_inds: torch.Tensor,
    match_line_scores: torch.Tensor,
    n_nodes: int,
    sorted_edge_inds: Tuple[int],
    edge_types: List[EdgeType],
    min_instance_peaks: int,
    min_line_scores: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group matched connections into full instances for a batch.

    Args:
        peaks: The sample-grouped detected peaks in a batch as a torch nested `torch.Tensor` of
            shape `(n_samples, (n_peaks), 2)` and dtype `torch.float32`. These should be
            `(x, y)` coordinates of each peak in the image scale.
        peak_vals: The sample-grouped scores of the detected peaks in a batch as a
            torch nested `torch.Tensor` of shape `(n_samples, (n_peaks))` and dtype `torch.float32`.
        peak_channel_inds: The sample-grouped indices of the channel (node) that each
            detected peak is associated with as a torch nested `torch.Tensor` of shape
            `(n_samples, (n_peaks))` and dtype `torch.int32`.
        match_edge_inds: Sample-grouped indices of the skeleton edge that each
            connection corresponds to as a torch nested `torch.Tensor` of shape
            `(n_samples, (n_connections))` and dtype `torch.int32`. This can be generated
            by `match_candidates_batch()`.
        match_src_peak_inds: Sample-grouped indices of the source peaks that form each
            connection as a torch nested `torch.Tensor` of shape `(n_samples, (n_connections))`
            and dtype `torch.int32`. Important: These indices correspond to the
            edge-grouped peaks, not the set of all peaks in each sample. This can be
            generated by `match_candidates_batch()`.
        match_dst_peak_inds: Sample-grouped indices of the destination peaks that form
            each connection as a torch nested `torch.Tensor` of shape
            `(n_samples, (n_connections))` and dtype `torch.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample. This can be generated by `match_candidates_batch()`.
        match_line_scores: Sample-grouped PAF line scores of the matched connections as
            a torch nested `torch.Tensor` of shape `(n_samples, (n_connections))` and dtype
            `torch.float32`. This can be generated by `match_candidates_batch()`.
        n_nodes: The total number of nodes in the skeleton as a scalar integer.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).
        edge_types: A torch nested `EdgeType`s associated with the skeleton.
        min_instance_peaks: If this is greater than 0, grouped instances with fewer
            assigned peaks than this threshold will be excluded. If a `float` in the
            range `(0., 1.]` is provided, this is interpreted as a fraction of the total
            number of nodes in the skeleton. If an `int` is provided, this is the
            absolute minimum number of peaks.
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs.

    Returns:
        A tuple of `torch.Tensor` with the grouped instances for the whole batch grouped by
        sample:

        `predicted_instances`: The sample- and instance-grouped coordinates for each
        instance as a torch nested `torch.Tensor` of shape `(n_samples, (n_instances), n_nodes, 2)`
        and dtype `torch.float32`. Missing peaks are represented by `NaN`s.

        `predicted_peak_scores`: The sample- and instance-grouped confidence map values
        for each peak as a torch nested `torch.Tensor` of shape `(n_samples, (n_instances), n_nodes)` and dtype
        `torch.float32`.

        `predicted_instance_scores`: The sample-grouped instance grouping score for each
        instance as a torch nested `torch.Tensor` of shape `(n_samples, (n_instances))` and dtype
        `torch.float32`.

    See also: match_candidates_batch, group_instances_sample
    """
    n_samples = len(peaks)
    predicted_instances_batch = []
    predicted_peak_scores_batch = []
    predicted_instance_scores_batch = []

    for sample in range(n_samples):
        (
            predicted_instances_sample,
            predicted_peak_scores_sample,
            predicted_instance_scores_sample,
        ) = group_instances_sample(
            peaks[sample],
            peak_vals[sample],
            peak_channel_inds[sample],
            match_edge_inds[sample],
            match_src_peak_inds[sample],
            match_dst_peak_inds[sample],
            match_line_scores[sample],
            n_nodes,
            sorted_edge_inds,
            edge_types,
            min_instance_peaks,
            min_line_scores,
        )

        predicted_instances_batch.append(torch.tensor(predicted_instances_sample))
        predicted_peak_scores_batch.append(torch.tensor(predicted_peak_scores_sample))
        predicted_instance_scores_batch.append(
            torch.tensor(predicted_instance_scores_sample)
        )

    return (
        predicted_instances_batch,
        predicted_peak_scores_batch,
        predicted_instance_scores_batch,
    )


@attrs.define
class PAFScorer:
    """Scoring pipeline based on part affinity fields.

    This class facilitates grouping of predicted peaks based on PAFs. It holds a set of
    common parameters that are used across different steps of the pipeline.

    Attributes:
        part_names: List of string node names in the skeleton.
        edges: List of (src_node, dst_node) names in the skeleton.
        pafs_stride: Output stride of the part affinity fields. This will be used to
            adjust the peak coordinates from full image to PAF subscripts.
        max_edge_length_ratio: The maximum expected length of a connected pair of points
            as a fraction of the image size. Candidate connections longer than this
            length will be penalized during matching.
        dist_penalty_weight: A coefficient to scale weight of the distance penalty as
            a scalar float. Set to values greater than 1.0 to enforce the distance
            penalty more strictly.
        n_points: Number of points to sample along the line integral.
        min_instance_peaks: Minimum number of peaks the instance should have to be
            considered a real instance. Instances with fewer peaks than this will be
            discarded (useful for filtering spurious detections).
        min_line_scores: Minimum line score (between -1 and 1) required to form a match
            between candidate point pairs. Useful for rejecting spurious detections when
            there are no better ones.
        edge_inds: The edges of the skeleton defined as a list of (source, destination)
            tuples of node indices. This is created automatically on initialization.
        edge_types: A list of `EdgeType` instances representing the edges of the
            skeleton. This is created automatically on initialization.
        n_nodes: The number of nodes in the skeleton as a scalar `int`. This is created
            automatically on initialization.
        n_edges: The number of edges in the skeleton as a scalar `int`. This is created
            automatically on initialization.
        sorted_edge_inds: A tuple of indices specifying the topological order that the
            edge types should be accessed in during instance assembly
            (`assign_connections_to_instances`).

    Notes:
        This class provides high level APIs for grouping peaks into instances using
        PAFs.

        The algorithm has three steps:

            1. Find all candidate connections between peaks and compute their matching
            score based on the PAFs.

            2. Match candidate connections using the connectivity score such that no
            peak is used in two connections of the same type.

            3. Group matched connections into complete instances.

        In general, the output from a peak finder (such as multi-peak confidence map
        prediction network) can be passed into `PAFScorer.predict()` to get back
        complete instances.

        For finer control over the grouping pipeline steps, use the instance methods in
        this class or the lower level functions in `sleap_nn.paf_grouping`.
    """

    part_names: List[Text]
    edges: List[Tuple[Text, Text]]
    pafs_stride: int
    max_edge_length_ratio: float = 0.25
    dist_penalty_weight: float = 1.0
    n_points: int = 10
    min_instance_peaks: Union[int, float] = 0
    min_line_scores: float = 0.25

    edge_inds: List[Tuple[int, int]] = attr.ib(init=False)
    edge_types: List[EdgeType] = attr.ib(init=False)
    n_nodes: int = attr.ib(init=False)
    n_edges: int = attr.ib(init=False)
    sorted_edge_inds: Tuple[int] = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Cache some computed attributes on initialization."""
        self.edge_inds = [
            (self.part_names.index(src), self.part_names.index(dst))
            for (src, dst) in self.edges
        ]
        self.edge_types = [
            EdgeType(src_node, dst_node) for src_node, dst_node in self.edge_inds
        ]

        self.n_nodes = len(self.part_names)
        self.n_edges = len(self.edges)
        self.sorted_edge_inds = toposort_edges(self.edge_types)

    @classmethod
    def from_config(
        cls,
        config: OmegaConf,
        max_edge_length_ratio: float = 0.25,
        dist_penalty_weight: float = 1.0,
        n_points: int = 10,
        min_instance_peaks: Union[int, float] = 0,
        min_line_scores: float = 0.25,
    ) -> "PAFScorer":
        """Initialize the PAF scorer from a `MultiInstanceConfig` head config.

        Args:
            config: An `OmegaConf` instance.
            max_edge_length_ratio: The maximum expected length of a connected pair of
                points as a fraction of the image size. Candidate connections longer
                than this length will be penalized during matching.
            dist_penalty_weight: A coefficient to scale weight of the distance penalty
                as a scalar float. Set to values greater than 1.0 to enforce the
                distance penalty more strictly.
            min_edge_score: Minimum score required to classify a connection as correct.
            n_points: Number of points to sample along the line integral.
            min_instance_peaks: Minimum number of peaks the instance should have to be
                considered a real instance. Instances with fewer peaks than this will be
                discarded (useful for filtering spurious detections).
            min_line_scores: Minimum line score (between -1 and 1) required to form a
                match between candidate point pairs. Useful for rejecting spurious
                detections when there are no better ones.

        Returns:
            The initialized instance of `PAFScorer`.
        """
        return cls(
            part_names=config.confmaps.part_names,
            edges=config.pafs.edges,
            pafs_stride=config.pafs.output_stride,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
        )

    def score_paf_lines(
        self,
        pafs: torch.Tensor,
        peaks: torch.Tensor,
        peak_channel_inds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create and score PAF lines formed between connection candidates.

        Args:
            pafs: A nested torch tensor of shape `(n_samples, height, width, 2 * n_edges)`
                containing the part affinity fields for each sample in the batch.
            peaks: A nested torch tensor of shape `(n_samples, (n_peaks), 2)` containing the
                (x, y) coordinates of the detected peaks for each sample.
            peak_channel_inds: A nested torch tensor of shape `(n_samples, (n_peaks))` indicating
                the channel (node) index that each peak corresponds to.

        Returns:
            A tuple containing three lists for each sample in the batch:
                - A nested torch tensor of shape `(n_samples, (n_connections,))` indicating the indices
                of the edges that each connection corresponds to.
                - A nested torch tensor of shape `(n_samples, (n_connections, 2))` containing the indices
                of the source and destination peaks forming each connection.
                - A nested torch tensor of shape `(n_samples, (n_connections,))` containing the scores
                for each connection based on the PAFs.

        Notes:
            This is a convenience wrapper for the standalone `score_paf_lines_batch()`.

        See also: score_paf_lines_batch
        """
        return score_paf_lines_batch(
            pafs,
            peaks,
            peak_channel_inds,
            self.edge_inds,
            self.n_points,
            self.pafs_stride,
            self.max_edge_length_ratio,
            self.dist_penalty_weight,
            self.n_nodes,
        )

    def match_candidates(
        self,
        edge_inds: torch.Tensor,
        edge_peak_inds: torch.Tensor,
        line_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match candidate connections for a batch based on PAF scores.

        Args:
            edge_inds: Sample-grouped edge indices as a nested `torch.Tensor` of shape
                `(n_samples, (n_candidates))` and dtype `torch.int32` indicating the
                indices of the edge that each of the candidate connections belongs to.
                Can be generated using `PAFScorer.score_paf_lines()`.
            edge_peak_inds: Sample-grouped indices of the peaks that form the source and
                destination of each candidate connection as a nested `torch.Tensor` of shape
                `(n_samples, (n_candidates), 2)` and dtype `torch.int32`. Can be generated
                using `PAFScorer.score_paf_lines()`.
            line_scores: Sample-grouped scores for each candidate connection as a
                nested `torch.Tensor` of shape `(n_samples, (n_candidates))` and dtype
                `torch.float32`. Can be generated using `PAFScorer.score_paf_lines()`.

        Returns:
            The connection peaks for each edge matched based on score as tuple of
            `(match_edge_inds, match_src_peak_inds, match_dst_peak_inds, match_line_scores)`

            `match_edge_inds`: Sample-grouped indices of the skeleton edge for each
            connection as a nested `torch.Tensor` of shape `(n_samples, (n_connections))`
            and dtype `torch.int32`.

            `match_src_peak_inds`: Sample-grouped indices of the source peaks that form
            each connection as a nested `torch.Tensor` of shape
            `(n_samples, (n_connections))` and dtype `torch.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample.

            `match_dst_peak_inds`: Sample-grouped indices of the destination peaks that
            form each connection as a nested `torch.Tensor` of shape
            `(n_samples, (n_connections))` and dtype `torch.int32`. Important: These
            indices correspond to the edge-grouped peaks, not the set of all peaks in
            the sample.

            `match_line_scores`: Sample-grouped PAF line scores of the matched
            connections as a nested `torch.Tensor` of shape `(n_samples, (n_connections))`
            and dtype `torch.float32`.

        Notes:
            This is a convenience wrapper for the standalone `match_candidates_batch()`.

        See also: PAFScorer.score_paf_lines, match_candidates_batch
        """
        return match_candidates_batch(
            edge_inds, edge_peak_inds, line_scores, self.n_edges
        )

    def group_instances(
        self,
        peaks: torch.Tensor,
        peak_vals: torch.Tensor,
        peak_channel_inds: torch.Tensor,
        match_edge_inds: torch.Tensor,
        match_src_peak_inds: torch.Tensor,
        match_dst_peak_inds: torch.Tensor,
        match_line_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group matched connections into full instances for a batch.

        Args:
            peaks: The sample-grouped detected peaks in a batch as a nested tensor `torch.Tensor`
                of shape `(n_samples, (n_peaks), 2)` and dtype `torch.float32`. These
                should be `(x, y)` coordinates of each peak in the image scale.
            peak_vals: The sample-grouped scores of the detected peaks in a batch as a
                nested tensor `torch.Tensor` of shape `(n_samples, (n_peaks))` and dtype
                `torch.float32`.
            peak_channel_inds: The sample-grouped indices of the channel (node) that
                each detected peak is associated with as a nested tensor `torch.Tensor` of shape
                `(n_samples, (n_peaks))` and dtype `torch.int32`.
            match_edge_inds: Sample-grouped indices of the skeleton edge that each
                connection corresponds to as a nested tensor `torch.Tensor` of shape
                `(n_samples, (n_connections))` and dtype `torch.int32`. This can be
                generated by `PAFScorer.match_candidates()`.
            match_src_peak_inds: Sample-grouped indices of the source peaks that form
                each connection as a nested tensor `torch.Tensor` of shape
                `(n_samples, (n_connections))` and dtype `torch.int32`. Important: These
                indices correspond to the edge-grouped peaks, not the set of all peaks
                in each sample. This can be generated by `PAFScorer.match_candidates()`.
            match_dst_peak_inds: Sample-grouped indices of the destination peaks that
                form each connection as a nested tensor `torch.Tensor` of shape
                `(n_samples, (n_connections))` and dtype `torch.int32`. Important: These
                indices correspond to the edge-grouped peaks, not the set of all peaks
                in the sample. This can be generated by `PAFScorer.match_candidates()`.
            match_line_scores: Sample-grouped PAF line scores of the matched connections
                as a nested tensor `torch.Tensor` of shape `(n_samples, (n_connections))` and dtype
                `torch.float32`. This can be generated by `PAFScorer.match_candidates()`.

        Returns:
            A tuple of arrays with the grouped instances for the whole batch grouped by
            sample:

            `predicted_instances`: The sample- and instance-grouped coordinates for each
            instance as nested `torch.Tensor` of shape
            `(n_samples, (n_instances), n_nodes, 2)` and dtype `torch.float32`. Missing
            peaks are represented by `NaN`s.

            `predicted_peak_scores`: The sample- and instance-grouped confidence map
            values for each peak as an array of `(n_samples, (n_instances), n_nodes)`
            and dtype `torch.float32`.

            `predicted_instance_scores`: The sample-grouped instance grouping score for
            each instance as an array of shape `(n_samples, (n_instances))` and dtype
            `torch.float32`.

        Notes:
            This is a convenience wrapper for the standalone `group_instances_batch()`.

        See also: PAFScorer.match_candidates, group_instances_batch
        """
        return group_instances_batch(
            peaks,
            peak_vals,
            peak_channel_inds,
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
            self.n_nodes,
            self.sorted_edge_inds,
            self.edge_types,
            self.min_instance_peaks,
            min_line_scores=self.min_line_scores,
        )

    def predict(
        self,
        pafs: torch.Tensor,
        peaks: torch.Tensor,
        peak_vals: torch.Tensor,
        peak_channel_inds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group a batch of predicted peaks into full instance predictions using PAFs.

        Args:
            pafs: The batch of part affinity fields as a `torch.Tensor` of shape
                `(n_samples, height, width, 2 * n_edges)` and type `torch.float32`.
            peaks: The coordinates of the peaks grouped by sample as a nested `torch.Tensor`
                of shape `(n_samples, (n_peaks), 2)`.
            peak_vals: The sample-grouped scores of the detected peaks in a batch as a
                nested `torch.Tensor` of shape `(n_samples, (n_peaks))` and dtype
                `torch.float32`.
            peak_channel_inds: The channel (node) that each peak in `peaks` corresponds
                to as a nested `torch.Tensor` of shape `(n_samples, (n_peaks))` and dtype
                `torch.int32`.

        Returns:
            A tuple of arrays with the grouped instances for the whole batch grouped by
            sample:

            `predicted_instances`: The sample- and instance-grouped coordinates for each
            instance as nested `torch.Tensor` of shape
            `(n_samples, (n_instances), n_nodes, 2)` and dtype `torch.float32`. Missing
            peaks are represented by `NaN`s.

            `predicted_peak_scores`: The sample- and instance-grouped confidence map
            values for each peak as an array of `(n_samples, (n_instances), n_nodes)`
            and dtype `torch.float32`.

            `predicted_instance_scores`: The sample-grouped instance grouping score for
            each instance as an array of shape `(n_samples, (n_instances))` and dtype
            `torch.float32`.

        Notes:
            This is a high level API for grouping peaks into instances using PAFs.

            See the `PAFScorer` class documentation for more details on the algorithm.

        See Also:
            PAFScorer.score_paf_lines, PAFScorer.match_candidates,
            PAFScorer.group_instances
        """
        edge_inds, edge_peak_inds, line_scores = self.score_paf_lines(
            pafs, peaks, peak_channel_inds
        )
        (
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
        ) = self.match_candidates(edge_inds, edge_peak_inds, line_scores)
        (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
        ) = self.group_instances(
            peaks,
            peak_vals,
            peak_channel_inds,
            match_edge_inds,
            match_src_peak_inds,
            match_dst_peak_inds,
            match_line_scores,
        )
        return (
            predicted_instances,
            predicted_peak_scores,
            predicted_instance_scores,
            edge_inds,
            edge_peak_inds,
            line_scores,
        )
