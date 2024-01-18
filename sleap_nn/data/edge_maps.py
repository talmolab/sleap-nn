"""Transformers for generating edge confidence maps and part affinity fields."""

import torch

from sleap_nn.data.utils import (
    expand_to_rank,
    make_grid_vectors,
    ensure_list,
    gaussian_pdf
)

def distance_to_edge(
    points: torch.Tensor, edge_source: torch.Tensor, edge_destination: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise distance between points and undirected edges.

    Args:
        points: Tensor of dtype torch.float32 of shape (d_0, ..., d_n, 2) where the last
            axis corresponds to x- and y-coordinates. Distances will be broadcast across
            all point dimensions.
        edge_source: Tensor of dtype torch.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype torch.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the source points of each
            edge.

    Returns:
        A tensor of dtype torch.float32 of shape (d_0, ..., d_n, n_edges) where the first
        axes correspond to the initial dimensions of `points`, and the last indicates
        the distance of each point to each edge.
    """
    # Ensure all points are at least rank 2.
    points = expand_to_rank(points, 2)
    edge_source = expand_to_rank(edge_source, 2)
    edge_destination = expand_to_rank(edge_destination, 2)

    # Compute number of point dimensions.
    n_pt_dims = points.dim() - 1

    # Direction vector.
    direction_vector = edge_destination - edge_source  # (n_edges, 2)

    # Edge length.
    edge_length = torch.maximum(
        direction_vector.square().sum(dim=1), torch.tensor(1.0)
    )  # (n_edges,)

    # Adjust query points relative to edge source point.
    source_relative_points = torch.unsqueeze(points, dim=-2) - expand_to_rank(
        edge_source, n_pt_dims + 2
    )  # (..., n_edges, 2)

    # Project points to edge line.
    line_projections = torch.sum(
        source_relative_points * expand_to_rank(direction_vector, n_pt_dims + 2), dim=3
    ) / expand_to_rank(
        edge_length, n_pt_dims + 1
    )  # (..., n_edges)

    # Crop to line segment.
    line_projections = torch.clamp(line_projections, min=0, max=1)

    # Compute distance from each point to the edge.
    distances = torch.sum(
        torch.square(
            (line_projections.unsqueeze(-1) * expand_to_rank(direction_vector, n_pt_dims + 2))
            - source_relative_points
        ),
        dim=-1
    )  # (..., n_edges)

    return distances
