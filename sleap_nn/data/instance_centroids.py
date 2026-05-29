"""Handle calculation of instance centroids."""

from typing import Optional
import torch


def find_points_mean(points: torch.Tensor) -> torch.Tensor:
    """Find the mean position of a set of points, ignoring NaNs.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.

    Returns:
        The NaN-ignoring mean across the ``n_points`` axis. Output shape ``(..., 2)``
        (rank reduced by 1). When *every* point in a slot is NaN, the corresponding
        output row is NaN.
    """
    mask = ~torch.isnan(points)
    counts = mask.any(dim=-1).sum(dim=-1, keepdim=True).clamp(min=1).to(points.dtype)
    safe = torch.where(mask, points, torch.zeros_like(points))
    means = safe.sum(dim=-2) / counts
    all_nan = (~mask.any(dim=-1)).all(dim=-1, keepdim=True)
    return torch.where(all_nan, torch.full_like(means, float("nan")), means)


def find_points_bbox_midpoint(points: torch.Tensor) -> torch.Tensor:
    """Find the midpoint of the bounding box of a set of points.

    Retained as a utility for callers that explicitly want bbox-midpoint behavior.
    The canonical anchor fallback used by :func:`generate_centroids` is
    :func:`find_points_mean` (mean of visible nodes) — see that function for the
    project-wide convention.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.

    Returns:
        The midpoints between the bounds of each set of points. The output will be of
        shape (..., 2), reducing the rank of the input by 1. NaNs will be ignored in the
        calculation.

    Notes:
        The midpoint is calculated as:
            xy_mid = xy_min + ((xy_max - xy_min) / 2)
                   = ((2 * xy_min) / 2) + ((xy_max - xy_min) / 2)
                   = (2 * xy_min + xy_max - xy_min) / 2
                   = (xy_min + xy_max) / 2
    """
    pts_min = torch.min(
        torch.where(torch.isnan(points), torch.inf, points), dim=-2
    ).values
    pts_max = torch.max(
        torch.where(torch.isnan(points), -torch.inf, points), dim=-2
    ).values

    return (pts_max + pts_min) * 0.5


def generate_centroids(
    points: torch.Tensor, anchor_ind: Optional[int] = None
) -> torch.Tensor:
    """Return centroids, falling back to the mean of visible nodes.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_nodes, 2),
            i.e., rank >= 2.
        anchor_ind: The index of the node to use as the anchor for the centroid.
            If not provided, or if the anchor node is NaN (not visible) for a given
            instance, the centroid falls back to the NaN-ignoring mean of all visible
            nodes for that instance.

    Returns:
        The centroids of the instances. The output will be of shape (..., 2),
        reducing the rank of the input by 1. NaNs will be ignored in the calculation.

    Note:
        The missing/occluded-anchor fallback is the mean of visible nodes
        (``find_points_mean``). Pre-#530 this was the bounding-box midpoint
        (``find_points_bbox_midpoint``). The two modes are tracked for a future
        revisit in https://github.com/talmolab/sleap-nn/issues/586 — keep this
        consistent with the centroid target generated during training.
    """
    if anchor_ind is not None:
        centroids = points[..., anchor_ind, :].clone()
    else:
        centroids = torch.full_like(points[..., 0, :], torch.nan).clone()

    missing_anchors = torch.isnan(centroids).any(dim=-1)
    if missing_anchors.any():
        centroids[missing_anchors] = find_points_mean(points[missing_anchors])

    return centroids  # (..., n_instances, 2)
