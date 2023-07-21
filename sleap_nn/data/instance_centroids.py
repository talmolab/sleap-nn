"""Handle calculation of instance centroids."""
import torchdata.datapipes.iter as dp
import lightning.pytorch as pl
from typing import Optional
import sleap_io as sio
import numpy as np
import torch


def find_points_bbox_midpoint(points: torch.Tensor) -> torch.Tensor:
    """Find the midpoint of the bounding box of a set of points.

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


def find_centroids(
    points: torch.Tensor, anchor_ind: Optional[int] = None
) -> torch.Tensor:
    """Return centroids, falling back to bounding box midpoints.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.
        anchor_ind: The index of the node to use as the anchor for the centroid. If not
            provided or if not present in the instance, the midpoint of the bounding box
            is used instead.

    Returns:
        The centroids of the instances. The output will be of shape (..., 2), reducing
        the rank of the input by 1. NaNs will be ignored in the calculation.
    """
    if anchor_ind is not None:
        centroids = points[..., anchor_ind, :]
    else:
        centroids = torch.full_like(points[..., 0, :], torch.nan)

    missing_anchors = torch.isnan(centroids).any(dim=-1)
    if missing_anchors.any():
        centroids[missing_anchors] = find_points_bbox_midpoint(points[missing_anchors])

    return centroids


class InstanceCentroidFinder(dp.IterDataPipe):
    """Datapipe for finding centroids of instances.

    This DataPipe will produce examples that contain a 'centroids' key.

    Attributes:
        source_dp: The previous `DataPipe` with samples that contain an `instances` key.
        anchor_ind: The index of the node to use as the anchor for the centroid. If not
            provided or if not present in the instance, the midpoint of the bounding box
            is used instead.
    """

    def __init__(
        self,
        source_dp: dp.IterDataPipe,
        anchor_ind: Optional[int] = None,
    ):
        """Initialize InstanceCentroidFinder with the source `DataPipe."""
        self.source_dp = source_dp
        self.anchor_ind = anchor_ind

    def __iter__(self):
        """Add `"centroids"` key to sample."""
        for example in self.source_dp:
            example["centroids"] = find_centroids(
                example["instances"], anchor_ind=self.anchor_ind
            )
            yield example
