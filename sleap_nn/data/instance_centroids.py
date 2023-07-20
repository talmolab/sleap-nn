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
        instances: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
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
    pts_min = torch.min(torch.where(torch.isnan(points), np.inf, points), dim=-2).values
    pts_max = torch.max(
        torch.where(torch.isnan(points), -np.inf, points), dim=-2
    ).values

    return (pts_max + pts_min) * 0.5


class InstanceCentroidFinder(dp.IterDataPipe):
    """Datapipe for finding centroids of instances.

    This DataPipe will produce examples that have been containing a 'centroid' key.

    Attributes:
        source_dp: the previous `DataPipe` with samples that contain an `instance`
    """

    def __init__(
        self,
        source_dp: dp.IterDataPipe,
    ):
        """Initialize InstanceCentroidFinder with the source `DataPipe."""
        self.source_dp = source_dp

    def __iter__(self):
        """Add 'centroid' key to sample."""

        def find_centroids(sample):
            mid_pts = find_points_bbox_midpoint(sample["instance"])
            sample["centroid"] = mid_pts

            return sample

        for sample in self.source_dp:
            find_centroids(sample)
            yield sample
