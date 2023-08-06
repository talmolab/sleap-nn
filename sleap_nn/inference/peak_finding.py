"""Peak finding for inference."""
import torch
import numpy as np
from typing import Tuple, Optional
from sleap_nn.data.instance_cropping import make_centered_bboxes, normalize_bboxes

def find_global_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the global maximum for each sample and channel.

    Args:
        cms: Tensor of shape (samples, channels, height, width).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will be replaced with NaNs.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    # Find the maximum values and their indices along the height and width axes.
    max_values, max_indices_y = torch.max(cms, dim=2, keepdim=True)
    max_values, max_indices_x = torch.max(max_values, dim=3, keepdim=True)

    max_indices_x = max_indices_x.squeeze(dim=(2, 3))  # (samples, channels)
    max_indices_y = max_indices_y.max(dim=3).values  # (samples, channels, 1)
    max_values = max_values.squeeze(-1).squeeze(-1)  # (samples, channels)
    peak_points = torch.cat([max_indices_x.unsqueeze(-1), max_indices_y], dim=-1)

    # Create masks for values below the threshold.
    below_threshold_mask = max_values < threshold

    # Replace values below the threshold with NaN.
    max_values[below_threshold_mask] = float("nan")

    return peak_points, max_values

