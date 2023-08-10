"""Peak finding for inference."""
import torch
import numpy as np
from typing import Tuple, Optional
from sleap_nn.data.instance_cropping import make_centered_bboxes
from kornia.geometry.transform import crop_and_resize


def crop_bboxes(
    images: torch.Tensor, bboxes: torch.Tensor, sample_inds: torch.Tensor
) -> torch.Tensor:
    """Crop bounding boxes from a batch of images.

    Args:
        images: Tensor of shape (samples, channels, height, width) of a batch of images.
        bboxes: Tensor of shape (n_bboxes, 4, 2) and dtype torch.float32, where n_bboxes is the number of centroids, and the second dimension
            represents the four corner points of the bounding boxes, each with x and y coordinates.
            The order of the corners follows a clockwise arrangement: top-left, top-right,
            bottom-right, and bottom-left. This can be generated from centroids using `make_centered_bboxes`.
        sample_inds: Tensor of shape (n_bboxes,) specifying which samples each bounding
            box should be cropped from.

    Returns:
        A tensor of shape (n_bboxes, crop_height, crop_width, channels) of the same
        dtype as the input image. The crop size is inferred from the bounding box
        coordinates.

    Notes:
        This function expects bounding boxes with coordinates at the centers of the
        pixels in the box limits. Technically, the box will span (x1 - 0.5, x2 + 0.5)
        and (y1 - 0.5, y2 + 0.5).

        For example, a 3x3 patch centered at (1, 1) would be specified by
        (y1, x1, y2, x2) = (0, 0, 2, 2). This would be exactly equivalent to indexing
        the image with `image[:, :, 0:3, 0:3]`.

    See also: `make_centered_bboxes`
    """
    # Compute bounding box size to use for crops.
    height = bboxes[0, 3, 1] - bboxes[0, 0, 1]
    width = bboxes[0, 1, 0] - bboxes[0, 0, 0]
    box_size = tuple(np.round((height + 1, width + 1)).astype(np.int32))

    # Crop.
    crops = crop_and_resize(
        images[sample_inds],  # (n_boxes, channels, height, width)
        boxes=bboxes,
        size=box_size,
    )

    # Cast back to original dtype and return.
    crops = crops.to(images.dtype)
    return crops


def integral_regression(
    cms: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute regression by integrating over the confidence maps on a grid.

    Args:
        cms: Confidence maps with shape (samples, channels, height, width).
        xv: X grid vector torch.float32 of grid coordinates to sample.
        yv: Y grid vector torch.float32 of grid coordinates to sample.

    Returns:
        A tuple of (x_hat, y_hat) with the regressed x- and y-coordinates for each
        channel of the confidence maps.

        x_hat and y_hat are of shape (samples, channels)
    """
    # Compute normalizing factor.
    z = torch.sum(cms, dim=[2, 3])

    # Regress to expectation.
    x_hat = torch.sum(xv.view(1, 1, 1, -1) * cms, dim=[2, 3]) / z
    y_hat = torch.sum(yv.view(1, 1, -1, 1) * cms, dim=[2, 3]) / z

    return x_hat, y_hat


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
    peak_points = torch.cat([max_indices_x.unsqueeze(-1), max_indices_y], dim=-1).to(
        torch.float32
    )

    # Create masks for values below the threshold.
    below_threshold_mask = max_values < threshold

    # Replace values below the threshold with NaN.
    peak_points[below_threshold_mask] = float("nan")

    return peak_points, max_values


def find_global_peaks(
    cms: torch.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find global peaks with optional refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, channels, height, width).
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression.
        integral_patch_size: Size of patches to crop around each rough peak as an
            integer scalar.

    Returns:
        A tuple of (peak_points, peak_vals).

        peak_points: float32 tensor of shape (samples, channels, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (samples, channels) containing the values at
        the peak points.
    """
    # Find grid aligned peaks.
    rough_peaks, peak_vals = find_global_peaks_rough(
        cms, threshold=threshold
    )  # (samples, channels, 2)

    # Return early if not refining or no rough peaks found.
    if refinement is None or torch.isnan(rough_peaks).all():
        return rough_peaks, peak_vals

    if refinement == "integral":
        crop_size = integral_patch_size
    else:
        return rough_peaks, peak_vals

    # Flatten samples and channels to (n_peaks, 2).
    samples = cms.size(0)
    channels = cms.size(1)
    rough_peaks = rough_peaks.view(samples * channels, 2)

    # Keep only peaks that are not NaNs.
    valid_idx = torch.where(~torch.isnan(rough_peaks[:, 0]))[0].squeeze(0)
    valid_peaks = rough_peaks[valid_idx]

    # Make bounding boxes for cropping around peaks.
    bboxes = make_centered_bboxes(
        valid_peaks, box_height=crop_size, box_width=crop_size
    )

    # Crop patch around each grid-aligned peak.
    cms = torch.reshape(
        cms,
        [samples * channels, 1, cms.size(2), cms.size(3)],
    )
    cm_crops = crop_bboxes(cms, bboxes, valid_idx)

    # Compute offsets via integral regression on a local patch.
    if refinement == "integral":
        gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
        dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
        offsets = torch.cat([dx_hat, dy_hat], dim=1)

    # Apply offsets.
    refined_peaks = rough_peaks.clone()
    refined_peaks[valid_idx] += offsets

    # Reshape to (samples, channels, 2).
    refined_peaks = refined_peaks.reshape(samples, channels, 2)

    return refined_peaks, peak_vals
