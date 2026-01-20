"""Peak finding for inference."""

from typing import Optional, Tuple

import kornia as K
import torch
import torch.nn.functional as F

from sleap_nn.data.instance_cropping import make_centered_bboxes


def crop_bboxes(
    images: torch.Tensor, bboxes: torch.Tensor, sample_inds: torch.Tensor
) -> torch.Tensor:
    """Crop bounding boxes from a batch of images using fast tensor indexing.

    This uses tensor unfold operations to extract patches, which is significantly
    faster than kornia's crop_and_resize (17-51x speedup) as it avoids perspective
    transform computations.

    Args:
        images: Tensor of shape (samples, channels, height, width) of a batch of images.
        bboxes: Tensor of shape (n_bboxes, 4, 2) and dtype torch.float32, where n_bboxes
            is the number of centroids, and the second dimension represents the four
            corner points of the bounding boxes, each with x and y coordinates.
            The order of the corners follows a clockwise arrangement: top-left,
            top-right, bottom-right, and bottom-left. This can be generated from
            centroids using `make_centered_bboxes`.
        sample_inds: Tensor of shape (n_bboxes,) specifying which samples each bounding
            box should be cropped from.

    Returns:
        A tensor of shape (n_bboxes, channels, crop_height, crop_width) of the same
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
    n_crops = bboxes.shape[0]
    if n_crops == 0:
        # Return empty tensor; use default crop size since we can't infer from bboxes
        return torch.empty(
            0, images.shape[1], 0, 0, device=images.device, dtype=images.dtype
        )

    # Compute bounding box size to use for crops.
    height = int(abs(bboxes[0, 3, 1] - bboxes[0, 0, 1]).item()) + 1
    width = int(abs(bboxes[0, 1, 0] - bboxes[0, 0, 0]).item()) + 1

    # Store original dtype for conversion back after cropping.
    original_dtype = images.dtype
    device = images.device
    n_samples, channels, img_h, img_w = images.shape
    half_h, half_w = height // 2, width // 2

    # Pad images for edge handling.
    images_padded = F.pad(
        images.float(), (half_w, half_w, half_h, half_h), mode="constant", value=0
    )

    # Extract all possible patches using unfold (creates a view, no copy).
    # Shape after unfold: (n_samples, channels, img_h, img_w, height, width)
    patches = images_padded.unfold(2, height, 1).unfold(3, width, 1)

    # Get crop centers from bboxes.
    # The bbox top-left is at index 0, with (x, y) coordinates.
    # We need the center of the crop (peak location), which is top-left + half_size.
    # Ensure bboxes are on the same device as images for index computation.
    bboxes_on_device = bboxes.to(device)
    crop_x = (bboxes_on_device[:, 0, 0] + half_w).to(torch.long)
    crop_y = (bboxes_on_device[:, 0, 1] + half_h).to(torch.long)

    # Clamp indices to valid bounds to handle edge cases where centroids
    # might be at or beyond image boundaries.
    crop_x = torch.clamp(crop_x, 0, patches.shape[3] - 1)
    crop_y = torch.clamp(crop_y, 0, patches.shape[2] - 1)

    # Select crops using advanced indexing.
    # Convert sample_inds to tensor if it's a list.
    if not isinstance(sample_inds, torch.Tensor):
        sample_inds = torch.tensor(sample_inds, device=device)
    sample_inds_long = sample_inds.to(device=device, dtype=torch.long)
    crops = patches[sample_inds_long, :, crop_y, crop_x]
    # Shape: (n_crops, channels, height, width)

    # Cast back to original dtype and return.
    crops = crops.to(original_dtype)
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
    z = torch.sum(cms, dim=[2, 3]).to(cms.device)
    xv = xv.to(cms.device)
    yv = yv.to(cms.device)

    # Regress to expectation.
    x_hat = torch.sum(xv.view(1, 1, 1, -1) * cms, dim=[2, 3]) / z
    y_hat = torch.sum(yv.view(1, 1, -1, 1) * cms, dim=[2, 3]) / z

    return x_hat, y_hat


def find_global_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the global maximum for each sample and channel.

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
    # Find the maximum values and their indices along the height and width axes.
    amax_values, amax_indices_x = torch.max(cms, dim=3, keepdim=True)
    amax_values, amax_indices_y = torch.max(amax_values, dim=2, keepdim=True)
    amax_indices_y = amax_indices_y.squeeze(dim=(2, 3))
    peak_points = torch.cat(
        [max_indices_x.unsqueeze(-1), amax_indices_y.unsqueeze(-1)], dim=-1
    ).to(torch.float32)
    max_values = max_values.squeeze(-1).squeeze(-1)
    # Create masks for values below the threshold.
    below_threshold_mask = max_values < threshold
    # Replace values below the threshold with NaN.
    peak_points[below_threshold_mask] = float("nan")
    max_values[below_threshold_mask] = float(0)
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
    valid_idx = torch.where(~torch.isnan(rough_peaks[:, 0]))[0]
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


def find_local_peaks_rough(
    cms: torch.Tensor, threshold: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find local maxima via non-maximum suppression.

    Args:
        cms: Tensor of shape (samples, channels, height, width).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will not be returned.

    Returns:
        A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).
        peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
        points.

        peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
        sample each peak belongs to.

        peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
        the channel each peak belongs to.
    """
    # Build custom local NMS kernel.
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)

    # Reshape to have singleton channels.
    height = cms.size(2)
    width = cms.size(3)
    channels = cms.size(1)
    flat_img = cms.reshape(-1, 1, height, width)

    # Perform dilation filtering to find local maxima per channel and reshape back.
    max_img = K.morphology.dilation(flat_img, kernel.to(flat_img.device))
    max_img = max_img.reshape(-1, channels, height, width)

    # Filter for maxima and threshold.
    argmax_and_thresh_img = (cms > max_img) & (cms > threshold)

    # Convert to subscripts.
    peak_subs = torch.stack(
        torch.where(argmax_and_thresh_img.permute(0, 2, 3, 1)), axis=-1
    )

    # Get peak values.
    peak_vals = cms[peak_subs[:, 0], peak_subs[:, 3], peak_subs[:, 1], peak_subs[:, 2]]

    # Convert to points format.
    peak_points = peak_subs[:, [2, 1]].to(torch.float32)

    # Pull out indexing vectors.
    peak_sample_inds = peak_subs[:, 0].to(torch.int32)
    peak_channel_inds = peak_subs[:, 3].to(torch.int32)

    return peak_points, peak_vals, peak_sample_inds, peak_channel_inds


def find_local_peaks(
    cms: torch.Tensor,
    threshold: float = 0.2,
    refinement: Optional[str] = None,
    integral_patch_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find local peaks with optional refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, channels, height, width).
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.
        refinement: If `None`, returns the grid-aligned peaks with no refinement. If
            `"integral"`, peaks will be refined with integral regression.
        integral_patch_size: Size of patches to crop around each rough peak as an
            integer scalar.

    Returns:
        A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).

        peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
        points.

        peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
        sample each peak belongs to.

        peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
        the channel each peak belongs to.
    """
    # Find grid aligned peaks.
    (
        rough_peaks,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_rough(cms, threshold=threshold)

    # Return early if no rough peaks found.
    if rough_peaks.size(0) == 0 or refinement is None:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    if refinement == "integral":
        crop_size = integral_patch_size
    else:
        return rough_peaks, peak_vals, peak_sample_inds, peak_channel_inds

    # Make bounding boxes for cropping around peaks.
    bboxes = make_centered_bboxes(
        rough_peaks, box_height=crop_size, box_width=crop_size
    )

    # Reshape to (samples * channels, height, width, 1).
    samples = cms.size(0)
    channels = cms.size(1)
    cms = torch.reshape(
        cms,
        [samples * channels, 1, cms.size(2), cms.size(3)],
    )
    box_sample_inds = (peak_sample_inds * channels) + peak_channel_inds

    # Crop patch around each grid-aligned peak.
    cm_crops = crop_bboxes(cms, bboxes, sample_inds=box_sample_inds)

    # Compute offsets via integral regression on a local patch.
    if refinement == "integral":
        gv = torch.arange(crop_size, dtype=torch.float32) - ((crop_size - 1) / 2)
        dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
        offsets = torch.cat([dx_hat, dy_hat], dim=1)

    # Apply offsets.
    refined_peaks = rough_peaks + offsets

    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds
