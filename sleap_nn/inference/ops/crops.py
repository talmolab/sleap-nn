"""Bbox creation + image cropping helpers used by inference.

Two responsibilities:

1. ``crop_bboxes`` — fast unfold-based patch extraction around a set of bboxes.
   Used by integral peak refinement and (later) top-down stage 2 crop pickup.
2. Re-export ``make_centered_bboxes`` from the data layer so callers in the
   inference path import from one place. The data layer keeps owning the
   function (the training pipeline uses it too).
"""

import torch
import torch.nn.functional as F

# Single source of truth for ``make_centered_bboxes`` is the data layer.
from sleap_nn.data.instance_cropping import make_centered_bboxes  # noqa: F401


def crop_bboxes(
    images: torch.Tensor, bboxes: torch.Tensor, sample_inds: torch.Tensor
) -> torch.Tensor:
    """Crop bounding boxes from a batch of images using fast tensor indexing.

    Uses tensor unfold operations to extract patches, which is significantly
    faster than ``kornia.crop_and_resize`` (17–51× speedup) because it avoids
    perspective transform computations.

    Args:
        images: Tensor of shape ``(samples, channels, height, width)``.
        bboxes: Tensor of shape ``(n_bboxes, 4, 2)`` and dtype ``float32``,
            where the second dim are the four corners of each bbox in
            top-left → top-right → bottom-right → bottom-left order. Build
            these with :func:`make_centered_bboxes`.
        sample_inds: Tensor of shape ``(n_bboxes,)`` saying which sample each
            bbox is cropped from.

    Returns:
        ``(n_bboxes, channels, crop_height, crop_width)`` of the same dtype
        as the input. The crop size is inferred from the first bbox.

    Notes:
        Bbox coordinates are at *pixel centers*; the box spans
        ``(x1 - 0.5, x2 + 0.5)`` and ``(y1 - 0.5, y2 + 0.5)``. A 3×3 patch
        centered at ``(1, 1)`` is specified by ``(y1, x1, y2, x2) = (0, 0, 2, 2)``
        and is equivalent to ``image[:, :, 0:3, 0:3]``.

    See Also:
        :func:`make_centered_bboxes`.
    """
    n_crops = bboxes.shape[0]
    if n_crops == 0:
        return torch.empty(
            0, images.shape[1], 0, 0, device=images.device, dtype=images.dtype
        )

    # Crop size is inferred from the first bbox.
    height = int(abs(bboxes[0, 3, 1] - bboxes[0, 0, 1]).item()) + 1
    width = int(abs(bboxes[0, 1, 0] - bboxes[0, 0, 0]).item()) + 1

    original_dtype = images.dtype
    device = images.device
    half_h, half_w = height // 2, width // 2

    images_padded = F.pad(
        images.float(), (half_w, half_w, half_h, half_h), mode="constant", value=0
    )

    # ``unfold`` creates a view (no copy). Shape:
    # (n_samples, channels, img_h, img_w, height, width)
    patches = images_padded.unfold(2, height, 1).unfold(3, width, 1)

    bboxes_on_device = bboxes.to(device)
    crop_x = (bboxes_on_device[:, 0, 0] + half_w).to(torch.long)
    crop_y = (bboxes_on_device[:, 0, 1] + half_h).to(torch.long)

    # Clamp for centroids at or beyond image boundaries.
    crop_x = torch.clamp(crop_x, 0, patches.shape[3] - 1)
    crop_y = torch.clamp(crop_y, 0, patches.shape[2] - 1)

    if not isinstance(sample_inds, torch.Tensor):
        sample_inds = torch.tensor(sample_inds, device=device)
    sample_inds_long = sample_inds.to(device=device, dtype=torch.long)
    crops = patches[sample_inds_long, :, crop_y, crop_x]

    return crops.to(original_dtype)
