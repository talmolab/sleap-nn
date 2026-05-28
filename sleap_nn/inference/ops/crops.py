"""Bbox creation + image cropping helpers used by inference.

Two responsibilities:

1. ``crop_bboxes`` — vectorized patch extraction around a set of bboxes.
   Used by integral peak refinement and top-down stage 2 crop pickup.
2. Re-export ``make_centered_bboxes`` from the data layer so callers in
   the inference path import from one place. The data layer keeps owning
   the function (the training pipeline uses it too).

PR 5 of #508 rewrote :func:`crop_bboxes` to remove the per-peak ``unfold``
formulation that the legacy TorchScript ONNX exporter rejected. The new
implementation:

* pads the image with zeros (matching old out-of-bounds behavior)
* gathers crops via ``advanced indexing`` on the padded tensor
* lowers cleanly to ONNX
* matches the previous integer-indexing behavior bit-exactly when bbox
  top-lefts are integer-aligned (which is the case for both call sites:
  integer peaks → make_centered_bboxes returns integer bboxes; centroid-
  driven crops floor the top-left)
"""

import torch
import torch.nn.functional as F

# Single source of truth for ``make_centered_bboxes`` is the data layer.
from sleap_nn.data.instance_cropping import make_centered_bboxes  # noqa: F401


def crop_bboxes(
    images: torch.Tensor, bboxes: torch.Tensor, sample_inds: torch.Tensor
) -> torch.Tensor:
    """Crop bounding boxes from a batch of images.

    Args:
        images: ``(samples, channels, height, width)`` tensor.
        bboxes: ``(n_bboxes, 4, 2)`` ``float32`` corner tensor in
            top-left → top-right → bottom-right → bottom-left order. Build
            these with :func:`make_centered_bboxes`.
        sample_inds: ``(n_bboxes,)`` int tensor selecting which sample each
            bbox crops from.

    Returns:
        ``(n_bboxes, channels, crop_height, crop_width)`` of the same dtype
        as ``images``. The crop size is inferred from the first bbox.

    Notes:
        Bbox top-lefts are floored to the integer grid before extraction,
        matching the prior ``.to(torch.long)`` behavior. Out-of-image
        sample positions are zero-padded.

    See Also:
        :func:`make_centered_bboxes`.
    """
    n_crops = bboxes.shape[0]
    if n_crops == 0:
        return torch.empty(
            0, images.shape[1], 0, 0, device=images.device, dtype=images.dtype
        )

    # Crop size from the first bbox. ``.item()`` makes this a Python int —
    # required to allocate fixed-shape index tensors. PR 7 (#515)
    # parameterizes ONNX wrappers with the constexpr crop size to bypass
    # this call.
    height = int(abs(bboxes[0, 3, 1] - bboxes[0, 0, 1]).item()) + 1
    width = int(abs(bboxes[0, 1, 0] - bboxes[0, 0, 0]).item()) + 1

    device = images.device
    bboxes_on_device = bboxes.to(device)

    # Floor the top-left so we sample integer pixel positions exactly. This
    # preserves the old ``.to(torch.long)`` truncation behavior for sub-pixel
    # centroid-driven crops, keeping topdown end-to-end parity with prior
    # output bit-exactly.
    crop_topleft = bboxes_on_device[:, 0, :].long()  # (n, 2) -- (x, y)

    # Pad the source image with zeros so out-of-bounds samples become 0.
    # The pad amount is at least ``max(width, height)`` so any in-image
    # sub-pixel can shift up to one full crop without escaping the padded
    # region (used as a static, ONNX-friendly upper bound).
    pad_h, pad_w = height, width
    images_padded = F.pad(
        images, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0
    )
    padded_h, padded_w = images_padded.shape[-2], images_padded.shape[-1]

    # Build per-crop sample indices over the padded image.
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.long, device=device),
        torch.arange(width, dtype=torch.long, device=device),
        indexing="ij",
    )
    # offsets shape: (height, width)
    # crop top-left in padded coords = original top-left + (pad_w, pad_h)
    abs_x = crop_topleft[:, 0:1, None] + xx + pad_w  # (n, h, w)
    abs_y = crop_topleft[:, 1:2, None] + yy + pad_h  # (n, h, w)

    # Clamp to padded bounds (defends against extremely-out-of-bounds bboxes).
    abs_x = abs_x.clamp(0, padded_w - 1)
    abs_y = abs_y.clamp(0, padded_h - 1)

    # Gather. Result shape: (n_crops, channels, height, width).
    if not isinstance(sample_inds, torch.Tensor):
        sample_inds = torch.tensor(sample_inds, device=device)
    sample_inds_long = sample_inds.to(device=device, dtype=torch.long)
    sample_idx = sample_inds_long[:, None, None]  # (n, 1, 1)
    crops = images_padded[sample_idx, :, abs_y, abs_x]  # (n, h, w, c)
    # Move channels back to dim 1: (n, c, h, w)
    return crops.permute(0, 3, 1, 2).contiguous()
