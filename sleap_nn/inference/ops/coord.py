"""Coordinate-ladder reversers for inference.

The current pipeline scatters ``stride`` / ``input_scale`` / ``eff_scale`` /
``crop_offset`` math across every Lightning ``forward()`` and across
``Predictor._make_labeled_frames_from_generator()``. That spread is the
single biggest source of "silent miscalibration" bugs.

This module is the single source of truth for the reverse direction (model
output → original-image coordinates). Every step is an early-returnable
identity for its no-op case so we don't pay extra ops when the user hasn't
configured a transform.

Apply order from a peak in confmap pixel space back to original-image space::

    peaks = undo_stride(peaks, info.output_stride)         # 1
    peaks = undo_input_scale(peaks, info.input_scale)      # 2
    peaks = add_crop_offset(peaks, info.crop_offsets)      # 3 (top-down only)
    peaks = undo_eff_scale(peaks, info.eff_scale)          # 4
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def undo_stride(coords: torch.Tensor, output_stride: int) -> torch.Tensor:
    """Scale peak coords from confmap pixel space to input-image pixel space.

    Args:
        coords: Tensor of any shape ending in 2 (the trailing axis is xy).
        output_stride: Stride between input pixels and confmap pixels (>=1).

    Returns:
        Coords scaled by ``output_stride``. ``stride == 1`` is an identity.
    """
    if output_stride == 1:
        return coords
    return coords * output_stride


def undo_input_scale(coords: torch.Tensor, input_scale: float) -> torch.Tensor:
    """Reverse the input-scale resize applied during preprocessing.

    Args:
        coords: Tensor with trailing xy axis.
        input_scale: Scale factor that was applied before the model
            (``preprocess_config.scale``). ``1.0`` is an identity.

    Returns:
        Coords divided by ``input_scale``.
    """
    if input_scale == 1.0:
        return coords
    return coords / input_scale


def undo_eff_scale(coords: torch.Tensor, eff_scale: torch.Tensor) -> torch.Tensor:
    """Reverse the per-sample sizematcher scale.

    The sizematcher fits each frame to ``(max_h, max_w)`` preserving aspect
    ratio, producing one scalar scale factor per batch sample (not constant).

    Args:
        coords: Tensor with shape ``(B, ...)`` whose trailing axis is xy.
        eff_scale: ``(B,)`` per-sample scale factors.

    Returns:
        Coords with each sample divided by its own ``eff_scale``.
    """
    # All-ones short-circuit avoids the broadcast div on the common path
    # where every frame had the same native size.
    if torch.all(eff_scale == 1.0):
        return coords
    shape = [eff_scale.shape[0]] + [1] * (coords.ndim - 1)
    return coords / eff_scale.view(shape).to(coords.device)


def add_crop_offset(peaks: torch.Tensor, crop_topleft: torch.Tensor) -> torch.Tensor:
    """Shift crop-local peaks back into full-image coordinates (top-down).

    Args:
        peaks: ``(B*I, N, 2)`` or ``(B, I, N, 2)`` peaks in crop-local space.
        crop_topleft: ``(B*I, 2)`` top-left corner of each crop bbox in
            ``(x, y)`` order.

    Returns:
        Peaks shifted into full-image space.
    """
    return peaks + crop_topleft.to(peaks.device).view(-1, 1, 2)


def apply_input_scale(image: torch.Tensor, input_scale: float) -> torch.Tensor:
    """Bilinear resize an image batch by ``input_scale`` (forward direction).

    Args:
        image: ``(B, C, H, W)`` float tensor.
        input_scale: Scale factor. ``1.0`` is an identity.

    Returns:
        Resized image. Output dtype matches input.
    """
    if input_scale == 1.0:
        return image
    h = int(image.shape[-2] * input_scale)
    w = int(image.shape[-1] * input_scale)
    return F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)


__all__: Tuple[str, ...] = (
    "undo_stride",
    "undo_input_scale",
    "undo_eff_scale",
    "add_crop_offset",
    "apply_input_scale",
)
