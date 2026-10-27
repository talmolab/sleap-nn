"""Resizing and augmentation of ``bbox_detr`` training frames (RF-DETR's recipe).

RF-DETR (``rfdetr`` @ d7d7e4c, ``datasets/coco.py`` + ``training/module_model.py``)
trains its detectors with:

1. **Resize or resize + crop**, picked 50/50 per sample: either the frame is
   resized straight to the training size, or its shortest side is resized to one
   of 400/500/600 px, a random square of side 384..600 px (at most the frame) is
   cropped out, and the crop is resized to the training size. Boxes are clipped
   to the crop; boxes left with no area are dropped.
2. **Horizontal flip** with probability 0.5.
3. **Per-batch multi-scale**: every sample is made at the largest scale and the
   collated batch is resized to one scale per optimizer step, drawn from the
   base resolution +/- 5 steps of ``patch_size * num_windows`` (the backbone's
   ``max_stride``) with ``random.Random(global_step)``, so every rank draws the
   same scale.

The "resize to the training size" is ``squash`` (RF-DETR's square resize, both
axes scaled independently) or ``letterbox`` (one scale to fit, padded on the
bottom/right), per ``data_config.preprocessing.resize_mode``. Boxes are
``(x1, y1, x2, y2)`` edge coordinates in pixels.
"""

import random
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvf

from sleap_nn.data.resizing import apply_sizematcher, compute_squash_scale

#: RF-DETR's pre-crop shortest sides and crop side range.
SHORTEST_SIDES = (400, 500, 600)
CROP_RANGE = (384, 600)


def multi_scale_sizes(
    base_hw: Tuple[int, int], block: int, steps: int = 5
) -> List[Tuple[int, int]]:
    """Training sizes of per-batch multi-scale: ``base +/- steps`` blocks per axis.

    RF-DETR's ``compute_multi_scale_scales(expanded_scales=True)``: each side is
    rounded down to a multiple of ``block`` and offset by ``-steps..steps`` blocks;
    sizes with a side under ``2 * block`` are left out.

    Args:
        base_hw: The base ``(height, width)`` (the inference size).
        block: The size step (``patch_size * num_windows``; the backbone's
            ``max_stride``).
        steps: Blocks to step up and down.

    Returns:
        Increasing ``(height, width)`` sizes.
    """
    base_h, base_w = base_hw[0] // block, base_hw[1] // block
    sizes = []
    for k in range(-steps, steps + 1):
        h, w = (base_h + k) * block, (base_w + k) * block
        if min(h, w) >= 2 * block:
            sizes.append((h, w))
    return sizes


def batch_scale(step: int, sizes: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    """The multi-scale size of optimizer step ``step``, the same on every rank."""
    return tuple(random.Random(step).choice(list(sizes)))


def resize_batch(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize a ``(B, C, H, W)`` batch to ``size`` (normalized boxes keep)."""
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    dtype = image.dtype
    out = F.interpolate(
        image.float(), size=tuple(size), mode="bilinear", align_corners=False
    )
    return out.to(dtype) if not dtype.is_floating_point else out


def resize_frame(
    image: torch.Tensor, boxes: torch.Tensor, out_hw: Tuple[int, int], mode: str
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]:
    """Fit a frame to ``out_hw`` by ``squash`` or ``letterbox``, with its boxes.

    Args:
        image: ``(..., C, H, W)`` image.
        boxes: ``(n, 4)`` xyxy boxes in the frame's pixels.
        out_hw: Output ``(height, width)``.
        mode: ``"squash"`` (resize each axis to fit exactly) or ``"letterbox"``
            (one scale to fit, zero-padded on the bottom/right).

    Returns:
        ``(image, boxes, (scale_x, scale_y))``.
    """
    in_hw = tuple(image.shape[-2:])
    if mode == "squash":
        sx, sy = compute_squash_scale(in_hw, out_hw)
        if in_hw != tuple(out_hw):
            image = tvf.resize(image, size=[int(out_hw[0]), int(out_hw[1])])
    elif mode == "letterbox":
        image, eff = apply_sizematcher(
            image, max_height=int(out_hw[0]), max_width=int(out_hw[1])
        )
        sx = sy = float(eff)
    else:
        raise ValueError(f"resize mode must be squash or letterbox, got {mode!r}.")
    return image, boxes * boxes.new_tensor([sx, sy, sx, sy]), (sx, sy)


def shortest_side_size(height: int, width: int, size: int) -> Tuple[int, int]:
    """Size with the shortest side at ``size`` and the aspect kept (RF-DETR rounding)."""
    if (width <= height and width == size) or (height <= width and height == size):
        return height, width
    if width < height:
        return int(round(size * height / width)), size
    return size, int(round(size * width / height))


def random_resize_crop(
    image: torch.Tensor,
    boxes: torch.Tensor,
    shortest_sides: Sequence[int] = SHORTEST_SIDES,
    crop_range: Tuple[int, int] = CROP_RANGE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RF-DETR's resize + random square crop (``RandomResize`` + ``RandomSizedCrop``).

    The shortest side is resized to a random entry of ``shortest_sides``; then a
    square of a random side in ``crop_range`` (capped by both sides) is cropped at
    a random position. Boxes move with the crop; clipping and dropping boxes left
    outside happen afterwards (see ``BBoxDataset``).

    Args:
        image: ``(..., C, H, W)`` image.
        boxes: ``(n, 4)`` xyxy boxes.
        shortest_sides: Candidate shortest sides of the pre-crop resize.
        crop_range: Inclusive ``(min, max)`` crop side.

    Returns:
        ``(crop, boxes in crop pixels)``.
    """
    height, width = image.shape[-2:]
    side = int(shortest_sides[int(torch.randint(len(shortest_sides), ()))])
    new_h, new_w = shortest_side_size(height, width, side)
    if (new_h, new_w) != (height, width):
        image = tvf.resize(image, size=[new_h, new_w])
        boxes = boxes * boxes.new_tensor(
            [new_w / width, new_h / height, new_w / width, new_h / height]
        )
    height, width = new_h, new_w

    min_crop, max_crop = int(crop_range[0]), int(crop_range[1])
    max_crop = min(max_crop, height, width)
    min_crop = min(min_crop, max_crop)
    crop = (
        int(torch.randint(min_crop, max_crop + 1, ()))
        if max_crop > min_crop
        else max_crop
    )
    crop_h, crop_w = min(crop, height), min(crop, width)
    top = int(torch.randint(0, height - crop_h + 1, ())) if height > crop_h else 0
    left = int(torch.randint(0, width - crop_w + 1, ())) if width > crop_w else 0
    image = image[..., top : top + crop_h, left : left + crop_w]
    boxes = boxes - boxes.new_tensor([left, top, left, top])
    return image, boxes


def hflip(
    image: torch.Tensor, boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror an image and its xyxy edge-coordinate boxes left-right."""
    width = image.shape[-1]
    image = image.flip(-1)
    boxes = torch.stack(
        [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=-1
    )
    return image, boxes


def detr_train_transform(
    image: torch.Tensor,
    boxes: torch.Tensor,
    out_hw: Tuple[int, int],
    mode: str,
    crop_p: float = 0.5,
    shortest_sides: Sequence[int] = SHORTEST_SIDES,
    crop_range: Tuple[int, int] = CROP_RANGE,
    flip_p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RF-DETR's training geometry: resize or resize + crop to ``out_hw``, and flip.

    The flip is applied before the final resize (RF-DETR flips after it; the
    distribution is the same) so a letterboxed frame keeps its padding on the
    bottom/right, as at inference.

    Args:
        image: ``(..., C, H, W)`` frame.
        boxes: ``(n, 4)`` xyxy boxes in frame pixels.
        out_hw: Training ``(height, width)``.
        mode: ``"squash"`` or ``"letterbox"`` (see :func:`resize_frame`).
        crop_p: Probability of the resize + crop branch.
        shortest_sides: Pre-crop shortest sides.
        crop_range: Crop side range.
        flip_p: Horizontal flip probability.

    Returns:
        ``(image, boxes)`` at ``out_hw`` (boxes unclipped).
    """
    if crop_p > 0 and float(torch.rand(())) < crop_p:
        image, boxes = random_resize_crop(image, boxes, shortest_sides, crop_range)
    if flip_p > 0 and float(torch.rand(())) < flip_p:
        image, boxes = hflip(image, boxes)
    image, boxes, _ = resize_frame(image, boxes, out_hw, mode)
    return image, boxes
