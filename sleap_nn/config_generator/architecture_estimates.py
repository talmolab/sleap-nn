"""Shared architecture-related estimates for UNet config recommendations.

Single source of truth for the formulas used by the config generator (TUI),
recommender, generator, and YAML emitter. Mirrors the formulas used by the
config picker web app at ``docs/configuration/config-picker/app.html`` so the
two surfaces produce equivalent recommendations and estimates.

References:
- Canonical receptive-field formula: ``example_notebooks/receptive_field_guide.py``
  (https://distill.pub/2019/computing-receptive-fields/, Eq. 2)
- UNet implementation (ground truth for parameter count):
  ``sleap_nn/architectures/unet.py``
- Web-app counterparts: ``computeReceptiveField``, ``estimateParamsAccurate``,
  ``computeAugmentationPadding``, ``computeSuggestedCropSize`` in app.html.
"""

import math
from typing import Optional, Tuple

SUPPORTED_MAX_STRIDES: Tuple[int, ...] = (8, 16, 32, 64, 128)


def compute_receptive_field(
    max_stride: int,
    convs_per_block: int = 2,
    kernel_size: int = 3,
) -> int:
    """Compute the receptive field of the deepest encoder layer of a UNet.

    Each downsampling block has ``convs_per_block`` convolutions (stride 1,
    kernel ``kernel_size``) followed by a 2x2 stride-2 pool. RF is built up
    layer-by-layer with the canonical formula::

        RF = 1 + sum((kernel[l] - 1) * prod(strides[:l])) for l in 0..L-1

    Args:
        max_stride: Total downsampling factor of the encoder (must be a
            positive power of 2).
        convs_per_block: Number of conv layers per down block.
        kernel_size: Kernel size of the conv layers.

    Returns:
        Receptive field in input pixels.
    """
    down_blocks = int(math.log2(max_stride))
    if 2**down_blocks != max_stride or max_stride < 1:
        raise ValueError(f"max_stride must be a positive power of 2, got {max_stride}")

    block_strides = [1] * convs_per_block + [2]
    block_kernels = [kernel_size] * convs_per_block + [2]

    strides = block_strides * down_blocks
    kernels = block_kernels * down_blocks

    rf = 1
    prod = 1
    for stride, kernel in zip(strides, kernels):
        rf += (kernel - 1) * prod
        prod *= stride
    return rf


def encoder_blocks(max_stride: int) -> int:
    """Number of encoder downsampling blocks for the given max stride."""
    return int(math.log2(max_stride))


def decoder_blocks(max_stride: int, output_stride: int) -> int:
    """Number of decoder upsampling blocks needed to reach ``output_stride``."""
    if output_stride <= 0:
        return encoder_blocks(max_stride)
    return int(math.log2(max_stride / output_stride))


def compute_max_stride_for_animal_size(
    animal_size: float,
    candidates: Tuple[int, ...] = SUPPORTED_MAX_STRIDES,
) -> int:
    """Smallest max_stride whose receptive field covers the animal.

    Args:
        animal_size: Maximum animal bounding-box dimension in input pixels
            (already scaled by ``input_scale`` if applicable).
        candidates: Strides to consider, ascending.

    Returns:
        Smallest stride in ``candidates`` whose RF >= ``animal_size``. Falls
        back to the largest candidate if none cover the animal.
    """
    for stride in sorted(candidates):
        if compute_receptive_field(stride) >= animal_size:
            return stride
    return max(candidates)


def recommend_default_max_stride(avg_animal_size: float, scale: float = 1.0) -> int:
    """Bucket-based default ``max_stride`` recommendation.

    Mirrors ``setDefaultParameters`` in
    ``docs/configuration/config-picker/app.html`` (lines 5371–5375): pick
    the stride based on the *average* animal bbox size after input scaling.

    Args:
        avg_animal_size: Average instance bbox diagonal in original pixels.
        scale: Input scale factor (multiplier applied before pickup).

    Returns:
        Recommended max_stride: 8 if effective size < 40, 32 if > 100, else 16.
    """
    effective = avg_animal_size * scale
    if effective < 40:
        return 8
    if effective > 100:
        return 32
    return 16


def estimate_unet_params(
    filters: int,
    max_stride: int,
    output_stride: int,
    in_channels: int,
    num_keypoints: int,
    filters_rate: float = 1.5,
) -> int:
    """Estimate trainable parameter count of a UNet head + body.

    Mirrors the web app's ``estimateParamsAccurate`` (app.html:3446) and
    matches the structure of the real UNet (``architectures/unet.py``):
    encoder + middle/bottleneck block + decoder + 1x1 head.

    Each encoder block is ``2x (kxk conv + bias)`` with k=3. Decoder blocks
    take a skip connection from the matching encoder level so their input
    channel count is ``f + skip_f``.

    Args:
        filters: Base filter count in the first encoder block.
        max_stride: Determines encoder depth (``log2(max_stride)`` blocks).
        output_stride: Determines decoder depth (``log2(max_stride/output_stride)``).
        in_channels: Network input channels (1 grayscale, 3 RGB).
        num_keypoints: Number of output channels in the head.
        filters_rate: Multiplier applied to filter count per encoder block.

    Returns:
        Estimated parameter count (weights + biases).
    """
    down_blocks = encoder_blocks(max_stride)
    up_blocks = decoder_blocks(max_stride, output_stride)

    total = 0
    ch = in_channels
    f = filters

    # Encoder
    for _ in range(down_blocks):
        total += ch * f * 9 + f
        total += f * f * 9 + f
        ch = f
        f = int(f * filters_rate)

    # Middle / bottleneck
    total += ch * f * 9 + f
    total += f * f * 9 + f
    middle_filters = f

    # Decoder
    f = middle_filters
    for i in range(up_blocks):
        next_f = int(f / filters_rate)
        skip_f = (
            int(filters * (filters_rate ** (down_blocks - 1 - i)))
            if i < down_blocks
            else 0
        )
        decoder_input = f + skip_f
        total += decoder_input * next_f * 9 + next_f
        total += next_f * next_f * 9 + next_f
        f = next_f

    # 1x1 head
    total += f * num_keypoints + num_keypoints

    return total


def compute_augmentation_padding(
    bbox_size: float,
    rotation_max: float = 0.0,
    scale_max: float = 1.0,
) -> int:
    """Pixels of padding required so a rotated/scaled bbox stays in bounds.

    For a square bbox rotated by angle theta, the worst-case bounding-box
    expansion is ``|cos(theta)| + |sin(theta)|``, which peaks at sqrt(2) at 45°.
    Scaling expands the bbox by ``max(scale_max, 1.0)``.

    Args:
        bbox_size: Original bbox dimension in pixels.
        rotation_max: Max absolute rotation in degrees.
        scale_max: Max scale factor (1.0 == no scaling).

    Returns:
        Padding in pixels (ceiling), 0 if no augmentation expansion needed.
    """
    if rotation_max == 0 and scale_max <= 1.0:
        return 0

    rotation_factor = 1.0
    if rotation_max > 0:
        if abs(rotation_max) >= 45:
            rotation_factor = math.sqrt(2)
        else:
            rad = math.radians(min(abs(rotation_max), 90))
            rotation_factor = abs(math.cos(rad)) + abs(math.sin(rad))

    expansion = rotation_factor * max(scale_max, 1.0)
    expanded = bbox_size * expansion
    return math.ceil(expanded - bbox_size)


def compute_suggested_crop_size(
    max_bbox_dim: float,
    max_stride: int,
    use_augmentation: bool = False,
    user_padding: Optional[int] = None,
    rotation_max: float = 0.0,
    scale_max: float = 1.0,
) -> int:
    """Suggest a crop size that fits the largest instance with optional padding.

    Mirrors the web app's ``computeSuggestedCropSize`` (app.html:3402).

    - If ``user_padding`` is provided, it overrides any auto-computed padding
      (including 0, which means "no padding").
    - Else if ``use_augmentation``, padding is computed from
      ``rotation_max`` / ``scale_max``.
    - Result is rounded UP to the next multiple of ``max_stride``.

    Args:
        max_bbox_dim: Largest instance bbox dimension (height or width).
        max_stride: Network max stride; result will be a multiple of this.
        use_augmentation: Whether to add padding for rotation/scale aug.
        user_padding: Explicit padding override.
        rotation_max: Max rotation in degrees (used when use_augmentation).
        scale_max: Max scale factor (used when use_augmentation).

    Returns:
        Suggested crop size in pixels, divisible by ``max_stride``.
    """
    if user_padding is not None and user_padding >= 0:
        padding = user_padding
    elif use_augmentation:
        padding = compute_augmentation_padding(max_bbox_dim, rotation_max, scale_max)
    else:
        padding = 0

    size_with_padding = max_bbox_dim + padding
    return math.ceil(size_with_padding / max_stride) * max_stride


def compute_pad_to_stride(height: int, width: int, max_stride: int) -> Tuple[int, int]:
    """Round (height, width) up so each is a multiple of ``max_stride``."""
    h_padded = math.ceil(height / max_stride) * max_stride
    w_padded = math.ceil(width / max_stride) * max_stride
    return h_padded, w_padded
