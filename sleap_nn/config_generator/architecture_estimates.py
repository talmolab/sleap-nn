"""Shared architecture-related estimates for config recommendations (UNet, bbox_detr).

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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


# Fixed per-family context margin (px) for non-UNet backbones under tiling.
# Pretrained backbones are EXCLUDED from tiling (see ``check_tiling``), so no
# margin is defined for them.
_BACKBONE_CONTEXT_MARGIN_PX = {"convnext": 128, "swint": 128}


def compute_backbone_context_margin(
    backbone_type: str,
    max_stride: int,
    convs_per_block: int = 2,
    kernel_size: int = 3,
) -> int:
    """Half the surrounding context (px) a tile edge needs to keep seams valid.

    A tile-edge output pixel has part of its receptive field off-tile; sizing
    the overlap by this margin ensures an adjacent tile's *center* (full RF)
    covers that region.

    - UNet: half the deepest-encoder receptive field (``compute_receptive_field``).
    - ConvNext / SwinT: a fixed per-family constant (windowed attention /
      patch-merging make an analytic RF invalid; see design DQ4).
    - Any other backbone (pretrained / unsupported): raises, since tiling is
      not supported there.

    Args:
        backbone_type: One of ``"unet"``, ``"convnext"``, ``"swint"``.
        max_stride: Backbone total downsampling factor (UNet only).
        convs_per_block: Convs per UNet down block (UNet only).
        kernel_size: UNet conv kernel size (UNet only).

    Returns:
        Context margin in input pixels.
    """
    if backbone_type == "unet":
        rf = compute_receptive_field(max_stride, convs_per_block, kernel_size)
        return int(math.ceil(rf / 2))
    if backbone_type in _BACKBONE_CONTEXT_MARGIN_PX:
        return _BACKBONE_CONTEXT_MARGIN_PX[backbone_type]
    raise ValueError(
        f"Tiling context margin is undefined for backbone {backbone_type!r} "
        "(pretrained / unsupported backbones cannot be tiled)."
    )


def compute_suggested_tile_size(
    max_bbox_dim: float,
    max_stride: int,
    output_stride: int,
    backbone_margin: int,
    object_multiple: float = 2.0,
    min_tile_multiples: int = 2,
) -> int:
    """Square tile side that fits an object plus context on both sides.

    Rounded UP to a multiple of ``lcm(max_stride, output_stride)`` so both the
    confmap target grid (subsamples by ``output_stride``) and the backbone
    (divides by ``max_stride``) stay exact. Depends ONLY on object extent +
    margin (no overlap) to avoid a cycle with ``compute_suggested_tile_overlap``.

    Args:
        max_bbox_dim: Largest instance bbox dimension (height or width).
        max_stride: Backbone total downsampling factor.
        output_stride: Head output stride.
        backbone_margin: Per-side context margin (``compute_backbone_context_margin``).
        object_multiple: Multiple of the object extent to span.
        min_tile_multiples: Floor on the tile side, in units of the divisor.

    Returns:
        Suggested square tile side in pixels, divisible by both strides.
    """
    divisor = math.lcm(int(max_stride), int(output_stride))
    raw = object_multiple * float(max_bbox_dim) + 2 * int(backbone_margin)
    tile = math.ceil(raw / divisor) * divisor
    return int(max(tile, min_tile_multiples * divisor))


def compute_suggested_tile_overlap(
    tile_size: int,
    max_bbox_dim: float,
    confmap_sigma: float,
    output_stride: int,
    backbone_margin: int,
    min_overlap_fraction: float = 0.25,
    sigma_multiple: float = 3.0,
) -> int:
    """Overlap (px) large enough that a seam-straddling object is whole in one tile.

    Covers half the object extent + a few confmap sigmas + backbone context, is
    at least ``min_overlap_fraction`` of the tile, rounded UP to a multiple of
    ``output_stride``, and clamped to leave a positive stride (``>= output_stride``).

    Args:
        tile_size: Square tile side (from ``compute_suggested_tile_size``).
        max_bbox_dim: Largest instance bbox dimension.
        confmap_sigma: Confidence-map Gaussian sigma (input pixels).
        output_stride: Head output stride.
        backbone_margin: Per-side context margin.
        min_overlap_fraction: Minimum overlap as a fraction of ``tile_size``.
        sigma_multiple: How many sigmas of blob to keep whole across a seam.

    Returns:
        Suggested overlap in pixels, divisible by ``output_stride``.
    """
    object_overlap = (
        0.5 * float(max_bbox_dim)
        + sigma_multiple * float(confmap_sigma)
        + int(backbone_margin)
    )
    frac_floor = float(min_overlap_fraction) * int(tile_size)
    overlap = math.ceil(max(object_overlap, frac_floor) / output_stride) * output_stride
    max_overlap = int(tile_size) - int(output_stride)
    return int(min(overlap, max_overlap))


# ---------------------------------------------------------------------------
# bbox_detr (RF-DETR-style detector) presets, sizes and input resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetrPreset:
    """A ``bbox_detr`` backbone + decoder size the config generator can emit.

    Attributes:
        name: Preset key (``DETR_PRESETS``).
        label: Human-readable name.
        backbone: ``model_config.backbone_config`` key: ``pretrained`` (an HF/timm
            plain ViT, here DINOv3) or ``dinov2_windowed`` (RF-DETR's backbone).
        model_name: Pretrained weights: a timm/HF id (``pretrained``) or the
            DINOv2 checkpoint the windowed backbone starts from.
        hidden_size: ViT token width.
        num_layers: ViT blocks.
        num_heads: ViT attention heads.
        dec_layers: DETR decoder layers.
        input_size: Default input side (RF-DETR's training resolution for the
            RF-DETR sizes; the DINOv3 presets size the input from the data).
        rfdetr_size: ``sleap-nn import-rfdetr`` size whose weights fit this preset
            exactly (``None`` for the DINOv3 presets).
    """

    name: str
    label: str
    backbone: str
    model_name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    dec_layers: int
    input_size: int
    rfdetr_size: Optional[str] = None


#: RF-DETR's Apache sizes (DINOv2-S windowed; they differ in resolution and decoder
#: depth) and the DINOv3 ViT-S / ViT-B backbones with a fresh projector + decoder.
DETR_PRESETS: Dict[str, DetrPreset] = {
    p.name: p
    for p in (
        DetrPreset(
            "rfdetr_n", "RF-DETR-N", "dinov2_windowed", "facebook/dinov2-small",
            384, 12, 6, 2, 384, "nano",
        ),
        DetrPreset(
            "rfdetr_s", "RF-DETR-S", "dinov2_windowed", "facebook/dinov2-small",
            384, 12, 6, 3, 512, "small",
        ),
        DetrPreset(
            "rfdetr_m", "RF-DETR-M", "dinov2_windowed", "facebook/dinov2-small",
            384, 12, 6, 4, 576, "medium",
        ),
        DetrPreset(
            "rfdetr_l", "RF-DETR-L", "dinov2_windowed", "facebook/dinov2-small",
            384, 12, 6, 4, 704, "large",
        ),
        DetrPreset(
            "dinov3_s", "DINOv3 ViT-S/16", "pretrained",
            "timm/vit_small_patch16_dinov3.lvd1689m", 384, 12, 6, 3, 640,
        ),
        DetrPreset(
            "dinov3_b", "DINOv3 ViT-B/16", "pretrained",
            "timm/vit_base_patch16_dinov3.lvd1689m", 768, 12, 12, 4, 704,
        ),
    )
}  # fmt: skip

#: The animal default: DINOv3 ViT-S with a fresh projector and decoder.
DEFAULT_DETR_PRESET = "dinov3_s"

#: ViT patch size (the DETR token stride) of every preset.
DETR_PATCH_SIZE = 16
#: Multi-scale step and input padding of every preset (``max_stride``).
DETR_MAX_STRIDE = 32


def estimate_vit_params(
    hidden_size: int,
    num_layers: int,
    patch_size: int = DETR_PATCH_SIZE,
    in_channels: int = 3,
    prefix_tokens: int = 5,
    pos_embed_tokens: int = 0,
) -> int:
    """Parameter count of a plain ViT encoder (DINOv2 / DINOv3).

    Each block has attention (``4 d^2 + 4 d``), a 4x MLP (``8 d^2 + 5 d``), two
    norms and two LayerScales (``6 d``); plus the patch embedding, prefix
    (CLS + register) tokens, an optional learned position embedding and the final
    norm. Within 0.1% of the real DINOv3 ViT-S/B and windowed DINOv2-S modules.

    Args:
        hidden_size: Token width ``d``.
        num_layers: Transformer blocks.
        patch_size: Patch size.
        in_channels: Input channels.
        prefix_tokens: CLS + register tokens (DINOv3: 1 + 4).
        pos_embed_tokens: Learned position-embedding entries (DINOv2: grid^2 + 1;
            DINOv3 uses RoPE: 0).

    Returns:
        Number of parameters.
    """
    d = int(hidden_size)
    block = 12 * d * d + 15 * d
    patch = in_channels * patch_size * patch_size * d + d
    return int(
        num_layers * block + patch + (prefix_tokens + pos_embed_tokens) * d + 2 * d
    )


def estimate_detr_head_params(
    in_channels: int,
    num_classes: int = 1,
    num_queries: int = 300,
    group_detr: int = 13,
    dec_layers: int = 3,
    hidden_dim: int = 256,
    dim_feedforward: int = 2048,
) -> int:
    """Exact parameter count of a ``bbox_detr`` head (projector, decoder, heads).

    Builds the head on PyTorch's ``meta`` device, which allocates nothing.

    Args:
        in_channels: Channels of the concatenated backbone taps (4 x ``d``).
        num_classes: Object classes.
        num_queries: Queries per group.
        group_detr: Query groups in training.
        dec_layers: Decoder layers.
        hidden_dim: Width of the memory and queries.
        dim_feedforward: Decoder FFN width.

    Returns:
        Number of parameters (the learned queries count every group).
    """
    import torch

    from sleap_nn.architectures.detr.model import BBoxDetrHead

    with torch.device("meta"):
        head = BBoxDetrHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            group_detr=group_detr,
            dec_layers=dec_layers,
            dim_feedforward=dim_feedforward,
        )
    return int(sum(p.numel() for p in head.parameters()))


def estimate_detr_params(
    preset: str,
    num_classes: int = 1,
    num_queries: int = 300,
    group_detr: int = 13,
    num_taps: int = 4,
) -> Dict[str, int]:
    """Backbone, head and total parameters of a ``bbox_detr`` preset.

    Args:
        preset: A ``DETR_PRESETS`` key.
        num_classes: Object classes.
        num_queries: Queries per group.
        group_detr: Query groups in training.
        num_taps: Tapped backbone blocks feeding the projector.

    Returns:
        ``{"backbone", "head", "total"}`` parameter counts.
    """
    p = DETR_PRESETS[preset]
    if p.backbone == "dinov2_windowed":
        # A learned position embedding over the training-size token grid + CLS
        # (RF-DETR's `positional_encoding_size`); no registers.
        grid = p.input_size // DETR_PATCH_SIZE
        backbone = estimate_vit_params(
            p.hidden_size,
            p.num_layers,
            prefix_tokens=1,
            pos_embed_tokens=grid * grid + 1,
        )
    else:
        backbone = estimate_vit_params(p.hidden_size, p.num_layers)
    head = estimate_detr_head_params(
        num_taps * p.hidden_size,
        num_classes=num_classes,
        num_queries=num_queries,
        group_detr=group_detr,
        dec_layers=p.dec_layers,
    )
    return {"backbone": backbone, "head": head, "total": backbone + head}


@dataclass
class DetrResolution:
    """Input size of a ``bbox_detr`` model chosen by the token-width rule.

    Attributes:
        max_side: Letterbox side (``preprocessing.max_height`` / ``max_width`` cap
            the longer frame side at this), a multiple of ``DETR_MAX_STRIDE``.
        max_height: Letterbox height for the dataset's frame aspect.
        max_width: Letterbox width for the dataset's frame aspect.
        scale: Frame-to-input scale.
        small_tokens: Tokens across a small animal (5th percentile) at this size.
        median_tokens: Tokens across the median animal at this size.
        needs_tiling: The rule asked for more than ``max_side_cap``; the input was
            capped and small animals stay under ``min_tokens`` wide.
        reason: One-line explanation.
    """

    max_side: int
    max_height: int
    max_width: int
    scale: float
    small_tokens: float
    median_tokens: float
    needs_tiling: bool
    reason: str


def recommend_detr_resolution(
    frame_height: int,
    frame_width: int,
    small_animal_size: float,
    median_animal_size: float,
    patch_size: int = DETR_PATCH_SIZE,
    min_tokens: float = 3.0,
    target_tokens: float = 4.0,
    min_side: int = 384,
    max_side_cap: int = 1280,
    multiple: int = DETR_MAX_STRIDE,
) -> DetrResolution:
    """Letterbox size that keeps animals at least 3-4 ViT tokens wide.

    A DETR on a stride-16 ViT predicts boxes from 16 x 16 px tokens; an animal
    under ~3 tokens across is often missed. The rule picks the smallest scale at
    which the small animals (``small_animal_size``, e.g. the 5th percentile of
    ``sqrt(w * h)`` of the training boxes) are ``min_tokens`` wide and the median
    animal ``target_tokens`` wide, never above the native resolution, then rounds
    the longer side up to ``multiple``, between ``min_side`` and ``max_side_cap``.
    ``max_side_cap`` bounds global attention cost (1280 px = 6,400 tokens); past it
    the arena is too large for the animals and ``needs_tiling`` is set.

    Args:
        frame_height: Frame height (px).
        frame_width: Frame width (px).
        small_animal_size: Size of a small animal (px), e.g. p5 of ``sqrt(w * h)``.
        median_animal_size: Median animal size (px).
        patch_size: Token stride.
        min_tokens: Tokens across a small animal.
        target_tokens: Tokens across the median animal.
        min_side: Smallest input side.
        max_side_cap: Largest input side before tiling is needed.
        multiple: Input sides are multiples of this (the backbone's max stride).

    Returns:
        The chosen :class:`DetrResolution`.
    """
    long_side = max(int(frame_height), int(frame_width))
    small = max(float(small_animal_size), 1.0)
    median = max(float(median_animal_size), small)
    want = max(min_tokens * patch_size / small, target_tokens * patch_size / median)
    scale = min(1.0, want)
    side = int(math.ceil(long_side * scale / multiple) * multiple)
    side = max(side, min(min_side, int(math.ceil(long_side / multiple) * multiple)))
    needs_tiling = side > max_side_cap
    side = min(side, max_side_cap)
    scale = min(1.0, side / long_side)

    def fit(n: int) -> int:
        return max(multiple, int(math.ceil(n * scale / multiple) * multiple))

    max_height, max_width = fit(frame_height), fit(frame_width)
    small_tokens = small * scale / patch_size
    median_tokens = median * scale / patch_size
    if needs_tiling:
        reason = (
            f"Animals need a {long_side * want:.0f} px input to stay {min_tokens:g} "
            f"tokens wide; capped at {max_side_cap} px, where small animals are "
            f"{small_tokens:.1f} tokens wide. The arena is large for these animals: "
            "raise the input size if GPU memory allows, or crop the arena."
        )
    elif want >= 1.0:
        reason = (
            f"Native resolution: small animals are {small_tokens:.1f} tokens wide "
            f"(target {min_tokens:g})"
            + ("." if small_tokens >= min_tokens else "; expect misses on them.")
        )
    else:
        reason = (
            f"Letterbox at {side} px (scale {scale:.2f}): small animals "
            f"{small_tokens:.1f} tokens wide, median {median_tokens:.1f}."
        )
    return DetrResolution(
        max_side=side,
        max_height=max_height,
        max_width=max_width,
        scale=scale,
        small_tokens=small_tokens,
        median_tokens=median_tokens,
        needs_tiling=needs_tiling,
        reason=reason,
    )
