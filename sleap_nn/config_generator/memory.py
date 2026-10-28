"""Memory estimation utilities for training configuration.

This module provides tools for estimating GPU and CPU memory requirements
based on training configuration parameters.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from sleap_nn.config_generator.analyzer import DatasetStats


@dataclass
class MemoryEstimate:
    """Memory estimation for training configuration.

    Attributes:
        model_weights_mb: Estimated model weights memory in MB.
        batch_images_mb: Estimated batch images memory in MB.
        activations_mb: Estimated activations memory in MB.
        gradients_mb: Estimated gradients memory in MB.
        confmaps_mb: Estimated confidence maps memory in MB.
        total_gpu_mb: Total estimated GPU memory in MB.
        cache_memory_mb: Estimated CPU cache memory in MB.
        gpu_status: Status indicator (green, yellow, red).
        gpu_message: Human-readable GPU memory message.
        cpu_fits_in_memory: Whether cache fits in available RAM.
        cpu_message: Human-readable CPU memory message.
        params_count: Estimated number of model parameters.
    """

    model_weights_mb: float
    batch_images_mb: float
    activations_mb: float
    gradients_mb: float
    confmaps_mb: float
    total_gpu_mb: float
    cache_memory_mb: float
    gpu_status: Literal["green", "yellow", "red"]
    gpu_message: str
    cpu_fits_in_memory: bool
    cpu_message: str
    params_count: int = 0

    @property
    def total_gpu_gb(self) -> float:
        """Total GPU memory in GB."""
        return self.total_gpu_mb / 1024

    @property
    def cache_memory_gb(self) -> float:
        """Cache memory in GB."""
        return self.cache_memory_mb / 1024

    def __str__(self) -> str:
        """Return human-readable summary."""
        status_icons = {"green": "[OK]", "yellow": "[WARN]", "red": "[HIGH]"}
        icon = status_icons.get(self.gpu_status, "")

        lines = [
            f"GPU Memory: {self.total_gpu_gb:.1f} GB {icon}",
            f"  - Model weights: {self.model_weights_mb:.0f} MB",
            f"  - Batch images: {self.batch_images_mb:.0f} MB",
            f"  - Activations: {self.activations_mb:.0f} MB",
            f"  - Gradients: {self.gradients_mb:.0f} MB",
            f"  {self.gpu_message}",
            "",
            f"CPU Cache: {self.cache_memory_gb:.1f} GB",
            f"  {self.cpu_message}",
        ]
        return "\n".join(lines)


def _estimate_params_accurate(
    filters: int,
    max_stride: int,
    output_stride: int,
    in_channels: int,
    num_keypoints: int,
    filters_rate: float = 1.5,
) -> int:
    """Estimate UNet parameter count.

    Thin wrapper kept for backwards compatibility; canonical implementation
    lives in :mod:`sleap_nn.config_generator.architecture_estimates`.
    """
    from sleap_nn.config_generator.architecture_estimates import estimate_unet_params

    return estimate_unet_params(
        filters=filters,
        max_stride=max_stride,
        output_stride=output_stride,
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        filters_rate=filters_rate,
    )


def estimate_memory(
    stats: "DatasetStats",
    backbone: str = "unet_medium_rf",
    batch_size: int = 4,
    input_scale: float = 1.0,
    output_stride: int = 1,
    filters: int = 32,
    filters_rate: float = 1.5,
    max_stride: int = 16,
    num_keypoints: int = None,
) -> MemoryEstimate:
    """Estimate GPU and CPU memory requirements (matches web app formula).

    Args:
        stats: DatasetStats from analyze_slp().
        backbone: Backbone architecture name.
        batch_size: Training batch size.
        input_scale: Input image scaling factor.
        output_stride: Output stride for confidence maps.
        filters: Base number of filters (for UNet).
        filters_rate: Filter multiplier per block (for UNet).
        max_stride: Maximum stride (determines encoder depth).
        num_keypoints: Number of keypoints (defaults to stats.num_nodes).

    Returns:
        MemoryEstimate with breakdown and recommendations.
    """
    # Get number of keypoints
    if num_keypoints is None:
        if hasattr(stats, "num_nodes"):
            num_keypoints = stats.num_nodes
        elif hasattr(stats, "node_names"):
            num_keypoints = len(stats.node_names)
        else:
            num_keypoints = 24

    # Scaled dimensions
    h = int(stats.max_height * input_scale)
    w = int(stats.max_width * input_scale)

    # Pad to max_stride for UNet (dimensions must be divisible by 2^num_blocks)
    h_padded = ((h + max_stride - 1) // max_stride) * max_stride
    w_padded = ((w + max_stride - 1) // max_stride) * max_stride

    in_channels = stats.num_channels

    # Estimate parameters based on backbone type
    if "convnext" in backbone or "swint" in backbone:
        # Pretrained backbones have fixed param counts
        param_estimates = {
            "convnext_tiny": 28_000_000,
            "convnext_small": 50_000_000,
            "convnext_base": 89_000_000,
            "swint_tiny": 28_000_000,
            "swint_small": 50_000_000,
        }
        num_params = param_estimates.get(backbone, 28_000_000)
    else:
        # UNet: compute params based on architecture
        num_params = _estimate_params_accurate(
            filters, max_stride, output_stride, in_channels, num_keypoints, filters_rate
        )

    # Memory calculations (in bytes, fp32 = 4 bytes)
    weights_bytes = num_params * 4

    # Batch images (FP32)
    batch_img_bytes = batch_size * h_padded * w_padded * in_channels * 4

    # Confidence map outputs
    confmap_h = h_padded // output_stride
    confmap_w = w_padded // output_stride
    confmap_bytes = batch_size * confmap_h * confmap_w * num_keypoints * 4

    # Activations: sum of feature map sizes through encoder layers
    act_bytes = 0
    h_act, w_act = h_padded, w_padded
    f = filters
    down_blocks = int(math.log2(max_stride))
    for i in range(down_blocks + 1):
        act_bytes += batch_size * h_act * w_act * f * 4
        h_act = max(1, h_act // 2)
        w_act = max(1, w_act // 2)
        f = int(f * filters_rate)
    act_bytes *= 2  # Encoder + decoder

    # Gradients roughly equal to activations
    gradient_bytes = act_bytes

    # Total in MB
    total_bytes = (
        weights_bytes + batch_img_bytes + confmap_bytes + act_bytes + gradient_bytes
    )
    total_mb = total_bytes / 1e6

    model_weights_mb = weights_bytes / 1e6
    batch_images_mb = batch_img_bytes / 1e6
    confmaps_mb = confmap_bytes / 1e6
    activations_mb = act_bytes / 1e6
    gradients_mb = gradient_bytes / 1e6

    # GPU status thresholds
    if total_mb < 4000:
        gpu_status: Literal["green", "yellow", "red"] = "green"
        gpu_message = "Should fit on most GPUs (8GB+)"
    elif total_mb < 8000:
        gpu_status = "yellow"
        gpu_message = "May require 12GB+ GPU"
    else:
        gpu_status = "red"
        gpu_message = "Requires 16GB+ GPU or reduce batch_size/scale"

    # CPU cache estimate (with 20% buffer for overhead)
    bytes_per_frame = stats.max_height * stats.max_width * stats.num_channels
    cache_mb = (bytes_per_frame * stats.num_labeled_frames * 1.2) / 1e6

    # Check available memory
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / 1e6
        cpu_fits = cache_mb < available_mb * 0.8
    except ImportError:
        cpu_fits = True  # Assume fits if can't check

    cpu_message = (
        "Fits in available memory"
        if cpu_fits
        else "Consider disk caching (data_pipeline_fw='torch_dataset_cache_img_disk')"
    )

    return MemoryEstimate(
        model_weights_mb=model_weights_mb,
        batch_images_mb=batch_images_mb,
        activations_mb=activations_mb,
        gradients_mb=gradients_mb,
        confmaps_mb=confmaps_mb,
        total_gpu_mb=total_mb,
        cache_memory_mb=cache_mb,
        gpu_status=gpu_status,
        gpu_message=gpu_message,
        cpu_fits_in_memory=cpu_fits,
        cpu_message=cpu_message,
        params_count=num_params,
    )


#: Training bytes per parameter: fp32 weights, gradients (4 of the 20), AdamW's two
#: moments and the EMA copy.
_DETR_BYTES_PER_PARAM = 20
#: Saved activation bytes per token per channel per ViT block (bf16 autocast),
#: calibrated on M2's measured 4.7 GB/GPU peak: RF-DETR-M, batch 4, multi-scale up
#: to 736 px.
_VIT_ACTIVATION_BYTES = 78
#: Decoder activation bytes per query per layer, in units of
#: ``(12 * hidden_dim + 2 * dim_feedforward)`` (bf16: 2 bytes).
_DECODER_ACTIVATION_BYTES = 2


def estimate_detr_memory(
    stats: "DatasetStats",
    preset: str = "dinov3_s",
    batch_size: int = 8,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    num_queries: int = 100,
    group_detr: int = 13,
    multi_scale_steps: int = 5,
    hidden_dim: int = 256,
    dim_feedforward: int = 2048,
) -> MemoryEstimate:
    """Estimate the training GPU memory of a ``bbox_detr`` model.

    Peak memory is set by the largest multi-scale step (``max_height`` /
    ``max_width`` plus ``multi_scale_steps`` x 32): the ViT's saved activations
    (every block sees every token), the decoder over ``group_detr x num_queries``
    queries, and ~20 bytes per parameter for weights, gradients, AdamW and EMA.
    Calibrated on a measured run (RF-DETR-M at 576, batch 4: 4.7 GB); treat it as
    +/- 30%.

    Args:
        stats: DatasetStats from analyze_slp() (frame size for the CPU cache).
        preset: A ``DETR_PRESETS`` key.
        batch_size: Training images per GPU.
        max_height: Letterbox height (default: the preset's input size).
        max_width: Letterbox width (default: the preset's input size).
        num_queries: Queries per group.
        group_detr: Query groups in training.
        multi_scale_steps: Multi-scale steps above the base size (0: off).
        hidden_dim: Width of the memory and queries.
        dim_feedforward: Decoder FFN width.

    Returns:
        MemoryEstimate with breakdown (``confmaps_mb`` holds the decoder's share).
    """
    from sleap_nn.config_generator.architecture_estimates import (
        DETR_MAX_STRIDE,
        DETR_PATCH_SIZE,
        DETR_PRESETS,
        estimate_detr_params,
    )

    p = DETR_PRESETS[preset]
    height = int(max_height or p.input_size) + multi_scale_steps * DETR_MAX_STRIDE
    width = int(max_width or p.input_size) + multi_scale_steps * DETR_MAX_STRIDE
    tokens = (height // DETR_PATCH_SIZE) * (width // DETR_PATCH_SIZE)
    params = estimate_detr_params(
        preset, num_queries=num_queries, group_detr=group_detr
    )["total"]

    weights_mb = params * (_DETR_BYTES_PER_PARAM - 4) / 1e6
    gradients_mb = params * 4 / 1e6
    batch_img_mb = batch_size * height * width * 3 * 4 / 1e6
    vit_mb = (
        batch_size * p.num_layers * tokens * p.hidden_size * _VIT_ACTIVATION_BYTES / 1e6
    )
    decoder_mb = (
        batch_size
        * p.dec_layers
        * group_detr
        * num_queries
        * (12 * hidden_dim + 2 * dim_feedforward)
        * _DECODER_ACTIVATION_BYTES
        / 1e6
    )
    total_mb = weights_mb + gradients_mb + batch_img_mb + vit_mb + decoder_mb

    if total_mb < 10_000:
        gpu_status: Literal["green", "yellow", "red"] = "green"
        gpu_message = "Should fit on a 12 GB GPU"
    elif total_mb < 20_000:
        gpu_status = "yellow"
        gpu_message = "Needs a 24 GB GPU"
    else:
        gpu_status = "red"
        gpu_message = "Needs a 48 GB GPU, or reduce batch_size / input size"

    # The CPU side is the same as for the other models.
    cpu = estimate_memory(stats, batch_size=1)
    return MemoryEstimate(
        model_weights_mb=weights_mb,
        batch_images_mb=batch_img_mb,
        activations_mb=vit_mb,
        gradients_mb=gradients_mb,
        confmaps_mb=decoder_mb,
        total_gpu_mb=total_mb,
        cache_memory_mb=cpu.cache_memory_mb,
        gpu_status=gpu_status,
        gpu_message=gpu_message,
        cpu_fits_in_memory=cpu.cpu_fits_in_memory,
        cpu_message=cpu.cpu_message,
        params_count=params,
    )
