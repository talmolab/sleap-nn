"""Memory estimation utilities for training configuration.

This module provides tools for estimating GPU and CPU memory requirements
based on training configuration parameters.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    """Estimate model parameters based on architecture (matches web app formula).

    Args:
        filters: Base number of filters in first encoder block.
        max_stride: Maximum stride (determines encoder depth).
        output_stride: Output stride for decoder depth.
        in_channels: Number of input channels.
        num_keypoints: Number of output keypoints.
        filters_rate: Multiplier for filters per encoder block.

    Returns:
        Estimated total parameter count.
    """
    down_blocks = int(math.log2(max_stride))
    up_blocks = int(math.log2(max_stride / output_stride)) if output_stride > 0 else down_blocks

    total_params = 0
    ch = in_channels
    f = filters

    # Encoder blocks: 2 convs per block (3x3 kernel)
    for i in range(down_blocks):
        total_params += ch * f * 9 + f  # First conv + bias
        total_params += f * f * 9 + f   # Second conv + bias
        ch = f
        f = int(f * filters_rate)

    # Middle block
    total_params += ch * f * 9 + f
    total_params += f * f * 9 + f
    middle_filters = f

    # Decoder blocks
    f = middle_filters
    for i in range(up_blocks):
        next_f = int(f / filters_rate)
        # Skip connection doubles input channels
        skip_f = int(filters * (filters_rate ** (down_blocks - 1 - i))) if i < down_blocks else 0
        decoder_input = f + skip_f
        total_params += decoder_input * next_f * 9 + next_f
        total_params += next_f * next_f * 9 + next_f
        f = next_f

    # Head: 1x1 conv to keypoints
    total_params += f * num_keypoints * 1 + num_keypoints

    return total_params


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
        if hasattr(stats, 'num_nodes'):
            num_keypoints = stats.num_nodes
        elif hasattr(stats, 'node_names'):
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
    total_bytes = weights_bytes + batch_img_bytes + confmap_bytes + act_bytes + gradient_bytes
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
