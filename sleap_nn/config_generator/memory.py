"""Memory estimation utilities for training configuration.

This module provides tools for estimating GPU and CPU memory requirements
based on training configuration parameters.
"""

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
        total_gpu_mb: Total estimated GPU memory in MB.
        cache_memory_mb: Estimated CPU cache memory in MB.
        gpu_status: Status indicator (green, yellow, red).
        gpu_message: Human-readable GPU memory message.
        cpu_fits_in_memory: Whether cache fits in available RAM.
        cpu_message: Human-readable CPU memory message.
    """

    model_weights_mb: float
    batch_images_mb: float
    activations_mb: float
    gradients_mb: float
    total_gpu_mb: float
    cache_memory_mb: float
    gpu_status: Literal["green", "yellow", "red"]
    gpu_message: str
    cpu_fits_in_memory: bool
    cpu_message: str

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


# Approximate parameter counts for different backbones
_PARAM_ESTIMATES = {
    "unet_medium_rf": 1_500_000,
    "unet_large_rf": 3_000_000,
    "convnext_tiny": 28_000_000,
    "convnext_small": 50_000_000,
    "convnext_base": 89_000_000,
    "convnext_large": 198_000_000,
    "swint_tiny": 28_000_000,
    "swint_small": 50_000_000,
    "swint_base": 88_000_000,
}


def estimate_memory(
    stats: "DatasetStats",
    backbone: str = "unet_medium_rf",
    batch_size: int = 4,
    input_scale: float = 1.0,
    output_stride: int = 1,
) -> MemoryEstimate:
    """Estimate GPU and CPU memory requirements.

    Args:
        stats: DatasetStats from analyze_slp().
        backbone: Backbone architecture name.
        batch_size: Training batch size.
        input_scale: Input image scaling factor.
        output_stride: Output stride for confidence maps.

    Returns:
        MemoryEstimate with breakdown and recommendations.

    Example:
        >>> stats = analyze_slp("labels.slp")
        >>> mem = estimate_memory(stats, batch_size=4)
        >>> print(f"GPU: {mem.total_gpu_gb:.1f} GB ({mem.gpu_status})")
    """
    # Scaled dimensions
    h = int(stats.max_height * input_scale)
    w = int(stats.max_width * input_scale)

    # Pad to stride 32 for safety
    h_padded = ((h + 31) // 32) * 32
    w_padded = ((w + 31) // 32) * 32

    # Model parameters estimate
    num_params = _PARAM_ESTIMATES.get(backbone, 2_000_000)

    # Memory calculations (in MB)
    # Model weights in FP32
    model_weights = (num_params * 4) / 1e6

    # Batch images (FP32)
    batch_images = (batch_size * h_padded * w_padded * stats.num_channels * 4) / 1e6

    # Activations estimate
    # Rough heuristic: ~0.5x model params per batch element for intermediate activations
    activations = (batch_size * num_params * 4 * 0.5) / 1e6

    # Gradients roughly equal to activations
    gradients = activations

    # Total with safety factor
    total_gpu = (model_weights + batch_images + activations + gradients) * 1.3

    # GPU status thresholds
    if total_gpu < 4000:
        gpu_status: Literal["green", "yellow", "red"] = "green"
        gpu_message = "Should fit on most GPUs (8GB+)"
    elif total_gpu < 8000:
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
        model_weights_mb=model_weights,
        batch_images_mb=batch_images,
        activations_mb=activations,
        gradients_mb=gradients,
        total_gpu_mb=total_gpu,
        cache_memory_mb=cache_mb,
        gpu_status=gpu_status,
        gpu_message=gpu_message,
        cpu_fits_in_memory=cpu_fits,
        cpu_message=cpu_message,
    )
