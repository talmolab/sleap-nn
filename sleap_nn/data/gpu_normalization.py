"""GPU normalization utilities for uint8 pipeline.

This module provides utilities to defer image normalization to the GPU,
enabling 4x bandwidth savings by transferring uint8 instead of float32.

Usage in model forward pass:
    from sleap_nn.data.gpu_normalization import normalize_image_gpu

    class Model(nn.Module):
        def forward(self, batch):
            # batch["image"] is uint8 tensor from DataLoader
            image = normalize_image_gpu(batch["image"])  # Now float32 on GPU
            # Continue with model forward pass...
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def normalize_image_gpu(
    image: torch.Tensor,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """Normalize uint8 image tensor on GPU.

    Converts uint8 [0, 255] to float32 [0, 1] (or normalized by mean/std).

    Args:
        image: Input tensor of shape (B, C, H, W) with dtype uint8 or float.
               If already float, assumes it's in [0, 1] range.
        mean: Optional per-channel mean for normalization (e.g., ImageNet mean).
        std: Optional per-channel std for normalization (e.g., ImageNet std).

    Returns:
        Float32 tensor normalized to [0, 1] or by mean/std if provided.
    """
    # Convert uint8 to float32 [0, 1]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0

    # Apply mean/std normalization if provided
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
        std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)
        image = (image - mean_tensor) / std_tensor

    return image


class GPUNormalize(nn.Module):
    """Module wrapper for GPU normalization.

    Can be used as the first layer in a model to handle uint8 input.

    Example:
        model = nn.Sequential(
            GPUNormalize(),
            backbone,
            head,
        )
    """

    def __init__(
        self,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ):
        """Initialize GPU normalization module.

        Args:
            mean: Optional per-channel mean for normalization.
            std: Optional per-channel std for normalization.
        """
        super().__init__()
        self.mean = mean
        self.std = std

        # Register as buffers if provided (moves with model to GPU)
        if mean is not None:
            self.register_buffer("mean_buffer", torch.tensor(mean).view(1, -1, 1, 1))
        if std is not None:
            self.register_buffer("std_buffer", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor (B, C, H, W), uint8 or float.

        Returns:
            Normalized float32 tensor.
        """
        # Convert uint8 to float32 [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Apply mean/std normalization if provided
        if self.mean is not None and self.std is not None:
            x = (x - self.mean_buffer) / self.std_buffer

        return x


def wrap_model_with_normalization(
    model: nn.Module,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> nn.Sequential:
    """Wrap a model with GPU normalization as the first layer.

    Args:
        model: The model to wrap.
        mean: Optional per-channel mean for normalization.
        std: Optional per-channel std for normalization.

    Returns:
        Sequential model with normalization as first layer.
    """
    return nn.Sequential(
        GPUNormalize(mean=mean, std=std),
        model,
    )
