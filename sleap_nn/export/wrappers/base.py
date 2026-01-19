"""Base classes and shared helpers for export wrappers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BaseExportWrapper(nn.Module):
    """Base class for ONNX-exportable wrappers."""

    def __init__(self, model: nn.Module):
        """Initialize wrapper with the underlying model.

        Args:
            model: The PyTorch model to wrap for export.
        """
        super().__init__()
        self.model = model

    @staticmethod
    def _normalize_uint8(image: torch.Tensor) -> torch.Tensor:
        """Normalize unnormalized uint8 (or [0, 255] float) images to [0, 1]."""
        if image.dtype != torch.float32:
            image = image.float()
        return image / 255.0

    @staticmethod
    def _extract_tensor(output, key_hints: Iterable[str]) -> torch.Tensor:
        if isinstance(output, dict):
            for key in output:
                for hint in key_hints:
                    if hint.lower() in key.lower():
                        return output[key]
            return next(iter(output.values()))
        return output

    @staticmethod
    def _find_topk_peaks(
        confmaps: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K peak finding with NMS via max pooling."""
        batch_size, _, height, width = confmaps.shape
        pooled = F.max_pool2d(confmaps, kernel_size=3, stride=1, padding=1)
        is_peak = (confmaps == pooled) & (confmaps > 0)

        confmaps_flat = confmaps.reshape(batch_size, height * width)
        is_peak_flat = is_peak.reshape(batch_size, height * width)
        masked = torch.where(
            is_peak_flat, confmaps_flat, torch.full_like(confmaps_flat, -1e9)
        )
        values, indices = torch.topk(masked, k=k, dim=1)

        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)
        valid = values > 0
        return peaks, values, valid

    @staticmethod
    def _find_topk_peaks_per_node(
        confmaps: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K peak finding per channel with NMS via max pooling."""
        batch_size, n_nodes, height, width = confmaps.shape
        pooled = F.max_pool2d(confmaps, kernel_size=3, stride=1, padding=1)
        is_peak = (confmaps == pooled) & (confmaps > 0)

        confmaps_flat = confmaps.reshape(batch_size, n_nodes, height * width)
        is_peak_flat = is_peak.reshape(batch_size, n_nodes, height * width)
        masked = torch.where(
            is_peak_flat, confmaps_flat, torch.full_like(confmaps_flat, -1e9)
        )
        values, indices = torch.topk(masked, k=k, dim=2)

        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)
        valid = values > 0
        return peaks, values, valid

    @staticmethod
    def _find_global_peaks(
        confmaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find global maxima per channel."""
        batch_size, channels, height, width = confmaps.shape
        flat = confmaps.reshape(batch_size, channels, height * width)
        values, indices = flat.max(dim=-1)
        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)
        return peaks, values
