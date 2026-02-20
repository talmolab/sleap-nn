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
    def _neighbor_max(x: torch.Tensor) -> torch.Tensor:
        """Compute max of 8 neighbors excluding center pixel.

        Uses -inf padding to match PyTorch dilation semantics (confmap heads
        are identity-activated, so negative values are possible).

        All ops (F.pad, slicing, torch.max) export cleanly to ONNX.
        """
        p = F.pad(x, [1, 1, 1, 1], mode="constant", value=float("-inf"))
        # 8 shifted views (excluding center)
        tl = p[:, :, :-2, :-2]  # top-left
        tc = p[:, :, :-2, 1:-1]  # top-center
        tr = p[:, :, :-2, 2:]  # top-right
        ml = p[:, :, 1:-1, :-2]  # middle-left
        mr = p[:, :, 1:-1, 2:]  # middle-right
        bl = p[:, :, 2:, :-2]  # bottom-left
        bc = p[:, :, 2:, 1:-1]  # bottom-center
        br = p[:, :, 2:, 2:]  # bottom-right
        return torch.max(
            torch.max(
                torch.max(
                    torch.max(torch.max(torch.max(torch.max(tl, tc), tr), ml), mr), bl
                ),
                bc,
            ),
            br,
        )

    @staticmethod
    def _find_topk_peaks(
        confmaps: torch.Tensor, k: int, peak_threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K peak finding with center-excluded neighbor-max NMS.

        Matches PyTorch path semantics: strict inequality (center > all
        neighbors) and configurable confidence threshold.
        """
        batch_size, _, height, width = confmaps.shape
        neighbor_max = BaseExportWrapper._neighbor_max(confmaps)
        is_peak = (confmaps > neighbor_max) & (confmaps > peak_threshold)

        confmaps_flat = confmaps.reshape(batch_size, height * width)
        is_peak_flat = is_peak.reshape(batch_size, height * width)
        masked = torch.where(
            is_peak_flat, confmaps_flat, torch.full_like(confmaps_flat, -1e9)
        )
        values, indices = torch.topk(masked, k=k, dim=1)

        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)
        valid = values > peak_threshold
        return peaks, values, valid

    @staticmethod
    def _find_topk_peaks_per_node(
        confmaps: torch.Tensor, k: int, peak_threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K peak finding per channel with center-excluded neighbor-max NMS.

        Matches PyTorch path semantics: strict inequality (center > all
        neighbors) and configurable confidence threshold.
        """
        batch_size, n_nodes, height, width = confmaps.shape
        neighbor_max = BaseExportWrapper._neighbor_max(confmaps)
        is_peak = (confmaps > neighbor_max) & (confmaps > peak_threshold)

        confmaps_flat = confmaps.reshape(batch_size, n_nodes, height * width)
        is_peak_flat = is_peak.reshape(batch_size, n_nodes, height * width)
        masked = torch.where(
            is_peak_flat, confmaps_flat, torch.full_like(confmaps_flat, -1e9)
        )
        values, indices = torch.topk(masked, k=k, dim=2)

        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)
        valid = values > peak_threshold
        return peaks, values, valid

    @staticmethod
    def _find_global_peaks(
        confmaps: torch.Tensor, peak_threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find global maxima per channel with threshold.

        Peaks with confidence below threshold are set to NaN coordinates and
        zero confidence, matching ``find_global_peaks_rough`` in the PyTorch
        path.
        """
        batch_size, channels, height, width = confmaps.shape
        flat = confmaps.reshape(batch_size, channels, height * width)
        values, indices = flat.max(dim=-1)
        y = indices // width
        x = indices % width
        peaks = torch.stack([x.float(), y.float()], dim=-1)

        below = values < peak_threshold
        peaks = peaks.masked_fill(below.unsqueeze(-1), float("nan"))
        values = values.masked_fill(below, 0.0)

        return peaks, values
