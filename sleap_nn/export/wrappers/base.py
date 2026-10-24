"""Base classes and shared helpers for export wrappers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.inference.ops.peaks import morphological_dilation_split


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
    def _local_peaks(
        confmaps: torch.Tensor, peak_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find local peaks with the flat-top rule of ``find_local_peaks_rough``.

        A pixel above ``peak_threshold`` that is greater than all 8 neighbors is
        a peak. A flat top of equal pixels (common with fp16 confidence maps) is
        one peak if it fits in a 2x2 block and every pixel around it is lower;
        its point is the mean of its pixels. A wider flat top, or one next to a
        higher pixel, gives no peak. See #781.

        This is a dense, fixed-shape version of the PyTorch path's sparse check
        (``_flat_tops``), so it exports to ONNX/TensorRT. A flat top is marked
        on its first pixel in raster order, which is ``>`` its 4 earlier
        neighbors and ``>=`` its 4 later ones (E, SW, S, SE); the later ones it
        equals are the rest of the top. The pixels of a top that fits in a 2x2
        block are all neighbors of each other, so the top is surrounded by
        lower pixels exactly when each of its pixels has ``|top| - 1``
        neighbors ``>=`` it. Without ties this is the strict rule (``>`` all 8
        neighbors) with zero offsets.

        Args:
            confmaps: ``(..., height, width)`` confidence maps.
            peak_threshold: Peaks must be above this.

        Returns:
            ``(is_peak, offset_x, offset_y)``, each shaped like ``confmaps``:
            whether a peak is marked at the pixel, and the ``float32`` offset
            from the pixel to the peak point (zero for one-pixel tops and
            non-peaks).
        """
        before, after = morphological_dilation_split(confmaps)
        candidate = (
            (confmaps > before) & (confmaps >= after) & (confmaps > peak_threshold)
        )

        def shifted(padded: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
            # The (dr, dc) neighbors of the image inside a 1-pixel border, sliced
            # without the image size so traced graphs keep dynamic H/W.
            return padded[..., 1 + dr : dr - 1 or None, 1 + dc : dc - 1 or None]

        padded = F.pad(confmaps, [1, 1, 1, 1], mode="constant", value=float("-inf"))
        # Number of neighbors >= each pixel.
        n_ge = torch.zeros_like(confmaps, dtype=torch.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    n_ge = n_ge + (shifted(padded, dr, dc) >= confmaps).float()
        n_ge = F.pad(n_ge, [1, 1, 1, 1])

        later = {"E": (0, 1), "SW": (1, -1), "S": (1, 0), "SE": (1, 1)}
        tied = {k: shifted(padded, *d) == confmaps for k, d in later.items()}
        t = {k: v.float() for k, v in tied.items()}
        n_top = t["E"] + t["SW"] + t["S"] + t["SE"] + 1

        # The top fits in a 2x2 block unless it has SW and also E or SE.
        is_peak = candidate & ~(tied["SW"] & (tied["E"] | tied["SE"]))
        for k, d in later.items():
            is_peak = is_peak & (~tied[k] | (shifted(n_ge, *d) == n_top - 1))

        zeros = torch.zeros_like(n_top)
        offset_x = torch.where(is_peak, (t["E"] + t["SE"] - t["SW"]) / n_top, zeros)
        offset_y = torch.where(is_peak, (t["SW"] + t["S"] + t["SE"]) / n_top, zeros)
        return is_peak, offset_x, offset_y

    @staticmethod
    def _topk_local_peaks(
        confmaps: torch.Tensor, k: int, peak_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K local peaks over the last two dims of ``(..., height, width)``.

        A flat top's point is the mean of its pixels, so it can be fractional
        (e.g. ``x + 0.5``). The wrappers do no sub-pixel refinement: they scale
        the points by the output stride and sample maps at them with bilinear
        ``grid_sample``, so fractional points need no rounding.
        """
        height, width = confmaps.shape[-2:]
        lead = confmaps.shape[:-2]
        is_peak, offset_x, offset_y = BaseExportWrapper._local_peaks(
            confmaps, peak_threshold
        )

        confmaps_flat = confmaps.reshape(*lead, height * width)
        is_peak_flat = is_peak.reshape(*lead, height * width)
        masked = torch.where(
            is_peak_flat, confmaps_flat, torch.full_like(confmaps_flat, -1e9)
        )
        values, indices = torch.topk(masked, k=k, dim=-1)

        y = indices // width
        x = indices % width
        dx = offset_x.reshape(*lead, height * width).gather(-1, indices)
        dy = offset_y.reshape(*lead, height * width).gather(-1, indices)
        peaks = torch.stack([x.float() + dx, y.float() + dy], dim=-1)
        valid = values > peak_threshold
        return peaks, values, valid

    @staticmethod
    def _find_topk_peaks(
        confmaps: torch.Tensor, k: int, peak_threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K local peaks of single-channel ``(batch, 1, height, width)`` maps.

        Same peak rule as ``find_local_peaks_rough`` (see ``_local_peaks``).
        """
        batch_size, _, height, width = confmaps.shape
        return BaseExportWrapper._topk_local_peaks(
            confmaps.reshape(batch_size, height, width), k, peak_threshold
        )

    @staticmethod
    def _find_topk_peaks_per_node(
        confmaps: torch.Tensor, k: int, peak_threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K local peaks per channel of ``(batch, nodes, height, width)``.

        Same peak rule as ``find_local_peaks_rough`` (see ``_local_peaks``).
        """
        return BaseExportWrapper._topk_local_peaks(confmaps, k, peak_threshold)

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
