"""Base classes and shared helpers for export wrappers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from sleap_nn.inference.ops.peaks import morphological_dilation_split

#: Frame heights / widths up to which :func:`resized_size_rule` is checked.
MAX_RESIZE_INPUT = 1 << 15


def resized_size_rule(scale: float) -> Tuple[int, int, bool]:
    """Integers ``(p, q)`` with ``(n * p) // q == int(n * scale)``, and whether exact.

    The checkpoint path sizes a resized frame with Python's ``int(n * scale)``
    (double precision). An exported graph computes the size from the input's shape
    with integer arithmetic, which TensorRT resolves while building (a float product
    would not floor like the double one: ``int(10 * 0.7)`` is 7, ``10 * 0.7`` in
    float32 floors to 6). The smallest ``q`` (up to 10,000) whose ``p / q`` gives
    ``int(n * scale)`` for every ``n`` up to :data:`MAX_RESIZE_INPUT` is used. A
    scale whose double rounding no fraction reproduces (0.7, 0.35, ...) gets the
    nearest fraction, and sizes that can differ from ``int(n * scale)`` by one pixel
    for some ``n`` (``exact`` is ``False``).

    Args:
        scale: The input scale (> 0).

    Returns:
        ``(p, q, exact)``.
    """
    import math
    from fractions import Fraction

    import numpy as np

    n = np.arange(MAX_RESIZE_INPUT + 1, dtype=np.int64)
    want = np.floor(n.astype(np.float64) * scale).astype(np.int64)
    head = slice(0, 1001)  # most candidates fail early: check small sizes first
    for q in range(1, 10_001):
        for p in {math.floor(scale * q), math.ceil(scale * q)}:
            if (
                p > 0
                and np.array_equal((n[head] * p) // q, want[head])
                and np.array_equal((n * p) // q, want)
            ):
                return p, q, True
    f = Fraction(scale).limit_denominator(10_000)
    return f.numerator, f.denominator, False


#: Smallest frame height / width the resize's tap count is sized for
#: (:func:`resize_taps`); smaller inputs are resized with fewer taps than they need.
MIN_RESIZE_INPUT = 32


def resize_taps(p: int, q: int) -> int:
    """Taps per output pixel the resize by ``p / q`` needs, for every input size.

    Output pixel ``i`` of ``O`` from ``N`` input pixels reads at most
    ``ceil(2 * max(N, O) / O) + 1`` taps (the triangle filter's support is
    ``max(N / O, 1)`` on each side). The maximum over every ``N`` from
    :data:`MIN_RESIZE_INPUT` to :data:`MAX_RESIZE_INPUT` is fixed at export, so the
    graph adds that many gathers one at a time.
    """
    import numpy as np

    n = np.arange(MIN_RESIZE_INPUT, MAX_RESIZE_INPUT + 1, dtype=np.int64)
    out = np.maximum((n * p) // q, 1)
    big = np.maximum(n, out)
    return int(((2 * big + out - 1) // out).max()) + 1


def _antialiased_resize_last_axis(
    x: torch.Tensor, out_size: torch.Tensor, n_taps: int
) -> torch.Tensor:
    """Antialiased bilinear resize of the last axis of ``x`` to ``out_size``.

    The same weights as ``F.interpolate(..., mode="bilinear", align_corners=False,
    antialias=True)`` (a triangle filter stretched by the downscale factor, weights
    renormalized over the taps inside the input), written with integer index
    arithmetic, gathers and sums, which ONNX Runtime and TensorRT both run (TensorRT
    parses no antialiased ``Resize``). Output pixel ``i`` of ``O`` from an input of
    ``N`` pixels is centered at ``(i + 0.5) * N / O``; with ``M = max(N, O)``, tap
    ``j``'s weight is ``max(0, 1 - |(2j + 1) O - (2i + 1) N| / (2M))``, computed from
    exact integers, so large coordinates lose no precision in reduced-precision
    engines. The taps are gathered and added one at a time, so the intermediates are
    the size of the output.

    Args:
        x: ``(..., N)`` float tensor.
        out_size: 0-d ``int64`` tensor, the output length ``O`` (>= 1).
        n_taps: Taps per output pixel (:func:`resize_taps`).

    Returns:
        ``(..., O)`` tensor.
    """
    # Sizes come from Shape ops (not from data), so TensorRT can resolve every
    # tensor shape below from the input's shape.
    n_in = torch._shape_as_tensor(x)[-1].to(x.device)  # N, int64
    out = out_size.to(device=x.device, dtype=torch.int64)
    big = torch.maximum(n_in, out)  # M: support * O
    i = torch.arange(out, device=x.device)  # (O,)
    # First tap of each output pixel: floor(center - support + 0.5), clamped at 0
    # (taps before 0 have no weight; clamping keeps the numerator non-negative, so
    # integer division truncating or flooring gives the same).
    first = torch.clamp((2 * i + 1) * n_in + out - 2 * big, min=0) // (2 * out)
    center2 = (2 * i + 1) * n_in  # 2 * O * center, exact
    span = (2 * big).float()
    total = None
    acc = None
    for k in range(n_taps):
        tap = first + k  # (O,)
        dist = torch.abs((2 * tap + 1) * out - center2)
        w = torch.clamp(1.0 - dist.float() / span, min=0.0) * (tap < n_in).float()
        part = torch.index_select(x, -1, torch.minimum(tap, n_in - 1)) * w
        acc = part if acc is None else acc + part
        total = w if total is None else total + w
    return acc / total


def antialiased_resize(
    image: torch.Tensor, p: int, q: int, n_taps: int
) -> torch.Tensor:
    """Resize ``(..., H, W)`` images by a scale, antialiased, as the checkpoint path.

    Matches ``torchvision.transforms.functional.resize`` / ``F.interpolate(...,
    mode="bilinear", align_corners=False, antialias=True)`` to ``(int(H * scale),
    int(W * scale))`` up to float rounding. Height and width are read from the
    input's shape when the graph runs (traced graphs keep them dynamic), and the
    output size is ``(H * p) // q``, ``(W * p) // q`` (at least 1).

    Args:
        image: ``(..., H, W)`` float images.
        p: Numerator of :func:`resized_size_rule` of the scale.
        q: Its denominator.
        n_taps: :func:`resize_taps` of ``p``, ``q``.

    Returns:
        The resized images.
    """
    out_hw = torch.clamp(
        torch._shape_as_tensor(image)[-2:].to(image.device) * p // q, min=1
    )
    image = _antialiased_resize_last_axis(image, out_hw[1], n_taps)
    image = _antialiased_resize_last_axis(image.transpose(-1, -2), out_hw[0], n_taps)
    return image.transpose(-1, -2)


def input_scale_resize_rule(input_scale: float) -> Optional[Tuple[int, int, int]]:
    """The resize rule ``(p, q, n_taps)`` of an input scale, or ``None`` for 1.0.

    Precomputed once per wrapper (:func:`resized_size_rule` and
    :func:`resize_taps`), for :func:`antialiased_resize`. A scale whose size rule is
    not exact gets a warning: the graph's resized size can then be one pixel off
    the checkpoint path's for some frame sizes.

    Args:
        input_scale: The model's input scale.

    Returns:
        ``(p, q, n_taps)``, or ``None`` when ``input_scale`` is 1.0 (no resize).
    """
    if input_scale == 1.0:
        return None
    p, q, exact = resized_size_rule(input_scale)
    if not exact:
        logger.warning(
            f"input_scale {input_scale}: no fraction reproduces Python's "
            f"int(size * {input_scale}) for every size; the exported graph "
            f"sizes the resized frame as size * {p} // {q}, which can differ "
            "from the checkpoint path by one pixel for some frame sizes."
        )
    return p, q, resize_taps(p, q)


def antialiased_resize_to(
    image: torch.Tensor, in_hw: Tuple[int, int], out_hw: Tuple[int, int]
) -> torch.Tensor:
    """Antialiased resize of ``(..., H, W)`` images of a fixed size to a fixed size.

    For inputs whose size is fixed by the graph (top-down crops of ``crop_size``):
    the output size is the checkpoint path's own ``(int(H * scale), int(W * scale))``,
    passed in, so no size rule is involved. Same weights as
    :func:`antialiased_resize`.

    Args:
        image: ``(..., H, W)`` float images with ``(H, W) == in_hw``.
        in_hw: The input's height and width.
        out_hw: The output's height and width (each >= 1).

    Returns:
        The resized images.
    """

    def taps(n: int, o: int) -> int:
        return -(-2 * max(n, o) // o) + 1

    # Sizes and tap counts are Python ints (no tensor-to-int in the trace).
    out_h, out_w = (max(int(v), 1) for v in out_hw)
    taps_h, taps_w = taps(int(in_hw[0]), out_h), taps(int(in_hw[1]), out_w)
    image = _antialiased_resize_last_axis(
        image, torch.full((), out_w, dtype=torch.int64), taps_w
    )
    image = _antialiased_resize_last_axis(
        image.transpose(-1, -2), torch.full((), out_h, dtype=torch.int64), taps_h
    )
    return image.transpose(-1, -2)


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
