"""Torch-native tiled inference: importance windows + accumulate/normalize canvas.

This module provides the two primitives needed to run a fixed-input-size model
over an image that is larger than the model's receptive field by splitting it
into overlapping tiles and stitching the per-tile predictions back into a single
full-frame output.

- :func:`build_importance_window` — a separable per-axis weighting window for one
  tile that de-emphasizes the tile borders (where predictions are least
  reliable). This is a pure-torch port of MONAI's gaussian importance map, plus a
  ``"pyramid"`` (triangular) window ported from pytorch-toolbelt and a
  ``"constant"`` (uniform) fallback.
- :class:`TileMerger` — a per-frame accumulate-and-normalize canvas (a torch-only
  port of pytorch-toolbelt's ``TileMerger``). Each tile is added into a weighted
  accumulator ``ACC`` and its window into a weight counter ``CNT``; the final
  merge is the elementwise ``ACC / CNT``, i.e. a per-pixel weighted average of
  every tile that covered it. With sum-of-weights normalization a uniform field
  stitches back to itself everywhere it is covered, independent of the window.

All coordinates and windows are expressed in **output-stride** pixels (the
resolution of the model's output canvas), not input pixels. No dependencies
beyond ``torch``.
"""

from typing import Optional, Tuple, Union

import torch


def build_importance_window(
    tile_hw: Tuple[int, int],
    mode: str = "gaussian",
    sigma_scale: float = 0.125,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a separable per-axis importance window for one tile.

    The window weights each pixel of a tile by how far it is from the tile
    border, so that overlapping tiles contribute most where they are most
    reliable (near their center) and least at their edges. It is built once per
    tile size and reused across every tile/frame.

    The window is **not** sum-normalized — normalization happens at merge time
    via the ``ACC``/``CNT`` accumulators in :class:`TileMerger`.

    Args:
        tile_hw: ``(th, tw)`` tile height and width, in output-stride pixels.
        mode: One of ``"gaussian"``, ``"pyramid"``, or ``"constant"``.

            - ``"gaussian"``: separable Gaussian bump, peak ``1.0`` at the
              center, with per-axis std ``sigma_scale * axis_length``.
            - ``"pyramid"``: separable triangular ramp (distance to the nearest
              edge along each axis), normalized to peak ``1.0`` at the center.
            - ``"constant"``: uniform weights (all ones).
        sigma_scale: Gaussian std as a fraction of each axis length. Only used
            for ``mode="gaussian"``.
        device: Device to build the window on.
        dtype: Output dtype of the returned window.

    Returns:
        A ``(th, tw)`` tensor with peak ``1.0`` at the center, clamped so the
        minimum covered weight is ``>= 1e-3``.

    Raises:
        ValueError: If ``mode`` is not one of the supported modes.
    """
    th, tw = tile_hw

    if mode == "gaussian":
        # MONAI-style separable gaussian importance map. Per-axis std.
        sy = sigma_scale * th
        sx = sigma_scale * tw
        # Centered coordinates: length th (resp. tw), symmetric about 0.
        y = torch.arange(-(th - 1) / 2.0, (th - 1) / 2.0 + 1, device=device)
        x = torch.arange(-(tw - 1) / 2.0, (tw - 1) / 2.0 + 1, device=device)
        gy = torch.exp(y**2 / (-2 * sy**2))
        gx = torch.exp(x**2 / (-2 * sx**2))
        # Separable outer product. Peak is 1.0 at the center by construction.
        w = gy[:, None] * gx[None, :]
    elif mode == "pyramid":
        # Separable triangular window: distance to the nearest edge along each
        # axis, normalized so the center peaks at 1.0 (port of pytorch-toolbelt's
        # compute_pyramid_patch_weight_loss, reduced to its separable form).
        iy = torch.arange(1, th + 1, device=device, dtype=torch.float32)
        ix = torch.arange(1, tw + 1, device=device, dtype=torch.float32)
        ry = torch.minimum(iy, th + 1 - iy)  # 1..ceil(th/2)..1
        rx = torch.minimum(ix, tw + 1 - ix)  # 1..ceil(tw/2)..1
        gy = ry / ry.max()  # peak 1.0 per axis
        gx = rx / rx.max()
        w = gy[:, None] * gx[None, :]
    elif mode == "constant":
        w = torch.ones((th, tw), device=device)
    else:
        raise ValueError(
            f"Unknown importance window mode: {mode!r}. "
            "Expected 'gaussian', 'pyramid', or 'constant'."
        )

    # Clamp away vanishing/zero weights so every covered pixel gets a finite,
    # strictly positive contribution (>= 1e-3). Not sum-normalized.
    min_non_zero = max(w.min().item(), 1e-3)
    w = torch.clamp(w, min=min_non_zero).to(dtype)
    return w


class TileMerger:
    """Per-frame accumulate-and-normalize canvas at output-stride resolution.

    Each integrated tile is added into a weighted accumulator ``ACC`` and its
    window into a weight counter ``CNT``. The final merge divides ``ACC`` by
    ``CNT`` elementwise, yielding a per-pixel weighted average over every tile
    that covered that pixel.

    Attributes:
        w: The importance window, shape ``(1, th, tw)``.
        acc: Weighted accumulator, shape ``(channels, H, W)``, float.
        cnt: Weight counter, shape ``(1, H, W)``, float.

    All coordinates ``(y0, x0)`` and the window are in output-stride pixels.
    Accumulation is done in ``dtype`` (float32 by default) even when tiles are
    passed in at lower precision (e.g. fp16).
    """

    def __init__(
        self,
        out_hw: Tuple[int, int],
        channels: int,
        window: torch.Tensor,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the accumulator canvas.

        Args:
            out_hw: ``(H, W)`` canvas size in output-stride pixels.
            channels: Number of output channels to accumulate.
            window: Importance window of shape ``(th, tw)`` from
                :func:`build_importance_window`.
            device: Device to hold the accumulators on.
            dtype: Accumulation dtype (float32 recommended).
        """
        H, W = out_hw
        self.w = window.to(device=device, dtype=dtype)[None]  # (1, th, tw)
        self.acc = torch.zeros((channels, H, W), device=device, dtype=dtype)
        self.cnt = torch.zeros((1, H, W), device=device, dtype=dtype)

    def integrate(self, tile: torch.Tensor, y0: int, x0: int) -> None:
        """Accumulate one tile at output-stride origin ``(y0, x0)``.

        The tile is moved to the accumulator device/dtype before accumulation,
        so fp16 tiles are accumulated in float32. If the tile is partial (clipped
        by the canvas edge), the window is cropped to match.

        Args:
            tile: Tile of shape ``(channels, th, tw)`` at output stride.
            y0: Top row of the tile within the canvas, in output-stride pixels.
            x0: Left column of the tile within the canvas, in output-stride
                pixels.
        """
        tile = tile.to(self.acc.device, self.acc.dtype)
        _, th, tw = tile.shape
        w = self.w[:, :th, :tw]  # guard partials (clipped tiles)
        self.acc[:, y0 : y0 + th, x0 : x0 + tw] += tile * w
        self.cnt[:, y0 : y0 + th, x0 : x0 + tw] += w

    def merge(self, eps: Optional[float] = None) -> torch.Tensor:
        """Normalize the accumulator by the weight counter.

        Args:
            eps: If given, clamp the counter to at least ``eps`` before dividing
                (avoids NaNs on uncovered pixels). If ``None``, divide directly;
                uncovered pixels (counter ``0``) become NaN.

        Returns:
            The merged output of shape ``(channels, H, W)``.
        """
        denom = self.cnt if eps is None else torch.clamp(self.cnt, min=eps)
        return self.acc / denom  # (channels, H, W)
