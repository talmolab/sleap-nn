"""Single-level projector of the ``bbox_detr`` detector.

Port of HF transformers' ``RfDetrScaleProjector`` / ``RfDetrC2FLayer``
(``models/rf_detr``, transformers 5.20.2): the backbone's tapped token maps are
concatenated on channels, mixed by a YOLOv8-style C2f block and layer-normalized,
giving one ``hidden_dim`` map at the token stride. Module and parameter names
follow HF's so RF-DETR weights load by renaming prefixes only.
"""

from typing import Sequence

import torch
from torch import nn

_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a ``(B, C, H, W)`` map (HF ``RfDetrLayerNorm``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its channel axis."""
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNormLayer(nn.Module):
    """Bias-free conv, channel LayerNorm and activation (HF ``RfDetrConvNormLayer``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "silu",
        eps: float = 1e-5,
    ):
        """Initialize the layer.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Square kernel size (stride 1, "same" padding).
            activation: ``"silu"``, ``"relu"`` or ``"gelu"``.
            eps: LayerNorm epsilon.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = LayerNorm2d(out_channels, eps=eps)
        self.activation = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv, norm and activation."""
        return self.activation(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    """Two 3x3 ``ConvNormLayer``s, no residual (HF ``RfDetrRepVggBlock``)."""

    def __init__(self, channels: int, activation: str = "silu", eps: float = 1e-5):
        """Initialize the block with ``channels`` in and out."""
        super().__init__()
        self.conv1 = ConvNormLayer(channels, channels, 3, activation, eps)
        self.conv2 = ConvNormLayer(channels, channels, 3, activation, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both convs."""
        return self.conv2(self.conv1(x))


class C2FLayer(nn.Module):
    """CSP block with all intermediate outputs concatenated (HF ``RfDetrC2FLayer``).

    ``conv1`` (1x1) splits the input into two halves of ``hidden`` channels; the
    second half runs through ``num_blocks`` bottlenecks in sequence, and every
    piece (both halves and each bottleneck output) is concatenated and fused by
    ``conv2`` (1x1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 0.5,
        activation: str = "silu",
        eps: float = 1e-5,
    ):
        """Initialize the layer.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            num_blocks: Number of bottlenecks.
            expansion: Hidden width as a fraction of ``out_channels``.
            activation: Activation of every conv.
            eps: LayerNorm epsilon of every conv.
        """
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(
            in_channels, 2 * self.hidden_channels, 1, activation, eps
        )
        self.conv2 = ConvNormLayer(
            (2 + num_blocks) * self.hidden_channels, out_channels, 1, activation, eps
        )
        self.bottlenecks = nn.ModuleList(
            RepVggBlock(self.hidden_channels, activation, eps)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split, run the bottlenecks, concatenate and fuse."""
        pieces = list(self.conv1(x).split(self.hidden_channels, dim=1))
        x = pieces[-1]
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
            pieces.append(x)
        return self.conv2(torch.cat(pieces, dim=1))


class ScaleProjector(nn.Module):
    """Concatenated backbone taps -> C2f -> LayerNorm (HF ``RfDetrScaleProjector``).

    Attributes:
        in_channels: Total input channels (taps x backbone width).
        out_channels: Width of the projected map (the decoder's ``hidden_dim``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        num_blocks: int = 3,
        expansion: float = 0.5,
        activation: str = "silu",
        eps: float = 1e-5,
    ):
        """Initialize the projector.

        Args:
            in_channels: Channels of the concatenated taps.
            out_channels: Output channels.
            num_blocks: Bottlenecks in the C2f block.
            expansion: C2f hidden width as a fraction of ``out_channels``.
            activation: C2f activation.
            eps: LayerNorm epsilon inside the C2f block. The final LayerNorm uses
                ``1e-6``, as in HF.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projector_layer = C2FLayer(
            in_channels, out_channels, num_blocks, expansion, activation, eps
        )
        self.layer_norm = LayerNorm2d(out_channels, eps=1e-6)

    def forward(self, taps: Sequence[torch.Tensor]) -> torch.Tensor:
        """Project ``taps`` (each ``(B, D, h, w)``) to one ``(B, out, h, w)`` map."""
        return self.layer_norm(self.projector_layer(torch.cat(list(taps), dim=1)))
