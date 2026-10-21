"""Feature-pyramid neck that lets a plain (isotropic) ViT feed spatial heads.

A plain ViT -- DINOv2/v3, CLIP, SigLIP, MAE, ... -- emits every layer at a single
stride (its patch size). The sleap-nn decoder needs a multi-scale pyramid to
upsample through, so this neck turns one or more single-stride token maps into
four levels at strides ``s/4, s/2, s, 2s`` (``s`` = the token stride), in the
style of ViTDet's Simple Feature Pyramid (Li et al. 2022) and DPT's reassemble
stage (Ranftl et al. 2021):

* ``"dpt"`` -- one tapped transformer layer per level, shallowest to the finest
  level (the default: intermediate layers keep more spatial detail).
* ``"sfp"`` -- every level is built from the last tapped layer alone (ViTDet).

The pyramid then feeds the existing
:class:`~sleap_nn.architectures.encoder_decoder.Decoder` exactly like the stage
outputs of a hierarchical backbone, so heads need no changes.
"""

from functools import partial
from typing import List, Sequence

import torch
from torch import nn
from torchvision.models.convnext import LayerNorm2d as _TorchvisionLayerNorm2d

# Relative scale of each pyramid level w.r.t. the token grid: x4, x2, x1, x0.5.
PYRAMID_SCALES = (4.0, 2.0, 1.0, 0.5)

# LayerNorm over the channels of an NCHW map (as the native ConvNeXt backbone uses).
LayerNorm2d = partial(_TorchvisionLayerNorm2d, eps=1e-6)


def _resample(in_channels: int, scale: float) -> nn.Module:
    """Build the resampling op that takes the token grid to one pyramid level."""
    if scale == 4.0:
        mid = max(in_channels // 2, 1)
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid, kernel_size=2, stride=2),
            LayerNorm2d(mid),
            nn.GELU(),
            nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
        )
    if scale == 2.0:
        return nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
    if scale == 1.0:
        return nn.Identity()
    if scale == 0.5:
        return nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
    raise ValueError(f"Unsupported pyramid scale: {scale}")


def _resampled_channels(in_channels: int, scale: float) -> int:
    """Number of channels `_resample` emits at a given scale."""
    return max(in_channels // 2, 1) if scale == 4.0 else in_channels


class ViTPyramidNeck(nn.Module):
    """Build a feature pyramid from single-stride ViT feature maps.

    Args:
        in_channels: Channel count of the ViT feature maps (its hidden size).
        out_channels: Channels of each output level, finest first (one per scale).
        num_inputs: Number of tapped layers fed to `forward`. ``"dpt"`` needs one
            per level; ``"sfp"`` uses only the last one.
        mode: ``"dpt"`` (one tapped layer per level) or ``"sfp"`` (the last
            tapped layer for every level).
        scales: The levels to build, as scale factors of the token grid, finest
            first -- a suffix of `PYRAMID_SCALES`. Levels finer than the model's
            output stride are left out: the decoder would never read them, and
            unread levels are parameters without a gradient (a DDP error).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        num_inputs: int = 4,
        mode: str = "dpt",
        scales: Sequence[float] = PYRAMID_SCALES,
    ) -> None:
        """Initialize the neck."""
        super().__init__()
        scales = [float(sc) for sc in scales]
        if mode not in ("dpt", "sfp"):
            raise ValueError(f"Unknown ViT neck mode {mode!r}; use 'dpt' or 'sfp'.")
        if len(out_channels) != len(scales):
            raise ValueError(
                f"ViT neck needs {len(scales)} output channel counts (one per level "
                f"at scales {scales}), got {list(out_channels)}."
            )
        if mode == "dpt" and num_inputs != len(scales):
            raise ValueError(
                f"The 'dpt' ViT neck taps one layer per pyramid level, so it needs "
                f"{len(scales)} layers, got {num_inputs}."
            )
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.scales = scales

        self.levels = nn.ModuleList()
        for scale, c_out in zip(self.scales, self.out_channels):
            c_mid = _resampled_channels(in_channels, scale)
            self.levels.append(
                nn.Sequential(
                    _resample(in_channels, scale),
                    nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False),
                    LayerNorm2d(c_out),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(c_out),
                )
            )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Map tapped token maps to the pyramid, finest level first.

        Args:
            feats: Tapped ViT feature maps ``(B, in_channels, h, w)``, shallowest
                first.

        Returns:
            One map per level, at relative scales `self.scales` of the token grid.
        """
        if self.mode == "sfp":
            inputs = [feats[-1]] * len(self.levels)
        else:
            inputs = feats
        return [level(f) for level, f in zip(self.levels, inputs)]


class DetailStem(nn.Module):
    """Light convolutional stream giving a plain ViT high-resolution skips.

    A plain ViT's finest real features sit at its token stride (16), so the
    pyramid levels the neck synthesizes below it carry no image detail of their
    own. This stem (in the spirit of ViTMatte's detail-capture stream, Yao et al.
    2024) reads the image directly and emits features at strides 2, 4 and 8,
    which are concatenated with the neck levels at the same stride (or added as
    new, finer levels) before the decoder.

    Args:
        in_channels: Input image channels.
        channels: Output channels at strides 2, 4, 8 (in that order).
    """

    def __init__(self, in_channels: int = 3, channels: List[int] = (32, 64, 128)):
        """Initialize the stem."""
        super().__init__()
        self.channels = list(channels)
        self.stages = nn.ModuleList()
        c_in = in_channels
        for c_out in self.channels:
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                    LayerNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    LayerNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            )
            c_in = c_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return features at strides 2, 4, 8 (finest first)."""
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
