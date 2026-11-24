"""Backbones of the ``bbox_detr`` detector.

A DETR backbone maps an image to the token maps of a few tapped blocks, all at one
stride and width::

    forward(x: (B, C, H, W)) -> [(B, out_channels, H / stride, W / stride), ...]

RF-DETR taps four blocks of a ViT (blocks 3, 6, 9 and 12 of DINOv2-S) at the patch
stride, and the projector concatenates them. Backbones are looked up by
``model_config.backbone_config`` type in :data:`DETR_BACKBONES`:
``dinov2_windowed`` (RF-DETR's DINOv2, :mod:`.dinov2_windowed`) and ``pretrained``
(any HF/timm plain ViT -- DINOv3, DINOv2 -- through sleap-nn's pretrained loader,
:mod:`.pretrained_vit`). This module also ships :class:`TinyDetrBackbone`, a small
convolutional stand-in for tests, and :func:`rfdetr_layer_decay_groups`, the
layer-wise learning rates both ViT backbones give the optimizer.
"""

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch
from omegaconf import DictConfig
from torch import nn


@runtime_checkable
class DetrBackbone(Protocol):
    """Interface of a ``bbox_detr`` backbone.

    Attributes:
        out_channels: Channels of each tapped token map.
        num_taps: Number of tapped maps ``forward`` returns.
        stride: Stride of the token maps relative to the input image.
    """

    out_channels: int
    num_taps: int
    stride: int

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Token maps of the tapped blocks, ``num_taps`` x ``(B, D, H/s, W/s)``."""
        ...


class TinyDetrBackbone(nn.Module):
    """A small convolutional backbone with the DETR backbone interface, for tests.

    A ``stride`` x ``stride`` patch embedding followed by ``num_blocks`` residual
    3x3 conv blocks; the outputs of the last ``num_taps`` blocks are tapped.

    Attributes:
        out_channels: Channels of each tapped map.
        num_taps: Number of tapped maps.
        stride: Patch size (the stride of every tapped map).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        stride: int = 16,
        num_blocks: int = 2,
        num_taps: int = 2,
    ):
        """Initialize the backbone.

        Args:
            in_channels: Input image channels.
            out_channels: Width of the token maps.
            stride: Patch size.
            num_blocks: Residual conv blocks after the patch embedding.
            num_taps: Number of trailing blocks whose outputs are returned.

        Raises:
            ValueError: If ``num_taps`` is not in ``[1, num_blocks]``.
        """
        super().__init__()
        if not 1 <= num_taps <= num_blocks:
            raise ValueError(
                f"num_taps must be in [1, num_blocks={num_blocks}], got {num_taps}."
            )
        self.out_channels = out_channels
        self.num_taps = num_taps
        self.stride = stride
        self.patch_embed = nn.Conv2d(in_channels, out_channels, stride, stride)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the last ``num_taps`` block outputs, each ``(B, D, H/s, W/s)``."""
        x = self.patch_embed(x)
        taps = []
        for block in self.blocks:
            x = x + block(x)
            taps.append(x)
        return taps[-self.num_taps :]


def rfdetr_layer_decay_groups(
    depths: Iterable[Tuple[torch.Tensor, int]],
    base_lr: float,
    num_layers: int,
    encoder_lr_scale: Optional[float] = None,
    layer_decay: Optional[float] = None,
) -> Optional[List[Dict]]:
    """Optimizer groups with RF-DETR's layer-wise learning-rate decay.

    RF-DETR (``get_dinov2_lr_decay_rate``) counts ``D = max(out_indices) + 1``
    layers (``num_layers`` here) and gives a parameter at depth ``d`` the rate
    ``layer_decay ** (D + 1 - d)``: the embeddings are depth 0, block ``k``
    (0-based) depth ``k + 1``, and what follows the blocks (the final norm) depth
    ``D + 1``.

    Args:
        depths: ``(parameter, depth)`` pairs of the trainable parameters.
        base_lr: The backbone's learning rate before layer decay.
        num_layers: RF-DETR's ``D``.
        encoder_lr_scale: Multiplies ``base_lr`` (``None`` = 1).
        layer_decay: The per-layer decay; ``None`` or 0: no decay.

    Returns:
        ``None`` when every parameter trains at ``base_lr`` (no scale, no decay),
        else ``{"params", "lr", "name"}`` groups (``depth<d>``, or one ``all``).
    """
    scale = 1.0 if encoder_lr_scale is None else float(encoder_lr_scale)
    if scale == 1.0 and not layer_decay:
        return None
    depths = list(depths)
    if not layer_decay:
        return [
            {"params": [p for p, _ in depths], "lr": base_lr * scale, "name": "all"}
        ]
    by_depth: Dict[int, List[torch.Tensor]] = {}
    for param, depth in depths:
        by_depth.setdefault(depth, []).append(param)
    return [
        {
            "params": params,
            "lr": base_lr * scale * layer_decay ** (num_layers + 1 - depth),
            "name": f"depth{depth}",
        }
        for depth, params in sorted(by_depth.items())
    ]


#: Builders of DETR backbones by ``backbone_config`` type, each taking the type's
#: config leaf. See :func:`register_detr_backbone`.
DETR_BACKBONES: Dict[str, Callable[[DictConfig], nn.Module]] = {}


def register_detr_backbone(
    name: str,
) -> Callable[[Callable[[DictConfig], nn.Module]], Callable[[DictConfig], nn.Module]]:
    """Register a DETR backbone builder for ``backbone_config`` type ``name``.

    Args:
        name: The backbone type (a key of ``model_config.backbone_config``).

    Returns:
        A decorator that registers the builder and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(
        fn: Callable[[DictConfig], nn.Module],
    ) -> Callable[[DictConfig], nn.Module]:
        if name in DETR_BACKBONES:
            raise ValueError(f"DETR backbone '{name}' is already registered.")
        DETR_BACKBONES[name] = fn
        return fn

    return decorator


def build_detr_backbone(backbone_type: str, backbone_config: DictConfig) -> nn.Module:
    """Build the DETR backbone for ``backbone_type`` from its config leaf.

    Args:
        backbone_type: Key of ``model_config.backbone_config`` that is set.
        backbone_config: That key's config.

    Returns:
        A module with the :class:`DetrBackbone` interface.

    Raises:
        ValueError: If no DETR backbone is registered for ``backbone_type``.
    """
    if backbone_type not in DETR_BACKBONES:
        raise ValueError(
            f"Backbone '{backbone_type}' has no bbox_detr adapter. Registered DETR "
            f"backbones: {sorted(DETR_BACKBONES) or 'none'}."
        )
    return DETR_BACKBONES[backbone_type](backbone_config)
