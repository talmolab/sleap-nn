"""A pretrained HF/timm plain ViT (DINOv3, DINOv2, ...) as a ``bbox_detr`` backbone.

:class:`PretrainedDetrBackbone` is sleap-nn's pretrained-encoder loader
(:class:`~sleap_nn.architectures.pretrained.PretrainedBackbone`) in its
``mode="tokens"``: the same checkpoint loading (HF ``AutoModel`` or ``timm/...``
through ``TimmWrapperModel``), normalization, drop path, gradient checkpointing,
pretrained-weight snapshot and freezing as the pose models' ``pretrained``
backbone, but ``forward`` returns the tapped layers as token maps for the DETR
projector instead of feeding a neck and decoder.

* **Taps.** Four evenly spaced blocks ending at the last one by default (3, 6, 9,
  12 of a 12-block ViT-S/B, 6, 12, 18, 24 of a 24-block ViT-L), as RF-DETR taps
  DINOv2. Each tap goes through the encoder's pretrained final norm.
* **Token grid.** Prefix tokens (CLS, registers) are dropped; the maps are at
  ``token_stride`` (16; a patch-14 DINOv2 input is resampled onto a 16-px grid).
  Attention is global; DINOv3's RoPE (and DINOv2's interpolated position
  embedding) handle any input size.
* **Optimizer.** :meth:`encoder_param_groups` gives RF-DETR's layer-wise
  learning-rate decay (``D = max(layers) + 1``), so ``layer_decay`` means what it
  means for ``dinov2_windowed``. Blocks after the last tap are frozen (they feed
  nothing).
"""

from typing import Dict, List, Optional

import torch
from loguru import logger
from omegaconf import DictConfig

from sleap_nn.architectures.detr.backbones import (
    register_detr_backbone,
    rfdetr_layer_decay_groups,
)
from sleap_nn.architectures.pretrained import PretrainedBackbone, encoder_param_depths


class PretrainedDetrBackbone(PretrainedBackbone):
    """A pretrained plain ViT's tapped token maps, for ``bbox_detr``.

    Implements the :class:`~sleap_nn.architectures.detr.backbones.DetrBackbone`
    interface. Takes :class:`PretrainedBackbone`'s arguments; ``mode`` is always
    ``"tokens"``.

    Attributes:
        out_channels: Token width (``hidden_size``).
        num_taps: Number of tapped layers.
        stride: The token stride.
    """

    def __init__(self, *args, mode: str = "tokens", **kwargs):
        """Build the encoder and its taps (see :class:`PretrainedBackbone`)."""
        if mode not in ("auto", "tokens"):
            logger.warning(
                f"pretrained.mode={mode!r} does not apply to a bbox_detr backbone, "
                "which always returns the tapped token maps (mode 'tokens')."
            )
        kwargs.pop("spatial_heads", None)
        super().__init__(*args, mode="tokens", **kwargs)
        self.out_channels = self.hidden_size
        self.num_taps = len(self.layers)
        self.stride = self.token_stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Token maps of the tapped layers.

        Args:
            x: ``(B, C, H, W)`` images in ``[0, 1]``; ``H`` and ``W`` must be
                multiples of :attr:`stride`.

        Returns:
            ``num_taps`` maps, each ``(B, hidden_size, H / stride, W / stride)``.

        Raises:
            ValueError: If the input size is not a multiple of the stride.
        """
        height, width = x.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(
                f"The pretrained bbox_detr backbone needs input sides that are "
                f"multiples of {self.stride}, got {height}x{width}."
            )
        return self.token_maps(x)

    def encoder_param_groups(
        self,
        base_lr: float,
        encoder_lr_scale: Optional[float] = None,
        layer_decay: Optional[float] = None,
    ) -> Optional[List[Dict]]:
        """Optimizer groups with RF-DETR's layer-wise learning-rate decay.

        Depths follow :func:`~sleap_nn.architectures.pretrained.
        encoder_param_depths` (embeddings and tokens 0, block ``k`` ``k + 1``),
        with RF-DETR's count ``D = max(layers) + 1``: the final norm and any tap
        norm outside the encoder are depth ``D + 1`` (undecayed). See
        :func:`~sleap_nn.architectures.detr.backbones.rfdetr_layer_decay_groups`.

        Args:
            base_lr: The backbone's learning rate before layer decay.
            encoder_lr_scale: Multiplies ``base_lr``; ``None`` keeps the config's
                ``encoder_lr_scale``.
            layer_decay: The per-layer decay; ``None`` keeps the config's
                ``layer_decay`` (``None`` or 0: no decay).

        Returns:
            ``None`` when every parameter trains at ``base_lr``, else
            ``{"params", "lr", "name"}`` groups over the trainable parameters.
        """
        scale = self.encoder_lr_scale if encoder_lr_scale is None else encoder_lr_scale
        decay = self.layer_decay if layer_decay is None else layer_decay
        num_layers = max(self.layers) + 1
        enc = [(n, p) for n, p in self.enc.named_parameters() if p.requires_grad]
        depths = encoder_param_depths(enc, self.num_layers, after_depth=num_layers + 1)
        depths += [
            (p, num_layers + 1)
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("enc.")
        ]
        return rfdetr_layer_decay_groups(depths, base_lr, num_layers, scale, decay)


@register_detr_backbone("pretrained")
def build_pretrained_detr(config: DictConfig) -> PretrainedDetrBackbone:
    """Build a :class:`PretrainedDetrBackbone` from a ``PretrainedConfig`` leaf."""
    return PretrainedDetrBackbone.from_config(config)
