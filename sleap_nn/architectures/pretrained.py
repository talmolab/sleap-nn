"""Pretrained image backbones from HuggingFace `transformers` as sleap-nn encoders.

This module wires external pretrained vision encoders (via
`transformers.AutoBackbone` / the model-specific ``*Backbone`` classes) into the
existing :class:`sleap_nn.architectures.model.Model` contract, so any head
(pose, centroid, segmentation, class-vectors) can sit on a frozen or fine-tuned
foundation/ImageNet backbone. It reuses the same backbone dict-contract as the
native UNet/ConvNeXt/SwinT wrappers — no new `Model` class and no head changes.

Two integration surfaces are supported (the wrapper auto-selects, or you can
force via ``mode``):

* **Case A — hierarchical encoder + sleap decoder** (``mode="decoder"``). A
  hierarchical backbone (ConvNeXtV2, ResNet, Swinv2, DINOv3-ConvNeXt, ...) emits
  a multi-scale ``feature_maps`` pyramid (strides 4/8/16/32). We feed those maps
  as skip connections into the existing
  :class:`sleap_nn.architectures.encoder_decoder.Decoder` (exactly what
  `ConvNextWrapper`/`SwinTWrapper` do for torchvision), so spatial heads work
  unchanged. This is the primary path for pose/segmentation/centroid models.

* **Case B — encoder-only pooled bottleneck** (``mode="encoder"``). An isotropic
  ViT (DINOv2 / DINOv2-with-registers) emits a single spatial bottleneck map
  (CLS/register tokens stripped, LayerNorm applied). We expose it as
  ``middle_output`` with ``middle_blocks[-1].filters == hidden_size`` so a pooled
  head (class-vectors / re-ID / embedding) consumes it. No decoder is built.

Key behaviors baked in here (rather than in the data pipeline, which is uint8 /
[0, 1] only): model-specific mean/std normalization (read from the matching
`AutoImageProcessor`, or passed explicitly), grayscale->3ch is handled upstream
in `Model.forward`, a `float32` dtype force (transformers v5 defaults to
``dtype="auto"`` which can silently load fp16/bf16), an optional pinned
``revision``, and a one-shot **stride probe** (HF backbones do not expose
stride metadata).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from sleap_nn.architectures.encoder_decoder import Decoder

# ImageNet stats, used as a last-resort normalization default when a model does
# not ship an `AutoImageProcessor` (most HF vision models do).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Config `model_type` strings that are isotropic (single-scale) ViTs. Case A
# (hierarchical decoder) cannot be built from these; they route to Case B.
_ISOTROPIC_MODEL_TYPES = {
    "vit",
    "deit",
    "beit",
    "dinov2",
    "dinov2_with_registers",
    "dinov2-with-registers",
    "dinov3_vit",
    "dinov3-vit",
    "ijepa",
    "vitdet",
    "vit_det",
    "vit_mae",
    "vit_msn",
}


def _import_transformers():
    """Lazily import `transformers`, with an actionable error if it is missing."""
    try:
        import transformers  # noqa: F401

        return transformers
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        message = (
            "Pretrained backbones require the `transformers` package, which is an "
            "optional dependency. Install it with `pip install 'sleap-nn[backbones]'` "
            "(or `uv sync --extra backbones`)."
        )
        logger.error(message)
        raise ImportError(message) from e


class _FiltersHolder(nn.Module):
    """Parameter-free module carrying an int ``filters`` attribute.

    :class:`sleap_nn.architectures.model.Model` reads
    ``backbone.middle_blocks[-1].filters`` to size the class-vectors head. Native
    wrappers expose a real conv block here; a pretrained encoder feeds the raw
    bottleneck to the decoder/head, so this lightweight holder satisfies the
    contract without adding parameters.
    """

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.filters = int(filters)


def _resolve_mode(config: Any, mode: str) -> str:
    """Resolve ``mode="auto"`` to ``"decoder"`` (Case A) or ``"encoder"`` (Case B)."""
    if mode in ("decoder", "encoder"):
        return mode
    model_type = getattr(config, "model_type", "") or ""
    stage_names = getattr(config, "stage_names", None)
    is_isotropic = model_type.lower() in _ISOTROPIC_MODEL_TYPES or not stage_names
    return "encoder" if is_isotropic else "decoder"


class PretrainedBackbone(nn.Module):
    """Wrap a HuggingFace pretrained backbone as a sleap-nn encoder.

    Args:
        source: Backbone source. Only ``"hf"`` (HuggingFace `transformers`) is
            currently supported.
        model_name: HuggingFace model id (e.g. ``"facebook/convnextv2-nano-22k-224"``,
            ``"microsoft/resnet-50"``, ``"facebook/dinov2-with-registers-base"``).
        in_channels: Number of input channels the wrapped stem expects. Pretrained
            vision stems are 3-channel; grayscale inputs are replicated to 3
            channels upstream in `Model.forward`. Should be ``3``.
        output_stride: Stride of the finest decoder output (Case A). Ignored for
            Case B (encoder-only).
        max_stride: Deepest stride the encoder reaches. For hierarchical CNN/Swin
            backbones this is ``32``; for a patch-``p`` ViT it is ``p``. Must match
            the real backbone (validated by the stride probe).
        weights: If ``True``, download and load the pretrained weights. If
            ``False``, build the architecture with random init from the model
            config (no network access) — useful for tests/CI and cold-start
            training.
        mode: One of ``"auto"``, ``"decoder"`` (Case A), ``"encoder"`` (Case B).
            ``"auto"`` picks encoder-only for isotropic ViTs and a decoder for
            hierarchical backbones.
        out_indices: Optional explicit stage indices to tap (Case A). If ``None``,
            all stages are requested and deduplicated by stride into a clean
            power-of-2 pyramid (recommended, family-agnostic).
        freeze: If ``True``, freeze the pretrained encoder (feature extraction);
            only the decoder/head train. Applied by the LightningModule after
            weight (re)loading.
        revision: Optional HuggingFace revision (commit sha / tag) to pin for
            reproducibility.
        normalize: If ``True``, apply model-specific mean/std normalization inside
            ``forward`` (the data pipeline only rescales to ``[0, 1]``).
        image_mean: Optional explicit per-channel mean (length 3). If ``None`` and
            ``normalize`` is set, read from the model's `AutoImageProcessor`,
            falling back to ImageNet stats.
        image_std: Optional explicit per-channel std (length 3).
        filters_rate: Decoder filter growth factor (Case A). Default ``2.0``.
        convs_per_block: Refinement convs per decoder block (Case A). Default ``2``.
        kernel_size: Decoder conv kernel size (Case A). Default ``3``.
        up_interpolate: If ``True``, bilinear upsampling in the decoder; else
            transposed convs (Case A). Default ``True``.
    """

    # Declared for type checkers; set via register_buffer in __init__.
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        source: str = "hf",
        model_name: str = "facebook/convnextv2-nano-22k-224",
        in_channels: int = 3,
        output_stride: int = 2,
        max_stride: int = 32,
        weights: bool = True,
        mode: str = "auto",
        out_indices: Optional[List[int]] = None,
        freeze: bool = False,
        revision: Optional[str] = None,
        normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        filters_rate: float = 2.0,
        convs_per_block: int = 2,
        kernel_size: int = 3,
        up_interpolate: bool = True,
    ) -> None:
        """Build the wrapped backbone and (Case A) decoder."""
        super().__init__()
        if source != "hf":
            message = (
                f"Unsupported pretrained backbone source: {source!r}. Only 'hf' "
                f"(HuggingFace transformers) is currently supported."
            )
            logger.error(message)
            raise ValueError(message)

        self.source = source
        self.model_name = model_name
        self.in_channels = in_channels
        self.output_stride = output_stride
        self.weights = weights
        self.freeze = freeze
        self.revision = revision
        self.normalize = normalize
        self.filters_rate = filters_rate
        self.convs_per_block = convs_per_block
        self.kernel_size = kernel_size
        self.up_interpolate = up_interpolate

        _import_transformers()  # actionable error if the extra is missing
        from transformers import AutoBackbone, AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, revision=revision)
        self.mode = _resolve_mode(hf_config, mode)

        # Build the HF backbone module (`self.enc`).
        if self.mode == "encoder":
            # Case B: single reshaped spatial bottleneck (tokens stripped, LN'd).
            backbone_kwargs = dict(
                out_indices=(-1,),
                reshape_hidden_states=True,
                apply_layernorm=True,
            )
        else:
            # Case A: request every stage; we dedupe by stride below.
            stage_names = getattr(hf_config, "stage_names", None)
            if out_indices is not None:
                idxs = tuple(out_indices)
            elif stage_names is not None:
                idxs = tuple(range(len(stage_names)))
            else:  # pragma: no cover - decoder mode implies stage_names
                idxs = (0, 1, 2, 3)
            backbone_kwargs = dict(out_indices=idxs)

        if weights:
            self.enc = AutoBackbone.from_pretrained(
                model_name,
                revision=revision,
                dtype=torch.float32,
                **backbone_kwargs,
            )
        else:
            cfg = AutoConfig.from_pretrained(model_name, revision=revision)
            for k, v in backbone_kwargs.items():
                setattr(cfg, k, v)
            self.enc = AutoBackbone.from_config(cfg)

        # Snapshot loaded weights on CPU so they can be re-applied after the
        # LightningModule's xavier init clobbers them (mirrors convnext/swint).
        self._pretrained_sd: Optional[Dict[str, torch.Tensor]] = None
        if weights:
            self._pretrained_sd = {
                k: v.detach().cpu().clone() for k, v in self.enc.state_dict().items()
            }

        # Register normalization buffers.
        mean, std = self._resolve_norm_stats(image_mean, image_std)
        self.register_buffer("image_mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, -1, 1, 1))

        # Probe strides/channels with a dummy forward (HF exposes no stride meta).
        strides, channels = self._probe(in_channels)

        if self.mode == "encoder":
            self._build_encoder_only(strides, channels)
        else:
            self._build_decoder(strides, channels, max_stride)

        # `from_pretrained` returns the model in eval mode; leave the encoder
        # trainable by default so fine-tuning (freeze=False) updates BatchNorm
        # running stats etc. Lightning respects module-level eval flags and will
        # NOT flip this back, so a stale eval here would silently freeze norm
        # layers during training. `freeze_encoder()` re-applies eval when frozen.
        self.enc.train()

    # ------------------------------------------------------------------ setup

    def _resolve_norm_stats(
        self, image_mean: Optional[List[float]], image_std: Optional[List[float]]
    ) -> Tuple[List[float], List[float]]:
        """Resolve per-channel mean/std for normalization.

        Priority: explicit args -> the model's `AutoImageProcessor` -> ImageNet.
        """
        if image_mean is not None and image_std is not None:
            return list(image_mean), list(image_std)
        if not self.normalize:
            # Identity normalization (mean 0, std 1) when disabled.
            return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        try:
            from transformers import AutoImageProcessor

            proc = AutoImageProcessor.from_pretrained(
                self.model_name, revision=self.revision
            )
            mean = getattr(proc, "image_mean", None)
            std = getattr(proc, "image_std", None)
            if mean is not None and std is not None:
                return list(mean), list(std)
        except Exception as e:  # pragma: no cover - network/availability dependent
            logger.warning(
                f"Could not read image_mean/image_std from an AutoImageProcessor for "
                f"'{self.model_name}' ({e}); falling back to ImageNet statistics."
            )
        return list(_IMAGENET_MEAN), list(_IMAGENET_STD)

    def _probe(self, in_channels: int) -> Tuple[List[int], List[int]]:
        """Dummy-forward to discover per-map strides and channel counts.

        Returns the multi-scale pyramid (Case A) or the single bottleneck (Case B)
        as ``(strides, channels)`` sorted by ascending stride, deduplicated so each
        stride appears once (keeping the deepest-processed map at that stride).
        """
        size = 256
        was_training = self.enc.training
        self.enc.eval()
        x = torch.zeros(1, in_channels, size, size)
        with torch.no_grad():
            feats = self.enc(x).feature_maps
        if was_training:
            self.enc.train()

        by_stride: Dict[int, int] = {}
        for f in feats:
            stride = size // f.shape[-2]
            by_stride[stride] = f.shape[1]
        strides = sorted(by_stride)
        channels = [by_stride[s] for s in strides]
        return strides, channels

    def _select_pyramid(self, feats) -> List[torch.Tensor]:
        """Dedupe raw HF ``feature_maps`` into a stride-ordered pyramid at runtime.

        Groups maps by spatial height (keeping the last/deepest map per size) and
        orders them finest -> coarsest (larger H == smaller stride first), matching
        the pyramid discovered by the init-time stride probe.
        """
        by_size: Dict[int, torch.Tensor] = {}
        for f in feats:
            by_size[f.shape[-2]] = f
        ordered_h = sorted(by_size, reverse=True)
        return [by_size[h] for h in ordered_h]

    def _build_decoder(
        self, strides: List[int], channels: List[int], max_stride: int
    ) -> None:
        """Case A: wire the multi-scale pyramid into the sleap decoder."""
        if len(strides) < 2:
            message = (
                f"Backbone '{self.model_name}' produced a single-scale feature map "
                f"(strides={strides}); it is isotropic and cannot feed a spatial "
                f"decoder. Use a hierarchical backbone (ConvNeXtV2/ResNet/Swinv2) "
                f"for pose/segmentation heads, or set mode='encoder' for a pooled "
                f"head."
            )
            logger.error(message)
            raise ValueError(message)

        self._pyramid_strides = strides
        self._pyramid_channels = channels
        deepest_stride = strides[-1]
        x_in_shape = channels[-1]

        if deepest_stride != max_stride:
            logger.warning(
                f"Backbone '{self.model_name}' has a deepest stride of "
                f"{deepest_stride}, but config max_stride={max_stride}. Using the "
                f"probed value {deepest_stride}."
            )
        self.max_stride = deepest_stride

        n_skips = len(strides) - 1  # every map except the bottleneck is a skip
        up_blocks = int(np.log2(deepest_stride / self.output_stride))
        if up_blocks < 1:
            message = (
                f"output_stride={self.output_stride} is >= the backbone's deepest "
                f"stride {deepest_stride}; nothing to decode. Lower output_stride."
            )
            logger.error(message)
            raise ValueError(message)

        # stem_blocks/down_blocks are chosen so decoder blocks past the available
        # skips use the no-concat (feat_concat=False) path. The finest HF stride is
        # typically 4, so reaching output_stride=2 adds one learned-upsample block
        # with no skip (see encoder_decoder.Decoder for the branch condition).
        stem_blocks = 1
        # Keep the feat_concat=False threshold (down_blocks + stem_blocks) equal to
        # n_skips so decoder blocks past the available skips take the no-concat
        # path. Must NOT clamp to 1: with only 2 feature maps (n_skips == 1, e.g. a
        # 2-entry out_indices) a clamp to 1 pushes the threshold to 2, so block 1 is
        # built expecting a skip but gets None at forward -> channel mismatch.
        down_blocks = max(0, n_skips - stem_blocks)
        encoder_channels = channels[:-1][::-1]  # skip channels, deepest-first

        self.dec = Decoder(
            x_in_shape=x_in_shape,
            output_stride=self.output_stride,
            current_stride=deepest_stride,
            filters=channels[0],
            up_blocks=up_blocks,
            down_blocks=down_blocks,
            stem_blocks=stem_blocks,
            filters_rate=self.filters_rate,
            convs_per_block=self.convs_per_block,
            kernel_size=self.kernel_size,
            up_interpolate=self.up_interpolate,
            encoder_channels=encoder_channels,
            prefix="pretrained_dec",
        )
        self.decoder_stride_to_filters = self.dec.stride_to_filters
        # Class-vectors head reads middle_blocks[-1].filters (bottleneck channels).
        self.middle_blocks = nn.ModuleList([_FiltersHolder(x_in_shape)])

    def _build_encoder_only(self, strides: List[int], channels: List[int]) -> None:
        """Case B: expose a pooled bottleneck for a class-vectors/embedding head."""
        self.max_stride = strides[-1]
        hidden = channels[-1]
        # No decoder: outputs=[] routes every head to `middle_output`.
        self.decoder_stride_to_filters = {}
        self.middle_blocks = nn.ModuleList([_FiltersHolder(hidden)])

    # ---------------------------------------------------------------- weights

    def reload_pretrained_weights(self) -> None:
        """Re-apply the snapshotted pretrained weights to the encoder.

        The LightningModule applies xavier init to the whole model (clobbering the
        encoder), then calls this to restore the pretrained weights — mirroring the
        convnext/swint ImageNet-load ordering.
        """
        if self._pretrained_sd is None:
            return
        self.enc.load_state_dict(self._pretrained_sd)
        logger.info(f"Restored pretrained weights for backbone '{self.model_name}'.")

    def freeze_encoder(self) -> None:
        """Freeze the pretrained encoder (feature extraction; decoder/head train)."""
        self.enc.eval()
        self.enc.requires_grad_(False)
        logger.info(f"Froze pretrained encoder '{self.model_name}'.")

    # ------------------------------------------------------------------ config

    @classmethod
    def from_config(cls, config: DictConfig) -> "PretrainedBackbone":
        """Create a `PretrainedBackbone` from a config leaf."""

        def get(key, default):
            return OmegaConf.select(config, key, default=default)

        return cls(
            source=get("source", "hf"),
            model_name=OmegaConf.select(config, "model_name"),
            in_channels=get("in_channels", 3),
            output_stride=get("output_stride", 2),
            max_stride=get("max_stride", 32),
            weights=get("weights", True),
            mode=get("mode", "auto"),
            out_indices=get("out_indices", None),
            freeze=get("freeze", False),
            revision=get("revision", None),
            normalize=get("normalize", True),
            image_mean=get("image_mean", None),
            image_std=get("image_std", None),
            filters_rate=get("filters_rate", 2.0),
            convs_per_block=get("convs_per_block", 2),
            kernel_size=get("kernel_size", 3),
            up_interpolate=get("up_interpolate", True),
        )

    # ------------------------------------------------------------------ forward

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific per-channel normalization to a [0, 1] tensor."""
        if not self.normalize:
            return x
        return (x - self.image_mean) / self.image_std

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass emitting the sleap-nn backbone dict contract.

        Args:
            x: Input tensor ``(B, in_channels, H, W)`` in ``[0, 1]`` (grayscale is
                already replicated to 3 channels upstream in `Model.forward`).

        Returns:
            Case A: ``{"outputs": [...], "strides": [...], "middle_output": bottleneck,
            "intermediate_feat": bottleneck}``.
            Case B: ``{"outputs": [], "strides": [], "middle_output": bottleneck,
            "intermediate_feat": bottleneck}``.
        """
        x = self._normalize(x)
        feats = self.enc(x).feature_maps

        if self.mode == "encoder":
            bottleneck = feats[-1]
            return {
                "outputs": [],
                "strides": [],
                "middle_output": bottleneck,
                "intermediate_feat": bottleneck,
            }

        # Case A: rebuild the stride-ordered pyramid, feed skips into the decoder.
        maps = self._select_pyramid(feats)  # finest -> coarsest (stride asc)
        bottleneck = maps[-1]
        skips = maps[:-1][::-1]  # deepest-first, matching encoder_channels
        out = self.dec(bottleneck, skips)
        out["middle_output"] = bottleneck
        return out
