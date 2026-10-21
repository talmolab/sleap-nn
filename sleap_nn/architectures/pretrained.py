"""Pretrained image backbones from HuggingFace `transformers` as sleap-nn encoders.

This module wires external pretrained vision encoders (via
`transformers.AutoBackbone` / the model-specific ``*Backbone`` classes) into the
existing :class:`sleap_nn.architectures.model.Model` contract, so any head
(pose, centroid, segmentation, class-vectors) can sit on a frozen or fine-tuned
foundation/ImageNet backbone. It reuses the same backbone dict-contract as the
native UNet/ConvNeXt/SwinT wrappers — no new `Model` class and no head changes.

Three integration surfaces are supported (the wrapper auto-selects, or you can
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

* **Case C — plain ViT + feature-pyramid neck + sleap decoder**
  (``mode="decoder"`` on an isotropic ViT; also what ``"auto"`` resolves to when
  the model has a spatial head). Any HuggingFace vision transformer -- DINOv2/v3,
  CLIP, SigLIP/SigLIP2, ViT, MAE, BEiT, I-JEPA, AIMv2, ... -- is loaded through
  ``AutoModel`` (its vision tower only, for two-tower checkpoints like CLIP) and
  run with ``output_hidden_states=True``. Selected transformer layers are cut out
  of the token sequence (prefix tokens -- CLS, registers -- stripped by count),
  reshaped to a grid, and turned into a stride ``s/4 .. 2s`` pyramid by
  :class:`~sleap_nn.architectures.vit_neck.ViTPyramidNeck`, which then feeds the
  same decoder as Case A. The input is resampled so the token grid lands on a
  power-of-two stride ``s`` (a patch-14 DINOv2 runs at an effective stride of
  16), so the rest of sleap-nn's stride bookkeeping is unchanged. Models that
  ship no ``*Backbone`` class (CLIP, SigLIP, ...) also use this extractor for
  Case B.

Key behaviors baked in here (rather than in the data pipeline, which is uint8 /
[0, 1] only): model-specific mean/std normalization (read from the matching
`AutoImageProcessor`, or passed explicitly), grayscale->3ch is handled upstream
in `Model.forward`, a `float32` dtype force (transformers v5 defaults to
``dtype="auto"`` which can silently load fp16/bf16), an optional pinned
``revision``, and a one-shot **stride probe** (HF backbones do not expose
stride metadata).
"""

import inspect
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from sleap_nn.architectures.common import FreezableEncoderMixin
from sleap_nn.architectures.encoder_decoder import Decoder
from sleap_nn.architectures.vit_neck import (
    PYRAMID_SCALES,
    DetailStem,
    ViTPyramidNeck,
)

# ImageNet stats, used as a last-resort normalization default when a model does
# not ship an `AutoImageProcessor` (most HF vision models do).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Config `model_type` strings that are isotropic (single-scale) ViTs. They have
# no stage pyramid for Case A: a spatial head routes them to Case C (pyramid
# neck), a pooled-only model to Case B.
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


class _LeadingBlocks(nn.ModuleList):
    """A transformer block list that runs only its first ``num_active`` blocks.

    Blocks after a plain ViT's last tapped layer feed nothing, but HF encoders loop
    over their whole block list. Iteration stops early here while every block
    stays registered, so parameters, the state dict and checkpoints are unchanged.
    """

    def __init__(self, blocks=None, num_active: Optional[int] = None) -> None:
        """Wrap ``blocks``; ``num_active=None`` runs them all."""
        super().__init__(blocks)
        self.num_active = num_active

    def __iter__(self):
        """Iterate over the active (leading) blocks only."""
        blocks = list(self._modules.values())
        return iter(blocks if self.num_active is None else blocks[: self.num_active])


def _is_isotropic(config: Any) -> bool:
    """Whether an HF config describes a single-stride (plain ViT) encoder."""
    model_type = getattr(config, "model_type", "") or ""
    stage_names = getattr(config, "stage_names", None)
    return model_type.lower() in _ISOTROPIC_MODEL_TYPES or not stage_names


def _resolve_mode(
    config: Any,
    mode: str,
    spatial_heads: bool = False,
    isotropic: Optional[bool] = None,
) -> str:
    """Resolve ``mode="auto"`` to ``"decoder"`` or ``"encoder"``.

    Args:
        config: A loaded HuggingFace config object.
        mode: ``"auto"``, ``"decoder"`` or ``"encoder"``.
        spatial_heads: Whether the model this backbone serves has a spatial head
            (confidence maps, PAFs, segmentation, ...). Under ``"auto"`` an
            isotropic ViT then resolves to ``"decoder"`` (Case C, the
            feature-pyramid neck); without one it stays encoder-only (Case B).
        isotropic: Whether the encoder is single-stride, when known from the built
            model rather than the config (timm checkpoints). ``None`` reads it from
            ``config``.

    Returns:
        ``"decoder"`` or ``"encoder"``.
    """
    if mode in ("decoder", "encoder"):
        return mode
    if isotropic is None:
        isotropic = _is_isotropic(config)
    if not isotropic:
        return "decoder"
    return "decoder" if spatial_heads else "encoder"


def _vision_config(config: Any) -> Any:
    """Return the vision-tower config of a (possibly multi-tower) HF config.

    CLIP, SigLIP, AIMv2, SAM, ... checkpoints bundle a text or prompt tower next
    to the image encoder; only ``config.vision_config`` is built so the unused
    towers never become (gradient-less) parameters of the backbone.
    """
    vision = getattr(config, "vision_config", None)
    return vision if vision is not None else config


def _patch_size(config: Any) -> int:
    """Read the (square) patch size of a ViT config."""
    patch = getattr(config, "patch_size", None)
    if isinstance(patch, (list, tuple)):
        if len(set(patch)) != 1:
            raise ValueError(f"Non-square patch size {patch} is not supported.")
        patch = patch[0]
    if not patch:
        raise ValueError(
            f"Could not read a patch size from the {type(config).__name__}; it does "
            "not look like a ViT."
        )
    return int(patch)


def _default_token_stride(patch_size: int) -> int:
    """Nearest power of two to the patch size (14 -> 16, 16 -> 16, 8 -> 8)."""
    return int(2 ** round(math.log2(patch_size)))


def _default_layers(num_layers: int, num_taps: int) -> List[int]:
    """Evenly spaced 1-indexed transformer-layer taps ending at the last layer.

    An encoder with fewer layers than taps repeats some layers (never layer 0).
    """
    return [max(1, round(num_layers * (i + 1) / num_taps)) for i in range(num_taps)]


def _resolve_backbone_class(config: Any):
    """Return the concrete ``*Backbone`` class for an HF config.

    ``AutoBackbone.from_pretrained`` performs a Hub existence check before it
    resolves anything, so it cannot load a cached model under
    ``HF_HUB_OFFLINE=1``. ``AutoConfig`` has no such check, so resolving the class
    from an already-loaded config and calling ``from_pretrained`` on it keeps the
    documented air-gapped path working.

    Args:
        config: A loaded HuggingFace config object.

    Returns:
        The mapped ``*Backbone`` class, or :class:`transformers.AutoBackbone` when
        the config type is not in the backbone mapping (a custom or very new
        model) — in which case the caller gets the original behavior, including
        its offline limitation.
    """
    from transformers import AutoBackbone

    backbone_cls = _backbone_class_or_none(config)
    if backbone_cls is None:
        logger.debug(
            f"No backbone mapping for {type(config).__name__}; falling back to "
            "AutoBackbone (which cannot load offline)."
        )
        return AutoBackbone
    return backbone_cls


def _backbone_class_or_none(config: Any):
    """Return the ``*Backbone`` class mapped to an HF config, or ``None``."""
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_BACKBONE_MAPPING,
        )

        return MODEL_FOR_BACKBONE_MAPPING[type(config)]
    except Exception:  # noqa: BLE001 - unmapped (CLIP, SigLIP, ...) or unknown
        return None


def _vision_model_class(vision_config: Any):
    """Return the base model class for a vision-tower config (offline-capable).

    Resolving through ``MODEL_MAPPING`` (rather than
    ``AutoModel.from_pretrained``) works from a config already in hand -- e.g.
    ``CLIPVisionConfig`` -> ``CLIPVisionModel`` -- which loads only the vision
    weights out of a two-tower checkpoint.
    """
    from transformers import AutoModel
    from transformers.models.auto.modeling_auto import MODEL_MAPPING

    try:
        return MODEL_MAPPING[type(vision_config)]
    except Exception:  # noqa: BLE001 - fall back to AutoModel's own resolution
        return AutoModel


def _build_timm_wrapper(
    model_name: str, hf_config: Any, weights: bool, revision: Optional[str]
) -> nn.Module:
    """Build a ``timm/...`` Hub checkpoint through transformers' TimmWrapperModel.

    ViT-family timm models are created with ``dynamic_img_size=True`` so they run
    at any input size (positional embeddings are resampled per call); families
    that do not take the flag (ConvNeXt, ...) are retried without it.
    """
    from transformers import TimmWrapperModel

    try:
        import timm  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        message = (
            f"'{model_name}' is a timm checkpoint, which needs the `timm` package. "
            "Install it with `pip install 'sleap-nn[backbones]'` (or "
            "`uv sync --extra backbones`)."
        )
        logger.error(message)
        raise ImportError(message) from e

    base_args = dict(getattr(hf_config, "model_args", None) or {})

    def build(model_args):
        hf_config.model_args = model_args
        if weights:
            return TimmWrapperModel.from_pretrained(
                model_name, config=hf_config, revision=revision, dtype=torch.float32
            )
        return TimmWrapperModel(hf_config)

    try:
        return build({**base_args, "dynamic_img_size": True})
    except TypeError:
        return build(base_args)


def _timm_is_isotropic(timm_model: nn.Module) -> bool:
    """Whether a timm model is a plain (single-stride) ViT."""
    info = getattr(timm_model, "feature_info", None) or []
    reductions = {int(f["reduction"]) for f in info}
    return len(reductions) <= 1 and hasattr(timm_model, "blocks")


def _timm_patch_size(timm_model: nn.Module) -> int:
    """Patch size of a timm ViT."""
    patch = getattr(getattr(timm_model, "patch_embed", None), "patch_size", None)
    if patch is None:
        patch = timm_model.feature_info[-1]["reduction"]
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    return int(patch)


class PretrainedBackbone(FreezableEncoderMixin, nn.Module):
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
            backbones this is ``32``; for a plain ViT (Case C) it is twice the
            token stride (``32`` by default) and must be a multiple of it.
            Validated against the real backbone at construction.
        weights: If ``True``, download and load the pretrained weights. If
            ``False``, build the architecture with random init from the model
            config (no network access) — useful for tests/CI and cold-start
            training.
        mode: One of ``"auto"``, ``"decoder"`` (Case A, or Case C for a plain
            ViT), ``"encoder"`` (Case B). ``"auto"`` picks a decoder for
            hierarchical backbones; for a plain ViT it picks Case C when the model
            has a spatial head (``spatial_heads``) and encoder-only otherwise.
        out_indices: Optional explicit stage indices to tap (Case A). If ``None``,
            all stages are requested and deduplicated by stride into a clean
            power-of-2 pyramid (recommended, family-agnostic). The indices follow
            the source: an HF backbone's ``stage_names``, where ``0`` is the stem
            (a ConvNeXt's four stages are ``[1, 2, 3, 4]``), or a ``timm/...``
            checkpoint's ``feature_info``, where ``0`` is the first stage
            (``[0, 1, 2, 3]``).
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
        layers: Transformer layers to tap for a plain ViT (Case C), 1-indexed
            (``hidden_states[i]`` is the output of block ``i``; negative indices
            count from the last block). ``None`` taps 4 evenly spaced layers ending
            at the last one for ``vit_neck="dpt"``, and the last layer for
            ``"sfp"`` / encoder mode.
        vit_neck: How a plain ViT's single-stride maps become a pyramid (Case C):
            ``"dpt"`` (one tapped layer per level, the default) or ``"sfp"``
            (ViTDet Simple Feature Pyramid from the last tapped layer).
        neck_channels: Channels of the 4 pyramid levels, finest first (Case C).
            ``None`` uses ``[h/8, h/4, h/2, h]`` for a hidden size ``h``.
        token_stride: Effective stride of a plain ViT's token grid (Case C). The
            input is resampled so the grid is ``H / token_stride`` tokens tall
            whatever the patch size; must be a power of two. ``None`` uses the
            power of two nearest the patch size (16 for patch-14 DINOv2). A smaller
            value (e.g. 8) gives a finer grid at ~4x the encoder cost.
        gradient_checkpointing: If ``True``, enable gradient checkpointing on the
            encoder (less activation memory, slower steps); active in training
            only.
        encoder_lr_scale: Learning-rate multiplier for the pretrained encoder's
            parameters relative to the base LR (neck/decoder/heads). Default
            ``1.0``.
        layer_decay: Optional layer-wise LR decay for a plain ViT encoder: block
            ``i`` of ``L`` trains at ``encoder_lr_scale * layer_decay**(L + 1 - i)``
            times the base LR (embeddings decay the most). ``None`` disables it.
        detail_stem: If ``True`` (Case C), add a light conv stream on the image
            whose stride-2/4/8 features join the pyramid as decoder skips, so a
            plain ViT's decoder sees real high-resolution detail.
        detail_channels: Channels of the detail stem at strides 2, 4, 8 (exactly
            three).
        spatial_heads: Whether the model has a spatial head; decides what
            ``mode="auto"`` means for a plain ViT (see ``_resolve_mode``). Set by
            `Model`, not by the config.
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
        layers: Optional[List[int]] = None,
        vit_neck: str = "dpt",
        neck_channels: Optional[List[int]] = None,
        token_stride: Optional[int] = None,
        gradient_checkpointing: bool = False,
        encoder_lr_scale: float = 1.0,
        layer_decay: Optional[float] = None,
        detail_stem: bool = False,
        detail_channels: Optional[List[int]] = None,
        spatial_heads: bool = False,
    ) -> None:
        """Build the wrapped backbone and (Case A/C) decoder."""
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
        self.vit_neck = vit_neck
        self.encoder_lr_scale = float(encoder_lr_scale)
        self.layer_decay = layer_decay
        self.detail_stem = detail_stem
        self.detail_channels = list(detail_channels or [32, 64, 128])
        if len(self.detail_channels) != 3:
            # One stage per stride 2, 4, 8: an extra stage would run without ever
            # reaching the decoder (parameters with no gradient, a DDP error).
            message = (
                f"detail_channels must give 3 channel counts (strides 2, 4, 8), got "
                f"{self.detail_channels}."
            )
            logger.error(message)
            raise ValueError(message)

        _import_transformers()  # actionable error if the extra is missing
        from transformers import AutoBackbone, AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, revision=revision)

        # `timm/...` Hub checkpoints (DINOv3, PE, EVA02, SigLIP2, ConvNeXt, ...)
        # load through transformers' TimmWrapperModel; whether they are plain ViTs
        # is read off the built model, since the wrapper config does not say.
        self.is_timm = getattr(hf_config, "model_type", "") == "timm_wrapper"
        if self.is_timm:
            self.enc = _build_timm_wrapper(model_name, hf_config, weights, revision)
            isotropic = _timm_is_isotropic(self.enc.timm_model)
            # Skip the blocks/stages after the last requested one.
            params = inspect.signature(
                self.enc.timm_model.forward_intermediates
            ).parameters
            self._timm_kwargs = {"stop_early": True} if "stop_early" in params else {}
        else:
            isotropic = _is_isotropic(hf_config)
        self.isotropic = isotropic
        self.mode = _resolve_mode(
            hf_config, mode, spatial_heads=spatial_heads, isotropic=isotropic
        )

        # Case C: a plain ViT is read through its hidden states (generic for any
        # HF vision transformer). So is an encoder-only ViT that ships no
        # `*Backbone` class (CLIP, SigLIP, timm ViTs, ...); the ones that do
        # (DINOv2) keep the AutoBackbone path below, so existing checkpoints load
        # unchanged.
        self.token_path = isotropic and (
            self.mode == "decoder"
            or self.is_timm
            or _backbone_class_or_none(hf_config) is None
        )
        if self.token_path:
            ignored = {"out_indices": out_indices}
            if self.mode == "encoder":
                # Encoder-only pools one tapped layer: there is no neck or decoder.
                ignored.update(
                    neck_channels=neck_channels,
                    detail_stem=detail_stem or None,
                    vit_neck=None if vit_neck == "dpt" else vit_neck,
                )
            ignored = [k for k, v in ignored.items() if v is not None]
            if ignored:
                path = "encoder-only" if self.mode == "encoder" else "pyramid-neck"
                logger.warning(
                    f"{', '.join(ignored)} do not apply to the plain-ViT {path} path "
                    f"'{model_name}' takes in mode '{self.mode}'; ignored."
                )
            self._init_token_encoder(
                hf_config,
                layers=layers,
                token_stride=token_stride,
                gradient_checkpointing=gradient_checkpointing,
            )
            self._snapshot_weights()
            mean, std = self._resolve_norm_stats(image_mean, image_std)
            self.register_buffer("image_mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("image_std", torch.tensor(std).view(1, -1, 1, 1))
            if self.mode == "encoder":
                self._probe_tokens()
                self._build_encoder_only([self.token_stride], [self.hidden_size])
            else:
                strides, channels = self._build_vit_neck(neck_channels, max_stride)
                # Detail-stem outputs the decoder reads (none finer than the
                # output stride); its last (stride-8) stage needs all the others.
                self._stem_strides = [
                    st for st in (2, 4, 8) if st >= self.output_stride
                ]
                if self.detail_stem and not self._stem_strides:
                    logger.warning(
                        f"detail_stem ignored: output_stride={self.output_stride} is "
                        "coarser than its finest level (8)."
                    )
                    self.detail_stem = False
                if self.detail_stem:
                    self.stem = DetailStem(self.in_channels, self.detail_channels)
                    by_stride = dict(zip(strides, channels))
                    for st, c in zip((2, 4, 8), self.detail_channels):
                        if st in self._stem_strides:
                            by_stride[st] = by_stride.get(st, 0) + c
                    strides = sorted(by_stride)
                    channels = [by_stride[st] for st in strides]
                self._build_decoder(strides, channels, max_stride)
            self.enc.train()
            return

        ignored = {
            "layers": layers,
            "token_stride": token_stride,
            "neck_channels": neck_channels,
            "detail_stem": detail_stem or None,
            "vit_neck": None if vit_neck == "dpt" else vit_neck,
        }
        ignored = [k for k, v in ignored.items() if v is not None]
        if ignored:
            logger.warning(
                f"{', '.join(ignored)} only apply to plain-ViT backbones on the "
                f"hidden-state path (Case C, or encoder-only without a `*Backbone` "
                f"class); ignored for '{model_name}' in mode '{self.mode}'."
            )

        # Build the HF backbone module (`self.enc`); timm models are already built.
        if self.is_timm:
            info = self.enc.timm_model.feature_info
            self._timm_stage_indices = list(range(len(info)))
            if out_indices is not None:
                self._timm_stage_indices = [int(i) for i in out_indices]
        elif self.mode == "encoder":
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

        if not self.is_timm:
            # BOTH branches configure the backbone the same way: set the options as
            # ATTRIBUTES on the config, then build from that config. Passing them as
            # `from_pretrained(**kwargs)` instead — which the weights=True path used to
            # do — forwards anything the config does not claim to the model's
            # `__init__`, and Case B's `reshape_hidden_states`/`apply_layernorm` exist
            # only on the ViT-family classes:
            #
            #     ConvNextV2Backbone.__init__() got an unexpected keyword argument
            #     'reshape_hidden_states'
            #
            # That made `mode="encoder"` unusable for every CNN backbone with real
            # weights, while the weights=False path (already config-based, where an
            # unclaimed attribute is silently ignored) worked — so the two paths
            # disagreed, and every test in `tests/architectures/test_pretrained.py`
            # exercises only the one that works. Configuring both identically removes
            # the divergence rather than special-casing the symptom.
            for k, v in backbone_kwargs.items():
                setattr(hf_config, k, v)

            if weights:
                # Resolve the concrete `*Backbone` class from the config instead of
                # going through `AutoBackbone.from_pretrained`, which probes
                # `hf_api().repo_exists()` before resolving anything and therefore
                # raises `OfflineModeIsEnabled` under `HF_HUB_OFFLINE=1` even for a
                # fully-cached model — breaking the air-gapped workflow the guide
                # documents. `AutoConfig` is offline-capable, so the class it maps to
                # is reachable without the network.
                backbone_cls = _resolve_backbone_class(hf_config)
                self.enc = backbone_cls.from_pretrained(
                    model_name,
                    config=hf_config,
                    revision=revision,
                    dtype=torch.float32,
                )
            else:
                self.enc = AutoBackbone.from_config(hf_config)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self._snapshot_weights()

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

    def _snapshot_weights(self) -> None:
        """Snapshot loaded weights on CPU for `reload_pretrained_weights`.

        The LightningModule's xavier init clobbers every Conv2d/Linear after
        construction; the snapshot lets it re-apply the pretrained encoder
        (mirrors convnext/swint).
        """
        self._pretrained_sd: Optional[Dict[str, torch.Tensor]] = None
        if self.weights:
            self._pretrained_sd = {
                k: v.detach().cpu().clone() for k, v in self.enc.state_dict().items()
            }

    def _enable_gradient_checkpointing(self) -> None:
        """Turn on HF gradient checkpointing for the encoder, if it supports it."""
        if self.is_timm and hasattr(self.enc.timm_model, "set_grad_checkpointing"):
            # Unlike HF's, timm's switch is not gated on training; `train` keeps it
            # off in eval (inference, export).
            self._timm_grad_ckpt = True
            self.enc.timm_model.set_grad_checkpointing(self.enc.training)
        elif getattr(self.enc, "supports_gradient_checkpointing", False):
            self.enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            logger.warning(
                f"'{self.model_name}' does not support gradient checkpointing; "
                "ignoring gradient_checkpointing=True."
            )

    def _init_token_encoder(
        self,
        hf_config: Any,
        layers: Optional[List[int]],
        token_stride: Optional[int],
        gradient_checkpointing: bool,
    ) -> None:
        """Case C: build a plain ViT's vision tower and its layer taps.

        Args:
            hf_config: The checkpoint's HF config (multi-tower configs are reduced
                to their ``vision_config``).
            layers: 1-indexed transformer layers to tap, or ``None`` for the
                default taps.
            token_stride: Effective token-grid stride, or ``None`` for the power
                of two nearest the patch size.
            gradient_checkpointing: Whether to enable HF gradient checkpointing.
        """
        from transformers import AutoModel

        if self.is_timm:
            timm_model = self.enc.timm_model
            vision_config = None
            self._identity_noise = False
            self.patch_size = _timm_patch_size(timm_model)
            self.hidden_size = int(
                getattr(timm_model, "embed_dim", None) or timm_model.num_features
            )
            self.num_layers = len(timm_model.blocks)
        else:
            vision_config = _vision_config(hf_config)
            model_type = getattr(vision_config, "model_type", "") or ""
            if model_type == "vit_mae":
                # Pretraining-time random masking would drop 75% of the tokens.
                vision_config.mask_ratio = 0.0
            self._identity_noise = model_type == "vit_mae"
            self.patch_size = _patch_size(vision_config)
            self.hidden_size = int(vision_config.hidden_size)
            self.num_layers = int(vision_config.num_hidden_layers)

        self.token_stride = int(token_stride or _default_token_stride(self.patch_size))
        if self.token_stride < 4 or self.token_stride & (self.token_stride - 1):
            message = (
                f"token_stride must be a power of two >= 4 (got {self.token_stride}); "
                "the pyramid neck reaches token_stride / 4."
            )
            logger.error(message)
            raise ValueError(message)

        # Pyramid levels the decoder will read: none finer than the output stride.
        self._neck_scales = [
            sc for sc in PYRAMID_SCALES if self.token_stride / sc >= self.output_stride
        ]
        if layers is None:
            dpt = self.mode == "decoder" and self.vit_neck == "dpt"
            num_taps = max(len(self._neck_scales), 1) if dpt else 1
            layers = _default_layers(self.num_layers, num_taps)
        elif len(layers) != 1 and (self.mode == "encoder" or self.vit_neck == "sfp"):
            # One tap feeds every level (sfp) / the pooled head (encoder); more
            # would build tap norms that never run.
            what = (
                "encoder mode pools"
                if self.mode == "encoder"
                else "vit_neck='sfp' builds every pyramid level from"
            )
            message = (
                f"layers={list(layers)}: {what} exactly one tapped layer. Give one "
                "(e.g. [-1]) or leave `layers` unset."
            )
            logger.error(message)
            raise ValueError(message)
        resolved = []
        for layer in layers:
            layer = int(layer)
            idx = layer if layer > 0 else self.num_layers + 1 + layer
            if not 1 <= idx <= self.num_layers:
                message = (
                    f"Layer tap {layer} is out of range for '{self.model_name}', which "
                    f"has {self.num_layers} transformer layers (valid: 1..."
                    f"{self.num_layers} or -1...-{self.num_layers})."
                )
                logger.error(message)
                raise ValueError(message)
            resolved.append(idx)
        self.layers = sorted(resolved)

        if self.is_timm:
            pass  # built in __init__ (needed to tell ViTs from hierarchical models)
        elif self.weights:
            model_cls = _vision_model_class(vision_config)
            self.enc = model_cls.from_pretrained(
                self.model_name,
                config=vision_config,
                revision=self.revision,
                dtype=torch.float32,
            )
        else:
            self.enc = AutoModel.from_config(vision_config)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        self._skip_blocks_after_last_tap()

        # Forward kwargs: always hidden states; turn on positional-embedding
        # interpolation where the model gates it behind a flag (CLIP, SigLIP, ViT,
        # MAE). DINOv2/v3 interpolate (or use RoPE) unconditionally.
        params = inspect.signature(self.enc.forward).parameters
        self._call_kwargs: Dict[str, Any] = {"output_hidden_states": True}
        if "interpolate_pos_encoding" in params and not self.is_timm:
            self._call_kwargs["interpolate_pos_encoding"] = True

        # One fresh LayerNorm per tap: intermediate ViT layers are not normalized
        # (and carry large outlier channels), and families disagree on whether
        # the final norm is part of `hidden_states[-1]`. Trainable, part of the
        # neck -- NOT the frozen encoder.
        self.tap_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size, eps=1e-6) for _ in self.layers]
        )

    def _skip_blocks_after_last_tap(self) -> None:
        """Stop the encoder after the last tapped layer instead of running them all.

        timm takes ``stop_early`` in ``forward_intermediates``. An HF encoder loops
        over its block list, so that list (the one ``nn.ModuleList`` with one entry
        per layer) is swapped for a `_LeadingBlocks` over the same blocks. Left
        alone when the last layer is tapped or the list cannot be found.
        """
        last = max(self.layers)
        if self.is_timm or last == self.num_layers:
            return
        found = [
            (name, m)
            for name, m in self.enc.named_modules()
            if isinstance(m, nn.ModuleList) and len(m) == self.num_layers
        ]
        if len(found) != 1:
            logger.debug(
                f"No unique block list in '{self.model_name}'; blocks after layer "
                f"{last} still run (they feed nothing)."
            )
            return
        name, blocks = found[0]
        parent_name, _, attr = name.rpartition(".")
        parent = self.enc.get_submodule(parent_name) if parent_name else self.enc
        setattr(parent, attr, _LeadingBlocks(blocks, num_active=last))

    def _build_vit_neck(
        self, neck_channels: Optional[List[int]], max_stride: int
    ) -> Tuple[List[int], List[int]]:
        """Case C: build the pyramid neck and probe the token grid.

        Returns:
            ``(strides, channels)`` of the neck's pyramid, finest first.
        """
        scales = self._neck_scales
        if len(scales) < 2:
            # The decoder upsamples from the deepest level through at least one
            # finer one, so output_stride must not exceed the token stride.
            message = (
                f"output_stride={self.output_stride} is too coarse for "
                f"'{self.model_name}': its pyramid levels are strides "
                f"{self.token_stride // 4}..{2 * self.token_stride} (token stride "
                f"{self.token_stride}) and the decoder needs at least two of them. "
                f"Use output_stride <= {self.token_stride}."
            )
            logger.error(message)
            raise ValueError(message)
        if neck_channels is None:
            h = self.hidden_size
            neck_channels = [max(h // 8, 16), max(h // 4, 16), max(h // 2, 16), h]
        neck_channels = list(neck_channels)
        if len(neck_channels) == len(PYRAMID_SCALES):
            # Given for all four levels; keep the ones that are built.
            neck_channels = neck_channels[len(PYRAMID_SCALES) - len(scales) :]
        self.neck = ViTPyramidNeck(
            in_channels=self.hidden_size,
            out_channels=neck_channels,
            num_inputs=len(self.layers),
            mode=self.vit_neck,
            scales=scales,
        )
        strides = [int(self.token_stride / scale) for scale in scales]
        self._neck_strides = strides
        deepest = strides[-1]
        if max_stride % deepest != 0:
            message = (
                f"max_stride={max_stride} must be a multiple of {deepest} (twice the "
                f"token stride {self.token_stride}) so padded inputs tile the "
                f"'{self.model_name}' token grid and its pyramid exactly."
            )
            logger.error(message)
            raise ValueError(message)
        self._probe_tokens()
        return strides, neck_channels

    def _probe_tokens(self) -> None:
        """Dummy forward/backward through the token extraction at construction.

        Catches models that cannot run off their pretraining resolution (fixed
        positional embeddings) or whose prefix-token count does not add up, at
        construction instead of at the first training step.

        The backward pass also finds encoder parameters the tapped layers never
        reach -- blocks after the last tap, final/pooler norms, pooling heads
        (SigLIP's MAP head, CLIP's ``post_layernorm``) -- and stops them from
        requiring grad. They would never receive a gradient anyway, but as
        trainable parameters they make DDP fail ("parameters that were not used
        in producing loss") and cost optimizer state. They stay in the state dict,
        so pretrained and trained weights load unchanged.
        """
        s = 2 * self.token_stride
        was_training = self.enc.training
        self.enc.eval()
        x = torch.zeros(1, self.in_channels, 4 * s, 3 * s)
        try:
            with torch.enable_grad():
                feats = self._tap_tokens(x)
                sum(f.float().mean() for f in feats).backward()
        except Exception as e:  # noqa: BLE001 - re-raised with context
            message = (
                f"Could not extract a dense token grid from '{self.model_name}' at a "
                f"{4 * s}x{3 * s} input ({type(e).__name__}: {e}). The model may not "
                "support inputs off its pretraining resolution."
            )
            logger.error(message)
            raise ValueError(message) from e
        finally:
            if was_training:
                self.enc.train()
        dead = [
            name
            for name, p in self.enc.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        for p in self.parameters():
            p.grad = None
        if dead:
            dead_names = set(dead)
            for name, p in self.enc.named_parameters():
                if name in dead_names:
                    p.requires_grad_(False)
            logger.info(
                f"{len(dead)} parameter tensors of '{self.model_name}' do not feed "
                f"the tapped layers {self.layers} (e.g. {dead[0]}); excluded from "
                "training."
            )
        expected = (self.hidden_size, 8, 6)
        for f in feats:
            if tuple(f.shape[1:]) != expected:
                message = (
                    f"Token maps of '{self.model_name}' have shape {tuple(f.shape[1:])}"
                    f", expected {expected}."
                )
                logger.error(message)
                raise ValueError(message)

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
        if getattr(self, "is_timm", False):
            pretrained_cfg = getattr(self.enc.timm_model, "pretrained_cfg", None) or {}
            mean, std = pretrained_cfg.get("mean"), pretrained_cfg.get("std")
            if mean is not None and std is not None:
                return list(mean), list(std)
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
            feats = self._stage_maps(x)
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

    def train(self, mode: bool = True):
        """Set train/eval mode; timm gradient checkpointing runs only in training.

        HF gates its gradient checkpointing on ``self.training``; timm checks only
        its flag, so it is switched to follow the encoder's mode (eval for
        inference/export, and for a frozen encoder that has nothing to backprop).
        """
        super().train(mode)
        if getattr(self, "_timm_grad_ckpt", False):
            self.enc.timm_model.set_grad_checkpointing(self.enc.training)
        return self

    def freeze_encoder(self) -> None:
        """Freeze the pretrained encoder (feature extraction; decoder/head train).

        The encoder stays in eval mode for the whole run (``FreezableEncoderMixin``
        re-applies it on every ``train()``), so BatchNorm keeps its pretrained
        running statistics.
        """
        super().freeze_encoder()
        logger.info(f"Froze pretrained encoder '{self.model_name}'.")

    def encoder_param_groups(self, base_lr: float) -> Optional[List[Dict[str, Any]]]:
        """Optimizer parameter groups for the encoder's trainable parameters.

        Args:
            base_lr: The base learning rate (used as-is by every non-encoder
                parameter).

        Returns:
            ``None`` when the encoder trains at the base LR (``encoder_lr_scale ==
            1`` and no ``layer_decay``) so the optimizer keeps a single group;
            otherwise a list of ``{"params", "lr"}`` groups covering exactly the
            encoder parameters that require grad.
        """
        if self.encoder_lr_scale == 1.0 and not self.layer_decay:
            return None
        named = [(n, p) for n, p in self.enc.named_parameters() if p.requires_grad]
        if not named:
            return None
        scale = self.encoder_lr_scale
        if not self.layer_decay:
            return [{"params": [p for _, p in named], "lr": base_lr * scale}]

        num_layers = getattr(self, "num_layers", None) or getattr(
            getattr(self.enc, "config", None), "num_hidden_layers", None
        )
        if not self.isotropic or not num_layers:
            logger.warning(
                f"layer_decay applies to plain-ViT encoders; '{self.model_name}' is "
                "hierarchical, so only encoder_lr_scale is used."
            )
            return [{"params": [p for _, p in named], "lr": base_lr * scale}]

        # Block i (0-indexed in its parameter name) -> depth i + 1. Parameters
        # registered before the first block (patch/position embeddings, CLS and
        # register tokens, CLIP's pre-norm) -> depth 0; after the last block
        # (final norms, pooling heads) -> depth L + 1. Registration order is
        # family-agnostic where name heuristics are not.
        block_re = re.compile(r"(?:^|\.)(?:layer|layers|blocks|block)\.(\d+)\.")
        groups: Dict[int, List[torch.Tensor]] = {}
        seen_block = False
        for name, param in named:
            match = block_re.search(name)
            if match:
                seen_block = True
                depth = min(int(match.group(1)) + 1, num_layers)
            else:
                depth = num_layers + 1 if seen_block else 0
            groups.setdefault(depth, []).append(param)
        return [
            {
                "params": params,
                "lr": base_lr * scale * self.layer_decay ** (num_layers + 1 - depth),
            }
            for depth, params in sorted(groups.items())
        ]

    # ------------------------------------------------------------------ config

    @classmethod
    def from_config(
        cls, config: DictConfig, spatial_heads: bool = False
    ) -> "PretrainedBackbone":
        """Create a `PretrainedBackbone` from a config leaf.

        Args:
            config: The ``backbone_config.pretrained`` leaf.
            spatial_heads: Whether the model has a spatial head (see
                ``_resolve_mode``).
        """

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
            layers=get("layers", None),
            vit_neck=get("vit_neck", "dpt"),
            neck_channels=get("neck_channels", None),
            token_stride=get("token_stride", None),
            gradient_checkpointing=get("gradient_checkpointing", False),
            encoder_lr_scale=get("encoder_lr_scale", 1.0),
            layer_decay=get("layer_decay", None),
            detail_stem=get("detail_stem", False),
            detail_channels=get("detail_channels", None),
            spatial_heads=spatial_heads,
        )

    # ------------------------------------------------------------------ forward

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific per-channel normalization to a [0, 1] tensor."""
        if not self.normalize:
            return x
        return (x - self.image_mean) / self.image_std

    def _merge_detail(
        self, maps: List[torch.Tensor], detail: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Concatenate detail-stem features (strides 2/4/8) into the pyramid."""
        by_stride: Dict[int, List[torch.Tensor]] = {
            st: [m] for st, m in zip(self._neck_strides, maps)
        }
        for st, d in zip((2, 4, 8), detail):
            if st in self._stem_strides:
                by_stride.setdefault(st, []).append(d)
        return [
            (
                torch.cat(by_stride[st], dim=1)
                if len(by_stride[st]) > 1
                else by_stride[st][0]
            )
            for st in sorted(by_stride)
        ]

    def _stage_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Stage outputs of a hierarchical encoder (Case A/B), finest first."""
        if self.is_timm:
            return self.enc.timm_model.forward_intermediates(
                x,
                indices=self._timm_stage_indices,
                output_fmt="NCHW",
                intermediates_only=True,
                **self._timm_kwargs,
            )
        return self.enc(x).feature_maps

    def _tap_tokens(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run a plain ViT and return the tapped layers as NCHW token maps.

        The input is resampled so its token grid is ``H / token_stride`` by
        ``W / token_stride`` (a no-op when the patch size equals the token
        stride), so each map sits exactly at stride ``token_stride``.
        """
        b, _, h, w = x.shape
        gh, gw = h // self.token_stride, w // self.token_stride
        size = (gh * self.patch_size, gw * self.patch_size)
        if (h, w) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if self.is_timm:
            # timm strips prefix tokens and returns NCHW grids itself, one per
            # distinct block (a repeated tap is looked up again below).
            unique = sorted(set(self.layers))
            maps = self.enc.timm_model.forward_intermediates(
                x,
                indices=[layer - 1 for layer in unique],
                output_fmt="NCHW",
                intermediates_only=True,
                **self._timm_kwargs,
            )
            by_layer = dict(zip(unique, maps))
            return [
                norm(by_layer[layer].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                for norm, layer in zip(self.tap_norms, self.layers)
            ]
        kwargs = dict(self._call_kwargs)
        if self._identity_noise:
            # MAE shuffles tokens by argsort(noise); monotone noise keeps the order.
            noise = torch.arange(gh * gw, device=x.device, dtype=x.dtype)
            kwargs["noise"] = noise.unsqueeze(0).expand(b, -1)
        hidden_states = self.enc(pixel_values=x, **kwargs).hidden_states

        maps = []
        for norm, layer in zip(self.tap_norms, self.layers):
            tokens = hidden_states[layer]
            if tokens.dim() == 4:  # (B, h, w, C) grids, e.g. SAM's ViT
                tokens = tokens.flatten(1, 2)
            # Prefix tokens (CLS, registers, ...) lead the sequence.
            tokens = norm(tokens[:, tokens.shape[1] - gh * gw :, :])
            maps.append(tokens.transpose(1, 2).reshape(b, -1, gh, gw))
        return maps

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass emitting the sleap-nn backbone dict contract.

        Args:
            x: Input tensor ``(B, in_channels, H, W)`` in ``[0, 1]`` (grayscale is
                already replicated to 3 channels upstream in `Model.forward`).

        Returns:
            Case A/C: ``{"outputs": [...], "strides": [...], "middle_output":
            bottleneck, "intermediate_feat": bottleneck}`` (Case C's bottleneck is
            the stride ``2 * token_stride`` neck level).
            Case B: ``{"outputs": [], "strides": [], "middle_output": bottleneck,
            "intermediate_feat": bottleneck}``.
        """
        x = self._normalize(x)
        if self.token_path:
            feats = self._tap_tokens(x)
        else:
            feats = self._stage_maps(x)

        if self.mode == "encoder":
            bottleneck = feats[-1]
            return {
                "outputs": [],
                "strides": [],
                "middle_output": bottleneck,
                "intermediate_feat": bottleneck,
            }

        if self.token_path:
            # Case C: the neck turns the tapped token maps into the pyramid.
            maps = self.neck(feats)  # finest -> coarsest (stride asc)
            if self.detail_stem:
                maps = self._merge_detail(maps, self.stem(x))
        else:
            # Case A: rebuild the stride-ordered pyramid from the stage outputs.
            maps = self._select_pyramid(feats)  # finest -> coarsest (stride asc)
        # Feed skips into the decoder.
        bottleneck = maps[-1]
        skips = maps[:-1][::-1]  # deepest-first, matching encoder_channels
        out = self.dec(bottleneck, skips)
        out["middle_output"] = bottleneck
        return out
