"""RF-DETR's windowed DINOv2 backbone (``dinov2_windowed``).

Port of HF transformers' ``RfDetrDinov2Backbone`` (``models/rf_detr``,
transformers 5.20.2). A DINOv2 ViT with three changes:

1. **Patch 16.** RF-DETR resamples DINOv2's 14x14 patch embedding to 16x16
   (FlexiViT), so tokens sit on a stride-16 grid. The learned position embedding
   covers a ``pos_embed_grid`` x ``pos_embed_grid`` grid and is interpolated
   (bicubic, antialiased) to other input grids.
2. **Windowed attention.** The token grid is cut into ``num_windows`` x
   ``num_windows`` windows that run as separate sequences, each with its own copy
   of the class token. The blocks in ``window_block_indexes`` attend within their
   window; the others merge the windows back into one sequence (class-token copies
   included) for global attention. Windows only reshape tokens, so DINOv2 weights
   load unchanged.
3. **Taps.** The outputs of the blocks in ``out_indices`` (hidden-state indices:
   0 is the embedding output, ``i`` the output of block ``i``) go through the final
   LayerNorm, lose their class tokens and come back as ``(B, D, H/16, W/16)`` maps.

Module and parameter names follow HF's ``RfDetrDinov2Backbone``, so RF-DETR
checkpoints load by renaming prefixes (:mod:`sleap_nn.convert.rfdetr`). Plain
``facebook/dinov2-*`` checkpoints load too (:func:`load_dinov2_weights`): their
patch embedding is resampled to ``patch_size`` and their position embedding
resized to ``pos_embed_grid``.

Inputs are images in ``[0, 1]``; the backbone applies the ImageNet normalization
DINOv2 and RF-DETR expect.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from sleap_nn.architectures.detr.backbones import register_detr_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_window_block_indexes(
    out_indices: Sequence[int], num_layers: int
) -> List[int]:
    """Blocks that attend within windows, as HF's ``RfDetrDinov2Config`` derives them.

    Every block up to the last tap except the tapped indices, so the blocks right
    after taps 3, 6 and 9 (blocks 3, 6 and 9, 0-based) attend globally.

    Args:
        out_indices: Tapped hidden-state indices.
        num_layers: Number of transformer blocks.

    Returns:
        Sorted block indices (0-based) with windowed attention.
    """
    windowed = set(range(max(out_indices) + 1)) - set(out_indices)
    return sorted(i for i in windowed if i < num_layers)


class PatchEmbeddings(nn.Module):
    """Patchify with a strided conv: ``(B, C, H, W) -> (B, H/p * W/p, D)``."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        """Initialize the projection."""
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, N, D)`` patch tokens in row-major order."""
        return self.projection(x).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    """Class token + patch tokens + (interpolated) position embedding."""

    def __init__(
        self, in_channels: int, hidden_size: int, patch_size: int, pos_embed_grid: int
    ):
        """Initialize the embeddings.

        Args:
            in_channels: Input image channels.
            hidden_size: Token width.
            patch_size: Patch size.
            pos_embed_grid: Side of the square grid the learned position embedding
                covers.
        """
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_grid = pos_embed_grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Unused (DINOv2's masked-image-modeling token), kept so checkpoints load
        # strictly.
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(in_channels, hidden_size, patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, pos_embed_grid**2 + 1, hidden_size)
        )
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def interpolate_pos_encoding(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """The position embedding for a ``grid_h`` x ``grid_w`` token grid.

        Bicubic with antialiasing and ``align_corners=False``, in float32, as HF's
        RF-DETR does. Always interpolates while tracing, so exported graphs take
        any input size.
        """
        pos = self.position_embeddings
        g = self.pos_embed_grid
        if not torch.jit.is_tracing() and (grid_h, grid_w) == (g, g):
            return pos
        dim = pos.shape[-1]
        patch_pos = pos[:, 1:].reshape(1, g, g, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([pos[:, :1], patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, 1 + H/p * W/p, D)`` (class token first)."""
        batch, _, height, width = x.shape
        tokens = self.patch_embeddings(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + self.interpolate_pos_encoding(
            height // self.patch_size, width // self.patch_size
        )


class Attention(nn.Module):
    """Multi-head self-attention with separate q/k/v/o projections."""

    def __init__(self, hidden_size: int, num_heads: int):
        """Initialize the projections."""
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the ``(B, S, D)`` sequence."""
        batch, seq, dim = x.shape
        shape = (batch, seq, self.num_heads, dim // self.num_heads)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, dim))


class LayerScale(nn.Module):
    """Per-channel residual branch scale (``lambda1``)."""

    def __init__(self, hidden_size: int, init_value: float = 1.0):
        """Initialize the scale."""
        super().__init__()
        self.lambda1 = nn.Parameter(init_value * torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` per channel."""
        return x * self.lambda1


class MLP(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, hidden_size: int, mlp_ratio: float):
        """Initialize the layers."""
        super().__init__()
        hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.fc2(F.gelu(self.fc1(x)))


class DropPath(nn.Module):
    """Stochastic depth on a residual branch (identity at eval)."""

    def __init__(self, drop_prob: float = 0.0):
        """Initialize with the drop probability."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drop whole samples of ``x`` with probability ``drop_prob`` in training."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Layer(nn.Module):
    """One DINOv2 block, windowed or global.

    Its input is always window-batched, ``(B * n_win, S_win, D)`` with
    ``n_win = num_windows**2``. A global block merges each image's windows into one
    sequence for attention and splits them again afterwards.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        num_windows: int,
        global_attention: bool,
        layer_norm_eps: float = 1e-6,
        drop_path: float = 0.0,
    ):
        """Initialize the block."""
        super().__init__()
        self.num_windows = num_windows
        self.global_attention = global_attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = Attention(hidden_size, num_heads)
        self.layer_scale1 = LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, mlp_ratio)
        self.layer_scale2 = LayerScale(hidden_size)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to window-batched tokens."""
        batch, seq, dim = x.shape
        n_win = self.num_windows**2
        h = x
        if self.global_attention and n_win > 1:
            h = h.reshape(batch // n_win, n_win * seq, dim)
        h = self.attention(self.norm1(h))
        if self.global_attention and n_win > 1:
            h = h.reshape(batch, seq, dim)
        x = x + self.drop_path(self.layer_scale1(h))
        return x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))


class Encoder(nn.Module):
    """The stack of blocks (``encoder.layer.{i}``, as in HF)."""

    def __init__(self, layers: Sequence[nn.Module]):
        """Hold the blocks."""
        super().__init__()
        self.layer = nn.ModuleList(layers)


def window_partition(tokens: torch.Tensor, grid_hw, num_windows: int) -> torch.Tensor:
    """Cut ``(B, 1 + h*w, D)`` tokens into ``num_windows**2`` windows.

    Returns ``(B * num_windows**2, 1 + h*w / num_windows**2, D)``: windows in
    row-major order per image, each led by a copy of the image's class token.
    """
    if num_windows == 1:
        return tokens
    batch, _, dim = tokens.shape
    h, w = grid_hw
    nw = num_windows
    cls, patches = tokens[:, :1], tokens[:, 1:]
    patches = patches.reshape(batch, nw, h // nw, nw, w // nw, dim)
    patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(
        batch * nw * nw, (h // nw) * (w // nw), dim
    )
    cls = cls.repeat_interleave(nw * nw, dim=0)
    return torch.cat([cls, patches], dim=1)


def window_unpartition(tokens: torch.Tensor, grid_hw, num_windows: int) -> torch.Tensor:
    """Inverse of :func:`window_partition` for the patch tokens only.

    Args:
        tokens: ``(B * num_windows**2, h*w / num_windows**2, D)`` window tokens
            without their class tokens.
        grid_hw: The image's token grid ``(h, w)``.
        num_windows: Windows per side.

    Returns:
        ``(B, D, h, w)`` token map.
    """
    h, w = grid_hw
    nw = num_windows
    dim = tokens.shape[-1]
    batch = tokens.shape[0] // (nw * nw)
    tokens = tokens.reshape(batch, nw, nw, h // nw, w // nw, dim)
    tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(batch, h, w, dim)
    return tokens.permute(0, 3, 1, 2).contiguous()


class WindowedDinov2Backbone(nn.Module):
    """RF-DETR's DINOv2 with windowed attention and tapped blocks.

    Implements the :class:`~sleap_nn.architectures.detr.backbones.DetrBackbone`
    interface.

    Attributes:
        out_channels: Token width (``hidden_size``).
        num_taps: ``len(out_indices)``.
        stride: The patch size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        pos_embed_grid: int = 36,
        out_indices: Sequence[int] = (3, 6, 9, 12),
        num_windows: int = 2,
        window_block_indexes: Optional[Sequence[int]] = None,
        layer_norm_eps: float = 1e-6,
        drop_path_rate: float = 0.0,
        normalize_input: bool = True,
    ):
        """Initialize the backbone.

        Args:
            in_channels: Input image channels (3 for DINOv2 weights).
            hidden_size: Token width (384 for ViT-S, 768 for ViT-B).
            num_layers: Transformer blocks.
            num_heads: Attention heads.
            mlp_ratio: MLP hidden width as a multiple of ``hidden_size``.
            patch_size: Patch size (16 in RF-DETR).
            pos_embed_grid: Side of the square token grid the learned position
                embedding covers (RF-DETR: resolution / patch size).
            out_indices: Tapped hidden-state indices (0 = embedding output).
            num_windows: Windows per side (1 = global attention everywhere).
            window_block_indexes: Blocks (0-based) with windowed attention.
                ``None``: :func:`default_window_block_indexes`, as in RF-DETR.
            layer_norm_eps: LayerNorm epsilon.
            drop_path_rate: Stochastic depth, ramped linearly over the blocks.
            normalize_input: Apply the ImageNet mean/std to the ``[0, 1]`` input.

        Raises:
            ValueError: If an ``out_indices`` entry is outside ``[1, num_layers]``.
        """
        super().__init__()
        out_indices = [int(i) for i in out_indices]
        if not all(1 <= i <= num_layers for i in out_indices):
            raise ValueError(
                f"out_indices must be in [1, num_layers={num_layers}], got "
                f"{out_indices}."
            )
        if window_block_indexes is None:
            window_block_indexes = default_window_block_indexes(out_indices, num_layers)
        self.out_channels = hidden_size
        self.num_taps = len(out_indices)
        self.stride = patch_size
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.out_indices = out_indices
        self.window_block_indexes = sorted(int(i) for i in window_block_indexes)
        self.normalize_input = normalize_input
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        self.embeddings = Embeddings(
            in_channels, hidden_size, patch_size, pos_embed_grid
        )
        rates = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.encoder = Encoder(
            Layer(
                hidden_size,
                num_heads,
                mlp_ratio,
                num_windows,
                global_attention=i not in self.window_block_indexes,
                layer_norm_eps=layer_norm_eps,
                drop_path=rates[i],
            )
            for i in range(num_layers)
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # Never used by `forward` (no masked-image modeling; blocks after the last
        # tap are skipped): kept in the state dict for checkpoint compatibility, but
        # frozen so DDP does not wait for their gradients.
        self.embeddings.mask_token.requires_grad_(False)
        for layer in self.encoder.layer[max(out_indices) :]:
            layer.requires_grad_(False)
        # Set by `load_dinov2_weights`, re-applied by `reload_pretrained_weights`.
        self._pretrained_state: Optional[Dict[str, torch.Tensor]] = None

    @property
    def token_multiple(self) -> int:
        """Input sides must be multiples of this (patch size x windows per side)."""
        return self.patch_size * self.num_windows

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Token maps of the tapped blocks.

        Args:
            x: ``(B, C, H, W)`` images in ``[0, 1]``; ``H`` and ``W`` must be
                multiples of :attr:`token_multiple`.

        Returns:
            ``num_taps`` maps, each ``(B, hidden_size, H / patch, W / patch)``.

        Raises:
            ValueError: If the input size does not split into whole windows.
        """
        height, width = x.shape[-2:]
        m = self.token_multiple
        if height % m or width % m:
            raise ValueError(
                f"dinov2_windowed needs input sides that are multiples of {m} "
                f"(patch {self.patch_size} x {self.num_windows} windows), got "
                f"{height}x{width}."
            )
        if self.normalize_input:
            x = (x - self.pixel_mean) / self.pixel_std
        grid = (height // self.patch_size, width // self.patch_size)

        hidden = window_partition(self.embeddings(x), grid, self.num_windows)
        taps = []
        last = max(self.out_indices)
        for i in range(last + 1):
            if i > 0:
                hidden = self.encoder.layer[i - 1](hidden)
            if i in self.out_indices:
                tap = self.layernorm(hidden)[:, 1:]
                taps.append(window_unpartition(tap, grid, self.num_windows))
        return taps

    def encoder_param_groups(
        self,
        base_lr: float,
        encoder_lr_scale: Optional[float] = None,
        layer_decay: Optional[float] = None,
    ) -> Optional[List[Dict]]:
        """Optimizer groups with RF-DETR's layer-wise learning-rate decay.

        RF-DETR (``get_dinov2_lr_decay_rate``) counts ``D = max(out_indices) + 1``
        layers and gives a parameter at depth ``d`` the rate
        ``layer_decay ** (D + 1 - d)``: the embeddings are depth 0, block ``k``
        (0-based) depth ``k + 1``, and the final LayerNorm depth ``D + 1``.
        (``PretrainedBackbone.encoder_param_groups`` counts the blocks instead, one
        decay step fewer.)

        Args:
            base_lr: The backbone's learning rate before layer decay.
            encoder_lr_scale: Multiplies ``base_lr`` (``None`` = 1).
            layer_decay: The per-layer decay; ``None`` or 0: no decay.

        Returns:
            ``None`` when every parameter trains at ``base_lr`` (no scale, no
            decay), else ``{"params", "lr", "name"}`` groups (``depth<d>``) over the
            trainable parameters.
        """
        scale = 1.0 if encoder_lr_scale is None else float(encoder_lr_scale)
        if scale == 1.0 and not layer_decay:
            return None
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if not layer_decay:
            return [
                {"params": [p for _, p in named], "lr": base_lr * scale, "name": "all"}
            ]
        num_layers = max(self.out_indices) + 1
        by_depth: Dict[int, List[torch.Tensor]] = {}
        for name, param in named:
            if name.startswith("embeddings."):
                depth = 0
            elif name.startswith("encoder.layer."):
                depth = int(name.split(".")[2]) + 1
            else:
                depth = num_layers + 1
            by_depth.setdefault(depth, []).append(param)
        return [
            {
                "params": params,
                "lr": base_lr * scale * layer_decay ** (num_layers + 1 - depth),
                "name": f"depth{depth}",
            }
            for depth, params in sorted(by_depth.items())
        ]

    def reload_pretrained_weights(self):
        """Re-apply the DINOv2 weights loaded at build time (after a re-init)."""
        if self._pretrained_state is not None:
            self.load_state_dict(self._pretrained_state, strict=False)


def resample_patch_embed(
    weight: torch.Tensor, new_size: int, interpolation: str = "bicubic"
) -> torch.Tensor:
    """Resample a ``(D, C, p, p)`` patch kernel to ``new_size``.

    Solves for the kernel whose response to a resized patch equals the old
    kernel's response to the original patch (the pseudo-inverse of the resize
    operator), as in FlexiViT and timm's ``resample_patch_embed``.

    Args:
        weight: Patch-embedding kernel.
        new_size: New patch size.
        interpolation: Interpolation of the resize operator.

    Returns:
        ``(D, C, new_size, new_size)`` kernel.
    """
    old = weight.shape[-1]
    if old == new_size:
        return weight
    eye = torch.eye(old * old, dtype=torch.float32).reshape(old * old, 1, old, old)
    resized = F.interpolate(
        eye, size=(new_size, new_size), mode=interpolation, antialias=True
    )
    resize_mat = resized.reshape(old * old, new_size * new_size).T  # (n^2, o^2)
    pinv = torch.linalg.pinv(resize_mat.T)  # (n^2, o^2)
    out = weight.float().reshape(-1, old * old) @ pinv.T
    return out.reshape(*weight.shape[:2], new_size, new_size).to(weight.dtype)


def resize_pos_embed(pos_embed: torch.Tensor, new_grid: int) -> torch.Tensor:
    """Resize a ``(1, 1 + g*g, D)`` position embedding to a ``new_grid`` square."""
    old = int(math.isqrt(pos_embed.shape[1] - 1))
    if old == new_grid:
        return pos_embed
    dim = pos_embed.shape[-1]
    patch = pos_embed[:, 1:].reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch = F.interpolate(
        patch.float(),
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(pos_embed.dtype)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat([pos_embed[:, :1], patch], dim=1)


#: DINOv2 (HF ``Dinov2Model``) -> ``WindowedDinov2Backbone`` key renames.
_DINOV2_RENAMES = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
)


def dinov2_state_dict(
    state_dict: Dict[str, torch.Tensor], patch_size: int, pos_embed_grid: int
) -> Dict[str, torch.Tensor]:
    """Convert an HF ``Dinov2Model`` state dict to a :class:`WindowedDinov2Backbone` one.

    Renames the attention projections, drops the pooler, resamples the patch
    kernel to ``patch_size`` and resizes the position embedding to
    ``pos_embed_grid``.

    Args:
        state_dict: HF DINOv2 tensors (optionally prefixed ``dinov2.``).
        patch_size: Target patch size.
        pos_embed_grid: Target position-embedding grid side.

    Returns:
        The converted state dict.

    Raises:
        ValueError: If the checkpoint has register tokens (DINOv2-with-registers).
    """
    out = {}
    for key, value in state_dict.items():
        if key.startswith("dinov2."):
            key = key[len("dinov2.") :]
        if key.startswith("pooler."):
            continue
        if "register_tokens" in key:
            raise ValueError(
                "DINOv2 checkpoints with register tokens are not supported by "
                "dinov2_windowed; use facebook/dinov2-{small,base}."
            )
        for old, new in _DINOV2_RENAMES:
            key = key.replace(old, new)
        out[key] = value
    proj = "embeddings.patch_embeddings.projection.weight"
    out[proj] = resample_patch_embed(out[proj], patch_size)
    pos = "embeddings.position_embeddings"
    out[pos] = resize_pos_embed(out[pos], pos_embed_grid)
    return out


def load_dinov2_weights(
    backbone: WindowedDinov2Backbone, model_id: str, revision: Optional[str] = None
) -> None:
    """Initialize ``backbone`` from a plain DINOv2 checkpoint on the HF Hub.

    Args:
        backbone: The backbone to load into.
        model_id: HF Hub id (e.g. ``facebook/dinov2-small``) or a local directory
            with ``config.json`` and ``model.safetensors``.
        revision: Optional Hub revision.

    Raises:
        ValueError: If the checkpoint's width, depth or heads differ from the
            backbone's.
    """
    directory = Path(model_id)
    if directory.is_dir():
        config_path = directory / "config.json"
        weights_path = directory / "model.safetensors"
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - depends on the env
            raise ImportError(
                "Loading DINOv2 weights from the Hub needs huggingface_hub; install "
                "sleap-nn[backbones]."
            ) from e
        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        weights_path = hf_hub_download(model_id, "model.safetensors", revision=revision)
    from safetensors.torch import load_file

    config = json.loads(Path(config_path).read_text())
    expected = {
        "hidden_size": backbone.out_channels,
        "num_hidden_layers": len(backbone.encoder.layer),
        "num_attention_heads": backbone.encoder.layer[0].attention.num_heads,
    }
    mismatched = {
        k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v
    }
    if mismatched:
        raise ValueError(
            f"{model_id} does not match the dinov2_windowed config "
            f"(checkpoint, config): {mismatched}."
        )
    state = dinov2_state_dict(
        load_file(str(weights_path)),
        patch_size=backbone.patch_size,
        pos_embed_grid=backbone.embeddings.pos_embed_grid,
    )
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected DINOv2 tensors: {unexpected}")
    if missing:
        logger.warning(f"DINOv2 tensors not in {model_id}: {missing}")
    backbone._pretrained_state = {k: v.detach().clone() for k, v in state.items()}
    logger.info(f"Loaded dinov2_windowed weights from {model_id}.")


@register_detr_backbone("dinov2_windowed")
def build_dinov2_windowed(config: DictConfig) -> WindowedDinov2Backbone:
    """Build a :class:`WindowedDinov2Backbone` from a ``Dinov2WindowedConfig`` leaf."""
    backbone = WindowedDinov2Backbone(
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        patch_size=config.patch_size,
        pos_embed_grid=config.pos_embed_grid,
        out_indices=list(config.out_indices),
        num_windows=config.num_windows,
        window_block_indexes=(
            None
            if config.window_block_indexes is None
            else list(config.window_block_indexes)
        ),
        drop_path_rate=config.drop_path_rate,
    )
    if config.weights:
        load_dinov2_weights(backbone, config.weights, revision=config.revision)
    return backbone
