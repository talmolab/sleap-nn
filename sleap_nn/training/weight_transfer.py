"""Shape-tolerant, skeleton-aware weight transfer, re-initialization and freezing.

`LightningModel` uses this to initialize a model from another sleap-nn checkpoint
whose architecture differs in places (``model_config.init_from``):

* Tensors whose key and shape match are copied as-is.
* Per-node output channels of a head (confidence maps, class maps) are copied by
  node NAME from the source skeleton into the target skeleton, so a COCO-17 model
  can seed a 46-node union model: the 17 shared channels are copied, the 29 new
  ones keep their fresh init. Reordered skeletons are permuted, not copied blindly.
* The class rows of a ``bbox_detr`` detector (its decoder class head and the
  per-group encoder class heads) are copied by class NAME the same way, so an
  imported 91-class COCO RF-DETR seeds a ``["person"]`` detector from its person
  row (see `detr_class_names`).
* Learned positional embeddings on a different token grid are resized
  (bicubic), keeping prefix (CLS / register) tokens.
* Anything else that does not fit is skipped (the target keeps its init) and
  reported -- never a silent ``strict=False`` and never a crash on a shape
  mismatch.

It also holds the module selectors shared by re-initialization and freezing
(``"head"``, ``"encoder"``, ``"blocks[-2:]"``, globs on parameter names, ...).
"""

import fnmatch
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

#: Key under which `LightningModel.on_save_checkpoint` stores the heads' channel
#: names, so a checkpoint describes its own output channels.
CHECKPOINT_HEADS_KEY = "sleap_nn_heads"

# Selector aliases -> globs on state-dict keys (relative to the `Model`).
_ALIASES: Dict[str, List[str]] = {
    "all": ["*"],
    "backbone": ["backbone.*"],
    "head": ["head_layers.*"],
    "heads": ["head_layers.*"],
    # `dec`: hierarchical/pyramid backbones; `decoders`: UNet; `vit_dec`: ViTPose head.
    "decoder": ["backbone.dec.*", "backbone.decoders.*", "backbone.vit_dec.*"],
    "neck": ["backbone.neck.*", "backbone.tap_norms.*"],
}
_BLOCKS_RE = re.compile(r"^blocks\[(-?\d*):(-?\d*)\]$")
_POS_EMBED_SUFFIXES = (
    "pos_embed",
    "position_embeddings",
    "position_embedding.weight",
)
_LAYER_SCALE_NAMES = {"gamma", "gamma_1", "gamma_2", "lambda1"}

#: Head name and class-row name of the never-positive extra logit of a
#: ``bbox_detr`` class head (``len(categories) + 1`` logits).
DETR_HEAD_NAME = "BBoxDetrHead"
DETR_EXTRA_LOGIT = "__extra__"
# The class heads of a `bbox_detr` model: one row per class, plus the extra logit.
_DETR_CLASS_KEY_RE = re.compile(
    r"^detr\.(?:class_embed|enc_out_class_embed\.\d+)\.(?:weight|bias)$"
)


# ----------------------------------------------------------------------- report


@dataclass
class TransferReport:
    """What a weight transfer did, tensor by tensor.

    Attributes:
        source: Where the weights came from.
        loaded: Keys copied unchanged.
        remapped: Keys whose per-node channels were copied by name -> description.
        resized: Keys resampled to the target shape (positional embeddings).
        reinitialized: Keys / channels left at (or reset to) a fresh init ->
            description.
        skipped: Selected source tensors that could not be used -> reason (the
            target keeps its init).
        missing: Selected target keys absent from the source (kept at init).
        unexpected: Source keys with no counterpart in the target.
        excluded: Number of target keys left out by ``include`` / ``exclude``.
        warnings: Transfers that went through but may be wrong (e.g. per-node
            head channels copied by position because the source node names are
            unknown).
    """

    source: str = ""
    loaded: List[str] = field(default_factory=list)
    remapped: Dict[str, str] = field(default_factory=dict)
    resized: Dict[str, str] = field(default_factory=dict)
    reinitialized: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    excluded: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def num_transferred(self) -> int:
        """Number of target tensors that received source values."""
        return len(self.loaded) + len(self.remapped) + len(self.resized)

    def summary(self, max_items: int = 6) -> str:
        """A multi-line, human-readable summary."""

        def items(xs) -> str:
            xs = list(xs.items()) if isinstance(xs, dict) else list(xs)
            lines = [
                f"      {x[0]}: {x[1]}" if isinstance(x, tuple) else f"      {x}"
                for x in xs[:max_items]
            ]
            if len(xs) > max_items:
                lines.append(f"      ... (+{len(xs) - max_items} more)")
            return "\n".join(lines)

        out = [f"Weight transfer from {self.source or '<state dict>'}:"]
        out.append(f"  loaded        {len(self.loaded):5d} tensors")
        for name in ("remapped", "resized", "reinitialized", "skipped", "missing"):
            val = getattr(self, name)
            out.append(f"  {name:<13} {len(val):5d}")
            if val:
                out.append(items(val))
        out.append(f"  unexpected    {len(self.unexpected):5d} (source-only keys)")
        if self.unexpected:
            out.append(items(self.unexpected))
        if self.excluded:
            out.append(f"  excluded      {self.excluded:5d} (by include/exclude)")
        for w in self.warnings:
            out.append(f"  WARNING: {w}")
        return "\n".join(out)

    def log(self) -> None:
        """Log the summary (WARNING when something was skipped or is suspect)."""
        (logger.warning if self.skipped or self.warnings else logger.info)(
            self.summary()
        )


# -------------------------------------------------------------------- selectors


def _model_of(root: nn.Module) -> nn.Module:
    """The `Model` (backbone + head_layers) inside ``root`` (or ``root`` itself)."""
    inner = getattr(root, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "head_layers"):
        return inner
    return root


def _key_prefix(root: nn.Module) -> str:
    """State-dict key prefix of the `Model` inside ``root`` (``"model."`` or ``""``)."""
    return "model." if _model_of(root) is not root else ""


def vit_blocks(backbone: nn.Module) -> Optional[Tuple[str, List[nn.Module]]]:
    """Locate a plain ViT's transformer block list inside a backbone.

    Returns:
        ``(qualified name relative to the backbone, blocks)`` or ``None`` if the
        backbone is not a plain ViT (`PretrainedBackbone` on the token path).
    """
    if not getattr(backbone, "token_path", False):
        return None
    if getattr(backbone, "is_timm", False):
        return "enc.timm_model.blocks", list(backbone.enc.timm_model.blocks)
    num_layers = getattr(backbone, "num_layers", None)
    found = [
        (name, m)
        for name, m in backbone.enc.named_modules()
        if isinstance(m, nn.ModuleList) and len(m._modules) == num_layers
    ]
    if len(found) != 1:
        return None
    name, blocks = found[0]
    return f"enc.{name}", list(blocks._modules.values())


def _encoder_prefixes(model: nn.Module) -> List[str]:
    """Key prefixes (relative to the `Model`) of the backbone's encoder modules."""
    backbone = model.backbone
    fn = getattr(backbone, "encoder_modules", None)
    if fn is None:
        return ["backbone.enc"] if hasattr(backbone, "enc") else []
    ids = {id(m) for m in fn()}
    return [f"backbone.{n}" for n, m in backbone.named_modules() if id(m) in ids]


def _embedding_param_names(model: nn.Module) -> List[str]:
    """Encoder parameters registered before the first ViT block.

    These are the patch/position embeddings and CLS/register tokens, named
    relative to the `Model`.
    """
    found = vit_blocks(model.backbone)
    if found is None:
        return []
    blocks_name = found[0]
    names = []
    for name, _ in model.backbone.named_parameters():
        if name.startswith(blocks_name + "."):
            break
        if name.startswith("enc."):
            names.append(f"backbone.{name}")
    return names


def resolve_selector(root: nn.Module, selector: str) -> List[str]:
    """Turn a selector into key globs relative to ``root``'s state dict.

    Selectors:
        ``"all"``, ``"backbone"``, ``"head"``/``"heads"``, ``"decoder"``,
        ``"neck"`` (plain-ViT neck + tap norms), ``"encoder"`` (the backbone's
        pretrained encoder, as ``freeze`` defines it), ``"embeddings"`` (a plain
        ViT's patch/position embeddings and prefix tokens), ``"blocks[a:b]"``
        (a Python slice over a plain ViT's transformer blocks, e.g.
        ``"blocks[-2:]"``), or a glob / exact key (``"backbone.enc.*norm*"``),
        with or without the ``model.`` prefix.
    """
    model = _model_of(root)
    prefix = _key_prefix(root)
    sel = selector.strip()
    if sel in _ALIASES:
        globs = _ALIASES[sel]
    elif sel == "encoder":
        globs = [f"{p}.*" for p in _encoder_prefixes(model)]
    elif sel == "embeddings":
        globs = _embedding_param_names(model)
    elif _BLOCKS_RE.match(sel):
        found = vit_blocks(model.backbone)
        if found is None:
            raise ValueError(
                f"Selector {selector!r} needs a plain-ViT `pretrained` backbone."
            )
        name, blocks = found
        a, b = _BLOCKS_RE.match(sel).groups()
        idx = list(range(len(blocks)))[
            slice(int(a) if a else None, int(b) if b else None)
        ]
        globs = [f"backbone.{name}.{i}.*" for i in idx]
    else:
        g = sel[len("model.") :] if sel.startswith("model.") else sel
        globs = [g]
    return [prefix + g for g in globs]


def _matches(key: str, globs: Sequence[str]) -> bool:
    return any(
        key == g or fnmatch.fnmatchcase(key, g) or key.startswith(g + ".")
        for g in globs
    )


def select_keys(
    root: nn.Module,
    keys: Iterable[str],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Keys selected by ``include`` (default: all) minus ``exclude``."""
    inc = [g for s in (include or ["all"]) for g in resolve_selector(root, s)]
    exc = [g for s in (exclude or []) for g in resolve_selector(root, s)]
    return [k for k in keys if _matches(k, inc) and not _matches(k, exc)]


# ---------------------------------------------------------------- head channels


def head_channel_names(head: Any) -> Optional[List[str]]:
    """Names of a head's output channels, or ``None`` if they are not per-node.

    Covers heads with one channel per node (confidence maps: single-instance,
    centered-instance, multi-instance) and class maps. PAFs / offsets are not
    named (their channels fall back to shape matching).
    """
    from sleap_nn.architectures.heads import ClassMapsHead

    if isinstance(head, ClassMapsHead):
        names = list(head.classes)
    else:
        names = getattr(head, "part_names", None)
        names = list(names) if names is not None else None
    if names is None or len(names) != head.channels:
        return None
    return [str(n) for n in names]


def detr_class_names(detr_config) -> List[str]:
    """Row names of a ``bbox_detr`` class head: its categories, then the extra logit."""
    from sleap_nn.config.model_config import DEFAULT_BBOX_CATEGORY

    categories = detr_config.categories or [DEFAULT_BBOX_CATEGORY]
    return [str(c) for c in categories] + [DETR_EXTRA_LOGIT]


def model_head_channels(root: nn.Module) -> List[Dict[str, Any]]:
    """``[{"name", "channel_names"}]`` for each head of the model in ``root``.

    A ``bbox_detr`` model has one entry, its class heads' rows
    (`detr_class_names`).
    """
    model = _model_of(root)
    if getattr(model, "model_type", None) == "bbox_detr":
        return [
            {
                "name": DETR_HEAD_NAME,
                "channel_names": detr_class_names(model.head_configs.detr),
            }
        ]
    return [
        {"name": h.name, "channel_names": head_channel_names(h)}
        for h in getattr(model, "heads", [])
    ]


def _heads_from_config(cfg: DictConfig) -> Optional[List[Dict[str, Any]]]:
    """Rebuild a checkpoint's head channel names from its ``training_config``."""
    from sleap_nn.architectures.model import get_head
    from sleap_nn.config.utils import get_model_type_from_cfg

    model_type = get_model_type_from_cfg(cfg)
    if model_type is None:
        return None
    if model_type == "bbox_detr":
        detr = cfg.model_config.head_configs.bbox_detr.detr
        return [{"name": DETR_HEAD_NAME, "channel_names": detr_class_names(detr)}]
    head_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.model_config.head_configs[model_type])
    )
    skeletons = OmegaConf.select(cfg, "data_config.skeletons", default=None)
    node_names = None
    if skeletons:
        sk = skeletons[0]
        nodes = sk.get("nodes", []) if hasattr(sk, "get") else []
        node_names = [n["name"] if hasattr(n, "get") else str(n) for n in nodes]
    for sub in head_cfg.values():
        if (
            sub is not None
            and "part_names" in sub
            and sub.part_names is None
            and node_names
        ):
            sub.part_names = node_names
    try:
        heads = get_head(model_type, head_cfg)
    except Exception as e:  # noqa: BLE001 - best effort; fall back to shapes
        logger.warning(f"Could not rebuild source heads from its config: {e}")
        return None
    return [{"name": h.name, "channel_names": head_channel_names(h)} for h in heads]


def load_source(
    path: str,
) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict[str, Any]]]]:
    """Load a checkpoint's state dict and (if known) its heads' channel names.

    Channel names come from the checkpoint itself (``CHECKPOINT_HEADS_KEY``,
    written by sleap-nn >= this version) or, for older checkpoints, from the
    ``training_config.yaml`` saved next to it.

    Args:
        path: A Lightning ``.ckpt``, a raw ``state_dict`` file, or a model
            directory (its ``best.ckpt`` is used).
    """
    p = Path(path)
    if p.is_dir():
        p = p / "best.ckpt"
    ckpt = torch.load(p.as_posix(), map_location="cpu", weights_only=False)
    if isinstance(ckpt, Mapping) and "state_dict" in ckpt:
        sd = dict(ckpt["state_dict"])
    else:
        sd = dict(ckpt)
    heads = ckpt.get(CHECKPOINT_HEADS_KEY) if isinstance(ckpt, Mapping) else None
    if heads is None:
        cfg_path = p.parent / "training_config.yaml"
        if cfg_path.exists():
            heads = _heads_from_config(OmegaConf.load(cfg_path.as_posix()))
    return sd, heads


# ------------------------------------------------------------------- transfer

_HEAD_KEY_RE = re.compile(r"^head_layers\.(\d+)\.(.*)$")


def _resize_pos_embed(
    src: torch.Tensor, tgt_shape: torch.Size
) -> Optional[torch.Tensor]:
    """Bicubically resize a learned ``(1, P + H*W, C)`` / ``(P + H*W, C)`` table."""
    squeeze = src.dim() == 2
    s = src[None] if squeeze else src
    t_n = tgt_shape[0] if squeeze else tgt_shape[1]
    if s.dim() != 3 or s.shape[-1] != tgt_shape[-1]:
        return None
    for p in range(0, 9):
        gs, gt = math.isqrt(s.shape[1] - p), math.isqrt(t_n - p)
        if gs * gs == s.shape[1] - p and gt * gt == t_n - p and gs > 0 and gt > 0:
            grid = s[:, p:].reshape(1, gs, gs, -1).permute(0, 3, 1, 2).float()
            grid = F.interpolate(
                grid, size=(gt, gt), mode="bicubic", align_corners=False
            )
            grid = grid.permute(0, 2, 3, 1).reshape(1, gt * gt, -1).to(src.dtype)
            out = torch.cat([s[:, :p], grid], dim=1)
            return out[0] if squeeze else out
    return None


def _remap_channels(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_names: List[str],
    tgt_names: List[str],
    node_map: Mapping[str, str],
) -> Tuple[Optional[torch.Tensor], List[str], List[str]]:
    """Copy dim-0 channels of ``src`` into a copy of ``tgt`` by name.

    Returns:
        ``(tensor or None, copied target names, fresh target names)``.
    """
    if (
        src.shape[0] != len(src_names)
        or tgt.shape[0] != len(tgt_names)
        or src.shape[1:] != tgt.shape[1:]
    ):
        return None, [], []
    src_index = {n: i for i, n in enumerate(src_names)}
    out = tgt.detach().clone()
    copied, fresh = [], []
    for ti, name in enumerate(tgt_names):
        si = src_index.get(node_map.get(name, name))
        if si is None:
            fresh.append(name)
        else:
            out[ti] = src[si].to(out.dtype)
            copied.append(name)
    return out, copied, fresh


def _head_output_keys(root: nn.Module) -> set:
    """State-dict keys of each head's output layer (its last layer with weights).

    Only these carry one slice per output channel (node); hidden layers of a
    deeper head (e.g. a ViTPose deconv stack) are matched by shape like any
    other tensor.
    """
    model = _model_of(root)
    prefix = _key_prefix(root)
    keys = set()
    for i, head in enumerate(getattr(model, "head_layers", [])):
        last = None
        for name, m in head.named_modules():
            if any(p is not None for p in m._parameters.values()):
                last = (name, m)
        if last is None:
            continue
        name, m = last
        for pname, p in m._parameters.items():
            if p is not None:
                sub = f"{name}.{pname}" if name else pname
                keys.add(f"{prefix}head_layers.{i}.{sub}")
    return keys


def _match_heads(
    target_heads: List[Dict[str, Any]], source_heads: Optional[List[Dict[str, Any]]]
) -> Dict[int, int]:
    """Target head index -> source head index (same head class, in order)."""
    if source_heads is None:
        return {i: i for i in range(len(target_heads))}
    used, out = set(), {}
    for ti, th in enumerate(target_heads):
        for si, sh in enumerate(source_heads):
            if si not in used and sh["name"] == th["name"]:
                out[ti] = si
                used.add(si)
                break
    return out


def transfer_state_dict(
    root: nn.Module,
    source_sd: Mapping[str, torch.Tensor],
    source_heads: Optional[List[Dict[str, Any]]] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    remap_nodes: bool = True,
    node_map: Optional[Mapping[str, str]] = None,
    source_part_names: Optional[Sequence[str]] = None,
    interpolate_pos_embed: bool = True,
    source: str = "",
) -> TransferReport:
    """Load what fits from ``source_sd`` into ``root``; report the rest.

    Args:
        root: A `LightningModel` or a bare `Model`.
        source_sd: Source state dict (keys with or without the ``model.`` prefix).
        source_heads: Source ``[{"name", "channel_names"}]`` (see `load_source`).
            ``None``: heads are matched by index and per-node channels by shape.
        include: Selectors of target keys to load (default: everything).
        exclude: Selectors of target keys to leave at their init.
        remap_nodes: Copy per-node head channels by node name.
        node_map: Optional ``{target node: source node}`` overrides, e.g. to seed a
            new node from a related one or to bridge a renamed node.
        source_part_names: Source node names, overriding ``source_heads`` for
            every per-node head (for checkpoints that do not record them).
        interpolate_pos_embed: Resize learned positional embeddings to the target
            grid instead of skipping them.
        source: Label for the report.

    Returns:
        A `TransferReport`.
    """
    report = TransferReport(source=source)
    prefix = _key_prefix(root)
    node_map = dict(node_map or {})

    # Normalize source keys to the target's convention.
    sd: Dict[str, torch.Tensor] = {}
    for k, v in source_sd.items():
        k = k[len("model.") :] if k.startswith("model.") else k
        sd[prefix + k] = v

    target_heads = model_head_channels(root)
    output_keys = _head_output_keys(root)
    if source_part_names is not None:
        source_heads = [
            {
                "name": h["name"],
                "channel_names": (
                    list(source_part_names) if h["channel_names"] is not None else None
                ),
            }
            for h in (source_heads or target_heads)
        ]
    head_map = _match_heads(target_heads, source_heads)

    target_sd = root.state_dict()
    selected = set(select_keys(root, target_sd.keys(), include, exclude))
    report.excluded = len(target_sd) - len(selected)
    consumed = set()
    new_sd: Dict[str, torch.Tensor] = {}
    fresh_reported = set()

    for key, tgt in target_sd.items():
        if key not in selected:
            continue
        rel = key[len(prefix) :]
        m = _HEAD_KEY_RE.match(rel)
        ti = int(m.group(1)) if m else None
        if m is not None:
            if ti not in head_map:
                report.missing.append(key)
                continue
            src_key = f"{prefix}head_layers.{head_map[ti]}.{m.group(2)}"
            # Head module names embed the head class (same class by construction).
        else:
            src_key = key
        if src_key not in sd:
            report.missing.append(key)
            continue
        src = sd[src_key]
        consumed.add(src_key)

        if (
            remap_nodes
            and _DETR_CLASS_KEY_RE.match(rel)
            and target_heads
            and target_heads[0]["name"] == DETR_HEAD_NAME
            and source_heads
            and source_heads[0]["name"] == DETR_HEAD_NAME
        ):
            s_names = source_heads[0]["channel_names"]
            t_names = target_heads[0]["channel_names"]
            if s_names != t_names or node_map:
                out, copied, fresh = _remap_channels(
                    src, tgt, s_names, t_names, node_map
                )
                if out is None:
                    report.skipped[key] = (
                        f"shape {tuple(src.shape)} does not match the source "
                        f"classes ({len(s_names)} rows)"
                    )
                    continue
                new_sd[key] = out
                report.remapped[key] = (
                    f"{len(copied)}/{len(t_names)} class rows copied by name "
                    f"from {len(s_names)} source rows"
                )
                if fresh and "detr (classes)" not in fresh_reported:
                    fresh_reported.add("detr (classes)")
                    report.reinitialized[f"{prefix}detr (classes)"] = (
                        f"{len(fresh)} new class(es) keep fresh init: "
                        + ", ".join(fresh[:8])
                    )
                continue

        if (
            key in output_keys
            and remap_nodes
            and source_heads is not None
            and target_heads[ti]["channel_names"] is not None
            and source_heads[head_map[ti]]["channel_names"] is not None
        ):
            s_names = source_heads[head_map[ti]]["channel_names"]
            t_names = target_heads[ti]["channel_names"]
            if (s_names != t_names or node_map) and src.dim() >= 1:
                out, copied, fresh = _remap_channels(
                    src, tgt, s_names, t_names, node_map
                )
                if out is not None:
                    new_sd[key] = out
                    report.remapped[key] = (
                        f"{len(copied)}/{len(t_names)} channels copied by node name "
                        f"from {len(s_names)} source channels"
                    )
                    head_tag = f"{prefix}head_layers.{ti} (channels)"
                    if fresh and head_tag not in fresh_reported:
                        fresh_reported.add(head_tag)
                        report.reinitialized[head_tag] = (
                            f"{len(fresh)} new node(s) keep fresh init: "
                            + ", ".join(fresh[:8])
                            + (" ..." if len(fresh) > 8 else "")
                        )
                    continue

        if (
            key in output_keys
            and remap_nodes
            and target_heads[ti]["channel_names"] is not None
            and (
                source_heads is None
                or source_heads[head_map[ti]]["channel_names"] is None
            )
            and src.shape == tgt.shape
        ):
            note = (
                f"{prefix}head_layers.{ti}: source node names unknown, channels "
                "copied by position (set init_from.source_part_names if the "
                "source skeleton differs in order)"
            )
            if note not in report.warnings:
                report.warnings.append(note)

        if src.shape == tgt.shape:
            new_sd[key] = src
            report.loaded.append(key)
        elif interpolate_pos_embed and key.endswith(_POS_EMBED_SUFFIXES):
            out = _resize_pos_embed(src, tgt.shape)
            if out is None:
                report.skipped[key] = (
                    f"shape {tuple(src.shape)} -> {tuple(tgt.shape)} (not a square grid)"
                )
            else:
                new_sd[key] = out
                report.resized[key] = f"{tuple(src.shape)} -> {tuple(tgt.shape)}"
        else:
            report.skipped[key] = (
                f"shape {tuple(src.shape)} != target {tuple(tgt.shape)}"
            )

    report.unexpected = [k for k in sd if k not in consumed and k not in target_sd]
    root.load_state_dict(new_sd, strict=False)
    return report


def init_from_checkpoint(
    root: nn.Module,
    path: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    remap_nodes: bool = True,
    node_map: Optional[Mapping[str, str]] = None,
    source_part_names: Optional[Sequence[str]] = None,
    interpolate_pos_embed: bool = True,
    strict: bool = False,
) -> TransferReport:
    """`load_source` + `transfer_state_dict`, logging the report.

    Args:
        root: A `LightningModel` or a bare `Model`.
        path: Checkpoint file or model directory (see `load_source`).
        include: Selectors of target weights to load (default: everything).
        exclude: Selectors of target weights to leave at their init.
        remap_nodes: Copy per-node head channels by node name.
        node_map: ``{target node: source node}`` overrides.
        source_part_names: Source node names (overrides the checkpoint's).
        interpolate_pos_embed: Resize learned positional embeddings.
        strict: Raise if any selected source tensor had to be skipped.

    Returns:
        The `TransferReport` (also logged).
    """
    sd, heads = load_source(path)
    report = transfer_state_dict(
        root,
        sd,
        source_heads=heads,
        include=include,
        exclude=exclude,
        remap_nodes=remap_nodes,
        node_map=node_map,
        source_part_names=source_part_names,
        interpolate_pos_embed=interpolate_pos_embed,
        source=str(path),
    )
    report.log()
    if strict and report.skipped:
        message = (
            f"init_from strict=True but {len(report.skipped)} tensor(s) were skipped: "
            f"{list(report.skipped)[:5]}"
        )
        logger.error(message)
        raise ValueError(message)
    if report.num_transferred == 0:
        logger.warning(f"No weights were transferred from {path}.")
    return report


# --------------------------------------------------------------- re-initialize


def _layer_scale_params(module: nn.Module) -> List[torch.nn.Parameter]:
    out = []
    for name, p in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        owner = name.rsplit(".", 2)[-2] if name.count(".") else ""
        if p.dim() == 1 and (
            leaf in _LAYER_SCALE_NAMES or owner.startswith(("ls", "layer_scale"))
        ):
            out.append(p)
    return out


@torch.no_grad()
def reinit_module(
    module: nn.Module, scheme: str = "default", layer_scale: float = 1e-5
) -> None:
    """Re-initialize every layer in ``module`` in place.

    Args:
        module: The module to reset.
        scheme: ``"xavier"`` (xavier-uniform conv/linear, zero bias -- sleap-nn's
            ``init_weights: xavier``), ``"default"`` (each layer's own
            ``reset_parameters``) or ``"vit"`` (truncated normal std 0.02 for
            linear layers, zero bias, timm/HF ViT style).
        layer_scale: Value for LayerScale parameters (``"vit"``). Small values
            make a re-initialized block start near identity on the residual stream.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif scheme == "vit":
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                m.reset_parameters()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)) or isinstance(
            m, nn.modules.batchnorm._BatchNorm
        ):
            m.reset_parameters()
    if scheme == "vit":
        for p in _layer_scale_params(module):
            p.fill_(layer_scale)


def reinit(
    root: nn.Module,
    selectors: Sequence[str],
    scheme: Optional[str] = None,
    layer_scale: float = 1e-5,
) -> Dict[str, str]:
    """Re-initialize the modules picked by ``selectors``.

    ViT blocks (``"blocks[a:b]"``, ``"embeddings"`` excluded -- they are not
    re-initialized) use the ``"vit"`` scheme; everything else uses ``scheme``
    (default: the model's ``init_weights``, i.e. ``"xavier"`` or ``"default"``).

    Returns:
        ``{module name: description}``.
    """
    model = _model_of(root)
    prefix = _key_prefix(root)
    if scheme is None:
        scheme = (
            "xavier" if getattr(root, "init_weights", None) == "xavier" else "default"
        )
    done: Dict[str, str] = {}
    named = dict(root.named_modules())
    for sel in selectors:
        if sel == "embeddings":
            raise ValueError("Re-initializing embeddings is not supported.")
        is_blocks = bool(_BLOCKS_RE.match(sel.strip()))
        globs = resolve_selector(root, sel)
        # Module-level: the shallowest modules whose name matches.
        picked = []
        for name, m in named.items():
            if not name or not _matches(name + ".x", globs):
                continue
            if any(name.startswith(p + ".") for p in picked):
                continue
            picked.append(name)
        if not picked:
            logger.warning(f"reinit selector {sel!r} matched no module.")
        sch = "vit" if is_blocks else scheme
        for name in picked:
            reinit_module(named[name], sch, layer_scale=layer_scale)
            n = sum(p.numel() for p in named[name].parameters())
            done[name] = f"{sch} init ({n} params) [{sel}]"
        if is_blocks:
            bb = model.backbone
            taps = getattr(bb, "layers", None)
            found = vit_blocks(bb)
            if taps and found and max(taps) < len(found[1]):
                logger.warning(
                    f"reinit {sel!r}: the backbone taps layers {taps}; blocks after "
                    f"layer {max(taps)} do not run."
                )
    for k, v in done.items():
        logger.info(f"Re-initialized {k}: {v}")
    return done


# --------------------------------------------------------------------- freeze


def select_params(root: nn.Module, selectors: Sequence[str]) -> List[str]:
    """Names of the parameters of ``root`` picked by ``selectors``."""
    globs = [g for s in selectors for g in resolve_selector(root, s)]
    return [name for name, _ in root.named_parameters() if _matches(name, globs)]


def set_requires_grad(
    root: nn.Module, selectors: Sequence[str], requires_grad: bool
) -> List[str]:
    """Set ``requires_grad`` on the parameters picked by ``selectors``.

    Returns:
        The parameter names changed (parameters already in the requested state
        are left out, so the caller can undo exactly what it did).
    """
    params = dict(root.named_parameters())
    changed = []
    for name in select_params(root, selectors):
        p = params[name]
        if p.requires_grad != requires_grad:
            p.requires_grad_(requires_grad)
            changed.append(name)
    return changed
