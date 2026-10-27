"""Optimizer parameter-group helpers (weight decay / no-decay split).

The ViT fine-tuning recipes (ViTPose, MAE, DINOv2, timm) apply weight decay to the
weight matrices / conv kernels only: biases, normalization weights, any other 1-D
parameter, and the learned position embeddings and CLS / register / mask tokens
are trained without it. `split_weight_decay` applies that rule to an existing list
of optimizer param groups (e.g. the per-depth groups `layer_decay` makes), so
every learning-rate group keeps its learning rate and gets the right decay.

`resolve_encoder_lr_options` reads the encoder learning-rate knobs
(``encoder_lr_scale`` / ``layer_decay``) from ``trainer_config.optimizer``, falling
back to their deprecated location on ``backbone_config.pretrained``.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

# Learned position embeddings / special tokens (names from timm, HF transformers
# and torchvision), plus Swin's relative-position table and Swinv2's attention
# temperature. Matched as a whole dotted-name component.
_NO_DECAY_NAME_RE = re.compile(
    r"(?:^|\.)(?:pos_embed|pos_embedding|position_embedding|position_embeddings|"
    r"absolute_pos_embed|cls_token|class_embedding|reg_token|register_tokens|"
    r"mask_token|dist_token|relative_position_bias_table|logit_scale)(?:$|\.)"
)


def no_weight_decay_names(module: nn.Module) -> Set[str]:
    """Parameter-name prefixes a submodule declares exempt from weight decay.

    timm models (and some others) list their position embeddings / tokens in a
    ``no_weight_decay()`` method, relative to themselves; this collects them for
    every submodule as fully-qualified names under ``module``.

    Args:
        module: The root module (e.g. the LightningModule).

    Returns:
        Fully-qualified names; a parameter is exempt when its name equals one or
        starts with one followed by ``"."``.
    """
    names: Set[str] = set()
    for mod_name, m in module.named_modules():
        fn = getattr(m, "no_weight_decay", None)
        if not callable(fn):
            continue
        try:
            declared = fn()
        except Exception:  # noqa: BLE001 - optional API; ignore odd signatures
            continue
        for n in declared or ():
            names.add(f"{mod_name}.{n}" if mod_name else str(n))
    return names


def is_embedding_or_token(name: str) -> bool:
    """Whether a parameter name is a learned position embedding or special token.

    Args:
        name: Fully-qualified parameter name.
    """
    return bool(_NO_DECAY_NAME_RE.search(name))


def is_no_decay(name: str, param: torch.Tensor, extra_names: Set[str] = ()) -> bool:
    """Whether a parameter trains without weight decay.

    Args:
        name: Fully-qualified parameter name.
        param: The parameter.
        extra_names: Names / name prefixes from `no_weight_decay_names`.
    """
    if param.ndim <= 1 or name.endswith(".bias"):
        return True
    if is_embedding_or_token(name):
        return True
    return any(name == n or name.startswith(n + ".") for n in extra_names)


def split_weight_decay(
    groups: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
    module: nn.Module,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Split optimizer param groups into decayed and no-decay halves.

    Args:
        groups: A flat iterable of parameters (one group) or a list of param-group
            dicts (``{"params": [...], "lr": ...}``, as made by
            ``LightningModel._trainable_param_groups``). Every key other than
            ``params`` (the group's ``lr``, ...) is kept on both halves.
        module: The module the parameters belong to (used for their names).
        weight_decay: Decay for the weight matrices / kernels.

    Returns:
        Param-group dicts, each with an explicit ``weight_decay`` and a ``name``
        (``"<group>/decay"`` or ``"<group>/no_decay"``, used e.g. by Lightning's
        ``LearningRateMonitor``). Empty halves are dropped.
    """
    groups = list(groups)
    if not groups:
        return []
    if not isinstance(groups[0], dict):
        groups = [{"params": groups, "name": "base"}]
    name_of = {id(p): n for n, p in module.named_parameters()}
    extra = no_weight_decay_names(module)
    out: List[Dict[str, Any]] = []
    for i, group in enumerate(groups):
        rest = {k: v for k, v in group.items() if k != "params"}
        base_name = rest.pop("name", f"group{i}")
        decay, no_decay = [], []
        for p in group["params"]:
            name = name_of.get(id(p), "")
            (no_decay if is_no_decay(name, p, extra) else decay).append(p)
        if decay:
            out.append(
                {
                    **rest,
                    "params": decay,
                    "weight_decay": float(weight_decay),
                    "name": f"{base_name}/decay",
                }
            )
        if no_decay:
            out.append(
                {
                    **rest,
                    "params": no_decay,
                    "weight_decay": 0.0,
                    "name": f"{base_name}/no_decay",
                }
            )
    return out


#: Encoder learning-rate knobs that moved from ``backbone_config.pretrained`` to
#: ``trainer_config.optimizer``, with the backbone key's default (= "not set").
_ENCODER_LR_KEYS = {"encoder_lr_scale": 1.0, "layer_decay": None}
_PRETRAINED_KEY = "model_config.backbone_config.pretrained"


def resolve_encoder_lr_options(
    config: DictConfig, warn: bool = True
) -> Tuple[Optional[float], Optional[float]]:
    """The effective ``(encoder_lr_scale, layer_decay)`` of a training config.

    ``trainer_config.optimizer.<key>`` wins. When it is unset (``None``), a
    non-default ``model_config.backbone_config.pretrained.<key>`` (the location
    before 0.4) is used instead, with a deprecation warning. Setting both to
    different values is an error.

    Args:
        config: The full training config.
        warn: Log the deprecation warning when a backbone key is used.

    Returns:
        ``(encoder_lr_scale, layer_decay)``; ``None`` for a knob that is set in
        neither place (``encoder_lr_scale`` then means ``1.0``).

    Raises:
        ValueError: If the two locations disagree.
    """
    out = []
    for key, backbone_default in _ENCODER_LR_KEYS.items():
        value = OmegaConf.select(
            config, f"trainer_config.optimizer.{key}", default=None
        )
        legacy = OmegaConf.select(config, f"{_PRETRAINED_KEY}.{key}", default=None)
        if legacy is not None and legacy != backbone_default:
            if value is None:
                if warn:
                    logger.warning(
                        f"backbone_config.pretrained.{key} is deprecated; set "
                        f"trainer_config.optimizer.{key}: {legacy} instead."
                    )
                value = legacy
            elif value != legacy:
                raise ValueError(
                    f"trainer_config.optimizer.{key}={value} conflicts with the "
                    f"deprecated backbone_config.pretrained.{key}={legacy}; remove "
                    "the backbone key."
                )
        out.append(value)
    return out[0], out[1]


def migrate_encoder_lr_options(config: DictConfig) -> DictConfig:
    """Move deprecated backbone ``encoder_lr_scale`` / ``layer_decay`` keys in place.

    Writes the resolved values (`resolve_encoder_lr_options`, which warns) to
    ``trainer_config.optimizer`` and resets the ``backbone_config.pretrained``
    keys to their defaults, so a saved ``training_config.yaml`` names each knob
    once, in its new location. A no-op for configs without the backbone keys.

    Args:
        config: The full training config (modified in place).

    Returns:
        The same config.
    """
    scale, decay = resolve_encoder_lr_options(config)
    for key, value in (("encoder_lr_scale", scale), ("layer_decay", decay)):
        if value is not None:
            OmegaConf.update(
                config, f"trainer_config.optimizer.{key}", value, force_add=True
            )
        if OmegaConf.select(config, f"{_PRETRAINED_KEY}.{key}", default=None) not in (
            None,
            _ENCODER_LR_KEYS[key],
        ):
            OmegaConf.update(config, f"{_PRETRAINED_KEY}.{key}", _ENCODER_LR_KEYS[key])
    return config
