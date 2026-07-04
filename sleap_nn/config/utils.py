"""Utilities for config building and validation."""

import math
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def resolve_model_dir(model_path: Union[str, Path]) -> str:
    """Resolve a user-supplied model path to its model *directory*.

    A trained model lives in a directory holding ``training_config.{yaml,json}``
    and a ``best.ckpt`` checkpoint. Callers historically had to pass that
    directory. This helper additionally accepts a path to a file *inside* it —
    either the ``training_config.{yaml,json}`` config or a ``.ckpt`` checkpoint —
    and returns the containing directory, so users can point at ``best.ckpt`` or
    ``training_config.yaml`` wherever a model directory is expected (issue #575).

    The directory's *contents* are intentionally NOT validated here: the caller's
    loader (e.g. :func:`sleap_nn.inference.loaders._load_training_config`) remains
    the single source of truth for whether the resolved directory holds a usable
    config and checkpoint, so its error messages stay attributable.

    A directory is always loaded via its ``best.ckpt``. If the path points at a
    *different* checkpoint (e.g. ``last.ckpt``), a warning is emitted and
    ``best.ckpt`` is loaded anyway — use ``backbone_ckpt_path`` / ``head_ckpt_path``
    to load a specific checkpoint.

    Args:
        model_path: A model directory, or a path to a ``.ckpt`` checkpoint or a
            ``training_config.{yaml,json,yml}`` file within one.

    Returns:
        The resolved model directory as a POSIX-style string. Relative paths are
        preserved (only the path separators are normalized); the path is not
        resolved against the filesystem root.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist, or is a file that is
            neither a config file nor a ``.ckpt`` checkpoint.
    """
    p = Path(model_path)
    if p.is_dir():
        # Backward-compatible fast path. Existence/contents of the config are
        # validated downstream so the original directory behavior is unchanged.
        return p.as_posix()
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml", ".json"):
            return p.parent.as_posix()
        if suffix == ".ckpt":
            if p.name.lower() != "best.ckpt":
                logger.warning(
                    f"Model path '{model_path}' points at a specific checkpoint, "
                    f"but inference always loads 'best.ckpt' from the model "
                    f"directory; '{p.name}' will be ignored. To load a different "
                    f"checkpoint, use 'backbone_ckpt_path' / 'head_ckpt_path'."
                )
            return p.parent.as_posix()
        raise FileNotFoundError(
            f"Model path '{model_path}' is not a recognized model file. Pass a "
            f"model directory, or a path to its 'best.ckpt' or "
            f"'training_config.yaml'/'training_config.json' file."
        )
    raise FileNotFoundError(
        f"Model path does not exist: {model_path}. Pass a model directory, or a "
        f"path to its 'best.ckpt' or 'training_config.yaml'/'training_config.json' "
        f"file."
    )


def get_model_type_from_cfg(config: DictConfig):
    """Return the model type from the config. One of [single_instance, centroid, centered_instance, bottomup]."""
    model_type = None
    for k, v in config.model_config.head_configs.items():
        if v is not None:
            model_type = k
            break
    return model_type


def get_backbone_type_from_cfg(config: DictConfig):
    """Return the backbone type from the config. One of [unet, swint, convnext]."""
    backbone_type = None
    for k, v in config.model_config.backbone_config.items():
        if v is not None:
            backbone_type = k
            break
    return backbone_type


def get_output_strides_from_heads(head_configs: DictConfig):
    """Get list of output strides from head configs."""
    output_strides_from_heads = []
    for head_type in head_configs:
        if head_configs[head_type] is not None:
            for head_layer in head_configs[head_type]:
                output_strides_from_heads.append(
                    head_configs[head_type][head_layer]["output_stride"]
                )
    return output_strides_from_heads


def check_output_strides(config: OmegaConf) -> OmegaConf:
    """Check max_stride and output_stride in backbone_config with head_config."""
    output_strides = get_output_strides_from_heads(config.model_config.head_configs)
    backbone_type = get_backbone_type_from_cfg(config)
    if output_strides:
        config.model_config.backbone_config[f"{backbone_type}"]["output_stride"] = min(
            output_strides
        )
        if config.model_config.backbone_config[f"{backbone_type}"]["max_stride"] < max(
            output_strides
        ):
            config.model_config.backbone_config[f"{backbone_type}"]["max_stride"] = max(
                output_strides
            )

    model_type = get_model_type_from_cfg(config)
    if model_type == "multi_class_topdown":
        config.model_config.head_configs.multi_class_topdown.class_vectors.output_stride = config.model_config.backbone_config[
            f"{backbone_type}"
        ][
            "max_stride"
        ]
    return config


def check_tiling(config: OmegaConf) -> OmegaConf:
    """Validate + reconcile tiling geometry against the finalized backbone/head.

    No-op unless ``data_config.preprocessing.tiling.enabled`` is ``True``.
    Must run *after* :func:`check_output_strides` so ``max_stride`` /
    ``output_stride`` are finalized, and after ``_setup_tiling_config`` has
    auto-sized ``tile_size`` / ``overlap`` from the labels. Enforces:

      - GUARD: pretrained-encoder / non-(unet|convnext|swint) backbone -> ValueError.
      - GUARD: ``multi_class_topdown`` / ClassVectorsHead -> ValueError.
      - ``tile_size`` divisible by ``lcm(max_stride, output_stride)`` (rounds UP + warns).
      - ``overlap`` divisible by ``output_stride``, ``>= min_overlap_fraction * tile_size``
        (raises overlap + warns), and ``0 <= overlap < tile_size`` (else ValueError).

    Guards are enforced explicitly here (not via attrs validators) because
    ``_setup_tiling_config`` mutates the config in place on the OmegaConf object,
    which does not re-run attrs validators.

    Args:
        config: The (finalized) training/inference config.

    Returns:
        The config, mutated in place with reconciled tiling geometry.
    """
    tiling = OmegaConf.select(config, "data_config.preprocessing.tiling")
    if tiling is None or not tiling.enabled:
        return config

    backbone_type = get_backbone_type_from_cfg(config)

    # GUARD 1: pretrained-encoder / unsupported backbone. A HuggingFace pretrained
    # encoder surfaces as backbone_type == "pretrained" (BatchNorm-bearing; DQ14),
    # so this also excludes it. A unet/convnext/swint that merely loaded pretrained
    # *weights* is seam-safe and intentionally NOT excluded.
    if backbone_type not in ("unet", "convnext", "swint"):
        message = (
            "data_config.preprocessing.tiling.enabled=True is not supported with "
            f"pretrained or non-UNet-family backbones (backbone={backbone_type!r}). "
            "Disable tiling or train a unet/convnext/swint backbone."
        )
        logger.error(message)
        raise ValueError(message)

    # GUARD 2: class-vector heads (global pool needs whole-instance context).
    head_configs = config.model_config.head_configs
    model_type = get_model_type_from_cfg(config)
    has_class_vectors = any(
        head_configs[h] is not None and "class_vectors" in head_configs[h]
        for h in head_configs
    )
    if model_type == "multi_class_topdown" or has_class_vectors:
        message = (
            "data_config.preprocessing.tiling.enabled=True is not supported for "
            "ClassVectorsHead / multi_class_topdown models (global pooling needs "
            "whole-instance context that per-tile stitching cannot recover)."
        )
        logger.error(message)
        raise ValueError(message)

    # GUARD 3: supported-model-types allowlist. Tiled training is only implemented
    # for these model types; enabling tiling elsewhere would otherwise silently
    # no-op (the dataset factory falls back to the whole-frame branch), so fail loud.
    _TILING_SUPPORTED_MODEL_TYPES = {"single_instance", "bottomup_segmentation"}
    if model_type not in _TILING_SUPPORTED_MODEL_TYPES:
        message = (
            f"tiling is not yet implemented for model_type={model_type} "
            "(supported: single_instance, bottomup_segmentation)"
        )
        logger.error(message)
        raise ValueError(message)

    max_stride = int(
        config.model_config.backbone_config[f"{backbone_type}"]["max_stride"]
    )
    output_strides = get_output_strides_from_heads(head_configs)
    output_stride = min(output_strides) if output_strides else 1
    divisor = math.lcm(max_stride, output_stride)

    # tile_size divisibility (auto-round up).
    tile_size = tiling.tile_size
    if tile_size is None:
        message = (
            "tiling.enabled=True but tile_size is unset in check_tiling; "
            "_setup_tiling_config must run first to auto-size it."
        )
        logger.error(message)
        raise ValueError(message)
    if tile_size % divisor != 0:
        snapped = math.ceil(tile_size / divisor) * divisor
        logger.warning(
            f"tiling.tile_size={tile_size} is not divisible by "
            f"lcm(max_stride={max_stride}, output_stride={output_stride})={divisor}; "
            f"rounding up to {snapped}."
        )
        config.data_config.preprocessing.tiling.tile_size = snapped
        tile_size = snapped

    # overlap: output_stride divisibility + min_overlap_fraction floor + range.
    overlap = tiling.overlap
    if overlap is None:
        message = (
            "tiling.enabled=True but overlap is unset in check_tiling; "
            "_setup_tiling_config must run first."
        )
        logger.error(message)
        raise ValueError(message)
    if overlap % output_stride != 0:
        snapped = math.ceil(overlap / output_stride) * output_stride
        logger.warning(
            f"tiling.overlap={overlap} not divisible by output_stride={output_stride}; "
            f"rounding up to {snapped}."
        )
        overlap = snapped
    frac_floor = (
        math.ceil(tiling.min_overlap_fraction * tile_size / output_stride)
        * output_stride
    )
    if overlap < frac_floor:
        logger.warning(
            f"tiling.overlap={overlap} is below the min_overlap_fraction "
            f"({tiling.min_overlap_fraction}) floor of {frac_floor}px; raising to {frac_floor}."
        )
        overlap = frac_floor
    if not (0 <= overlap < tile_size):
        message = (
            f"tiling.overlap={overlap} must satisfy 0 <= overlap < tile_size={tile_size} "
            "(a tile must have a positive stride)."
        )
        logger.error(message)
        raise ValueError(message)
    config.data_config.preprocessing.tiling.overlap = overlap
    return config


def check_tiling_parity(
    config: OmegaConf,
    tile_size_override: Optional[int] = None,
    overlap_override: Optional[int] = None,
) -> OmegaConf:
    """Re-check inference tiling geometry against the trained model config.

    Tiling requires train-time == infer-time geometry (scale parity); the trained
    geometry lives in the model config. A user-supplied geometry override that
    diverges from the trained values requires a retrain, so this raises on a
    mismatch. No-op unless tiling is enabled.

    Args:
        config: The loaded (trained) model config.
        tile_size_override: Optional inference-time ``tile_size`` override.
        overlap_override: Optional inference-time ``overlap`` override.

    Returns:
        The config unchanged (parity check only).
    """
    tiling = OmegaConf.select(config, "data_config.preprocessing.tiling")
    if tiling is None or not tiling.enabled:
        return config
    if tile_size_override is not None and tile_size_override != tiling.tile_size:
        message = (
            f"tile_size override ({tile_size_override}) does not match the trained "
            f"tiling geometry (tile_size={tiling.tile_size}). Tiling geometry is fixed "
            "at train time (scale parity) — retrain to change it."
        )
        logger.error(message)
        raise ValueError(message)
    if overlap_override is not None and overlap_override != tiling.overlap:
        message = (
            f"overlap override ({overlap_override}) does not match the trained tiling "
            f"geometry (overlap={tiling.overlap}). Tiling geometry is fixed at train "
            "time (scale parity) — retrain to change it."
        )
        logger.error(message)
        raise ValueError(message)
    return config


def oneof(attrs_cls, must_be_set: bool = False):
    """Ensure that the decorated attrs class only has a single attribute set.

    This decorator is inspired by the `oneof` protobuffer field behavior.

    Args:
        attrs_cls: An attrs decorated class.
        must_be_set: If True, raise an error if none of the attributes are set. If not,
            error will only be raised if more than one attribute is set.

    Returns:
        The `attrs_cls` with an `__init__` method that checks for the number of
        attributes that are set.
    """
    # Check if the class is an attrs class at all.
    if not hasattr(attrs_cls, "__attrs_attrs__"):
        message = "Classes decorated with oneof must also be attr.s decorated."
        logger.error(message)
        raise ValueError(message)

    # Pull out attrs generated class attributes.
    attribs = attrs_cls.__attrs_attrs__
    init_fn = attrs_cls.__init__

    # Define a new __init__ function that wraps the attrs generated one.
    def new_init_fn(self, *args, **kwargs):
        # Execute the standard attrs-generated __init__.
        init_fn(self, *args, **kwargs)

        # Check for attribs with set values.
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]

        class_name = self.__class__.__name__

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            message = (
                f"{class_name}: Only one attribute of this class can be set (not None)."
            )
            logger.error(message)
            raise ValueError(message)

        if len(attribs_with_value) == 0 and must_be_set:
            # Raise error if none are set.
            message = f"{class_name}: At least one attribute of this class must be set."
            logger.error(message)
            raise ValueError(message)

    # Replace with wrapped __init__.
    attrs_cls.__init__ = new_init_fn

    # Define convenience method for getting the set attribute.
    def which_oneof_attrib_name(self):
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]
        class_name = self.__class__.__name__

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            message = (
                f"{class_name}: Only one attribute of this class can be set (not None)."
            )
            logger.error(message)
            raise ValueError(message)

        if len(attribs_with_value) == 0:
            if must_be_set:
                # Raise error if none are set.
                message = (
                    f"{class_name}: At least one attribute of this class must be set."
                )
                logger.error(message)
                raise ValueError(message)
            else:
                return None

        return attribs_with_value[0].name

    def which_oneof(self):
        attrib_name = self.which_oneof_attrib_name()

        if attrib_name is None:
            return None

        return getattr(self, attrib_name)

    attrs_cls.which_oneof_attrib_name = which_oneof_attrib_name
    attrs_cls.which_oneof = which_oneof

    return attrs_cls
