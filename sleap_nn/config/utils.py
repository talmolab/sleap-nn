"""Utilities for config building and validation."""

import math
import os
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from omegaconf import DictConfig, OmegaConf

#: Upper bound on the worker count picked by ``num_workers: "auto"``.
#:
#: Dataloading here is dominated by decode + cache lookup rather than heavy CPU
#: transforms, so returns flatten quickly, and with in-memory caching each extra
#: worker adds a copy-on-write share of the image cache (see
#: :func:`sleap_nn.data.utils.check_cache_memory`, which warns from 4 workers up).
#: Capping at 4 keeps "auto" safe on a 64-core node rather than optimal on one.
MAX_AUTO_NUM_WORKERS = 4


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
    if model_type == "embedding":
        # Pin the pooled head's stride to the backbone max_stride so the decoder is
        # empty and the head taps `middle_output` (the lone-head tap; SPEC §3.0).
        max_stride = config.model_config.backbone_config[f"{backbone_type}"][
            "max_stride"
        ]
        config.model_config.head_configs.embedding.embedding.output_stride = max_stride
        _check_embedding_backbone(config, backbone_type)
        _resolve_embedding_pool(config, backbone_type)
        if backbone_type == "pretrained":
            # The `pretrained` wrapper spells "no decoder" as `mode="encoder"`, not
            # with strides -- and it treats `output_stride == max_stride` as a user
            # error ("nothing to decode"). Pinning the stride here, as the native
            # backbones need, is therefore exactly what made the DEFAULT
            # `mode: auto` refuse to build for every hierarchical backbone. Say it
            # in the wrapper's own vocabulary instead and leave its stride alone.
            _set_pretrained_encoder_mode(config)
        else:
            config.model_config.backbone_config[f"{backbone_type}"][
                "output_stride"
            ] = max_stride
    return config


def _has_pretrained_backbone_weights(config: OmegaConf, backbone_type: str) -> bool:
    """Whether training starts the backbone from pretrained (not random) weights."""
    if OmegaConf.select(config, "model_config.pretrained_backbone_weights") is not None:
        return True
    backbone_cfg = config.model_config.backbone_config[backbone_type]
    if backbone_type == "pretrained":
        return bool(OmegaConf.select(backbone_cfg, "weights", default=True))
    if backbone_type in ("convnext", "swint"):
        return OmegaConf.select(backbone_cfg, "pre_trained_weights") is not None
    return False


def _check_embedding_backbone(config: OmegaConf, backbone_type: str) -> None:
    """Validate the backbone of an `embedding` model at config time.

    Args:
        config: The full training job config of an `embedding` model.
        backbone_type: Its backbone type.

    Raises:
        ValueError: If a `unet` backbone sets `stem_stride`.
    """
    backbone_cfg = config.model_config.backbone_config[backbone_type]
    if (
        backbone_type == "unet"
        and OmegaConf.select(backbone_cfg, "stem_stride") is not None
    ):
        # Same hazard as the rejected `pretrained.mode='decoder'`: a UNet with a stem
        # builds `log2(stem_stride)` decoder blocks even when output_stride ==
        # max_stride (up_blocks = log2(max/out) + stem_blocks), and the pooled head
        # reads the bottleneck, so that decoder never receives gradient (33,138 of
        # 90,936 parameters on a small UNet) -- dead weight and a DDP
        # unused-parameter error.
        message = (
            "model_config.backbone_config.unet.stem_stride is not supported for the "
            "`embedding` model type: a stem makes the UNet build decoder blocks that "
            "its pooled head never reads, so they would receive no gradient. Unset "
            "stem_stride (the stride the head pools at is max_stride)."
        )
        logger.error(message)
        raise ValueError(message)

    freeze = OmegaConf.select(
        config,
        "model_config.head_configs.embedding.embedding.freeze_backbone",
        default=False,
    )
    if freeze and not _has_pretrained_backbone_weights(config, backbone_type):
        logger.warning(
            "head_configs.embedding.embedding.freeze_backbone=True, but the "
            f"`{backbone_type}` backbone starts from random weights (no "
            "pretrained.weights / pre_trained_weights / "
            "model_config.pretrained_backbone_weights): its encoder will stay at its "
            "random initialization for the whole run. This is almost certainly a "
            "mistake -- load pretrained weights or set freeze_backbone=False."
        )


def _resolve_embedding_pool(config: OmegaConf, backbone_type: str) -> None:
    """Write the default pooling into an `embedding` config that leaves it unset.

    Resolved here, at training setup, so the saved ``training_config.yaml`` records
    the pooling the model was trained with and inference never has to re-derive it.

    Args:
        config: The full training job config of an `embedding` model.
        backbone_type: Its backbone type.
    """
    from sleap_nn.architectures.heads import (
        default_embedding_pool,
        embedding_encoder_is_frozen,
    )

    leaf = config.model_config.head_configs.embedding.embedding
    if OmegaConf.select(leaf, "pool", default=None) is not None:
        return
    frozen = embedding_encoder_is_frozen(
        backbone_type, config.model_config.backbone_config[backbone_type], leaf
    )
    leaf.pool = default_embedding_pool(frozen)
    logger.info(
        f"Setting `head_configs.embedding.embedding.pool` to '{leaf.pool}' "
        + (
            "(frozen pretrained encoder: its LayerNorm output is about half "
            "negative, and GeM pools only the positive part)."
            if frozen
            else "(the default)."
        )
    )


def _set_pretrained_encoder_mode(config: OmegaConf) -> None:
    """Put a `pretrained` backbone in encoder-only mode for a pooled-head model.

    Args:
        config: The full training job config, with `backbone_type == "pretrained"`
            and a lone pooled head (the `embedding` model type).

    Raises:
        ValueError: If the config explicitly asks for `mode: decoder`.
    """
    pretrained_cfg = config.model_config.backbone_config.pretrained
    mode = OmegaConf.select(pretrained_cfg, "mode", default="auto")
    if mode == "decoder":
        # Not a stride problem, so do not let it surface as one. A pooled head
        # reads the bottleneck (`Model.forward` routes it to `intermediate_feat`),
        # so a decoder built underneath it never receives gradient -- measured at
        # 4.9 M of 20.1 M parameters (24%) on convnextv2-nano, which is dead weight
        # and a DDP hazard (unused parameters).
        message = (
            "model_config.backbone_config.pretrained.mode='decoder' is not valid "
            "for the `embedding` model type: its lone pooled head reads the "
            "encoder bottleneck, so the decoder would receive no gradient. Use "
            "mode='encoder' (or 'auto', which resolves to it here)."
        )
        logger.error(message)
        raise ValueError(message)
    if mode != "encoder":
        pretrained_cfg.mode = "encoder"
        logger.info(
            "Setting `model_config.backbone_config.pretrained.mode` to 'encoder' "
            "for the `embedding` model type (its pooled head taps the encoder "
            "bottleneck; there is nothing for a decoder to do)."
        )


def check_centroid_methods(config: OmegaConf) -> OmegaConf:
    """Validate every head config's centroid-method knobs (#586).

    A contradictory pair (``anchor_part`` plus a non-anchor ``centroid_method``)
    or an unknown method name must fail at setup with a message naming the head,
    not deep inside the first ``__getitem__`` of a dataloader worker — where the
    traceback is a multiprocessing wrapper and the run has already spent minutes
    caching images.

    Args:
        config: The full training job config.

    Returns:
        The config, unchanged (validation only).

    Raises:
        ValueError: For any head whose centroid knobs do not resolve.
    """
    from sleap_nn.data.instance_centroids import resolve_centroid_method

    head_configs = OmegaConf.select(config, "model_config.head_configs", default=None)
    if head_configs is None:
        return config
    for head_type, head in head_configs.items():
        if head is None:
            continue
        for leaf_name, leaf in head.items():
            if leaf is None or not OmegaConf.is_config(leaf):
                continue
            if OmegaConf.select(leaf, "centroid_method", default=None) is None and (
                OmegaConf.select(leaf, "centroid_fallback", default=None) is None
            ):
                continue
            try:
                resolve_centroid_method(
                    anchor_part=OmegaConf.select(leaf, "anchor_part", default=None),
                    centroid_method=OmegaConf.select(
                        leaf, "centroid_method", default=None
                    ),
                    centroid_fallback=OmegaConf.select(
                        leaf, "centroid_fallback", default=None
                    ),
                )
            except ValueError as e:
                message = (
                    f"Invalid centroid config in "
                    f"`head_configs.{head_type}.{leaf_name}`: {e}"
                )
                logger.error(message)
                raise ValueError(message) from e
    return config


def usable_cpu_count() -> int:
    """Return the number of CPUs actually usable by this process.

    Prefers the process's CPU affinity mask over the machine's total core count,
    so a run confined to a subset of cores (cgroups/cpusets, SLURM, Docker
    ``--cpuset-cpus``) sizes itself to its allocation instead of to the host.
    Falls back to :func:`os.cpu_count` where affinity is unavailable (macOS,
    Windows).

    Returns:
        The number of usable CPUs (at least 1).
    """
    count = None
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - platform-dependent
        try:
            import psutil

            count = len(psutil.Process().cpu_affinity())
        except (AttributeError, ImportError):  # pragma: no cover
            count = None
    if not count:
        count = os.cpu_count()
    return max(1, count or 1)


def resolve_auto_num_workers(
    data_pipeline_fw: str,
    train_labels: Optional[list] = None,
    val_labels: Optional[list] = None,
    memory_buffer: float = 0.2,
    log: bool = True,
) -> int:
    """Pick a dataloader worker count for ``num_workers: "auto"``.

    Workers are bounded by two independent limits, and ``auto`` takes the
    smaller:

    **CPU.** One core is left for the main/training process and the result is
    capped at :data:`MAX_AUTO_NUM_WORKERS`, since dataloading here is dominated
    by decode and cache lookup rather than heavy CPU transforms.

    **Memory.** Under ``torch_dataset_cache_img_memory`` every worker adds a
    share of the image cache (see
    :func:`sleap_nn.data.utils.worker_memory_overhead_factor` — ~25% of it per
    worker when forking on Linux, ~50% when spawning on macOS/Windows). Workers
    are therefore a *memory multiplier* on exactly the pipeline that is already
    the most memory-hungry, so ``auto`` inverts the same estimate
    ``_setup_datasets`` uses and returns the largest count that still fits in
    available RAM. Without this, ``auto`` could itself push a run past the
    memory check and silently downgrade it to disk caching.

    Two cases short-circuit:

    - ``torch_dataset`` (streaming) always resolves to ``0``: it reads frames
      through the video backend inside the dataset, and those backends (e.g.
      open ``h5py`` handles) cannot be pickled to worker processes. This is the
      constraint documented on ``data_config.data_pipeline_fw``.
    - If the cache does not fit even at zero workers, the run will fall back to
      disk caching anyway, where workers are cheap — so the CPU bound applies
      rather than a pointless ``0``.

    ``torch_dataset_cache_img_disk`` holds no large in-process cache, so it is
    CPU-bound only.

    Args:
        data_pipeline_fw: The configured ``data_config.data_pipeline_fw``.
        train_labels: Training labels, used to size the in-memory cache. When
            omitted, the memory bound is skipped and only the CPU bound applies.
        val_labels: Validation labels, as above.
        memory_buffer: Fraction of memory reserved for training overhead, matching
            the value ``_setup_datasets`` passes to the memory check.
        log: Whether to report when memory rather than CPU is the binding limit.
            Set ``False`` when resolving speculatively (e.g. to suggest a count
            to a user who did not ask for ``"auto"``), so the log does not claim
            an ``"auto"`` that was never requested.

    Returns:
        The resolved worker count.
    """
    if data_pipeline_fw == "torch_dataset":
        return 0

    cpu_bound = max(0, min(MAX_AUTO_NUM_WORKERS, usable_cpu_count() - 1))

    if (
        data_pipeline_fw != "torch_dataset_cache_img_memory"
        or not train_labels
        or cpu_bound == 0
    ):
        return cpu_bound

    # Imported here rather than at module scope: `sleap_nn.data.utils` imports
    # from this module, so a top-level import would be circular.
    from sleap_nn.data.utils import (
        estimate_cache_memory,
        worker_memory_overhead_factor,
    )

    estimate = estimate_cache_memory(
        train_labels=train_labels,
        val_labels=val_labels if val_labels is not None else [],
        num_workers=0,
        memory_buffer=memory_buffer,
    )

    raw_cache_bytes = estimate["raw_cache_bytes"]
    if raw_cache_bytes <= 0:
        return cpu_bound

    # `estimate_cache_memory` computes
    #   total(w) = (raw + python_overhead + raw * factor * w) * (1 + memory_buffer)
    # which is linear in `w`, so solve `total(w) <= available` directly instead of
    # re-estimating per candidate (each estimate walks every labeled frame).
    headroom = (
        estimate["available_bytes"] / (1.0 + memory_buffer)
        - raw_cache_bytes
        - estimate["python_overhead_bytes"]
    )
    if headroom < 0:
        # Does not fit even with zero workers: the run downgrades to disk
        # caching in `_setup_datasets`, where workers are cheap again.
        return cpu_bound

    memory_bound = int(headroom // (raw_cache_bytes * worker_memory_overhead_factor()))

    resolved = max(0, min(cpu_bound, memory_bound))
    if log and resolved < cpu_bound:
        logger.info(
            f"`num_workers: auto` limited to {resolved} worker(s) by available "
            f"memory (CPU alone would allow {cpu_bound}): in-memory cache is "
            f"{raw_cache_bytes / (1024**3):.2f} GB and each worker adds ~"
            f"{worker_memory_overhead_factor() * 100:.0f}% of it on "
            f"{sys.platform}."
        )
    return resolved


def check_num_workers(
    config: OmegaConf,
    train_labels: Optional[list] = None,
    val_labels: Optional[list] = None,
    memory_buffer: float = 0.2,
) -> OmegaConf:
    """Resolve ``num_workers: "auto"`` on the train/val dataloader configs.

    Resolution happens once, up front, and the resolved integer is written back
    into the config so that every downstream consumer (dataloader construction,
    cache-memory estimation) sees a plain ``int``, and the saved
    ``training_config.yaml`` records the worker count the run actually used.
    ``initial_config.yaml`` keeps the literal ``"auto"``, so a config reused on
    another machine re-resolves there.

    Args:
        config: The full training job config.
        train_labels: Training labels. Passing them lets ``auto`` apply the
            memory bound described in :func:`resolve_auto_num_workers`; without
            them only the CPU bound applies.
        val_labels: Validation labels, as above.
        memory_buffer: Fraction of memory reserved for training overhead.

    Also emits a one-line hint when a caching pipeline is configured with fewer
    workers than this machine could actually support — see
    :func:`suggest_num_workers`.

    Returns:
        The config with any ``"auto"`` worker counts replaced by integers.
    """
    data_pipeline_fw = OmegaConf.select(
        config, "data_config.data_pipeline_fw", default="torch_dataset"
    )

    requested_auto, explicit = [], {}
    for loader in ("train_data_loader", "val_data_loader"):
        num_workers = OmegaConf.select(
            config, f"trainer_config.{loader}.num_workers", default=0
        )
        if isinstance(num_workers, str) and num_workers.lower() == "auto":
            requested_auto.append(loader)
        else:
            explicit[loader] = num_workers

    resolved = None

    def _resolve():
        """Resolve once; the memory estimate is the expensive part."""
        nonlocal resolved
        if resolved is None:
            resolved = resolve_auto_num_workers(
                data_pipeline_fw,
                train_labels=train_labels,
                val_labels=val_labels,
                memory_buffer=memory_buffer,
                # Only narrate the memory bound if someone actually asked for
                # `auto`; otherwise this call is speculative, for the hint below.
                log=bool(requested_auto),
            )
        return resolved

    for loader in requested_auto:
        config.trainer_config[loader].num_workers = _resolve()
        logger.info(
            f"`trainer_config.{loader}.num_workers` set to `auto`: using "
            f"{resolved} worker(s) "
            f"({usable_cpu_count()} usable CPU(s), "
            f"data_pipeline_fw=`{data_pipeline_fw}`)."
        )

    suggest_num_workers(data_pipeline_fw, explicit, _resolve)

    return config


def suggest_num_workers(data_pipeline_fw: str, explicit: dict, resolve) -> None:
    """Hint that a cached run is leaving dataloading throughput on the table.

    Caching (memory or disk) is what makes ``num_workers > 0`` usable at all —
    the streaming pipeline cannot fork its video backends — so a cached run left
    at the default of 0 workers is decoding batches serially in the training
    process for no reason. This nudges the user toward the worker count their
    machine can actually support.

    The suggested count is whatever ``num_workers: "auto"`` would pick here, not
    a fixed "use 2-4": on the in-memory pipeline that number is already bounded
    by the dataset's cache size and free RAM, so the hint can never talk a user
    into a setting that would push their run past the memory check and downgrade
    it to disk caching.

    Silent when the pipeline is streaming, when every loader is already at or
    above the suggestion, or when nothing could be suggested (``auto`` resolves
    to 0 on this machine).

    Args:
        data_pipeline_fw: The configured ``data_config.data_pipeline_fw``.
        explicit: Mapping of loader name to its explicitly-configured worker
            count, i.e. the loaders that did *not* ask for ``"auto"``.
        resolve: Zero-arg callable returning the resolved ``"auto"`` count.
            Deferred so the memory estimate is skipped when no hint is possible.
    """
    if data_pipeline_fw == "torch_dataset":
        return

    # Only loaders with room to grow are worth estimating for.
    candidates = {
        loader: n
        for loader, n in explicit.items()
        if isinstance(n, int) and n < MAX_AUTO_NUM_WORKERS
    }
    if not candidates:
        return

    suggested = resolve()
    below = {loader: n for loader, n in candidates.items() if n < suggested}
    if not below:
        return

    fields = ", ".join(
        f"`trainer_config.{loader}.num_workers` ({n})" for loader, n in below.items()
    )
    logger.info(
        f"Data caching is enabled (`data_pipeline_fw={data_pipeline_fw}`), which "
        f"supports parallel data loading, but {fields} "
        f"{'is' if len(below) == 1 else 'are'} below what this machine can "
        f"support. Consider setting {'it' if len(below) == 1 else 'them'} to "
        f"{suggested} — or to `auto`, which picks this for you — to speed up "
        f"training."
    )


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
    _TILING_SUPPORTED_MODEL_TYPES = {
        "single_instance",
        "bottomup_segmentation",
        "semantic_segmentation",
    }
    if model_type not in _TILING_SUPPORTED_MODEL_TYPES:
        message = (
            f"tiling is not yet implemented for model_type={model_type} "
            "(supported: single_instance, bottomup_segmentation, "
            "semantic_segmentation)"
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
