"""Serializable configuration classes for specifying all trainer config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the trainer config.
"""

from attrs import define, field, validators
from pathlib import Path
import time
from typing import Optional, List, Any
from loguru import logger
import re

#: Lightning ``Trainer(precision=...)`` values accepted by ``trainer_config.precision``.
TRAINER_PRECISIONS = {"32", "32-true", "bf16-mixed", "16-mixed"}

#: ``torch.compile`` modes accepted by ``trainer_config.compile`` (``True`` = ``"default"``).
COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


@define
class DataLoaderConfig:
    """Train DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. This is the per-GPU batch size; with multi-GPU (DDP) training the effective (global) batch size is `batch_size × num_GPUs`. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `False`.
        num_workers: (int or "auto") Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Set to `"auto"` to size the pool from the CPUs usable by this process (capped, and leaving one core for the main process); `"auto"` resolves to `0` for the `torch_dataset` (streaming) pipeline, whose video backends cannot be pickled to worker processes. The resolved integer is written back into the saved `training_config.yaml`. *Default*: `0`.
        prefetch_factor: (int) Number of batches each worker loads ahead (ignored when `num_workers` is 0), so at most `prefetch_factor x num_workers` batches are buffered per loader. `None` = `2` (the PyTorch default). Before 0.4 this was tied to `batch_size`, which buffered `batch_size x num_workers` batches and could exhaust host/shared memory with large batches. *Default*: `2`.
        pin_memory: (bool) Use pinned (page-locked) host memory for batches. `None` = `True` for the train loader; the val loader falls back to the train loader's setting. *Default*: `None`.
    """

    batch_size: int = 4
    shuffle: bool = False
    num_workers: Any = field(
        default=0,
        validator=lambda inst, attr, val: DataLoaderConfig.validate_num_workers(val),
    )
    prefetch_factor: Optional[int] = field(
        default=2,
        validator=validators.optional(validators.ge(1)),
    )
    pin_memory: Optional[bool] = None

    @staticmethod
    def validate_num_workers(value):
        """Validate the value of num_workers."""
        if isinstance(value, bool):
            pass
        elif isinstance(value, int) and value >= 0:
            return
        elif isinstance(value, str) and value.lower() == "auto":
            return
        message = 'num_workers must be an integer >= 0, or the string "auto".'
        logger.error(message)
        raise ValueError(message)


@define
class TrainDataLoaderConfig(DataLoaderConfig):
    """Train DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. This is the per-GPU batch size; with multi-GPU (DDP) training the effective (global) batch size is `batch_size × num_GPUs`. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `True`.
        num_workers: (int or "auto") Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Set to `"auto"` to size the pool from the CPUs usable by this process. *Default*: `0`.
        drop_last: (bool) Drop the partial batch at the end of each pass over the training data. `None` = `True` only when `trainer_config.compile` is enabled (a new batch size triggers a recompile), else `False`. Never applied when the data is smaller than one batch. *Default*: `None`.
    """

    shuffle: bool = True
    drop_last: Optional[bool] = None


@define
class ValDataLoaderConfig(DataLoaderConfig):
    """Validation DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. This is the per-GPU batch size; with multi-GPU (DDP) training the effective (global) batch size is `batch_size × num_GPUs`. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `False`.
        num_workers: (int or "auto") Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Set to `"auto"` to size the pool from the CPUs usable by this process. *Default*: `0`.
    """

    shuffle: bool = False


@define
class ModelCkptConfig:
    """Configuration for model checkpoint.

    Any parameters from Lightning's ModelCheckpoint could be used.

    Attributes:
        save_top_k: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False. *Default*: `1`.
        save_last: (bool) When True, also writes a last.ckpt at the end of every epoch (overwritten each time), so the most recent epoch can always be resumed from, whether or not it improved the monitored metric. *Default*: `None`.
        monitor: (str) Metric name the checkpoint tracks to pick the "best" model. Any key present in ``trainer.callback_metrics`` — e.g. ``"val/loss"`` (default), or, for segmentation runs with ``eval.enabled``, a full-resolution quality metric such as ``"eval/val/fg_mean_cldice"`` (semantic) or ``"eval/val/mask_mean_iou"`` (instance) logged by ``SegmentationEvaluationCallback``. *Default*: `"val/loss"`.
        mode: (str) ``"min"`` (default) or ``"max"`` — direction of improvement for ``monitor`` (use ``"max"`` for IoU/clDice metrics). *Default*: `"min"`.
    """

    save_top_k: int = 1
    save_last: Optional[bool] = None
    monitor: str = "val/loss"
    mode: str = "min"


@define
class WandBConfig:
    """Configuration for WandB.

    Only if use_wandb is True, else skip this

    Attributes:
        entity: (str) Entity of wandb project. *Default*: `None`.
        project: (str) Project name for the wandb project. *Default*: `None`.
        name: (str) Name of the current run. *Default*: `None`.
        save_viz_imgs_wandb: (bool) If set to `True`, sample predictions (keypoints + confidence maps) that are saved to local `viz` folder in the ckpt dir would also be uploaded to wandb. *Default*: `False`.
        api_key: (str) API key. The API key is masked when saved to config files. *Default*: `None`.
        wandb_mode: (str) "offline" if only local logging is required. *Default*: `"None"`.
        prv_runid: (str) Previous run ID if training should be resumed from a previous ckpt. *Default*: `None`.
        group: (str) Group for wandb logging. *Default*: `None`.
        current_run_id: (str) Run ID for the current model training. (stored once the training starts). *Default*: `None`.
        viz_enabled: (bool) If True, log pre-rendered matplotlib images to wandb. *Default*: `True`.
        viz_boxes: (bool) If True, log interactive keypoint boxes. *Default*: `False`.
        viz_masks: (bool) If True, log confidence map overlay masks. *Default*: `False`.
        viz_box_size: (float) Size of keypoint boxes in pixels (for viz_boxes). *Default*: `5.0`.
        viz_confmap_threshold: (float) Threshold for confidence map masks (for viz_masks). *Default*: `0.1`.
        log_viz_table: (bool) If True, also log images to a wandb.Table for backwards compatibility. *Default*: `False`.
        delete_local_logs: (bool, optional) If True, delete local wandb logs folder after
            training. If False, keep the folder. If None (default), automatically delete
            if logging online (wandb_mode != "offline") and keep if logging offline.
            *Default*: `None`.
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    save_viz_imgs_wandb: bool = False
    api_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    prv_runid: Optional[str] = None
    group: Optional[str] = None
    current_run_id: Optional[str] = None
    viz_enabled: bool = True
    viz_boxes: bool = False
    viz_masks: bool = False
    viz_box_size: float = 5.0
    viz_confmap_threshold: float = 0.1
    log_viz_table: bool = False
    delete_local_logs: Optional[bool] = None


@define
class OptimizerConfig:
    """Configuration for optimizer.

    Attributes:
        lr: (float) Learning rate of type float. *Default*: `1e-4`.
        amsgrad: (bool) Enable AMSGrad with the optimizer. *Default*: `False`.
        weight_decay: (float) Weight decay applied to the weight matrices/kernels
            only. When set, the optimizer's parameters are split into a decayed
            group (>= 2-D weights) and a no-decay group (biases, norm weights, any
            other 1-D parameter, and position embeddings / CLS / register / mask
            tokens, including a timm model's ``no_weight_decay()`` names); each
            encoder learning-rate group (``encoder_lr_scale`` / ``layer_decay``)
            is split the same way and keeps its learning rate. ``None`` keeps the
            torch default for the optimizer applied to EVERY parameter (``0`` for
            Adam, ``0.01`` for AdamW). *Default*: `None`.
        encoder_lr_scale: (float) Learning-rate multiplier for a ``pretrained``
            backbone's encoder; the neck, decoder and heads train at ``lr``. Has no
            effect on other backbones. ``None`` = ``1.0``. Replaces
            ``model_config.backbone_config.pretrained.encoder_lr_scale``, which is
            still read (with a deprecation warning) when this is unset.
            *Default*: `None`.
        layer_decay: (float) Layer-wise learning-rate decay for a plain-ViT
            ``pretrained`` encoder: block ``i`` of ``L`` trains at
            ``encoder_lr_scale * layer_decay**(L + 1 - i)`` times ``lr``
            (embeddings decay the most). ``None`` disables it. Replaces
            ``model_config.backbone_config.pretrained.layer_decay``, which is still
            read (with a deprecation warning) when this is unset. *Default*: `None`.
    """

    lr: float = field(default=1e-4, validator=validators.gt(0))
    amsgrad: bool = False
    weight_decay: Optional[float] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    encoder_lr_scale: Optional[float] = field(
        default=None, validator=validators.optional(validators.gt(0))
    )
    layer_decay: Optional[float] = field(
        default=None,
        validator=validators.optional([validators.gt(0), validators.le(1)]),
    )


@define
class StepLRConfig:
    """Configuration for StepLR scheduler.

    Attributes:
        step_size: (int) Period of learning rate decay. If step_size=10, then every 10 epochs, learning rate will be reduced by a factor of gamma. *Default*: `10`.
        gamma: (float) Multiplicative factor of learning rate decay. *Default*: `0.1`.
    """

    step_size: int = field(default=10, validator=validators.gt(0))
    gamma: float = 0.1


@define
class ReduceLROnPlateauConfig:
    """Configuration for ReduceLROnPlateau scheduler.

    Attributes:
        threshold: (float) Threshold for measuring the new optimum, to only focus on significant changes. *Default*: `1e-6`.
        threshold_mode: (str) One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. *Default*: `"abs"`.
        cooldown: (int) Number of epochs to wait before resuming normal operation after lr has been reduced. *Default*: `3`.
        patience: (int) Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn't improved then. *Default*: `5`.
        factor: (float) Factor by which the learning rate will be reduced. new_lr = lr * factor. *Default*: `0.5`.
        min_lr: (float or List[float]) A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. *Default*: `1e-8`.
    """

    threshold: float = 1e-6
    threshold_mode: str = "abs"
    cooldown: int = 3
    patience: int = 5
    factor: float = 0.5
    min_lr: Any = field(
        default=1e-8, validator=lambda instance, attr, value: instance.validate_min_lr()
    )

    def validate_min_lr(self):
        """min_lr Validation.

        Ensures min_lr is a float>=0 or list of floats>=0
        """
        if isinstance(self.min_lr, float) and self.min_lr >= 0:
            return
        if isinstance(self.min_lr, list) and all(
            isinstance(x, float) and x >= 0 for x in self.min_lr
        ):
            return
        message = "min_lr must be a float or a list of floats."
        logger.error(message)
        raise ValueError(message)


@define
class CosineAnnealingWarmupConfig:
    """Configuration for Cosine Annealing with Linear Warmup scheduler.

    The learning rate increases linearly during warmup, then decreases following
    a cosine curve to the minimum value.

    Attributes:
        warmup_epochs: (int) Number of epochs for linear warmup phase. *Default*: `5`.
        max_epochs: (int) Total number of training epochs. Will be overridden by
            trainer's max_epochs if not specified. *Default*: `None`.
        warmup_start_lr: (float) Learning rate at start of warmup. *Default*: `0.0`.
        eta_min: (float) Minimum learning rate at end of cosine decay. *Default*: `0.0`.
        warmup_steps: (int) Warmup length in optimizer steps (iterations). Setting
            it (``0`` included) switches the schedule to per-step updates: the
            learning rate changes every optimizer step instead of every epoch,
            ``warmup_epochs`` is ignored, and the cosine runs over ``max_steps``.
            ``None`` keeps the per-epoch schedule, whose warmup holds the LR at
            ``warmup_start_lr`` (``0`` by default) for the whole first epoch.
            *Default*: `None`.
        max_steps: (int) Total schedule length in optimizer steps when
            ``warmup_steps`` is set. ``None`` uses this schedule's ``max_epochs``
            times the optimizer steps per epoch when that is set, else the run's
            total optimizer steps (``trainer_config.max_epochs``, capped by
            ``trainer_config.max_steps``). *Default*: `None`.
    """

    warmup_epochs: int = field(default=5, validator=validators.ge(0))
    max_epochs: Optional[int] = None
    warmup_start_lr: float = field(default=0.0, validator=validators.ge(0))
    eta_min: float = field(default=0.0, validator=validators.ge(0))
    warmup_steps: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    max_steps: Optional[int] = field(
        default=None, validator=validators.optional(validators.gt(0))
    )


@define
class LinearWarmupLinearDecayConfig:
    """Configuration for Linear Warmup + Linear Decay scheduler.

    The learning rate increases linearly during warmup, then decreases linearly
    to the end learning rate.

    Attributes:
        warmup_epochs: (int) Number of epochs for linear warmup phase. *Default*: `5`.
        max_epochs: (int) Total number of training epochs. Will be overridden by
            trainer's max_epochs if not specified. *Default*: `None`.
        warmup_start_lr: (float) Learning rate at start of warmup. *Default*: `0.0`.
        end_lr: (float) Learning rate at end of training. *Default*: `0.0`.
        warmup_steps: (int) Warmup length in optimizer steps; setting it switches
            the schedule to per-step updates (see
            ``CosineAnnealingWarmupConfig.warmup_steps``). *Default*: `None`.
        max_steps: (int) Total schedule length in optimizer steps when
            ``warmup_steps`` is set. ``None`` uses this schedule's ``max_epochs``
            times the optimizer steps per epoch when that is set, else the run's
            total optimizer steps (``trainer_config.max_epochs``, capped by
            ``trainer_config.max_steps``). *Default*: `None`.
    """

    warmup_epochs: int = field(default=5, validator=validators.ge(0))
    max_epochs: Optional[int] = None
    warmup_start_lr: float = field(default=0.0, validator=validators.ge(0))
    end_lr: float = field(default=0.0, validator=validators.ge(0))
    warmup_steps: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    max_steps: Optional[int] = field(
        default=None, validator=validators.optional(validators.gt(0))
    )


@define
class MultiStepLRConfig:
    """Configuration for the step-decay schedule with an optional linear warmup.

    The ViTPose / mmpose recipe: a linear warmup over ``warmup_steps`` optimizer
    steps from ``warmup_start_factor`` x the learning rate, then the learning rate
    is multiplied by ``gamma`` at every milestone. The schedule is always updated
    per optimizer step; epoch milestones are converted with the number of
    optimizer steps per epoch, so a milestone of ``170`` takes effect on the first
    step of epoch ``170`` (0-indexed), as in mmcv's ``step`` policy. Each param
    group keeps its own (encoder_lr_scale / layer_decay) learning rate ratio.

    Attributes:
        milestones: (List[int]) Increasing epochs (or steps, per
            ``milestone_unit``) at which the learning rate is multiplied by
            ``gamma``. Required. *Default*: `None`.
        gamma: (float) Multiplicative factor at each milestone. *Default*: `0.1`.
        milestone_unit: (str) ``"epoch"`` or ``"step"``. *Default*: `"epoch"`.
        warmup_steps: (int) Linear warmup length in optimizer steps. ``0``
            disables warmup. *Default*: `0`.
        warmup_start_factor: (float) Fraction of the learning rate the warmup
            starts from (mmcv's ``warmup_ratio``). *Default*: `0.001`.
    """

    milestones: Optional[List[int]] = field(
        default=None,
        validator=lambda inst, attr, val: MultiStepLRConfig.validate_milestones(val),
    )
    gamma: float = field(default=0.1, validator=validators.gt(0))
    milestone_unit: str = field(
        default="epoch", validator=validators.in_(["epoch", "step"])
    )
    warmup_steps: int = field(default=0, validator=validators.ge(0))
    warmup_start_factor: float = field(
        default=0.001, validator=[validators.gt(0), validators.le(1)]
    )

    @staticmethod
    def validate_milestones(value):
        """Validate that milestones are strictly increasing non-negative ints."""
        if value is None:
            return
        value = list(value)
        if all(isinstance(m, int) and not isinstance(m, bool) for m in value) and (
            all(m >= 0 for m in value) and all(b > a for a, b in zip(value, value[1:]))
        ):
            return
        message = (
            "multi_step_lr.milestones must be strictly increasing non-negative "
            f"integers, got {value!r}."
        )
        logger.error(message)
        raise ValueError(message)


@define
class LRSchedulerConfig:
    """Configuration for lr_scheduler.

    Only one scheduler is used. If multiple are set, priority order is:
    cosine_annealing_warmup > linear_warmup_linear_decay > multi_step_lr >
    step_lr > reduce_lr_on_plateau. ``reduce_lr_on_plateau`` is populated by
    default, so setting any other scheduler replaces it (they never stack).
    Choosing a fixed-length schedule (``cosine_annealing_warmup``,
    ``linear_warmup_linear_decay`` or ``multi_step_lr``) also turns early stopping
    off unless ``early_stopping.stop_training_on_plateau`` is set explicitly, and
    training saves the config with ``reduce_lr_on_plateau: null``.

    Attributes:
        step_lr: Configuration for StepLR scheduler.
        reduce_lr_on_plateau: Configuration for ReduceLROnPlateau scheduler.
        cosine_annealing_warmup: Configuration for Cosine Annealing with Linear Warmup scheduler.
        linear_warmup_linear_decay: Configuration for Linear Warmup + Linear Decay scheduler.
        multi_step_lr: Configuration for the step-decay (milestones) schedule
            with a per-step linear warmup.
    """

    step_lr: Optional[StepLRConfig] = None
    reduce_lr_on_plateau: Optional[ReduceLROnPlateauConfig] = field(
        factory=ReduceLROnPlateauConfig
    )
    cosine_annealing_warmup: Optional[CosineAnnealingWarmupConfig] = None
    linear_warmup_linear_decay: Optional[LinearWarmupLinearDecayConfig] = None
    multi_step_lr: Optional[MultiStepLRConfig] = None


#: ``lr_scheduler`` branches planned around a fixed training length, in priority
#: order (all of them outrank ``step_lr`` and ``reduce_lr_on_plateau``).
FIXED_LENGTH_LR_SCHEDULES = (
    "cosine_annealing_warmup",
    "linear_warmup_linear_decay",
    "multi_step_lr",
)


def fixed_length_lr_schedule(lr_scheduler: Any) -> Optional[str]:
    """Name of the fixed-length LR schedule that training will run, if any.

    Args:
        lr_scheduler: The ``trainer_config.lr_scheduler`` section (``DictConfig``,
            dict or `LRSchedulerConfig`), or ``None``.

    Returns:
        The highest-priority entry of `FIXED_LENGTH_LR_SCHEDULES` that is set, or
        ``None`` when the run uses ``step_lr``, ``reduce_lr_on_plateau`` or no
        scheduler.
    """
    if lr_scheduler is None:
        return None
    for name in FIXED_LENGTH_LR_SCHEDULES:
        if hasattr(lr_scheduler, "get"):
            value = lr_scheduler.get(name, None)
        else:
            value = getattr(lr_scheduler, name, None)
        if value is not None:
            return name
    return None


@define
class EarlyStoppingConfig:
    """Configuration for early_stopping.

    Attributes:
        stop_training_on_plateau: (bool) True if early stopping should be enabled. `None` = on, except with a fixed-length LR schedule (`lr_scheduler.cosine_annealing_warmup`, `linear_warmup_linear_decay` or `multi_step_lr`), whose decay would be cut short by it; there it is off. Training writes the resolved value into the saved `training_config.yaml`. *Default*: `None`.
        min_delta: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement. *Default*: `1e-8`.
        patience: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. For the ``embedding`` model type, which selects on a retrieval metric measured every ``eval.frequency`` epochs, one check happens per evaluation, so patience counts evaluations. *Default*: `10`.
    """

    min_delta: float = field(default=1e-8, validator=validators.ge(0))
    patience: int = field(default=10, validator=validators.ge(0))
    stop_training_on_plateau: Optional[bool] = None


@define
class EMAConfig:
    """Configuration for an exponential moving average (EMA) of the model weights.

    When enabled, a float32 copy of the trainable weights (and floating-point
    buffers) is updated after every optimizer step as
    ``ema = d * ema + (1 - d) * weights`` with ``d = decay * (1 - exp(-step /
    warmup_steps))`` (mmdet/RTMPose ``ExpMomentumEMA``; ``warmup_steps=0`` uses
    ``decay`` from the first step). The EMA weights are swapped in for
    validation (so ``val/loss``, epoch-end evaluation and checkpoint selection
    see them) and are what checkpoints store as ``state_dict``, so inference and
    export load them with no changes. The raw (online) weights ride along in the
    checkpoint under ``"ema"`` for resuming.

    Attributes:
        enabled: (bool) Track and use an EMA of the weights. *Default*: `False`.
        decay: (float) EMA decay after warmup (``1 - momentum``). *Default*:
            `0.9998`.
        warmup_steps: (int) Time constant (in optimizer steps) of the decay
            warmup; ``0`` disables it. *Default*: `2000`.
    """

    enabled: bool = False
    decay: float = field(default=0.9998, validator=[validators.ge(0), validators.lt(1)])
    warmup_steps: int = field(default=2000, validator=validators.ge(0))


@define
class EvalConfig:
    """Configuration for epoch-end evaluation.

    Attributes:
        enabled: (bool) Enable epoch-end evaluation metrics. *Default*: `False`.
        frequency: (int) Evaluate every N epochs. *Default*: `1`.
        oks_stddev: (float) OKS standard deviation for evaluation. *Default*: `0.025`.
        oks_scale: (float) OKS scale override. If None, uses default. *Default*: `None`.
        oks_sigmas: Per-node OKS sigmas for post-training evaluation, overriding
            ``oks_stddev``: a preset (``"coco"``, ``"crowdpose"``, ``"ochuman"``),
            ``"provenance"`` / ``"auto"`` (the GT file's ``node_sigmas``), or a list
            with one value per node. *Default*: `None`.
        coco_metrics: (bool) Also report COCOeval-faithful keypoint AP/AR
            (``coco.AP``, ``coco.AP50``, ...; see ``sleap_nn.evaluation_coco``) in
            post-training evaluation. For top-down models
            (``centered_instance``, ``multi_class_topdown``) it also switches the
            post-training predictions to the COCO protocol: no keypoint is
            dropped (``peak_threshold`` 0) and instances are rescored as mmpose
            does (stage-1 score x mean keypoint score above 0.2). Without it,
            low-confidence keypoints are dropped at 0.2, which costs several AP
            on COCO. *Default*: `False`.
        match_threshold: (float) Match threshold for post-training evaluation.
            For centroid models it is the maximum centroid distance in PIXELS
            (default `50.0`). For ``bottomup_segmentation`` it is reinterpreted
            as a minimum mask IoU in ``(0, 1]``; the default `50.0` is out of
            that range, so segmentation eval falls back to `0.5` unless you set
            a value in ``(0, 1]`` (see ``train._run_segmentation_split_eval``).
            *Default*: `50.0`.
        select_metric: (str) For the ``embedding`` model type, which retrieval /
            verification metric selects the best checkpoint (and drives early
            stopping): one of ``rank1``, ``mAP``, ``auc``, ``knn_acc`` (higher is
            better) or ``eer`` (lower is better). For the ``bbox_*`` detectors, one
            of ``bbox_mAP``, ``bbox_AP50``, ``bbox_AP75``, ``bbox_AR``, ``bbox_f1``
            or ``val_loss``; left at the default, detectors select on ``bbox_mAP``
            when ``enabled`` and on ``val/loss`` otherwise. Box and retrieval
            metrics are checked on evaluation epochs only. Ignored by every other
            model type, which selects on ``val/loss``. *Default*: `"rank1"`.
    """

    enabled: bool = False
    frequency: int = field(default=1, validator=validators.ge(1))
    oks_stddev: float = field(default=0.025, validator=validators.gt(0))
    oks_scale: Optional[float] = None
    oks_sigmas: Optional[Any] = None
    coco_metrics: bool = False
    match_threshold: float = field(default=50.0, validator=validators.gt(0))
    # Declared here so it is reachable from YAML at all: a structured merge against
    # this dataclass raises `ConfigKeyError` for any key it does not define, so
    # `trainer_config.eval.select_metric` in a config file could only ever fail --
    # the mode table in `model_trainer` was dead for config users and selection was
    # always rank-1. The valid names are validated there (one table, one place).
    select_metric: str = "rank1"


@define
class HardKeypointMiningConfig:
    """Configuration for online hard keypoint mining.

    Attributes:
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When this is enabled, the loss is computed per keypoint (or edge for PAFs) and sorted from lowest (easy) to highest (hard). The hard keypoint loss will be scaled to have a higher weight in the total loss, encouraging the training to focus on tricky body parts that are more difficult to learn. If False, no mining will be performed and all keypoints will be weighted equally in the loss. *Default*: `False`.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with respect to the lowest keypoint loss in order to be considered as "hard". This helps to switch focus on across groups of keypoints during training. *Default*: `2.0`.
        min_hard_keypoints: The minimum number of keypoints that will be considered as "hard", even if they are not below the `hard_to_easy_ratio`. *Default*: `2`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to. This can help when there are few very easy keypoints which may skew the ratio and result in loss scaling being applied to most keypoints, which can reduce the impact of hard mining altogether. *Default*: `None`.
        loss_scale: Factor to scale the hard keypoint losses by. *Default*: `5.0`.
        per_sample: If True, hard keypoints are ranked within each sample, over the
            nodes with non-zero loss weight (see `data_config.node_loss_mask`), and the
            loss is the mean of the selected (sample, node) losses. If False, they are
            ranked on losses summed over the batch (the behavior before sleap-nn 0.4,
            which cannot see node weights). Applies to the single-instance,
            centered-instance and bottom-up confmap heads; results differ from
            `False` for batch size > 1. *Default*: `True`.
    """

    online_mining: bool = False
    hard_to_easy_ratio: float = 2.0
    min_hard_keypoints: int = 2
    max_hard_keypoints: Optional[int] = None
    loss_scale: float = 5.0
    per_sample: bool = True


@define
class ZMQConfig:
    """Configuration of ZeroMQ-based monitoring of the training.

    ZMQ monitoring is **disabled by default** and exists for training launched
    from the SLEAP GUI, which supplies these ports itself. Each channel is
    enabled independently by setting its port; leaving a port as `None` (the
    default) keeps that channel off, and no ZMQ callback is attached.

    Attributes:
        controller_port: Port number of the endpoint to listen for command messages from, i.e. "tcp://127.0.0.1:{port_number}". Leave as `None` to disable the controller (the default). *Default*: `None`.
        controller_polling_timeout: Polling timeout in microseconds specified as an integer. This controls how long the poller should wait to receive a response and should be set to a small value to minimize the impact on training speed. Only used when `controller_port` is set. *Default*: `10`.
        publish_port: Port number of the endpoint to publish updates to, i.e. "tcp://127.0.0.1:{port_number}". Leave as `None` to disable progress publishing (the default). *Default*: `None`.
    """

    controller_port: Optional[int] = None
    controller_polling_timeout: int = 10
    publish_port: Optional[int] = None


@define
class FreezeScheduleConfig:
    """Freeze part of the model for the first steps of training.

    Attributes:
        targets: (List[str]) What to freeze: ``"encoder"`` (the backbone's
            pretrained encoder), ``"embeddings"``, ``"blocks[:6]"`` (a slice of a
            plain ViT's blocks), ``"backbone"``, ``"decoder"``, ``"head"`` or a glob
            on parameter names. *Default*: ``None`` (``["encoder"]``).
        unfreeze_at_step: (int) Global step at which the targets are unfrozen;
            ``None`` keeps them frozen for the whole run. Requires the backbone's
            ``freeze: false`` (the parameters must be in the optimizer).
            *Default*: ``None``.

    Under DDP, ``find_unused_parameters`` is turned on automatically (frozen
    parameters get no gradient); an explicit
    ``trainer_strategy: ddp_find_unused_parameters_false`` is refused.
    """

    targets: Optional[List[str]] = None
    unfreeze_at_step: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(0))
    )


@define
class TrainerConfig:
    """Configuration for trainer.

    Attributes:
        train_data_loader: (Note: Any parameters from Torch's DataLoader could be used.)
        val_data_loader: (Similar to train_data_loader)
        model_ckpt: (Note: Any parameters from Lightning's ModelCheckpoint could be used.)
        trainer_num_devices: (int) Number of devices to use or "auto" to let Lightning decide. If `None`, it defaults to `"auto"` when `trainer_device_indices` is also `None`, otherwise its value is inferred from trainer_device_indices. Default: None.
        trainer_device_indices: (list) List of device indices to use. For example, `[0, 1]` selects two devices and overrides `trainer_devices`, while `[2]` with `trainer_devices=2` still runs only on `device 2` (not two devices). If `None`, the number of devices is taken from `trainer_devices`, starting from index 0. Default: `None`.
        trainer_accelerator: (str) One of the ("cpu", "gpu", "mps", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the Trainer to be connected to. *Default*: `"auto"`.
        profiler: (str) Profiler for pytorch Trainer. One of ["advanced", "passthrough", "pytorch", "simple"]. *Default*: `None`.
        trainer_strategy: (str) Training strategy, one of ["auto", "ddp", "fsdp", "ddp_find_unused_parameters_false", "ddp_find_unused_parameters_true", ...]. This supports any training strategy that is supported by `lightning.Trainer`. *Default*: `"auto"`.
        enable_progress_bar: (bool) When True, enables printing the logs during training. *Default*: `True`.
        min_train_steps_per_epoch: (int) Minimum number of iterations in a single epoch. (Useful if model is trained with very few data points). Refer limit_train_batches parameter of Torch Trainer. For ``embedding``, one iteration is a P x K batch (``objective.sampler``), not ``train_data_loader.batch_size`` samples. *Default*: `200`.
        train_steps_per_epoch: (int) Number of minibatches (steps) to train for in an epoch. If set to `None`, this is set to the number of batches in the training data (for ``embedding``, the number of P x K batches that covers it once) or `min_train_steps_per_epoch`, whichever is largest. *Default*: `None`. **Note**: With multiple devices this counts batches summed over ALL devices (so an epoch stays one pass over the data): each device runs `train_steps_per_epoch / trainer_devices` optimizer steps per epoch. `min_train_steps_per_epoch` is counted the same way.
        visualize_preds_during_training: (bool) If set to `True`, sample predictions (keypoints + confidence maps) are saved to `viz` folder in the ckpt dir and in wandb table. *Default*: `False`.
        keep_viz: (bool) If set to `True`, the `viz` folder will be kept after training. If `False`, the `viz` folder will be deleted after training. Only applies when `visualize_preds_during_training` is `True`. *Default*: `False`.
        viz_img_format: (str) Image format for the visualization figures saved to the local `viz` folder, one of `"png"` or `"jpg"`. `"jpg"` produces much smaller files, which helps when training a battery of models on the same dataset (#644). Only applies when `visualize_preds_during_training` is `True`. *Default*: `"png"`.
        max_epochs: (int) Maximum number of epochs to run. *Default*: `100`.
        seed: (int) Seed value for the current experiment. This ensures deterministic train/val splits across runs, which is critical for safe checkpoint resume. *Default*: `42`.
        use_wandb: (bool) True to enable wandb logging. *Default*: `False`.
        save_ckpt: (bool) True to enable checkpointing. *Default*: `False`.
        ckpt_dir: (str) Directory path where the `<run_name>` folder is created. If `None`, a new folder for the current run is created in the working dir. **Default**: `None`
        run_name: (str) Name of the current run. The ckpts will be created in `<ckpt_dir>/<run_name>`. If `None`, a run name is generated with `<timestamp>_<head_name>`. *Default*: `None`.
        resume_ckpt_path: (str) Path to `.ckpt` file from which training is resumed. *Default*: `None`.
        wandb: (Only if use_wandb is True, else skip this)
        optimizer_name: (str) Optimizer to be used. One of ["Adam", "AdamW"]. *Default*: `"Adam"`.
        optimizer: create an optimizer configuration
        lr_scheduler: create an lr_scheduler configuration
        early_stopping: create an early_stopping configuration
        ema: Weight EMA configuration (``EMAConfig``); off by default.
        zmq: Zmq config with publish and controller port addresses.
        precision: (str) Lightning training precision: `"32"` (full fp32), `"bf16-mixed"` (bf16 autocast, fp32 weights/optimizer; recommended on Ampere+ GPUs), or `"16-mixed"` (fp16 autocast with a gradient scaler). *Default*: `"32"`.
        float32_matmul_precision: (str) Passed to `torch.set_float32_matmul_precision` for the duration of `fit` on CUDA, and restored afterwards. `"high"` lets fp32 matmuls use TF32 tensor cores (~1.4-1.6x faster ViT training on Ampere+; ~10 mantissa bits for matmul inputs, fp32 accumulation); `"highest"` forces strict fp32 matmuls; `None` leaves the process setting untouched (PyTorch's default is `"highest"`). cuDNN convolutions already use TF32 by default (`torch.backends.cudnn.allow_tf32`) and are not affected. Has no effect under `bf16-mixed`/`16-mixed` autocast regions, only on ops that stay fp32. *Default*: `None`.
        compile: (bool or str) `torch.compile` the model (backbone + heads) for training and validation; `true` = `"default"` mode, or one of `"reduce-overhead"`, `"max-autotune"`, `"max-autotune-no-cudagraphs"`. Compiles in place, so checkpoint keys are unchanged. The first steps are slow while compiling. *Default*: `False`.
        gradient_clip_val: (float) Clip gradients at this value (see `gradient_clip_algorithm`). `None` disables clipping. *Default*: `None`.
        gradient_clip_algorithm: (str) `"norm"` (global L2 norm) or `"value"`. *Default*: `"norm"`.
        accumulate_grad_batches: (int) Accumulate gradients over this many batches per optimizer step (effective batch = `batch_size x num_devices x accumulate_grad_batches`). *Default*: `1`.
        log_every_n_steps: (int) How often step-level metrics (e.g. `loss`) are sent to the loggers (wandb). Each logged step forces a host-device sync, so large-batch runs should use ~50. *Default*: `1`.
        max_steps: (int) Total optimizer-step budget. Training stops at `max_steps` or `max_epochs`, whichever comes first (set `max_epochs: -1` to train by steps only; epoch-based LR schedulers then need their own `max_epochs`). `None` = no step limit. *Default*: `None`.
        val_check_interval: (int or float) Run validation every N training batches (int, counted across epoch boundaries, so it may exceed `train_steps_per_epoch`) or every fraction of an epoch (float in `(0, 1]`). With an int, validation no longer runs at every epoch end, and early-stopping `patience` counts validation runs. `None` = once per epoch. *Default*: `None`.
        throughput_log_every_n_steps: (int) Window (in training batches) for the `ThroughputMonitor` callback, which logs train-only samples/s, iteration time, data-wait time and fraction, GPU idle fraction, peak GPU memory and the LR of every param group under `perf/*`. `0`/`None` disables it. *Default*: `50`.
        confmap_fg_bg_diagnostics: (bool) Log the confmap foreground/background MSE split (`{train,val}/confmap_loss_{fg,bg}`, `confmap_fg_frac`). `None` = only for `centroid` models. *Default*: `None`.
    """

    train_data_loader: TrainDataLoaderConfig = field(factory=TrainDataLoaderConfig)
    val_data_loader: ValDataLoaderConfig = field(factory=ValDataLoaderConfig)
    model_ckpt: ModelCkptConfig = field(factory=ModelCkptConfig)
    trainer_devices: Optional[Any] = field(
        default=None,
        validator=lambda inst, attr, val: TrainerConfig.validate_trainer_devices(val),
    )
    trainer_device_indices: Optional[List[int]] = None
    trainer_accelerator: str = "auto"
    profiler: Optional[str] = None
    trainer_strategy: str = "auto"
    enable_progress_bar: bool = True
    min_train_steps_per_epoch: int = 200
    train_steps_per_epoch: Optional[int] = None
    visualize_preds_during_training: bool = False
    keep_viz: bool = False
    viz_img_format: str = "png"
    max_epochs: int = 100
    seed: Optional[int] = 42
    use_wandb: bool = False
    save_ckpt: bool = False
    ckpt_dir: Optional[str] = "."
    run_name: Optional[str] = None
    resume_ckpt_path: Optional[str] = None
    wandb: WandBConfig = field(factory=WandBConfig)
    optimizer_name: str = field(
        default="Adam",
        validator=lambda inst, attr, val: TrainerConfig.validate_optimizer_name(val),
    )
    optimizer: OptimizerConfig = field(factory=OptimizerConfig)
    lr_scheduler: Optional[LRSchedulerConfig] = field(factory=LRSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(factory=EarlyStoppingConfig)
    ema: EMAConfig = field(factory=EMAConfig)
    freeze_schedule: Optional[FreezeScheduleConfig] = None
    online_hard_keypoint_mining: Optional[HardKeypointMiningConfig] = field(
        factory=HardKeypointMiningConfig
    )
    zmq: Optional[ZMQConfig] = field(factory=ZMQConfig)  # Required for SLEAP GUI
    eval: EvalConfig = field(factory=EvalConfig)  # Epoch-end evaluation config
    precision: Any = field(
        default="32",
        validator=lambda inst, attr, val: TrainerConfig.validate_precision(val),
    )
    float32_matmul_precision: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.in_(["highest", "high", "medium"])),
    )
    compile: Any = field(
        default=False,
        validator=lambda inst, attr, val: TrainerConfig.validate_compile(val),
    )
    gradient_clip_val: Optional[float] = field(
        default=None, validator=validators.optional(validators.gt(0))
    )
    gradient_clip_algorithm: str = field(
        default="norm", validator=validators.in_(["norm", "value"])
    )
    accumulate_grad_batches: int = field(default=1, validator=validators.ge(1))
    log_every_n_steps: int = field(default=1, validator=validators.ge(1))
    max_steps: Optional[int] = field(
        default=None, validator=validators.optional(validators.ge(1))
    )
    val_check_interval: Optional[Any] = field(
        default=None,
        validator=lambda inst, attr, val: TrainerConfig.validate_val_check_interval(
            val
        ),
    )
    throughput_log_every_n_steps: Optional[int] = field(
        default=50, validator=validators.optional(validators.ge(0))
    )
    confmap_fg_bg_diagnostics: Optional[bool] = None

    @staticmethod
    def validate_precision(value):
        """Validate the Lightning training precision."""
        if str(value) in TRAINER_PRECISIONS:
            return True
        message = (
            f"precision must be one of {sorted(TRAINER_PRECISIONS)}, got {value!r}."
        )
        logger.error(message)
        raise ValueError(message)

    @staticmethod
    def validate_val_check_interval(value):
        """Validate `val_check_interval` (int >= 1 batches, or a float in (0, 1])."""
        if value is None:
            return True
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return True
        if isinstance(value, float) and 0.0 < value <= 1.0:
            return True
        message = (
            "val_check_interval must be an int >= 1 (training batches) or a float "
            f"in (0, 1] (fraction of an epoch), got {value!r}."
        )
        logger.error(message)
        raise ValueError(message)

    @staticmethod
    def validate_compile(value):
        """Validate the `compile` option (bool or a `torch.compile` mode)."""
        if isinstance(value, bool) or value is None or value in COMPILE_MODES:
            return True
        message = (
            f"compile must be a bool or one of {list(COMPILE_MODES)}, got {value!r}."
        )
        logger.error(message)
        raise ValueError(message)

    @staticmethod
    def validate_optimizer_name(value):
        """Validate that optimizer_name is one of the allowed values."""
        if value not in ["Adam", "AdamW"]:
            message = "optimizer_name must be one of: Adam, AdamW"
            logger.error(message)
            raise ValueError(message)
        return True

    @staticmethod
    def validate_trainer_devices(value):
        """Validate the value of trainer_devices."""
        if value is None:
            return
        if isinstance(value, int) and value >= 0:
            return
        if isinstance(value, str) and value == "auto":
            return
        message = "trainer_devices must be an integer >= 0, or the string 'auto'."
        logger.error(message)
        raise ValueError(message)


def trainer_mapper(legacy_config: dict) -> TrainerConfig:
    """Map the legacy trainer configuration to the new trainer configuration.

    Args:
        legacy_config: A dictionary containing the legacy trainer configuration.

    Returns:
        An instance of `TrainerConfig` with the mapped configuration.
    """
    legacy_config_optimization = legacy_config.get("optimization", {})
    legacy_config_outputs = legacy_config.get("outputs", {})
    run_name = legacy_config_outputs.get("run_name", None)
    run_name = run_name if run_name is not None else ""
    run_name_prefix = legacy_config_outputs.get("run_name_prefix", "")
    run_name_suffix = legacy_config_outputs.get("run_name_suffix", "")
    run_name = (
        run_name_prefix
        if run_name_prefix is not None
        else "" + run_name + run_name_suffix if run_name_prefix is not None else ""
    )
    run_name = None if run_name == "" else run_name

    trainer_cfg_args = {}
    train_dataloader_cfg_args = {}
    val_dataloader_cfg_args = {}
    model_ckpt_cfg_args = {}
    optimizer_cfg_args = {}
    lr_scheduler_cfg_args = {}
    reduce_lr_on_plateau_cfg_args = {}
    early_stopping_cfg_args = {}
    zmq_cfg_args = {}
    online_hard_keypoint_mining_cfg_args = {}

    # train dataloader
    if legacy_config_optimization.get("batch_size", None) is not None:
        train_dataloader_cfg_args["batch_size"] = legacy_config_optimization[
            "batch_size"
        ]

    if legacy_config_optimization.get("online_shuffling", None) is not None:
        train_dataloader_cfg_args["shuffle"] = legacy_config_optimization[
            "online_shuffling"
        ]

    if legacy_config_optimization.get("num_workers", None) is not None:
        train_dataloader_cfg_args["num_workers"] = legacy_config_optimization[
            "num_workers"
        ]

    trainer_cfg_args["train_data_loader"] = TrainDataLoaderConfig(
        **train_dataloader_cfg_args
    )

    # val dataloader
    if legacy_config_optimization.get("batch_size", None) is not None:
        val_dataloader_cfg_args["batch_size"] = legacy_config_optimization["batch_size"]

    if legacy_config_optimization.get("num_workers", None) is not None:
        val_dataloader_cfg_args["num_workers"] = legacy_config_optimization[
            "num_workers"
        ]

    trainer_cfg_args["val_data_loader"] = ValDataLoaderConfig(**val_dataloader_cfg_args)

    # model ckpt
    if (
        legacy_config_outputs.get("checkpointing", {}).get("latest_model", None)
        is not None
    ):
        model_ckpt_cfg_args["save_last"] = legacy_config_outputs["checkpointing"][
            "latest_model"
        ]

    trainer_cfg_args["model_ckpt"] = ModelCkptConfig(**model_ckpt_cfg_args)

    if legacy_config_outputs.get("save_visualizations", None) is not None:
        trainer_cfg_args["visualize_preds_during_training"] = legacy_config_outputs[
            "save_visualizations"
        ]

    # Handle legacy delete_viz_images parameter
    if legacy_config_outputs.get("keep_viz_images", None) is not None:
        trainer_cfg_args["keep_viz"] = legacy_config_outputs["keep_viz_images"]

    if legacy_config_optimization.get("epochs", None) is not None:
        trainer_cfg_args["max_epochs"] = legacy_config_optimization["epochs"]

    if legacy_config_optimization.get("min_batches_per_epoch", None) is not None:
        trainer_cfg_args["min_train_steps_per_epoch"] = legacy_config_optimization[
            "min_batches_per_epoch"
        ]

    if legacy_config_optimization.get("batches_per_epoch", None) is not None:
        trainer_cfg_args["train_steps_per_epoch"] = legacy_config_optimization[
            "batches_per_epoch"
        ]

    trainer_cfg_args["save_ckpt"] = True
    trainer_cfg_args["ckpt_dir"] = (
        Path(legacy_config_outputs.get("runs_folder", "."))
    ).as_posix()
    trainer_cfg_args["run_name"] = run_name

    trainer_cfg_args["optimizer_name"] = re.sub(
        r"^[a-z]",
        lambda x: x.group().upper(),
        legacy_config_optimization.get("optimizer", "adam"),
    )
    if legacy_config_optimization.get("initial_learning_rate", None) is not None:
        optimizer_cfg_args["lr"] = legacy_config_optimization["initial_learning_rate"]

    trainer_cfg_args["optimizer"] = OptimizerConfig(**optimizer_cfg_args)

    if (
        legacy_config_optimization.get("learning_rate_schedule", {}).get(
            "reduce_on_plateau", None
        )
        is not None
    ):
        if legacy_config_optimization["learning_rate_schedule"]["reduce_on_plateau"]:
            if (
                legacy_config_optimization.get("learning_rate_schedule", {}).get(
                    "plateau_min_delta", None
                )
                is not None
            ):
                reduce_lr_on_plateau_cfg_args["threshold"] = legacy_config_optimization[
                    "learning_rate_schedule"
                ]["plateau_min_delta"]

            if (
                legacy_config_optimization.get("learning_rate_schedule", {}).get(
                    "plateau_cooldown", None
                )
                is not None
            ):
                reduce_lr_on_plateau_cfg_args["cooldown"] = legacy_config_optimization[
                    "learning_rate_schedule"
                ]["plateau_cooldown"]

            if (
                legacy_config_optimization.get("learning_rate_schedule", {}).get(
                    "reduction_factor", None
                )
                is not None
            ):
                reduce_lr_on_plateau_cfg_args["factor"] = legacy_config_optimization[
                    "learning_rate_schedule"
                ]["reduction_factor"]

            if (
                legacy_config_optimization.get("learning_rate_schedule", {}).get(
                    "plateau_patience", None
                )
                is not None
            ):
                reduce_lr_on_plateau_cfg_args["patience"] = legacy_config_optimization[
                    "learning_rate_schedule"
                ]["plateau_patience"]

            if (
                legacy_config_optimization.get("learning_rate_schedule", {}).get(
                    "min_learning_rate", None
                )
                is not None
            ):
                reduce_lr_on_plateau_cfg_args["min_lr"] = legacy_config_optimization[
                    "learning_rate_schedule"
                ]["min_learning_rate"]

            lr_scheduler_cfg_args["reduce_lr_on_plateau"] = ReduceLROnPlateauConfig(
                **reduce_lr_on_plateau_cfg_args
            )

    trainer_cfg_args["lr_scheduler"] = LRSchedulerConfig(**lr_scheduler_cfg_args)

    if (
        legacy_config_optimization.get("early_stopping", {}).get(
            "stop_training_on_plateau", None
        )
        is not None
    ):
        early_stopping_cfg_args["stop_training_on_plateau"] = (
            legacy_config_optimization["early_stopping"]["stop_training_on_plateau"]
        )
        if (
            legacy_config_optimization.get("early_stopping", {}).get(
                "plateau_min_delta", None
            )
            is not None
        ):
            early_stopping_cfg_args["min_delta"] = legacy_config_optimization[
                "early_stopping"
            ]["plateau_min_delta"]

        if (
            legacy_config_optimization.get("early_stopping", {}).get(
                "plateau_patience", None
            )
            is not None
        ):
            early_stopping_cfg_args["patience"] = legacy_config_optimization[
                "early_stopping"
            ]["plateau_patience"]

    trainer_cfg_args["early_stopping"] = EarlyStoppingConfig(**early_stopping_cfg_args)

    if (
        legacy_config_optimization.get("hard_keypoint_mining", {}).get(
            "online_mining", None
        )
        is not None
    ):
        if legacy_config_optimization["hard_keypoint_mining"]["online_mining"]:
            online_hard_keypoint_mining_cfg_args["online_mining"] = True

        if (
            legacy_config_optimization.get("hard_keypoint_mining", {}).get(
                "hard_to_easy_ratio", None
            )
            is not None
        ):
            online_hard_keypoint_mining_cfg_args["hard_to_easy_ratio"] = (
                legacy_config_optimization["hard_keypoint_mining"]["hard_to_easy_ratio"]
            )

        if (
            legacy_config_optimization.get("hard_keypoint_mining", {}).get(
                "min_hard_keypoints", None
            )
            is not None
        ):
            online_hard_keypoint_mining_cfg_args["min_hard_keypoints"] = (
                legacy_config_optimization["hard_keypoint_mining"]["min_hard_keypoints"]
            )

        if (
            legacy_config_optimization.get("hard_keypoint_mining", {}).get(
                "max_hard_keypoints", None
            )
            is not None
        ):
            online_hard_keypoint_mining_cfg_args["max_hard_keypoints"] = (
                legacy_config_optimization["hard_keypoint_mining"]["max_hard_keypoints"]
            )

        if (
            legacy_config_optimization.get("hard_keypoint_mining", {}).get(
                "loss_scale", None
            )
            is not None
        ):
            online_hard_keypoint_mining_cfg_args["loss_scale"] = (
                legacy_config_optimization["hard_keypoint_mining"]["loss_scale"]
            )

    trainer_cfg_args["online_hard_keypoint_mining"] = HardKeypointMiningConfig(
        **online_hard_keypoint_mining_cfg_args
    )

    legacy_zmq = legacy_config_outputs.get("zmq", {}) or {}

    # ZMQ monitoring is opt-in: only carry the ports over when the legacy config
    # actually enabled the corresponding channel. `subscribe_to_controller` /
    # `publish_updates` are booleans, so a `False` here must leave the port unset
    # (i.e. ZMQ off) rather than being treated as "present".
    if legacy_zmq.get("subscribe_to_controller", False) and legacy_zmq.get(
        "controller_address", None
    ):
        zmq_cfg_args["controller_port"] = int(
            legacy_zmq["controller_address"].split(":")[-1]
        )

    if legacy_zmq.get("publish_updates", False) and legacy_zmq.get(
        "publish_address", None
    ):
        zmq_cfg_args["publish_port"] = int(legacy_zmq["publish_address"].split(":")[-1])

    if legacy_zmq.get("controller_polling_timeout", None) is not None:
        zmq_cfg_args["controller_polling_timeout"] = legacy_zmq[
            "controller_polling_timeout"
        ]

    trainer_cfg_args["zmq"] = ZMQConfig(**zmq_cfg_args)

    return TrainerConfig(**trainer_cfg_args)
