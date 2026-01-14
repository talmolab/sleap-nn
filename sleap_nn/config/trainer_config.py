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


@define
class DataLoaderConfig:
    """Train DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `False`.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. *Default*: `0`.
    """

    batch_size: int = 4
    shuffle: bool = False
    num_workers: int = 0


@define
class TrainDataLoaderConfig(DataLoaderConfig):
    """Train DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `True`.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. *Default*: `0`.
    """

    shuffle: bool = True


@define
class ValDataLoaderConfig(DataLoaderConfig):
    """Validation DataLoaderConfig.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. *Default*: `4`.
        shuffle: (bool) True to have the data reshuffled at every epoch. *Default*: `False`.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. *Default*: `0`.
    """

    shuffle: bool = False


@define
class ModelCkptConfig:
    """Configuration for model checkpoint.

    Any parameters from Lightning's ModelCheckpoint could be used.

    Attributes:
        save_top_k: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False. *Default*: `1`.
        save_last: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. *Default*: `None`.
    """

    save_top_k: int = 1
    save_last: Optional[bool] = None


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
    """

    lr: float = field(default=1e-4, validator=validators.gt(0))
    amsgrad: bool = False


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
class LRSchedulerConfig:
    """Configuration for lr_scheduler.

    Attributes:
        step_lr: Configuration for StepLR scheduler.
        reduce_lr_on_plateau: Configuration for ReduceLROnPlateau scheduler.
    """

    step_lr: Optional[StepLRConfig] = None
    reduce_lr_on_plateau: Optional[ReduceLROnPlateauConfig] = field(
        factory=ReduceLROnPlateauConfig
    )


@define
class EarlyStoppingConfig:
    """Configuration for early_stopping.

    Attributes:
        stop_training_on_plateau: (bool) True if early stopping should be enabled. *Default*: `True`.
        min_delta: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement. *Default*: `1e-8`.
        patience: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. *Default*: `10`.
    """

    min_delta: float = field(default=1e-8, validator=validators.ge(0))
    patience: int = field(default=10, validator=validators.ge(0))
    stop_training_on_plateau: bool = True


@define
class EvalConfig:
    """Configuration for epoch-end evaluation.

    Attributes:
        enabled: (bool) Enable epoch-end evaluation metrics. *Default*: `False`.
        frequency: (int) Evaluate every N epochs. *Default*: `1`.
        oks_stddev: (float) OKS standard deviation for evaluation. *Default*: `0.025`.
        oks_scale: (float) OKS scale override. If None, uses default. *Default*: `None`.
    """

    enabled: bool = False
    frequency: int = field(default=1, validator=validators.ge(1))
    oks_stddev: float = field(default=0.025, validator=validators.gt(0))
    oks_scale: Optional[float] = None


@define
class HardKeypointMiningConfig:
    """Configuration for online hard keypoint mining.

    Attributes:
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When this is enabled, the loss is computed per keypoint (or edge for PAFs) and sorted from lowest (easy) to highest (hard). The hard keypoint loss will be scaled to have a higher weight in the total loss, encouraging the training to focus on tricky body parts that are more difficult to learn. If False, no mining will be performed and all keypoints will be weighted equally in the loss. *Default*: `False`.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with respect to the lowest keypoint loss in order to be considered as "hard". This helps to switch focus on across groups of keypoints during training. *Default*: `2.0`.
        min_hard_keypoints: The minimum number of keypoints that will be considered as "hard", even if they are not below the `hard_to_easy_ratio`. *Default*: `2`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to. This can help when there are few very easy keypoints which may skew the ratio and result in loss scaling being applied to most keypoints, which can reduce the impact of hard mining altogether. *Default*: `None`.
        loss_scale: Factor to scale the hard keypoint losses by. *Default*: `5.0`.
    """

    online_mining: bool = False
    hard_to_easy_ratio: float = 2.0
    min_hard_keypoints: int = 2
    max_hard_keypoints: Optional[int] = None
    loss_scale: float = 5.0


@define
class ZMQConfig:
    """Configuration of ZeroMQ-based monitoring of the training.

    Attributes:
        controller_port: Port number of the endpoint to listen for command messages from. "tcp://tcp://127.0.0.1:{port_number}". Set to `None` to disable log publishing. *Default*: `None`.
        controller_polling_timeout: Polling timeout in microseconds specified as an integer. This controls how long the poller should wait to receive a response and should be set to a small value to minimize the impact on training speed. *Default*: `10`.
        publish_port: Port number of the endpoint to publish updates to. "tcp://tcp://127.0.0.1:{port_number}". Set to `None` to disable log publishing. *Default*: `None`.
    """

    controller_port: Optional[int] = None
    controller_polling_timeout: int = 10
    publish_port: Optional[int] = None


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
        min_train_steps_per_epoch: (int) Minimum number of iterations in a single epoch. (Useful if model is trained with very few data points). Refer limit_train_batches parameter of Torch Trainer. *Default*: `200`.
        train_steps_per_epoch: (int) Number of minibatches (steps) to train for in an epoch. If set to `None`, this is set to the number of batches in the training data or `min_train_steps_per_epoch`, whichever is largest. *Default*: `None`. **Note**: In a multi-gpu training setup, the effective steps during training would be the `trainer_steps_per_epoch` / `trainer_devices`.
        visualize_preds_during_training: (bool) If set to `True`, sample predictions (keypoints + confidence maps) are saved to `viz` folder in the ckpt dir and in wandb table. *Default*: `False`.
        keep_viz: (bool) If set to `True`, the `viz` folder will be kept after training. If `False`, the `viz` folder will be deleted after training. Only applies when `visualize_preds_during_training` is `True`. *Default*: `False`.
        max_epochs: (int) Maximum number of epochs to run. *Default*: `100`.
        seed: (int) Seed value for the current experiment. If None, no seeding is applied. *Default*: `None`.
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
        zmq: Zmq config with publish and controller port addresses.
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
    max_epochs: int = 100
    seed: Optional[int] = None
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
    online_hard_keypoint_mining: Optional[HardKeypointMiningConfig] = field(
        factory=HardKeypointMiningConfig
    )
    zmq: Optional[ZMQConfig] = field(factory=ZMQConfig)  # Required for SLEAP GUI
    eval: EvalConfig = field(factory=EvalConfig)  # Epoch-end evaluation config

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

    if (
        legacy_config_outputs.get("zmq", {}).get("subscribe_to_controller", None)
        is not None
    ):
        zmq_cfg_args["controller_port"] = int(
            legacy_config_outputs["zmq"]["controller_address"].split(":")[-1]
        )

    if legacy_config_outputs.get("zmq", {}).get("publish_updates", None) is not None:
        zmq_cfg_args["publish_port"] = int(
            legacy_config_outputs["zmq"]["publish_address"].split(":")[-1]
        )

    if (
        legacy_config_outputs.get("zmq", {}).get("controller_polling_timeout", None)
        is not None
    ):
        zmq_cfg_args["controller_polling_timeout"] = legacy_config_outputs["zmq"][
            "controller_polling_timeout"
        ]

    trainer_cfg_args["zmq"] = ZMQConfig(**zmq_cfg_args)

    return TrainerConfig(**trainer_cfg_args)
