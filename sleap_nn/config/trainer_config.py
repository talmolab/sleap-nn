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
    """Train and val DataLoaderConfig.

    Any parameters from Torch's DataLoader could be used.

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. Default = 1.
        shuffle: (bool) True to have the data reshuffled at every epoch. Default: False.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default: 0.
    """

    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0


@define
class ModelCkptConfig:
    """Configuration for model checkpoint.

    Any parameters from Lightning's ModelCheckpoint could be used.

    Attributes:
        save_top_k: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False.
        save_last: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. Default: None.
    """

    save_top_k: int = 1
    save_last: Optional[bool] = None


@define
class WandBConfig:
    """Configuration for WandB.

    Only if use_wandb is True, else skip this

    Attributes:
        entity: (str) Entity of wandb project.
        project: (str) Project name for the wandb project.
        name: (str) Name of the current run.
        api_key: (str) API key. The API key is masked when saved to config files.
        wandb_mode: (str) "offline" if only local logging is required. Default: "None".
        prv_runid: (str) Previous run ID if training should be resumed from a previous ckpt. Default: None.
        group: (str) Group for wandb logging.
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    api_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    prv_runid: Optional[str] = None
    group: Optional[str] = None


@define
class OptimizerConfig:
    """Configuration for optimizer.

    Attributes:
        lr: (float) Learning rate of type float. Default: 1e-3
        amsgrad: (bool) Enable AMSGrad with the optimizer. Default: False
    """

    lr: float = field(default=1e-3, validator=validators.gt(0))
    amsgrad: bool = False


@define
class StepLRConfig:
    """Configuration for StepLR scheduler.

    Attributes:
        step_size: (int) Period of learning rate decay. If step_size=10, then every 10 epochs, learning rate will be reduced by a factor of gamma.
        gamma: (float) Multiplicative factor of learning rate decay. Default: 0.1.
    """

    step_size: int = field(default=10, validator=validators.gt(0))
    gamma: float = 0.1


@define
class ReduceLROnPlateauConfig:
    """Configuration for ReduceLROnPlateau scheduler.

    Attributes:
        threshold: (float) Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode: (str) One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: "rel".
        cooldown: (int) Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0
        patience: (int) Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn't improved then. Default: 10.
        factor: (float) Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        min_lr: (float or List[float]) A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
    """

    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    patience: int = 10
    factor: float = 0.1
    min_lr: Any = field(
        default=0.0, validator=lambda instance, attr, value: instance.validate_min_lr()
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
    reduce_lr_on_plateau: Optional[ReduceLROnPlateauConfig] = None


@define
class EarlyStoppingConfig:
    """Configuration for early_stopping.

    Attributes:
        stop_training_on_plateau: (bool) True if early stopping should be enabled.
        min_delta: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement.
        patience: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch.
    """

    min_delta: float = field(default=0.0, validator=validators.ge(0))
    patience: int = field(default=1, validator=validators.ge(0))
    stop_training_on_plateau: bool = False


@define
class TrainerConfig:
    """Configuration for trainer.

    Attributes:
        train_data_loader: (Note: Any parameters from Torch's DataLoader could be used.)
        val_data_loader: (Similar to train_data_loader)
        model_ckpt: (Note: Any parameters from Lightning's ModelCheckpoint could be used.)
        trainer_devices: (int) Number of devices to train on (int), which devices to train on (list or str), or "auto" to select automatically.
        trainer_accelerator: (str) One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the Trainer to be connected to.
        profiler: (str) Profiler for pytorch Trainer. One of ["advanced", "passthrough", "pytorch", "simple"].
        trainer_strategy: (str) Training strategy, one of ["auto", "ddp", "fsdp", "ddp_find_unused_parameters_false", "ddp_find_unused_parameters_true", ...]. This supports any training strategy that is supported by `lightning.Trainer`.
        enable_progress_bar: (bool) When True, enables printing the logs during training.
        steps_per_epoch: (int) Minimum number of iterations in a single epoch. (Useful if model is trained with very few data points). Refer limit_train_batches parameter of Torch Trainer. If None, the number of iterations depends on the number of samples in the train dataset.
        max_epochs: (int) Maxinum number of epochs to run.
        seed: (int) Seed value for the current experiment.
        use_wandb: (bool) True to enable wandb logging.
        save_ckpt: (bool) True to enable checkpointing.
        save_ckpt_path: (str) Directory path to save the training config and checkpoint files. Default: "./"
        resume_ckpt_path: (str) Path to .ckpt file from which training is resumed. Default: None.
        wandb: (Only if use_wandb is True, else skip this)
        optimizer_name: (str) Optimizer to be used. One of ["Adam", "AdamW"].
        optimizer: create an optimizer configuration
        lr_scheduler: create an lr_scheduler configuration
        early_stopping: create an early_stopping configuration
        zmq: Dict with keys ["publish_adddress", "controller_address"]. `publish_address` specifies the address and port to which the training logs (loss values) should be sent to. `controller_address` specifies the address and port to listen to to stop the training (specific to SLEAP GUI).
    """

    train_data_loader: DataLoaderConfig = field(factory=DataLoaderConfig)
    val_data_loader: DataLoaderConfig = field(factory=DataLoaderConfig)
    model_ckpt: ModelCkptConfig = field(factory=ModelCkptConfig)
    trainer_devices: Any = field(
        default="auto",
        validator=lambda inst, attr, val: TrainerConfig.validate_trainer_devices(val),
    )
    trainer_accelerator: str = "auto"
    profiler: Optional[str] = None
    trainer_strategy: str = "auto"
    enable_progress_bar: bool = True
    steps_per_epoch: Optional[int] = None
    max_epochs: int = 10
    seed: int = 0
    use_wandb: bool = False
    save_ckpt: bool = False
    save_ckpt_path: Optional[str] = None
    resume_ckpt_path: Optional[str] = None
    wandb: WandBConfig = field(factory=WandBConfig)
    optimizer_name: str = field(
        default="Adam",
        validator=lambda inst, attr, val: TrainerConfig.validate_optimizer_name(val),
    )
    optimizer: OptimizerConfig = field(factory=OptimizerConfig)
    lr_scheduler: Optional[LRSchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    zmq: Optional[dict] = None  # Required for SLEAP GUI

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
        if isinstance(value, int) and value >= 0:
            return
        if isinstance(value, list) and all(
            isinstance(x, int) and x >= 0 for x in value
        ):
            return
        if isinstance(value, str) and value == "auto":
            return
        message = "trainer_devices must be an integer >= 0, a list of integers >= 0, or the string 'auto'."
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
    return TrainerConfig(
        train_data_loader=DataLoaderConfig(
            batch_size=legacy_config_optimization.get("batch_size", 1),
            shuffle=legacy_config_optimization.get("online_shuffling", False),
            num_workers=0,
        ),
        val_data_loader=DataLoaderConfig(
            batch_size=legacy_config_optimization.get("batch_size", 1), num_workers=0
        ),
        model_ckpt=ModelCkptConfig(
            save_last=legacy_config_outputs.get("checkpointing", {}).get(
                "latest_model", False
            ),
        ),
        max_epochs=legacy_config_optimization.get("epochs", 10),
        save_ckpt=True,
        save_ckpt_path=(
            Path(legacy_config_outputs.get("runs_folder", "."))
            / legacy_config_outputs.get("run_name", f"training_{time.time()}")
        ).as_posix(),
        optimizer_name=re.sub(
            r"^[a-z]",
            lambda x: x.group().upper(),
            legacy_config_optimization.get("optimizer", "adam"),
        ),
        optimizer=OptimizerConfig(
            lr=legacy_config_optimization.get("initial_learning_rate", 1e-4),
        ),
        lr_scheduler=(
            LRSchedulerConfig(
                reduce_lr_on_plateau=ReduceLROnPlateauConfig(
                    threshold=legacy_config_optimization.get(
                        "learning_rate_schedule", {}
                    ).get("plateau_min_delta", 1e-4),
                    cooldown=legacy_config_optimization.get(
                        "learning_rate_schedule", {}
                    ).get("plateau_cooldown", 3),
                    factor=legacy_config_optimization.get(
                        "learning_rate_schedule", {}
                    ).get("reduction_factor", 0.1),
                    patience=legacy_config_optimization.get(
                        "learning_rate_schedule", {}
                    ).get("plateau_patience", 10),
                    min_lr=legacy_config_optimization.get(
                        "learning_rate_schedule", {}
                    ).get("min_learning_rate", 0.0),
                )
            )
            if legacy_config_optimization.get("learning_rate_schedule", {}).get(
                "reduce_on_plateau", False
            )
            else None
        ),
        early_stopping=(
            EarlyStoppingConfig(
                stop_training_on_plateau=legacy_config_optimization.get(
                    "early_stopping", {}
                ).get("stop_training_on_plateau", False),
                min_delta=legacy_config_optimization.get("early_stopping", {}).get(
                    "plateau_min_delta", 0.0
                ),
                patience=legacy_config_optimization.get("early_stopping", {}).get(
                    "plateau_patience", 1
                ),
            )
            if legacy_config_optimization.get("early_stopping")
            else None
        ),
        zmq={
            "publish_adddress": (
                legacy_config_outputs.get("zmq", {}).get("publish_address", None)
                if legacy_config_outputs.get("zmq", {}).get("publish_updates", False)
                else None
            ),
            "controller_address": (
                legacy_config_outputs.get("zmq", {}).get("controller_address", None)
                if legacy_config_outputs.get("zmq", {}).get(
                    "subscribe_to_controller", False
                )
                else None
            ),
        },
    )
