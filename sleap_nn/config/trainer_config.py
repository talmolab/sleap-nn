import attrs
from omegaconf import OmegaConf
from typing import Optional, Union, List, Text, Any


"""Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
        human, as well as easily be serialized/deserialized to/from simple dictionaries
        and JSON.

    2. Decoupling from the implementation. This makes it easier to design functional
        modules with attributes/parameters that contain objects that may not be easily
        serializable or may implement additional logic that relies on runtime
        information or other parameters.

In general, classes that implement the actual functionality related to these
configuration classes should provide a classmethod for instantiation from the
configuration class instances. This makes it easier to implement other logic not related
to the high level parameters at creation time.

Conveniently, this format also provides a single location where all user-facing
parameters are aggregated and documented for end users (as opposed to developers).
"""


@attrs.define
class DataLoaderConfig:
    """train and val DataLoaderConfig: (Note: Any parameters from Torch's DataLoader could be used.)

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. Default = 1.
        shuffle: (bool) True to have the data reshuffled at every epoch. Default: False.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default: 0.
    """

    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0


@attrs.define
class ModelCkptConfig:
    """modelCkptConfig: (Note: Any parameters from Lightning's ModelCheckpoint could be used.)

    Attributes:
        save_top_k: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False.
        save_last: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. Default: None.
    """

    save_top_k: int = 1
    save_last: Optional[bool] = None


@attrs.define
class WandBConfig:
    """wandb: (Only if use_wandb is True, else skip this)

    Attributes:
        entity: (str) Entity of wandb project.
        project: (str) Project name for the wandb project.
        name: (str) Name of the current run.
        api_key: (str) API key. The API key is masked when saved to config files.
        wandb_mode: (str) "offline" if only local logging is required. Default: "None".
        prv_runid: (str) Previous run ID if training should be resumed from a previous ckpt. Default: None.
        log_params: (List[str]) List of config parameters to save it in wandb logs. For example, to save learning rate from trainer config section, use "trainer_config.optimizer.lr" (provide the full path to the specific config parameter).
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    api_key: Optional[str] = None
    wandb_mode: Optional[str] = "None"
    prv_runid: Optional[str] = None
    log_params: Optional[List[str]] = None


@attrs.define
class OptimizerConfig:
    """optimizer configuration

    Attributes:
        lr: (float) Learning rate of type float. Default: 1e-3
        amsgrad: (bool) Enable AMSGrad with the optimizer. Default: False
    """

    lr: float = 1e-3
    amsgrad: bool = False


@attrs.define
class LRSchedulerConfig:
    """lr_scheduler configuration

    Attributes:
        mode: (str) One of "min", "max". In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: "min".
        threshold: (float) Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode: (str) One of "rel", "abs". In rel mode, dynamic_threshold = best * ( 1 + threshold ) in max mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: "rel".
        cooldown: (int) Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0
        patience: (int) Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the third epoch if the loss still hasn’t improved then. Default: 10.
        factor: (float) Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        min_lr: (float or List[float]) A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
    """

    mode: str = "min"
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    patience: int = 10
    factor: float = 0.1
    min_lr: Any = 0.0 

    def __post_init__(self):
        self.validate_min_lr()

    def validate_min_lr(self):
        if isinstance(self.min_lr, float):
            return
        if isinstance(self.min_lr, list) and all(isinstance(x, float) for x in self.min_lr):
            return
        raise ValueError("min_lr must be a float or a list of floats.")


@attrs.define
class EarlyStoppingConfig:
    """early_stopping configuration

    Attributes:
        stop_training_on_plateau: (bool) True if early stopping should be enabled.
        min_delta: (float) Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement.
        patience: (int) Number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch.
    """

    stop_training_on_plateau: bool = False
    min_delta: float = 0.0
    patience: int = 1


@attrs.define
class TrainerConfig:
    """Configuration of Trainer.

    Attributes:
        train_data_loader: (Note: Any parameters from Torch's DataLoader could be used.)
        val_data_loader: (Similar to train_data_loader)
        model_ckpt: (Note: Any parameters from Lightning's ModelCheckpoint could be used.)
        trainer_devices: (int) Number of devices to train on (int), which devices to train on (list or str), or "auto" to select automatically.
        trainer_accelerator: (str) One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises the machine the model is running on and chooses the appropriate accelerator for the Trainer to be connected to.
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
    """

    train_data_loader: DataLoaderConfig = DataLoaderConfig()
    val_data_loader: DataLoaderConfig = DataLoaderConfig()
    model_ckpt: ModelCkptConfig = ModelCkptConfig()
    trainer_devices: Any = "auto"
    trainer_accelerator: str = "auto"
    enable_progress_bar: bool = True
    steps_per_epoch: Optional[int] = None
    max_epochs: int = 10
    seed: Optional[int] = None
    use_wandb: bool = False
    save_ckpt: bool = False
    save_ckpt_path: str = "./"
    resume_ckpt_path: Optional[str] = None
    wandb: Optional[WandBConfig] = attrs.field(init=False)
    optimizer: Optional[OptimizerConfig] = OptimizerConfig()
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()

    # post-initialization
    def __attrs_post_init__(self):
        self.validate_trainer_devices()
        # Set wandb configuration only if use_wandb is True
        if self.use_wandb:
            self.wandb = (
                WandBConfig()
            )  # Initialize WandBConfig with defaults or passed parameters
        else:
            self.wandb = None

    def validate_trainer_devices(self):
        """Validate the value of trainer_devices."""
        if isinstance(self.trainer_devices, int) and self.trainer_devices >= 0:
            return
        if isinstance(self.trainer_devices, list) and all(isinstance(x, int) and x >= 0 for x in self.trainer_devices):
            return
        if isinstance(self.trainer_devices, str) and self.trainer_devices == "auto":
            return
        raise ValueError(
            "trainer_devices must be an integer >= 0, a list of integers >= 0, or the string 'auto'."
        )

    @classmethod
    def from_yaml(cls, yaml_data: Text) -> "TrainerConfig":
        """Create TrainerConfig from YAML-formatted string.

        Arguments:
            yaml_data: YAML-formatted string that specifies the configurations.

        Returns:
            A TrainerConfig instance parsed from the YAML text.
        """
        config = OmegaConf.create(yaml_data)
        return OmegaConf.to_object(config, cls)

    @classmethod
    def load_yaml(cls, filename: Text) -> "TrainerConfig":
        """Load a training job configuration from a yaml file.

        Arguments:
            filename: Path to a training job configuration YAML file or a directory
                containing `"training_job.yaml"`.

        Returns:
          A TrainerConfig instance parsed from the YAML file.
        """
        config = OmegaConf.load(filename)
        return OmegaConf.to_object(config, cls)

    def to_yaml(self) -> str:
        """Serialize the configuration into YAML-encoded string format.

        Returns:
            The YAML encoded string representation of the configuration.
        """
        config = self.to_dict()
        return OmegaConf.to_yaml(config)

    def save_yaml(self, filename: Text):
        """Save the configuration to a YAML file.

        Arguments:
            filename: Path to save the training job file to.
        """
        with open(filename, "w") as f:
            f.write(self.to_yaml())
