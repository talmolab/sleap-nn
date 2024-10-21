from omegaconf import OmegaConf

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
    """
    train_data_loader: DataLoaderConfig = attrs.field(factory=DataLoaderConfig)
    val_data_loader: DataLoaderConfig = attrs.field(factory=DataLoaderConfig)
    model_ckpt: ModelCkptConfig = attrs.field(factory=ModelCkptConfig)
    trainer_devices: Union[int, List[int], str] = "auto"
    trainer_accelerator: str="auto"
    enable_progress_bar: bool = True
    steps_per_epoch: Optional[int] = None
    max_epochs: int = 10
    seed: Optional[int] = None
    use_wandb: bool = False
    save_ckpt: bool = False
    save_ckpt_path: str = "./"
    resume_ckpt_path: Optional[str] = None
    wandb: 


@attrs.define
class DataLoaderConfig:
    '''train and val DataLoaderConfig: (Note: Any parameters from Torch's DataLoader could be used.)

    Attributes:
        batch_size: (int) Number of samples per batch or batch size for training/validation data. Default = 1.
        shuffle: (bool) True to have the data reshuffled at every epoch. Default: False.
        num_workers: (int) Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default: 0.
    '''
    batch_size: int = 1
    shuffle: bool=False
    num_workers: int=0

@attrs.define 
class ModelCkptConfig:
    '''modelCkptConfig: (Note: Any parameters from Lightning's ModelCheckpoint could be used.)
    
    Attributes:
        save_top_k: (int) If save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved. Please note that the monitors are checked every every_n_epochs epochs. if save_top_k >= 2 and the callback is called multiple times inside an epoch, the name of the saved file will be appended with a version count starting with v1 unless enable_version_counter is set to False.
        save_last: (bool) When True, saves a last.ckpt whenever a checkpoint file gets saved. On a local filesystem, this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing the latest checkpoint in a deterministic manner. Default: None.
    '''
    save_top_k: int = 1
    save_last: Optional[bool]=None
