"""Tests for the serializable configuration classes for specifying all trainer config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the trainer config.
"""

import pytest
from omegaconf import OmegaConf
from omegaconf import ValidationError
from attrs import asdict
from sleap_nn.config.trainer_config import (
    DataLoaderConfig,
    ModelCkptConfig,
    WandBConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    StepLRConfig,
    ReduceLROnPlateauConfig,
    EarlyStoppingConfig,
    TrainerConfig,
)


def test_reduce_lr_on_plateau_config():
    """reduce_lr_on_plateau_config tests.

    Check default values and validators
    """
    # Check default values
    conf = OmegaConf.structured(ReduceLROnPlateauConfig)
    assert conf.factor == 0.1
    assert conf.patience == 10
    assert conf.threshold == 0.0001
    assert conf.threshold_mode == "rel"
    assert conf.cooldown == 0
    assert conf.min_lr == 0

    # Test validators
    with pytest.raises(ValueError):
        OmegaConf.structured(ReduceLROnPlateauConfig(min_lr=-0.1))


def test_dataloader_config():
    """dataloader_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(DataLoaderConfig)
    conf_instance = OmegaConf.structured(DataLoaderConfig())
    assert conf == conf_instance
    assert conf.batch_size == 1
    assert conf.shuffle is False
    assert conf.num_workers == 0

    # Test customization
    custom_conf = OmegaConf.structured(
        DataLoaderConfig(batch_size=16, shuffle=True, num_workers=4)
    )
    assert custom_conf.batch_size == 16
    assert custom_conf.shuffle is True
    assert custom_conf.num_workers == 4


def test_model_ckpt_config():
    """model_ckpt_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(ModelCkptConfig)
    assert conf.save_top_k == 1
    assert conf.save_last is None

    # Test customization
    custom_conf = OmegaConf.structured(ModelCkptConfig(save_top_k=5, save_last=True))
    assert custom_conf.save_top_k == 5
    assert custom_conf.save_last is True


def test_wandb_config():
    """wandb_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(WandBConfig)
    assert conf.entity is None
    assert conf.project is None
    assert conf.wandb_mode == None

    # Test customization
    custom_conf = OmegaConf.structured(
        WandBConfig(entity="test_entity", project="test_project")
    )
    assert custom_conf.entity == "test_entity"
    assert custom_conf.project == "test_project"


def test_optimizer_config():
    """optimizer_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(OptimizerConfig)
    assert conf.lr == 1e-3
    assert conf.amsgrad is False

    # Test customization
    custom_conf = OmegaConf.structured(OptimizerConfig(lr=0.01, amsgrad=True))
    assert custom_conf.lr == 0.01
    assert custom_conf.amsgrad is True

    # Test validation
    with pytest.raises(ValueError):
        OmegaConf.structured(OptimizerConfig(lr=-0.01, amsgrad=True))


def test_lr_scheduler_config():
    """lr_scheduler_config tests.

    Check default values and customization
    """
    # Check default values
    conf = TrainerConfig(
        lr_scheduler=LRSchedulerConfig(
            scheduler="ReduceLROnPlateau",
            reduce_lr_on_plateau=ReduceLROnPlateauConfig(),
        )
    )
    conf_dict = asdict(conf)  # Convert to dict for OmegaConf
    conf_structured = OmegaConf.create(conf_dict)

    # Test ReduceLROnPlateau
    assert conf_structured.lr_scheduler.scheduler == "ReduceLROnPlateau"
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.threshold == 1e-4
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.patience == 10
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.factor == 0.1

    # Test StepLR configuration
    custom_conf = TrainerConfig(
        lr_scheduler=LRSchedulerConfig(
            scheduler="StepLR", step_lr=StepLRConfig(step_size=5, gamma=0.5)
        )
    )
    custom_dict = asdict(custom_conf)  # Convert to dict for OmegaConf
    custom_structured = OmegaConf.create(custom_dict)

    assert custom_structured.lr_scheduler.scheduler == "StepLR"
    assert custom_structured.lr_scheduler.step_lr.step_size == 5
    assert custom_structured.lr_scheduler.step_lr.gamma == 0.5

    # Test validation
    with pytest.raises(ValueError, match="scheduler must be one of"):
        LRSchedulerConfig(scheduler="InvalidScheduler")
    with pytest.raises(ValueError):
        LRSchedulerConfig(
            scheduler="StepLR", step_lr=StepLRConfig(step_size=-5, gamma=0.5)
        )


def test_early_stopping_config():
    """early_stopping_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(EarlyStoppingConfig)
    assert conf.stop_training_on_plateau is False
    assert conf.min_delta == 0.0
    assert conf.patience == 1

    # Test customization
    custom_conf = OmegaConf.structured(
        EarlyStoppingConfig(stop_training_on_plateau=True, patience=5)
    )
    assert custom_conf.stop_training_on_plateau is True
    assert custom_conf.patience == 5

    # Test validation
    with pytest.raises(ValueError):
        OmegaConf.structured(EarlyStoppingConfig(patience=-5))
    with pytest.raises(ValueError):
        OmegaConf.structured(EarlyStoppingConfig(min_delta=-5))


def test_trainer_config():
    """trainer_config tests.

    Check default values and customization
    """
    # Check default values by creating an instance first
    conf = TrainerConfig()
    conf_dict = asdict(conf)  # Convert to dict for OmegaConf
    conf_structured = OmegaConf.create(conf_dict)

    assert conf_structured.train_data_loader.batch_size == 1
    assert conf_structured.val_data_loader.shuffle is False
    assert conf_structured.model_ckpt.save_top_k == 1
    assert conf_structured.optimizer.lr == 1e-3
    assert conf_structured.lr_scheduler is None
    assert conf_structured.early_stopping is None
    assert conf_structured.use_wandb is False
    assert conf_structured.save_ckpt_path is None

    # Test customization
    custom_conf = TrainerConfig(
        max_epochs=20,
        train_data_loader=DataLoaderConfig(batch_size=32),
        val_data_loader=DataLoaderConfig(batch_size=32),
        optimizer=OptimizerConfig(lr=0.01),
        use_wandb=True,
    )
    custom_dict = asdict(custom_conf)  # Convert to dict for OmegaConf
    custom_structured = OmegaConf.structured(custom_dict)

    assert custom_structured.max_epochs == 20
    assert custom_structured.train_data_loader.batch_size == 32
    assert custom_structured.optimizer.lr == 0.01
    assert custom_structured.use_wandb is True

    # Test validation
    with pytest.raises(ValueError):
        TrainerConfig(optimizer_name="InvalidOptimizer")
    with pytest.raises(ValueError, match="trainer_devices"):
        TrainerConfig(trainer_devices=-1)
