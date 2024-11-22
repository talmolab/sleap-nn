import pytest
from omegaconf import OmegaConf
from sleap_nn.config.trainer_config import (
    DataLoaderConfig,
    ModelCkptConfig,
    WandBConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    EarlyStoppingConfig,
    TrainerConfig,
)

def test_dataloader_config():
    # Check default values
    conf = OmegaConf.structured(DataLoaderConfig)
    conf_instance = OmegaConf.structured(DataLoaderConfig())
    assert conf == conf_instance
    assert conf.batch_size == 1
    assert conf.shuffle is False
    assert conf.num_workers == 0

    # Test customization
    custom_conf = OmegaConf.structured(DataLoaderConfig(batch_size=16, shuffle=True, num_workers=4))
    assert custom_conf.batch_size == 16
    assert custom_conf.shuffle is True
    assert custom_conf.num_workers == 4

def test_model_ckpt_config():
    # Check default values
    conf = OmegaConf.structured(ModelCkptConfig)
    assert conf.save_top_k == 1
    assert conf.save_last is None

    # Test customization
    custom_conf = OmegaConf.structured(ModelCkptConfig(save_top_k=5, save_last=True))
    assert custom_conf.save_top_k == 5
    assert custom_conf.save_last is True

def test_wandb_config():
    # Check default values
    conf = OmegaConf.structured(WandBConfig)
    assert conf.entity is None
    assert conf.project is None
    assert conf.wandb_mode == "None"

    # Test customization
    custom_conf = OmegaConf.structured(WandBConfig(entity="test_entity", project="test_project"))
    assert custom_conf.entity == "test_entity"
    assert custom_conf.project == "test_project"

def test_optimizer_config():
    # Check default values
    conf = OmegaConf.structured(OptimizerConfig)
    assert conf.lr == 1e-3
    assert conf.amsgrad is False

    # Test customization
    custom_conf = OmegaConf.structured(OptimizerConfig(lr=0.01, amsgrad=True))
    assert custom_conf.lr == 0.01
    assert custom_conf.amsgrad is True

def test_lr_scheduler_config():
    # Check default values
    conf = OmegaConf.structured(LRSchedulerConfig)
    assert conf.mode == "min"
    assert conf.threshold == 1e-4
    assert conf.patience == 10

    # Test customization
    custom_conf = OmegaConf.structured(LRSchedulerConfig(mode="max", patience=5, factor=0.5))
    assert custom_conf.mode == "max"
    assert custom_conf.patience == 5
    assert custom_conf.factor == 0.5


def test_early_stopping_config():
    # Check default values
    conf = OmegaConf.structured(EarlyStoppingConfig)
    assert conf.stop_training_on_plateau is False
    assert conf.min_delta == 0.0
    assert conf.patience == 1

    # Test customization
    custom_conf = OmegaConf.structured(EarlyStoppingConfig(stop_training_on_plateau=True, patience=5))
    assert custom_conf.stop_training_on_plateau is True
    assert custom_conf.patience == 5


def test_trainer_config():
    # Check default values
    conf = OmegaConf.structured(TrainerConfig)
    assert conf.train_data_loader.batch_size == 1
    assert conf.val_data_loader.shuffle is False
    assert conf.model_ckpt.save_top_k == 1
    assert conf.optimizer.lr == 1e-3
    assert conf.lr_scheduler.mode == "min"
    assert conf.early_stopping.patience == 1
    assert conf.use_wandb is False
    assert conf.save_ckpt_path == "./"

    # Test customization
    custom_conf = OmegaConf.structured(
        TrainerConfig(
            max_epochs=20,
            train_data_loader=DataLoaderConfig(batch_size=32),
            optimizer=OptimizerConfig(lr=0.01),
            use_wandb=True,
        )
    )
    assert custom_conf.max_epochs == 20
    assert custom_conf.train_data_loader.batch_size == 32
    assert custom_conf.optimizer.lr == 0.01
    assert custom_conf.use_wandb is True
