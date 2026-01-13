"""Tests for the serializable configuration classes for specifying all trainer config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the trainer config.
"""

import pytest
from omegaconf import OmegaConf
from attrs import asdict
from loguru import logger
from _pytest.logging import LogCaptureFixture

from sleap_nn.config.trainer_config import (
    TrainDataLoaderConfig,
    ValDataLoaderConfig,
    ModelCkptConfig,
    WandBConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    StepLRConfig,
    ReduceLROnPlateauConfig,
    EarlyStoppingConfig,
    TrainerConfig,
    trainer_mapper,
)


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def test_reduce_lr_on_plateau_config(caplog):
    """reduce_lr_on_plateau_config tests.

    Check default values and validators
    """
    # Check default values
    conf = OmegaConf.structured(ReduceLROnPlateauConfig)
    assert conf.factor == 0.5
    assert conf.patience == 5
    assert conf.threshold == 1e-6
    assert conf.threshold_mode == "abs"
    assert conf.cooldown == 3
    assert conf.min_lr == 1e-8

    # Test validators
    with pytest.raises(ValueError):
        OmegaConf.structured(ReduceLROnPlateauConfig(min_lr=-0.1))
    assert "min_lr" in caplog.text


def test_dataloader_config():
    """dataloader_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(TrainDataLoaderConfig)
    conf_instance = OmegaConf.structured(TrainDataLoaderConfig())
    assert conf == conf_instance
    assert conf.batch_size == 4
    assert conf.shuffle is True
    assert conf.num_workers == 0

    # Check default values
    conf = OmegaConf.structured(ValDataLoaderConfig)
    conf_instance = OmegaConf.structured(ValDataLoaderConfig())
    assert conf == conf_instance
    assert conf.batch_size == 4
    assert conf.shuffle is False
    assert conf.num_workers == 0

    # Test customization
    custom_conf = OmegaConf.structured(
        TrainDataLoaderConfig(batch_size=16, shuffle=True, num_workers=4)
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


def test_wandb_config_delete_local_logs():
    """Test delete_local_logs field in WandBConfig.

    Verifies default value and auto-detection logic.
    """
    # Check default value is None
    conf = OmegaConf.structured(WandBConfig)
    assert conf.delete_local_logs is None

    # Test explicit True
    custom_conf = OmegaConf.structured(WandBConfig(delete_local_logs=True))
    assert custom_conf.delete_local_logs is True

    # Test explicit False
    custom_conf = OmegaConf.structured(WandBConfig(delete_local_logs=False))
    assert custom_conf.delete_local_logs is False

    # Test auto-detection logic: online mode (wandb_mode=None) should delete
    config = WandBConfig(wandb_mode=None)
    should_delete = config.delete_local_logs is True or (
        config.delete_local_logs is None and config.wandb_mode != "offline"
    )
    assert should_delete is True

    # Test auto-detection logic: offline mode should keep
    config = WandBConfig(wandb_mode="offline")
    should_delete = config.delete_local_logs is True or (
        config.delete_local_logs is None and config.wandb_mode != "offline"
    )
    assert should_delete is False

    # Test explicit True overrides offline mode
    config = WandBConfig(wandb_mode="offline", delete_local_logs=True)
    should_delete = config.delete_local_logs is True or (
        config.delete_local_logs is None and config.wandb_mode != "offline"
    )
    assert should_delete is True

    # Test explicit False overrides online mode
    config = WandBConfig(wandb_mode=None, delete_local_logs=False)
    should_delete = config.delete_local_logs is True or (
        config.delete_local_logs is None and config.wandb_mode != "offline"
    )
    assert should_delete is False


def test_optimizer_config(caplog):
    """optimizer_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(OptimizerConfig)
    assert conf.lr == 1e-4
    assert conf.amsgrad is False

    # Test customization
    custom_conf = OmegaConf.structured(OptimizerConfig(lr=0.01, amsgrad=True))
    assert custom_conf.lr == 0.01
    assert custom_conf.amsgrad is True

    # Test validation
    with pytest.raises(ValueError):
        OmegaConf.structured(OptimizerConfig(lr=-0.01, amsgrad=True))


def test_lr_scheduler_config(caplog):
    """lr_scheduler_config tests.

    Check default values and customization
    """
    # Check default values
    conf = TrainerConfig(
        lr_scheduler=LRSchedulerConfig(
            reduce_lr_on_plateau=ReduceLROnPlateauConfig(),
        )
    )
    conf_dict = asdict(conf)  # Convert to dict for OmegaConf
    conf_structured = OmegaConf.create(conf_dict)

    # Test ReduceLROnPlateau
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.threshold == 1e-6
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.patience == 5
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau.factor == 0.5

    # Test StepLR configuration
    custom_conf = TrainerConfig(
        lr_scheduler=LRSchedulerConfig(step_lr=StepLRConfig(step_size=5, gamma=0.5))
    )
    custom_dict = asdict(custom_conf)  # Convert to dict for OmegaConf
    custom_structured = OmegaConf.create(custom_dict)

    assert custom_structured.lr_scheduler.step_lr.step_size == 5
    assert custom_structured.lr_scheduler.step_lr.gamma == 0.5

    with pytest.raises(ValueError):
        LRSchedulerConfig(step_lr=StepLRConfig(step_size=-5, gamma=0.5))


def test_early_stopping_config():
    """early_stopping_config tests.

    Check default values and customization
    """
    # Check default values
    conf = OmegaConf.structured(EarlyStoppingConfig)
    assert conf.stop_training_on_plateau is True
    assert conf.min_delta == 1e-8
    assert conf.patience == 10

    # Test customization
    custom_conf = OmegaConf.structured(
        EarlyStoppingConfig(stop_training_on_plateau=False, patience=5)
    )
    assert custom_conf.stop_training_on_plateau is False
    assert custom_conf.patience == 5

    # Test validation
    with pytest.raises(ValueError):
        OmegaConf.structured(EarlyStoppingConfig(patience=-5))
    with pytest.raises(ValueError):
        OmegaConf.structured(EarlyStoppingConfig(min_delta=-5))


def test_trainer_config(caplog):
    """trainer_config tests.

    Check default values and customization
    """
    # Check default values by creating an instance first
    conf = TrainerConfig()
    conf_dict = asdict(conf)  # Convert to dict for OmegaConf
    conf_structured = OmegaConf.create(conf_dict)

    assert conf_structured.train_data_loader.batch_size == 4
    assert conf_structured.val_data_loader.batch_size == 4
    assert conf_structured.val_data_loader.shuffle is False
    assert conf_structured.model_ckpt.save_top_k == 1
    assert conf_structured.max_epochs == 100
    assert conf_structured.seed is None
    assert conf_structured.optimizer.lr == 1e-4
    assert conf_structured.lr_scheduler is not None
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau is not None
    assert conf_structured.early_stopping.stop_training_on_plateau is True
    assert conf_structured.use_wandb is False
    assert conf_structured.ckpt_dir == "."
    assert conf_structured.run_name is None

    # Test customization
    custom_conf = TrainerConfig(
        max_epochs=20,
        train_data_loader=TrainDataLoaderConfig(batch_size=32),
        val_data_loader=ValDataLoaderConfig(batch_size=32),
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
    assert "optimizer_name" in caplog.text
    with pytest.raises(ValueError, match="trainer_devices"):
        TrainerConfig(trainer_devices=-1)
    assert "trainer_devices" in caplog.text


def test_trainer_mapper():
    """Test the trainer_mapper function with a sample legacy configuration."""
    legacy_config = {
        "optimization": {
            "batch_size": 32,
            "online_shuffling": True,
            "epochs": 20,
            "learning_rate_schedule": {
                "plateau_patience": 5,
                "min_learning_rate": 0.0001,
                "reduce_on_plateau": True,
                "plateau_min_delta": 0.01,
            },
            "optimizer": "Adam",
            "initial_learning_rate": 0.001,
            "checkpointing": {},
            "early_stopping": {
                "stop_training_on_plateau": True,
                "plateau_min_delta": 1e-06,
                "plateau_patience": 10,
            },
            "hard_keypoint_mining": {
                "online_mining": False,
                "hard_to_easy_ratio": 2.0,
                "min_hard_keypoints": 2,
                "max_hard_keypoints": None,
                "loss_scale": 5.0,
            },
        },
        "outputs": {"save_outputs": True, "save_visualizations": True, "zmq": {}},
    }

    config = trainer_mapper(legacy_config)

    # Assertions to check if the output matches expected values
    assert config.train_data_loader.batch_size == 32
    assert config.train_data_loader.shuffle is True
    assert config.train_data_loader.num_workers == 0
    assert config.max_epochs == 20
    assert config.optimizer_name == "Adam"
    assert config.optimizer.lr == 0.001
    assert config.model_ckpt.save_last is None
    assert config.visualize_preds_during_training is True

    # Test for default values (unspecified by legacy config)
    assert config.trainer_devices is None
    assert config.trainer_device_indices is None
    assert config.trainer_accelerator == "auto"
    assert config.enable_progress_bar is True
    assert config.min_train_steps_per_epoch == 200
    assert config.seed is None
    assert config.use_wandb is False
    assert config.save_ckpt is True
    assert config.resume_ckpt_path is None
    assert config.wandb.entity is None
    assert config.wandb.project is None
    assert config.wandb.name is None
    assert config.wandb.api_key is None
    assert config.wandb.wandb_mode is None
    assert config.wandb.prv_runid is None
    assert config.wandb.group is None
    assert config.lr_scheduler is not None
    assert config.lr_scheduler.reduce_lr_on_plateau is not None
    assert config.lr_scheduler.reduce_lr_on_plateau.patience == 5
    assert config.lr_scheduler.reduce_lr_on_plateau.min_lr == 0.0001
    assert config.early_stopping is not None
    assert config.early_stopping.patience == 10
    assert config.early_stopping.min_delta == 1e-6
    assert config.early_stopping.stop_training_on_plateau is True
    assert config.online_hard_keypoint_mining.online_mining is False
    assert config.online_hard_keypoint_mining.hard_to_easy_ratio == 2.0
    assert config.online_hard_keypoint_mining.min_hard_keypoints == 2
    assert config.online_hard_keypoint_mining.max_hard_keypoints is None
    assert config.online_hard_keypoint_mining.loss_scale == 5.0
