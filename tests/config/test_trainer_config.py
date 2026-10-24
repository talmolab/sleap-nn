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
    # Check default values (defaults keep the historical val/loss selection)
    conf = OmegaConf.structured(ModelCkptConfig)
    assert conf.save_top_k == 1
    assert conf.save_last is None
    assert conf.monitor == "val/loss"
    assert conf.mode == "min"

    # Test customization (e.g. select best.ckpt on a full-eval quality metric)
    custom_conf = OmegaConf.structured(
        ModelCkptConfig(
            save_top_k=5,
            save_last=True,
            monitor="eval/val/fg_mean_cldice",
            mode="max",
        )
    )
    assert custom_conf.save_top_k == 5
    assert custom_conf.save_last is True
    assert custom_conf.monitor == "eval/val/fg_mean_cldice"
    assert custom_conf.mode == "max"


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
    # `None` is resolved at training time against the LR schedule.
    assert conf.stop_training_on_plateau is None
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
    assert conf_structured.seed == 42
    assert conf_structured.optimizer.lr == 1e-4
    assert conf_structured.lr_scheduler is not None
    assert conf_structured.lr_scheduler.reduce_lr_on_plateau is not None
    assert conf_structured.early_stopping.stop_training_on_plateau is None
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
    assert config.seed == 42
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


def _zmq_legacy_config(zmq_block):
    """Minimal legacy config carrying only the `outputs.zmq` block under test."""
    return {
        "optimization": {},
        "outputs": {"zmq": zmq_block},
    }


def test_trainer_mapper_zmq_disabled():
    """ZMQ off in the legacy config must stay off in the converted config.

    The legacy flags are booleans, so `False` must not be read as "port present".
    """
    config = trainer_mapper(
        _zmq_legacy_config(
            {
                "subscribe_to_controller": False,
                "controller_address": "tcp://127.0.0.1:9000",
                "controller_polling_timeout": 10,
                "publish_updates": False,
                "publish_address": "tcp://127.0.0.1:9001",
            }
        )
    )

    assert config.zmq.controller_port is None
    assert config.zmq.publish_port is None
    assert config.zmq.controller_polling_timeout == 10


def test_trainer_mapper_zmq_enabled():
    """Ports are carried over only for the channels the legacy config enabled."""
    config = trainer_mapper(
        _zmq_legacy_config(
            {
                "subscribe_to_controller": True,
                "controller_address": "tcp://127.0.0.1:9000",
                "controller_polling_timeout": 20,
                "publish_updates": True,
                "publish_address": "tcp://127.0.0.1:9001",
            }
        )
    )

    assert config.zmq.controller_port == 9000
    assert config.zmq.publish_port == 9001
    assert config.zmq.controller_polling_timeout == 20


def test_trainer_mapper_zmq_partially_enabled():
    """Each ZMQ channel is opted into independently."""
    config = trainer_mapper(
        _zmq_legacy_config(
            {
                "subscribe_to_controller": False,
                "controller_address": "tcp://127.0.0.1:9000",
                "publish_updates": True,
                "publish_address": "tcp://127.0.0.1:9001",
            }
        )
    )

    assert config.zmq.controller_port is None
    assert config.zmq.publish_port == 9001


def test_trainer_mapper_zmq_missing_or_empty():
    """A missing/empty `zmq` block leaves ZMQ off rather than raising."""
    for zmq_block in ({}, None):
        config = trainer_mapper(_zmq_legacy_config(zmq_block))
        assert config.zmq.controller_port is None
        assert config.zmq.publish_port is None

    config = trainer_mapper({"optimization": {}, "outputs": {}})
    assert config.zmq.controller_port is None
    assert config.zmq.publish_port is None


def test_trainer_mapper_zmq_enabled_without_address():
    """An enabled flag with no address must not crash; the channel stays off."""
    config = trainer_mapper(
        _zmq_legacy_config({"subscribe_to_controller": True, "publish_updates": True})
    )

    assert config.zmq.controller_port is None
    assert config.zmq.publish_port is None


def test_dataloader_num_workers_accepts_auto():
    """`num_workers` accepts a non-negative int or the string "auto"."""
    assert TrainDataLoaderConfig(num_workers="auto").num_workers == "auto"
    assert ValDataLoaderConfig(num_workers="AUTO").num_workers == "AUTO"
    assert TrainDataLoaderConfig(num_workers=0).num_workers == 0
    assert TrainDataLoaderConfig(num_workers=8).num_workers == 8


@pytest.mark.parametrize("bad", [-1, "many", "", True, 1.5, None])
def test_dataloader_num_workers_rejects_invalid(bad):
    """Anything else is rejected at config construction."""
    with pytest.raises(ValueError):
        TrainDataLoaderConfig(num_workers=bad)


def test_trainer_config_num_workers_auto_survives_omegaconf():
    """`auto` round-trips through the structured OmegaConf schema."""
    cfg = OmegaConf.structured(
        TrainerConfig(train_data_loader=TrainDataLoaderConfig(num_workers="auto"))
    )
    assert cfg.train_data_loader.num_workers == "auto"


def test_recipe_knob_defaults_are_off():
    """weight_decay / per-step schedules / multi-step / EMA default to legacy behavior."""
    from sleap_nn.config.trainer_config import (
        CosineAnnealingWarmupConfig,
        EMAConfig,
        LRSchedulerConfig,
        MultiStepLRConfig,
        OptimizerConfig,
        TrainerConfig,
    )

    assert OptimizerConfig().weight_decay is None
    assert CosineAnnealingWarmupConfig().warmup_steps is None
    assert LRSchedulerConfig().multi_step_lr is None
    cfg = TrainerConfig()
    assert cfg.ema == EMAConfig() and cfg.ema.enabled is False
    msl = MultiStepLRConfig(milestones=[170, 200])
    assert (msl.gamma, msl.milestone_unit, msl.warmup_steps) == (0.1, "epoch", 0)
    assert msl.warmup_start_factor == 0.001
    # Round-trips through OmegaConf like every saved training_config.yaml.
    oc = OmegaConf.structured(
        TrainerConfig(
            optimizer=OptimizerConfig(weight_decay=0.1),
            lr_scheduler=LRSchedulerConfig(multi_step_lr=msl),
            ema=EMAConfig(enabled=True),
        )
    )
    assert oc.optimizer.weight_decay == 0.1
    assert list(oc.lr_scheduler.multi_step_lr.milestones) == [170, 200]
    assert oc.ema.enabled is True


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        ("OptimizerConfig", {"weight_decay": -0.1}),
        ("MultiStepLRConfig", {"milestone_unit": "iters"}),
        ("MultiStepLRConfig", {"warmup_start_factor": 0.0}),
        ("CosineAnnealingWarmupConfig", {"warmup_steps": -1}),
        ("EMAConfig", {"decay": 1.0}),
        ("EMAConfig", {"warmup_steps": -5}),
        ("OptimizerConfig", {"encoder_lr_scale": 0.0}),
        ("OptimizerConfig", {"layer_decay": 1.5}),
        ("MultiStepLRConfig", {"milestones": [200, 170]}),
        ("MultiStepLRConfig", {"milestones": [170, 170]}),
        ("MultiStepLRConfig", {"milestones": [-1, 5]}),
        ("MultiStepLRConfig", {"milestones": [1.5]}),
    ],
)
def test_recipe_knob_validation(factory, kwargs):
    import sleap_nn.config.trainer_config as tc

    with pytest.raises((ValueError, TypeError)):
        getattr(tc, factory)(**kwargs)


@pytest.mark.parametrize(
    "lr_scheduler, expected",
    [
        (None, None),
        ({"reduce_lr_on_plateau": {}}, None),
        ({"step_lr": {}, "reduce_lr_on_plateau": {}}, None),
        ({"multi_step_lr": {"milestones": [1]}}, "multi_step_lr"),
        (
            {"linear_warmup_linear_decay": {}, "multi_step_lr": {"milestones": [1]}},
            "linear_warmup_linear_decay",
        ),
        (
            {"cosine_annealing_warmup": {}, "linear_warmup_linear_decay": {}},
            "cosine_annealing_warmup",
        ),
    ],
)
def test_fixed_length_lr_schedule(lr_scheduler, expected):
    """Same priority as `configure_optimizers`, on dicts, DictConfigs and attrs."""
    from sleap_nn.config.trainer_config import (
        LRSchedulerConfig,
        fixed_length_lr_schedule,
    )

    assert fixed_length_lr_schedule(lr_scheduler) == expected
    if lr_scheduler is not None:
        assert fixed_length_lr_schedule(OmegaConf.create(lr_scheduler)) == expected
    assert fixed_length_lr_schedule(LRSchedulerConfig()) is None


def test_get_trainer_config_plumbs_recipe_keys():
    from sleap_nn.config.get_config import get_trainer_config

    tc = get_trainer_config(
        optimizer="AdamW",
        weight_decay=0.1,
        encoder_lr_scale=0.5,
        layer_decay=0.75,
        ema=True,
        ema_decay=0.999,
        ema_warmup_steps=100,
        early_stopping=None,
        lr_scheduler={"multi_step_lr": {"milestones": [170, 200], "warmup_steps": 500}},
    )
    assert (
        tc.optimizer.weight_decay,
        tc.optimizer.encoder_lr_scale,
        tc.optimizer.layer_decay,
    ) == (0.1, 0.5, 0.75)
    assert (tc.ema.enabled, tc.ema.decay, tc.ema.warmup_steps) == (True, 0.999, 100)
    assert tc.lr_scheduler.multi_step_lr.milestones == [170, 200]
    assert tc.lr_scheduler.multi_step_lr.warmup_steps == 500
    assert tc.early_stopping.stop_training_on_plateau is None

    defaults = get_trainer_config()
    assert defaults.optimizer.weight_decay is None
    assert defaults.ema.enabled is False
    assert defaults.early_stopping.stop_training_on_plateau is False


@pytest.mark.parametrize(
    "name", ["cosine_annealing_warmup", "linear_warmup_linear_decay"]
)
def test_get_trainer_config_accepts_fixed_schedule_names(name):
    from sleap_nn.config.get_config import get_trainer_config

    tc = get_trainer_config(lr_scheduler=name)
    assert getattr(tc.lr_scheduler, name) is not None
    # A dict form used to be dropped silently, leaving the plateau default.
    tc = get_trainer_config(lr_scheduler={name: {"warmup_epochs": 2}})
    assert getattr(tc.lr_scheduler, name).warmup_epochs == 2


def test_get_trainer_config_multi_step_lr_needs_milestones():
    from sleap_nn.config.get_config import get_trainer_config

    with pytest.raises(ValueError, match="milestones"):
        get_trainer_config(lr_scheduler="multi_step_lr")


def test_train_api_and_get_trainer_config_take_the_same_recipe_keys():
    import inspect

    from sleap_nn.config.get_config import get_trainer_config
    from sleap_nn.train import train

    keys = [
        "weight_decay",
        "encoder_lr_scale",
        "layer_decay",
        "ema",
        "ema_decay",
        "ema_warmup_steps",
        "early_stopping",
    ]
    train_params = inspect.signature(train).parameters
    config_params = inspect.signature(get_trainer_config).parameters
    for key in keys:
        assert train_params[key].default == config_params[key].default, key


def test_hard_keypoint_mining_per_sample_default():
    from sleap_nn.config.trainer_config import HardKeypointMiningConfig

    assert HardKeypointMiningConfig().per_sample is True
    assert HardKeypointMiningConfig(per_sample=False).per_sample is False
