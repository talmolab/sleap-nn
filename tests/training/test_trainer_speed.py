"""Tests for the trainer speed/precision knobs and the ThroughputMonitor callback."""

import csv
import inspect
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_trainer_config
from sleap_nn.config.trainer_config import (
    TrainDataLoaderConfig,
    TrainerConfig,
    ValDataLoaderConfig,
)
from sleap_nn.data.custom_datasets import (
    DEFAULT_PREFETCH_FACTOR,
    _get_prefetch_factor,
    _get_train_drop_last,
)
from sleap_nn.training.callbacks import ThroughputMonitor, _batch_num_samples
from sleap_nn.training.model_trainer import ModelTrainer

#: The new `trainer_config` keys and the value each defaults to.
NEW_TRAINER_KEYS = {
    "precision": "32",
    "float32_matmul_precision": None,
    "compile": False,
    "gradient_clip_val": None,
    "gradient_clip_algorithm": "norm",
    "accumulate_grad_batches": 1,
    "log_every_n_steps": 1,
    "max_steps": None,
    "val_check_interval": None,
    "throughput_log_every_n_steps": 50,
    "confmap_fg_bg_diagnostics": None,
}


def _cfg(**trainer):
    base = {
        "train_data_loader": {"batch_size": 4, "num_workers": 2},
        "val_data_loader": {"batch_size": 4, "num_workers": 0},
    }
    base.update(trainer)
    return OmegaConf.create({"trainer_config": base})


def _speed_kwargs(cfg):
    return ModelTrainer._get_trainer_speed_kwargs(SimpleNamespace(config=cfg))


def _cpu_trainer(**kwargs):
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_schema_defaults():
    """The new keys default to the pre-0.4 behaviour."""
    cfg = TrainerConfig()
    for key, default in NEW_TRAINER_KEYS.items():
        assert getattr(cfg, key) == default, key
    assert cfg.train_data_loader.prefetch_factor == 2
    assert cfg.train_data_loader.pin_memory is None
    assert cfg.train_data_loader.drop_last is None
    assert cfg.val_data_loader.prefetch_factor == 2
    assert not hasattr(cfg.val_data_loader, "drop_last")


def test_speed_kwargs_defaults_match_previous_behaviour():
    """A config without the new keys gives the pre-0.4 Trainer settings."""
    assert _speed_kwargs(_cfg()) == {
        "precision": "32",
        "log_every_n_steps": 1,
        "accumulate_grad_batches": 1,
    }


def test_speed_kwargs_schema_defaults_match_missing_keys():
    """Schema defaults and an old config without the keys give the same kwargs."""
    schema = OmegaConf.structured(TrainerConfig())
    assert _speed_kwargs(OmegaConf.create({"trainer_config": schema})) == (
        _speed_kwargs(_cfg())
    )


def test_default_kwargs_build_a_default_trainer():
    """The default kwargs leave every Trainer attribute at Lightning's default."""
    ours = _cpu_trainer(**_speed_kwargs(_cfg()))
    stock = _cpu_trainer(log_every_n_steps=1)
    for attr in (
        "precision",
        "accumulate_grad_batches",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "max_steps",
        "val_check_interval",
        "check_val_every_n_epoch",
        "log_every_n_steps",
    ):
        assert getattr(ours, attr) == getattr(stock, attr), attr
    assert ours.precision == "32-true"


def test_speed_kwargs_forwarded():
    cfg = _cfg(
        precision="bf16-mixed",
        log_every_n_steps=50,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        max_steps=1000,
        val_check_interval=250,
    )
    kwargs = _speed_kwargs(cfg)
    assert kwargs == {
        "precision": "bf16-mixed",
        "log_every_n_steps": 50,
        "accumulate_grad_batches": 2,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "value",
        "max_steps": 1000,
        "val_check_interval": 250,
        "check_val_every_n_epoch": None,
    }


def test_every_key_reaches_lightning_trainer():
    """Each kwarg is accepted by `L.Trainer` and lands on the matching attribute."""
    trainer = _cpu_trainer(
        **_speed_kwargs(
            _cfg(
                precision="bf16-mixed",
                log_every_n_steps=25,
                accumulate_grad_batches=4,
                gradient_clip_val=0.5,
                max_steps=123,
                val_check_interval=1000,
            )
        )
    )
    assert trainer.precision == "bf16-mixed"
    assert trainer.log_every_n_steps == 25
    assert trainer.accumulate_grad_batches == 4
    assert trainer.gradient_clip_val == 0.5
    assert trainer.gradient_clip_algorithm == "norm"
    assert trainer.max_steps == 123
    assert trainer.val_check_interval == 1000
    # An int interval counts batches across epochs (it may exceed an epoch).
    assert trainer.check_val_every_n_epoch is None


def test_float_val_check_interval_keeps_epoch_cadence():
    kwargs = _speed_kwargs(_cfg(val_check_interval=0.25))
    assert kwargs["val_check_interval"] == 0.25
    assert "check_val_every_n_epoch" not in kwargs
    assert _cpu_trainer(**kwargs).check_val_every_n_epoch == 1


@pytest.mark.parametrize("value", ["32", 32, "bf16-mixed", "16-mixed", "32-true"])
def test_precision_validator_accepts(value):
    TrainerConfig(precision=value)


@pytest.mark.parametrize("value", ["bf16", "64", "fp32", 16])
def test_precision_validator_rejects(value):
    with pytest.raises(ValueError):
        TrainerConfig(precision=value)


def test_compile_validator():
    TrainerConfig(compile=True)
    TrainerConfig(compile="max-autotune")
    with pytest.raises(ValueError):
        TrainerConfig(compile="fast")


@pytest.mark.parametrize("value", [1, 500, 0.25, 1.0])
def test_val_check_interval_validator_accepts(value):
    TrainerConfig(val_check_interval=value)


@pytest.mark.parametrize("value", [0, -1, 0.0, 1.5, True, "epoch"])
def test_val_check_interval_validator_rejects(value):
    with pytest.raises(ValueError):
        TrainerConfig(val_check_interval=value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("float32_matmul_precision", "tf32"),
        ("gradient_clip_val", 0.0),
        ("gradient_clip_algorithm", "l2"),
        ("accumulate_grad_batches", 0),
        ("log_every_n_steps", 0),
        ("max_steps", 0),
        ("throughput_log_every_n_steps", -1),
    ],
)
def test_validators_reject(key, value):
    with pytest.raises((ValueError, TypeError)):
        TrainerConfig(**{key: value})


def test_dataloader_keys_validated():
    with pytest.raises(ValueError):
        TrainDataLoaderConfig(prefetch_factor=0)
    ValDataLoaderConfig(prefetch_factor=None, pin_memory=False)


def test_yaml_round_trip_through_schema():
    """The new keys survive the structured-config merge used to load YAML."""
    from sleap_nn.config.training_job_config import TrainingJobConfig

    schema = OmegaConf.structured(TrainingJobConfig())
    user = OmegaConf.create(
        {
            "trainer_config": {
                "precision": "bf16-mixed",
                "float32_matmul_precision": "high",
                "compile": "max-autotune",
                "val_check_interval": 0.5,
                "train_data_loader": {
                    "prefetch_factor": 4,
                    "pin_memory": False,
                    "drop_last": True,
                },
            }
        }
    )
    merged = OmegaConf.merge(schema, user)
    obj = OmegaConf.to_object(merged)
    assert obj.trainer_config.precision == "bf16-mixed"
    assert obj.trainer_config.compile == "max-autotune"
    assert obj.trainer_config.val_check_interval == 0.5
    assert obj.trainer_config.train_data_loader.prefetch_factor == 4
    assert obj.trainer_config.train_data_loader.drop_last is True


def test_get_trainer_config_plumbs_new_keys():
    tc = get_trainer_config(
        precision="bf16-mixed",
        float32_matmul_precision="high",
        compile=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=2,
        log_every_n_steps=50,
        max_steps=10,
        val_check_interval=5,
        throughput_log_every_n_steps=0,
        confmap_fg_bg_diagnostics=True,
        prefetch_factor=3,
        pin_memory=False,
        drop_last=True,
    )
    assert tc.precision == "bf16-mixed"
    assert tc.float32_matmul_precision == "high"
    assert tc.compile is True
    assert tc.gradient_clip_val == 1.0
    assert tc.gradient_clip_algorithm == "value"
    assert tc.accumulate_grad_batches == 2
    assert tc.log_every_n_steps == 50
    assert tc.max_steps == 10
    assert tc.val_check_interval == 5
    assert tc.throughput_log_every_n_steps == 0
    assert tc.confmap_fg_bg_diagnostics is True
    for loader in (tc.train_data_loader, tc.val_data_loader):
        assert loader.prefetch_factor == 3
        assert loader.pin_memory is False
    assert tc.train_data_loader.drop_last is True


def test_get_trainer_config_defaults_match_schema():
    tc = get_trainer_config()
    for key, default in NEW_TRAINER_KEYS.items():
        assert getattr(tc, key) == default, key
    assert tc.train_data_loader.prefetch_factor == 2
    assert tc.train_data_loader.drop_last is None


def test_train_api_exposes_and_forwards_new_keys(monkeypatch):
    """`sleap_nn.train.train()` takes every new key and puts it in the config."""
    import sleap_nn.train as train_module

    params = inspect.signature(train_module.train).parameters
    gtc_params = inspect.signature(get_trainer_config).parameters
    new = set(NEW_TRAINER_KEYS) | {"prefetch_factor", "pin_memory", "drop_last"}
    for key in new:
        assert key in params, key
        assert params[key].default == gtc_params[key].default, key

    captured = {}
    monkeypatch.setattr(
        train_module, "run_training", lambda cfg: captured.setdefault("cfg", cfg)
    )
    train_module.train(
        train_labels_path=["dummy.slp"],
        head_configs="centered_instance",
        precision="bf16-mixed",
        max_steps=7,
        val_check_interval=0.5,
        prefetch_factor=6,
    )
    tc = captured["cfg"].trainer_config
    assert tc.precision == "bf16-mixed"
    assert tc.max_steps == 7
    assert tc.val_check_interval == 0.5
    assert tc.train_data_loader.prefetch_factor == 6


def test_float32_matmul_precision_set_and_restored(monkeypatch):
    prev = torch.get_float32_matmul_precision()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    stub = SimpleNamespace(config=_cfg(float32_matmul_precision="medium"), trainer=None)
    try:
        returned = ModelTrainer._set_float32_matmul_precision(stub)
        assert returned == prev
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(prev)


def test_float32_matmul_precision_default_is_untouched(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    prev = torch.get_float32_matmul_precision()
    stub = SimpleNamespace(config=_cfg(), trainer=None)
    assert ModelTrainer._set_float32_matmul_precision(stub) is None
    assert torch.get_float32_matmul_precision() == prev


def test_float32_matmul_precision_skipped_off_cuda():
    """A CPU trainer never changes the process-wide setting."""
    prev = torch.get_float32_matmul_precision()
    stub = SimpleNamespace(
        config=_cfg(float32_matmul_precision="medium"),
        trainer=SimpleNamespace(
            strategy=SimpleNamespace(root_device=torch.device("cpu"))
        ),
    )
    assert ModelTrainer._set_float32_matmul_precision(stub) is None
    assert torch.get_float32_matmul_precision() == prev


@pytest.mark.parametrize("value,mode", [(True, "default"), ("max-autotune", None)])
def test_compile_in_place_keeps_state_dict_keys(value, mode):
    """`nn.Module.compile` keeps parameter names (no `_orig_mod.` prefix)."""
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.ReLU())
    keys = list(net.state_dict())
    stub = SimpleNamespace(
        config=_cfg(compile=value), lightning_model=SimpleNamespace(model=net)
    )
    ModelTrainer._maybe_compile_model(stub)
    assert net._compiled_call_impl is not None
    assert list(net.state_dict()) == keys


def test_compile_multi_scale_is_static_with_a_graph_per_size(monkeypatch):
    """Per-batch multi-scale compiles static graphs and fits every size."""
    import torch._dynamo

    config = torch._dynamo.config
    name = (
        "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    )
    monkeypatch.setattr(config, name, 8)
    net = torch.nn.Linear(2, 2)
    calls = []
    monkeypatch.setattr(net, "compile", lambda **kw: calls.append(kw))
    sizes = [(s, s) for s in range(416, 737, 32)]
    stub = SimpleNamespace(
        config=_cfg(compile=True),
        lightning_model=SimpleNamespace(model=net, multi_scale_sizes=sizes),
    )
    ModelTrainer._maybe_compile_model(stub)
    assert calls == [{"mode": "default", "dynamic": False}]
    assert getattr(config, name) == len(sizes) + 8


@pytest.mark.parametrize("value", [False, None])
def test_compile_off_is_a_no_op(value):
    net = torch.nn.Linear(2, 2)
    stub = SimpleNamespace(
        config=_cfg(compile=value), lightning_model=SimpleNamespace(model=net)
    )
    ModelTrainer._maybe_compile_model(stub)
    assert net._compiled_call_impl is None


def test_prefetch_factor_no_longer_tied_to_batch_size():
    cfg = _cfg()
    cfg.trainer_config.train_data_loader.batch_size = 256
    assert _get_prefetch_factor(cfg, "train_data_loader") == DEFAULT_PREFETCH_FACTOR
    # No workers -> must be None (DataLoader rejects a prefetch_factor then).
    assert _get_prefetch_factor(cfg, "val_data_loader") is None
    cfg.trainer_config.train_data_loader.prefetch_factor = 6
    assert _get_prefetch_factor(cfg, "train_data_loader") == 6


def test_train_drop_last_auto():
    ds = list(range(10))
    cfg = _cfg()
    assert not _get_train_drop_last(cfg, ds, 1, False)
    cfg.trainer_config.compile = True
    assert _get_train_drop_last(cfg, ds, 1, False)
    # A rank's share smaller than one batch would make the repeating sampler hang.
    assert not _get_train_drop_last(cfg, ds, 4, False)
    assert not _get_train_drop_last(cfg, ds, 1, True)  # tiling keeps its sampler
    cfg.trainer_config.train_data_loader.drop_last = False
    assert not _get_train_drop_last(cfg, ds, 1, False)
    cfg.trainer_config.compile = False
    cfg.trainer_config.train_data_loader.drop_last = True
    assert _get_train_drop_last(cfg, ds, 1, False)


def test_dataloaders_get_prefetch_pin_memory_and_drop_last(config, tmp_path):
    """The resolved loader settings reach the real train/val DataLoaders."""
    from sleap_nn.data.custom_datasets import (
        get_train_val_dataloaders,
        get_train_val_datasets,
    )

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", "loader_settings")
    OmegaConf.update(
        cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(cfg, "trainer_config.train_data_loader.num_workers", 1)
    OmegaConf.update(cfg, "trainer_config.val_data_loader.num_workers", 1)
    OmegaConf.update(cfg, "trainer_config.train_data_loader.batch_size", 32)
    OmegaConf.update(cfg, "trainer_config.train_data_loader.pin_memory", False)
    OmegaConf.update(cfg, "trainer_config.val_data_loader.prefetch_factor", 3)
    OmegaConf.update(cfg, "trainer_config.compile", True)
    model_trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    train_ds, val_ds = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_dl, val_dl = get_train_val_dataloaders(
        train_dataset=train_ds, val_dataset=val_ds, config=model_trainer.config
    )
    # Batch 32 is well above the default prefetch (it used to BE the prefetch).
    assert train_dl.prefetch_factor == DEFAULT_PREFETCH_FACTOR
    assert val_dl.prefetch_factor == 3
    # The val loader follows the train loader's pin_memory when unset.
    assert train_dl.pin_memory is False and val_dl.pin_memory is False
    # Compile asks for drop_last, but this tiny dataset is smaller than one
    # batch, so the guard keeps the partial batch (else the sampler hangs).
    assert train_dl.drop_last is False


def test_batch_num_samples():
    assert _batch_num_samples({"instance_image": torch.zeros(3, 1, 1, 4, 4)}) == 3
    assert _batch_num_samples({"image": torch.zeros(5, 1, 1, 4, 4)}) == 5
    assert _batch_num_samples({"frame_idx": [1, 2]}) == 2
    assert _batch_num_samples([torch.zeros(6, 2)]) == 6
    assert _batch_num_samples(torch.zeros(7)) == 7
    assert _batch_num_samples({}) == 0


def test_throughput_monitor_windows():
    """Logs one metrics dict per window with train-only rates and per-group LRs."""
    logged = []

    class _Logger:
        def log_metrics(self, metrics, step):
            logged.append((metrics, step))

    params = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.SGD(
        [
            {"params": [params[0]], "lr": 0.1, "name": "encoder"},
            {"params": [params[1]], "lr": 0.01},
        ]
    )
    trainer = SimpleNamespace(
        world_size=2,
        optimizers=[opt],
        is_global_zero=True,
        loggers=[_Logger()],
        global_step=0,
    )
    module = SimpleNamespace(device=torch.device("cpu"))
    batch = {"instance_image": torch.zeros(4, 1, 3, 8, 8)}
    cb = ThroughputMonitor(log_every_n_steps=2, log_to_console=False)

    cb.on_train_epoch_start(trainer, module)
    for i in range(5):
        cb.on_train_batch_start(trainer, module, batch, i)
        trainer.global_step += 1
        cb.on_train_batch_end(trainer, module, None, batch, i)
    assert len(cb.history) == 2 and len(logged) == 2
    m = cb.history[0]
    assert m["perf/samples_per_sec"] > 0
    assert m["perf/steps_per_sec"] > 0
    assert 0.0 <= m["perf/data_wait_frac"] <= 1.0
    assert m["perf/lr/encoder"] == 0.1 and m["perf/lr/1"] == 0.01
    assert "perf/gpu_idle_frac" not in m  # CPU: no CUDA events
    assert logged[0][1] == 2 and logged[1][1] == 4
    assert all(k.startswith("perf/") for k in logged[0][0])

    # Validation drops the partial window (no train-time contamination).
    cb.on_validation_start(trainer, module)
    assert cb._n == 0 and cb._window_start is None


def test_throughput_monitor_logs_only_on_rank_zero():
    logged = []
    trainer = SimpleNamespace(
        world_size=2,
        optimizers=[],
        is_global_zero=False,
        loggers=[SimpleNamespace(log_metrics=lambda m, step: logged.append(m))],
        global_step=0,
    )
    module = SimpleNamespace(device=torch.device("cpu"))
    cb = ThroughputMonitor(log_every_n_steps=1, log_to_console=False)
    cb.on_train_epoch_start(trainer, module)
    cb.on_train_batch_start(trainer, module, {"image": torch.zeros(2, 1)}, 0)
    cb.on_train_batch_end(trainer, module, None, {"image": torch.zeros(2, 1)}, 0)
    assert len(cb.history) == 1 and not logged


def test_training_run_with_speed_knobs(config, tmp_path):
    """A CPU run exercises bf16, clipping, accumulation, max_steps, val cadence.

    These knobs are all handled by Lightning; this checks they reach it through
    `ModelTrainer` and that a run with all of them on trains, logs and saves.
    """
    cfg = config.copy()
    run_name = "speed_knobs"
    OmegaConf.update(cfg, "trainer_config.trainer_accelerator", "cpu")
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 1)
    OmegaConf.update(cfg, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", run_name)
    OmegaConf.update(cfg, "trainer_config.max_epochs", 10)
    OmegaConf.update(cfg, "trainer_config.min_train_steps_per_epoch", 4)
    OmegaConf.update(cfg, "trainer_config.precision", "bf16-mixed")
    OmegaConf.update(cfg, "trainer_config.gradient_clip_val", 1.0)
    OmegaConf.update(cfg, "trainer_config.accumulate_grad_batches", 2)
    # 3 optimizer steps = 6 batches: stops inside the second 4-batch epoch.
    OmegaConf.update(cfg, "trainer_config.max_steps", 3)
    # Every 3 batches, across the epoch boundary: after batches 3 and 6.
    OmegaConf.update(cfg, "trainer_config.val_check_interval", 3)
    OmegaConf.update(cfg, "trainer_config.throughput_log_every_n_steps", 2)
    OmegaConf.update(
        cfg, "trainer_config.early_stopping.stop_training_on_plateau", False
    )
    # The plateau scheduler must tolerate the epoch end before the first val.
    OmegaConf.update(
        cfg,
        "trainer_config.lr_scheduler",
        {"reduce_lr_on_plateau": {"patience": 1}},
    )

    model_trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    model_trainer.train()
    trainer = model_trainer.trainer

    assert trainer.precision == "bf16-mixed"
    assert trainer.gradient_clip_val == 1.0
    assert trainer.accumulate_grad_batches == 2
    assert trainer.global_step == 3
    assert trainer.check_val_every_n_epoch is None
    # Weights stay fp32 under bf16-mixed.
    assert all(
        p.dtype == torch.float32 for p in model_trainer.lightning_model.parameters()
    )
    monitor = next(c for c in trainer.callbacks if isinstance(c, ThroughputMonitor))
    assert len(monitor.history) >= 2
    assert "perf/lr/0" in monitor.history[-1]

    run_dir = tmp_path / run_name
    assert (run_dir / "best.ckpt").exists()
    with open(run_dir / "training_log.csv") as f:
        rows = list(csv.DictReader(f))
    # One row per epoch that validated, each with that epoch's train loss/LR.
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    assert all(r["train/loss"] and r["learning_rate"] for r in rows)
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    assert saved.trainer_config.precision == "bf16-mixed"
    assert saved.trainer_config.max_steps == 3
