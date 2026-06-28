"""Test ModelTrainer classes."""

import re
import torch
import numpy as np
from PIL import Image
import pytest
import zmq
import time
import jsonpickle
import threading
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import pandas as pd
import sys
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.training.lightning_modules import (
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
)
from sleap_nn.config.model_config import ConvNextConfig, SingleInstanceConfig
import sleap_io as sio
from torch.nn.functional import mse_loss
import os
import wandb
from lightning.pytorch.loggers import WandbLogger
import shutil
from loguru import logger
from _pytest.logging import LogCaptureFixture
from sleap_nn.data.custom_datasets import (
    get_train_val_datasets,
    get_train_val_dataloaders,
)
from sleap_nn.config.training_job_config import TrainingJobConfig


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


@pytest.fixture(autouse=True)
def cleanup_wandb():
    """Ensure wandb runs in offline mode and is cleaned up after each test.

    This fixture:
    1. Sets WANDB_MODE=offline to prevent network hangs on CI
    2. Cleans up any active wandb run after the test to prevent state leakage
    """
    # Save original mode and force offline to prevent network hangs on CI
    original_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"

    yield

    # Finish any active wandb run to prevent contamination between tests
    if wandb.run is not None:
        wandb.finish()

    # Restore original WANDB_MODE
    if original_mode is not None:
        os.environ["WANDB_MODE"] = original_mode
    else:
        os.environ.pop("WANDB_MODE", None)


def test_cfg_without_val_labels_path(config, tmp_path, minimal_instance):
    """Test Model Trainer if no val labels path is provided."""
    labels = sio.load_slp(minimal_instance)
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_vals_fraction")
    config.data_config.val_labels_path = None
    # val labels will be split from train labels.
    trainer = ModelTrainer.get_model_trainer_from_config(config)
    assert np.all(trainer.train_labels[0][0].instances[0].numpy()) == np.all(
        labels[0].instances[0].numpy()
    )
    assert np.all(trainer.val_labels[0][0].instances[0].numpy()) == np.all(
        labels[0].instances[0].numpy()
    )


def test_setup_data_loaders_torch_dataset(caplog, config, tmp_path, minimal_instance):
    """Test _create_data_loader function of ModelTrainer class."""
    ## torch_dataset: test centered-instance pipeline
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_model_trainer")
    # without explicitly providing crop_size
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_size", None)
    OmegaConf.update(config_copy, "trainer_config.train_data_loader.num_workers", 0)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_padding", 0)
    OmegaConf.update(config_copy, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config_copy)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_dataloader, _ = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    assert len(list(iter(train_dataset))) == 2
    assert len(list(iter(val_dataset))) == 2
    sample = next(iter(train_dataloader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    ## with memory caching
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_size", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_padding", 0)
    OmegaConf.update(
        config_copy, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    model_trainer = ModelTrainer.get_model_trainer_from_config(config_copy)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_dataloader, _ = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    assert len(list(iter(train_dataset))) == 2
    assert len(list(iter(val_dataset))) == 2
    sample = next(iter(train_dataloader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    ## with caching imgs on disk
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_size", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_padding", 0)
    OmegaConf.update(
        config_copy, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(
        config_copy, "data_config.cache_img_path", Path(tmp_path) / f"./cache_imgs/"
    )
    model_trainer = ModelTrainer.get_model_trainer_from_config(
        config_copy,
        train_labels=[sio.load_slp(minimal_instance)],
        val_labels=[sio.load_slp(minimal_instance)],
    )
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_dataloader, _ = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    assert (
        Path(config_copy.data_config.cache_img_path).joinpath("train_imgs")
    ).exists()
    assert any(
        (Path(config_copy.data_config.cache_img_path).joinpath("train_imgs")).iterdir()
    )
    assert (Path(config_copy.data_config.cache_img_path).joinpath("val_imgs")).exists()
    assert any(
        (Path(config_copy.data_config.cache_img_path).joinpath("val_imgs")).iterdir()
    )
    assert len(list(iter(train_dataset))) == 2
    assert len(list(iter(val_dataset))) == 2
    sample = next(iter(train_dataloader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    ## raise exception if no imgs are found when use_existing_imgs = True.
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(config, "data_config.cache_img_path", "test_reuse_cache")
    OmegaConf.update(config, "data_config.use_existing_imgs", True)
    with pytest.raises(Exception):
        model_trainer = ModelTrainer.get_model_trainer_from_config(
            config,
        )
        train_dataset, val_dataset = get_train_val_datasets(
            train_labels=model_trainer.train_labels,
            val_labels=model_trainer.val_labels,
            config=model_trainer.config,
        )
        train_data_loader, val_data_loader = get_train_val_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=model_trainer.config,
        )
    assert "There are no images in the path" in caplog.text

    # test with non-empty `train_imgs` but empty `val_imgs`
    Path.mkdir(Path(tmp_path) / "train_imgs", parents=True)
    file_path = Path(tmp_path) / "train_imgs" / "sample.jpg"
    Image.fromarray(
        np.random.randint(low=0, high=255, size=(100, 100)).astype(np.uint8)
    ).save(file_path, format="JPEG")

    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(config, "data_config.cache_img_path", tmp_path)
    OmegaConf.update(config, "data_config.use_existing_imgs", True)

    with pytest.raises(Exception):
        model_trainer = ModelTrainer.get_model_trainer_from_config(config)
        train_dataset, val_dataset = get_train_val_datasets(
            train_labels=model_trainer.train_labels,
            val_labels=model_trainer.val_labels,
            config=model_trainer.config,
        )
        train_data_loader, val_data_loader = get_train_val_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=model_trainer.config,
        )
    assert "There are no images in the path" in caplog.text


def test_wandb():
    """Test wandb integration."""
    old_mode = os.environ.get("WANDB_MODE")
    try:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init()
        assert wandb.run is not None
        wandb.finish()
    finally:
        # Restore original WANDB_MODE
        if old_mode is not None:
            os.environ["WANDB_MODE"] = old_mode
        elif "WANDB_MODE" in os.environ:
            del os.environ["WANDB_MODE"]


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_centered_instance(caplog, config, tmp_path: str):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(config, "trainer_config.run_name", "test_trainer")
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.profiler", None)

    ## invalid profiler: raise exception
    invalid_profiler_cfg = config.copy()
    OmegaConf.update(invalid_profiler_cfg, "trainer_config.profiler", "simple_torch")
    with pytest.raises(ValueError):
        model_trainer = ModelTrainer.get_model_trainer_from_config(
            invalid_profiler_cfg,
        )
        model_trainer.train()
    assert f"simple_torch is not a valid option" in caplog.text

    ## disable save ckpt
    no_save_ckpt_cfg = config.copy()
    OmegaConf.update(
        no_save_ckpt_cfg, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(no_save_ckpt_cfg, "trainer_config.save_ckpt", False)
    OmegaConf.update(
        no_save_ckpt_cfg, "trainer_config.run_name", "test_trainer_no_save_ckpt"
    )
    OmegaConf.update(
        no_save_ckpt_cfg,
        "trainer_config.ckpt_dir",
        f"{tmp_path}",
    )
    OmegaConf.update(
        no_save_ckpt_cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(
        no_save_ckpt_cfg, "data_config.cache_img_path", f"{tmp_path}/cache_imgs/"
    )

    model_trainer = ModelTrainer.get_model_trainer_from_config(no_save_ckpt_cfg)
    model_trainer.train()

    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("training_config.yaml")
        .exists()
    )
    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("viz")
        .exists()
    )
    assert not (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("best.ckpt")
        .exists()
    )

    ## update save_ckpt to True and test files in the ckpt folder
    training_cfg = config.copy()
    OmegaConf.update(training_cfg, "trainer_config.save_ckpt", True)
    OmegaConf.update(training_cfg, "trainer_config.use_wandb", True)
    OmegaConf.update(training_cfg, "trainer_config.max_epochs", 2)
    OmegaConf.update(
        training_cfg, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(training_cfg, "data_config.preprocessing.crop_size", None)
    OmegaConf.update(training_cfg, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(training_cfg, "data_config.preprocessing.crop_padding", 0)
    OmegaConf.update(training_cfg, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(training_cfg, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(training_cfg, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer.get_model_trainer_from_config(training_cfg)
    model_trainer.train()

    assert (
        Path(model_trainer.config.trainer_config.ckpt_dir)
        / model_trainer.config.trainer_config.run_name
    ).exists()
    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("wandb")
        .exists()
    )  # check wandb folder

    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("viz")
        .exists()
    )  # check if viz folder is created and non-empty
    assert any(
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
            / "viz"
        ).glob("*.png")
    )

    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("last.ckpt")
        .exists()
    )
    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("best.ckpt")
        .exists()
    )

    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("initial_config.yaml")
        .exists()
    )
    assert (
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        )
        .joinpath("training_config.yaml")
        .exists()
    )
    training_config = OmegaConf.load(
        f"{model_trainer.config.trainer_config.ckpt_dir}/{model_trainer.config.trainer_config.run_name}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.current_run_id is not None
    assert training_config.model_config.total_params is not None
    assert training_config.trainer_config.wandb.api_key == ""
    assert training_config.data_config.skeletons
    assert training_config.data_config.preprocessing.crop_size == 104

    # Verify API key is also masked in initial_config.yaml
    initial_config = OmegaConf.load(
        f"{model_trainer.config.trainer_config.ckpt_dir}/{model_trainer.config.trainer_config.run_name}/initial_config.yaml"
    )
    assert initial_config.trainer_config.wandb.api_key == ""

    checkpoint = torch.load(
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        ).joinpath("last.ckpt"),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 1

    # check for training metrics csv
    path = (
        Path(model_trainer.config.trainer_config.ckpt_dir)
        / model_trainer.config.trainer_config.run_name
    )
    assert path.joinpath("training_log.csv").exists()
    df = pd.read_csv(
        (
            Path(model_trainer.config.trainer_config.ckpt_dir)
            / model_trainer.config.trainer_config.run_name
        ).joinpath("training_log.csv")
    )
    # Verify training log has data
    assert len(df) > 0, "Training log CSV is empty"
    # Check learning rate if any non-NaN values exist
    lr_values = df["learning_rate"].dropna()
    if len(lr_values) > 0:
        assert (
            abs(lr_values.iloc[0] - config.trainer_config.optimizer.lr) <= 1e-4
        ), f"Learning rate mismatch: {lr_values.iloc[0]} vs {config.trainer_config.optimizer.lr}"
    assert not df["val/loss"].isnull().all()
    assert not df["train/loss"].isnull().all()
    # check if part loss is logged
    assert not df["train/confmaps/A"].isnull().all()


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_single_instance(config, tmp_path, minimal_instance):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    single_instance_config = config.copy()
    head_config = single_instance_config.model_config.head_configs.centered_instance
    del single_instance_config.model_config.head_configs.centered_instance
    OmegaConf.update(
        single_instance_config, "model_config.head_configs.single_instance", head_config
    )
    del (
        single_instance_config.model_config.head_configs.single_instance.confmaps.anchor_part
    )
    OmegaConf.update(
        single_instance_config,
        "data_config.data_pipeline_fw",
        "torch_dataset_cache_img_memory",
    )
    OmegaConf.update(single_instance_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(
        single_instance_config,
        "trainer_config.run_name",
        "test_model_trainer_single_instance",
    )
    OmegaConf.update(
        single_instance_config,
        "trainer_config.ckpt_dir",
        f"{tmp_path}",
    )
    OmegaConf.update(
        single_instance_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(single_instance_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(
        single_instance_config,
        "trainer_config.online_hard_keypoint_mining.online_mining",
        True,
    )
    OmegaConf.update(
        single_instance_config,
        "data_config.data_pipeline_fw",
        "torch_dataset_cache_img_disk",
    )
    OmegaConf.update(
        single_instance_config,
        "trainer_config.train_data_loader.num_workers",
        2,
    )
    OmegaConf.update(
        single_instance_config,
        "trainer_config.val_data_loader.num_workers",
        2,
    )

    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = [lf.instances[0]]

    trainer = ModelTrainer.get_model_trainer_from_config(
        single_instance_config, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()
    assert isinstance(trainer.lightning_model, SingleInstanceLightningModule)
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.0000.png"
    ).exists()
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "best.ckpt"
    ).exists()


def _make_single_instance_config(config, tmp_path, run_name):
    """Convert the shared `config` fixture into a single-instance config."""
    single_instance_config = config.copy()
    head_config = single_instance_config.model_config.head_configs.centered_instance
    del single_instance_config.model_config.head_configs.centered_instance
    OmegaConf.update(
        single_instance_config, "model_config.head_configs.single_instance", head_config
    )
    del (
        single_instance_config.model_config.head_configs.single_instance.confmaps.anchor_part
    )
    OmegaConf.update(single_instance_config, "trainer_config.run_name", run_name)
    OmegaConf.update(single_instance_config, "trainer_config.ckpt_dir", f"{tmp_path}")
    return single_instance_config


def test_single_instance_multiple_instances_error(config, tmp_path, minimal_instance):
    """Single-instance training should error on frames with >1 instance."""
    single_instance_config = _make_single_instance_config(
        config, tmp_path, "test_single_instance_multiple_instances_error"
    )

    # `minimal_instance` has frames with two instances; do NOT strip them.
    labels = sio.load_slp(minimal_instance)

    with pytest.raises(ValueError, match="at most one instance per frame"):
        ModelTrainer.get_model_trainer_from_config(
            single_instance_config, train_labels=[labels], val_labels=[labels]
        )


def test_single_instance_single_instance_ok(config, tmp_path, minimal_instance):
    """One instance per frame should pass single-instance label validation."""
    single_instance_config = _make_single_instance_config(
        config, tmp_path, "test_single_instance_single_instance_ok"
    )

    labels = sio.load_slp(minimal_instance)
    for lf in labels:
        lf.instances = [lf.instances[0]]

    # Should not raise during label setup / validation.
    trainer = ModelTrainer.get_model_trainer_from_config(
        single_instance_config, train_labels=[labels], val_labels=[labels]
    )
    assert trainer.model_type == "single_instance"


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_centroid(config, tmp_path):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # Centroid model
    centroid_config = config.copy()
    head_config = centroid_config.model_config.head_configs.centered_instance
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    OmegaConf.update(
        centroid_config, "trainer_config.run_name", "test_model_trainer_centroid"
    )
    OmegaConf.update(centroid_config, "trainer_config.ckpt_dir", f"{tmp_path}")
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.min_train_steps_per_epoch", 10)

    OmegaConf.update(centroid_config, "data_config.data_pipeline_fw", "torch_dataset")
    OmegaConf.update(
        centroid_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)

    trainer = ModelTrainer.get_model_trainer_from_config(centroid_config)
    trainer.train()
    assert isinstance(trainer.lightning_model, CentroidLightningModule)
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.0000.png"
    ).exists()
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "best.ckpt"
    ).exists()


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_zmq_callbacks(config, tmp_path: str):
    # Setup ZMQ subscriber
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.subscribe("")
    socket.bind("tcp://127.0.0.1:9510")
    received_msgs = []

    def zmq_listener():
        # Give time for trainer to start publishing
        start = time.time()
        while time.time() - start < 5:
            if socket.poll(timeout=100):
                msg = socket.recv_string()
                received_msgs.append(jsonpickle.decode(msg))

    # Start subscriber thread before training
    listener_thread = threading.Thread(target=zmq_listener)
    listener_thread.start()

    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(config, "trainer_config.run_name", "test_zmq_callbacks")
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.zmq.publish_port", "9510")
    OmegaConf.update(config, "trainer_config.zmq.controller_port", "9004")
    OmegaConf.update(config, "trainer_config.max_epochs", 1)

    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    model_trainer.train()

    listener_thread.join()

    # Verify at least one message was received
    assert any(
        "logs" in msg for msg in received_msgs
    ), "No ZMQ messages received from training."


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_bottomup(config, tmp_path):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # bottom up model
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.profiler", "simple")
    OmegaConf.update(config, "trainer_config.run_name", "bottomup_trainer_test")
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(config, "trainer_config.enable_progress_bar", True)
    OmegaConf.update(config, "trainer_config.max_epochs", 1)
    OmegaConf.update(config, "data_config.delete_cache_imgs_after_training", False)
    OmegaConf.update(
        config, "trainer_config.online_hard_keypoint_mining.online_mining", True
    )

    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    head_config = config.model_config.head_configs.centered_instance
    bottomup_config = config.copy()
    OmegaConf.update(bottomup_config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": [("A", "B")],
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del bottomup_config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del bottomup_config.model_config.head_configs.centered_instance
    bottomup_config.model_config.head_configs.bottomup.confmaps.part_names = ["A", "B"]
    bottomup_config.model_config.head_configs.bottomup["pafs"] = paf
    bottomup_config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    trainer = ModelTrainer.get_model_trainer_from_config(bottomup_config)
    trainer.train()
    assert isinstance(trainer.lightning_model, BottomUpLightningModule)
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
    ).exists()
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.pafs_magnitude.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "validation.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "validation.pafs_magnitude.0000.png"
    )


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_multi_class_bottomup(config, tmp_path, minimal_instance):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # bottom up model
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.profiler", "simple")
    OmegaConf.update(
        config, "trainer_config.run_name", "multiclass_bottomup_trainer_test"
    )
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(config, "trainer_config.enable_progress_bar", True)
    OmegaConf.update(config, "trainer_config.max_epochs", 1)
    OmegaConf.update(config, "data_config.delete_cache_imgs_after_training", False)
    OmegaConf.update(
        config, "trainer_config.online_hard_keypoint_mining.online_mining", True
    )

    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    head_config = config.model_config.head_configs.centered_instance
    bottomup_config = config.copy()
    OmegaConf.update(
        bottomup_config, "model_config.head_configs.multi_class_bottomup", head_config
    )
    class_maps = {
        "classes": None,
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del bottomup_config.model_config.head_configs.multi_class_bottomup[
        "confmaps"
    ].anchor_part
    del bottomup_config.model_config.head_configs.centered_instance
    bottomup_config.model_config.head_configs.multi_class_bottomup.confmaps.part_names = [
        "A",
        "B",
    ]
    bottomup_config.model_config.head_configs.multi_class_bottomup["class_maps"] = (
        class_maps
    )
    bottomup_config.model_config.head_configs.multi_class_bottomup.confmaps.loss_weight = (
        1.0
    )

    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()

    trainer = ModelTrainer.get_model_trainer_from_config(
        bottomup_config, train_labels=[tracked_labels], val_labels=[tracked_labels]
    )
    trainer.train()
    assert isinstance(trainer.lightning_model, BottomUpMultiClassLightningModule)
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
    ).exists()
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.class_maps.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "validation.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "validation.class_maps.0000.png"
    )


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_multi_classtopdown(config, tmp_path, minimal_instance, caplog):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.profiler", "simple")
    OmegaConf.update(
        config, "trainer_config.run_name", "multiclass_topdown_trainer_test"
    )
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(config, "trainer_config.enable_progress_bar", True)
    OmegaConf.update(config, "trainer_config.max_epochs", 1)
    OmegaConf.update(config, "data_config.delete_cache_imgs_after_training", False)
    OmegaConf.update(
        config, "trainer_config.online_hard_keypoint_mining.online_mining", True
    )

    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    confmaps = config.model_config.head_configs.centered_instance
    class_vectors = {
        "classes": None,
        "num_fc_layers": 1,
        "num_fc_units": 64,
        "output_stride": 16,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.multi_class_topdown", confmaps)
    config.model_config.head_configs["multi_class_topdown"][
        "class_vectors"
    ] = class_vectors
    config.model_config.head_configs.multi_class_topdown.confmaps.loss_weight = 1.0

    with pytest.raises(Exception):
        trainer = ModelTrainer.get_model_trainer_from_config(
            config,
            train_labels=[sio.load_slp(minimal_instance)],
            val_labels=[sio.load_slp(minimal_instance)],
        )
    assert "No tracks found. ID models need tracks to be defined." in caplog.text

    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()

    trainer = ModelTrainer.get_model_trainer_from_config(
        config, train_labels=[tracked_labels], val_labels=[tracked_labels]
    )
    trainer.train()
    assert isinstance(
        trainer.lightning_model, TopDownCenteredInstanceMultiClassLightningModule
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
    ).exists()
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "train.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
        / "validation.0000.png"
    )


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_resume_training(config, tmp_path):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # train a model for 2 epochs:
    OmegaConf.update(config, "trainer_config.run_name", "test_resume_trainer")
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    trainer = ModelTrainer.get_model_trainer_from_config(config)
    trainer.train()

    config_copy = config.copy()
    OmegaConf.update(config_copy, "trainer_config.max_epochs", 4)
    OmegaConf.update(
        config_copy,
        "trainer_config.resume_ckpt_path",
        f"{(Path(trainer.config.trainer_config.ckpt_dir) / trainer.config.trainer_config.run_name).joinpath('best.ckpt')}",
    )
    training_config = OmegaConf.load(
        (
            Path(trainer.config.trainer_config.ckpt_dir)
            / trainer.config.trainer_config.run_name
            / "training_config.yaml"
        ).as_posix()
    )
    prv_runid = training_config.trainer_config.wandb.current_run_id
    OmegaConf.update(config_copy, "trainer_config.wandb.prv_runid", prv_runid)
    OmegaConf.update(config_copy, "data_config.data_pipeline_fw", "torch_dataset")
    OmegaConf.update(config_copy, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config_copy, "trainer_config.run_name", "test_resume_trainer")
    OmegaConf.update(config_copy, "trainer_config.save_ckpt", True)
    trainer = ModelTrainer.get_model_trainer_from_config(config_copy)
    trainer.train()

    checkpoint = torch.load(
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 3

    training_config = OmegaConf.load(
        (
            Path(trainer.config.trainer_config.ckpt_dir)
            / trainer.config.trainer_config.run_name
            / "training_config.yaml"
        ).as_posix()
    )
    assert training_config.trainer_config.wandb.current_run_id == prv_runid


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_early_stopping(config, tmp_path):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    config_early_stopping = config.copy()
    OmegaConf.update(
        config_early_stopping, "trainer_config.early_stopping.min_delta", 1e-1
    )
    OmegaConf.update(
        config_early_stopping,
        "trainer_config.online_hard_keypoint_mining.online_mining",
        True,
    )
    OmegaConf.update(config_early_stopping, "trainer_config.early_stopping.patience", 1)
    OmegaConf.update(config_early_stopping, "trainer_config.save_ckpt", True)
    OmegaConf.update(config_early_stopping, "trainer_config.model_ckpt.save_last", True)
    OmegaConf.update(config_early_stopping, "trainer_config.max_epochs", 10)
    OmegaConf.update(
        config_early_stopping, "trainer_config.lr_scheduler", {"step_lr": None}
    )
    OmegaConf.update(
        config_early_stopping, "trainer_config.run_name", "test_early_stopping"
    )
    OmegaConf.update(config_early_stopping, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config_early_stopping, "data_config.data_pipeline_fw", "torch_dataset"
    )

    trainer = ModelTrainer.get_model_trainer_from_config(config_early_stopping)
    trainer.train()

    checkpoint = torch.load(
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 1


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_reuse_cache_img_files(config, tmp_path: str):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # Centroid model
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    centroid_config = config.copy()
    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    OmegaConf.update(
        centroid_config,
        "trainer_config.run_name",
        "test_model_trainer_reuse_imgs_cache",
    )
    OmegaConf.update(centroid_config, "trainer_config.ckpt_dir", f"{tmp_path}")

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.profiler", "simple")
    OmegaConf.update(centroid_config, "trainer_config.min_train_steps_per_epoch", 10)
    OmegaConf.update(
        centroid_config, "data_config.delete_cache_imgs_after_training", False
    )
    OmegaConf.update(
        centroid_config,
        "data_config.cache_img_path",
        Path(tmp_path) / "new_imags",
    )

    # test reusing bin files
    trainer1 = ModelTrainer.get_model_trainer_from_config(centroid_config)
    trainer1.train()

    OmegaConf.update(
        centroid_config,
        "data_config.cache_img_path",
        (trainer1.config.data_config.cache_img_path),
    )
    OmegaConf.update(
        centroid_config,
        "data_config.use_existing_imgs",
        True,
    )
    trainer2 = ModelTrainer.get_model_trainer_from_config(centroid_config)
    trainer2.train()


def test_keep_viz_behavior(config, tmp_path, minimal_instance):
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # Test keep_viz = True (viz folder should be kept)
    cfg_keep = config.copy()
    OmegaConf.update(cfg_keep, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg_keep, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(cfg_keep, "trainer_config.keep_viz", True)
    OmegaConf.update(cfg_keep, "trainer_config.run_name", "test_keep_viz_true")
    OmegaConf.update(cfg_keep, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg_keep, "trainer_config.max_epochs", 1)
    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg_keep, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()
    viz_path = (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
    )
    assert viz_path.exists() and any(
        viz_path.glob("*.png")
    ), "viz folder should be kept when keep_viz=True"

    # Test keep_viz = False (viz folder should be deleted)
    cfg_del = config.copy()
    OmegaConf.update(cfg_del, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg_del, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(cfg_del, "trainer_config.keep_viz", False)
    OmegaConf.update(cfg_del, "trainer_config.run_name", "test_keep_viz_false")
    OmegaConf.update(cfg_del, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg_del, "trainer_config.max_epochs", 1)
    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg_del, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()
    viz_path = (
        Path(trainer.config.trainer_config.ckpt_dir)
        / trainer.config.trainer_config.run_name
        / "viz"
    )
    assert not viz_path.exists(), "viz folder should be deleted when keep_viz=False"


def test_backbone_oneof_validation_error(config, caplog):
    # Test that an error is raised when a oneof field is not set
    config_unet_and_convnext = config.copy()
    OmegaConf.update(
        config_unet_and_convnext,
        "model_config.backbone_config.convnext",
        ConvNextConfig(),
    )
    with pytest.raises(ValueError):
        ModelTrainer.get_model_trainer_from_config(config_unet_and_convnext)
    assert "Only one attribute" in caplog.text
    assert "BackboneConfig" in caplog.text


def test_backbone_oneof_validation_error_no_backbone(config, caplog):
    # Test that an error is raised when no backbone field is set
    config_no_backbone = config.copy()
    OmegaConf.update(
        config_no_backbone,
        "model_config.backbone_config.unet",
        None,
    )
    OmegaConf.update(
        config_no_backbone,
        "model_config.backbone_config.convnext",
        None,
    )
    OmegaConf.update(
        config_no_backbone,
        "model_config.backbone_config.swint",
        None,
    )
    with pytest.raises(ValueError):
        ModelTrainer.get_model_trainer_from_config(config_no_backbone)


def test_head_configs_oneof_validation_error(config, caplog):
    # Test that an error is raised when a oneof field is not set
    config_two_head_configs = config.copy()
    OmegaConf.update(
        config_two_head_configs,
        "model_config.head_configs.single_instance",
        SingleInstanceConfig(),
    )
    with pytest.raises(ValueError):
        ModelTrainer.get_model_trainer_from_config(config_two_head_configs)
    assert "Only one attribute" in caplog.text
    assert "HeadConfig" in caplog.text


def test_head_config_oneof_validation_error_no_head(config, caplog):
    # Test that an error is raised when no head field is set
    config_no_head = config.copy()
    OmegaConf.update(
        config_no_head,
        "model_config.head_configs.single_instance",
        None,
    )
    OmegaConf.update(
        config_no_head,
        "model_config.head_configs.centroid",
        None,
    )
    OmegaConf.update(
        config_no_head,
        "model_config.head_configs.centered_instance",
        None,
    )
    OmegaConf.update(
        config_no_head,
        "model_config.head_configs.bottomup",
        None,
    )
    with pytest.raises(ValueError):
        ModelTrainer.get_model_trainer_from_config(config_no_head)


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_loading_pretrained_weights(
    config,
    sleap_centered_instance_model_path,
    minimal_instance,
    caplog,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """Test loading pretrained weights for model initialization."""
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"
    # with keras (.h5 weights)
    sleap_nn_config = TrainingJobConfig.load_sleap_config(
        Path(sleap_centered_instance_model_path) / "training_config.json"
    )
    sleap_nn_config.model_config.pretrained_backbone_weights = (
        Path(sleap_centered_instance_model_path) / "best_model.h5"
    )
    sleap_nn_config.model_config.pretrained_head_weights = (
        Path(sleap_centered_instance_model_path) / "best_model.h5"
    )
    sleap_nn_config.data_config.preprocessing.ensure_rgb = True
    sleap_nn_config.trainer_config.max_epochs = 1
    sleap_nn_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    sleap_nn_config.trainer_config.ckpt_dir = f"{tmp_path}"
    sleap_nn_config.trainer_config.run_name = "test_loading_weights"

    trainer = ModelTrainer.get_model_trainer_from_config(
        config=sleap_nn_config,
        train_labels=[sio.load_slp(minimal_instance)],
        val_labels=[sio.load_slp(minimal_instance)],
    )
    trainer.train()

    assert "Loading backbone weights from" in caplog.text
    assert "Successfully loaded 28/28 weights from legacy model" in caplog.text
    assert "Loading head weights from" in caplog.text
    assert "Successfully loaded 2/2 weights from legacy model" in caplog.text

    # loading `.ckpt`
    sleap_nn_config = TrainingJobConfig.load_sleap_config(
        Path(sleap_centered_instance_model_path) / "initial_config.json"
    )
    sleap_nn_config.model_config.pretrained_backbone_weights = (
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt"
    )
    sleap_nn_config.model_config.pretrained_head_weights = (
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt"
    )
    sleap_nn_config.data_config.preprocessing.ensure_rgb = True
    sleap_nn_config.trainer_config.max_epochs = 1
    sleap_nn_config.trainer_config.ckpt_dir = f"{tmp_path}"
    sleap_nn_config.trainer_config.run_name = "test_loading_weights"
    sleap_nn_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    trainer = ModelTrainer.get_model_trainer_from_config(
        config=sleap_nn_config,
        train_labels=[sio.load_slp(minimal_instance)],
        val_labels=[sio.load_slp(minimal_instance)],
    )
    trainer.train()

    assert "Loading backbone weights from" in caplog.text
    assert "Loading head weights from" in caplog.text


def test_file_not_found_handling(config, tmp_path, caplog, minimal_instance):
    """Test ModelTrainer handles missing video files gracefully."""
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"

    # Load labels and modify video filename to point to non-existent file
    labels = sio.load_slp(minimal_instance)
    labels.videos[0].filename = "/nonexistent/path/video.mp4"

    OmegaConf.update(config, "trainer_config.max_epochs", 1)
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_missing_video")

    with pytest.raises(FileNotFoundError):
        trainer = ModelTrainer.get_model_trainer_from_config(
            config, train_labels=[labels], val_labels=[labels]
        )


def test_model_ckpt_path_duplication(config, caplog, tmp_path, minimal_instance):
    # Test model checkpoint path duplication in the config
    if torch.mps.is_available():
        config.trainer_config.trainer_accelerator = "cpu"
    else:
        config.trainer_config.trainer_accelerator = "auto"

    # if run name is empty string
    cfg_copy = config.copy()
    OmegaConf.update(
        cfg_copy,
        "trainer_config.ckpt_dir",
        f"{tmp_path}",
    )
    OmegaConf.update(
        cfg_copy,
        "trainer_config.save_ckpt",
        True,
    )
    OmegaConf.update(cfg_copy, "trainer_config.run_name", "")
    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg_copy, train_labels=[labels], val_labels=[labels]
    )

    trainer.train()

    # use an existing run name
    config_duplicate_ckpt_path = config.copy()
    OmegaConf.update(
        config_duplicate_ckpt_path,
        "trainer_config.ckpt_dir",
        f"{tmp_path}",
    )
    OmegaConf.update(
        config_duplicate_ckpt_path,
        "trainer_config.run_name",
        "test_saved_ckpt",
    )
    OmegaConf.update(
        config_duplicate_ckpt_path,
        "trainer_config.save_ckpt",
        True,
    )
    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        config_duplicate_ckpt_path, train_labels=[labels], val_labels=[labels]
    )

    trainer.train()
    assert (
        Path(config_duplicate_ckpt_path.trainer_config.ckpt_dir)
        / config_duplicate_ckpt_path.trainer_config.run_name
    ).exists()

    trainer = ModelTrainer.get_model_trainer_from_config(
        config_duplicate_ckpt_path, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()
    assert Path(
        f"{config_duplicate_ckpt_path.trainer_config.ckpt_dir}/{config_duplicate_ckpt_path.trainer_config.run_name}-1"
    ).exists()

    trainer = ModelTrainer.get_model_trainer_from_config(
        config_duplicate_ckpt_path, train_labels=[labels], val_labels=[labels]
    )
    trainer.train()
    assert Path(
        f"{config_duplicate_ckpt_path.trainer_config.ckpt_dir}/{config_duplicate_ckpt_path.trainer_config.run_name}-2"
    ).exists()


def test_multi_gpu_disk_cache_requires_run_name(config, tmp_path, minimal_instance):
    """Test that multi-GPU + disk cache without explicit run_name raises an error."""
    from unittest.mock import patch

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", None)  # No run_name provided
    OmegaConf.update(
        cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 2)  # Multi-GPU

    labels = sio.load_slp(minimal_instance)

    # Mock _get_trainer_devices to return 2 (simulating multi-GPU)
    with patch.object(ModelTrainer, "_get_trainer_devices", return_value=2):
        with pytest.raises(ValueError) as exc_info:
            ModelTrainer.get_model_trainer_from_config(
                cfg, train_labels=[labels], val_labels=[labels]
            )

        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "Multi-GPU training with disk caching requires" in error_msg
        assert "run_name" in error_msg
        assert "2 device(s)" in error_msg


def test_multi_gpu_disk_cache_with_explicit_run_name_succeeds(
    config, tmp_path, minimal_instance
):
    """Test that multi-GPU + disk cache WITH explicit run_name works."""
    from unittest.mock import patch

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", "my_explicit_run")  # Explicit name
    OmegaConf.update(
        cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(cfg, "data_config.cache_img_path", f"{tmp_path}/cache")
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 2)

    labels = sio.load_slp(minimal_instance)

    # Mock _get_trainer_devices to return 2 (simulating multi-GPU)
    with patch.object(ModelTrainer, "_get_trainer_devices", return_value=2):
        # Should NOT raise - explicit run_name provided
        trainer = ModelTrainer.get_model_trainer_from_config(
            cfg, train_labels=[labels], val_labels=[labels]
        )
        assert trainer.config.trainer_config.run_name == "my_explicit_run"


def test_single_gpu_disk_cache_auto_generates_run_name(
    config, tmp_path, minimal_instance
):
    """Test that single GPU + disk cache auto-generates run_name (unchanged behavior)."""
    import re

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", None)  # No run_name
    OmegaConf.update(
        cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(cfg, "data_config.cache_img_path", f"{tmp_path}/cache")
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 1)  # Single GPU

    labels = sio.load_slp(minimal_instance)

    # Should NOT raise - single GPU doesn't have the sync issue
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )

    # run_name should be auto-generated with timestamp pattern: YYMMDD_HHMMSS.*
    assert trainer.config.trainer_config.run_name is not None
    assert re.match(r"\d{6}_\d{6}\.", trainer.config.trainer_config.run_name)


def test_multi_gpu_memory_cache_auto_generates_run_name(
    config, tmp_path, minimal_instance
):
    """Test that multi-GPU + memory cache auto-generates run_name (no sync issue)."""
    from unittest.mock import patch
    import re

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", None)  # No run_name
    OmegaConf.update(
        cfg, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 2)

    labels = sio.load_slp(minimal_instance)

    # Mock _get_trainer_devices to return 2
    with patch.object(ModelTrainer, "_get_trainer_devices", return_value=2):
        # Should NOT raise - memory caching doesn't have the path sync issue
        trainer = ModelTrainer.get_model_trainer_from_config(
            cfg, train_labels=[labels], val_labels=[labels]
        )

        # run_name should be auto-generated
        assert trainer.config.trainer_config.run_name is not None
        assert re.match(r"\d{6}_\d{6}\.", trainer.config.trainer_config.run_name)


def test_multi_gpu_no_cache_auto_generates_run_name(config, tmp_path, minimal_instance):
    """Test that multi-GPU without caching auto-generates run_name."""
    from unittest.mock import patch
    import re

    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", None)
    OmegaConf.update(cfg, "data_config.data_pipeline_fw", "torch_dataset")  # No caching
    OmegaConf.update(cfg, "trainer_config.trainer_devices", 2)

    labels = sio.load_slp(minimal_instance)

    with patch.object(ModelTrainer, "_get_trainer_devices", return_value=2):
        # Should NOT raise - no disk caching
        trainer = ModelTrainer.get_model_trainer_from_config(
            cfg, train_labels=[labels], val_labels=[labels]
        )

        assert trainer.config.trainer_config.run_name is not None
        assert re.match(r"\d{6}_\d{6}\.", trainer.config.trainer_config.run_name)


class TestEmbeddingChannelAutoSync:
    """The convnext/swint+ImageNet channel auto-sync is decoupled for embedding.

    Every other model type flips to RGB when a 3-channel ImageNet stem is requested;
    the ``embedding`` model type keeps its (default grayscale) data channels — the
    1-channel crop is repeated to 3 in ``Model.forward`` (P2 #7).
    """

    @staticmethod
    def _cfg(model_type):
        head = (
            {"embedding": {"embedding": {"embedding_dim": 128}}}
            if model_type == "embedding"
            else {"centered_instance": {"confmaps": {}}}
        )
        return OmegaConf.create(
            {
                "data_config": {
                    "preprocessing": {"ensure_rgb": False, "ensure_grayscale": True}
                },
                "model_config": {
                    "backbone_config": {
                        "convnext": {
                            "in_channels": 1,
                            "pre_trained_weights": "ConvNeXt_Tiny_Weights",
                        }
                    },
                    "head_configs": head,
                },
            }
        )

    def _verify(self, model_type):
        mt = ModelTrainer.__new__(ModelTrainer)
        mt.config = self._cfg(model_type)
        mt.backbone_type = "convnext"
        mt.model_type = model_type
        mt.train_labels = [None]  # skip the image-derived channel block
        mt._verify_model_input_channels()
        return mt.config

    def test_embedding_keeps_grayscale_with_imagenet_stem(self):
        cfg = self._verify("embedding")
        assert cfg.model_config.backbone_config.convnext.in_channels == 3
        assert cfg.data_config.preprocessing.ensure_grayscale is True
        assert cfg.data_config.preprocessing.ensure_rgb is False

    def test_non_embedding_still_flips_to_rgb(self):
        cfg = self._verify("centered_instance")
        assert cfg.model_config.backbone_config.convnext.in_channels == 3
        assert cfg.data_config.preprocessing.ensure_rgb is True
        assert cfg.data_config.preprocessing.ensure_grayscale is False


class TestEmbeddingMemoryFallback:
    """Low-RAM cache fallback for the embedding model type (P0.2 follow-up)."""

    @staticmethod
    def _cfg():
        return OmegaConf.create(
            {
                "data_config": {
                    "data_pipeline_fw": "torch_dataset_cache_img_memory",
                    "cache_img_path": None,
                },
                "trainer_config": {
                    "train_data_loader": {"num_workers": 0},
                    "val_data_loader": {"num_workers": 0},
                    "run_name": "expt",
                },
            }
        )

    def _run(self, model_type):
        from unittest.mock import MagicMock, patch

        cfg = self._cfg()
        mt = ModelTrainer(config=cfg)
        mt._initial_config = cfg.copy()
        mt.model_type = model_type
        mt.train_labels = []
        mt.val_labels = []
        mt.trainer = MagicMock(num_devices=1, global_rank=0)
        with (
            patch(
                "sleap_nn.training.model_trainer.check_cache_memory", return_value=False
            ),
            patch(
                "sleap_nn.training.model_trainer.get_train_val_datasets",
                return_value=("TRAIN", "VAL"),
            ) as gtv,
        ):
            out = mt._setup_datasets()
        return mt, out, gtv

    def test_embedding_low_memory_falls_back_to_uncached(self):
        """Embedding + insufficient RAM -> uncached `torch_dataset` (NOT lossy disk)."""
        mt, out, gtv = self._run("embedding")
        assert mt.config.data_config.data_pipeline_fw == "torch_dataset"
        assert mt.config.data_config.cache_img_path is None
        assert out == ("TRAIN", "VAL")
        gtv.assert_called_once()

    def test_non_embedding_low_memory_still_falls_back_to_disk(self):
        """Non-embedding + insufficient RAM keeps the existing disk-cache fallback."""
        mt, _, _ = self._run("centroid")
        assert mt.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk"


class TestClassUuidMinting:
    """Train-time minting of per-class canonical identity UUIDs (the A2 bridge).

    ``ModelTrainer._setup_head_config`` populates ``class_uuids`` parallel to
    ``classes`` so the frozen per-class uuid persists into
    ``training_config.yaml`` and is re-emitted at inference. These tests exercise
    the minting logic in isolation (no full training run) by merging the raw head
    config with the structured schema (mirroring ``verify_training_cfg``, which is
    what surfaces the ``class_uuids`` key in production) and calling
    ``_setup_head_config`` directly.
    """

    _HEX32 = re.compile(r"^[0-9a-f]{32}$")

    @staticmethod
    def _tracked_labels(minimal_instance, track_names=("female", "male")):
        labels = sio.load_slp(minimal_instance)
        tracks = [sio.Track(name=n) for n in track_names]
        for lf in labels:
            for i, instance in enumerate(lf.instances):
                instance.track = tracks[i % len(tracks)]
        labels.tracks = tracks
        labels.update()
        return labels, tracks

    @staticmethod
    def _build_trainer(model_type, sub_key, sub_cfg, train_labels):
        """Merge with the schema (to surface ``class_uuids``), build a trainer.

        Includes a ``confmaps`` head (with explicit ``part_names`` so no skeleton
        lookup is needed) alongside the class head, mirroring the real
        multi-class head shape.
        """
        confmaps = {
            "part_names": ["A", "B"],
            "sigma": 1.5,
            "output_stride": 2,
            "loss_weight": 1.0,
        }
        if model_type == "multi_class_topdown":
            confmaps["anchor_part"] = "A"
        head = {"confmaps": confmaps, sub_key: sub_cfg}
        raw = OmegaConf.create({"model_config": {"head_configs": {model_type: head}}})
        schema = OmegaConf.structured(TrainingJobConfig())
        merged = OmegaConf.merge(schema, raw)
        trainer = ModelTrainer(config=merged)
        trainer.model_type = model_type
        trainer.train_labels = train_labels
        trainer.skeletons = train_labels[0].skeletons
        return trainer

    def test_topdown_class_uuids_minted(self, minimal_instance):
        """class_vectors: classes + class_uuids populated, equal len, 32-hex."""
        labels, _ = self._tracked_labels(minimal_instance)
        trainer = self._build_trainer(
            "multi_class_topdown",
            "class_vectors",
            {"classes": None},
            [labels],
        )
        trainer._setup_head_config()
        cv = trainer.config.model_config.head_configs.multi_class_topdown.class_vectors
        assert cv.classes is not None and len(cv.classes) == 2
        assert cv.class_uuids is not None
        assert len(cv.class_uuids) == len(cv.classes)
        assert all(self._HEX32.match(u) for u in cv.class_uuids)
        # UUIDs are unique per class.
        assert len(set(cv.class_uuids)) == len(cv.class_uuids)

    def test_bottomup_class_uuids_minted(self, minimal_instance):
        """class_maps: classes + class_uuids populated, equal len, 32-hex."""
        labels, _ = self._tracked_labels(minimal_instance)
        trainer = self._build_trainer(
            "multi_class_bottomup",
            "class_maps",
            {"classes": None},
            [labels],
        )
        trainer._setup_head_config()
        cm = trainer.config.model_config.head_configs.multi_class_bottomup.class_maps
        assert cm.classes is not None and len(cm.classes) == 2
        assert cm.class_uuids is not None
        assert len(cm.class_uuids) == len(cm.classes)
        assert all(self._HEX32.match(u) for u in cm.class_uuids)

    def test_class_uuids_stable_across_yaml_reload(self, minimal_instance, tmp_path):
        """Minted uuids survive a write+reload of training_config.yaml unchanged."""
        labels, _ = self._tracked_labels(minimal_instance)
        trainer = self._build_trainer(
            "multi_class_topdown",
            "class_vectors",
            {"classes": None},
            [labels],
        )
        trainer._setup_head_config()
        cv = trainer.config.model_config.head_configs.multi_class_topdown.class_vectors
        classes_before = list(cv.classes)
        uuids_before = list(cv.class_uuids)

        cfg_path = tmp_path / "training_config.yaml"
        OmegaConf.save(trainer.config, cfg_path)
        reloaded = OmegaConf.load(cfg_path)
        cv2 = reloaded.model_config.head_configs.multi_class_topdown.class_vectors
        assert list(cv2.classes) == classes_before
        assert list(cv2.class_uuids) == uuids_before

        # Re-running setup on the reloaded config must NOT re-mint (idempotent).
        trainer2 = ModelTrainer(config=reloaded)
        trainer2.model_type = "multi_class_topdown"
        trainer2.train_labels = [labels]
        trainer2.skeletons = labels.skeletons
        trainer2._setup_head_config()
        cv3 = (
            trainer2.config.model_config.head_configs.multi_class_topdown.class_vectors
        )
        assert list(cv3.class_uuids) == uuids_before

    def test_class_uuids_reused_from_gt_identities(self, minimal_instance):
        """When labels carry sio.Identity matching a class name, reuse its uuid."""
        labels, tracks = self._tracked_labels(minimal_instance)
        # Attach GT identities (name-matched to the track/class names).
        gt_identities = [sio.Identity(name=t.name) for t in tracks]
        labels.identities = gt_identities
        gt_uuid_by_name = {i.name: i.uuid for i in gt_identities}

        trainer = self._build_trainer(
            "multi_class_topdown",
            "class_vectors",
            {"classes": None},
            [labels],
        )
        trainer._setup_head_config()
        cv = trainer.config.model_config.head_configs.multi_class_topdown.class_vectors
        for name, uuid in zip(cv.classes, cv.class_uuids):
            assert (
                uuid == gt_uuid_by_name[name]
            ), f"class {name!r} uuid {uuid} != GT uuid {gt_uuid_by_name[name]}"

    def test_class_uuids_minted_when_classes_user_provided(self, minimal_instance):
        """User-provided classes with class_uuids None still get minted uuids."""
        labels, _ = self._tracked_labels(minimal_instance)
        trainer = self._build_trainer(
            "multi_class_topdown",
            "class_vectors",
            {"classes": ["female", "male"], "class_uuids": None},
            [labels],
        )
        trainer._setup_head_config()
        cv = trainer.config.model_config.head_configs.multi_class_topdown.class_vectors
        assert list(cv.classes) == ["female", "male"]
        assert cv.class_uuids is not None
        assert len(cv.class_uuids) == 2
        assert all(self._HEX32.match(u) for u in cv.class_uuids)
