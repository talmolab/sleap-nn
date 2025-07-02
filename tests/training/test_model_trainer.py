"""Test ModelTrainer classes."""

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
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
)
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


def test_cfg_without_val_labels_path(config, tmp_path, minimal_instance):
    """Test Model Trainer if no val labels path is provided."""
    labels = sio.load_slp(minimal_instance)
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_vals_fraction/"
    )
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
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    # without explicitly providing crop_hw
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "trainer_config.train_data_loader.num_workers", 0)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
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
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
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
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
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

    ## raise exception if no imsg are found when use_existing_imgs = True.
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

    # test with non-empty `train_imgs` but emtpy `val_imgs`
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
    os.environ["WANDB_MODE"] = "offline"
    wandb_logger = WandbLogger()
    wandb.init()
    assert wandb.run is not None
    wandb.finish()


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_centered_instance(caplog, config, tmp_path: str):
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(config, "trainer_config.save_ckpt_path", None)
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
        no_save_ckpt_cfg,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_trainer_no_save_ckpt",
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
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("viz")
        .exists()
    )
    assert not (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
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
    OmegaConf.update(training_cfg, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(training_cfg, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(training_cfg, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(training_cfg, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(training_cfg, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer.get_model_trainer_from_config(training_cfg)
    model_trainer.train()

    assert Path(model_trainer.config.trainer_config.save_ckpt_path).exists()
    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("wandb")
        .exists()
    )  # check wandb folder

    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("viz")
        .exists()
    )  # check if viz folder is created and non-empty
    assert any(
        (Path(model_trainer.config.trainer_config.save_ckpt_path) / "viz").glob("*.png")
    )

    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("last.ckpt")
        .exists()
    )
    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("best.ckpt")
        .exists()
    )

    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("initial_config.yaml")
        .exists()
    )
    assert (
        Path(model_trainer.config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    training_config = OmegaConf.load(
        f"{model_trainer.config.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.current_run_id is not None
    assert training_config.model_config.total_params is not None
    assert training_config.trainer_config.wandb.api_key == ""
    assert training_config.data_config.skeletons
    assert training_config.data_config.preprocessing.crop_hw == (104, 104)

    checkpoint = torch.load(
        Path(model_trainer.config.trainer_config.save_ckpt_path).joinpath("last.ckpt"),
        map_location="cpu",
    )
    assert checkpoint["epoch"] == 1

    # check if skeleton is saved in ckpt file
    assert checkpoint["config"]
    assert checkpoint["config"]["trainer_config"]["wandb"]["api_key"] == ""
    assert len(checkpoint["config"]["data_config"]["skeletons"].keys()) == 1

    # check for training metrics csv
    path = Path(model_trainer.config.trainer_config.save_ckpt_path)
    assert path.joinpath("training_log.csv").exists()
    df = pd.read_csv(
        Path(model_trainer.config.trainer_config.save_ckpt_path).joinpath(
            "training_log.csv"
        )
    )
    assert (
        abs(
            df[~np.isnan(df["learning_rate"])].reset_index()["learning_rate"][0]
            - config.trainer_config.optimizer.lr
        )
        <= 1e-4
    )
    assert not df["val_loss"].isnull().all()
    assert not df["train_loss"].isnull().all()
    # check if part loss is logged
    assert not df["A"].isnull().all()


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_single_instance(config, tmp_path, minimal_instance):
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
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer_single_instan e",
    )
    OmegaConf.update(
        single_instance_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(single_instance_config, "trainer_config.max_epochs", 2)
    OmegaConf.update(
        single_instance_config,
        "trainer_config.online_hard_keypoint_mining.online_mining",
        True,
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
        Path(trainer.config.trainer_config.save_ckpt_path) / "viz" / "train.0000.png"
    ).exists()
    assert (Path(trainer.config.trainer_config.save_ckpt_path) / "best.ckpt").exists()


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_centroid(config, tmp_path):
    # Centroid model
    centroid_config = config.copy()
    head_config = centroid_config.model_config.head_configs.centered_instance
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    OmegaConf.update(
        centroid_config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer_centroid",
    )
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
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 2)

    trainer = ModelTrainer.get_model_trainer_from_config(centroid_config)
    trainer.train()
    assert isinstance(trainer.lightning_model, CentroidLightningModule)
    assert (
        Path(trainer.config.trainer_config.save_ckpt_path) / "viz" / "train.0000.png"
    ).exists()
    assert (Path(trainer.config.trainer_config.save_ckpt_path) / "best.ckpt").exists()


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_zmq_callbacks(config, tmp_path: str):
    # Setup ZMQ subscriber
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
    OmegaConf.update(config, "trainer_config.save_ckpt_path", None)
    OmegaConf.update(
        config, "trainer_config.zmq.publish_address", "tcp://127.0.0.1:9510"
    )
    OmegaConf.update(
        config, "trainer_config.zmq.controller_address", "tcp://127.0.0.1:9004"
    )
    OmegaConf.update(config, "trainer_config.max_epochs", 1)

    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    model_trainer.train()

    listener_thread.join()

    # Verify at least one message was received
    assert any(
        "logs" in msg for msg in received_msgs
    ), "No ZMQ messages received from training."


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_model_trainer_bottomup(config, tmp_path):
    # bottom up model
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.profiler", "simple")
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{Path(tmp_path) / 'bottomup_trainer_test'}",
    )
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)
    OmegaConf.update(config, "trainer_config.enable_progress_bar", True)
    OmegaConf.update(config, "trainer_config.max_epochs", 2)
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
    assert (Path(trainer.config.trainer_config.save_ckpt_path) / "viz").exists()
    assert Path(trainer.config.trainer_config.save_ckpt_path) / "viz" / "train.0000.png"
    assert (
        Path(trainer.config.trainer_config.save_ckpt_path)
        / "viz"
        / "train.pafs_magnitude.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.save_ckpt_path)
        / "viz"
        / "validation.0000.png"
    )
    assert (
        Path(trainer.config.trainer_config.save_ckpt_path)
        / "viz"
        / "validation.pafs_magnitude.0000.png"
    )


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_resume_training(config):
    # train a model for 2 epochs:
    OmegaConf.update(config, "trainer_config.save_ckpt_path", None)
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    trainer = ModelTrainer.get_model_trainer_from_config(config)
    trainer.train()

    config_copy = config.copy()
    OmegaConf.update(config_copy, "trainer_config.max_epochs", 4)
    OmegaConf.update(
        config_copy,
        "trainer_config.resume_ckpt_path",
        f"{Path(trainer.config.trainer_config.save_ckpt_path).joinpath('best.ckpt')}",
    )
    training_config = OmegaConf.load(
        f"{trainer.config.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    prv_runid = training_config.trainer_config.wandb.current_run_id
    OmegaConf.update(config_copy, "trainer_config.wandb.prv_runid", prv_runid)
    OmegaConf.update(config_copy, "data_config.data_pipeline_fw", "torch_dataset")
    OmegaConf.update(config_copy, "trainer_config.save_ckpt_path", None)
    OmegaConf.update(config_copy, "trainer_config.save_ckpt", True)
    trainer = ModelTrainer.get_model_trainer_from_config(config_copy)
    trainer.train()

    checkpoint = torch.load(
        Path(trainer.config.trainer_config.save_ckpt_path).joinpath("last.ckpt"),
        map_location="cpu",
    )
    assert checkpoint["epoch"] == 3

    training_config = OmegaConf.load(
        f"{trainer.config.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.current_run_id == prv_runid


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_early_stopping(config, tmp_path):
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
        config_early_stopping,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_early_stopping/",
    )
    OmegaConf.update(
        config_early_stopping, "data_config.data_pipeline_fw", "torch_dataset"
    )

    trainer = ModelTrainer.get_model_trainer_from_config(config_early_stopping)
    trainer.train()

    checkpoint = torch.load(
        Path(trainer.config.trainer_config.save_ckpt_path).joinpath("best.ckpt"),
        map_location="cpu",
    )
    assert checkpoint["epoch"] == 1


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_reuse_cache_img_files(config, tmp_path: str):
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
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer_reuse_imgs_cache/",
    )

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
