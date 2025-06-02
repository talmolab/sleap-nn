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
    trainer = ModelTrainer(config)
    assert np.all(trainer.train_labels[0].instances[0].numpy()) == np.all(
        labels[0].instances[0].numpy()
    )
    assert np.all(trainer.val_labels[0].instances[0].numpy()) == np.all(
        labels[0].instances[0].numpy()
    )


def test_create_data_loader_torch_dataset(caplog, config, tmp_path):
    """Test _create_data_loader function of ModelTrainer class."""
    # test centered-instance pipeline

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    # without explicitly providing crop_hw
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "trainer_config.train_data_loader.num_workers", 0)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config_copy, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders_torch_dataset()
    assert len(list(iter(model_trainer.train_dataset))) == 2
    assert len(list(iter(model_trainer.val_dataset))) == 2
    sample = next(iter(model_trainer.train_data_loader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    # memory caching
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(
        config_copy, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders_torch_dataset()
    model_trainer.train_dataset._fill_cache()
    model_trainer.val_dataset._fill_cache()
    assert len(list(iter(model_trainer.train_dataset))) == 2
    assert len(list(iter(model_trainer.val_dataset))) == 2
    sample = next(iter(model_trainer.train_data_loader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    # with saving imgs
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(
        config_copy, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(
        config_copy, "data_config.cache_img_path", f"{tmp_path}/cache_imgs/"
    )
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders_torch_dataset()
    model_trainer.train_dataset._fill_cache()
    model_trainer.val_dataset._fill_cache()
    assert len(list(iter(model_trainer.train_dataset))) == 2
    assert len(list(iter(model_trainer.val_dataset))) == 2
    sample = next(iter(model_trainer.train_data_loader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    # test exception
    config_copy = config.copy()
    head_config = config_copy.model_config.head_configs.centered_instance
    del config_copy.model_config.head_configs.centered_instance
    OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
    model_trainer = ModelTrainer(config_copy)
    with pytest.raises(ValueError):
        model_trainer._create_data_loaders_torch_dataset()
    assert "Model type: topdown. Ensure the heads config" in caplog.text


def test_create_data_loader_litdata(caplog, config, tmp_path: str):
    """Test _create_data_loader function of ModelTrainer class."""
    # test centered-instance pipeline
    OmegaConf.update(config, "data_config.data_pipeline_fw", "litdata")
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_create_data_loader_litdata_1/",
    )
    # without explicitly providing crop_hw
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders_litdata()
    assert len(list(iter(model_trainer.train_data_loader))) == 2
    assert len(list(iter(model_trainer.val_data_loader))) == 2
    sample = next(iter(model_trainer.train_data_loader))
    assert sample["instance_image"].shape == (1, 1, 1, 104, 104)

    # test exception
    config_copy = config.copy()
    OmegaConf.update(
        config_copy,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_create_data_loader_litdata_2/",
    )
    head_config = config_copy.model_config.head_configs.centered_instance
    del config_copy.model_config.head_configs.centered_instance
    OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
    model_trainer = ModelTrainer(config_copy)
    with pytest.raises(Exception):
        model_trainer._create_data_loaders_litdata()
    assert "topdown is not defined." in caplog.text


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
def test_trainer_litdata(caplog, config, tmp_path: str):
    OmegaConf.update(config, "trainer_config.save_ckpt_path", None)
    model_trainer = ModelTrainer(config)
    assert model_trainer.dir_path == "."

    #####
    # test incorrect value for data fww
    with pytest.raises(ValueError):
        OmegaConf.update(config, "data_config.data_pipeline_fw", "torch")
        model_trainer = ModelTrainer(config)
        model_trainer.train()
    assert "torch is not a valid option." in caplog.text

    #####
    OmegaConf.update(config, "data_config.data_pipeline_fw", "litdata")
    # # for topdown centered instance model
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_trainer_litdata/"
    )

    model_trainer = ModelTrainer(config)
    model_trainer.train()

    # disable ckpt, check if ckpt is created
    folder_created = Path(config.trainer_config.save_ckpt_path).exists()
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    assert not (
        Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()
    )

    #######

    # update save_ckpt to True and test step lr
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_trainer_litdata_2/"
    )
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)

    model_trainer = ModelTrainer(config)
    model_trainer.train()

    # check if wandb folder is created
    assert Path(config.trainer_config.save_ckpt_path).joinpath("wandb").exists()

    folder_created = Path(config.trainer_config.save_ckpt_path).exists()
    assert folder_created
    files = [
        str(x)
        for x in Path(config.trainer_config.save_ckpt_path).iterdir()
        if x.is_file()
    ]
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("initial_config.yaml")
        .exists()
    )
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    training_config = OmegaConf.load(
        f"{config.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.run_id is not None
    assert training_config.model_config.total_params is not None
    assert training_config.trainer_config.wandb.api_key == ""
    assert training_config.data_config.skeletons
    assert training_config.data_config.preprocessing.crop_hw == (104, 104)

    # check if ckpt is created
    assert Path(config.trainer_config.save_ckpt_path).joinpath("last.ckpt").exists()
    assert Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()

    checkpoint = torch.load(
        Path(config.trainer_config.save_ckpt_path).joinpath("last.ckpt")
    )
    assert checkpoint["epoch"] == 1

    # check if skeleton is saved in ckpt file
    assert checkpoint["config"]
    assert checkpoint["config"]["trainer_config"]["wandb"]["api_key"] == ""
    assert len(checkpoint["config"]["data_config"]["skeletons"].keys()) == 1

    # check for training metrics csv
    path = Path(config.trainer_config.save_ckpt_path).joinpath(
        "lightning_logs/version_0/"
    )
    files = [str(x) for x in Path(path).iterdir() if x.is_file()]
    metrics = False
    for i in files:
        if "metrics.csv" in i:
            metrics = True
            break
    assert metrics
    df = pd.read_csv(
        Path(config.trainer_config.save_ckpt_path).joinpath(
            "lightning_logs/version_0/metrics.csv"
        )
    )
    assert (
        abs(
            df[~np.isnan(df["learning_rate_step"])].reset_index()["learning_rate_step"][
                0
            ]
            - config.trainer_config.optimizer.lr
        )
        <= 1e-4
    )
    assert not df["val_loss_step"].isnull().all()
    assert not df["train_loss_step"].isnull().all()

    #######

    # For Single instance model
    single_instance_config = config.copy()
    head_config = single_instance_config.model_config.head_configs.centered_instance
    del single_instance_config.model_config.head_configs.centered_instance
    OmegaConf.update(
        single_instance_config, "model_config.head_configs.single_instance", head_config
    )
    del (
        single_instance_config.model_config.head_configs.single_instance.confmaps.anchor_part
    )

    trainer = ModelTrainer(single_instance_config)
    trainer._initialize_model()
    assert isinstance(trainer.model, SingleInstanceLightningModule)

    #######

    # Centroid model
    centroid_config = config.copy()
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    if (Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)

    trainer = ModelTrainer(centroid_config)

    trainer._initialize_model()
    assert isinstance(trainer.model, CentroidLightningModule)

    #######

    # bottom up model
    bottomup_config = config.copy()
    OmegaConf.update(bottomup_config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": [("part1", "part2")],
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del bottomup_config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del bottomup_config.model_config.head_configs.centered_instance
    bottomup_config.model_config.head_configs.bottomup["pafs"] = paf
    bottomup_config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    if (Path(bottomup_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(bottomup_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(bottomup_config, "trainer_config.use_wandb", False)
    OmegaConf.update(bottomup_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(bottomup_config, "trainer_config.steps_per_epoch", 10)

    trainer = ModelTrainer(bottomup_config)
    trainer._initialize_model()
    assert isinstance(trainer.model, BottomUpLightningModule)


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

    model_trainer = ModelTrainer(config)
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
def test_trainer_torch_dataset(caplog, config, tmp_path: str):
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_memory"
    )
    OmegaConf.update(config, "trainer_config.save_ckpt_path", None)
    model_trainer = ModelTrainer(config)
    assert model_trainer.dir_path == "."

    ### invalid profiler
    OmegaConf.update(config, "trainer_config.profiler", "simple_torch")
    with pytest.raises(ValueError):
        model_trainer = ModelTrainer(
            config,
        )
        model_trainer.train()
    assert f"simple_torch is not a valid option" in caplog.text

    ##### test for reusing imgs path
    OmegaConf.update(config, "trainer_config.profiler", "simple")
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(config, "data_config.cache_img_path", tmp_path)
    OmegaConf.update(config, "data_config.use_existing_imgs", True)
    with pytest.raises(Exception):
        model_trainer = ModelTrainer(
            config,
        )
    assert "There are no images in the path" in caplog.text

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
        model_trainer = ModelTrainer(config)
    assert "There are no images in the path" in caplog.text

    #####

    # # for topdown centered instance model
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_trainer_torch_dataset/",
    )
    OmegaConf.update(
        config, "data_config.data_pipeline_fw", "torch_dataset_cache_img_disk"
    )
    OmegaConf.update(config, "data_config.cache_img_path", f"{tmp_path}/cache_imgs/")
    OmegaConf.update(config, "data_config.use_existing_imgs", False)

    model_trainer = ModelTrainer(config)
    model_trainer.train()

    # disable ckpt, check if ckpt is created
    folder_created = Path(config.trainer_config.save_ckpt_path).exists()
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    assert not (
        Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()
    )

    #######

    # update save_ckpt to True and test step lr
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "trainer_config.visualize_preds_during_training", True)
    OmegaConf.update(config, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.step_size", 10)
    OmegaConf.update(config, "trainer_config.lr_scheduler.step_lr.gamma", 0.5)

    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer(config)
    model_trainer.train()

    # check if wandb folder is created
    assert Path(config.trainer_config.save_ckpt_path).joinpath("wandb").exists()

    # check if viz folder is created and non-empty
    assert Path(config.trainer_config.save_ckpt_path).joinpath("viz").exists()
    assert any((Path(config.trainer_config.save_ckpt_path) / "viz").glob("*.png"))

    folder_created = Path(config.trainer_config.save_ckpt_path).exists()
    assert folder_created
    files = [
        str(x)
        for x in Path(config.trainer_config.save_ckpt_path).iterdir()
        if x.is_file()
    ]
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("initial_config.yaml")
        .exists()
    )
    assert (
        Path(config.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    training_config = OmegaConf.load(
        f"{config.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.run_id is not None
    assert training_config.model_config.total_params is not None
    assert training_config.trainer_config.wandb.api_key == ""
    assert training_config.data_config.skeletons
    assert training_config.data_config.preprocessing.crop_hw == (104, 104)

    # check if ckpt is created
    assert Path(config.trainer_config.save_ckpt_path).joinpath("last.ckpt").exists()
    assert Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()

    checkpoint = torch.load(
        Path(config.trainer_config.save_ckpt_path).joinpath("last.ckpt")
    )
    assert checkpoint["epoch"] == 1

    # check if skeleton is saved in ckpt file
    assert checkpoint["config"]
    assert checkpoint["config"]["trainer_config"]["wandb"]["api_key"] == ""
    assert len(checkpoint["config"]["data_config"]["skeletons"].keys()) == 1

    # check for training metrics csv
    path = Path(config.trainer_config.save_ckpt_path).joinpath(
        "lightning_logs/version_0/"
    )
    files = [str(x) for x in Path(path).iterdir() if x.is_file()]
    metrics = False
    for i in files:
        if "metrics.csv" in i:
            metrics = True
            break
    assert metrics
    df = pd.read_csv(
        Path(config.trainer_config.save_ckpt_path).joinpath(
            "lightning_logs/version_0/metrics.csv"
        )
    )
    assert (
        abs(
            df[~np.isnan(df["learning_rate_step"])].reset_index()["learning_rate_step"][
                0
            ]
            - config.trainer_config.optimizer.lr
        )
        <= 1e-4
    )
    assert not df["val_loss_step"].isnull().all()
    assert not df["train_loss_step"].isnull().all()
    #######

    # check resume training
    config_copy = config.copy()
    OmegaConf.update(config_copy, "trainer_config.max_epochs", 4)
    OmegaConf.update(
        config_copy,
        "trainer_config.resume_ckpt_path",
        f"{Path(config.trainer_config.save_ckpt_path).joinpath('last.ckpt')}",
    )
    training_config = OmegaConf.load(
        f"{config_copy.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    prv_runid = training_config.trainer_config.wandb.run_id
    OmegaConf.update(config_copy, "trainer_config.wandb.prv_runid", prv_runid)
    OmegaConf.update(config_copy, "data_config.data_pipeline_fw", "torch_dataset")
    trainer = ModelTrainer(config_copy)
    trainer.train()

    checkpoint = torch.load(
        Path(config_copy.trainer_config.save_ckpt_path).joinpath("last.ckpt")
    )
    assert checkpoint["epoch"] == 3

    training_config = OmegaConf.load(
        f"{config_copy.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    assert training_config.trainer_config.wandb.run_id == prv_runid
    os.remove((Path(trainer.dir_path) / "best.ckpt").as_posix())
    os.remove((Path(trainer.dir_path) / "last.ckpt").as_posix())
    shutil.rmtree((Path(trainer.dir_path) / "lightning_logs").as_posix())

    #######

    # check early stopping
    config_early_stopping = config.copy()
    OmegaConf.update(
        config_early_stopping, "trainer_config.early_stopping.min_delta", 1e-1
    )
    OmegaConf.update(config_early_stopping, "trainer_config.early_stopping.patience", 1)
    OmegaConf.update(config_early_stopping, "trainer_config.max_epochs", 10)
    OmegaConf.update(
        config_early_stopping, "trainer_config.lr_scheduler", {"step_lr": None}
    )
    OmegaConf.update(
        config_early_stopping,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer/",
    )
    OmegaConf.update(
        config_early_stopping, "data_config.data_pipeline_fw", "torch_dataset"
    )

    trainer = ModelTrainer(config_early_stopping)
    trainer.train()

    checkpoint = torch.load(
        Path(config_early_stopping.trainer_config.save_ckpt_path).joinpath("last.ckpt")
    )
    assert checkpoint["epoch"] == 1

    #######

    # For Single instance model
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
        single_instance_config, "data_config.data_pipeline_fw", "torch_dataset"
    )
    OmegaConf.update(
        single_instance_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(single_instance_config, "trainer_config.max_epochs", 2)

    trainer = ModelTrainer(single_instance_config)
    trainer._initialize_model()
    assert isinstance(trainer.model, SingleInstanceLightningModule)

    #######

    # Centroid model
    centroid_config = config.copy()
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    if (Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)

    OmegaConf.update(centroid_config, "data_config.data_pipeline_fw", "torch_dataset")
    OmegaConf.update(
        centroid_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 2)

    trainer = ModelTrainer(centroid_config)

    trainer._initialize_model()
    trainer.train()
    assert isinstance(trainer.model, CentroidLightningModule)

    #######

    # bottom up model
    bottomup_config = config.copy()
    OmegaConf.update(bottomup_config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": None,
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del bottomup_config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del bottomup_config.model_config.head_configs.centered_instance
    bottomup_config.model_config.head_configs.bottomup.confmaps.part_names = None
    bottomup_config.model_config.head_configs.bottomup["pafs"] = paf
    bottomup_config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    if (Path(bottomup_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(bottomup_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(bottomup_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(bottomup_config, "trainer_config.use_wandb", False)
    OmegaConf.update(bottomup_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(bottomup_config, "trainer_config.steps_per_epoch", 10)
    OmegaConf.update(
        bottomup_config, "trainer_config.visualize_preds_during_training", True
    )
    OmegaConf.update(bottomup_config, "trainer_config.max_epochs", 2)

    trainer = ModelTrainer(bottomup_config)
    trainer._initialize_model()
    trainer.train()
    assert isinstance(trainer.model, BottomUpLightningModule)


def test_trainer_load_trained_ckpts(config, tmp_path, minimal_instance_ckpt):
    """Test loading trained weights for backbone and head layers."""

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(
        config,
        "model_config.pretrained_backbone_weights",
        (Path(minimal_instance_ckpt) / "best.ckpt").as_posix(),
    )
    OmegaConf.update(
        config,
        "model_config.pretrained_head_weights",
        (Path(minimal_instance_ckpt) / "best.ckpt").as_posix(),
    )

    # check loading trained weights for backbone
    load_weights_config = config.copy()
    ckpt = torch.load((Path(minimal_instance_ckpt) / "best.ckpt").as_posix())
    first_layer_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

    # load head ckpts
    head_layer_ckpt = ckpt["state_dict"]["model.head_layers.0.0.weight"][
        0, 0, :
    ].numpy()

    trainer = ModelTrainer(load_weights_config)
    trainer._initialize_model()
    model_ckpt = next(trainer.model.parameters())[0, 0, :].detach().numpy()

    assert np.all(np.abs(first_layer_ckpt - model_ckpt) < 1e-6)

    model_ckpt = (
        next(trainer.model.model.head_layers.parameters())[0, 0, :].detach().numpy()
    )

    assert np.all(np.abs(head_layer_ckpt - model_ckpt) < 1e-6)


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
def test_reuse_bin_files(config, tmp_path: str):
    """Test reusing `.bin` files."""
    # Centroid model
    OmegaConf.update(config, "data_config.data_pipeline_fw", "litdata")
    centroid_config = config.copy()
    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    OmegaConf.update(
        centroid_config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer/",
    )

    if (Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)
    OmegaConf.update(
        centroid_config, "data_config.delete_cache_imgs_after_training", False
    )
    OmegaConf.update(
        centroid_config,
        "data_config.litdata_chunks_path",
        Path(tmp_path) / "new_chunks",
    )

    # test reusing bin files
    trainer1 = ModelTrainer(centroid_config)
    trainer1.train()

    OmegaConf.update(
        centroid_config,
        "data_config.litdata_chunks_path",
        (trainer1.train_litdata_chunks_path).split("train_chunks")[0],
    )
    OmegaConf.update(
        centroid_config,
        "data_config.use_existing_imgs",
        True,
    )
    trainer2 = ModelTrainer(centroid_config)
    trainer2.train()


@pytest.mark.skipif(
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
# TODO: Revisit this test later (Failing on ubuntu)
# torch dataset
def test_reuse_npz_files(config, tmp_path: str):
    """Test reusing `.npz` files."""
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
        f"{tmp_path}/test_model_trainer/",
    )

    if (Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt").exists():
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "best.ckpt"
            ).as_posix()
        )
        os.remove(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "last.ckpt"
            ).as_posix()
        )
        shutil.rmtree(
            (
                Path(centroid_config.trainer_config.save_ckpt_path) / "lightning_logs"
            ).as_posix()
        )

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.profiler", "simple")
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)
    OmegaConf.update(
        centroid_config, "data_config.delete_cache_imgs_after_training", False
    )
    OmegaConf.update(
        centroid_config,
        "data_config.cache_img_path",
        Path(tmp_path) / "new_imags",
    )

    # test reusing bin files
    trainer1 = ModelTrainer(centroid_config)
    trainer1.train()

    OmegaConf.update(
        centroid_config,
        "data_config.cache_img_path",
        (trainer1.cache_img_path.as_posix()),
    )
    OmegaConf.update(
        centroid_config,
        "data_config.use_existing_imgs",
        True,
    )
    trainer2 = ModelTrainer(centroid_config)
    trainer2.train()
