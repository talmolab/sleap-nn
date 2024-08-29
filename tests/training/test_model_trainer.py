"""Test ModelTrainer and TrainingModule classes."""

import torch
import sleap_io as sio
from typing import Text
import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig
import lightning as L
from pathlib import Path
import pandas as pd
from sleap_nn.training.model_trainer import (
    ModelTrainer,
    TopDownCenteredInstanceModel,
    SingleInstanceModel,
    CentroidModel,
    BottomUpModel,
)
from torch.nn.functional import mse_loss
import os
import wandb
from lightning.pytorch.loggers import WandbLogger
import shutil


def test_create_data_loader(config, tmp_path: str):
    """Test _create_data_loader function of ModelTrainer class."""
    # test centered-instance pipeline
    model_trainer = ModelTrainer(config)
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer._create_data_loaders()
    assert isinstance(
        model_trainer.train_data_loader, torch.utils.data.dataloader.DataLoader
    )
    assert isinstance(
        model_trainer.val_data_loader, torch.utils.data.dataloader.DataLoader
    )
    assert len(list(iter(model_trainer.train_data_loader))) == 2
    assert len(list(iter(model_trainer.val_data_loader))) == 2

    # without explicitly providing crop_hw
    config_copy = config.copy()
    OmegaConf.update(config_copy, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config_copy, "data_config.preprocessing.min_crop_size", 100)
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders()
    assert len(list(iter(model_trainer.train_data_loader))) == 2
    assert len(list(iter(model_trainer.val_data_loader))) == 2
    sample = next(iter(model_trainer.train_data_loader))
    assert sample["instance_image"].shape == (1, 1, 1, 112, 112)

    # test exception
    config_copy = config.copy()
    head_config = config_copy.model_config.head_configs.centered_instance
    del config_copy.model_config.head_configs.centered_instance
    OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
    model_trainer = ModelTrainer(config_copy)
    with pytest.raises(Exception):
        model_trainer._create_data_loaders()

    # test single instance pipeline
    config_copy = config.copy()
    del config_copy.model_config.head_configs.centered_instance
    OmegaConf.update(
        config_copy, "model_config.head_configs.single_instance", head_config
    )
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders()
    assert len(list(iter(model_trainer.train_data_loader))) == 1
    assert len(list(iter(model_trainer.val_data_loader))) == 1

    # test centroid pipeline
    config_copy = config.copy()
    del config_copy.model_config.head_configs.centered_instance
    OmegaConf.update(config_copy, "model_config.head_configs.centroid", head_config)
    model_trainer = ModelTrainer(config_copy)
    model_trainer._create_data_loaders()
    assert len(list(iter(model_trainer.train_data_loader))) == 1
    assert len(list(iter(model_trainer.val_data_loader))) == 1
    ex = next(iter(model_trainer.train_data_loader))
    assert ex["centroids_confidence_maps"].shape == (1, 1, 1, 192, 192)


def test_wandb():
    """Test wandb integration."""
    os.environ["WANDB_MODE"] = "offline"
    wandb_logger = WandbLogger()
    wandb.init()
    assert wandb.run is not None
    wandb.finish()


def test_trainer(config, tmp_path: str):
    # # for topdown centered instance model
    model_trainer = ModelTrainer(config)
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
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

    # update save_ckpt to True
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "data_config.preprocessing.crop_hw", None)
    OmegaConf.update(config, "data_config.preprocessing.min_crop_size", 100)

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
    assert training_config.data_config.preprocessing.crop_hw == (112, 112)

    # check if ckpt is created
    assert Path(config.trainer_config.save_ckpt_path).joinpath("last.ckpt").exists()
    assert Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()

    checkpoint = torch.load(
        Path(config.trainer_config.save_ckpt_path).joinpath("best.ckpt")
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
    assert abs(df.loc[0, "learning_rate"] - config.trainer_config.optimizer.lr) <= 1e-4
    assert not df.val_loss.isnull().all()
    assert not df.train_loss.isnull().all()

    # check resume training
    config_copy = config.copy()
    OmegaConf.update(config_copy, "trainer_config.max_epochs", 4)
    OmegaConf.update(
        config_copy,
        "trainer_config.resume_ckpt_path",
        f"{Path(config.trainer_config.save_ckpt_path).joinpath('best.ckpt')}",
    )
    training_config = OmegaConf.load(
        f"{config_copy.trainer_config.save_ckpt_path}/training_config.yaml"
    )
    prv_runid = training_config.trainer_config.wandb.run_id
    OmegaConf.update(config_copy, "trainer_config.wandb.prv_runid", prv_runid)
    trainer = ModelTrainer(config_copy)
    trainer.train()

    checkpoint = torch.load(
        Path(config_copy.trainer_config.save_ckpt_path).joinpath("best.ckpt")
    )
    assert checkpoint["epoch"] == 3

    wandb_folders = list(
        Path(f"{config_copy.trainer_config.save_ckpt_path}/wandb").iterdir()
    )
    for f in wandb_folders:
        if f.is_dir():
            assert prv_runid in str(f.stem)
    shutil.rmtree(f"{config_copy.trainer_config.save_ckpt_path}")

    # check early stopping
    config_early_stopping = config.copy()
    OmegaConf.update(
        config_early_stopping, "trainer_config.early_stopping.min_delta", 1e-1
    )
    OmegaConf.update(config_early_stopping, "trainer_config.early_stopping.patience", 1)
    OmegaConf.update(config_early_stopping, "trainer_config.max_epochs", 10)
    OmegaConf.update(
        config_early_stopping,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_model_trainer/",
    )

    trainer = ModelTrainer(config_early_stopping)
    trainer.train()

    checkpoint = torch.load(
        Path(config_early_stopping.trainer_config.save_ckpt_path).joinpath("best.ckpt")
    )
    assert checkpoint["epoch"] == 1

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
    trainer._create_data_loaders()
    trainer._initialize_model()
    assert isinstance(trainer.model, SingleInstanceModel)

    # Centroid model
    centroid_config = config.copy()
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    if Path(centroid_config.trainer_config.save_ckpt_path).exists():
        shutil.rmtree(centroid_config.trainer_config.save_ckpt_path)

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)

    trainer = ModelTrainer(centroid_config)
    trainer._create_data_loaders()

    trainer._initialize_model()
    assert isinstance(trainer.model, CentroidModel)

    trainer.train()

    checkpoint = torch.load(
        Path(centroid_config.trainer_config.save_ckpt_path).joinpath("best.ckpt")
    )
    assert checkpoint["epoch"] == 0
    assert checkpoint["global_step"] == 10

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

    if Path(bottomup_config.trainer_config.save_ckpt_path).exists():
        shutil.rmtree(bottomup_config.trainer_config.save_ckpt_path)

    OmegaConf.update(bottomup_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(bottomup_config, "trainer_config.use_wandb", False)
    OmegaConf.update(bottomup_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(bottomup_config, "trainer_config.steps_per_epoch", 10)

    trainer = ModelTrainer(bottomup_config)
    trainer._create_data_loaders()

    trainer._initialize_model()
    assert isinstance(trainer.model, BottomUpModel)

    trainer.train()

    checkpoint = torch.load(
        Path(bottomup_config.trainer_config.save_ckpt_path).joinpath("best.ckpt")
    )
    assert checkpoint["epoch"] == 0
    assert checkpoint["global_step"] == 10


def test_topdown_centered_instance_model(config, tmp_path: str):

    # unet
    model = TopDownCenteredInstanceModel(config, None, "centered_instance")
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the output shape
    assert preds.shape == (1, 2, 80, 80)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm)) < 1e-3

    # convnext with pretrained weights
    OmegaConf.update(
        config, "model_config.pre_trained_weights", "ConvNeXt_Tiny_Weights"
    )
    OmegaConf.update(config, "model_config.backbone_type", "convnext")
    OmegaConf.update(
        config,
        "model_config.backbone_config",
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
        },
    )
    model = TopDownCenteredInstanceModel(config, None, "centered_instance")
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the output shape
    assert preds.shape == (1, 2, 80, 80)
    print(next(model.parameters())[0, 0, 0, :].detach().numpy())
    assert all(
        np.abs(
            next(model.parameters())[0, 0, 0, :].detach().numpy()
            - np.array([-0.1019, -0.1258, -0.0777, -0.0484])
        )
        < 1e-4
    )


def test_centroid_model(config, tmp_path: str):
    """Test CentroidModel training."""
    OmegaConf.update(
        config,
        "model_config.head_configs.centroid",
        config.model_config.head_configs.centered_instance,
    )
    del config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centroid["confmaps"].part_names

    model = CentroidModel(config, None, "centroid")

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["centroids_confidence_maps"]
    preds = model(input_["image"])

    # check the output shape
    assert preds.shape == (1, 1, 192, 192)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm.squeeze(dim=1))) < 1e-3


def test_single_instance_model(config, tmp_path: str):
    """Test the SingleInstanceModel training."""
    head_config = config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.single_instance", head_config)
    del config.model_config.head_configs.single_instance.confmaps.anchor_part

    OmegaConf.update(config, "model_config.init_weights", "xavier")

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    model = SingleInstanceModel(config, None, "single_instance")

    img = input_["image"]
    img_shape = img.shape[-2:]
    preds = model(img)

    # check the output shape
    assert preds.shape == (
        1,
        2,
        int(
            img_shape[0]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
        int(
            img_shape[1]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
    )

    # check the loss value
    input_["confidence_maps"] = input_["confidence_maps"][:, :, :2, :, :]
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_["confidence_maps"].squeeze(dim=1))) < 1e-3


def test_bottomup_model(config, tmp_path: str):
    """Test BottomUp model training."""
    config_copy = config.copy()

    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": [("part1", "part2")],
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.bottomup["pafs"] = paf
    config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))

    model = BottomUpModel(config, None, "bottomup")

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["PartAffinityFieldsHead"].shape == (1, 2, 96, 96)

    # with edges as None
    config = config_copy
    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": None,
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.bottomup["pafs"] = paf
    config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    skeletons = model_trainer.skeletons
    input_ = next(iter(model_trainer.train_data_loader))

    model = BottomUpModel(config, skeletons, "bottomup")

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["PartAffinityFieldsHead"].shape == (1, 2, 96, 96)
