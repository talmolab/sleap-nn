"""Test ModelTrainer and TrainingModule classes."""

import torch
import sleap_io as sio
from typing import Text
import numpy as np
import pytest
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import pandas as pd
from sleap_nn.model_trainer import (
    ModelTrainer,
    TopDownCenteredInstanceModel,
    SingleInstanceModel,
    CentroidModel,
)
from torch.nn.functional import mse_loss
import os
import wandb
from lightning.pytorch.loggers import WandbLogger
import shutil


def test_create_data_loader(config, tmp_path: str):
    """Test _create_data_loader function of ModelTrainer class."""
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

    OmegaConf.update(config, "data_config.pipeline", "TopDown")
    model_trainer = ModelTrainer(config)
    with pytest.raises(Exception, match="TopDown is not defined."):
        model_trainer._create_data_loaders()

    OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    assert len(list(iter(model_trainer.train_data_loader))) == 1
    assert len(list(iter(model_trainer.val_data_loader))) == 1

    OmegaConf.update(config, "data_config.pipeline", "CentroidConfmaps")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    assert len(list(iter(model_trainer.train_data_loader))) == 1
    assert len(list(iter(model_trainer.val_data_loader))) == 1
    ex = next(iter(model_trainer.train_data_loader))
    assert ex["centroids"].shape == (1, 1, 2, 2)


def test_wandb():
    """Test wandb integration."""
    os.environ["WANDB_MODE"] = "offline"
    wandb_logger = WandbLogger()
    wandb.init()
    assert wandb.run is not None
    wandb.finish()


def test_trainer(config, tmp_path: str):
    # for topdown centered instance model
    model_trainer = ModelTrainer(config)
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer.train()
    assert all(
        [
            not isinstance(i, L.pytorch.loggers.wandb.WandbLogger)
            for i in model_trainer.logger
        ]
    )

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
    OmegaConf.update(config, "model_config.init_weights", "xavier")

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

    # check early stopping
    config_early_stopping = config.copy()
    OmegaConf.update(
        config_early_stopping, "trainer_config.early_stopping.min_delta", 1e-3
    )
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
    OmegaConf.update(
        single_instance_config, "data_config.pipeline", "SingleInstanceConfmaps"
    )
    OmegaConf.update(
        single_instance_config,
        "model_config.head_configs.head_type",
        "SingleInstanceConfmapsHead",
    )
    del single_instance_config.model_config.head_configs.head_config.anchor_part

    trainer = ModelTrainer(single_instance_config)
    trainer._initialize_model()
    assert isinstance(trainer.model, SingleInstanceModel)

    # Centroid model
    centroid_config = config.copy()
    OmegaConf.update(centroid_config, "data_config.pipeline", "CentroidConfmaps")
    OmegaConf.update(
        centroid_config,
        "model_config.head_configs.head_type",
        "CentroidConfmapsHead",
    )

    del centroid_config.model_config.head_configs.head_config.part_names

    if Path(config.trainer_config.save_ckpt_path).exists():
        shutil.rmtree(config.trainer_config.save_ckpt_path)

    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 1)
    OmegaConf.update(centroid_config, "trainer_config.steps_per_epoch", 10)

    trainer = ModelTrainer(centroid_config)
    trainer._initialize_model()
    assert isinstance(trainer.model, CentroidModel)

    trainer.train()

    checkpoint = torch.load(
        Path(centroid_config.trainer_config.save_ckpt_path).joinpath("best.ckpt")
    )
    assert checkpoint["epoch"] == 0
    assert checkpoint["global_step"] == 10


def test_topdown_centered_instance_model(config, tmp_path: str):

    # unet
    model = TopDownCenteredInstanceModel(config)
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
    OmegaConf.update(config, "model_config.backbone_config.backbone_type", "convnext")
    OmegaConf.update(
        config,
        "model_config.backbone_config.backbone_config",
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters_rate": 2,
            "up_blocks": 3,
            "convs_per_block": 2,
            "arch": {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
        },
    )
    model = TopDownCenteredInstanceModel(config)
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
    OmegaConf.update(config, "data_config.pipeline", "CentroidConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "CentroidConfmapsHead"
    )
    del config.model_config.head_configs.head_config.part_names

    model = CentroidModel(config)

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
    assert abs(loss - mse_loss(preds, input_cm)) < 1e-3


def test_single_instance_model(config, tmp_path: str):
    OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
    )
    del config.model_config.head_configs.head_config.anchor_part

    model = SingleInstanceModel(config)
    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    img = input_["image"]
    img_shape = img.shape[-2:]
    preds = model(img)

    # check the output shape
    assert preds.shape == (
        1,
        2,
        int(
            img_shape[0]
            / config.data_config.train.preprocessing.conf_map_gen.output_stride
        ),
        int(
            img_shape[1]
            / config.data_config.train.preprocessing.conf_map_gen.output_stride
        ),
    )

    # check the loss value
    input_["confidence_maps"] = input_["confidence_maps"][:, :, :2, :, :]
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_["confidence_maps"])) < 1e-3
