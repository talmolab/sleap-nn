import torch
import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import pandas as pd
from sleap_nn.model_trainer import ModelTrainer, TopDownCenteredInstanceModel
from torch.nn.functional import mse_loss
import os
import wandb
from lightning.pytorch.loggers import WandbLogger


def test_create_data_loader(config, tmp_path: str):
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


def test_wandb():
    # check for wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb_logger = WandbLogger()
    assert wandb.run is not None
    wandb.finish()


def test_trainer(config, tmp_path: str):
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
    assert not folder_created

    # update save_ckpt to True
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "model_config.init_weights", "xavier")

    model_trainer = ModelTrainer(config)
    model_trainer.train()

    # check if wandb folder is created
    print(
        "------------------", list(Path(config.trainer_config.save_ckpt_path).iterdir())
    )
    assert Path(config.trainer_config.save_ckpt_path).joinpath("wandb").exists()

    folder_created = Path(config.trainer_config.save_ckpt_path).exists()
    assert folder_created
    files = [
        str(x)
        for x in Path(config.trainer_config.save_ckpt_path).iterdir()
        if x.is_file()
    ]
    ckpt = False
    yaml = False
    for i in files:
        if i.endswith("ckpt"):
            ckpt = True
        if i.endswith("yaml"):
            yaml = True

    # check if ckpt is created
    assert ckpt and yaml
    checkpoint = torch.load(Path(config.trainer_config.save_ckpt_path).joinpath(i))
    assert checkpoint["epoch"] == 1

    # check if skeleton is saved in ckpt file
    assert isinstance(checkpoint["skeleton"][0], sio.Skeleton)
    assert checkpoint["config"]

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


def test_topdown_centered_instance_model(config, tmp_path: str):
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
