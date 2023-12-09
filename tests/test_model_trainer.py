import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from typing import Text
import lightning.pytorch as pl
import pytest
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import TopdownConfmapsPipeline

# import wandb
import os
import pytest
from torch import nn
import pandas as pd
from sleap_nn.architectures.model import Model
from sleap_nn.model_trainer import ModelTrainer, TopDownCenteredInstanceModel
from torch.nn.functional import mse_loss


def test_create_data_loader(config, sleap_data_dir):
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    assert isinstance(
        model_trainer.train_data_loader, torch.utils.data.dataloader.DataLoader
    )
    assert isinstance(
        model_trainer.val_data_loader, torch.utils.data.dataloader.DataLoader
    )
    assert model_trainer.test_data_loader is None
    assert len(list(iter(model_trainer.train_data_loader))) == 2
    assert len(list(iter(model_trainer.val_data_loader))) == 2

    config_test = OmegaConf.create(
        {
            "labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
            "general": {
                "keep_keys": [
                    "instance_image",
                    "confidence_maps",
                    "instance",
                    "video_idx",
                    "frame_idx",
                    "instance_bbox",
                ]
            },
            "preprocessing": {
                "anchor_ind": 0,
                "crop_hw": (160, 160),
                "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
            },
            "augmentation_config": {
                "random_crop": {"random_crop_p": 0, "random_crop_hw": (160, 160)},
                "use_augmentations": False,
                "augmentations": {
                    "intensity": {
                        "uniform_noise": (0.0, 0.04),
                        "uniform_noise_p": 0,
                        "gaussian_noise_mean": 0.02,
                        "gaussian_noise_std": 0.004,
                        "gaussian_noise_p": 0,
                        "contrast": (0.5, 2.0),
                        "contrast_p": 0,
                        "brightness": 0.0,
                        "brightness_p": 0,
                    },
                    "geometric": {
                        "rotation": 180.0,
                        "scale": 0,
                        "translate": (0, 0),
                        "affine_p": 0.5,
                        "erase_scale": (0.0001, 0.01),
                        "erase_ratio": (1, 1),
                        "erase_p": 0,
                        "mixup_lambda": None,
                        "mixup_p": 0,
                    },
                },
            },
        },
    )

    OmegaConf.update(config, "data_config.test", config_test)
    print(config.data_config.test)
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    assert isinstance(
        model_trainer.test_data_loader, torch.utils.data.dataloader.DataLoader
    )
    assert len(list(iter(model_trainer.test_data_loader))) == 2


def test_trainer(config):
    model_trainer = ModelTrainer(config)
    model_trainer.train()
    assert all(
        [
            not isinstance(i, L.pytorch.loggers.wandb.WandbLogger)
            for i in model_trainer.logger
        ]
    )

    # disable ckpt, check if ckpt is created
    files = os.listdir(config.trainer_config.save_ckpt_path)
    yaml = False
    for i in files:
        if i.endswith(".yaml"):
            yaml = True
        assert not i.endswith(".ckpt")
    # check if yaml file is created
    assert not yaml

    # update save_ckpt to True
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    # OmegaConf.update(config, "trainer_config.use_wandb", True)
    model_trainer = ModelTrainer(config)
    # model_trainer._set_wandb()
    # assert wandb.run is not None
    model_trainer.train()
    # assert any(
    #     [
    #         isinstance(i, L.pytorch.loggers.wandb.WandbLogger)
    #         for i in model_trainer.logger
    #     ]
    # )

    files = os.listdir(config.trainer_config.save_ckpt_path)
    ckpt = False
    yaml = False
    for i in files:
        if i.endswith("ckpt"):
            ckpt = True
        if i.endswith("yaml"):
            yaml = True
    # check if ckpt is created
    assert ckpt and yaml
    checkpoint = torch.load(os.path.join(config.trainer_config.save_ckpt_path, i))
    assert checkpoint["epoch"] == 1
    labels = sio.load_slp("minimal_instance.pkg.slp")
    # check if skeleton is saved in ckpt file
    assert checkpoint["skeleton"] == labels.skeletons

    # check for training metrics csv
    files = os.listdir(
        os.path.join(config.trainer_config.save_ckpt_path, "lightning_logs/version_0/")
    )
    assert "metrics.csv" in files
    df = pd.read_csv(
        os.path.join(
            config.trainer_config.save_ckpt_path, "lightning_logs/version_0/metrics.csv"
        )
    )
    assert (
        abs(df.loc[0, "learning_rate"] - config.trainer_config.optimizer.learning_rate)
        <= 1e-4
    )
    assert not df.val_loss.isnull().all()
    assert not df.train_loss.isnull().all()


def test_topdown_centered_instance_model(config):
    model = TopDownCenteredInstanceModel(config)
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["confidence_maps"].squeeze(dim=1).to("cuda")
    preds = model(input_)
    # check the output shape
    assert preds.shape == (1, 2, 80, 80)

    # check the loss
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm)) < 1e-3

    # check if cuda is activated
    OmegaConf.update(config, "trainer_config.device", "cuda")
    model = TopDownCenteredInstanceModel(config)
    assert "cuda" in str(model.device)


if __name__ == "__main__":
    pytest.main([f"{__file__}::test_create_data_loader"])
