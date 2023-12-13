"""This module is to train a sleap-nn model using Lightning."""

import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from typing import Text
import os
import lightning.pytorch as pl
import pandas as pd
from omegaconf import OmegaConf
import lightning as L
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import TopdownConfmapsPipeline
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch import nn
import pandas as pd
from sleap_nn.architectures.model import Model
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelTrainer:
    """Train sleap-nn model using PyTorch Lightning.

    This class is used to train a sleap-nn model and save the model checkpoints/ logs with options to logging
    with wandb and csvlogger.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_cong: trainer configs like accelerator, optimiser params.

    """

    def __init__(self, config: OmegaConf):
        """Initialise the class with configs and set the seed and device as class attributes."""
        self.config = config

        self.m_device = self.config.trainer_config.device
        self.seed = self.config.trainer_config.seed
        # set seed
        torch.manual_seed(self.seed)

    def _create_data_loaders(self):
        """Creates a DataLoader for train, validation and test sets using the data_config."""
        self.provider = self.config.data_config.provider
        if self.provider == "LabelsReader":
            self.provider = LabelsReader

        # create pipelines
        train_pipeline = TopdownConfmapsPipeline(
            data_config=self.config.data_config.train
        )
        train_labels_reader = self.provider(
            sio.load_slp(self.config.data_config.train.labels_path)
        )
        train_datapipe = train_pipeline.make_training_pipeline(
            data_provider=train_labels_reader,
        )

        # to remove duplicates when multiprocessing is used
        train_datapipe = train_datapipe.sharding_filter()
        # create torch Data Loaders
        self.train_data_loader = DataLoader(
            train_datapipe,
            **dict(self.config.trainer_config.train_data_loader),
        )

        val_pipeline = TopdownConfmapsPipeline(data_config=self.config.data_config.val)
        val_labels_reader = self.provider(
            sio.load_slp(self.config.data_config.val.labels_path)
        )
        val_datapipe = val_pipeline.make_training_pipeline(
            data_provider=val_labels_reader,
        )
        val_datapipe = val_datapipe.sharding_filter()
        self.val_data_loader = DataLoader(
            val_datapipe,
            **dict(self.config.trainer_config.val_data_loader),
        )

        self.test_data_loader = None
        use_test_data = "test" in self.config.data_config.keys()
        if use_test_data:
            test_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.test
            )
            test_labels_reader = self.provider(
                sio.load_slp(self.config.data_config.test.labels_path)
            )
            test_datapipe = test_pipeline.make_training_pipeline(
                data_provider=test_labels_reader,
            )
            test_datapipe = test_datapipe.sharding_filter()
            self.test_data_loader = DataLoader(
                test_datapipe,
                **dict(self.config.trainer_config.test_data_loader),
            )

    def _set_wandb(self):
        wandb.login()
        self.wandb_logger = WandbLogger(**dict(self.config.trainer_config.wandb))

    def train(self):
        """Function to initiate the training by calling the fit method of Trainer."""
        self._create_data_loaders()
        self.logger = []
        if self.config.trainer_config.save_ckpt:
            if not self.config.trainer_config.save_ckpt_path:
                dir_path = "./"
            else:
                dir_path = self.config.trainer_config.save_ckpt_path

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # create checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                **dict(self.config.trainer_config.model_ckpt),
                dirpath=dir_path,
            )
            callbacks = [checkpoint_callback]
            # logger to create csv with metrics values over the epochs
            csv_logger = CSVLogger(dir_path)
            self.logger.append(csv_logger)
            OmegaConf.save(config=self.config, f=dir_path + "config.yaml")

        else:
            callbacks = []

        if self.config.trainer_config.use_wandb:
            self._set_wandb()
            self.logger.append(self.wandb_logger)

        model = TopDownCenteredInstanceModel(self.config)
        trainer = L.Trainer(
            callbacks=callbacks,
            logger=self.logger,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
        )

        trainer.fit(model, self.train_data_loader, self.val_data_loader)
        # save the configs as yaml in the checkpoint dir

        wandb.finish()


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


class TopDownCenteredInstanceModel(L.LightningModule):
    """PyTorch Lightning Module.

    This class is a sub-class of Lightning module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_cong: trainer configs like accelerator, optimiser params.

    """

    def __init__(self, config: OmegaConf):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.trainer_config = self.config.trainer_config
        self.data_config = self.config.data_config
        self.m_device = self.trainer_config.device
        self.model = Model(
            backbone_config=self.model_config.backbone_config,
            head_configs=[self.model_config.head_configs],
        ).to(self.m_device)

        # Initializing the model weights
        if self.model_config.init_weights == "xavier":
            self.model.apply(xavier_init_weights)
        self.seed = self.trainer_config.seed
        self.training_loss = {}
        self.val_loss = {}
        self.learning_rate = {}
        torch.manual_seed(self.seed)

    @property
    def device(self):
        """Save the device as an attribute to the class."""
        return next(self.model.parameters()).device

    def forward(self, inputs):
        """Forward pass of the model."""
        imgs = torch.squeeze(inputs["instance_image"], dim=1).to(self.m_device)
        return self.model(imgs)["CenteredInstanceConfmapsHead"]

    def on_save_checkpoint(self, checkpoint):
        """Configure checkpoint to save parameters."""
        labels_gt = sio.load_slp(self.data_config.train.labels_path)
        # save the skeletons and seed to the checkpoint file
        checkpoint["skeleton"] = labels_gt.skeletons
        checkpoint["seed"] = self.seed

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )
        X = X.to(self.m_device)
        y = y.to(self.m_device)

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        self.training_loss[self.current_epoch] = loss.detach().cpu().numpy()
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1).to(
            self.m_device
        ), torch.squeeze(batch["confidence_maps"], dim=1).to(self.m_device)

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        val_loss = nn.MSELoss()(y_preds, y)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.val_loss[self.current_epoch] = val_loss.detach().cpu().numpy()
        self.learning_rate[self.current_epoch] = lr

    def configure_optimizers(self):
        """Configure optimiser and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            **dict(self.trainer_config.optimizer),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **dict(self.trainer_config.lr_scheduler),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
