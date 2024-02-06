"""This module is to train a sleap-nn model using Lightning."""

import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from typing import Text
from pathlib import Path
from omegaconf import OmegaConf
import lightning as L
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
import wandb
import time
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch import nn
import pandas as pd
from sleap_nn.architectures.model import Model
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def xavier_init_weights(x):
    """Function to initilaise the model weights with Xavier initialization method."""
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        nn.init.constant_(x.bias, 0)


class ModelTrainer:
    """Train sleap-nn model using PyTorch Lightning.

    This class is used to train a sleap-nn model and save the model checkpoints/ logs with options to logging
    with wandb and csvlogger.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.

    """

    def __init__(self, config: OmegaConf):
        """Initialise the class with configs and set the seed and device as class attributes."""
        self.config = config

        self.m_device = self.config.trainer_config.device
        self.seed = self.config.trainer_config.seed
        # set seed
        torch.manual_seed(self.seed)

        self.is_single_instance_model = False
        if self.config.data_config.pipeline == "SingleInstanceConfmaps":
            self.is_single_instance_model = True

    def _create_data_loaders(self):
        """Creates a DataLoader for train, validation and test sets using the data_config."""
        self.provider = self.config.data_config.provider
        if self.provider == "LabelsReader":
            self.provider = LabelsReader

        # create pipelines
        if self.is_single_instance_model:
            train_pipeline = SingleInstanceConfmapsPipeline(
                data_config=self.config.data_config.train
            )
            val_pipeline = SingleInstanceConfmapsPipeline(
                data_config=self.config.data_config.val
            )
            max_instances = 1

        elif self.config.data_config.pipeline == "TopdownConfmaps":
            train_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.train
            )
            val_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.val
            )
            max_instances = self.config.data_config.max_instances

        else:
            raise Exception(f"{self.config.data_config.pipeline} is not defined.")

        # train
        train_labels = sio.load_slp(self.config.data_config.train.labels_path)
        self.skeletons = train_labels.skeletons

        train_labels_reader = self.provider(train_labels, max_instances=max_instances)
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

        # val
        val_labels_reader = self.provider(
            sio.load_slp(self.config.data_config.val.labels_path),
            max_instances=max_instances,
        )
        val_datapipe = val_pipeline.make_training_pipeline(
            data_provider=val_labels_reader,
        )
        val_datapipe = val_datapipe.sharding_filter()
        self.val_data_loader = DataLoader(
            val_datapipe,
            **dict(self.config.trainer_config.val_data_loader),
        )

    def _set_wandb(self):
        wandb.login(key=self.config.trainer_config.wandb.api_key)

    def _initialize_model(self):
        if self.is_single_instance_model:
            self.model = SingleInstanceModel(self.config)
        else:
            self.model = TopDownCenteredInstanceModel(self.config)

    def train(self):
        """Function to initiate the training by calling the fit method of Trainer."""
        self._create_data_loaders()
        self.logger = []
        if self.config.trainer_config.save_ckpt:
            if not self.config.trainer_config.save_ckpt_path:
                dir_path = "."
            else:
                dir_path = self.config.trainer_config.save_ckpt_path

            if not Path(dir_path).exists():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    print(
                        f"Cannot create a new folder. Check the permissions to the given Checkpoint directory. \n {e}"
                    )

            # create checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                **dict(self.config.trainer_config.model_ckpt),
                dirpath=dir_path,
                filename="best",
            )
            callbacks = [checkpoint_callback]
            # logger to create csv with metrics values over the epochs
            csv_logger = CSVLogger(dir_path)
            self.logger.append(csv_logger)

        else:
            callbacks = []

        if self.config.trainer_config.use_wandb:
            wandb_config = self.config.trainer_config.wandb
            if wandb_config.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            else:
                self._set_wandb()
            self.wandb_logger = WandbLogger(
                project=wandb_config.project, name=wandb_config.name, save_dir=dir_path
            )
            self.logger.append(self.wandb_logger)

        if self.config.trainer_config.save_ckpt:
            # save the configs as yaml in the checkpoint dir
            self.config.trainer_config.wandb.api_key = ""
            OmegaConf.save(config=self.config, f=f"{dir_path}/initial_config.yaml")

            # save the skeleton in the config
            self.config["data_config"]["skeletons"] = {}
            for skl in self.skeletons:
                if skl.symmetries:
                    symm = [list(s.nodes) for s in skl.symmetries]
                else:
                    symm = None
                self.config["data_config"]["skeletons"][skl.name] = {
                    "nodes": skl.nodes,
                    "edges": skl.edges,
                    "symmetries": symm,
                }

        self._initialize_model()

        trainer = L.Trainer(
            callbacks=callbacks,
            logger=self.logger,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
        )

        trainer.fit(self.model, self.train_data_loader, self.val_data_loader)

        if self.config.trainer_config.use_wandb:
            wandb.finish()

        if self.config.trainer_config.save_ckpt:
            # save the configs as yaml in the checkpoint dir
            OmegaConf.save(config=self.config, f=f"{dir_path}/training_config.yaml")


class TrainingModel(L.LightningModule):
    """Base PyTorch Lightning Module for all sleap-nn models.

    This class is a sub-class of Torch Lightning Module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.

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

    def forward(self, img):
        """Forward pass of the model."""
        pass

    def on_save_checkpoint(self, checkpoint):
        """Configure checkpoint to save parameters."""
        # save the config to the checkpoint file
        checkpoint["config"] = self.config

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        self.train_start_time = time.time()

    def on_train_epoch_end(self):
        """Configure the train timer at the end of every epoch."""
        train_time = time.time() - self.train_start_time
        self.log(
            "train_time",
            train_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_validation_epoch_start(self):
        """Configure the val timer at the beginning of each epoch."""
        self.val_start_time = time.time()

    def on_validation_epoch_end(self):
        """Configure the val timer at the end of every epoch."""
        val_time = time.time() - self.val_start_time
        self.log(
            "val_time",
            val_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        pass

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pass

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


class SingleInstanceModel(TrainingModel):
    """Lightning Module for SingleInstance Model.

    This is a subclass of the `TrainingModel` to configure the training/ validation steps and
    forward pass specific to Single Instance model.

    Args:
        config: OmegaConf dictionary which has the following:
            (i) data_config: data loading pre-processing configs to be passed to
            `TopdownConfmapsPipeline` class.
            (ii) model_config: backbone and head configs to be passed to `Model` class.
            (iii) trainer_config: trainer configs like accelerator, optimiser params.

    """

    def __init__(self, config: OmegaConf):
        """Initialise the configs and the model."""
        super().__init__(config)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1)
        img = img.to(self.m_device)
        return self.model(img)["SingleInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.m_device), torch.squeeze(
            batch["confidence_maps"], dim=1
        ).to(self.m_device)

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
        y = y.to(self.m_device)
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.m_device), torch.squeeze(
            batch["confidence_maps"], dim=1
        ).to(self.m_device)

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
        y = y.to(self.m_device)
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


class TopDownCenteredInstanceModel(TrainingModel):
    """Lightning Module for TopDownCenteredInstance Model.

    This is a subclass of the `TrainingModel` to configure the training/ validation steps and
    forward pass specific to TopDown Centered instance model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.

    """

    def __init__(self, config: OmegaConf):
        """Initialise the configs and the model."""
        super().__init__(config)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1)
        img = img.to(self.m_device)
        return self.model(img)["CenteredInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1).to(
            self.m_device
        ), torch.squeeze(batch["confidence_maps"], dim=1).to(self.m_device)

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        y = y.to(self.m_device)
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1).to(
            self.m_device
        ), torch.squeeze(batch["confidence_maps"], dim=1).to(self.m_device)

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        y = y.to(self.m_device)
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
