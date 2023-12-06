"""This module is to train a sleap-nn model using Lightning."""

import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from typing import Text
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
        train_datapipe = train_pipeline.make_training_pipeline(
            data_provider=self.provider,
            filename=self.config.data_config.train.labels_path,
        )

        val_pipeline = TopdownConfmapsPipeline(
            data_config=self.config.data_config.valid
        )
        val_datapipe = val_pipeline.make_training_pipeline(
            data_provider=LabelsReader,
            filename=self.config.data_config.valid.labels_path,
        )

        use_test_data = "test" in self.config.data_config.keys()
        if use_test_data:
            test_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.test
            )
            test_datapipe = test_pipeline.make_training_pipeline(
                data_provider=LabelsReader,
                filename=self.config.data_config.test.labels_path,
            )
            test_datapipe = test_datapipe.sharding_filter()

        # to remove duplicates when multiprocessing is used
        train_datapipe = train_datapipe.sharding_filter()
        val_datapipe = val_datapipe.sharding_filter()

        num_cores = self.config.trainer_config.num_cores
        # create torch Data Loaders
        self.train_data_loader = DataLoader(
            train_datapipe,
            batch_size=self.config.trainer_config.train_batch_size,
            shuffle=True,
            num_workers=num_cores,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
        self.val_data_loader = DataLoader(
            val_datapipe,
            batch_size=self.config.trainer_config.val_batch_size,
            shuffle=False,
            num_workers=num_cores,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
        self.test_data_loader = None
        if use_test_data:
            self.test_data_loader = DataLoader(
                test_datapipe,
                batch_size=self.config.trainer_config.test_batch_size,
                shuffle=False,
                num_workers=num_cores,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2,
            )

    def _set_wandb(self):
        wandb.login()
        self.wandb_logger = WandbLogger(
            project=self.config.trainer_config.wandb.project,
            name=self.config.trainer_config.wandb.entity_name,
        )

    def train(self):
        """Function to initiate the training by calling the fit method of Trainer"""
        self._create_data_loaders()
        dir_path = self.config.trainer_config.save_ckpt_path
        # create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            every_n_epochs=0,
            monitor="val_loss",
            mode="min",
            dirpath=dir_path,
            filename="{epoch:02d}-{val_loss:.2f}",
        )
        enable_checkpoint = True
        callbacks = [checkpoint_callback]
        # default logger to create csv with metrics values over the epochs
        csv_logger = CSVLogger(dir_path)
        self.logger = [csv_logger]
        if self.config.trainer_config.use_wandb:
            self._set_wandb()
            self.logger = [self.wandb_logger, csv_logger]
        if not self.config.trainer_config.save_ckpt:
            enable_checkpoint = False
            callbacks = []
        model = TopDownCenteredInstanceModel(self.config)
        trainer = L.Trainer(
            callbacks=callbacks,
            logger=self.logger,
            enable_checkpointing=enable_checkpoint,
            devices=self.config.trainer_config.devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
        )

        trainer.fit(model, self.train_data_loader, self.val_data_loader)
        # save the configs as yaml in the checkpoint dir
        OmegaConf.save(config=self.config, f=dir_path + "config.yaml")

        wandb.finish()


def xavier_init_weights(x):
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
        return next(self.model.parameters()).device

    def forward(self, inputs):
        """Forward pass of the model."""
        imgs = inputs["instance_image"].to(self.m_device)
        return self.model(imgs)["CenteredInstanceConfmapsHead"]

    def on_save_checkpoint(self, checkpoint):
        """Configure checkpoint to save parameters."""
        labels_gt = sio.load_slp(self.data_config.train.labels_path)
        # save the skeletons and seed to the checkpoint file
        checkpoint["skeleton"] = labels_gt.skeletons
        checkpoint["seed"] = self.seed

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch["instance_image"], batch["confidence_maps"]

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
        X, y = batch["instance_image"].to(self.m_device), batch["confidence_maps"].to(
            self.m_device
        )

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
            lr=self.trainer_config.optimizer.learning_rate,
            amsgrad=self.trainer_config.optimizer.use_amsgrad,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            threshold=self.trainer_config.optimizer.lr_scheduler_threshold,
            threshold_mode="abs",
            cooldown=self.trainer_config.optimizer.lr_scheduler_cooldown,
            patience=self.trainer_config.optimizer.lr_scheduler_patience,
            factor=self.trainer_config.optimizer.lr_reduction_factor,
            min_lr=self.trainer_config.optimizer.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
