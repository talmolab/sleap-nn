"""This module has the LightningModule classes for all model types."""

from typing import Optional, List
import time
from torch import nn
import numpy as np
import torch
from pathlib import Path
import sleap_io as sio
from omegaconf import OmegaConf
import lightning as L
from PIL import Image
import wandb
from torchvision.models.swin_transformer import (
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)
from torchvision.models.convnext import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Large_Weights,
)
import sleap_io as sio
from sleap_nn.inference.topdown import CentroidCrop, FindInstancePeaks
from sleap_nn.inference.single_instance import SingleInstanceInferenceModel
from sleap_nn.inference.bottomup import BottomUpInferenceModel
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.architectures.model import Model
from loguru import logger
from sleap_nn.training.utils import xavier_init_weights
import matplotlib.pyplot as plt

MODEL_WEIGHTS = {
    "Swin_T_Weights": Swin_T_Weights,
    "Swin_S_Weights": Swin_S_Weights,
    "Swin_B_Weights": Swin_B_Weights,
    "Swin_V2_T_Weights": Swin_V2_T_Weights,
    "Swin_V2_S_Weights": Swin_V2_S_Weights,
    "Swin_V2_B_Weights": Swin_V2_B_Weights,
    "ConvNeXt_Base_Weights": ConvNeXt_Base_Weights,
    "ConvNeXt_Tiny_Weights": ConvNeXt_Tiny_Weights,
    "ConvNeXt_Small_Weights": ConvNeXt_Small_Weights,
    "ConvNeXt_Large_Weights": ConvNeXt_Large_Weights,
}


class BaseLightningModule(L.LightningModule):
    """Base PyTorch Lightning Module for all sleap-nn models.

    This class is a sub-class of Torch Lightning Module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                a pipeline class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
    """

    def __init__(
        self,
        config: OmegaConf,
        model_type: str,
        backbone_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.trainer_config = self.config.trainer_config
        self.data_config = self.config.data_config
        self.model_type = model_type
        self.backbone_type = backbone_type
        self.pretrained_backbone_weights = (
            self.config.model_config.pretrained_backbone_weights
        )
        self.pretrained_head_weights = self.config.model_config.pretrained_head_weights
        self.in_channels = self.model_config.backbone_config[f"{self.backbone_type}"][
            "in_channels"
        ]
        self.input_expand_channels = self.in_channels
        if self.model_config.pre_trained_weights:  # only for swint and convnext
            ckpt = MODEL_WEIGHTS[
                self.model_config.pre_trained_weights
            ].DEFAULT.get_state_dict(progress=True, check_hash=True)
            input_channels = ckpt["features.0.0.weight"].shape[-3]
            if self.in_channels != input_channels:  # TODO: not working!
                self.input_expand_channels = input_channels
                OmegaConf.update(
                    self.model_config,
                    f"backbone_config.{self.backbone_type}.in_channels",
                    input_channels,
                )

        self.model = Model(
            backbone_type=self.backbone_type,
            backbone_config=self.model_config.backbone_config[f"{self.backbone_type}"],
            head_configs=self.model_config.head_configs[self.model_type],
            model_type=self.model_type,
        )

        if len(self.model_config.head_configs[self.model_type]) > 1:
            self.loss_weights = [
                (
                    self.model_config.head_configs[self.model_type][x].loss_weight
                    if self.model_config.head_configs[self.model_type][x].loss_weight
                    is not None
                    else 1.0
                )
                for x in self.model_config.head_configs[self.model_type]
            ]

        self.training_loss = {}
        self.val_loss = {}
        self.learning_rate = {}

        # Initialization for encoder and decoder stacks.
        if self.model_config.init_weights == "xavier":
            self.model.apply(xavier_init_weights)

        # Pre-trained weights for the encoder stack - only for swint and convnext
        if self.model_config.pre_trained_weights:
            self.model.backbone.enc.load_state_dict(ckpt, strict=False)

        # TODO: Handling different input channels
        # Initializing backbone (encoder + decoder) with trained ckpts
        if self.pretrained_backbone_weights is not None:
            logger.info(
                f"Loading backbone weights from `{self.pretrained_backbone_weights}` ..."
            )
            ckpt = torch.load(self.pretrained_backbone_weights)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            self.load_state_dict(ckpt["state_dict"], strict=False)

        # Initializing head layers with trained ckpts.
        if self.pretrained_head_weights is not None:
            logger.info(
                f"Loading head weights from `{self.pretrained_head_weights}` ..."
            )
            ckpt = torch.load(self.pretrained_head_weights)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            self.load_state_dict(ckpt["state_dict"], strict=False)

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
        if self.trainer_config.optimizer_name == "Adam":
            optim = torch.optim.Adam
        elif self.trainer_config.optimizer_name == "AdamW":
            optim = torch.optim.AdamW

        optimizer = optim(
            self.parameters(),
            lr=self.trainer_config.optimizer.lr,
            amsgrad=self.trainer_config.optimizer.amsgrad,
        )

        scheduler = None
        for k, v in self.trainer_config.lr_scheduler.items():
            if v is not None:
                if k == "step_lr":
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=self.trainer_config.lr_scheduler.step_lr.step_size,
                        gamma=self.trainer_config.lr_scheduler.step_lr.gamma,
                    )
                    break
                elif k == "reduce_lr_on_plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        threshold=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.threshold,
                        threshold_mode=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.threshold_mode,
                        cooldown=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.cooldown,
                        patience=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.patience,
                        factor=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.factor,
                        min_lr=self.trainer_config.lr_scheduler.reduce_lr_on_plateau.min_lr,
                    )
                    break

        if scheduler is None:
            return {
                "optimizer": optimizer,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class SingleInstanceLightningModule(BaseLightningModule):
    """Lightning Module for SingleInstance Model.

    This is a subclass of the `BaseLightningModule` to configure the training/ validation steps and
    forward pass specific to Single Instance model.

    Args:
        config: OmegaConf dictionary which has the following:
            (i) data_config: data loading pre-processing configs to be passed to
            `TopdownConfmapsPipeline` class.
            (ii) model_config: backbone and head configs to be passed to `Model` class.
            (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            model_type=model_type,
            backbone_type=backbone_type,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["SingleInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
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
            on_step=True,
            on_epoch=True,
            logger=True,
        )


class TopDownCenteredInstanceLightningModule(BaseLightningModule):
    """Lightning Module for TopDownCenteredInstance Model.

    This is a subclass of the `BaseLightningModule` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CenteredInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
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
            on_step=True,
            on_epoch=True,
            logger=True,
        )


class CentroidLightningModule(BaseLightningModule):
    """Lightning Module for Centroid Model.

    This is a subclass of the `BaseLightningModule` to configure the training/ validation steps
    and forward pass specific to centroid model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `CentroidConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CentroidConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        )

        y_preds = self.model(X)["CentroidConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        )

        y_preds = self.model(X)["CentroidConfmapsHead"]
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
            on_step=True,
            on_epoch=True,
            logger=True,
        )


class BottomUpLightningModule(BaseLightningModule):
    """Lightning Module for BottomUp Model.

    This is a subclass of the `BaseLightningModule` to configure the training/ validation steps
    and forward pass specific to BottomUp model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `BottomUpPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        output = self.model(img)
        return {
            "MultiInstanceConfmapsHead": output["MultiInstanceConfmapsHead"],
            "PartAffinityFieldsHead": output["PartAffinityFieldsHead"],
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        X = torch.squeeze(batch["image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_paf = batch["part_affinity_fields"]
        preds = self.model(X)
        pafs = preds["PartAffinityFieldsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]
        losses = {
            "MultiInstanceConfmapsHead": nn.MSELoss()(confmaps, y_confmap),
            "PartAffinityFieldsHead": nn.MSELoss()(pafs, y_paf),
        }
        loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X = torch.squeeze(batch["image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_paf = batch["part_affinity_fields"]

        preds = self.model(X)
        pafs = preds["PartAffinityFieldsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]
        losses = {
            "MultiInstanceConfmapsHead": nn.MSELoss()(confmaps, y_confmap),
            "PartAffinityFieldsHead": nn.MSELoss()(pafs, y_paf),
        }
        val_loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
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
            on_step=True,
            on_epoch=True,
            logger=True,
        )
