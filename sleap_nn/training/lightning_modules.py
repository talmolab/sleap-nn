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
from sleap_nn.architectures.model import Model, MultiHeadModel
from loguru import logger
from sleap_nn.training.utils import xavier_init_weights, plot_pred_confmaps_peaks
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
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
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
            on_step=False,
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
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
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
            on_step=False,
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
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
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
            on_step=False,
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
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
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
            on_step=False,
            on_epoch=True,
            logger=True,
        )


class MultiHeadLightningModule(L.LightningModule):
    """Base PyTorch Lightning Module for multi-head models.

    This class is a sub-class of Torch Lightning Module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
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
        self.save_ckpt = self.config.trainer_config.save_ckpt
        self.use_wandb = self.config.trainer_config.use_wandb
        if self.save_ckpt:
            self.results_path = (
                Path(self.config.trainer_config.save_ckpt_path) / "visualizations"
            )
            if not Path(self.results_path).exists():
                Path(self.results_path).mkdir(parents=True, exist_ok=True)
        self.backbone_type = backbone_type
        self.pretrained_backbone_weights = (
            self.config.model_config.pretrained_backbone_weights
        )
        self.pretrained_head_weights = self.config.model_config.pretrained_head_weights
        self.in_channels = self.model_config.backbone_config[f"{self.backbone_type}"][
            "in_channels"
        ]
        self.input_expand_channels = self.in_channels

        # only for swint and convnext
        if self.model_config.pre_trained_weights:
            ckpt = MODEL_WEIGHTS[
                self.model_config.pre_trained_weights
            ].DEFAULT.get_state_dict(progress=True, check_hash=True)
            input_channels = ckpt["features.0.0.weight"].shape[-3]
            if self.in_channels != input_channels:
                self.input_expand_channels = input_channels
                OmegaConf.update(
                    self.model_config,
                    f"backbone_config.{self.backbone_type}.in_channels",
                    input_channels,
                )

        self.model = MultiHeadModel(
            backbone_type=self.backbone_type,
            backbone_config=self.model_config.backbone_config[f"{self.backbone_type}"],
            head_configs=self.model_config.head_configs[self.model_type],
            model_type=self.model_type,
        )

        self.dataset_loss_weights = self.config.get(
            "model_config.dataset_loss_weights",
            {k: 1.0 for k in self.config.dataset_mapper},
        )

        if (
            len(self.model_config.head_configs[self.model_type]) > 1
        ):  # TODO: online mining for each dataset
            self.loss_weights = [
                (
                    self.model_config.head_configs[self.model_type][x][1].loss_weight
                    if self.model_config.head_configs[self.model_type][x][1].loss_weight
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

        self.automatic_optimization = False

        self.loss_func = nn.MSELoss()

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


class TopDownCenteredInstanceMultiHeadLightningModule(MultiHeadLightningModule):
    """Lightning Module for TopDownCenteredInstanceMultiHeadLightningModule Model.

    This is a subclass of the `MultiHeadLightningModule` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance multi-head model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
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
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )
        self.inf_layer = FindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            centered_fitbbox=False,
        )
        self.part_names = {}
        for (
            d_num,
            cfg,
        ) in self.config.model_config.head_configs.centered_instance.confmaps.items():
            self.part_names[d_num] = cfg.part_names

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        # add eval
        if self.config.trainer_config.log_inf_epochs is not None:
            if (
                self.current_epoch > 0
                and self.global_rank == 0
                and (self.current_epoch % self.config.trainer_config.log_inf_epochs)
                == 0
            ):
                img_array = []
                for d_num in self.config.dataset_mapper:
                    sample = next(iter(self.trainer.val_dataloaders[d_num]))
                    sample["eff_scale"] = torch.ones(sample["video_idx"].shape)
                    sample["pad_shifts"] = torch.zeros(
                        (sample["video_idx"].shape[0], 2)
                    )
                    # for fit bbox cropping
                    # sample["eff_scale_crops"] = torch.ones(sample["video_idx"].shape)
                    # sample["padding_shifts_crops"] = torch.zeros(
                    #     (sample["video_idx"].shape[0], 2)
                    # )
                    for k, v in sample.items():
                        sample[k] = v.to(device=self.device)
                    self.inf_layer.output_stride = self.config.model_config.head_configs.centered_instance.confmaps[
                        d_num
                    ][
                        "output_stride"
                    ]
                    output = self.inf_layer(sample, output_head_skeleton_num=d_num)
                    batch_idx = 0

                    # plot predictions on sample image
                    if self.use_wandb or self.save_ckpt:
                        peaks = output["pred_instance_peaks"][batch_idx].cpu().numpy()
                        gt_instances = sample["instance"][batch_idx, 0].cpu().numpy()
                        img = output["instance_image"][batch_idx, 0].cpu().numpy()
                        confmaps = output["pred_confmaps"][batch_idx].cpu().numpy()
                        fig = plot_pred_confmaps_peaks(
                            img=img,
                            confmaps=confmaps,
                            peaks=np.expand_dims(peaks, axis=0),
                            gt_instances=np.expand_dims(gt_instances, axis=0),
                            plot_title=f"{self.config.dataset_mapper[d_num]}",
                        )

                    if self.save_ckpt:
                        curr_results_path = (
                            Path(self.config.trainer_config.save_ckpt_path)
                            / "visualizations"
                            / f"epoch_{self.current_epoch}"
                        )
                        if not Path(curr_results_path).exists():
                            Path(curr_results_path).mkdir(parents=True, exist_ok=True)
                        fig.savefig(
                            (Path(curr_results_path) / f"pred_on_{d_num}").as_posix(),
                            bbox_inches="tight",
                        )

                    if self.use_wandb:
                        fig.canvas.draw()
                        img = Image.frombytes(
                            "RGB",
                            fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb(),
                        )

                        img_array.append(wandb.Image(img))

                    plt.close(fig)

                if self.use_wandb and img_array:
                    # wandb logging metrics in table

                    wandb_table = wandb.Table(
                        columns=[
                            "epoch",
                            "Predictions on test set",
                        ],
                        data=[[self.current_epoch, img_array]],
                    )
                    wandb.log({"Performance": wandb_table})

        self.train_start_time = time.time()

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CenteredInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = 0
        opt = self.optimizers()
        opt.zero_grad()
        for d_num in batch.keys():
            batch_data = batch[d_num]
            X, y = torch.squeeze(batch_data["instance_image"], dim=1).to(
                self.device
            ), torch.squeeze(batch_data["confidence_maps"], dim=1)

            output = self.model(X)["CenteredInstanceConfmapsHead"]

            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output[h_num] = output[h_num].detach()

            y_preds = output[d_num]

            for c in range(y.shape[-3]):
                l = self.loss_func(y_preds[..., c, :, :], y[..., c, :, :])
                self.log(
                    f"node_inv_losses_dataset:{d_num}_node:`{self.part_names[d_num][c]}`",
                    1 / l,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

            curr_loss = self.dataset_loss_weights[d_num] * self.loss_func(y_preds, y)
            loss += curr_loss

            self.manual_backward(curr_loss, retain_graph=True)

            self.log(
                f"train_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        total_loss = 0
        for d_num in batch.keys():
            X, y = torch.squeeze(batch[d_num]["instance_image"], dim=1).to(
                self.device
            ), torch.squeeze(batch[d_num]["confidence_maps"], dim=1)

            y_preds = self.model(X)["CenteredInstanceConfmapsHead"][d_num]
            curr_loss = self.dataset_loss_weights[d_num] * nn.MSELoss()(y_preds, y)
            total_loss += curr_loss

            self.log(
                f"val_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )


class SingleInstanceMultiHeadLightningModule(MultiHeadLightningModule):
    """Lightning Module for SingleInstanceMultiHeadLightningModule Model.

    This is a subclass of the `MultiHeadLightningModule` to configure the training/ validation steps
    and forward pass specific to single-instance multi-head model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
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
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )
        self.single_instance_inf_layer = SingleInstanceInferenceModel(
            torch_model=self.forward,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["SingleInstanceConfmapsHead"]

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        # add eval
        if self.config.trainer_config.log_inf_epochs is not None:
            if (
                self.current_epoch > 0
                and self.global_rank == 0
                and (self.current_epoch % self.config.trainer_config.log_inf_epochs)
                == 0
            ):
                img_array = []
                for d_num in self.config.dataset_mapper:
                    sample = next(iter(self.trainer.val_dataloaders[d_num]))
                    sample["eff_scale"] = torch.ones(sample["video_idx"].shape)
                    for k, v in sample.items():
                        sample[k] = v.to(device=self.device)
                    self.single_instance_inf_layer.output_head_skeleton_num = d_num
                    self.single_instance_inf_layer.output_stride = (
                        self.config.model_config.head_configs.single_instance.confmaps[
                            d_num
                        ]["output_stride"]
                    )
                    output = self.single_instance_inf_layer(sample)
                    batch_idx = 0

                    # plot predictions on sample image
                    if self.use_wandb or self.save_ckpt:
                        peaks = output["pred_instance_peaks"][batch_idx].cpu().numpy()
                        img = output["image"][batch_idx, 0].cpu().numpy()
                        gt_instances = sample["instances"][batch_idx, 0].cpu().numpy()
                        confmaps = output["pred_confmaps"][batch_idx].cpu().numpy()
                        fig = plot_pred_confmaps_peaks(
                            img=img,
                            confmaps=confmaps,
                            peaks=np.expand_dims(peaks, axis=0),
                            gt_instances=np.expand_dims(gt_instances, axis=0),
                            plot_title=f"{self.config.dataset_mapper[d_num]}",
                        )

                    if self.save_ckpt:
                        curr_results_path = (
                            Path(self.config.trainer_config.save_ckpt_path)
                            / "visualizations"
                            / f"epoch_{self.current_epoch}"
                        )
                        if not Path(curr_results_path).exists():
                            Path(curr_results_path).mkdir(parents=True, exist_ok=True)
                        fig.savefig(
                            (Path(curr_results_path) / f"pred_on_{d_num}").as_posix(),
                            bbox_inches="tight",
                        )

                    if self.use_wandb:
                        fig.canvas.draw()
                        img = Image.frombytes(
                            "RGB",
                            fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb(),
                        )

                        img_array.append(wandb.Image(img))

                    plt.close(fig)

                if self.use_wandb and img_array:
                    # wandb logging metrics in table

                    wandb_table = wandb.Table(
                        columns=[
                            "epoch",
                            "Predictions on test set",
                        ],
                        data=[[self.current_epoch, img_array]],
                    )
                    wandb.log({"Performance": wandb_table})

        self.train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = 0
        opt = self.optimizers()
        opt.zero_grad()
        for d_num in batch.keys():
            batch_data = batch[d_num]
            X, y = torch.squeeze(batch_data["image"], dim=1).to(
                self.device
            ), torch.squeeze(batch_data["confidence_maps"], dim=1)

            output = self.model(X)["SingleInstanceConfmapsHead"]

            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output[h_num] = output[h_num].detach()

            y_preds = output[d_num]
            curr_loss = self.dataset_loss_weights[d_num] * self.loss_func(y_preds, y)
            loss += curr_loss

            self.manual_backward(curr_loss, retain_graph=True)

            self.log(
                f"train_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        total_loss = 0
        for d_num in batch.keys():
            X, y = torch.squeeze(batch[d_num]["image"], dim=1).to(
                self.device
            ), torch.squeeze(batch[d_num]["confidence_maps"], dim=1)

            y_preds = self.model(X)["SingleInstanceConfmapsHead"][d_num]
            curr_loss = self.dataset_loss_weights[d_num] * nn.MSELoss()(y_preds, y)
            total_loss += curr_loss

            self.log(
                f"val_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )


class CentroidMultiHeadLightningModule(MultiHeadLightningModule):
    """Lightning Module for CentroidMultiHeadLightningModule Model.

    This is a subclass of the `MultiHeadLightningModule` to configure the training/ validation steps
    and forward pass specific to centroid multi-head model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
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
        super().__init__(
            config=config,
            backbone_type=backbone_type,
            model_type=model_type,
        )
        self.centroid_inf_layer = CentroidCrop(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.config.model_config.head_configs.centroid.confmaps[0][
                "output_stride"
            ],
            input_scale=1.0,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CentroidConfmapsHead"]

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        # add eval
        if self.config.trainer_config.log_inf_epochs is not None:
            if (
                self.current_epoch > 0
                and self.global_rank == 0
                and (self.current_epoch % self.config.trainer_config.log_inf_epochs)
                == 0
            ):
                img_array = []
                for d_num in self.config.dataset_mapper:
                    sample = next(iter(self.trainer.val_dataloaders[d_num]))
                    gt_centroids = sample["centroids"]
                    sample["eff_scale"] = torch.ones(sample["video_idx"].shape)
                    sample["pad_shifts"] = torch.zeros(
                        (sample["video_idx"].shape[0], 2)
                    )
                    for k, v in sample.items():
                        sample[k] = v.to(device=self.device)
                    output = self.centroid_inf_layer(sample)
                    batch_idx = 1

                    # plot predictions on sample image
                    if self.use_wandb or self.save_ckpt:
                        centroids = output["centroids"][batch_idx, 0].cpu().numpy()
                        img = output["image"][batch_idx, 0].cpu().numpy()
                        confmaps = (
                            output["pred_centroid_confmaps"][batch_idx].cpu().numpy()
                        )
                        gt_centroids = gt_centroids[batch_idx, 0].cpu().numpy()
                        fig = plot_pred_confmaps_peaks(
                            img=img,
                            confmaps=confmaps,
                            peaks=np.expand_dims(centroids, axis=0),
                            gt_instances=np.expand_dims(gt_centroids, axis=0),
                            plot_title=f"{self.config.dataset_mapper[d_num]}",
                        )
                    if self.save_ckpt:
                        curr_results_path = (
                            Path(self.config.trainer_config.save_ckpt_path)
                            / "visualizations"
                            / f"epoch_{self.current_epoch}"
                        )
                        if not Path(curr_results_path).exists():
                            Path(curr_results_path).mkdir(parents=True, exist_ok=True)
                        fig.savefig(
                            (Path(curr_results_path) / f"pred_on_{d_num}").as_posix(),
                            bbox_inches="tight",
                        )

                    if self.use_wandb:
                        fig.canvas.draw()
                        img = Image.frombytes(
                            "RGB",
                            fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb(),
                        )

                        img_array.append(wandb.Image(img))

                    plt.close(fig)

                if self.use_wandb and img_array:
                    # wandb logging metrics in table

                    wandb_table = wandb.Table(
                        columns=[
                            "epoch",
                            "Predictions on test set",
                        ],
                        data=[[self.current_epoch, img_array]],
                    )
                    wandb.log({"Performance": wandb_table})

        self.train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = 0
        opt = self.optimizers()
        opt.zero_grad()
        for d_num in batch.keys():
            batch_data = batch[d_num]
            X, y = torch.squeeze(batch_data["image"], dim=1).to(
                self.device
            ), torch.squeeze(batch_data["centroids_confidence_maps"], dim=1).to(
                self.device
            )

            output = self.model(X)["CentroidConfmapsHead"]


            y_preds = output[0]
            curr_loss = self.dataset_loss_weights[d_num] * self.loss_func(y_preds, y)
            loss += curr_loss

            self.manual_backward(curr_loss, retain_graph=True)

            self.log(
                f"train_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        total_loss = 0
        for d_num in batch.keys():
            X, y = torch.squeeze(batch[d_num]["image"], dim=1).to(
                self.device
            ), torch.squeeze(batch[d_num]["centroids_confidence_maps"], dim=1).to(
                self.device
            )

            y_preds = self.model(X)["CentroidConfmapsHead"][0]
            curr_loss = self.dataset_loss_weights[d_num] * nn.MSELoss()(y_preds, y)
            total_loss += curr_loss

            self.log(
                f"val_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )


class BottomUpMultiHeadLightningModule(MultiHeadLightningModule):
    """Lightning Module for BottomUpMultiHeadLightningModule Model.

    This is a subclass of the `MultiHeadLightningModule` to configure the training/ validation steps
    and forward pass specific to bottom up multi-head model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
        skeletons_dict: Dict of `sio.Skeleton` objects from the input `.slp` file for all the datasets.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons_dict: List[sio.Skeleton],
        model_type: str,
        backbone_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            skeletons_dict=skeletons_dict,
            backbone_type=backbone_type,
            model_type=model_type,
        )
        paf_scorer = PAFScorer(
            part_names=self.config.model_config.head_configs.bottomup.confmaps[0][
                "part_names"
            ],
            edges=self.config.model_config.head_configs.bottomup.pafs[0]["edges"],
            pafs_stride=self.config.model_config.head_configs.bottomup.pafs[0][
                "output_stride"
            ],
        )
        self.inf_layer = BottomUpInferenceModel(
            torch_model=self.forward,
            paf_scorer=paf_scorer,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
            cms_output_stride=self.config.model_config.head_configs.bottomup.confmaps[
                0
            ]["output_stride"],
            pafs_output_stride=self.config.model_config.head_configs.bottomup.pafs[0][
                "output_stride"
            ],
        )

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        # add eval
        if self.config.trainer_config.log_inf_epochs is not None:
            if (
                self.current_epoch > 0
                and self.global_rank == 0
                and (self.current_epoch % self.config.trainer_config.log_inf_epochs)
                == 0
            ):
                img_array = []
                for d_num in self.config.dataset_mapper:
                    sample = next(iter(self.trainer.val_dataloaders[d_num]))
                    sample["eff_scale"] = torch.ones(sample["video_idx"].shape)
                    for k, v in sample.items():
                        sample[k] = v.to(device=self.device)

                    paf_scorer = PAFScorer(
                        part_names=self.config.model_config.head_configs.bottomup.confmaps[
                            d_num
                        ][
                            "part_names"
                        ],
                        edges=self.config.model_config.head_configs.bottomup.pafs[
                            d_num
                        ]["edges"],
                        pafs_stride=self.config.model_config.head_configs.bottomup.pafs[
                            d_num
                        ]["output_stride"],
                    )
                    self.inf_layer.paf_scorer = paf_scorer
                    self.inf_layer.cms_output_stride = (
                        self.config.model_config.head_configs.bottomup.confmaps[d_num][
                            "output_stride"
                        ]
                    )
                    self.inf_layer.pafs_output_stride = (
                        self.config.model_config.head_configs.bottomup.pafs[d_num][
                            "output_stride"
                        ]
                    )

                    output = self.inf_layer(sample, output_head_skeleton_num=d_num)
                    batch_idx = 0

                    # plot predictions on sample image
                    if self.use_wandb or self.save_ckpt:
                        peaks = output["pred_instance_peaks"][batch_idx].cpu().numpy()
                        img = output["image"][batch_idx, 0].cpu().numpy()
                        confmaps = output["pred_confmaps"][batch_idx].cpu().numpy()
                        gt_instances = sample["instances"][batch_idx, 0].cpu().numpy()
                        fig = plot_pred_confmaps_peaks(
                            img=img,
                            confmaps=confmaps,
                            peaks=peaks,
                            gt_instances=gt_instances,
                            plot_title=f"{self.config.dataset_mapper[d_num]}",
                        )
                        plt.imshow(
                            output["image"][batch_idx, 0]
                            .cpu()
                            .numpy()
                            .transpose(1, 2, 0)
                        )
                        plt.plot(
                            peaks[:, 0],
                            peaks[:, 1],
                            "rx",
                            label="Predicted",
                        )
                        plt.legend()
                        plt.title(f"{self.config.dataset_mapper[d_num]}")
                        plt.axis("off")

                    if self.save_ckpt:
                        curr_results_path = (
                            Path(self.config.trainer_config.save_ckpt_path)
                            / "visualizations"
                            / f"epoch_{self.current_epoch}"
                        )
                        if not Path(curr_results_path).exists():
                            Path(curr_results_path).mkdir(parents=True, exist_ok=True)
                        plt.savefig(
                            (Path(curr_results_path) / f"pred_on_{d_num}").as_posix()
                        )

                    if self.use_wandb:
                        fig = plt.gcf()
                        fig.canvas.draw()
                        img = Image.frombytes(
                            "RGB",
                            fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb(),
                        )

                        img_array.append(wandb.Image(img))

                    plt.close(fig)

                if self.use_wandb and img_array:
                    # wandb logging metrics in table

                    wandb_table = wandb.Table(
                        columns=[
                            "epoch",
                            "Predictions on test set",
                        ],
                        data=[[self.current_epoch, img_array]],
                    )
                    wandb.log({"Performance": wandb_table})

        self.train_start_time = time.time()

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
        loss = 0
        opt = self.optimizers()
        opt.zero_grad()
        for d_num in batch.keys():
            batch_data = batch[d_num]
            X = torch.squeeze(batch_data["image"], dim=1)
            y_confmap = torch.squeeze(batch_data["confidence_maps"], dim=1).to(
                self.device
            )
            y_paf = batch_data["part_affinity_fields"]

            output = self.model(X)
            output_confmaps = output["MultiInstanceConfmapsHead"]
            output_pafs = output["PartAffinityFieldsHead"]

            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output_confmaps[h_num] = output_confmaps[h_num].detach()
                        output_pafs[h_num] = output_pafs[h_num].detach()

            losses = {
                "MultiInstanceConfmapsHead": nn.MSELoss()(
                    output_confmaps[d_num], y_confmap
                ),
                "PartAffinityFieldsHead": nn.MSELoss()(output_pafs[d_num], y_paf),
            }
            curr_loss = self.dataset_loss_weights[d_num] * sum(
                [s * losses[t] for s, t in zip(self.loss_weights, losses)]
            )

            loss += curr_loss

            self.manual_backward(curr_loss, retain_graph=True)

            self.log(
                f"train_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        total_loss = 0
        for d_num in batch.keys():
            batch_data = batch[d_num]
            X = torch.squeeze(batch_data["image"], dim=1)
            y_confmap = torch.squeeze(batch_data["confidence_maps"], dim=1).to(
                self.device
            )
            y_paf = batch_data["part_affinity_fields"]

            output = self.model(X)
            output_confmaps = output["MultiInstanceConfmapsHead"]
            output_pafs = output["PartAffinityFieldsHead"]

            losses = {
                "MultiInstanceConfmapsHead": nn.MSELoss()(
                    output_confmaps[d_num], y_confmap
                ),
                "PartAffinityFieldsHead": nn.MSELoss()(output_pafs[d_num], y_paf),
            }

            curr_loss = self.dataset_loss_weights[d_num] * sum(
                [s * losses[t] for s, t in zip(self.loss_weights, losses)]
            )
            total_loss += curr_loss

            self.log(
                f"val_loss_on_head_{d_num}",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.log(
            f"val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
