"""This module has the LightningModule classes for all model types."""

from typing import Optional, List
import time
from torch import nn
import numpy as np
import torch
from pathlib import Path
import sleap_io as sio
from omegaconf import OmegaConf, DictConfig
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
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
    BottomUpMultiClassInferenceModel,
)
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.architectures.model import Model
from sleap_nn.training.losses import compute_ohkm_loss
from loguru import logger
from sleap_nn.training.utils import (
    xavier_init_weights,
    plot_confmaps,
    plot_img,
    plot_peaks,
)
import matplotlib.pyplot as plt
from sleap_nn.config.utils import get_backbone_type_from_cfg, get_model_type_from_cfg

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


class LightningModel(L.LightningModule):
    """Base PyTorch Lightning Module for all sleap-nn models.

    This class is a sub-class of Torch Lightning Module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                a pipeline class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, , `multi_class_bottomup`.
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
            ckpt = torch.load(
                self.pretrained_backbone_weights,
                map_location=self.config.trainer_config.trainer_accelerator,
            )
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
            ckpt = torch.load(
                self.pretrained_head_weights,
                map_location=self.config.trainer_config.trainer_accelerator,
            )
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".head_layers" in k
            }
            self.load_state_dict(ckpt["state_dict"], strict=False)

    @classmethod
    def get_lightning_model_from_config(cls, config: DictConfig):
        """Get lightning model from config."""
        model_type = get_model_type_from_cfg(config)
        backbone_type = get_backbone_type_from_cfg(config)

        lightning_models = {
            "single_instance": SingleInstanceLightningModule,
            "centroid": CentroidLightningModule,
            "centered_instance": TopDownCenteredInstanceLightningModule,
            "bottomup": BottomUpLightningModule,
            "multi_class_bottomup": BottomUpMultiClassLightningModule,
        }

        if model_type not in lightning_models:
            message = f"Incorrect model type. Please check if one of the following keys in the head configs is not None: [`single_instance`, `centroid`, `centered_instance`, `bottomup`, `multi_class_bottomup`]"
            logger.error(message)
            raise ValueError(message)

        lightning_model = lightning_models[model_type](
            config=config,
            model_type=model_type,
            backbone_type=backbone_type,
        )

        return lightning_model

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


class SingleInstanceLightningModule(LightningModel):
    """Lightning Module for SingleInstance Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps and
    forward pass specific to Single Instance model.

    Args:
        config: OmegaConf dictionary which has the following:
            (i) data_config: data loading pre-processing configs to be passed to
            `TopdownConfmapsPipeline` class.
            (ii) model_config: backbone and head configs to be passed to `Model` class.
            (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`.

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
        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            self.single_instance_inf_layer = SingleInstanceInferenceModel(
                torch_model=self.forward,
                peak_threshold=0.2,
                input_scale=1.0,
                return_confmaps=True,
                output_stride=self.config.model_config.head_configs.single_instance.confmaps.output_stride,
            )
        self.ohkm_cfg = OmegaConf.select(
            self.config, "trainer_config.online_hard_keypoint_mining", default=None
        )
        self.node_names = (
            self.config.model_config.head_configs.single_instance.confmaps.part_names
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.single_instance_inf_layer(ex)[0]
        peaks = output["pred_instance_peaks"].cpu().numpy()
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = (
            output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plot_peaks(gt_instances, peaks, paired=True)
        return fig

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

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            loss = loss + ohkm_loss

        # for part-wise loss
        if self.node_names is not None:
            batch_size, _, h, w = y.shape
            mse = (y - y_preds) ** 2
            channel_wise_loss = torch.sum(mse, dim=(0, 2, 3)) / (batch_size * h * w)
            for node_idx, name in enumerate(self.node_names):
                self.log(
                    f"{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
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
        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            val_loss = val_loss + ohkm_loss
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=True,
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


class TopDownCenteredInstanceLightningModule(LightningModel):
    """Lightning Module for TopDownCenteredInstance Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`.

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
        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            self.instance_peaks_inf_layer = FindInstancePeaks(
                torch_model=self.forward,
                peak_threshold=0.2,
                return_confmaps=True,
                output_stride=self.config.model_config.head_configs.centered_instance.confmaps.output_stride,
            )
        self.ohkm_cfg = OmegaConf.select(
            self.config, "trainer_config.online_hard_keypoint_mining", default=None
        )

        self.node_names = (
            self.config.model_config.head_configs.centered_instance.confmaps.part_names
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0)
        output = self.instance_peaks_inf_layer(ex)
        peaks = output["pred_instance_peaks"].cpu().numpy()
        img = (
            output["instance_image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        gt_instances = ex["instance"].cpu().numpy()
        confmaps = (
            output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plot_peaks(gt_instances, peaks, paired=True)
        return fig

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

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            loss = loss + ohkm_loss

        # for part-wise loss
        if self.node_names is not None:
            batch_size, _, h, w = y.shape
            mse = (y - y_preds) ** 2
            channel_wise_loss = torch.sum(mse, dim=(0, 2, 3)) / (batch_size * h * w)
            for node_idx, name in enumerate(self.node_names):
                self.log(
                    f"{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

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
        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            val_loss = val_loss + ohkm_loss
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=True,
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


class CentroidLightningModule(LightningModel):
    """Lightning Module for Centroid Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to centroid model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `CentroidConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`.

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
        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            self.centroid_inf_layer = CentroidCrop(
                torch_model=self.forward,
                peak_threshold=0.2,
                return_confmaps=True,
                output_stride=self.config.model_config.head_configs.centroid.confmaps.output_stride,
                input_scale=1.0,
            )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        gt_centroids = ex["centroids"].cpu().numpy()
        output = self.centroid_inf_layer(ex)
        peaks = output["centroids"][0].cpu().numpy()
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        confmaps = (
            output["pred_centroid_confmaps"][0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plot_peaks(gt_centroids, peaks, paired=False)
        return fig

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
            on_step=True,
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


class BottomUpLightningModule(LightningModel):
    """Lightning Module for BottomUp Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to BottomUp model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `BottomUpPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`.

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

        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            paf_scorer = PAFScorer(
                part_names=self.config.model_config.head_configs.bottomup.confmaps.part_names,
                edges=self.config.model_config.head_configs.bottomup.pafs.edges,
                pafs_stride=self.config.model_config.head_configs.bottomup.pafs.output_stride,
            )
            self.bottomup_inf_layer = BottomUpInferenceModel(
                torch_model=self.forward,
                paf_scorer=paf_scorer,
                peak_threshold=0.2,
                input_scale=1.0,
                return_confmaps=True,
                return_pafs=True,
                cms_output_stride=self.config.model_config.head_configs.bottomup.confmaps.output_stride,
                pafs_output_stride=self.config.model_config.head_configs.bottomup.pafs.output_stride,
            )
        self.ohkm_cfg = OmegaConf.select(
            self.config, "trainer_config.online_hard_keypoint_mining", default=None
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]
        peaks = output["pred_instance_peaks"][0].cpu().numpy()
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = (
            output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plot_peaks(gt_instances, peaks, paired=False)
        return fig

    def visualize_pafs_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        pafs = output["pred_part_affinity_fields"].cpu().numpy()[0]  # (h, w, 2*edges)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)

        pafs = pafs.reshape((pafs.shape[0], pafs.shape[1], -1, 2))
        pafs_mag = np.sqrt(pafs[..., 0] ** 2 + pafs[..., 1] ** 2)
        plot_confmaps(pafs_mag, output_scale=pafs_mag.shape[0] / img.shape[0])
        return fig

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

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        pafs_loss = nn.MSELoss()(pafs, y_paf)

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            pafs_ohkm_loss = compute_ohkm_loss(
                y_gt=y_paf,
                y_pr=pafs,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss
            pafs_loss += pafs_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "PartAffinityFieldsHead": pafs_loss,
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

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        pafs_loss = nn.MSELoss()(pafs, y_paf)

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            pafs_ohkm_loss = compute_ohkm_loss(
                y_gt=y_paf,
                y_pr=pafs,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss
            pafs_loss += pafs_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "PartAffinityFieldsHead": pafs_loss,
        }

        val_loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=True,
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


class BottomUpMultiClassLightningModule(LightningModel):
    """Lightning Module for BottomUp ID Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to BottomUp ID model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `BottomUpMultiClassDataset` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`.

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
        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            self.bottomup_inf_layer = BottomUpMultiClassInferenceModel(
                torch_model=self.forward,
                peak_threshold=0.2,
                input_scale=1.0,
                return_confmaps=True,
                return_class_maps=True,
                cms_output_stride=self.config.model_config.head_configs.multi_class_bottomup.confmaps.output_stride,
                class_maps_output_stride=self.config.model_config.head_configs.multi_class_bottomup.class_maps.output_stride,
            )
        self.ohkm_cfg = OmegaConf.select(
            self.config, "trainer_config.online_hard_keypoint_mining", default=None
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]
        peaks = output["pred_instance_peaks"][0].cpu().numpy()
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = (
            output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plot_peaks(gt_instances, peaks, paired=False)
        return fig

    def visualize_class_maps_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]
        img = (
            output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        )  # convert from (C, H, W) to (H, W, C)
        classmaps = (
            output["pred_class_maps"].cpu().numpy()[0].transpose(1, 2, 0)
        )  # (n_classes, h, w)
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)

        plot_confmaps(classmaps, output_scale=classmaps.shape[0] / img.shape[0])
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        output = self.model(img)
        return {
            "MultiInstanceConfmapsHead": output["MultiInstanceConfmapsHead"],
            "ClassMapsHead": output["ClassMapsHead"],
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        X = torch.squeeze(batch["image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_classmap = torch.squeeze(batch["class_maps"], dim=1)
        preds = self.model(X)
        classmaps = preds["ClassMapsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        classmaps_loss = nn.MSELoss()(classmaps, y_classmap)

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "ClassMapsHead": classmaps_loss,
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
        y_classmap = torch.squeeze(batch["class_maps"], dim=1)

        preds = self.model(X)
        classmaps = preds["ClassMapsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        classmaps_loss = nn.MSELoss()(classmaps, y_classmap)

        if self.ohkm_cfg is not None and self.ohkm_cfg.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.ohkm_cfg.hard_to_easy_ratio,
                min_hard_keypoints=self.ohkm_cfg.min_hard_keypoints,
                max_hard_keypoints=self.ohkm_cfg.max_hard_keypoints,
                loss_scale=self.ohkm_cfg.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "ClassMapsHead": classmaps_loss,
        }

        val_loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=True,
            on_step=True,
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
