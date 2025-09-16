"""This module has the LightningModule classes for all model types."""

from typing import Optional, Union, Dict, Any
import time
from torch import nn
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
import lightning as L
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
from sleap_nn.inference.topdown import (
    CentroidCrop,
    FindInstancePeaks,
    TopDownMultiClassFindInstancePeaks,
)
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
from sleap_nn.config.trainer_config import (
    LRSchedulerConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)
from sleap_nn.config.get_config import get_backbone_config
from sleap_nn.legacy_models import (
    load_legacy_model_weights,
)

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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.model_type = model_type
        self.backbone_type = backbone_type
        if not isinstance(backbone_config, DictConfig):
            backbone_cfg = get_backbone_config(backbone_config)
            config = OmegaConf.structured(backbone_cfg)
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            config = DictConfig(config)
        else:
            config = backbone_config
        self.backbone_config = config
        self.head_configs = head_configs
        self.pretrained_backbone_weights = pretrained_backbone_weights
        self.pretrained_head_weights = pretrained_head_weights
        self.in_channels = self.backbone_config[f"{self.backbone_type}"]["in_channels"]
        self.input_expand_channels = self.in_channels
        self.init_weights = init_weights
        self.lr_scheduler = lr_scheduler
        self.online_mining = online_mining
        self.hard_to_easy_ratio = hard_to_easy_ratio
        self.min_hard_keypoints = min_hard_keypoints
        self.max_hard_keypoints = max_hard_keypoints
        self.loss_scale = loss_scale
        self.optimizer = optimizer
        self.lr = learning_rate
        self.amsgrad = amsgrad

        self.model = Model(
            backbone_type=self.backbone_type,
            backbone_config=self.backbone_config[f"{self.backbone_type}"],
            head_configs=self.head_configs[self.model_type],
            model_type=self.model_type,
        )

        if len(self.head_configs[self.model_type]) > 1:
            self.loss_weights = [
                (
                    self.head_configs[self.model_type][x].loss_weight
                    if self.head_configs[self.model_type][x].loss_weight is not None
                    else 1.0
                )
                for x in self.head_configs[self.model_type]
            ]

        self.training_loss = {}
        self.val_loss = {}
        self.learning_rate = {}

        # Initialization for encoder and decoder stacks.
        if self.init_weights == "xavier":
            self.model.apply(xavier_init_weights)

        # Pre-trained weights for the encoder stack - only for swint and convnext
        if self.backbone_type == "convnext" or self.backbone_type == "swint":
            if (
                self.backbone_config[f"{self.backbone_type}"]["pre_trained_weights"]
                is not None
            ):
                ckpt = MODEL_WEIGHTS[
                    self.backbone_config[f"{self.backbone_type}"]["pre_trained_weights"]
                ].DEFAULT.get_state_dict(progress=True, check_hash=True)
                self.model.backbone.enc.load_state_dict(ckpt, strict=False)

        # Initializing backbone (encoder + decoder) with trained ckpts
        if self.pretrained_backbone_weights is not None:
            logger.info(
                f"Loading backbone weights from `{self.pretrained_backbone_weights}` ..."
            )
            if self.pretrained_backbone_weights.endswith(".ckpt"):
                ckpt = torch.load(
                    self.pretrained_backbone_weights,
                    map_location="cpu",
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".backbone" in k
                }
                self.load_state_dict(ckpt["state_dict"], strict=False)

            elif self.pretrained_backbone_weights.endswith(".h5"):
                # load from sleap model weights
                load_legacy_model_weights(
                    self.model.backbone, self.pretrained_backbone_weights
                )

            else:
                message = f"Unsupported file extension for pretrained backbone weights. Please provide a .ckpt or .h5 file."
                logger.error(message)
                raise ValueError(message)

        # Initializing head layers with trained ckpts.
        if self.pretrained_head_weights is not None:
            logger.info(
                f"Loading head weights from `{self.pretrained_head_weights}` ..."
            )
            if self.pretrained_head_weights.endswith(".ckpt"):
                ckpt = torch.load(
                    self.pretrained_head_weights,
                    map_location="cpu",
                    weights_only=False,
                )
                ckpt["state_dict"] = {
                    k: ckpt["state_dict"][k]
                    for k in ckpt["state_dict"].keys()
                    if ".head_layers" in k
                }
                self.load_state_dict(ckpt["state_dict"], strict=False)

            elif self.pretrained_head_weights.endswith(".h5"):
                # load from sleap model weights
                load_legacy_model_weights(
                    self.model.head_layers, self.pretrained_head_weights
                )

            else:
                message = f"Unsupported file extension for pretrained head weights. Please provide a .ckpt or .h5 file."
                logger.error(message)
                raise ValueError(message)

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
            "multi_class_topdown": TopDownCenteredInstanceMultiClassLightningModule,
        }

        if model_type not in lightning_models:
            message = f"Incorrect model type. Please check if one of the following keys in the head configs is not None: [`single_instance`, `centroid`, `centered_instance`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`]"
            logger.error(message)
            raise ValueError(message)

        lightning_model = lightning_models[model_type](
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=config.model_config.backbone_config,
            head_configs=config.model_config.head_configs,
            pretrained_backbone_weights=config.model_config.pretrained_backbone_weights,
            pretrained_head_weights=config.model_config.pretrained_head_weights,
            init_weights=config.model_config.init_weights,
            lr_scheduler=config.trainer_config.lr_scheduler,
            online_mining=config.trainer_config.online_hard_keypoint_mining.online_mining,
            hard_to_easy_ratio=config.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
            min_hard_keypoints=config.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
            max_hard_keypoints=config.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
            loss_scale=config.trainer_config.online_hard_keypoint_mining.loss_scale,
            optimizer=config.trainer_config.optimizer_name,
            learning_rate=config.trainer_config.optimizer.lr,
            amsgrad=config.trainer_config.optimizer.amsgrad,
        )

        return lightning_model

    def forward(self, img):
        """Forward pass of the model."""
        pass

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
            sync_dist=True,
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
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        pass

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pass

    def configure_optimizers(self):
        """Configure optimiser and learning rate scheduler."""
        if self.optimizer == "Adam":
            optim = torch.optim.Adam
        elif self.optimizer == "AdamW":
            optim = torch.optim.AdamW

        optimizer = optim(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad,
        )

        lr_scheduler_cfg = LRSchedulerConfig()
        if self.lr_scheduler is None:
            return {
                "optimizer": optimizer,
            }

        scheduler = None
        if isinstance(self.lr_scheduler, str):
            if self.lr_scheduler == "step_lr":
                lr_scheduler_cfg.step_lr = StepLRConfig()
            elif self.lr_scheduler == "reduce_lr_on_plateau":
                lr_scheduler_cfg.reduce_lr_on_plateau = ReduceLROnPlateauConfig()

        elif isinstance(self.lr_scheduler, dict):
            lr_scheduler_cfg = self.lr_scheduler

        for k, v in self.lr_scheduler.items():
            if v is not None:
                if k == "step_lr":
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=self.lr_scheduler.step_lr.step_size,
                        gamma=self.lr_scheduler.step_lr.gamma,
                    )
                    break
                elif k == "reduce_lr_on_plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        threshold=self.lr_scheduler.reduce_lr_on_plateau.threshold,
                        threshold_mode=self.lr_scheduler.reduce_lr_on_plateau.threshold_mode,
                        cooldown=self.lr_scheduler.reduce_lr_on_plateau.cooldown,
                        patience=self.lr_scheduler.reduce_lr_on_plateau.patience,
                        factor=self.lr_scheduler.reduce_lr_on_plateau.factor,
                        min_lr=self.lr_scheduler.reduce_lr_on_plateau.min_lr,
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
    forward pass specific to Single Instance model. Single Instance models predict keypoint locations
    directly from the input image without requiring a separate detection step.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )

        self.single_instance_inf_layer = SingleInstanceInferenceModel(
            torch_model=self.forward,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
            output_stride=self.head_configs.single_instance.confmaps.output_stride,
        )
        self.node_names = self.head_configs.single_instance.confmaps.part_names

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

        if self.online_mining is not None and self.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
                    sync_dist=True,
                )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
        val_loss = nn.MSELoss()(y_preds, y)
        if self.online_mining is not None and self.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )


class TopDownCenteredInstanceLightningModule(LightningModel):
    """Lightning Module for TopDownCenteredInstance Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance model. Top-Down models use a two-stage
    approach: first detecting centroids, then predicting keypoints for each detected centroid.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )

        self.instance_peaks_inf_layer = FindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centered_instance.confmaps.output_stride,
        )

        self.node_names = self.head_configs.centered_instance.confmaps.part_names

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

        if self.online_mining is not None and self.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
                    sync_dist=True,
                )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1), torch.squeeze(
            batch["confidence_maps"], dim=1
        )

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        val_loss = nn.MSELoss()(y_preds, y)
        if self.online_mining is not None and self.online_mining:
            ohkm_loss = compute_ohkm_loss(
                y_gt=y,
                y_pr=y_preds,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )


class CentroidLightningModule(LightningModel):
    """Lightning Module for Centroid Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to centroid model. Centroid models detect the center points
    of animals in the image, which are then used by Top-Down models for keypoint prediction.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )

        self.centroid_inf_layer = CentroidCrop(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centroid.confmaps.output_stride,
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
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )


class BottomUpLightningModule(LightningModel):
    """Lightning Module for BottomUp Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to BottomUp model. Bottom-Up models predict all keypoints
    simultaneously and use Part Affinity Fields (PAFs) to group keypoints into individual animals.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )

        paf_scorer = PAFScorer(
            part_names=self.head_configs.bottomup.confmaps.part_names,
            edges=self.head_configs.bottomup.pafs.edges,
            pafs_stride=self.head_configs.bottomup.pafs.output_stride,
        )
        self.bottomup_inf_layer = BottomUpInferenceModel(
            torch_model=self.forward,
            paf_scorer=paf_scorer,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
            return_pafs=True,
            cms_output_stride=self.head_configs.bottomup.confmaps.output_stride,
            pafs_output_stride=self.head_configs.bottomup.pafs.output_stride,
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

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            pafs_ohkm_loss = compute_ohkm_loss(
                y_gt=y_paf,
                y_pr=pafs,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss
            pafs_loss += pafs_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "PartAffinityFieldsHead": pafs_loss,
        }
        loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
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

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            pafs_ohkm_loss = compute_ohkm_loss(
                y_gt=y_paf,
                y_pr=pafs,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )


class BottomUpMultiClassLightningModule(LightningModel):
    """Lightning Module for BottomUp ID Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to BottomUp ID model. Multi-Class Bottom-Up models predict
    all keypoints simultaneously and classify instances using class maps to identify
    individual animals across frames.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )
        self.bottomup_inf_layer = BottomUpMultiClassInferenceModel(
            torch_model=self.forward,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
            return_class_maps=True,
            cms_output_stride=self.head_configs.multi_class_bottomup.confmaps.output_stride,
            class_maps_output_stride=self.head_configs.multi_class_bottomup.class_maps.output_stride,
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

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss

        losses = {
            "MultiInstanceConfmapsHead": confmap_loss,
            "ClassMapsHead": classmaps_loss,
        }
        loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
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

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )


class TopDownCenteredInstanceMultiClassLightningModule(LightningModel):
    """Lightning Module for TopDownCenteredInstance ID Model.

    This is a subclass of the `LightningModel` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance model. Multi-Class Top-Down models
    use a two-stage approach: first detecting centroids, then predicting keypoints and
    classifying instances using supervised learning with ground truth track IDs.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration. Can be:
            - String: One of the preset backbone types:
                - UNet variants: ["unet", "unet_medium_rf", "unet_large_rf"]
                - ConvNeXt variants: ["convnext", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
                - SwinT variants: ["swint", "swint_tiny", "swint_small", "swint_base"]
            - Dictionary: Custom configuration with structure:
                {
                    "unet": {UNetConfig parameters},
                    "convnext": {ConvNextConfig parameters},
                    "swint": {SwinTConfig parameters}
                }
                Only one backbone type should be specified in the dictionary.
            - DictConfig: OmegaConf DictConfig object containing backbone configuration.
        head_configs: Head configuration dictionary containing model-specific parameters.
            For Single Instance: confmaps with part_names, sigma, output_stride.
            For Centroid: confmaps with anchor_part, sigma, output_stride.
            For Centered Instance: confmaps with part_names, anchor_part, sigma, output_stride.
            For Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; pafs with edges, sigma, output_stride, loss_weight.
            For Multi-Class Bottom-Up: confmaps with part_names, sigma, output_stride, loss_weight; class_maps with classes, sigma, output_stride, loss_weight.
            For Multi-Class Top-Down: confmaps with part_names, anchor_part, sigma, output_stride, loss_weight; class_vectors with classes, num_fc_layers, num_fc_units, global_pool, output_stride, loss_weight.
        pretrained_backbone_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for backbone initialization. If None, random initialization is used.
        pretrained_head_weights: Path to checkpoint `.ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file for head layers initialization. If None, random initialization is used.
        init_weights: Model weights initialization method. "default" uses kaiming uniform initialization, "xavier" uses Xavier initialization.
        lr_scheduler: Learning rate scheduler configuration. Can be string ("step_lr", "reduce_lr_on_plateau") or dictionary with scheduler-specific parameters.
        online_mining: If True, online hard keypoint mining (OHKM) is enabled. Loss is computed per keypoint and sorted from lowest (easy) to highest (hard).
        hard_to_easy_ratio: Minimum ratio of individual keypoint loss to lowest keypoint loss to be considered "hard". Default: 2.0.
        min_hard_keypoints: Minimum number of keypoints considered as "hard", even if below hard_to_easy_ratio. Default: 2.
        max_hard_keypoints: Maximum number of hard keypoints to apply scaling to. If None, no limit is applied.
        loss_scale: Factor to scale hard keypoint losses by. Default: 5.0.
        optimizer: Optimizer name. One of ["Adam", "AdamW"].
        learning_rate: Learning rate for the optimizer. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
    """

    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_config: Union[str, Dict[str, Any], DictConfig],
        head_configs: DictConfig,
        pretrained_backbone_weights: Optional[str] = None,
        pretrained_head_weights: Optional[str] = None,
        init_weights: Optional[str] = "xavier",
        lr_scheduler: Optional[Union[str, DictConfig]] = None,
        online_mining: Optional[bool] = False,
        hard_to_easy_ratio: Optional[float] = 2.0,
        min_hard_keypoints: Optional[int] = 2,
        max_hard_keypoints: Optional[int] = None,
        loss_scale: Optional[float] = 5.0,
        optimizer: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
        amsgrad: Optional[bool] = False,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            model_type=model_type,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            head_configs=head_configs,
            pretrained_backbone_weights=pretrained_backbone_weights,
            pretrained_head_weights=pretrained_head_weights,
            init_weights=init_weights,
            lr_scheduler=lr_scheduler,
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
            optimizer=optimizer,
            learning_rate=learning_rate,
            amsgrad=amsgrad,
        )
        self.instance_peaks_inf_layer = TopDownMultiClassFindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.multi_class_topdown.confmaps.output_stride,
        )

        self.node_names = self.head_configs.multi_class_topdown.confmaps.part_names

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
        output = self.model(img)
        return {
            "CenteredInstanceConfmapsHead": output["CenteredInstanceConfmapsHead"],
            "ClassVectorsHead": output["ClassVectorsHead"],
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        X = torch.squeeze(batch["instance_image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_classvector = batch["class_vectors"]
        preds = self.model(X)
        classvector = preds["ClassVectorsHead"]
        confmaps = preds["CenteredInstanceConfmapsHead"]

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        classvector_loss = nn.CrossEntropyLoss()(classvector, y_classvector)

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss

        losses = {
            "CenteredInstanceConfmapsHead": confmap_loss,
            "ClassVectorsHead": classvector_loss,
        }
        loss = sum([s * losses[t] for s, t in zip(self.loss_weights, losses)])

        # for part-wise loss
        if self.node_names is not None:
            batch_size, _, h, w = y_confmap.shape
            mse = (y_confmap - confmaps) ** 2
            channel_wise_loss = torch.sum(mse, dim=(0, 2, 3)) / (batch_size * h * w)
            for node_idx, name in enumerate(self.node_names):
                self.log(
                    f"{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X = torch.squeeze(batch["instance_image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_classvector = batch["class_vectors"]
        preds = self.model(X)
        classvector = preds["ClassVectorsHead"]
        confmaps = preds["CenteredInstanceConfmapsHead"]

        confmap_loss = nn.MSELoss()(confmaps, y_confmap)
        classvector_loss = nn.CrossEntropyLoss()(classvector, y_classvector)

        if self.online_mining is not None and self.online_mining:
            confmap_ohkm_loss = compute_ohkm_loss(
                y_gt=y_confmap,
                y_pr=confmaps,
                hard_to_easy_ratio=self.hard_to_easy_ratio,
                min_hard_keypoints=self.min_hard_keypoints,
                max_hard_keypoints=self.max_hard_keypoints,
                loss_scale=self.loss_scale,
            )
            confmap_loss += confmap_ohkm_loss

        losses = {
            "CenteredInstanceConfmapsHead": confmap_loss,
            "ClassVectorsHead": classvector_loss,
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
            sync_dist=True,
        )
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
