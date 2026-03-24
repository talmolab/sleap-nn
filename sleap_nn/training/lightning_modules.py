"""This module has the LightningModule classes for all model types."""

from typing import Optional, Union, Dict, Any, List
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
from sleap_nn.architectures.model import Model, MultiHeadModel
from sleap_nn.data.normalization import normalize_on_gpu
from sleap_nn.training.losses import compute_ohkm_loss
from loguru import logger
from sleap_nn.training.utils import (
    xavier_init_weights,
    plot_confmaps,
    plot_img,
    plot_peaks,
    VisualizationData,
)
import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend to avoid tkinter issues on Windows CI
import matplotlib.pyplot as plt
from sleap_nn.config.utils import get_backbone_type_from_cfg, get_model_type_from_cfg
from sleap_nn.config.trainer_config import (
    CosineAnnealingWarmupConfig,
    LinearWarmupLinearDecayConfig,
    LRSchedulerConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)
from sleap_nn.training.schedulers import (
    LinearWarmupCosineAnnealingLR,
    LinearWarmupLinearDecayLR,
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
        negative_loss_weight: Optional[float] = 1.0,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.negative_loss_weight = negative_loss_weight
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

        # For epoch-averaged loss tracking
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

        # For epoch-end evaluation
        self.val_predictions: List[Dict] = []
        self.val_ground_truth: List[Dict] = []
        self._collect_val_predictions: bool = False

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
                    self.model.backbone,
                    self.pretrained_backbone_weights,
                    component="backbone",
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
                    self.model.head_layers,
                    self.pretrained_head_weights,
                    component="head",
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

        negative_loss_weight = getattr(config.data_config, "negative_loss_weight", 1.0)

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
            negative_loss_weight=negative_loss_weight,
        )

        return lightning_model

    def forward(self, img):
        """Forward pass of the model."""
        pass

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        self.train_start_time = time.time()
        # Reset epoch loss tracking
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def _accumulate_loss(self, loss: torch.Tensor):
        """Accumulate loss for epoch-averaged logging. Call this in training_step."""
        self._epoch_loss_sum += loss.detach().item()
        self._epoch_loss_count += 1

    def on_train_epoch_end(self):
        """Configure the train timer at the end of every epoch."""
        train_time = time.time() - self.train_start_time
        self.log(
            "train/time",
            train_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch explicitly for custom x-axis support in wandb
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch-averaged training loss
        if self._epoch_loss_count > 0:
            avg_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.log(
                "train/loss",
                avg_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # Log current learning rate (useful for monitoring LR schedulers)
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                "train/lr",
                lr,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_validation_epoch_start(self):
        """Configure the val timer at the beginning of each epoch."""
        self.val_start_time = time.time()
        # Clear accumulated predictions for new epoch
        self.val_predictions = []
        self.val_ground_truth = []

    def on_validation_epoch_end(self):
        """Configure the val timer at the end of every epoch."""
        val_time = time.time() - self.val_start_time
        self.log(
            "val/time",
            val_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch explicitly so val/* metrics can use it as x-axis in wandb
        # (mirrors what on_train_epoch_end does for train/* metrics)
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _compute_negative_weighted_loss(
        self, y_preds: torch.Tensor, y: torch.Tensor, batch: Dict
    ) -> torch.Tensor:
        """Compute MSE loss with optional negative sample weighting.

        When ``is_negative`` is absent from the batch or ``negative_loss_weight``
        is 1.0, this returns plain ``nn.MSELoss()``.  Otherwise, per-sample MSE
        is computed and negative samples are weighted by ``negative_loss_weight``.

        Also logs split metrics (positive/negative loss and counts) when
        negatives are present.

        Args:
            y_preds: Predicted tensor.
            y: Ground truth tensor.
            batch: Batch dictionary, may contain ``is_negative`` key.

        Returns:
            The (optionally weighted) loss tensor.
        """
        is_negative = batch.get("is_negative", None)
        if is_negative is None:
            return nn.MSELoss()(y_preds, y)

        # Per-sample loss for split tracking and weighting
        per_sample = (
            (y_preds - y).pow(2).mean(dim=list(range(1, y_preds.ndim)))
        )  # (batch,)

        is_neg = is_negative.to(y_preds.device)
        is_pos = ~is_neg
        n_neg = int(is_neg.sum().item())
        n_pos = int(is_pos.sum().item())

        # Log split losses and counts
        if n_pos > 0:
            loss_pos = per_sample[is_pos].mean()
            self.log(
                "train/loss_positive",
                loss_pos,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if n_neg > 0:
            loss_neg = per_sample[is_neg].mean()
            self.log(
                "train/loss_negative",
                loss_neg,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "train/n_positive",
            float(n_pos),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )
        self.log(
            "train/n_negative",
            float(n_neg),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )

        # Apply weighting
        if self.negative_loss_weight == 1.0:
            return per_sample.mean()

        weights = torch.where(
            is_neg,
            torch.tensor(self.negative_loss_weight, device=y_preds.device),
            torch.tensor(1.0, device=y_preds.device),
        )
        return (per_sample * weights).mean()

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
            elif self.lr_scheduler == "cosine_annealing_warmup":
                lr_scheduler_cfg.cosine_annealing_warmup = CosineAnnealingWarmupConfig()
            elif self.lr_scheduler == "linear_warmup_linear_decay":
                lr_scheduler_cfg.linear_warmup_linear_decay = (
                    LinearWarmupLinearDecayConfig()
                )

        elif isinstance(self.lr_scheduler, dict):
            lr_scheduler_cfg = self.lr_scheduler

        for k, v in self.lr_scheduler.items():
            if v is not None:
                if k == "cosine_annealing_warmup":
                    cfg = self.lr_scheduler.cosine_annealing_warmup
                    # Use trainer's max_epochs if not specified in config
                    max_epochs = (
                        cfg.max_epochs
                        if cfg.max_epochs is not None
                        else self.trainer.max_epochs
                    )
                    scheduler = LinearWarmupCosineAnnealingLR(
                        optimizer=optimizer,
                        warmup_epochs=cfg.warmup_epochs,
                        max_epochs=max_epochs,
                        warmup_start_lr=cfg.warmup_start_lr,
                        eta_min=cfg.eta_min,
                    )
                    break
                elif k == "linear_warmup_linear_decay":
                    cfg = self.lr_scheduler.linear_warmup_linear_decay
                    # Use trainer's max_epochs if not specified in config
                    max_epochs = (
                        cfg.max_epochs
                        if cfg.max_epochs is not None
                        else self.trainer.max_epochs
                    )
                    scheduler = LinearWarmupLinearDecayLR(
                        optimizer=optimizer,
                        warmup_epochs=cfg.warmup_epochs,
                        max_epochs=max_epochs,
                        warmup_start_lr=cfg.warmup_start_lr,
                        end_lr=cfg.end_lr,
                    )
                    break
                elif k == "step_lr":
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
                "monitor": "val/loss",
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
        )

        self.single_instance_inf_layer = SingleInstanceInferenceModel(
            torch_model=self.forward,
            peak_threshold=0.2,
            input_scale=1.0,
            return_confmaps=True,
            output_stride=self.head_configs.single_instance.confmaps.output_stride,
        )
        self.node_names = self.head_configs.single_instance.confmaps.part_names

    def get_visualization_data(self, sample) -> VisualizationData:
        """Extract visualization data from a sample.

        Args:
            sample: A sample dictionary from the data pipeline.

        Returns:
            VisualizationData containing image, confmaps, peaks, etc.
        """
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.single_instance_inf_layer(ex)[0]

        peaks = output["pred_instance_peaks"].cpu().numpy()
        peak_values = output["pred_peak_values"].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=list(self.node_names) if self.node_names else [],
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=True,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        # Only squeeze n_samples dim if 5D (batch, n_samples, C, H, W) -> (batch, C, H, W)
        # Avoid double-squeezing when called from validation_step which already squeezes
        if img.ndim == 5:
            img = img.squeeze(1)
        img = img.to(self.device)
        img = normalize_on_gpu(img)
        return self.model(img)["SingleInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = (
            torch.squeeze(batch["image"], dim=1),
            torch.squeeze(batch["confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]

        loss = self._compute_negative_weighted_loss(y_preds, y, batch)

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
                    f"train/confmaps/{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = (
            torch.squeeze(batch["image"], dim=1),
            torch.squeeze(batch["confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

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
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            with torch.no_grad():
                # Squeeze n_samples dim from image for inference (batch, 1, C, H, W) -> (batch, C, H, W)
                inference_batch = {k: v for k, v in batch.items()}
                if inference_batch["image"].ndim == 5:
                    inference_batch["image"] = inference_batch["image"].squeeze(1)
                inference_output = self.single_instance_inf_layer(inference_batch)
                if isinstance(inference_output, list):
                    inference_output = inference_output[0]

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions are already in original image space (inference divides by eff_scale)
                pred_peaks = inference_output["pred_instance_peaks"][i].cpu().numpy()
                pred_scores = inference_output["pred_peak_values"][i].cpu().numpy()

                # Transform GT from preprocessed to original image space
                # Note: instances have shape (1, max_inst, n_nodes, 2) - squeeze n_samples dim
                gt_prep = batch["instances"][i].cpu().numpy()
                if gt_prep.ndim == 4:
                    gt_prep = gt_prep.squeeze(0)  # (max_inst, n_nodes, 2)
                gt_orig = gt_prep / eff
                num_inst = batch["num_instances"][i].item()
                gt_orig = gt_orig[:num_inst]  # Only valid instances

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_peaks,
                        "pred_scores": pred_scores,
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_orig,
                        "num_instances": num_inst,
                    }
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
        )

        self.instance_peaks_inf_layer = FindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centered_instance.confmaps.output_stride,
        )

        self.node_names = self.head_configs.centered_instance.confmaps.part_names

    def get_visualization_data(self, sample) -> VisualizationData:
        """Extract visualization data from a sample."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0)
        output = self.instance_peaks_inf_layer(ex)

        peaks = output["pred_instance_peaks"].cpu().numpy()
        peak_values = output["pred_peak_values"].cpu().numpy()
        img = output["instance_image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instance"].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=list(self.node_names) if self.node_names else [],
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=True,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        return self.model(img)["CenteredInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = (
            torch.squeeze(batch["instance_image"], dim=1),
            torch.squeeze(batch["confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

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
                    f"train/confmaps/{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X, y = (
            torch.squeeze(batch["instance_image"], dim=1),
            torch.squeeze(batch["confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

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
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            # SAVE bbox BEFORE inference (it modifies in-place!)
            bbox_prep_saved = batch["instance_bbox"].clone()

            with torch.no_grad():
                inference_output = self.instance_peaks_inf_layer(batch)

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions from inference (crop-relative, original scale)
                pred_peaks_crop = (
                    inference_output["pred_instance_peaks"][i].cpu().numpy()
                )
                pred_scores = inference_output["pred_peak_values"][i].cpu().numpy()

                # Compute bbox offset in original space from SAVED prep bbox
                # bbox has shape (n_samples=1, 4, 2) where 4 corners
                bbox_prep = bbox_prep_saved[i].squeeze(0).cpu().numpy()  # (4, 2)
                bbox_top_left_orig = (
                    bbox_prep[0] / eff
                )  # Top-left corner in original space

                # Full image coordinates (original space)
                pred_peaks_full = pred_peaks_crop + bbox_top_left_orig

                # GT transform: crop-relative preprocessed -> full image original
                gt_crop_prep = (
                    batch["instance"][i].squeeze(0).cpu().numpy()
                )  # (n_nodes, 2)
                gt_crop_orig = gt_crop_prep / eff
                gt_full_orig = gt_crop_orig + bbox_top_left_orig

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_peaks_full.reshape(
                            1, -1, 2
                        ),  # (1, n_nodes, 2)
                        "pred_scores": pred_scores.reshape(1, -1),  # (1, n_nodes)
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_full_orig.reshape(
                            1, -1, 2
                        ),  # (1, n_nodes, 2)
                        "num_instances": 1,
                    }
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
        )

        self.centroid_inf_layer = CentroidCrop(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centroid.confmaps.output_stride,
            input_scale=1.0,
        )
        self.node_names = ["centroid"]

    def get_visualization_data(self, sample) -> VisualizationData:
        """Extract visualization data from a sample."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        gt_centroids = ex["centroids"].cpu().numpy()
        output = self.centroid_inf_layer(ex)

        peaks = output["centroids"][0].cpu().numpy()
        centroid_vals = output["centroid_vals"][0].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        confmaps = output["pred_centroid_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=centroid_vals,
            gt_instances=gt_centroids,
            node_names=self.node_names,
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=False,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        return self.model(img)["CentroidConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = (
            torch.squeeze(batch["image"], dim=1),
            torch.squeeze(batch["centroids_confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

        y_preds = self.model(X)["CentroidConfmapsHead"]
        loss = self._compute_negative_weighted_loss(y_preds, y, batch)
        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = (
            torch.squeeze(batch["image"], dim=1),
            torch.squeeze(batch["centroids_confidence_maps"], dim=1),
        )
        X = normalize_on_gpu(X)

        y_preds = self.model(X)["CentroidConfmapsHead"]
        val_loss = nn.MSELoss()(y_preds, y)
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            # Save GT centroids before inference (inference overwrites batch["centroids"])
            batch["gt_centroids"] = batch["centroids"].clone()

            with torch.no_grad():
                inference_output = self.centroid_inf_layer(batch)

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions are in original image space (inference divides by eff_scale)
                # centroids shape: (batch, 1, max_instances, 2) - squeeze to (max_instances, 2)
                pred_centroids = (
                    inference_output["centroids"][i].squeeze(0).cpu().numpy()
                )
                pred_vals = inference_output["centroid_vals"][i].cpu().numpy()

                # Transform GT centroids from preprocessed to original image space
                # Use "gt_centroids" since inference overwrites "centroids" with predictions
                gt_centroids_prep = (
                    batch["gt_centroids"][i].cpu().numpy()
                )  # (n_samples=1, max_inst, 2)
                gt_centroids_orig = gt_centroids_prep.squeeze(0) / eff  # (max_inst, 2)
                num_inst = batch["num_instances"][i].item()

                # Filter to valid instances (non-NaN)
                valid_pred_mask = ~np.isnan(pred_centroids).any(axis=1)
                pred_centroids = pred_centroids[valid_pred_mask]
                pred_vals = pred_vals[valid_pred_mask]

                gt_centroids_valid = gt_centroids_orig[:num_inst]

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_centroids.reshape(
                            -1, 1, 2
                        ),  # (n_inst, 1, 2)
                        "pred_scores": pred_vals.reshape(-1, 1),  # (n_inst, 1)
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_centroids_valid.reshape(
                            -1, 1, 2
                        ),  # (n_inst, 1, 2)
                        "num_instances": num_inst,
                    }
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
        )

        paf_scorer = PAFScorer(
            part_names=self.head_configs.bottomup.confmaps.part_names,
            edges=self.head_configs.bottomup.pafs.edges,
            pafs_stride=self.head_configs.bottomup.pafs.output_stride,
        )
        self.bottomup_inf_layer = BottomUpInferenceModel(
            torch_model=self.forward,
            paf_scorer=paf_scorer,
            peak_threshold=0.1,  # Lower threshold for epoch-end eval during training
            input_scale=1.0,
            return_confmaps=True,
            return_pafs=True,
            cms_output_stride=self.head_configs.bottomup.confmaps.output_stride,
            pafs_output_stride=self.head_configs.bottomup.pafs.output_stride,
            max_peaks_per_node=100,  # Prevents combinatorial explosion in early training
        )
        self.node_names = list(self.head_configs.bottomup.confmaps.part_names)

    def get_visualization_data(
        self, sample, include_pafs: bool = False
    ) -> VisualizationData:
        """Extract visualization data from a sample."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]

        peaks = output["pred_instance_peaks"][0].cpu().numpy()
        peak_values = output["pred_peak_values"][0].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        pred_pafs = None
        if include_pafs:
            pafs = output["pred_part_affinity_fields"].cpu().numpy()[0]
            pred_pafs = pafs  # (h, w, 2*edges)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=self.node_names,
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=False,
            pred_pafs=pred_pafs,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def visualize_pafs_example(self, sample):
        """Visualize PAF predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample, include_pafs=True)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)

        pafs = data.pred_pafs
        pafs = pafs.reshape((pafs.shape[0], pafs.shape[1], -1, 2))
        pafs_mag = np.sqrt(pafs[..., 0] ** 2 + pafs[..., 1] ** 2)
        plot_confmaps(pafs_mag, output_scale=pafs_mag.shape[0] / data.image.shape[0])
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
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
        X = normalize_on_gpu(X)
        preds = self.model(X)
        pafs = preds["PartAffinityFieldsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]

        confmap_loss = self._compute_negative_weighted_loss(confmaps, y_confmap, batch)
        pafs_loss = self._compute_negative_weighted_loss(pafs, y_paf, batch)

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
        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        self.log(
            "train/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/paf_loss",
            pafs_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X = torch.squeeze(batch["image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_paf = batch["part_affinity_fields"]
        X = normalize_on_gpu(X)

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
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/paf_loss",
            pafs_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            with torch.no_grad():
                # Note: Do NOT squeeze the image here - the forward() method expects
                # (batch, n_samples, C, H, W) and handles the n_samples squeeze internally
                inference_output = self.bottomup_inf_layer(batch)
                if isinstance(inference_output, list):
                    inference_output = inference_output[0]

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions are already in original space (variable number of instances)
                pred_peaks = inference_output["pred_instance_peaks"][i]
                pred_scores = inference_output["pred_peak_values"][i]
                if torch.is_tensor(pred_peaks):
                    pred_peaks = pred_peaks.cpu().numpy()
                if torch.is_tensor(pred_scores):
                    pred_scores = pred_scores.cpu().numpy()

                # Transform GT to original space
                # Note: instances have shape (1, max_inst, n_nodes, 2) - squeeze n_samples dim
                gt_prep = batch["instances"][i].cpu().numpy()
                if gt_prep.ndim == 4:
                    gt_prep = gt_prep.squeeze(0)  # (max_inst, n_nodes, 2)
                gt_orig = gt_prep / eff
                num_inst = batch["num_instances"][i].item()
                gt_orig = gt_orig[:num_inst]  # Only valid instances

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_peaks,  # Original space, variable instances
                        "pred_scores": pred_scores,
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_orig,  # Original space
                        "num_instances": num_inst,
                    }
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
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
        self.node_names = list(
            self.head_configs.multi_class_bottomup.confmaps.part_names
        )

    def get_visualization_data(
        self, sample, include_class_maps: bool = False
    ) -> VisualizationData:
        """Extract visualization data from a sample."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        output = self.bottomup_inf_layer(ex)[0]

        peaks = output["pred_instance_peaks"][0].cpu().numpy()
        peak_values = output["pred_peak_values"][0].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instances"][0].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        pred_class_maps = None
        if include_class_maps:
            pred_class_maps = (
                output["pred_class_maps"].cpu().numpy()[0].transpose(1, 2, 0)
            )

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=self.node_names,
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=False,
            pred_class_maps=pred_class_maps,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def visualize_class_maps_example(self, sample):
        """Visualize class map predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample, include_class_maps=True)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(
            data.pred_class_maps,
            output_scale=data.pred_class_maps.shape[0] / data.image.shape[0],
        )
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
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
        X = normalize_on_gpu(X)
        preds = self.model(X)
        classmaps = preds["ClassMapsHead"]
        confmaps = preds["MultiInstanceConfmapsHead"]

        confmap_loss = self._compute_negative_weighted_loss(confmaps, y_confmap, batch)
        classmaps_loss = self._compute_negative_weighted_loss(
            classmaps, y_classmap, batch
        )

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
        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        self.log(
            "train/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/classmap_loss",
            classmaps_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute classification accuracy at GT keypoint locations
        with torch.no_grad():
            # Get output stride for class maps
            cms_stride = self.head_configs.multi_class_bottomup.class_maps.output_stride

            # Get GT instances and sample class maps at those locations
            instances = batch["instances"]  # (batch, n_samples, max_inst, n_nodes, 2)
            if instances.dim() == 5:
                instances = instances.squeeze(1)  # (batch, max_inst, n_nodes, 2)
            num_instances = batch["num_instances"]  # (batch,)

            correct = 0
            total = 0
            for b in range(instances.shape[0]):
                n_inst = num_instances[b].item()
                for inst_idx in range(n_inst):
                    for node_idx in range(instances.shape[2]):
                        # Get keypoint location (in input image space)
                        kp = instances[b, inst_idx, node_idx]  # (2,) = (x, y)
                        if torch.isnan(kp).any():
                            continue

                        # Convert to class map space
                        x_cm = (
                            (kp[0] / cms_stride)
                            .long()
                            .clamp(0, classmaps.shape[-1] - 1)
                        )
                        y_cm = (
                            (kp[1] / cms_stride)
                            .long()
                            .clamp(0, classmaps.shape[-2] - 1)
                        )

                        # Sample predicted and GT class at this location
                        pred_class = classmaps[b, :, y_cm, x_cm].argmax()
                        gt_class = y_classmap[b, :, y_cm, x_cm].argmax()

                        if pred_class == gt_class:
                            correct += 1
                        total += 1

            if total > 0:
                class_accuracy = torch.tensor(correct / total, device=X.device)
                self.log(
                    "train/class_accuracy",
                    class_accuracy,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X = torch.squeeze(batch["image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_classmap = torch.squeeze(batch["class_maps"], dim=1)
        X = normalize_on_gpu(X)

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
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/classmap_loss",
            classmaps_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute classification accuracy at GT keypoint locations
        with torch.no_grad():
            # Get output stride for class maps
            cms_stride = self.head_configs.multi_class_bottomup.class_maps.output_stride

            # Get GT instances and sample class maps at those locations
            instances = batch["instances"]  # (batch, n_samples, max_inst, n_nodes, 2)
            if instances.dim() == 5:
                instances = instances.squeeze(1)  # (batch, max_inst, n_nodes, 2)
            num_instances = batch["num_instances"]  # (batch,)

            correct = 0
            total = 0
            for b in range(instances.shape[0]):
                n_inst = num_instances[b].item()
                for inst_idx in range(n_inst):
                    for node_idx in range(instances.shape[2]):
                        # Get keypoint location (in input image space)
                        kp = instances[b, inst_idx, node_idx]  # (2,) = (x, y)
                        if torch.isnan(kp).any():
                            continue

                        # Convert to class map space
                        x_cm = (
                            (kp[0] / cms_stride)
                            .long()
                            .clamp(0, classmaps.shape[-1] - 1)
                        )
                        y_cm = (
                            (kp[1] / cms_stride)
                            .long()
                            .clamp(0, classmaps.shape[-2] - 1)
                        )

                        # Sample predicted and GT class at this location
                        pred_class = classmaps[b, :, y_cm, x_cm].argmax()
                        gt_class = y_classmap[b, :, y_cm, x_cm].argmax()

                        if pred_class == gt_class:
                            correct += 1
                        total += 1

            if total > 0:
                class_accuracy = torch.tensor(correct / total, device=X.device)
                self.log(
                    "val/class_accuracy",
                    class_accuracy,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            with torch.no_grad():
                # Note: Do NOT squeeze the image here - the forward() method expects
                # (batch, n_samples, C, H, W) and handles the n_samples squeeze internally
                inference_output = self.bottomup_inf_layer(batch)
                if isinstance(inference_output, list):
                    inference_output = inference_output[0]

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions are already in original space (variable number of instances)
                pred_peaks = inference_output["pred_instance_peaks"][i]
                pred_scores = inference_output["pred_peak_values"][i]
                if torch.is_tensor(pred_peaks):
                    pred_peaks = pred_peaks.cpu().numpy()
                if torch.is_tensor(pred_scores):
                    pred_scores = pred_scores.cpu().numpy()

                # Transform GT to original space
                # Note: instances have shape (1, max_inst, n_nodes, 2) - squeeze n_samples dim
                gt_prep = batch["instances"][i].cpu().numpy()
                if gt_prep.ndim == 4:
                    gt_prep = gt_prep.squeeze(0)  # (max_inst, n_nodes, 2)
                gt_orig = gt_prep / eff
                num_inst = batch["num_instances"][i].item()
                gt_orig = gt_orig[:num_inst]  # Only valid instances

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_peaks,  # Original space, variable instances
                        "pred_scores": pred_scores,
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_orig,  # Original space
                        "num_instances": num_inst,
                    }
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
        negative_loss_weight: Optional[float] = 1.0,
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
            negative_loss_weight=negative_loss_weight,
        )
        self.instance_peaks_inf_layer = TopDownMultiClassFindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.multi_class_topdown.confmaps.output_stride,
        )

        self.node_names = self.head_configs.multi_class_topdown.confmaps.part_names

    def get_visualization_data(self, sample) -> VisualizationData:
        """Extract visualization data from a sample."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0)
        output = self.instance_peaks_inf_layer(ex)

        peaks = output["pred_instance_peaks"].cpu().numpy()
        peak_values = output["pred_peak_values"].cpu().numpy()
        img = output["instance_image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instance"].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=list(self.node_names) if self.node_names else [],
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=True,
        )

    def visualize_example(self, sample):
        """Visualize predictions during training (used with callbacks)."""
        data = self.get_visualization_data(sample)
        scale = 1.0
        if data.image.shape[0] < 512:
            scale = 2.0
        if data.image.shape[0] < 256:
            scale = 4.0
        fig = plot_img(data.image, dpi=72 * scale, scale=scale)
        plot_confmaps(data.pred_confmaps, output_scale=data.output_scale)
        plot_peaks(data.gt_instances, data.pred_peaks, paired=data.is_paired)
        return fig

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
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
        X = normalize_on_gpu(X)
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
                    f"train/confmaps/{name}",
                    channel_wise_loss[node_idx],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Log step-level loss (every batch, uses global_step x-axis)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        # Accumulate for epoch-averaged loss (logged in on_train_epoch_end)
        self._accumulate_loss(loss)
        self.log(
            "train/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/classvector_loss",
            classvector_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute classification accuracy
        with torch.no_grad():
            pred_classes = torch.argmax(classvector, dim=1)
            gt_classes = torch.argmax(y_classvector, dim=1)
            class_accuracy = (pred_classes == gt_classes).float().mean()
        self.log(
            "train/class_accuracy",
            class_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X = torch.squeeze(batch["instance_image"], dim=1)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1)
        y_classvector = batch["class_vectors"]
        X = normalize_on_gpu(X)
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
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/confmaps_loss",
            confmap_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/classvector_loss",
            classvector_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute classification accuracy
        with torch.no_grad():
            pred_classes = torch.argmax(classvector, dim=1)
            gt_classes = torch.argmax(y_classvector, dim=1)
            class_accuracy = (pred_classes == gt_classes).float().mean()
        self.log(
            "val/class_accuracy",
            class_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Collect predictions for epoch-end evaluation if enabled
        if self._collect_val_predictions:
            # SAVE bbox BEFORE inference (it modifies in-place!)
            bbox_prep_saved = batch["instance_bbox"].clone()

            with torch.no_grad():
                inference_output = self.instance_peaks_inf_layer(batch)

            batch_size = len(batch["frame_idx"])
            for i in range(batch_size):
                eff = batch["eff_scale"][i].cpu().numpy()

                # Predictions from inference (crop-relative, original scale)
                pred_peaks_crop = (
                    inference_output["pred_instance_peaks"][i].cpu().numpy()
                )
                pred_scores = inference_output["pred_peak_values"][i].cpu().numpy()

                # Compute bbox offset in original space from SAVED prep bbox
                # bbox has shape (n_samples=1, 4, 2) where 4 corners
                bbox_prep = bbox_prep_saved[i].squeeze(0).cpu().numpy()  # (4, 2)
                bbox_top_left_orig = (
                    bbox_prep[0] / eff
                )  # Top-left corner in original space

                # Full image coordinates (original space)
                pred_peaks_full = pred_peaks_crop + bbox_top_left_orig

                # GT transform: crop-relative preprocessed -> full image original
                gt_crop_prep = (
                    batch["instance"][i].squeeze(0).cpu().numpy()
                )  # (n_nodes, 2)
                gt_crop_orig = gt_crop_prep / eff
                gt_full_orig = gt_crop_orig + bbox_top_left_orig

                self.val_predictions.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "pred_peaks": pred_peaks_full.reshape(
                            1, -1, 2
                        ),  # (1, n_nodes, 2)
                        "pred_scores": pred_scores.reshape(1, -1),  # (1, n_nodes)
                    }
                )
                self.val_ground_truth.append(
                    {
                        "video_idx": batch["video_idx"][i].item(),
                        "frame_idx": batch["frame_idx"][i].item(),
                        "gt_instances": gt_full_orig.reshape(
                            1, -1, 2
                        ),  # (1, n_nodes, 2)
                        "num_instances": 1,
                    }
                )


# =============================================================================
# Multi-Head Models
# =============================================================================


class MultiHeadLightningModel(L.LightningModule):
    """Base PyTorch Lightning Module for multi-head sleap-nn models.

    This class enables training with multiple datasets, where each dataset has its own
    head while sharing a common backbone. Uses manual optimization to control gradient
    flow per head.

    Args:
        model_type: Type of the model. One of `single_instance`, `centered_instance`,
            `centroid`, `bottomup`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        backbone_config: Backbone configuration (string preset, dict, or DictConfig).
        head_configs: Head configuration dictionary with per-dataset configurations.
        pretrained_backbone_weights: Path to checkpoint for backbone initialization.
        pretrained_head_weights: Path to checkpoint for head initialization.
        init_weights: Model weights initialization method ("default" or "xavier").
        lr_scheduler: Learning rate scheduler configuration.
        online_mining: If True, enable online hard keypoint mining.
        hard_to_easy_ratio: Ratio threshold for hard keypoint detection.
        min_hard_keypoints: Minimum number of hard keypoints.
        max_hard_keypoints: Maximum number of hard keypoints.
        loss_scale: Factor to scale hard keypoint losses.
        optimizer: Optimizer name ("Adam" or "AdamW").
        learning_rate: Learning rate for the optimizer.
        amsgrad: Enable AMSGrad with the optimizer.
        backbone_feats: If set, return backbone features ("last" or "all").
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
        backbone_feats: Optional[str] = None,
        dataset_loss_weights: Optional[Dict[int, float]] = None,
        dataset_names: Optional[Dict[int, str]] = None,
        apply_dwa: bool = False,
        dwa_temperature: float = 2.0,
    ):
        """Initialize the configs and the model.

        Args:
            dataset_loss_weights: Optional dict mapping dataset indices to loss weights.
                If provided, overrides the per-head loss_weight values from config.
                Example: {0: 1.0, 1: 0.5, 2: 2.0} to weight dataset 0 at 1.0, dataset 1 at 0.5.
            dataset_names: Optional dict mapping dataset indices to human-readable names.
                Example: {0: "mice", 1: "flies"}.  Falls back to ``"dataset_{d_num}"``.
            apply_dwa: If True, enable Dynamic Weight Averaging (DWA). Loss weights are
                adjusted each epoch based on rate of loss change across heads.
            dwa_temperature: Temperature for the DWA softmax (default: 2.0). Lower values
                produce more aggressive reweighting toward struggling heads.
        """
        super().__init__()
        self.model_type = model_type
        self.backbone_type = backbone_type

        # Process backbone config
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
        self.backbone_feats = backbone_feats
        self.dataset_names = dataset_names

        # Gradient logging config (set via config after init)
        self.grad_logging_frequency: int = 0  # 0 = disabled, N = every N epochs
        self._grad_logging_enabled: bool = False  # Set per-epoch in on_train_epoch_start

        # Create multi-head model
        self.model = MultiHeadModel(
            backbone_type=self.backbone_type,
            backbone_config=self.backbone_config[f"{self.backbone_type}"],
            head_configs=self.head_configs[self.model_type],
            model_type=self.model_type,
        )

        # Setup per-dataset loss weights
        # Priority: dataset_loss_weights param > config loss_weight > default 1.0
        self.dataset_loss_weights = {}
        if hasattr(self.head_configs[self.model_type], "confmaps"):
            for d_num in self.head_configs[self.model_type].confmaps:
                if dataset_loss_weights is not None and d_num in dataset_loss_weights:
                    # Use explicitly provided dataset loss weight
                    self.dataset_loss_weights[d_num] = dataset_loss_weights[d_num]
                else:
                    # Fall back to config loss_weight or default
                    cfg = self.head_configs[self.model_type].confmaps[d_num]
                    self.dataset_loss_weights[d_num] = (
                        cfg.loss_weight if hasattr(cfg, "loss_weight") and cfg.loss_weight is not None else 1.0
                    )

        # Alias for backward compatibility
        self.loss_weights = self.dataset_loss_weights

        self.training_loss = {}
        self.val_loss = {}
        self.learning_rate = {}

        # For epoch-averaged loss tracking
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        # Per-head loss accumulators for val-train gap
        self._per_head_train_loss_sum: Dict[int, float] = {}
        self._per_head_train_loss_count: Dict[int, int] = {}
        self._per_head_val_loss_sum: Dict[int, float] = {}
        self._per_head_val_loss_count: Dict[int, int] = {}

        # Dynamic Weight Averaging (DWA) state
        self.apply_dwa = apply_dwa
        self.dwa_temperature = dwa_temperature
        # Stores per-head average training loss for the last 2 epochs: {d_num: [loss_t-2, loss_t-1]}
        self._dwa_loss_history: Dict[int, List[float]] = {}
        # Store initial weights as base weights for DWA (used to preserve static weight corrections)
        self._dwa_base_weights: Dict[int, float] = dict(self.dataset_loss_weights) if apply_dwa else {}

        self.loss_func = nn.MSELoss()
        self.automatic_optimization = False  # Manual optimization for multi-head

        # Initialize weights
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
                logger.info(f"Loaded pretrained encoder weights from config")

        # Load pretrained backbone weights from file
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
                logger.info(f"Loaded backbone weights from .ckpt file")

            elif self.pretrained_backbone_weights.endswith(".h5"):
                # load from sleap model weights
                load_legacy_model_weights(
                    self.model.backbone,
                    self.pretrained_backbone_weights,
                    component="backbone",
                )
                logger.info(f"Loaded backbone weights from .h5 file")

            elif self.pretrained_backbone_weights in MODEL_WEIGHTS:
                # ImageNet pretrained weights (convnext/swint)
                self.model.backbone.encoder.load_state_dict(
                    MODEL_WEIGHTS[self.pretrained_backbone_weights].IMAGENET1K_V1.get_state_dict(
                        progress=True, check_hash=True
                    )
                )
                logger.info(f"Loaded pretrained weights: {self.pretrained_backbone_weights}")

            else:
                message = f"Unsupported file extension for pretrained backbone weights. Please provide a .ckpt or .h5 file."
                logger.error(message)
                raise ValueError(message)

        # Load pretrained head weights from file
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
                logger.info(f"Loaded head weights from .ckpt file")

            elif self.pretrained_head_weights.endswith(".h5"):
                # load from sleap model weights
                load_legacy_model_weights(
                    self.model.head_layers,
                    self.pretrained_head_weights,
                    component="head",
                )
                logger.info(f"Loaded head weights from .h5 file")

            else:
                message = f"Unsupported file extension for pretrained head weights. Please provide a .ckpt or .h5 file."
                logger.error(message)
                raise ValueError(message)

    def on_train_epoch_start(self):
        """Configure the train timer at the beginning of each epoch."""
        self.train_start_time = time.time()
        # Reset epoch loss tracking
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        # Reset per-head train loss accumulators (for val-train gap)
        self._per_head_train_loss_sum: Dict[int, float] = {}
        self._per_head_train_loss_count: Dict[int, int] = {}
        # Check if gradient logging should be active this epoch
        self._grad_logging_enabled = (
            self.grad_logging_frequency > 0
            and (self.current_epoch + 1) % self.grad_logging_frequency == 0
        )

    def _accumulate_loss(self, loss: torch.Tensor):
        """Accumulate loss for epoch-averaged logging. Call this in training_step."""
        self._epoch_loss_sum += loss.detach().item()
        self._epoch_loss_count += 1

    def on_train_epoch_end(self):
        """Configure the train timer at the end of every epoch."""
        train_time = time.time() - self.train_start_time
        self.log(
            "train/time",
            train_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch explicitly for custom x-axis support in wandb
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch-averaged training loss
        if self._epoch_loss_count > 0:
            avg_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.log(
                "train/loss",
                avg_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # Log current learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                "train/lr",
                lr,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Dynamic Weight Averaging: update loss weights for next epoch
        if self.apply_dwa:
            self._update_dwa_weights()

    def on_validation_epoch_start(self):
        """Configure the val timer at the beginning of each epoch."""
        self.val_start_time = time.time()
        # Reset per-head val loss accumulators (for val-train gap)
        self._per_head_val_loss_sum: Dict[int, float] = {}
        self._per_head_val_loss_count: Dict[int, int] = {}

    def on_validation_epoch_end(self):
        """Configure the val timer at the end of every epoch."""
        val_time = time.time() - self.val_start_time
        self.log(
            "val/time",
            val_time,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log epoch explicitly so val/* metrics can use it as x-axis in wandb
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Log per-head val-train loss gap
        for d_num in self._per_head_val_loss_sum:
            if (
                d_num in self._per_head_train_loss_sum
                and self._per_head_val_loss_count.get(d_num, 0) > 0
                and self._per_head_train_loss_count.get(d_num, 0) > 0
            ):
                avg_val = (
                    self._per_head_val_loss_sum[d_num]
                    / self._per_head_val_loss_count[d_num]
                )
                avg_train = (
                    self._per_head_train_loss_sum[d_num]
                    / self._per_head_train_loss_count[d_num]
                )
                gap = avg_val - avg_train
                head_name = self._dataset_name(d_num)
                self.log(
                    f"gap/head_{head_name}",
                    gap,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Manual LR scheduler stepping (needed with automatic_optimization=False)
        sch = self.lr_schedulers()
        if sch is not None:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = self.trainer.callback_metrics.get("val_loss")
                if val_loss is not None:
                    sch.step(val_loss)
            else:
                sch.step()

    def _dataset_name(self, d_num: int) -> str:
        """Return a human-readable name for dataset index ``d_num``."""
        if self.dataset_names and d_num in self.dataset_names:
            return self.dataset_names[d_num]
        return f"dataset_{d_num}"

    def _update_dwa_weights(self):
        """Update loss weights using Dynamic Weight Averaging (Liu et al., 2019).

        Computes the rate of loss change for each head over the last 2 epochs,
        then applies a softmax to produce weights that emphasize heads with slower
        progress (higher loss ratio = loss not decreasing much = needs more attention).

        Weights are scaled to preserve total loss magnitude (sum = num_datasets).
        If static dataset weights were provided, DWA modulates on top of them.
        """
        import math

        # Record this epoch's per-head average training loss
        for d_num in self._per_head_train_loss_sum:
            count = self._per_head_train_loss_count.get(d_num, 0)
            if count > 0:
                avg_loss = self._per_head_train_loss_sum[d_num] / count
                if d_num not in self._dwa_loss_history:
                    self._dwa_loss_history[d_num] = []
                self._dwa_loss_history[d_num].append(avg_loss)
                # Only keep last 2 epochs
                if len(self._dwa_loss_history[d_num]) > 2:
                    self._dwa_loss_history[d_num] = self._dwa_loss_history[d_num][-2:]

        # Need at least 2 epochs of history to compute rate of change
        all_heads = list(self._dwa_loss_history.keys())
        if not all_heads or any(len(self._dwa_loss_history[d]) < 2 for d in all_heads):
            return

        # Compute loss ratios: r_i = L_i(t-1) / L_i(t-2)
        # r > 1 means loss increased (struggling), r < 1 means loss decreased (progressing)
        ratios = {}
        for d_num in all_heads:
            prev, curr = self._dwa_loss_history[d_num]
            # Guard against zero/near-zero previous loss
            ratios[d_num] = curr / max(prev, 1e-12)

        # Softmax with temperature: w_i = exp(r_i / T) / sum(exp(r_j / T))
        T = self.dwa_temperature
        max_ratio = max(ratios.values())  # For numerical stability
        exp_vals = {d: math.exp((r - max_ratio) / T) for d, r in ratios.items()}
        exp_sum = sum(exp_vals.values())
        num_heads = len(all_heads)

        # Scale so weights sum to num_heads (preserves loss magnitude)
        dwa_weights = {d: (v / exp_sum) * num_heads for d, v in exp_vals.items()}

        # If static base weights exist, multiply DWA factors on top and re-normalize
        if self._dwa_base_weights:
            combined = {d: self._dwa_base_weights.get(d, 1.0) * dwa_weights[d] for d in all_heads}
            combined_sum = sum(combined.values())
            # Re-normalize to preserve total loss magnitude (sum = num_heads)
            final_weights = {d: (w / combined_sum) * num_heads for d, w in combined.items()}
        else:
            final_weights = dwa_weights

        # Update the live weights used in training_step
        self.dataset_loss_weights.update(final_weights)
        self.loss_weights = self.dataset_loss_weights

        # Log DWA weights
        for d_num, w in final_weights.items():
            self.log(
                f"dwa_weight/{self._dataset_name(d_num)}",
                w,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    @staticmethod
    def get_lightning_model_from_config(
        config: DictConfig,
        dataset_loss_weights: Optional[Dict[int, float]] = None,
        dataset_names: Optional[Dict[int, str]] = None,
        apply_dwa: bool = False,
        dwa_temperature: float = 2.0,
    ):
        """Create the appropriate multi-head Lightning model based on config.

        Args:
            config: Training job configuration.
            dataset_loss_weights: Optional dict mapping dataset indices to loss weights.
                If provided, overrides the per-head loss_weight values from config.
                Example: {0: 1.0, 1: 0.5, 2: 2.0} to weight dataset 0 at 1.0, dataset 1 at 0.5.
            dataset_names: Optional dict mapping dataset indices to human-readable names.
            apply_dwa: If True, enable Dynamic Weight Averaging.
            dwa_temperature: Temperature for the DWA softmax.

        Returns:
            MultiHeadLightningModel: Instance of the appropriate Lightning module.
        """
        model_type = get_model_type_from_cfg(config)
        backbone_type = get_backbone_type_from_cfg(config)

        lightning_models = {
            "single_instance": SingleInstanceMultiHeadLightningModule,
            "centroid": CentroidMultiHeadLightningModule,
            "centered_instance": TopDownCenteredInstanceMultiHeadLightningModule,
        }

        if model_type not in lightning_models:
            message = f"Multi-head training not supported for model type: {model_type}. Supported: {list(lightning_models.keys())}"
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
            dataset_loss_weights=dataset_loss_weights,
            dataset_names=dataset_names,
            apply_dwa=apply_dwa,
            dwa_temperature=dwa_temperature,
        )

        return lightning_model

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        if self.optimizer == "Adam":
            optim = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
            )
        elif self.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
            )
        else:
            message = f"Unknown optimizer: {self.optimizer}. Choose from ['Adam', 'AdamW']"
            logger.error(message)
            raise ValueError(message)

        # Learning rate scheduler setup
        if self.lr_scheduler is None:
            return {"optimizer": optim}

        # Handle different scheduler types
        if isinstance(self.lr_scheduler, DictConfig):
            # Warmup schedulers
            if OmegaConf.select(self.lr_scheduler, "cosine_annealing_warmup", default=None) is not None:
                cfg = self.lr_scheduler.cosine_annealing_warmup
                # Use trainer's max_epochs if not specified in config
                max_epochs = (
                    cfg.max_epochs
                    if cfg.max_epochs is not None
                    else self.trainer.max_epochs
                )
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optim,
                    warmup_epochs=cfg.warmup_epochs,
                    max_epochs=max_epochs,
                    warmup_start_lr=cfg.warmup_start_lr,
                    eta_min=cfg.eta_min,
                )
                return {
                    "optimizer": optim,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif OmegaConf.select(self.lr_scheduler, "linear_warmup_linear_decay", default=None) is not None:
                cfg = self.lr_scheduler.linear_warmup_linear_decay
                # Use trainer's max_epochs if not specified in config
                max_epochs = (
                    cfg.max_epochs
                    if cfg.max_epochs is not None
                    else self.trainer.max_epochs
                )
                scheduler = LinearWarmupLinearDecayLR(
                    optimizer=optim,
                    warmup_epochs=cfg.warmup_epochs,
                    max_epochs=max_epochs,
                    warmup_start_lr=cfg.warmup_start_lr,
                    end_lr=cfg.end_lr,
                )
                return {
                    "optimizer": optim,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif OmegaConf.select(self.lr_scheduler, "step_lr", default=None) is not None:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optim,
                    step_size=self.lr_scheduler.step_lr.step_size,
                    gamma=self.lr_scheduler.step_lr.gamma,
                )
                return {
                    "optimizer": optim,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif OmegaConf.select(self.lr_scheduler, "reduce_lr_on_plateau", default=None) is not None:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim,
                    mode="min",
                    threshold=self.lr_scheduler.reduce_lr_on_plateau.threshold,
                    threshold_mode=self.lr_scheduler.reduce_lr_on_plateau.threshold_mode,
                    cooldown=self.lr_scheduler.reduce_lr_on_plateau.cooldown,
                    patience=self.lr_scheduler.reduce_lr_on_plateau.patience,
                    factor=self.lr_scheduler.reduce_lr_on_plateau.factor,
                    min_lr=self.lr_scheduler.reduce_lr_on_plateau.min_lr,
                )
                return {
                    "optimizer": optim,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "val_loss",
                    },
                }

        return {"optimizer": optim}


class CentroidMultiHeadLightningModule(MultiHeadLightningModel):
    """Lightning Module for multi-head centroid detection.

    Trains a shared backbone with multiple centroid detection heads, one per dataset.
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
        backbone_feats: Optional[str] = None,
        dataset_loss_weights: Optional[Dict[int, float]] = None,
        dataset_names: Optional[Dict[int, str]] = None,
        apply_dwa: bool = False,
        dwa_temperature: float = 2.0,
    ):
        """Initialize the centroid multi-head module."""
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
            backbone_feats=backbone_feats,
            dataset_loss_weights=dataset_loss_weights,
            dataset_names=dataset_names,
            apply_dwa=apply_dwa,
            dwa_temperature=dwa_temperature,
        )

        # Inference layer for visualization
        self.centroid_inf_layer = CentroidCrop(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centroid.confmaps[0]["output_stride"],
            input_scale=1.0,
        )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        output = self.model(
            img,
            include_backbone_features=self.backbone_feats is not None,
            backbone_outputs=self.backbone_feats,
        )
        return {
            "head": output["CentroidConfmapsHead"],
            "backbone_features": output.get("backbone_features"),
            "strides": output.get("backbone_features_strides"),
        }

    def visualize_example(self, sample, d_idx):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        gt_centroids = ex["centroids"].cpu().numpy()

        self.centroid_inf_layer.output_stride = self.head_configs.centroid.confmaps[
            d_idx
        ]["output_stride"]
        output = self.centroid_inf_layer(ex, output_head_skeleton_num=d_idx)

        peaks = output["centroids"][0].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        confmaps = output["pred_centroid_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plot_peaks(gt_centroids, peaks, paired=False)
        return fig

    def _forward_no_ddp(self, img):
        """Forward pass bypassing DDP wrapper.

        Used for visualization on rank 0 only, where calling the DDP-wrapped
        model would corrupt reducer state (other ranks don't participate).
        """
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        raw_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        output = raw_model(
            img,
            include_backbone_features=self.backbone_feats is not None,
            backbone_outputs=self.backbone_feats,
        )
        return {
            "head": output["CentroidConfmapsHead"],
            "backbone_features": output.get("backbone_features"),
            "strides": output.get("backbone_features_strides"),
        }

    def get_visualization_data(self, sample, dataset_idx: int = 0) -> VisualizationData:
        """Extract visualization data from a sample for a specific dataset/head.

        Args:
            sample: A sample from the visualization dataset.
            dataset_idx: Index of the dataset/head to use for inference.

        Returns:
            VisualizationData object with all needed fields for visualization.
        """
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["image"] = ex["image"].unsqueeze(dim=0)
        gt_centroids = ex["centroids"].cpu().numpy()

        viz_inf_layer = CentroidCrop(
            torch_model=self._forward_no_ddp,
            peak_threshold=self.centroid_inf_layer.peak_threshold,
            return_confmaps=True,
            output_stride=self.head_configs.centroid.confmaps[dataset_idx]["output_stride"],
            input_scale=self.centroid_inf_layer.input_scale,
        )
        with torch.no_grad():
            output = viz_inf_layer(ex, output_head_skeleton_num=dataset_idx)

        peaks = output["centroids"][0].cpu().numpy()
        img = output["image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        confmaps = output["pred_centroid_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=None,
            gt_instances=gt_centroids,
            node_names=["centroid"],
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=False,
        )

    def training_step(self, batch, batch_idx):
        """Training step for multi-head centroid model."""
        total_loss = 0
        opt = self.optimizers()
        opt.zero_grad()

        # Get batch size from first dataset
        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            batch_data = batch[d_num]
            X = torch.squeeze(batch_data["image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch_data["centroids_confidence_maps"], dim=1)

            # Forward pass through all heads
            output = self.model(X)["CentroidConfmapsHead"]

            # Detach gradients from non-matching heads
            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output[h_num] = output[h_num].detach()

            # Get prediction for this dataset's head
            y_preds = output[d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * self.loss_func(y_preds, y)
            total_loss += curr_loss.detach()

            self.manual_backward(curr_loss)

            self.log(
                f"train/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # Log step-level loss for progress bar
        self.log(
            "loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Accumulate loss for epoch-averaged logging
        self._accumulate_loss(total_loss)

        opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-head centroid model."""
        total_loss = 0
        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            X = torch.squeeze(batch[d_num]["image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch[d_num]["centroids_confidence_maps"], dim=1)

            y_preds = self.model(X)["CentroidConfmapsHead"][d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * nn.MSELoss()(y_preds, y)
            total_loss += curr_loss

            self.log(
                f"val/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        self.log(
            "val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # Also log with val/ prefix for consistency
        self.log(
            "val/loss",
            total_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )


class TopDownCenteredInstanceMultiHeadLightningModule(MultiHeadLightningModel):
    """Lightning Module for multi-head top-down centered instance pose estimation.

    Trains a shared backbone with multiple instance pose estimation heads, one per dataset.
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
        backbone_feats: Optional[str] = None,
        dataset_loss_weights: Optional[Dict[int, float]] = None,
        dataset_names: Optional[Dict[int, str]] = None,
        apply_dwa: bool = False,
        dwa_temperature: float = 2.0,
    ):
        """Initialize the top-down centered instance multi-head module."""
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
            backbone_feats=backbone_feats,
            dataset_loss_weights=dataset_loss_weights,
            dataset_names=dataset_names,
            apply_dwa=apply_dwa,
            dwa_temperature=dwa_temperature,
        )

        # Inference layer for visualization
        self.instance_peaks_inf_layer = FindInstancePeaks(
            torch_model=self.forward,
            peak_threshold=0.2,
            return_confmaps=True,
            output_stride=self.head_configs.centered_instance.confmaps[0]["output_stride"],
        )

        # Store node names per head for logging
        self.node_names = {}
        for d_num in self.head_configs.centered_instance.confmaps:
            self.node_names[d_num] = self.head_configs.centered_instance.confmaps[d_num].get(
                "part_names", []
            )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        output = self.model(
            img,
            include_backbone_features=self.backbone_feats is not None,
            backbone_outputs=self.backbone_feats,
        )
        return {
            "head": output["CenteredInstanceConfmapsHead"],
            "backbone_features": output.get("backbone_features"),
            "strides": output.get("backbone_features_strides"),
        }

    def visualize_example(self, sample, d_idx):
        """Visualize predictions during training (used with callbacks)."""
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0)

        self.instance_peaks_inf_layer.output_stride = (
            self.head_configs.centered_instance.confmaps[d_idx]["output_stride"]
        )
        output = self.instance_peaks_inf_layer(ex, output_head_skeleton_num=d_idx)

        peaks = output["pred_instance_peaks"].cpu().numpy()
        img = output["instance_image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instance"].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0
        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(confmaps, output_scale=confmaps.shape[0] / img.shape[0])
        plot_peaks(gt_instances, peaks, paired=True)
        return fig

    def _forward_no_ddp(self, img):
        """Forward pass bypassing DDP wrapper.

        Used for visualization on rank 0 only, where calling the DDP-wrapped
        model would corrupt reducer state (other ranks don't participate).
        """
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        # Access unwrapped model to avoid DDP collective ops
        raw_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        output = raw_model(
            img,
            include_backbone_features=self.backbone_feats is not None,
            backbone_outputs=self.backbone_feats,
        )
        return {
            "head": output["CenteredInstanceConfmapsHead"],
            "backbone_features": output.get("backbone_features"),
            "strides": output.get("backbone_features_strides"),
        }

    def get_visualization_data(self, sample, dataset_idx: int = 0) -> VisualizationData:
        """Extract visualization data from a sample for a specific dataset/head.

        Args:
            sample: A sample from the visualization dataset.
            dataset_idx: Index of the dataset/head to use for inference.

        Returns:
            VisualizationData object with all needed fields for visualization.
        """
        ex = sample.copy()
        ex["eff_scale"] = torch.tensor([1.0])
        for k, v in ex.items():
            if isinstance(v, torch.Tensor):
                ex[k] = v.to(device=self.device)
        ex["instance_image"] = ex["instance_image"].unsqueeze(dim=0)

        # Use a DDP-safe inference layer that bypasses the DDP wrapper.
        # The main instance_peaks_inf_layer uses self.forward → self.model (DDP),
        # which corrupts DDP reducer state when called on rank 0 only.
        viz_inf_layer = FindInstancePeaks(
            torch_model=self._forward_no_ddp,
            peak_threshold=self.instance_peaks_inf_layer.peak_threshold,
            return_confmaps=True,
            output_stride=self.head_configs.centered_instance.confmaps[dataset_idx]["output_stride"],
        )
        with torch.no_grad():
            output = viz_inf_layer(ex, output_head_skeleton_num=dataset_idx)

        peaks = output["pred_instance_peaks"].cpu().numpy()
        peak_values = output["pred_peak_values"].cpu().numpy()
        img = output["instance_image"][0, 0].cpu().numpy().transpose(1, 2, 0)
        gt_instances = ex["instance"].cpu().numpy()
        confmaps = output["pred_confmaps"][0].cpu().numpy().transpose(1, 2, 0)

        node_names = self.node_names.get(dataset_idx, [])

        return VisualizationData(
            image=img,
            pred_confmaps=confmaps,
            pred_peaks=peaks,
            pred_peak_values=peak_values,
            gt_instances=gt_instances,
            node_names=list(node_names) if node_names else [],
            output_scale=confmaps.shape[0] / img.shape[0],
            is_paired=True,
        )

    def training_step(self, batch, batch_idx):
        """Training step for multi-head top-down model."""
        total_loss = 0
        opt = self.optimizers()
        opt.zero_grad()

        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            batch_data = batch[d_num]
            X = torch.squeeze(batch_data["instance_image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch_data["confidence_maps"], dim=1)

            # Forward pass through all heads
            output = self.model(X)["CenteredInstanceConfmapsHead"]

            # Detach gradients from non-matching heads
            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output[h_num] = output[h_num].detach()

            # Get prediction for this dataset's head
            y_preds = output[d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * self.loss_func(y_preds, y)

            # Online hard keypoint mining
            if self.online_mining:
                ohkm_loss = compute_ohkm_loss(
                    y_gt=y,
                    y_pr=y_preds,
                    hard_to_easy_ratio=self.hard_to_easy_ratio,
                    min_hard_keypoints=self.min_hard_keypoints,
                    max_hard_keypoints=self.max_hard_keypoints,
                    loss_scale=self.loss_scale,
                )
                curr_loss = curr_loss + ohkm_loss

            total_loss += curr_loss.detach()

            # Accumulate per-head train loss for val-train gap
            loss_val = curr_loss.detach().item()
            self._per_head_train_loss_sum[d_num] = (
                self._per_head_train_loss_sum.get(d_num, 0.0) + loss_val
            )
            self._per_head_train_loss_count[d_num] = (
                self._per_head_train_loss_count.get(d_num, 0) + 1
            )

            self.manual_backward(curr_loss)

            # Log backbone gradient norm after this head's backward pass.
            # Since gradients accumulate across heads,
            # this is the cumulative norm after head d_num. Comparing across
            # heads shows each head's relative contribution to backbone updates.
            if self._grad_logging_enabled:
                backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.backbone.parameters(), max_norm=float("inf")
                )
                self.log(
                    f"grad_norm/backbone_cumulative_after_{self._dataset_name(d_num)}",
                    backbone_grad_norm,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # Per-node loss logging
            if d_num in self.node_names and len(self.node_names[d_num]) > 0:
                batch_size_local, _, h, w = y.shape
                mse = (y - y_preds) ** 2
                channel_wise_loss = torch.sum(mse, dim=(0, 2, 3)) / (batch_size_local * h * w)
                head_name = self._dataset_name(d_num)
                for node_idx, name in enumerate(self.node_names[d_num]):
                    if node_idx < channel_wise_loss.shape[0]:
                        self.log(
                            f"train/head_{head_name}/confmaps/{name}",
                            channel_wise_loss[node_idx],
                            prog_bar=False,
                            on_step=False,
                            on_epoch=True,
                            logger=True,
                            sync_dist=True,
                            batch_size=batch_size,
                        )

            self.log(
                f"train/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # Log step-level loss for progress bar
        self.log(
            "loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Accumulate loss for epoch-averaged logging
        self._accumulate_loss(total_loss)

        opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-head top-down model."""
        total_loss = 0
        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            X = torch.squeeze(batch[d_num]["instance_image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch[d_num]["confidence_maps"], dim=1)

            y_preds = self.model(X)["CenteredInstanceConfmapsHead"][d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * nn.MSELoss()(y_preds, y)

            # Online hard keypoint mining for validation
            if self.online_mining:
                ohkm_loss = compute_ohkm_loss(
                    y_gt=y,
                    y_pr=y_preds,
                    hard_to_easy_ratio=self.hard_to_easy_ratio,
                    min_hard_keypoints=self.min_hard_keypoints,
                    max_hard_keypoints=self.max_hard_keypoints,
                    loss_scale=self.loss_scale,
                )
                curr_loss = curr_loss + ohkm_loss

            total_loss += curr_loss

            # Accumulate per-head val loss for val-train gap
            self._per_head_val_loss_sum[d_num] = (
                self._per_head_val_loss_sum.get(d_num, 0.0)
                + curr_loss.detach().item()
            )
            self._per_head_val_loss_count[d_num] = (
                self._per_head_val_loss_count.get(d_num, 0) + 1
            )

            self.log(
                f"val/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        self.log(
            "val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # Also log with val/ prefix for consistency
        self.log(
            "val/loss",
            total_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )


class SingleInstanceMultiHeadLightningModule(MultiHeadLightningModel):
    """Lightning Module for multi-head single instance pose estimation.

    Trains a shared backbone with multiple single-instance pose estimation heads,
    one per dataset.
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
        backbone_feats: Optional[str] = None,
        dataset_loss_weights: Optional[Dict[int, float]] = None,
        dataset_names: Optional[Dict[int, str]] = None,
        apply_dwa: bool = False,
        dwa_temperature: float = 2.0,
    ):
        """Initialize the single instance multi-head module."""
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
            backbone_feats=backbone_feats,
            dataset_loss_weights=dataset_loss_weights,
            dataset_names=dataset_names,
            apply_dwa=apply_dwa,
            dwa_temperature=dwa_temperature,
        )

        # Store node names per head for logging
        self.node_names = {}
        for d_num in self.head_configs.single_instance.confmaps:
            self.node_names[d_num] = self.head_configs.single_instance.confmaps[d_num].get(
                "part_names", []
            )

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        img = normalize_on_gpu(img)
        output = self.model(
            img,
            include_backbone_features=self.backbone_feats is not None,
            backbone_outputs=self.backbone_feats,
        )
        return {
            "head": output["SingleInstanceConfmapsHead"],
            "backbone_features": output.get("backbone_features"),
            "strides": output.get("backbone_features_strides"),
        }

    def training_step(self, batch, batch_idx):
        """Training step for multi-head single instance model."""
        total_loss = 0
        opt = self.optimizers()
        opt.zero_grad()

        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            batch_data = batch[d_num]
            X = torch.squeeze(batch_data["image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch_data["confidence_maps"], dim=1)

            # Forward pass through all heads
            output = self.model(X)["SingleInstanceConfmapsHead"]

            # Detach gradients from non-matching heads
            for h_num in batch.keys():
                if d_num != h_num:
                    with torch.no_grad():
                        output[h_num] = output[h_num].detach()

            # Get prediction for this dataset's head
            y_preds = output[d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * self.loss_func(y_preds, y)

            # Online hard keypoint mining
            if self.online_mining:
                ohkm_loss = compute_ohkm_loss(
                    y_gt=y,
                    y_pr=y_preds,
                    hard_to_easy_ratio=self.hard_to_easy_ratio,
                    min_hard_keypoints=self.min_hard_keypoints,
                    max_hard_keypoints=self.max_hard_keypoints,
                    loss_scale=self.loss_scale,
                )
                curr_loss = curr_loss + ohkm_loss

            total_loss += curr_loss.detach()

            self.manual_backward(curr_loss)

            self.log(
                f"train/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # Log step-level loss for progress bar
        self.log(
            "loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Accumulate loss for epoch-averaged logging
        self._accumulate_loss(total_loss)

        opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step for multi-head single instance model."""
        total_loss = 0
        first_key = list(batch.keys())[0]
        batch_size = batch[first_key]["frame_idx"].shape[0]

        for d_num in batch.keys():
            X = torch.squeeze(batch[d_num]["image"], dim=1)
            X = normalize_on_gpu(X)  # Normalize input
            y = torch.squeeze(batch[d_num]["confidence_maps"], dim=1)

            y_preds = self.model(X)["SingleInstanceConfmapsHead"][d_num]
            loss_weight = self.loss_weights.get(d_num, 1.0)
            curr_loss = loss_weight * nn.MSELoss()(y_preds, y)

            if self.online_mining:
                ohkm_loss = compute_ohkm_loss(
                    y_gt=y,
                    y_pr=y_preds,
                    hard_to_easy_ratio=self.hard_to_easy_ratio,
                    min_hard_keypoints=self.min_hard_keypoints,
                    max_hard_keypoints=self.max_hard_keypoints,
                    loss_scale=self.loss_scale,
                )
                curr_loss = curr_loss + ohkm_loss

            total_loss += curr_loss

            self.log(
                f"val/head_{self._dataset_name(d_num)}_loss",
                curr_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        self.log(
            "val_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # Also log with val/ prefix for consistency
        self.log(
            "val/loss",
            total_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
