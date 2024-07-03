"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
from typing import Optional, List
import time
from torch import nn
import os
import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import lightning as L
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
    CentroidConfmapsPipeline,
    BottomUpPipeline,
)
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from sleap_nn.architectures.model import Model
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sleap_nn.data.cycler import CyclerIterDataPipe as Cycler
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
        self.steps_per_epoch = self.config.trainer_config.steps_per_epoch

        # initialize attributes
        self.model = None
        self.provider = None
        self.skeletons = None
        self.train_data_loader = None
        self.val_data_loader = None

        # set seed
        torch.manual_seed(self.seed)

    def _create_data_loaders(self):
        """Create a DataLoader for train, validation and test sets using the data_config."""
        self.provider = self.config.data_config.provider
        if self.provider == "LabelsReader":
            self.provider = LabelsReader

        if self.config.data_config.pipeline == "SingleInstanceConfmaps":
            train_pipeline = SingleInstanceConfmapsPipeline(
                data_config=self.config.data_config.train,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )
            val_pipeline = SingleInstanceConfmapsPipeline(
                data_config=self.config.data_config.val,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )

        elif self.config.data_config.pipeline == "TopdownConfmaps":
            train_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.train,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )
            val_pipeline = TopdownConfmapsPipeline(
                data_config=self.config.data_config.val,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )

        elif self.config.data_config.pipeline == "CentroidConfmaps":
            train_pipeline = CentroidConfmapsPipeline(
                data_config=self.config.data_config.train,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )
            val_pipeline = CentroidConfmapsPipeline(
                data_config=self.config.data_config.val,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
            )

        elif self.config.data_config.pipeline == "BottomUp":
            train_pipeline = BottomUpPipeline(
                data_config=self.config.data_config.train,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
                pafs_head=self.config.model_config.head_configs.pafs.head_config,
            )
            val_pipeline = BottomUpPipeline(
                data_config=self.config.data_config.val,
                max_stride=self.config.model_config.backbone_config.backbone_config.max_stride,
                confmap_head=self.config.model_config.head_configs.confmaps.head_config,
                pafs_head=self.config.model_config.head_configs.pafs.head_config,
            )

        else:
            raise Exception(f"{self.config.data_config.pipeline} is not defined.")

        # train
        train_labels = sio.load_slp(self.config.data_config.train.labels_path)
        self.skeletons = train_labels.skeletons

        train_labels_reader = self.provider(train_labels)

        train_datapipe = train_pipeline.make_training_pipeline(
            data_provider=train_labels_reader,
        )
        if self.steps_per_epoch is not None:
            train_datapipe = Cycler(train_datapipe)

        # to remove duplicates when multiprocessing is used
        train_datapipe = train_datapipe.sharding_filter()
        self.train_data_loader = DataLoader(
            train_datapipe,
            **dict(self.config.trainer_config.train_data_loader),
        )

        # val
        val_labels_reader = self.provider.from_filename(
            self.config.data_config.val.labels_path,
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
        models = {
            "SingleInstanceConfmaps": SingleInstanceModel,
            "TopdownConfmaps": TopDownCenteredInstanceModel,
            "CentroidConfmaps": CentroidModel,
            "BottomUp": BottomUpModel,
        }
        self.model = models[self.config.data_config.pipeline](
            self.config, self.skeletons
        )

    def _get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        """Initiate the training by calling the fit method of Trainer."""
        self._create_data_loaders()
        logger = []
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
        if self.config.trainer_config.save_ckpt:

            # create checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                **dict(self.config.trainer_config.model_ckpt),
                dirpath=dir_path,
                filename="best",
                monitor="val_loss",
                mode="min",
            )
            callbacks = [checkpoint_callback]
            # logger to create csv with metrics values over the epochs
            csv_logger = CSVLogger(dir_path)
            logger.append(csv_logger)

        else:
            callbacks = []

        if self.config.trainer_config.early_stopping.stop_training_on_plateau:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=False,
                    min_delta=self.config.trainer_config.early_stopping.min_delta,
                    patience=self.config.trainer_config.early_stopping.patience,
                )
            )

        if self.config.trainer_config.use_wandb:
            wandb_config = self.config.trainer_config.wandb
            if wandb_config.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            else:
                self._set_wandb()
            wandb_logger = WandbLogger(
                project=wandb_config.project, name=wandb_config.name, save_dir=dir_path
            )
            logger.append(wandb_logger)

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
            logger=logger,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            limit_train_batches=self.steps_per_epoch,
        )

        trainer.fit(self.model, self.train_data_loader, self.val_data_loader)

        total_params = self._get_param_count()

        if self.config.trainer_config.use_wandb:
            self.config.trainer_config.wandb.run_id = wandb.run.id
            self.config.model_config.total_params = total_params
            for m in self.config.trainer_config.wandb.log_params:
                list_keys = m.split(".")
                key = list_keys[-1]
                value = self.config[list_keys[0]]
                for l in list_keys[1:]:
                    value = value[l]
                wandb_logger.experiment.config.update({key: value})
            wandb_logger.experiment.config.update({"model_params": total_params})
            wandb.finish()

        # save the configs as yaml in the checkpoint dir
        OmegaConf.save(config=self.config, f=f"{dir_path}/training_config.yaml")


class TrainingModel(L.LightningModule):
    """Base PyTorch Lightning Module for all sleap-nn models.

    This class is a sub-class of Torch Lightning Module to configure the training and validation steps.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                a pipeline class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        skeletons: List of `sio.Skeleton` objects from the input `.slp` file.
    """

    def __init__(
        self, config: OmegaConf, skeletons: Optional[List[sio.Skeleton]] = None
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.skeletons = skeletons
        self.model_config = self.config.model_config
        self.trainer_config = self.config.trainer_config
        self.data_config = self.config.data_config
        self.m_device = self.trainer_config.device
        self.input_expand_channels = (
            self.model_config.backbone_config.backbone_config.in_channels
        )
        if self.model_config.pre_trained_weights:
            ckpt = eval(self.model_config.pre_trained_weights).DEFAULT.get_state_dict(
                progress=True, check_hash=True
            )
            input_channels = ckpt["features.0.0.weight"].shape[-3]
            if (
                self.model_config.backbone_config.backbone_config.in_channels
                != input_channels
            ):
                self.input_expand_channels = input_channels
                OmegaConf.update(
                    self.model_config,
                    "backbone_config.backbone_config.in_channels",
                    input_channels,
                )

        # if edges and part names aren't set in config, get it from `sio.Labels` object.
        head_configs = self.model_config.head_configs
        for key in head_configs:
            if "part_names" in head_configs[key].head_config.keys():
                if head_configs[key].head_config["part_names"] is None:
                    part_names = [x.name for x in self.skeletons[0].nodes]
                    head_configs[key].head_config["part_names"] = part_names

            if "edges" in head_configs[key].head_config.keys():
                if head_configs[key].head_config["edges"] is None:
                    edges = [
                        (x.source.name, x.destination.name)
                        for x in self.skeletons[0].edges
                    ]
                    head_configs[key].head_config["edges"] = edges

        self.model = Model(
            backbone_config=self.model_config.backbone_config,
            head_configs=head_configs,
            input_expand_channels=self.input_expand_channels,
        ).to(self.m_device)

        self.loss_weights = [
            self.model_config.head_configs[x].head_config.loss_weight
            for x in self.model_config.head_configs
        ]
        self.training_loss = {}
        self.val_loss = {}
        self.learning_rate = {}

        # Initialization for encoder and decoder stacks.
        if self.model_config.init_weights == "xavier":
            self.model.apply(xavier_init_weights)

        # Pre-trained weights for the encoder stack.
        if self.model_config.pre_trained_weights:
            self.model.backbone.enc.load_state_dict(ckpt, strict=False)

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
        if self.trainer_config.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                **dict(self.trainer_config.optimizer),
            )
        elif self.trainer_config.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
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
        skeletons: List of `sio.Skeleton` objects from the input `.slp` file.

    """

    def __init__(
        self, config: OmegaConf, skeletons: Optional[List[sio.Skeleton]] = None
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons)

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

    This is a subclass of the `TrainingModel` to configure the training/ validation steps
    and forward pass specific to TopDown Centered instance model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `TopdownConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        skeletons: List of `sio.Skeleton` objects from the input `.slp` file.

    """

    def __init__(
        self, config: OmegaConf, skeletons: Optional[List[sio.Skeleton]] = None
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons)

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
        """Perform validation step."""
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


class CentroidModel(TrainingModel):
    """Lightning Module for Centroid Model.

    This is a subclass of the `TrainingModel` to configure the training/ validation steps
    and forward pass specific to centroid model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `CentroidConfmapsPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        skeletons: List of `sio.Skeleton` objects from the input `.slp` file.

    """

    def __init__(
        self, config: OmegaConf, skeletons: Optional[List[sio.Skeleton]] = None
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1)
        img = img.to(self.m_device)
        return self.model(img)["CentroidConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.m_device), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        ).to(self.m_device)

        y_preds = self.model(X)["CentroidConfmapsHead"]
        y = y.to(self.m_device)
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.m_device), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        ).to(self.m_device)

        y_preds = self.model(X)["CentroidConfmapsHead"]
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


class BottomUpModel(TrainingModel):
    """Lightning Module for BottomUp Model.

    This is a subclass of the `TrainingModel` to configure the training/ validation steps
    and forward pass specific to BottomUp model.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs to be passed to
                `BottomUpPipeline` class.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params.
        skeletons: List of `sio.Skeleton` objects from the input `.slp` file.

    """

    def __init__(
        self, config: OmegaConf, skeletons: Optional[List[sio.Skeleton]] = None
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1)
        img = img.to(self.m_device)
        output = self.model(img)
        return {
            "MultiInstanceConfmapsHead": output["MultiInstanceConfmapsHead"],
            "PartAffinityFieldsHead": output["PartAffinityFieldsHead"],
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        X = torch.squeeze(batch["image"], dim=1).to(self.m_device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.m_device)
        y_paf = batch["part_affinity_fields"].to(self.m_device)
        preds = self.model(X)
        pafs = preds["PartAffinityFieldsHead"].permute(0, 2, 3, 1)
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
        X = torch.squeeze(batch["image"], dim=1).to(self.m_device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.m_device)
        y_paf = batch["part_affinity_fields"].to(self.m_device)

        preds = self.model(X)
        pafs = preds["PartAffinityFieldsHead"].permute(0, 2, 3, 1)
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
