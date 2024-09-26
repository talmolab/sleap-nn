"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
from typing import Optional, List
import functools
import time
from torch import nn
import os
import shutil
import torch
import sleap_io as sio
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import lightning as L
import litdata as ld
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
    CentroidConfmapsPipeline,
    BottomUpPipeline,
)
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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
from sleap_nn.architectures.model import Model
from sleap_nn.data.cycler import CyclerIterDataPipe as Cycler
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_instances
from sleap_nn.data.get_data_chunks import (
    bottomup_data_chunks,
    centered_instance_data_chunks,
    centroid_data_chunks,
    single_instance_data_chunks,
)
from sleap_nn.data.streaming_datasets import (
    BottomUpStreamingDataset,
    CenteredInstanceStreamingDataset,
    CentroidStreamingDataset,
    SingleInstanceStreamingDataset,
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

        self.seed = self.config.trainer_config.seed
        self.steps_per_epoch = self.config.trainer_config.steps_per_epoch

        # initialize attributes
        self.model = None
        self.provider = None
        self.skeletons = None
        self.train_data_loader = None
        self.val_data_loader = None

        # check which head type to choose the model
        for k, v in self.config.model_config.head_configs.items():
            if v is not None:
                self.model_type = k
                break

        if not self.config.trainer_config.save_ckpt_path:
            self.dir_path = "."
        else:
            self.dir_path = self.config.trainer_config.save_ckpt_path

        if not Path(self.dir_path).exists():
            try:
                Path(self.dir_path).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(
                    f"Cannot create a new folder. Check the permissions to the given Checkpoint directory. \n {e}"
                )

        # set seed
        torch.manual_seed(self.seed)

    def _get_data_chunks(self, func, train_labels, val_labels):
        """Create a new folder with pre-processed data stored as `.bin` files."""
        ld.optimize(
            fn=func,
            inputs=[(x, train_labels.videos.index(x.video)) for x in train_labels],
            output_dir=(Path(self.dir_path) / "train_chunks").as_posix(),
            num_workers=self.config.trainer_config.train_data_loader.num_workers,
            chunk_size=(
                self.config.data_config.chunk_size
                if "chunk_size" in self.config.data_config
                and self.config.data_config.chunk_size is not None
                else 100
            ),
        )

        ld.optimize(
            fn=func,
            inputs=[(x, val_labels.videos.index(x.video)) for x in val_labels],
            output_dir=(Path(self.dir_path) / "val_chunks").as_posix(),
            num_workers=self.config.trainer_config.train_data_loader.num_workers,
            chunk_size=(
                self.config.data_config.chunk_size
                if "chunk_size" in self.config.data_config
                and self.config.data_config.chunk_size is not None
                else 100
            ),
        )

    def _create_data_loaders(self):
        """Create a DataLoader for train, validation and test sets using the data_config."""
        self.provider = self.config.data_config.provider
        if self.provider == "LabelsReader":
            self.provider = LabelsReader

        train_labels = sio.load_slp(self.config.data_config.train_labels_path)
        val_labels = sio.load_slp(self.config.data_config.val_labels_path)
        user_instances_only = (
            self.config.data_config.user_instances_only
            if "user_instances_only" in self.config.data_config
            and self.config.data_config.user_instances_only is not None
            else True
        )
        self.skeletons = train_labels.skeletons
        max_stride = self.config.model_config.backbone_config.max_stride
        max_instances = get_max_instances(train_labels)
        edge_inds = train_labels.skeletons[0].edge_inds

        if self.model_type == "single_instance":

            factory_get_chunks = functools.partial(
                single_instance_data_chunks,
                data_config=self.config.data_config,
                user_instances_only=user_instances_only,
            )

            self._get_data_chunks(
                func=factory_get_chunks,
                train_labels=train_labels,
                val_labels=val_labels,
            )

            train_dataset = SingleInstanceStreamingDataset(
                input_dir=(Path(self.dir_path) / "train_chunks").as_posix(),
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = SingleInstanceStreamingDataset(
                input_dir=(Path(self.dir_path) / "val_chunks").as_posix(),
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

        elif self.model_type == "centered_instance":
            # compute crop size
            crop_hw = self.config.data_config.preprocessing.crop_hw
            if crop_hw is None:

                min_crop_size = (
                    self.config.data_config.preprocessing.min_crop_size
                    if "min_crop_size" in self.config.data_config.preprocessing
                    else None
                )
                crop_size = find_instance_crop_size(
                    train_labels,
                    maximum_stride=max_stride,
                    input_scaling=self.config.data_config.preprocessing.scale,
                    min_crop_size=min_crop_size,
                )
                crop_hw = (crop_size, crop_size)
            self.config.data_config.preprocessing.crop_hw = crop_hw

            factory_get_chunks = functools.partial(
                centered_instance_data_chunks,
                data_config=self.config.data_config,
                max_instances=max_instances,
                crop_size=crop_hw,
                anchor_ind=self.config.model_config.head_configs.centered_instance.confmaps.anchor_part,
                user_instances_only=user_instances_only,
            )

            self._get_data_chunks(
                func=factory_get_chunks,
                train_labels=train_labels,
                val_labels=val_labels,
            )

            train_dataset = CenteredInstanceStreamingDataset(
                input_dir=(Path(self.dir_path) / "train_chunks").as_posix(),
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=max_stride,
                crop_hw=crop_hw,
                scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = CenteredInstanceStreamingDataset(
                input_dir=(Path(self.dir_path) / "val_chunks").as_posix(),
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=max_stride,
                crop_hw=crop_hw,
                scale=self.config.data_config.preprocessing.scale,
            )

        elif self.model_type == "centroid":
            factory_get_chunks = functools.partial(
                centroid_data_chunks,
                data_config=self.config.data_config,
                max_instances=max_instances,
                anchor_ind=self.config.model_config.head_configs.centroid.confmaps.anchor_part,
                user_instances_only=user_instances_only,
            )

            self._get_data_chunks(
                func=factory_get_chunks,
                train_labels=train_labels,
                val_labels=val_labels,
            )

            train_dataset = CentroidStreamingDataset(
                input_dir=(Path(self.dir_path) / "train_chunks").as_posix(),
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = CentroidStreamingDataset(
                input_dir=(Path(self.dir_path) / "val_chunks").as_posix(),
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

        elif self.model_type == "bottomup":
            factory_get_chunks = functools.partial(
                bottomup_data_chunks,
                data_config=self.config.data_config,
                max_instances=max_instances,
                user_instances_only=user_instances_only,
            )

            self._get_data_chunks(
                func=factory_get_chunks,
                train_labels=train_labels,
                val_labels=val_labels,
            )

            train_dataset = BottomUpStreamingDataset(
                input_dir=(Path(self.dir_path) / "train_chunks").as_posix(),
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head=self.config.model_config.head_configs.bottomup.pafs,
                edge_inds=edge_inds,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = BottomUpStreamingDataset(
                input_dir=(Path(self.dir_path) / "val_chunks").as_posix(),
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head=self.config.model_config.head_configs.bottomup.pafs,
                edge_inds=edge_inds,
                max_stride=max_stride,
                scale=self.config.data_config.preprocessing.scale,
            )

        else:
            raise Exception(
                f"{self.model_type} is not defined. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`."
            )

        # train
        # TODO: cycler - to ensure minimum steps per epoch
        self.train_data_loader = ld.StreamingDataLoader(
            train_dataset,
            batch_size=self.config.trainer_config.train_data_loader.batch_size,
            num_workers=self.config.trainer_config.train_data_loader.num_workers,
        )

        # val
        self.val_data_loader = ld.StreamingDataLoader(
            val_dataset,
            batch_size=self.config.trainer_config.val_data_loader.batch_size,
            num_workers=self.config.trainer_config.val_data_loader.num_workers,
        )

    def _set_wandb(self):
        wandb.login(key=self.config.trainer_config.wandb.api_key)

    def _initialize_model(self):
        models = {
            "single_instance": SingleInstanceModel,
            "centered_instance": TopDownCenteredInstanceModel,
            "centroid": CentroidModel,
            "bottomup": BottomUpModel,
        }
        self.model = models[self.model_type](
            self.config, self.skeletons, self.model_type
        )

    def _get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        """Initiate the training by calling the fit method of Trainer."""
        self._create_data_loaders()
        logger = []

        if self.config.trainer_config.save_ckpt:

            # create checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                save_top_k=self.config.trainer_config.model_ckpt.save_top_k,
                save_last=self.config.trainer_config.model_ckpt.save_last,
                dirpath=self.dir_path,
                filename="best",
                monitor="val_loss",
                mode="min",
            )
            callbacks = [checkpoint_callback]
            # logger to create csv with metrics values over the epochs
            csv_logger = CSVLogger(self.dir_path)
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
                project=wandb_config.project,
                name=wandb_config.name,
                save_dir=self.dir_path,
                id=self.config.trainer_config.wandb.prv_runid,
            )
            logger.append(wandb_logger)

            # save the configs as yaml in the checkpoint dir
            self.config.trainer_config.wandb.api_key = ""

        OmegaConf.save(config=self.config, f=f"{self.dir_path}/initial_config.yaml")

        # save the skeleton in the config
        self.config["data_config"]["skeletons"] = {}
        for skl in self.skeletons:
            if skl.symmetries:
                symm = [list(s.nodes) for s in skl.symmetries]
            else:
                symm = None
            skl_name = skl.name if skl.name is not None else "skeleton-0"
            self.config["data_config"]["skeletons"][skl_name] = {
                "nodes": skl.nodes,
                "edges": skl.edges,
                "symmetries": symm,
            }

        self._initialize_model()
        total_params = self._get_param_count()

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

        try:
            trainer.fit(
                self.model,
                self.train_data_loader,
                self.val_data_loader,
                ckpt_path=self.config.trainer_config.resume_ckpt_path,
            )

            if self.config.trainer_config.use_wandb:
                for m in self.config.trainer_config.wandb.log_params:
                    list_keys = m.split(".")
                    key = list_keys[-1]
                    value = self.config[list_keys[0]]
                    for l in list_keys[1:]:
                        value = value[l]
                    wandb_logger.experiment.config.update({key: value})
                wandb_logger.experiment.config.update({"model_params": total_params})

        except KeyboardInterrupt:
            print("Stopping training...")

        finally:
            if self.config.trainer_config.use_wandb:
                self.config.trainer_config.wandb.run_id = wandb.run.id
                wandb.finish()
            self.config.model_config.total_params = total_params
            # save the configs as yaml in the checkpoint dir
            OmegaConf.save(
                config=self.config, f=f"{self.dir_path}/training_config.yaml"
            )

            # shutil.rmtree((Path(self.dir_path) / "train_chunks").as_posix())
            # shutil.rmtree((Path(self.dir_path) / "val_chunks").as_posix())


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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.skeletons = skeletons
        self.model_config = self.config.model_config
        self.trainer_config = self.config.trainer_config
        self.data_config = self.config.data_config
        self.model_type = model_type
        self.input_expand_channels = self.model_config.backbone_config.in_channels
        if self.model_config.pre_trained_weights:
            ckpt = eval(self.model_config.pre_trained_weights).DEFAULT.get_state_dict(
                progress=True, check_hash=True
            )
            input_channels = ckpt["features.0.0.weight"].shape[-3]
            if self.model_config.backbone_config.in_channels != input_channels:
                self.input_expand_channels = input_channels
                OmegaConf.update(
                    self.model_config,
                    "backbone_config.in_channels",
                    input_channels,
                )

        # if edges and part names aren't set in config, get it from `sio.Labels` object.
        head_config = self.model_config.head_configs[self.model_type]
        for key in head_config:
            if "part_names" in head_config[key].keys():
                if head_config[key]["part_names"] is None:
                    part_names = [x.name for x in self.skeletons[0].nodes]
                    head_config[key]["part_names"] = part_names

            if "edges" in head_config[key].keys():
                if head_config[key]["edges"] is None:
                    edges = [
                        (x.source.name, x.destination.name)
                        for x in self.skeletons[0].edges
                    ]
                    head_config[key]["edges"] = edges

        self.model = Model(
            backbone_type=self.model_config.backbone_type,
            backbone_config=self.model_config.backbone_config,
            head_configs=head_config,
            input_expand_channels=self.input_expand_channels,
            model_type=self.model_type,
        )

        if len(self.model_config.head_configs[self.model_type]) > 1:
            self.loss_weights = [
                self.model_config.head_configs[self.model_type][x].loss_weight
                for x in self.model_config.head_configs[self.model_type]
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold=self.trainer_config.lr_scheduler.threshold,
            threshold_mode="rel",
            cooldown=self.trainer_config.lr_scheduler.cooldown,
            patience=self.trainer_config.lr_scheduler.patience,
            factor=self.trainer_config.lr_scheduler.factor,
            min_lr=self.trainer_config.lr_scheduler.min_lr,
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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons, model_type)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["SingleInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.device), torch.squeeze(
            batch["confidence_maps"], dim=1
        ).to(self.device)

        y_preds = self.model(X)["SingleInstanceConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.device), torch.squeeze(
            batch["confidence_maps"], dim=1
        ).to(self.device)

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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons, model_type)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CenteredInstanceConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1).to(
            self.device
        ), torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)

        y_preds = self.model(X)["CenteredInstanceConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        X, y = torch.squeeze(batch["instance_image"], dim=1).to(
            self.device
        ), torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)

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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons, model_type)

    def forward(self, img):
        """Forward pass of the model."""
        img = torch.squeeze(img, dim=1).to(self.device)
        return self.model(img)["CentroidConfmapsHead"]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.device), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        ).to(self.device)

        y_preds = self.model(X)["CentroidConfmapsHead"]
        loss = nn.MSELoss()(y_preds, y)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, y = torch.squeeze(batch["image"], dim=1).to(self.device), torch.squeeze(
            batch["centroids_confidence_maps"], dim=1
        ).to(self.device)

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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(config, skeletons, model_type)

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
        X = torch.squeeze(batch["image"], dim=1).to(self.device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)
        y_paf = batch["part_affinity_fields"].to(self.device)
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
        X = torch.squeeze(batch["image"], dim=1).to(self.device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)
        y_paf = batch["part_affinity_fields"].to(self.device)

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
