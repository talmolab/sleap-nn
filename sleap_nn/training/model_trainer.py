"""This module is to train a sleap-nn model using Lightning."""

import os
import shutil
import copy
import attrs
import torch
import sleap_io as sio
import time
import lightning as L
import wandb
import yaml

from pathlib import Path
from typing import List, Optional
from datetime import datetime
from itertools import cycle
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from sleap_nn.data.utils import check_cache_memory
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
from sleap_io.io.skeleton import SkeletonYAMLEncoder
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.custom_datasets import (
    get_train_val_dataloaders,
    get_steps_per_epoch,
    get_train_val_datasets,
)
from loguru import logger
from sleap_nn.config.utils import (
    get_backbone_type_from_cfg,
    get_model_type_from_cfg,
)
from sleap_nn.training.lightning_modules import LightningModel
from sleap_nn.config.utils import check_output_strides
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.callbacks import (
    ProgressReporterZMQ,
    TrainingControllerZMQ,
    MatplotlibSaver,
    WandBPredImageLogger,
    CSVLoggerCallback,
)
from sleap_nn import RANK
from sleap_nn.legacy_models import get_keras_first_layer_channels

MEMORY_BUFFER = 0.2  # Default memory buffer for caching


@attrs.define
class ModelTrainer:
    """Train sleap-nn model using PyTorch Lightning.

    This class is used to create dataloaders, train a sleap-nn model and save the model checkpoints/ logs with options to logging
    with wandb and csvlogger.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) data_config: data loading pre-processing configs.
                (ii) model_config: backbone and head configs to be passed to `Model` class.
                (iii) trainer_config: trainer configs like accelerator, optimiser params, etc.
        train_labels: List of `sio.Labels` objects for training dataset.
        val_labels: List of `sio.Labels` objects for validation dataset.
        skeletons: List of `sio.Skeleton` objects in a single slp file.
        lightning_model: One of the child classes of `sleap_nn.training.lightning_modules.LightningModel`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`, `multi_class_bottomup`, `multi_class_topdown`.
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        trainer: Instance of the `lightning.Trainer` initialized with loggers and callbacks.
    """

    config: DictConfig
    _initial_config: Optional[DictConfig] = None
    train_labels: List[sio.Labels] = []
    val_labels: List[sio.Labels] = []
    skeletons: Optional[List[sio.Skeleton]] = None

    lightning_model: Optional[LightningModel] = None
    model_type: Optional[str] = None
    backbone_type: Optional[str] = None

    _profilers: dict = {
        "advanced": AdvancedProfiler(),
        "passthrough": PassThroughProfiler(),
        "pytorch": PyTorchProfiler(),
        "simple": SimpleProfiler(),
    }

    trainer: Optional[L.Trainer] = None

    @classmethod
    def get_model_trainer_from_config(
        cls,
        config: DictConfig,
        train_labels: Optional[List[sio.Labels]] = None,
        val_labels: Optional[List[sio.Labels]] = None,
    ):
        """Create a model trainer instance from config."""
        model_trainer = cls(config=config)

        model_trainer.model_type = get_model_type_from_cfg(model_trainer.config)
        model_trainer.backbone_type = get_backbone_type_from_cfg(model_trainer.config)

        if train_labels is None and val_labels is None:
            # read labels from paths provided in the config
            train_labels = [
                sio.load_slp(path)
                for path in model_trainer.config.data_config.train_labels_path
            ]
            val_labels = (
                [
                    sio.load_slp(path)
                    for path in model_trainer.config.data_config.val_labels_path
                ]
                if model_trainer.config.data_config.val_labels_path is not None
                else None
            )
            model_trainer._setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )
        else:
            model_trainer._setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )

        model_trainer._initial_config = model_trainer.config.copy()
        # update config parameters
        model_trainer.setup_config()

        # Check if all videos exist across all labels
        all_videos_exist = all(
            video.exists(check_all=True)
            for labels in [*model_trainer.train_labels, *model_trainer.val_labels]
            for video in labels.videos
        )

        if not all_videos_exist:
            raise FileNotFoundError(
                "One or more video files do not exist or are not accessible."
            )

        return model_trainer

    def _setup_train_val_labels(
        self,
        labels: Optional[List[sio.Labels]] = None,
        val_labels: Optional[List[sio.Labels]] = None,
    ):
        """Create train and val labels objects. (Initialize `self.train_labels` and `self.val_labels`)."""
        logger.info(f"Creating train-val split...")
        total_train_lfs = 0
        total_val_lfs = 0
        self.skeletons = labels[0].skeletons

        # check if all `.slp` file shave same skeleton structure (if multiple slp file paths are provided)
        skeleton = self.skeletons[0]
        for index, train_label in enumerate(labels):
            skel_temp = train_label.skeletons[0]
            nodes_equal = [node.name for node in skeleton.nodes] == [
                node.name for node in skel_temp.nodes
            ]
            edge_inds_equal = [tuple(edge) for edge in skeleton.edge_inds] == [
                tuple(edge) for edge in skel_temp.edge_inds
            ]
            skeletons_equal = nodes_equal and edge_inds_equal
            if skeletons_equal:
                total_train_lfs += len(train_label)
            else:
                message = f"The skeletons in the training labels: {index+1} do not match the skeleton in the first training label file."
                logger.error(message)
                raise ValueError(message)

        if val_labels is None or not len(val_labels):
            # if val labels are not provided, split from train
            total_train_lfs = 0
            val_fraction = OmegaConf.select(
                self.config, "data_config.validation_fraction", default=0.1
            )
            for label in labels:
                train_split, val_split = label.make_training_splits(
                    n_train=1 - val_fraction, n_val=val_fraction, seed=42
                )
                self.train_labels.append(train_split)
                self.val_labels.append(val_split)
                total_train_lfs += len(train_split)
                total_val_lfs += len(val_split)
        else:
            self.train_labels = labels
            self.val_labels = val_labels
            for val_l in self.val_labels:
                total_val_lfs += len(val_l)

        logger.info(f"# Train Labeled frames: {total_train_lfs}")
        logger.info(f"# Val Labeled frames: {total_val_lfs}")

    def _setup_preprocessing_config(self):
        """Setup preprocessing config."""
        # compute max_heigt, max_width, and crop_hw (if not provided in the config)
        max_height = self.config.data_config.preprocessing.max_height
        max_width = self.config.data_config.preprocessing.max_width
        if self.model_type == "centered_instance":
            crop_hw = self.config.data_config.preprocessing.crop_hw

        max_h, max_w = 0, 0
        max_crop_size = 0

        for train_label in self.train_labels:
            # compute max h and w from slp file if not provided
            if max_height is None or max_width is None:
                current_max_h, current_max_w = get_max_height_width(train_label)

                if current_max_h > max_h:
                    max_h = current_max_h
                if current_max_w > max_w:
                    max_w = current_max_w

            if self.model_type == "centered_instance":
                # compute crop size if not provided in config
                if crop_hw is None:

                    crop_size = find_instance_crop_size(
                        labels=train_label,
                        maximum_stride=self.config.model_config.backbone_config[
                            f"{self.backbone_type}"
                        ]["max_stride"],
                        min_crop_size=self.config.data_config.preprocessing.min_crop_size,
                        input_scaling=self.config.data_config.preprocessing.scale,
                    )

                    if crop_size > max_crop_size:
                        max_crop_size = crop_size

        # if preprocessing params were None, replace with computed params
        if max_height is None or max_width is None:
            self.config.data_config.preprocessing.max_height = max_h
            self.config.data_config.preprocessing.max_width = max_w

        if self.model_type == "centered_instance" and crop_hw is None:
            self.config.data_config.preprocessing.crop_hw = [
                max_crop_size,
                max_crop_size,
            ]

    def _setup_head_config(self):
        """Setup node, edge and class names in head config."""
        # if edges and part names aren't set in head configs, get it from labels object.
        head_config = self.config.model_config.head_configs[self.model_type]
        for key in head_config:
            if "part_names" in head_config[key].keys():
                if head_config[key]["part_names"] is None:
                    self.config.model_config.head_configs[self.model_type][key][
                        "part_names"
                    ] = self.skeletons[0].node_names

            if "edges" in head_config[key].keys():
                if head_config[key]["edges"] is None:
                    edges = [
                        (x.source.name, x.destination.name)
                        for x in self.skeletons[0].edges
                    ]
                    self.config.model_config.head_configs[self.model_type][key][
                        "edges"
                    ] = edges

            if "classes" in head_config[key].keys():
                if head_config[key]["classes"] is None:
                    tracks = []
                    for train_label in self.train_labels:
                        tracks.extend(
                            [x.name for x in train_label.tracks if x is not None]
                        )
                    classes = list(set(tracks))
                    if not len(classes):
                        message = (
                            f"No tracks found. ID models need tracks to be defined."
                        )
                        logger.error(message)
                        raise Exception(message)
                    self.config.model_config.head_configs[self.model_type][key][
                        "classes"
                    ] = classes

    def _setup_ckpt_path(self):
        """Setup checkpoint path."""
        # if save_ckpt_path is None, assign a new dir name
        ckpt_path = self.config.trainer_config.save_ckpt_path
        if ckpt_path is None:
            trainer_devices = (
                self.config.trainer_config.trainer_devices
                if self.config.trainer_config.trainer_devices is not None
                else "auto"
            )
            if trainer_devices == "auto":
                if torch.cuda.is_available():
                    trainer_devices = torch.cuda.device_count()
                elif torch.backends.mps.is_available():
                    trainer_devices = 1
                elif torch.xpu.is_available():
                    trainer_devices = torch.xpu.device_count()
                else:
                    trainer_devices = 1
            if trainer_devices > 1:
                ckpt_path = (
                    f"{self.model_type}.n={len(self.train_labels)+len(self.val_labels)}"
                )
            else:
                ckpt_path = (
                    datetime.now().strftime("%y%m%d_%H%M%S")
                    + f".{self.model_type}.n={len(self.train_labels)+len(self.val_labels)}"
                )

        self.config.trainer_config.save_ckpt_path = ckpt_path

        # set output dir for cache img
        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk":
            if self.config.data_config.cache_img_path is None:
                self.config.data_config.cache_img_path = Path(
                    self.config.trainer_config.save_ckpt_path
                )

    def _verify_model_input_channels(self):
        """Verify input channels in model_config based on input image and pretrained model weights."""
        # check in channels, verify with img channels / ensure_rgb/ ensure_grayscale
        if self.train_labels[0] is not None:
            img_channels = self.train_labels[0][0].image.shape[-1]
            if self.config.data_config.preprocessing.ensure_rgb:
                img_channels = 3
            if self.config.data_config.preprocessing.ensure_grayscale:
                img_channels = 1
            if (
                self.config.model_config.backbone_config[
                    f"{self.backbone_type}"
                ].in_channels
                != img_channels
            ):
                self.config.model_config.backbone_config[
                    f"{self.backbone_type}"
                ].in_channels = img_channels
                logger.info(
                    f"Updating backbone in_channels to {img_channels} based on the input image channels."
                )

        # verify input img channels with pretrained model ckpts (if any)
        if (
            self.backbone_type == "convnext" or self.backbone_type == "swint"
        ) and self.config.model_config.backbone_config[
            f"{self.backbone_type}"
        ].pre_trained_weights is not None:
            if (
                self.config.model_config.backbone_config[
                    f"{self.backbone_type}"
                ].in_channels
                != 3
            ):
                self.config.model_config.backbone_config[
                    f"{self.backbone_type}"
                ].in_channels = 3
                self.config.data_config.preprocessing.ensure_rgb = True
                self.config.data_config.preprocessing.ensure_grayscale = False
                logger.info(
                    f"Updating backbone in_channels to 3 based on the pretrained model weights."
                )

        elif (
            self.backbone_type == "unet"
            and self.config.model_config.pretrained_backbone_weights is not None
        ):

            if self.config.model_config.pretrained_backbone_weights.endswith(".ckpt"):
                pretrained_backbone_ckpt = torch.load(
                    self.config.model_config.pretrained_backbone_weights,
                    map_location=(
                        self.config.trainer_config.trainer_accelerator
                        if self.config.trainer_config.trainer_accelerator is not None
                        or self.config.trainer_config.trainer_accelerator != "auto"
                        else "cpu"
                    ),
                    weights_only=False,
                )
                input_channels = list(pretrained_backbone_ckpt["state_dict"].values())[
                    0
                ].shape[
                    -3
                ]  # get input channels from first layer
                if (
                    self.config.model_config.backbone_config.unet.in_channels
                    != input_channels
                ):
                    self.config.model_config.backbone_config.unet.in_channels = (
                        input_channels
                    )
                    logger.info(
                        f"Updating backbone in_channels to {input_channels} based on the pretrained model weights."
                    )

                    if input_channels == 1:
                        self.config.data_config.preprocessing.ensure_grayscale = True
                        self.config.data_config.preprocessing.ensure_rgb = False
                        logger.info(
                            f"Updating data preprocessing to ensure_grayscale to True based on the pretrained model weights."
                        )
                    elif input_channels == 3:
                        self.config.data_config.preprocessing.ensure_rgb = True
                        self.config.data_config.preprocessing.ensure_grayscale = False
                        logger.info(
                            f"Updating data preprocessing to ensure_rgb to True based on the pretrained model weights."
                        )

            elif self.config.model_config.pretrained_backbone_weights.endswith(".h5"):
                input_channels = get_keras_first_layer_channels(
                    self.config.model_config.pretrained_backbone_weights
                )
                if (
                    self.config.model_config.backbone_config.unet.in_channels
                    != input_channels
                ):
                    self.config.model_config.backbone_config.unet.in_channels = (
                        input_channels
                    )
                    logger.info(
                        f"Updating backbone in_channels to {input_channels} based on the pretrained model weights."
                    )

                    if input_channels == 1:
                        self.config.data_config.preprocessing.ensure_grayscale = True
                        self.config.data_config.preprocessing.ensure_rgb = False
                        logger.info(
                            f"Updating data preprocessing to ensure_grayscale to True based on the pretrained model weights."
                        )
                    elif input_channels == 3:
                        self.config.data_config.preprocessing.ensure_rgb = True
                        self.config.data_config.preprocessing.ensure_grayscale = False
                        logger.info(
                            f"Updating data preprocessing to ensure_rgb to True based on the pretrained model weights."
                        )

    def setup_config(self):
        """Compute config parameters."""
        # Verify config structure.
        logger.info("Setting up config...")
        self.config = verify_training_cfg(self.config)

        # compute preprocessing parameters from the labels objects and fill in the config
        self._setup_preprocessing_config()

        # save skeleton to config
        skeleton_yaml = yaml.safe_load(SkeletonYAMLEncoder().encode(self.skeletons))
        skeleton_names = skeleton_yaml.keys()
        self.config["data_config"]["skeletons"] = []
        for skeleton_name in skeleton_names:
            skl = skeleton_yaml[skeleton_name]
            skl["name"] = skeleton_name
            self.config["data_config"]["skeletons"].append(skl)

        # setup head config - partnames, edges and class names
        self._setup_head_config()

        # set max stride for the backbone: convnext and swint
        if self.backbone_type == "convnext":
            self.config.model_config.backbone_config.convnext.max_stride = (
                self.config.model_config.backbone_config.convnext.stem_patch_stride
                * (2**3)
                * 2
            )
        elif self.backbone_type == "swint":
            self.config.model_config.backbone_config.swint.max_stride = (
                self.config.model_config.backbone_config.swint.stem_patch_stride
                * (2**3)
                * 2
            )

        # set output stride for backbone from head config and verify max stride
        self.config = check_output_strides(self.config)

        # setup checkpoint path
        self._setup_ckpt_path()

        # verify input_channels in model_config based on input image and pretrained model weights
        self._verify_model_input_channels()

    def _setup_model_ckpt_dir(self):
        """Create the model ckpt folder."""
        ckpt_path = self.config.trainer_config.save_ckpt_path
        logger.info(f"Setting up model ckpt dir: `{ckpt_path}`...")

        if not Path(ckpt_path).exists():
            try:
                Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                message = f"Cannot create a new folder in {ckpt_path}. Check the permissions to the given Checkpoint directory. \n {e}"
                logger.error(message)
                raise OSError(message)

        if RANK in [0, -1]:
            for idx, (train, val) in enumerate(zip(self.train_labels, self.val_labels)):
                train.save(
                    Path(ckpt_path) / f"labels_train_gt_{idx}.slp",
                    restore_original_videos=False,
                )
                val.save(
                    Path(ckpt_path) / f"labels_val_gt_{idx}.slp",
                    restore_original_videos=False,
                )

    def _setup_viz_datasets(self):
        """Setup dataloaders."""
        data_viz_config = self.config.copy()
        data_viz_config.data_config.data_pipeline_fw = "torch_dataset"

        return get_train_val_datasets(
            train_labels=self.train_labels,
            val_labels=self.val_labels,
            config=data_viz_config,
            rank=-1,
        )

    def _setup_datasets(self):
        """Setup dataloaders."""
        base_cache_img_path = None
        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_memory":
            # check available memory. If insufficient memory, default to disk caching.
            mem_available = check_cache_memory(
                self.train_labels, self.val_labels, memory_buffer=MEMORY_BUFFER
            )
            if not mem_available:
                self.config.data_config.data_pipeline_fw = (
                    "torch_dataset_cache_img_disk"
                )
                base_cache_img_path = Path("./")
                logger.info(
                    f"Insufficient memory for in-memory caching. `jpg` files will be created for disk-caching."
                )
            self.config.data_config.cache_img_path = base_cache_img_path

        elif self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk":
            # Get cache img path
            base_cache_img_path = (
                Path(self.config.data_config.cache_img_path)
                if self.config.data_config.cache_img_path is not None
                else Path(self.config.trainer_config.save_ckpt_path)
            )

            if self.config.data_config.cache_img_path is None:
                self.config.data_config.cache_img_path = base_cache_img_path

        return get_train_val_datasets(
            train_labels=self.train_labels,
            val_labels=self.val_labels,
            config=self.config,
            rank=self.trainer.global_rank,
        )

    def _setup_loggers_callbacks(self, viz_train_dataset, viz_val_dataset):
        """Create loggers and callbacks."""
        logger.info("Setting up callbacks and loggers...")
        loggers = []
        callbacks = []
        if self.config.trainer_config.save_ckpt:

            # checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                save_top_k=self.config.trainer_config.model_ckpt.save_top_k,
                save_last=self.config.trainer_config.model_ckpt.save_last,
                dirpath=self.config.trainer_config.save_ckpt_path,
                filename="best",
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

            # csv log callback
            csv_log_keys = [
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "train_time",
                "val_time",
            ]
            if self.model_type in [
                "single_instance",
                "centered_instance",
                "multi_class_topdown",
            ]:
                csv_log_keys.extend(self.skeletons[0].node_names)
            csv_logger = CSVLoggerCallback(
                filepath=Path(self.config.trainer_config.save_ckpt_path)
                / "training_log.csv",
                keys=csv_log_keys,
            )
            callbacks.append(csv_logger)

        if self.config.trainer_config.early_stopping.stop_training_on_plateau:
            # early stopping callback
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
            # wandb logger
            wandb_config = self.config.trainer_config.wandb
            if wandb_config.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
            else:
                if RANK in [0, -1]:
                    wandb.login(key=self.config.trainer_config.wandb.api_key)
            wandb_logger = WandbLogger(
                entity=wandb_config.entity,
                project=wandb_config.project,
                name=wandb_config.name,
                save_dir=self.config.trainer_config.save_ckpt_path,
                id=self.config.trainer_config.wandb.prv_runid,
                group=self.config.trainer_config.wandb.group,
            )
            loggers.append(wandb_logger)

            # save the configs as yaml in the checkpoint dir
            self.config.trainer_config.wandb.api_key = ""

        # zmq callbacks
        controller_address = OmegaConf.select(
            self.config, "trainer_config.zmq.controller_address", default=None
        )
        publish_address = OmegaConf.select(
            self.config, "trainer_config.zmq.publish_address", default=None
        )
        if controller_address is not None:
            callbacks.append(TrainingControllerZMQ(address=controller_address))
        if publish_address is not None:
            callbacks.append(ProgressReporterZMQ(address=publish_address))

        # viz callbacks
        if self.config.trainer_config.visualize_preds_during_training:
            train_viz_pipeline = cycle(viz_train_dataset)
            val_viz_pipeline = cycle(viz_val_dataset)

            viz_dir = Path(self.config.trainer_config.save_ckpt_path) / "viz"
            if not Path(viz_dir).exists():
                if RANK in [0, -1]:
                    Path(viz_dir).mkdir(parents=True, exist_ok=True)

            callbacks.append(
                MatplotlibSaver(
                    save_folder=viz_dir,
                    plot_fn=lambda: self.lightning_model.visualize_example(
                        next(train_viz_pipeline)
                    ),
                    prefix="train",
                )
            )
            callbacks.append(
                MatplotlibSaver(
                    save_folder=viz_dir,
                    plot_fn=lambda: self.lightning_model.visualize_example(
                        next(val_viz_pipeline)
                    ),
                    prefix="validation",
                )
            )

            if self.model_type == "bottomup":
                train_viz_pipeline1 = cycle(copy.deepcopy(viz_train_dataset))
                val_viz_pipeline1 = cycle(copy.deepcopy(viz_val_dataset))
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.lightning_model.visualize_pafs_example(
                            next(train_viz_pipeline1)
                        ),
                        prefix="train.pafs_magnitude",
                    )
                )
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.lightning_model.visualize_pafs_example(
                            next(val_viz_pipeline1)
                        ),
                        prefix="validation.pafs_magnitude",
                    )
                )

            if self.model_type == "multi_class_bottomup":
                train_viz_pipeline1 = cycle(copy.deepcopy(viz_train_dataset))
                val_viz_pipeline1 = cycle(copy.deepcopy(viz_val_dataset))
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.lightning_model.visualize_class_maps_example(
                            next(train_viz_pipeline1)
                        ),
                        prefix="train.class_maps",
                    )
                )
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.lightning_model.visualize_class_maps_example(
                            next(val_viz_pipeline1)
                        ),
                        prefix="validation.class_maps",
                    )
                )

            if self.config.trainer_config.use_wandb:
                callbacks.append(
                    WandBPredImageLogger(
                        viz_folder=viz_dir,
                        wandb_run_name=self.config.trainer_config.wandb.name,
                        is_bottomup=(self.model_type == "bottomup"),
                    )
                )

        return loggers, callbacks

    def _delete_cache_imgs(self):
        """Delete cache images in disk."""
        base_cache_img_path = Path(self.config.data_config.cache_img_path)
        train_cache_img_path = Path(base_cache_img_path) / "train_imgs"
        val_cache_img_path = Path(base_cache_img_path) / "val_imgs"

        if (train_cache_img_path).exists():
            logger.info(f"Deleting cache imgs from `{train_cache_img_path}`...")
            shutil.rmtree(
                (train_cache_img_path).as_posix(),
                ignore_errors=True,
            )

        if (val_cache_img_path).exists():
            logger.info(f"Deleting cache imgs from `{val_cache_img_path}`...")
            shutil.rmtree(
                (val_cache_img_path).as_posix(),
                ignore_errors=True,
            )

    def train(self):
        """Train the lightning model."""
        logger.info(f"Setting up for training...")
        start_setup_time = time.time()

        # initialize the labels object and update config.
        if not len(self.train_labels) or not len(self.val_labels):
            self._setup_train_val_labels(self.config)
            self.setup_config()

        # create the ckpt dir.
        self._setup_model_ckpt_dir()

        # create the train and val datasets for visualization.
        viz_train_dataset = None
        viz_val_dataset = None
        if self.config.trainer_config.visualize_preds_during_training:
            logger.info(f"Setting up visualization train and val datasets...")
            viz_train_dataset, viz_val_dataset = self._setup_viz_datasets()

        # setup loggers and callbacks for Trainer.
        logger.info(f"Setting up Trainer...")
        loggers, callbacks = self._setup_loggers_callbacks(
            viz_train_dataset=viz_train_dataset, viz_val_dataset=viz_val_dataset
        )
        # set up the strategy (for multi-gpu training)
        strategy = OmegaConf.select(
            self.config, "trainer_config.trainer_strategy", default="auto"
        )
        # set up profilers
        cfg_profiler = self.config.trainer_config.profiler
        profiler = None
        if cfg_profiler is not None:
            if cfg_profiler in self._profilers:
                profiler = self._profilers[cfg_profiler]
            else:
                message = f"{cfg_profiler} is not a valid option. Please choose one of {list(self._profilers.keys())}"
                logger.error(message)
                raise ValueError(message)

        # create lightning.Trainer instance.
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            strategy=strategy,
            profiler=profiler,
            log_every_n_steps=1,
        )

        self.trainer.strategy.barrier()

        # setup datasets
        train_dataset, val_dataset = self._setup_datasets()

        # set-up steps per epoch
        train_steps_per_epoch = self.config.trainer_config.train_steps_per_epoch
        if train_steps_per_epoch is None:
            train_steps_per_epoch = get_steps_per_epoch(
                dataset=train_dataset,
                batch_size=self.config.trainer_config.train_data_loader.batch_size,
            )
        if self.config.trainer_config.min_train_steps_per_epoch > train_steps_per_epoch:
            train_steps_per_epoch = self.config.trainer_config.min_train_steps_per_epoch
        self.config.trainer_config.train_steps_per_epoch = train_steps_per_epoch

        val_steps_per_epoch = get_steps_per_epoch(
            dataset=val_dataset,
            batch_size=self.config.trainer_config.val_data_loader.batch_size,
        )

        # set devices and accelrator
        if (
            self.config.trainer_config.trainer_devices is None
            or self.config.trainer_config.trainer_devices == "auto"
        ):
            self.config.trainer_config.trainer_devices = self.trainer.num_devices
        if (
            self.config.trainer_config.trainer_accelerator is None
            or self.config.trainer_config.trainer_accelerator == "auto"
        ):
            self.config.trainer_config.trainer_accelerator = (
                self.trainer.strategy.root_device
            )

        # initialize the lightning model.
        # need to initialize after Trainer is initialized (for trainer accelerator)
        logger.info(f"Setting up lightning module for {self.model_type} model...")
        self.lightning_model = LightningModel.get_lightning_model_from_config(
            config=self.config
        )
        total_params = sum(p.numel() for p in self.lightning_model.parameters())
        self.config.model_config.total_params = total_params

        # setup dataloaders
        # need to set up dataloaders after Trainer is initialized (for ddp). DistributedSampler depends on the rank
        train_dataloader, val_dataloader = get_train_val_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            rank=self.trainer.global_rank,
            train_steps_per_epoch=self.config.trainer_config.train_steps_per_epoch,
            val_steps_per_epoch=val_steps_per_epoch,
            trainer_devices=self.config.trainer_config.trainer_devices,
        )

        if self.trainer.global_rank == 0:  # save config only in rank 0 process
            ckpt_path = self.config.trainer_config.save_ckpt_path
            OmegaConf.save(
                self._initial_config,
                (Path(ckpt_path) / "initial_config.yaml").as_posix(),
            )

            if self.config.trainer_config.use_wandb:
                if wandb.run is None:
                    wandb.init(
                        dir=self.config.trainer_config.save_ckpt_path,
                        project=self.config.trainer_config.wandb.project,
                        entity=self.config.trainer_config.wandb.entity,
                        name=self.config.trainer_config.wandb.name,
                        id=self.config.trainer_config.wandb.prv_runid,
                        group=self.config.trainer_config.wandb.group,
                    )
                self.config.trainer_config.wandb.current_run_id = wandb.run.id
                wandb.config["run_name"] = self.config.trainer_config.wandb.name
                wandb.config["run_config"] = OmegaConf.to_container(
                    self.config, resolve=True
                )

            OmegaConf.save(
                self.config,
                (
                    Path(self.config.trainer_config.save_ckpt_path)
                    / "training_config.yaml"
                ).as_posix(),
            )

        self.trainer.strategy.barrier()

        try:
            logger.info(
                f"Finished trainer set up. [{time.time() - start_setup_time:.1f}s]"
            )
            logger.info(f"Starting training loop...")
            start_train_time = time.time()
            self.trainer.fit(
                self.lightning_model,
                train_dataloader,
                val_dataloader,
                ckpt_path=self.config.trainer_config.resume_ckpt_path,
            )

        except KeyboardInterrupt:
            logger.info("Stopping training...")

        finally:
            logger.info(
                f"Finished training loop. [{(time.time() - start_train_time) / 60:.1f} min]"
            )
            if self.trainer.global_rank == 0 and self.config.trainer_config.use_wandb:
                wandb.finish()

            # delete image disk caching
            if (
                self.config.data_config.data_pipeline_fw
                == "torch_dataset_cache_img_disk"
                and self.config.data_config.delete_cache_imgs_after_training
            ):
                if self.trainer.global_rank == 0:
                    self._delete_cache_imgs()

            # delete viz folder if requested
            if (
                self.config.trainer_config.visualize_preds_during_training
                and not self.config.trainer_config.keep_viz
            ):
                if self.trainer.global_rank == 0:
                    viz_dir = Path(self.config.trainer_config.save_ckpt_path) / "viz"
                    if viz_dir.exists():
                        logger.info(f"Deleting viz folder at {viz_dir}...")
                        shutil.rmtree(viz_dir, ignore_errors=True)
