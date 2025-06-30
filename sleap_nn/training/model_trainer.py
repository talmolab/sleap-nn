"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
import os
import shutil
import copy
import attrs
import sleap_io as sio
from itertools import cycle
from omegaconf import DictConfig, OmegaConf
import lightning as L
import wandb
from typing import List, Optional
import time
from datetime import datetime
from lightning.pytorch.loggers import WandbLogger
from sleap_nn.data.utils import check_cache_memory
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
import sleap_io as sio
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.custom_datasets import get_train_val_dataloaders
from loguru import logger
from sleap_nn.config.utils import (
    get_backbone_type_from_cfg,
    get_model_type_from_cfg,
)
from sleap_nn.training.lightning_modules import LightningModel
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.callbacks import (
    ProgressReporterZMQ,
    TrainingControllerZMQ,
    MatplotlibSaver,
    WandBPredImageLogger,
    CSVLoggerCallback,
)


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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
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
            print(f"train labels: {train_labels}-----------------")
            val_labels = (
                [
                    sio.load_slp(path)
                    for path in model_trainer.config.data_config.val_labels_path
                ]
                if model_trainer.config.data_config.val_labels_path is not None
                else None
            )
            model_trainer.setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )
        else:
            model_trainer.setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )

        model_trainer._initial_config = model_trainer.config.copy()
        # update config parameters
        model_trainer._setup_config()

        return model_trainer

    def setup_train_val_labels(
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
                    n_train=1 - val_fraction, n_val=val_fraction
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

    def _setup_config(self):
        """Compute preprocessing parameters."""
        # Verify config structure.
        logger.info("Setting up config...")
        self.config = verify_training_cfg(self.config)

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

        # save skeleton to config
        self.config["data_config"]["skeletons"] = {}
        for skl in self.skeletons:
            if skl.symmetries:
                symm = [list(s.nodes) for s in skl.symmetries]
            else:
                symm = None
            skl_name = skl.name if skl.name is not None else "skeleton-0"
            self.config["data_config"]["skeletons"] = {
                skl_name: {
                    "nodes": skl.nodes,
                    "edges": skl.edges,
                    "symmetries": symm,
                }
            }

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

        # if save_ckpt_path is None, assign a new dir name
        ckpt_path = self.config.trainer_config.save_ckpt_path
        if ckpt_path is None:
            ckpt_path = datetime.now().strftime("%y%m%d_%H%M%S") + f".{self.model_type}"

        self.config.trainer_config.save_ckpt_path = ckpt_path

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

        OmegaConf.save(
            self._initial_config, (Path(ckpt_path) / "initial_config.yaml").as_posix()
        )
        for idx, (train, val) in enumerate(zip(self.train_labels, self.val_labels)):
            train.save(Path(ckpt_path) / f"labels_train_gt_{idx}.slp")
            val.save(Path(ckpt_path) / f"labels_val_gt_{idx}.slp")

    def _setup_dataloaders(self):
        """Setup dataloaders."""
        base_cache_img_path = None
        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_memory":
            # check available memory. If insufficient memory, default to disk caching.
            mem_available = check_cache_memory(
                self.train_labels, self.val_labels, self.config
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

        return get_train_val_dataloaders(
            train_labels=self.train_labels,
            val_labels=self.val_labels,
            config=self.config,
            rank=self.trainer.global_rank if self.trainer is not None else None,
        )

    def _setup_loggers_callbacks(self, train_dataset, val_dataset):
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
            if self.model_type in ["single_instance", "centered_instance"]:
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
            train_viz_pipeline = cycle(train_dataset)
            val_viz_pipeline = cycle(val_dataset)

            viz_dir = Path(self.config.trainer_config.save_ckpt_path) / "viz"
            if not Path(viz_dir).exists():
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
                train_viz_pipeline1 = cycle(copy.deepcopy(train_dataset))
                val_viz_pipeline1 = cycle(copy.deepcopy(val_dataset))
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
            self.setup_train_val_labels(self.config)
            self._setup_config()

        # create the ckpt dir.
        self._setup_model_ckpt_dir()

        # initialize the lightning model.
        logger.info(f"Setting up lightning module for {self.model_type} model...")
        self.lightning_model = LightningModel.get_lightning_model_from_config(
            config=self.config
        )
        total_params = sum(p.numel() for p in self.lightning_model.parameters())
        self.config.model_config.total_params = total_params

        # create the train and val dataloaders.
        logger.info(f"Setting up train and val data loaders...")
        train_dataloader, val_dataloader = self._setup_dataloaders()

        # setup loggers and callbacks for Trainer.
        logger.info(f"Setting up Trainer...")
        loggers, callbacks = self._setup_loggers_callbacks(
            train_dataloader.dataset, val_dataloader.dataset
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

        # create lightning.Trainer insatnce.
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            limit_train_batches=self.config.trainer_config.min_train_steps_per_epoch,  # TODO
            strategy=strategy,
            profiler=profiler,
            log_every_n_steps=1,
        )  # TODO check any other methods to use rank in dataset creations!

        if (
            self.trainer.global_rank == 0
        ):  # save config if there are no distributed process

            if self.config.trainer_config.use_wandb:
                wandb.init()
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
            if self.config.trainer_config.use_wandb:
                wandb.finish()

            # delete image disk caching
            if (
                self.config.data_config.data_pipeline_fw
                == "torch_dataset_cache_img_disk"
                and self.config.data_config.delete_cache_imgs_after_training
            ):
                self._delete_cache_imgs()
