"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import time
from torch import nn
import os
import psutil
import shutil
import subprocess
import torch
import sleap_io as sio
from omegaconf import OmegaConf
import lightning as L
import litdata as ld
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
from sleap_nn.config.data_config import DataConfig, PreprocessingConfig
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.trainer_config import (
    TrainerConfig,
    DataLoaderConfig,
    ModelCkptConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    WandBConfig,
    EarlyStoppingConfig,
    StepLRConfig,
    ReduceLROnPlateauConfig,
)
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    CenteredInstanceDataset,
    CentroidDataset,
    SingleInstanceDataset,
    CyclerDataLoader,
)
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.streaming_datasets import (
    BottomUpStreamingDataset,
    CenteredInstanceStreamingDataset,
    CentroidStreamingDataset,
    SingleInstanceStreamingDataset,
)
from sleap_nn.training.utils import (
    check_memory,
    get_aug_config,
    xavier_init_weights,
    get_backbone_config,
    get_head_configs,
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

    def __init__(
        self,
        config: OmegaConf,
    ):
        """Initialise the class with configs and set the seed and device as class attributes."""
        self.config = config
        self.data_pipeline_fw = self.config.data_config.data_pipeline_fw
        self.use_existing_chunks = self.config.data_config.use_existing_chunks

        # Get ckpt dir path
        self.dir_path = self.config.trainer_config.save_ckpt_path
        if self.dir_path is None:
            self.dir_path = "."

        if not Path(self.dir_path).exists():
            try:
                Path(self.dir_path).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(
                    f"Cannot create a new folder in {self.dir_path}. Check the permissions to the given Checkpoint directory. \n {e}"
                )

        if self.data_pipeline_fw == "litdata":
            # Get litdata chunks path
            self.litdata_chunks_path = (
                Path(self.config.data_config.litdata_chunks_path)
                if self.config.data_config.litdata_chunks_path is not None
                else Path(self.dir_path)
            )

            if not Path(self.litdata_chunks_path).exists():
                Path(self.litdata_chunks_path).mkdir(parents=True, exist_ok=True)

            self.train_litdata_chunks_path = (
                Path(self.litdata_chunks_path) / "train_chunks"
            ).as_posix()
            self.val_litdata_chunks_path = (
                Path(self.litdata_chunks_path) / "val_chunks"
            ).as_posix()

        elif (
            self.data_pipeline_fw == "torch_dataset"
            or self.data_pipeline_fw == "torch_dataset_np_chunks"
        ):
            self.train_dataset = None
            self.val_dataset = None
            # Get np chunks path
            self.np_chunks = True if "np_chunks" in self.data_pipeline_fw else False
            self.train_np_chunks_path = (
                Path(self.config.data_config.np_chunks_path) / "train_chunks"
                if self.config.data_config.np_chunks_path is not None
                else Path(self.dir_path) / "train_chunks"
            )
            self.val_np_chunks_path = (
                Path(self.config.data_config.np_chunks_path) / "val_chunks"
                if self.config.data_config.np_chunks_path is not None
                else Path(self.dir_path) / "val_chunks"
            )
            if self.use_existing_chunks:
                if not (
                    self.train_np_chunks_path.exists()
                    and self.train_np_chunks_path.is_dir()
                    and any(self.train_np_chunks_path.glob("*.npz"))
                ):
                    raise Exception(
                        f"There are no numpy chunks in the path: {self.train_np_chunks_path}"
                    )
                if not (
                    self.val_np_chunks_path.exists()
                    and self.val_np_chunks_path.is_dir()
                    and any(self.val_np_chunks_path.glob("*.npz"))
                ):
                    raise Exception(
                        f"There are no numpy chunks in the path: {self.val_np_chunks_path}"
                    )

        self.seed = self.config.trainer_config.seed
        self.steps_per_epoch = self.config.trainer_config.steps_per_epoch

        # initialize attributes
        self.model = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.trainer = None
        self.crop_hw = -1

        # check which backbone architecture
        for k, v in self.config.model_config.backbone_config.items():
            if v is not None:
                self.backbone_type = k
                break

        # check which head type to choose the model
        for k, v in self.config.model_config.head_configs.items():
            if v is not None:
                self.model_type = k
                break

        OmegaConf.save(config=self.config, f=f"{self.dir_path}/initial_config.yaml")

        # set seed
        torch.manual_seed(self.seed)

        train_labels = sio.load_slp(self.config.data_config.train_labels_path)
        self.user_instances_only = (
            self.config.data_config.user_instances_only
            if "user_instances_only" in self.config.data_config
            and self.config.data_config.user_instances_only is not None
            else True
        )  # TODO: defaults should be handles in config validation.
        self.skeletons = train_labels.skeletons
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

        self.max_stride = self.config.model_config.backbone_config[
            f"{self.backbone_type}"
        ]["max_stride"]
        self.edge_inds = train_labels.skeletons[0].edge_inds

        self.max_height, self.max_width = get_max_height_width(train_labels)
        if (
            self.config.data_config.preprocessing.max_height is None
            and self.config.data_config.preprocessing.max_width is None
        ):
            self.config.data_config.preprocessing.max_height = self.max_height
            self.config.data_config.preprocessing.max_width = self.max_width

        if self.model_type == "centered_instance":
            # compute crop size
            self.crop_hw = self.config.data_config.preprocessing.crop_hw
            if self.crop_hw is None:

                min_crop_size = (
                    self.config.data_config.preprocessing.min_crop_size
                    if "min_crop_size" in self.config.data_config.preprocessing
                    else None
                )
                crop_size = find_instance_crop_size(
                    train_labels,
                    maximum_stride=self.max_stride,
                    min_crop_size=min_crop_size,
                    input_scaling=self.config.data_config.preprocessing.scale,
                )
                self.crop_hw = crop_size
                self.config.data_config.preprocessing.crop_hw = (
                    self.crop_hw,
                    self.crop_hw,
                )
            else:
                self.crop_hw = self.crop_hw[0]

        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

    def _create_data_loaders_torch_dataset(self):
        """Create a torch DataLoader for train, validation and test sets using the data_config."""
        train_labels = sio.load_slp(self.config.data_config.train_labels_path)
        val_labels = sio.load_slp(self.config.data_config.val_labels_path)
        if self.data_pipeline_fw == "torch_dataset":
            train_cache_memory = check_memory(
                train_labels,
                max_hw=(self.max_height, self.max_width),
                model_type=self.model_type,
                input_scaling=self.config.data_config.preprocessing.scale,
                crop_size=self.crop_hw if self.crop_hw != -1 else None,
            )
            val_cache_memory = check_memory(
                val_labels,
                max_hw=(self.max_height, self.max_width),
                model_type=self.model_type,
                input_scaling=self.config.data_config.preprocessing.scale,
                crop_size=self.crop_hw if self.crop_hw != -1 else None,
            )
            total_cache_memory = train_cache_memory + val_cache_memory
            total_cache_memory += 0.1 * total_cache_memory  # memory required in bytes
            available_memory = (
                psutil.virtual_memory().available
            )  # available memory in bytes

            if total_cache_memory > available_memory:
                self.data_pipeline_fw = "torch_dataset_np_chunks"
                self.np_chunks = True
                self.train_np_chunks_path = Path("./train_chunks")
                self.val_np_chunks_path = Path("./val_chunks")
                print(
                    f"Insufficient memory for in-memory caching. `npz` files will be created."
                )

        if self.model_type == "bottomup":
            self.train_dataset = BottomUpDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )
            self.val_dataset = BottomUpDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )

        elif self.model_type == "centered_instance":
            self.train_dataset = CenteredInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )
            self.val_dataset = CenteredInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                apply_aug=False,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )

        elif self.model_type == "centroid":
            self.train_dataset = CentroidDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )
            self.val_dataset = CentroidDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )

        elif self.model_type == "single_instance":
            self.train_dataset = SingleInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )
            self.val_dataset = SingleInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )

        else:
            raise ValueError(
                f"Model type: {self.model_type}. Ensure the heads config has one of the keys: [`bottomup`, `centroid`, `centered_instance`, `single_instance`]."
            )

        if self.steps_per_epoch is None:
            self.steps_per_epoch = (
                len(self.train_dataset)
                // self.config.trainer_config.train_data_loader.batch_size
            )

        pin_memory = (
            self.config.trainer_config.train_data_loader.pin_memory
            if "pin_memory" in self.config.trainer_config.train_data_loader
            and self.config.trainer_config.train_data_loader.pin_memory is not None
            else True
        )

        # train
        self.train_data_loader = CyclerDataLoader(
            dataset=self.train_dataset,
            steps_per_epoch=self.steps_per_epoch,
            shuffle=self.config.trainer_config.train_data_loader.shuffle,
            batch_size=self.config.trainer_config.train_data_loader.batch_size,
            num_workers=self.config.trainer_config.train_data_loader.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                True
                if self.config.trainer_config.train_data_loader.num_workers > 0
                else None
            ),
            prefetch_factor=(
                self.config.trainer_config.train_data_loader.batch_size
                if self.config.trainer_config.train_data_loader.num_workers > 0
                else None
            ),
        )

        # val
        self.val_data_loader = CyclerDataLoader(
            dataset=self.val_dataset,
            steps_per_epoch=len(self.val_dataset)
            // self.config.trainer_config.val_data_loader.batch_size,
            shuffle=False,
            batch_size=self.config.trainer_config.val_data_loader.batch_size,
            num_workers=self.config.trainer_config.val_data_loader.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                True
                if self.config.trainer_config.val_data_loader.num_workers > 0
                else None
            ),
            prefetch_factor=(
                self.config.trainer_config.val_data_loader.batch_size
                if self.config.trainer_config.val_data_loader.num_workers > 0
                else None
            ),
        )

    def _create_data_loaders_litdata(self):
        """Create a StreamingDataLoader for train, validation and test sets using the data_config."""
        self.chunk_size = (
            self.config.data_config.chunk_size
            if "chunk_size" in self.config.data_config
            and self.config.data_config.chunk_size is not None
            else 100
        )

        def run_subprocess():
            process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    f"sleap_nn.training.get_bin_files",
                    "--dir_path",
                    f"{self.dir_path}",
                    "--bin_files_path",
                    f"{self.litdata_chunks_path}",
                    "--user_instances_only",
                    "1" if self.user_instances_only else "0",
                    "--model_type",
                    f"{self.model_type}",
                    "--num_workers",
                    f"{self.config.trainer_config.train_data_loader.num_workers}",
                    "--chunk_size",
                    f"{self.chunk_size}",
                    "--scale",
                    f"{self.config.data_config.preprocessing.scale}",
                    "--crop_hw",
                    f"{self.crop_hw}",
                    "--max_height",
                    f"{self.max_height}",
                    "--max_width",
                    f"{self.max_width}",
                    "--backbone_type",
                    f"{self.backbone_type}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Use communicate() to read output and avoid hanging
            stdout, stderr = process.communicate()

            # Print the logs
            print("Standard Output:\n", stdout)
            print("Standard Error:\n", stderr)

        if not self.use_existing_chunks:
            try:
                run_subprocess()

            except Exception as e:
                raise Exception(f"Error while creating the `.bin` files... {e}")

        else:
            print(f"Using `.bin` files from {self.litdata_chunks_path}.")
            self.train_litdata_chunks_path = (
                Path(self.litdata_chunks_path) / "train_chunks"
            ).as_posix()
            self.val_litdata_chunks_path = (
                Path(self.litdata_chunks_path) / "val_chunks"
            ).as_posix()

        if self.model_type == "single_instance":

            train_dataset = SingleInstanceStreamingDataset(
                input_dir=self.train_litdata_chunks_path,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
            )

            val_dataset = SingleInstanceStreamingDataset(
                input_dir=self.val_litdata_chunks_path,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
            )

        elif self.model_type == "centered_instance":

            train_dataset = CenteredInstanceStreamingDataset(
                input_dir=self.train_litdata_chunks_path,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                crop_hw=(self.crop_hw, self.crop_hw),
                input_scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = CenteredInstanceStreamingDataset(
                input_dir=self.val_litdata_chunks_path,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                crop_hw=(self.crop_hw, self.crop_hw),
                input_scale=self.config.data_config.preprocessing.scale,
            )

        elif self.model_type == "centroid":
            train_dataset = CentroidStreamingDataset(
                input_dir=self.train_litdata_chunks_path,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
            )

            val_dataset = CentroidStreamingDataset(
                input_dir=self.val_litdata_chunks_path,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
            )

        elif self.model_type == "bottomup":
            train_dataset = BottomUpStreamingDataset(
                input_dir=self.train_litdata_chunks_path,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head=self.config.model_config.head_configs.bottomup.pafs,
                edge_inds=self.edge_inds,
                max_stride=self.max_stride,
            )

            val_dataset = BottomUpStreamingDataset(
                input_dir=self.val_litdata_chunks_path,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head=self.config.model_config.head_configs.bottomup.pafs,
                edge_inds=self.edge_inds,
                max_stride=self.max_stride,
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
            pin_memory=True,
            persistent_workers=(
                True
                if self.config.trainer_config.train_data_loader.num_workers > 0
                else None
            ),
            prefetch_factor=(
                self.config.trainer_config.train_data_loader.batch_size
                if self.config.trainer_config.train_data_loader.num_workers > 0
                else None
            ),
        )

        # val
        self.val_data_loader = ld.StreamingDataLoader(
            val_dataset,
            batch_size=self.config.trainer_config.val_data_loader.batch_size,
            num_workers=self.config.trainer_config.val_data_loader.num_workers,
            pin_memory=True,
            persistent_workers=(
                True
                if self.config.trainer_config.val_data_loader.num_workers > 0
                else None
            ),
            prefetch_factor=(
                self.config.trainer_config.val_data_loader.batch_size
                if self.config.trainer_config.val_data_loader.num_workers > 0
                else None
            ),
        )

    def _set_wandb(self):
        wandb.login(key=self.config.trainer_config.wandb.api_key)

    def _initialize_model(
        self,
    ):
        models = {
            "single_instance": SingleInstanceModel,
            "centered_instance": TopDownCenteredInstanceModel,
            "centroid": CentroidModel,
            "bottomup": BottomUpModel,
        }
        self.model = models[self.model_type](
            config=self.config,
            skeletons=self.skeletons,
            model_type=self.model_type,
            backbone_type=self.backbone_type,
        )

    def _get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(self):
        """Initiate the training by calling the fit method of Trainer."""
        self._initialize_model()
        total_params = self._get_param_count()
        self.config.model_config.total_params = total_params

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
                entity=wandb_config.entity,
                project=wandb_config.project,
                name=wandb_config.name,
                save_dir=self.dir_path,
                id=self.config.trainer_config.wandb.prv_runid,
                group=self.config.trainer_config.wandb.group,
            )
            logger.append(wandb_logger)

            # save the configs as yaml in the checkpoint dir
            self.config.trainer_config.wandb.api_key = ""

            wandb_logger.experiment.config.update({"run_name": wandb_config.name})
            wandb_logger.experiment.config.update(
                {"run_config": OmegaConf.to_container(self.config, resolve=True)}
            )
            wandb_logger.experiment.config.update({"model_params": total_params})

        # save the configs as yaml in the checkpoint dir
        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

        if self.data_pipeline_fw == "litdata":
            self._create_data_loaders_litdata()

        elif (
            self.data_pipeline_fw == "torch_dataset"
            or self.data_pipeline_fw == "torch_dataset_np_chunks"
        ):
            self._create_data_loaders_torch_dataset()

        else:
            raise ValueError(
                f"{self.data_pipeline_fw} is not a valid option. Please choose one of `litdata` or `torch_dataset`."
            )

        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            limit_train_batches=self.steps_per_epoch,
            strategy=(
                "ddp_find_unused_parameters_false"
                if isinstance(self.config.trainer_config.trainer_devices, int)
                and self.config.trainer_config.trainer_devices > 1
                else "auto"
            ),
        )

        try:

            self.trainer.fit(
                self.model,
                self.train_data_loader,
                self.val_data_loader,
                ckpt_path=self.config.trainer_config.resume_ckpt_path,
            )

        except KeyboardInterrupt:
            print("Stopping training...")

        finally:
            if self.config.trainer_config.use_wandb:
                self.config.trainer_config.wandb.run_id = wandb.run.id
                wandb.finish()

            # save the config with wandb runid
            OmegaConf.save(
                config=self.config, f=f"{self.dir_path}/training_config.yaml"
            )

            if (
                self.data_pipeline_fw == "torch_dataset_np_chunks"
                and self.config.data_config.delete_chunks_after_training
            ):
                if (self.train_np_chunks_path).exists():
                    shutil.rmtree(
                        (self.train_np_chunks_path).as_posix(),
                        ignore_errors=True,
                    )

                if (self.val_np_chunks_path).exists():
                    shutil.rmtree(
                        (self.val_np_chunks_path).as_posix(),
                        ignore_errors=True,
                    )

            # TODO: (ubuntu test failing (running for > 6hrs) with the below lines)
            if (
                self.data_pipeline_fw == "litdata"
                and self.config.data_config.delete_chunks_after_training
            ):
                print("Deleting training and validation files...")
                if (Path(self.train_litdata_chunks_path)).exists():
                    shutil.rmtree(
                        (Path(self.train_litdata_chunks_path)).as_posix(),
                        ignore_errors=True,
                    )
                if (Path(self.val_litdata_chunks_path)).exists():
                    shutil.rmtree(
                        (Path(self.val_litdata_chunks_path)).as_posix(),
                        ignore_errors=True,
                    )


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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.skeletons = skeletons
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
            if self.in_channels != input_channels:
                self.input_expand_channels = input_channels
                OmegaConf.update(
                    self.model_config,
                    f"backbone_config.{self.backbone_type}.in_channels",
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
                    print(f"edges: {edges}")
                    head_config[key]["edges"] = edges

        self.model = Model(
            backbone_type=self.backbone_type,
            backbone_config=self.model_config.backbone_config[f"{self.backbone_type}"],
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

        # Pre-trained weights for the encoder stack - only for swint and convnext
        if self.model_config.pre_trained_weights:
            self.model.backbone.enc.load_state_dict(ckpt, strict=False)

        # TODO: Handling different input channels
        # Initializing backbone (encoder + decoder) with trained ckpts
        if self.pretrained_backbone_weights is not None:
            print(
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
            print(f"Loading head weights from `{self.pretrained_head_weights}` ...")
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

        if self.trainer_config.lr_scheduler is None:
            return {
                "optimizer": optimizer,
            }

        else:

            if self.trainer_config.lr_scheduler.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer,
                    step_size=self.trainer_config.lr_scheduler.step_lr.step_size,
                    gamma=self.trainer_config.lr_scheduler.step_lr.gamma,
                )

            elif self.trainer_config.lr_scheduler.scheduler == "ReduceLROnPlateau":
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

            elif self.trainer_config.lr_scheduler.scheduler is not None:
                raise ValueError(
                    f"{self.trainer_config.lr_scheduler.scheduler} is not a valid scheduler. Valid schedulers: `'StepLR'`, `'ReduceLROnPlateau'`"
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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            skeletons=skeletons,
            model_type=model_type,
            backbone_type=backbone_type,
        )

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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            skeletons=skeletons,
            backbone_type=backbone_type,
            model_type=model_type,
        )

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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            skeletons=skeletons,
            backbone_type=backbone_type,
            model_type=model_type,
        )

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
        backbone_type: Backbone model. One of `unet`, `convnext` and `swint`.
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        backbone_type: str,
        model_type: str,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config=config,
            skeletons=skeletons,
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
        X = torch.squeeze(batch["image"], dim=1).to(self.device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)
        y_paf = batch["part_affinity_fields"].to(self.device)
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
        X = torch.squeeze(batch["image"], dim=1).to(self.device)
        y_confmap = torch.squeeze(batch["confidence_maps"], dim=1).to(self.device)
        y_paf = batch["part_affinity_fields"].to(self.device)

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


def train(
    train_labels_path: str,
    val_labels_path: str,
    provider: str = "LabelsReader",
    user_instances_only: bool = True,
    data_pipeline_fw: str = "torch_dataset",
    np_chunks_path: Optional[str] = None,
    litdata_chunks_path: Optional[str] = None,
    use_existing_chunks: bool = False,
    chunk_size: int = 100,
    delete_chunks_after_training: bool = True,
    is_rgb: bool = False,
    scale: float = 1.0,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
    min_crop_size: int = 100,
    use_augmentations_train: bool = False,
    intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    geometry_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    init_weight: str = "default",
    pre_trained_weights: Optional[str] = None,
    pretrained_backbone_weights: Optional[str] = None,
    pretrained_head_weights: Optional[str] = None,
    backbone_config: Union[str, Dict[str, Any]] = "unet",
    head_configs: Union[str, Dict[str, Any]] = None,
    batch_size: int = 4,
    shuffle_train: bool = True,
    num_workers: int = 2,
    ckpt_save_top_k: int = 1,
    ckpt_save_last: bool = False,
    trainer_num_devices: Union[str, int] = "auto",
    trainer_accelerator: str = "auto",
    enable_progress_bar: bool = False,
    steps_per_epoch: Optional[int] = None,
    max_epochs: int = 100,
    seed: int = 1000,
    use_wandb: bool = False,
    save_ckpt: bool = False,
    save_ckpt_path: Optional[str] = None,
    resume_ckpt_path: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_resume_prv_runid: Optional[str] = None,
    wandb_group_name: Optional[str] = None,
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    amsgrad: bool = False,
    lr_scheduler: Optional[
        Union[str, Dict[str, Any]]
    ] = None,  # {"lr_scheduler": {"step_lr": {}, "reduce_lr_on_plateau": {}}}
    early_stopping: bool = False,
    early_stopping_min_delta: float = 0.0,
    early_stopping_patience: int = 1,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
        train_labels_path: Path to training data (`.slp` file).
        val_labels_path: Path to validation data (`.slp` file).
        provider: Provider class to read the input sleap files. Only "LabelsReader"
            supported for the training pipeline. Default: "LabelsReader".
        user_instances_only: `True` if only user labeled instances should be used for
            training. If `False`, both user labeled and predicted instances would be used.
            Default: `True`.
        data_pipeline_fw: Framework to create the data loaders. One of [`litdata`,
            `torch_dataset`, `torch_dataset_np_chunks`]. Default: "torch_dataset".
        np_chunks_path: Path to save `.npz` chunks created with `torch_dataset_np_chunks`
            data pipeline framework. If `None`, the path provided in `trainer_config.save_ckpt`
            is used (else working dir is used). The `train_chunks` and `val_chunks` dirs
            are created inside this path. Default: None.
        litdata_chunks_path: Path to save `.bin` files created with `litdata` data pipeline
            framework. If `None`, the path provided in `trainer_config.save_ckpt` is used
            (else working dir is used). The `train_chunks` and `val_chunks` dirs are created
            inside this path. Default: None.
        use_existing_chunks: Use existing train and val chunks in the `np_chunks_path` or
            `chunks_path` for `torch_dataset_np_chunks` or `litdata` frameworks. If `True`,
            the `np_chunks_path` (or `chunks_path`) should have `train_chunks` and `val_chunks` dirs.
            Default: False.
        chunk_size: Size of each chunk (in MB). Default: 100.
        delete_chunks_after_training: If `False`, the chunks (numpy or litdata chunks) are
            retained after training. Else, the chunks are deleted. Default: True.
        is_rgb: True if the image has 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If input has three channels and this
            is set to False, then we convert the image to grayscale (single-channel)
            image. Default: False.
        scale: Factor to resize the image dimensions by, specified as a float. Default: 1.0.
        max_height: Maximum height the image should be padded to. If not provided, the
            original image size will be retained. Default: None.
        max_width: Maximum width the image should be padded to. If not provided, the
            original image size will be retained. Default: None.
        crop_hw: Crop height and width of each instance (h, w) for centered-instance model.
            If `None`, this would be automatically computed based on the largest instance
            in the `sio.Labels` file. Default: None.
        min_crop_size: Minimum crop size to be used if `crop_hw` is `None`. Default: 100.
        use_augmentations_train: True if the data augmentation should be applied to the
            training data, else False. Default: False.
        intensity_aug: One of ["uniform_noise", "gaussian_noise", "contrast", "brightness"]
            or list of strings from the above allowed values. To have custom values, pass
            a dict with the structure in `sleap_nn.config.data_config.IntensityConfig`.
            For eg: {"intensity": {"uniform_noise_min": 1.0, "uniform_noise_p": 1.0}}
        geometry_aug: One of ["rotation", "scale", "translate", "erase_scale", "mixup"].
            or list of strings from the above allowed values. To have custom values, pass
            a dict with the structure in `sleap_nn.config.data_config.GeometryConfig`.
            For eg: {"geometric": {"rotation": 45, "affine_p": 1.0}}
        init_weight: model weights initialization method. "default" uses kaiming uniform
            initialization and "xavier" uses Xavier initialization method. Default: "default".
        pre_trained_weights: Pretrained weights file name supported only for ConvNext and
            SwinT backbones. For ConvNext, one of ["ConvNeXt_Base_Weights","ConvNeXt_Tiny_Weights",
            "ConvNeXt_Small_Weights", "ConvNeXt_Large_Weights"]. For SwinT, one of ["Swin_T_Weights",
            "Swin_S_Weights", "Swin_B_Weights"]. Default: None.
        pretrained_backbone_weights: Path of the `ckpt` file with which the backbone is
            initialized. If `None`, random init is used. Default: None.
        pretrained_head_weights: Path of the `ckpt` file with which the head layers are
            initialized. If `None`, random init is used. Default: None.
        backbone_config: One of ["unet", "unet_medium_rf", "unet_large_rf", "convnext",
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "swint",
            "swint_tiny", "swint_small", "swint_base"]. If custom values need to be set,
            then pass a dictionary with the structure: {"unet((or) convnext (or)swint)":
            {(params in the corresponding architecture given in `sleap_nn.config.model_config.backbone_config`)}}.
            For eg: {"unet": {"in_channels": 3, "filters": 64, "max_stride": 32, "output_stride": 2}}
        head_configs: One of ["bottomup", "centered_instance", "centroid", "single_instance"].
            The default `sigma` and `output_strides` are used if a string is passed. To
            set custom parameters, pass in a dictionary with the structure:
            {"bottomup" (or "centroid" or "single_instance" or "centered_instance"):
            {"confmaps": {# params in the corresponding head type given in `sleap_nn.config.model_config.head_configs`},
            "pafs": {# only for bottomup}}}.
            For eg: {"single_instance": "confmaps": {{"part_names": None, "sigma": 2.5, "output_stride": 2}}}
        batch_size: Number of samples per batch or batch size for training data. Default: 4.
        shuffle_train: True to have the train data reshuffled at every epoch. Default: False.
        num_workers: Number of subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process. Default: 2.
        ckpt_save_top_k: If save_top_k == k, the best k models according to the quantity
            monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1,
            all models are saved. Please note that the monitors are checked every every_n_epochs
            epochs. if save_top_k >= 2 and the callback is called multiple times inside an
            epoch, the name of the saved file will be appended with a version count starting
            with v1 unless enable_version_counter is set to False. Default: 1.
        ckpt_save_last: When True, saves a last.ckpt whenever a checkpoint file gets saved.
            On a local filesystem, this will be a symbolic link, and otherwise a copy of
            the checkpoint file. This allows accessing the latest checkpoint in a deterministic
            manner. Default: False.
        trainer_num_devices: Number of devices to train on (int), which devices to train
            on (list or str), or "auto" to select automatically. Default: "auto".
        trainer_accelerator: One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises
            the machine the model is running on and chooses the appropriate accelerator for
            the `Trainer` to be connected to. Default: "auto".
        enable_progress_bar: When True, enables printing the logs during training. Default: False.
        steps_per_epoch: Minimum number of iterations in a single epoch. (Useful if model
            is trained with very few data points). Refer `limit_train_batches` parameter
            of Torch `Trainer`. If `None`, the number of iterations depends on the number
            of samples in the train dataset. Default: None.
        max_epochs: Maxinum number of epochs to run. Default: 100.
        seed: Seed value for the current experiment. default: 1000.
        save_ckpt: True to enable checkpointing. Default: False.
        save_ckpt_path: Directory path to save the training config and checkpoint files.
            If `None` and `save_ckpt` is `True`, then the current working dir is used as
            the ckpt path. Default: None
        resume_ckpt_path: Path to `.ckpt` file from which training is resumed. Default: None.
        use_wandb: True to enable wandb logging. Default: False.
        wandb_entity: Entity of wandb project. Default: None.
            (The default entity in the user profile settings is used)
        wandb_project: Project name for the current wandb run. Default: None.
        wandb_name: Name of the current wandb run. Default: None.
        wandb_api_key: API key. The API key is masked when saved to config files. Default: None.
        wandb_mode: "offline" if only local logging is required. Default: None.
        wandb_resume_prv_runid: Previous run ID if training should be resumed from a previous
            ckpt. Default: None
        wandb_group_name: Group name fo the wandb run. Default: None.
        optimizer: Optimizer to be used. One of ["Adam", "AdamW"]. Default: "Adam".
        learning_rate: Learning rate of type float. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Defaul: False.
        lr_scheduler: One of ["StepLR", "ReduceLROnPlateau"] (the default values in
            `sleap_nn.config.trainer_config` are used). To use custom values, pass a
            dictionary with the structure in `sleap_nn.config.trainer_config.LRSchedulerConfig`.
            For eg, {"scheduler": "StepLR", "step_lr": {(params in `sleap_nn.config.trainer_config.StepLRConfig`)}}
        early_stopping: True if early stopping should be enabled. Default: False.
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as
            an improvement, i.e. an absolute change of less than or equal to min_delta,
            will count as no improvement. Default: 0.0.
        early_stopping_patience: Number of checks with no improvement after which training
            will be stopped. Under the default configuration, one check happens after every
            training epoch. Default: 1.
    """
    preprocessing_config = PreprocessingConfig(
        is_rgb=is_rgb,
        max_height=max_height,
        max_width=max_width,
        scale=scale,
        crop_hw=crop_hw,
        min_crop_size=min_crop_size,
    )
    augmentation_config = None
    if use_augmentations_train:
        augmentation_config = get_aug_config(
            intensity_aug=intensity_aug, geometric_aug=geometry_aug
        )

    # construct data config
    data_config = DataConfig(
        train_labels_path=train_labels_path,
        val_labels_path=val_labels_path,
        provider=provider,
        user_instances_only=user_instances_only,
        data_pipeline_fw=data_pipeline_fw,
        np_chunks_path=np_chunks_path,
        litdata_chunks_path=litdata_chunks_path,
        use_existing_chunks=use_existing_chunks,
        chunk_size=chunk_size,
        delete_chunks_after_training=delete_chunks_after_training,
        preprocessing=preprocessing_config,
        use_augmentations_train=use_augmentations_train,
        augmentation_config=augmentation_config,
    )

    # construct model config
    backbone_config = get_backbone_config(backbone_cfg=backbone_config)
    head_configs = get_head_configs(head_cfg=head_configs)
    model_config = ModelConfig(
        init_weights=init_weight,
        pre_trained_weights=pre_trained_weights,
        pretrained_backbone_weights=pretrained_backbone_weights,
        pretrained_head_weights=pretrained_head_weights,
        backbone_config=backbone_config,
        head_configs=head_configs,
    )

    # constrict trainer config
    train_dataloader_cfg = DataLoaderConfig(
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_dataloader_cfg = DataLoaderConfig(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    lr_scheduler_cfg = None
    if isinstance(lr_scheduler, str):
        lr_scheduler_cfg = LRSchedulerConfig(scheduler=lr_scheduler)
        if lr_scheduler == "StepLR":
            lr_scheduler_cfg.step_lr = StepLRConfig()
        if lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_cfg.reduce_lr_on_plateau = ReduceLROnPlateauConfig()
    elif isinstance(lr_scheduler, dict):
        lr_scheduler_cfg = LRSchedulerConfig(**lr_scheduler)

    trainer_config = TrainerConfig(
        train_data_loader=train_dataloader_cfg,
        val_data_loader=val_dataloader_cfg,
        model_ckpt=ModelCkptConfig(
            save_top_k=ckpt_save_top_k, save_last=ckpt_save_last
        ),
        trainer_devices=trainer_num_devices,
        trainer_accelerator=trainer_accelerator,
        enable_progress_bar=enable_progress_bar,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs,
        seed=seed,
        use_wandb=use_wandb,
        wandb=WandBConfig(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_name,
            api_key=wandb_api_key,
            wandb_mode=wandb_mode,
            prv_runid=wandb_resume_prv_runid,
            group=wandb_group_name,
        ),
        save_ckpt=save_ckpt,
        save_ckpt_path=save_ckpt_path,
        resume_ckpt_path=resume_ckpt_path,
        optimizer_name=optimizer,
        optimizer=OptimizerConfig(lr=learning_rate, amsgrad=amsgrad),
        lr_scheduler=lr_scheduler_cfg,
        early_stopping=EarlyStoppingConfig(
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            stop_training_on_plateau=early_stopping,
        ),
    )

    # create omegaconf object (or attrs class object)
    training_job_config = TrainingJobConfig(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    omegaconf_config = OmegaConf.structured(training_job_config)
    print(f"omegaconf: {omegaconf_config}")

    # create an instance of the ModelTrainer class
    trainer = ModelTrainer(omegaconf_config)
    trainer.train()

    print(f"Training completed!")

    # compute metrics on train and val set
