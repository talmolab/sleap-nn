"""This module is to train a sleap-nn model using Lightning."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
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
from sleap_nn.training.utils import check_memory, xavier_init_weights

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
        data_pipeline_fw: Framework to create the data loaders. One of [`litdata`, `torch_dataset`, `torch_dataset_np_chunks`]
        np_chunks_path: Path to save `.npz` chunks created with `torch_dataset_np_chunks` data pipeline framework.
        use_existing_np_chunks: Use existing train and val chunks in the `np_chunks_path`.
    """

    def __init__(
        self,
        config: OmegaConf,
        data_pipeline_fw: str = "litdata",
        np_chunks_path: Optional[str] = None,
        use_existing_np_chunks: bool = False,
    ):
        """Initialise the class with configs and set the seed and device as class attributes."""
        self.config = config
        self.data_pipeline_fw = data_pipeline_fw
        self.np_chunks = True if "np_chunks" in self.data_pipeline_fw else False
        self.train_np_chunks_path = (
            Path(np_chunks_path) / "train_chunks"
            if np_chunks_path is not None
            else None
        )
        self.val_np_chunks_path = (
            Path(np_chunks_path) / "val_chunks" if np_chunks_path is not None else None
        )
        self.use_existing_np_chunks = use_existing_np_chunks
        if self.use_existing_np_chunks:
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
        self.train_dataset = None
        self.val_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.bin_files_path = None
        self.trainer = None
        self.crop_hw = -1

        # check which head type to choose the model
        for k, v in self.config.model_config.head_configs.items():
            if v is not None:
                self.model_type = k
                break

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

        self.max_stride = self.config.model_config.backbone_config.max_stride
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
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )
            self.val_dataset = BottomUpDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )

        elif self.model_type == "centered_instance":
            self.train_dataset = CenteredInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )
            self.val_dataset = CenteredInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=False,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )

        elif self.model_type == "centroid":
            self.train_dataset = CentroidDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )
            self.val_dataset = CentroidDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )

        elif self.model_type == "single_instance":
            self.train_dataset = SingleInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
            )
            self.val_dataset = SingleInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.config.model_config.backbone_config.max_stride,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_np_chunks,
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

    def _create_data_loaders_litdata(
        self,
        chunks_dir_path: Optional[str] = None,
    ):
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
                    f"{self.bin_files_path}",
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

        if chunks_dir_path is None:
            try:
                self.bin_files_path = self.config.trainer_config.bin_files_path
                if self.bin_files_path is None:
                    self.bin_files_path = self.dir_path

                self.bin_files_path = f"{self.bin_files_path}/chunks_{datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S-%f')}"
                print(
                    f"New dir is created and `.bin` files are saved in {self.bin_files_path}"
                )

                if not Path(self.bin_files_path).exists():
                    try:
                        Path(self.bin_files_path).mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        raise OSError(
                            f"Cannot create a new folder in {self.bin_files_path}. Check the permissions to the given Checkpoint directory. \n {e}"
                        )

                self.config.trainer_config.saved_bin_files_path = self.bin_files_path

                self.train_input_dir = (
                    Path(self.bin_files_path) / "train_chunks"
                ).as_posix()
                self.val_input_dir = (
                    Path(self.bin_files_path) / "val_chunks"
                ).as_posix()

                run_subprocess()

            except Exception as e:
                raise Exception(f"Error while creating the `.bin` files... {e}")

        else:
            print(f"Using `.bin` files from {chunks_dir_path}.")
            self.train_input_dir = (Path(chunks_dir_path) / "train_chunks").as_posix()
            self.val_input_dir = (Path(chunks_dir_path) / "val_chunks").as_posix()
            self.config.trainer_config.saved_bin_files_path = chunks_dir_path

        if self.model_type == "single_instance":

            train_dataset = SingleInstanceStreamingDataset(
                input_dir=self.train_input_dir,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
            )

            val_dataset = SingleInstanceStreamingDataset(
                input_dir=self.val_input_dir,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
            )

        elif self.model_type == "centered_instance":

            train_dataset = CenteredInstanceStreamingDataset(
                input_dir=self.train_input_dir,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                crop_hw=(self.crop_hw, self.crop_hw),
                input_scale=self.config.data_config.preprocessing.scale,
            )

            val_dataset = CenteredInstanceStreamingDataset(
                input_dir=self.val_input_dir,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                crop_hw=(self.crop_hw, self.crop_hw),
                input_scale=self.config.data_config.preprocessing.scale,
            )

        elif self.model_type == "centroid":
            train_dataset = CentroidStreamingDataset(
                input_dir=self.train_input_dir,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
            )

            val_dataset = CentroidStreamingDataset(
                input_dir=self.val_input_dir,
                shuffle=False,
                apply_aug=False,
                confmap_head=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
            )

        elif self.model_type == "bottomup":
            train_dataset = BottomUpStreamingDataset(
                input_dir=self.train_input_dir,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
                apply_aug=self.config.data_config.use_augmentations_train,
                augmentation_config=self.config.data_config.augmentation_config,
                confmap_head=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head=self.config.model_config.head_configs.bottomup.pafs,
                edge_inds=self.edge_inds,
                max_stride=self.max_stride,
            )

            val_dataset = BottomUpStreamingDataset(
                input_dir=self.val_input_dir,
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
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        models = {
            "single_instance": SingleInstanceModel,
            "centered_instance": TopDownCenteredInstanceModel,
            "centroid": CentroidModel,
            "bottomup": BottomUpModel,
        }
        self.model = models[self.model_type](
            self.config,
            self.skeletons,
            self.model_type,
            backbone_trained_ckpts_path=backbone_trained_ckpts_path,
            head_trained_ckpts_path=head_trained_ckpts_path,
        )

    def _get_param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def train(
        self,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
        delete_bin_files_after_training: bool = True,
        chunks_dir_path: Optional[str] = None,
        delete_np_chunks_after_training: bool = True,
    ):
        """Initiate the training by calling the fit method of Trainer.

        Args:
            backbone_trained_ckpts_path: Path of the `ckpt` file with which the backbone
                 is initialized. If `None`, random init is used.
            head_trained_ckpts_path: Path of the `ckpt` file with which the head layers
                 are initialized. If `None`, random init is used.
            delete_bin_files_after_training: If `False`, the `bin` files are retained after
                training. Else, the `bin` files are deleted.
            chunks_dir_path: Path to chunks dir (this dir should contain `train_chunks`
                and `val_chunks` folder.). If `None`, `bin` files are generated.
            delete_np_chunks_after_training: If `False`, the numpy chunks are retained after
                training. Else, the numpy chunks are deleted.

        """
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

        self._initialize_model(
            backbone_trained_ckpts_path=backbone_trained_ckpts_path,
            head_trained_ckpts_path=head_trained_ckpts_path,
        )
        total_params = self._get_param_count()
        self.config.model_config.total_params = total_params
        # save the configs as yaml in the checkpoint dir
        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

        if self.data_pipeline_fw == "litdata":
            self._create_data_loaders_litdata(chunks_dir_path)

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
        )

        try:
            if self.config.trainer_config.use_wandb:
                wandb_logger.experiment.config.update({"run_name": wandb_config.name})
                wandb_logger.experiment.config.update(dict(self.config))
                wandb_logger.experiment.config.update({"model_params": total_params})

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

            if self.np_chunks and delete_np_chunks_after_training:
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
            if self.data_pipeline_fw == "litdata" and delete_bin_files_after_training:
                print("Deleting training and validation files...")
                if (Path(self.train_input_dir)).exists():
                    shutil.rmtree(
                        (Path(self.train_input_dir)).as_posix(),
                        ignore_errors=True,
                    )
                if (Path(self.val_input_dir)).exists():
                    shutil.rmtree(
                        (Path(self.val_input_dir)).as_posix(),
                        ignore_errors=True,
                    )
                if self.bin_files_path is not None:
                    shutil.rmtree(
                        (Path(self.bin_files_path)).as_posix(),
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
        backbone_trained_ckpts_path: Path to trained ckpts for backbone.
        head_trained_ckpts_path: Path to trained ckpts for head layer.
    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        """Initialise the configs and the model."""
        super().__init__()
        self.config = config
        self.skeletons = skeletons
        self.model_config = self.config.model_config
        self.trainer_config = self.config.trainer_config
        self.data_config = self.config.data_config
        self.model_type = model_type
        self.backbone_trained_ckpts_path = backbone_trained_ckpts_path
        self.head_trained_ckpts_path = head_trained_ckpts_path
        self.input_expand_channels = self.model_config.backbone_config.in_channels
        if self.model_config.pre_trained_weights:  # only for swint and convnext
            ckpt = MODEL_WEIGHTS[
                self.model_config.pre_trained_weights
            ].DEFAULT.get_state_dict(progress=True, check_hash=True)
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

        # Pre-trained weights for the encoder stack - only for swint and convnext
        if self.model_config.pre_trained_weights:
            self.model.backbone.enc.load_state_dict(ckpt, strict=False)

        # TODO: Handling different input channels
        # Initializing backbone (encoder + decoder) with trained ckpts
        if backbone_trained_ckpts_path is not None:
            print(f"Loading backbone weights from `{backbone_trained_ckpts_path}` ...")
            ckpt = torch.load(backbone_trained_ckpts_path)
            ckpt["state_dict"] = {
                k: ckpt["state_dict"][k]
                for k in ckpt["state_dict"].keys()
                if ".backbone" in k
            }
            self.load_state_dict(ckpt["state_dict"], strict=False)

        # Initializing head layers with trained ckpts.
        if head_trained_ckpts_path is not None:
            print(f"Loading head weights from `{head_trained_ckpts_path}` ...")
            ckpt = torch.load(head_trained_ckpts_path)
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

        if self.trainer_config.lr_scheduler.scheduler is None:
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
        backbone_trained_ckpts_path: Path to trained ckpts for backbone.
        head_trained_ckpts_path: Path to trained ckpts for head layer.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config,
            skeletons,
            model_type,
            backbone_trained_ckpts_path,
            head_trained_ckpts_path,
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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
        backbone_trained_ckpts_path: Path to trained ckpts for backbone.
        head_trained_ckpts_path: Path to trained ckpts for head layer.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config,
            skeletons,
            model_type,
            backbone_trained_ckpts_path,
            head_trained_ckpts_path,
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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
        backbone_trained_ckpts_path: Path to trained ckpts for backbone.
        head_trained_ckpts_path: Path to trained ckpts for head layer.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config,
            skeletons,
            model_type,
            backbone_trained_ckpts_path,
            head_trained_ckpts_path,
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
        model_type: Type of the model. One of `single_instance`, `centered_instance`, `centroid`, `bottomup`.
        backbone_trained_ckpts_path: Path to trained ckpts for backbone.
        head_trained_ckpts_path: Path to trained ckpts for head layer.

    """

    def __init__(
        self,
        config: OmegaConf,
        skeletons: Optional[List[sio.Skeleton]],
        model_type: str,
        backbone_trained_ckpts_path: Optional[str] = None,
        head_trained_ckpts_path: Optional[str] = None,
    ):
        """Initialise the configs and the model."""
        super().__init__(
            config,
            skeletons,
            model_type,
            backbone_trained_ckpts_path,
            head_trained_ckpts_path,
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
