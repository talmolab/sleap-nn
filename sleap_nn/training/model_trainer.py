"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
import os
import psutil
import shutil
import subprocess
import torch
from torch.utils.data import DistributedSampler, DataLoader
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
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    CenteredInstanceDataset,
    CentroidDataset,
    SingleInstanceDataset,
    CyclerDataLoader,
)
from lightning.pytorch.utilities import CombinedLoader
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.streaming_datasets import (
    BottomUpStreamingDataset,
    CenteredInstanceStreamingDataset,
    CentroidStreamingDataset,
    SingleInstanceStreamingDataset,
)
from loguru import logger
from sleap_nn.training.utils import check_memory


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
from sleap_nn.training.lightning_modules import (
    BottomUpModel,
    CentroidModel,
    TopDownCenteredInstanceModel,
    SingleInstanceModel,
    BottomUpMultiHeadModel,
    CentroidMultiHeadModel,
    TopDownCenteredInstanceMultiHeadModel,
    SingleInstanceMultiHeadModel,
)


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
                message = f"Cannot create a new folder in {self.dir_path}. Check the permissions to the given Checkpoint directory. \n {e}"
                logger.error(message)
                raise OSError(message)

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
                    message = f"There are no numpy chunks in the path: {self.train_np_chunks_path}"
                    logger.error(message)
                    raise Exception(message)

                if not (
                    self.val_np_chunks_path.exists()
                    and self.val_np_chunks_path.is_dir()
                    and any(self.val_np_chunks_path.glob("*.npz"))
                ):
                    message = f"There are no numpy chunks in the path: {self.val_np_chunks_path}"
                    logger.error(message)
                    raise Exception(message)

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

        # if edges and part names aren't set in config, get it from `sio.Labels` object.
        head_config = self.config.model_config.head_configs[self.model_type]
        for key in head_config:
            if "part_names" in head_config[key].keys():
                if head_config[key]["part_names"] is None:
                    part_names = [x.name for x in self.skeletons[0].nodes]
                    self.config.model_config.head_configs[self.model_type][key][
                        "part_names"
                    ] = part_names

            if "edges" in head_config[key].keys():
                if head_config[key]["edges"] is None:
                    edges = [
                        (x.source.name, x.destination.name)
                        for x in self.skeletons[0].edges
                    ]
                    self.config.model_config.head_configs[self.model_type][key][
                        "edges"
                    ] = edges

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

        if self.config.data_config.preprocessing.scale is None:
            self.config.data_config.preprocessing.scale = 1.0

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
                logger.info(
                    f"Insufficient memory for in-memory caching. `npz` files will be created."
                )

        if self.model_type == "bottomup":
            self.train_dataset = BottomUpDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
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
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_path,
                use_existing_chunks=self.use_existing_chunks,
            )

        else:
            message = f"Model type: {self.model_type}. Ensure the heads config has one of the keys: [`bottomup`, `centroid`, `centered_instance`, `single_instance`]."
            logger.error(message)
            raise ValueError(message)

        if self.steps_per_epoch is None:
            self.steps_per_epoch = (
                len(self.train_dataset)
                // self.config.trainer_config.train_data_loader.batch_size
            )
            if self.steps_per_epoch == 0:
                self.steps_per_epoch = 1

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
        val_steps_per_epoch = (
            len(self.val_dataset)
            // self.config.trainer_config.val_data_loader.batch_size
        )
        self.val_data_loader = CyclerDataLoader(
            dataset=self.val_dataset,
            steps_per_epoch=val_steps_per_epoch if val_steps_per_epoch != 0 else 1,
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

            # logger.info the logs
            logger.info("Standard Output:\n", stdout)
            logger.info("Standard Error:\n", stderr)

        if not self.use_existing_chunks:
            try:
                run_subprocess()

            except Exception as e:
                message = f"Error while creating the `.bin` files... {e}"
                logger.error(message)
                raise Exception(message)

        else:
            logger.info(f"Using `.bin` files from {self.litdata_chunks_path}.")
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
            message = f"{self.model_type} is not defined. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`."
            logger.error(message)
            raise ValueError(message)

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

        training_loggers = []

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
            training_loggers.append(csv_logger)

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
            training_loggers.append(wandb_logger)

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
            message = f"{self.data_pipeline_fw} is not a valid option. Please choose one of `litdata` or `torch_dataset`."
            logger.error(message)
            raise ValueError(message)

        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=training_loggers,
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
            logger.info("Stopping training...")

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
                logger.info("Deleting training and validation files...")
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


class MultiHeadModelTrainer:
    """Train sleap-nn multi-head model using PyTorch Lightning.

    This class is used to train a multi-head model and save the model checkpoints/ logs with options to logging
    with wandb and csvlogger.

    Args:
        config: OmegaConf dictionary which has the following:
                (i) dataset_mapper: mapping between dataset numbers and dataset name.
                (ii) data_config: data loading pre-processing configs.
                (iii) model_config: backbone and head configs to be passed to `Model` class.
                (iv) trainer_config: trainer configs like accelerator, optimiser params.
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
                message = f"Cannot create a new folder in {self.dir_path}. Check the permissions to the given Checkpoint directory. \n {e}"
                logger.error(message)
                raise OSError(message)

        if self.data_pipeline_fw == "litdata":  # TODO
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
            self.train_datasets, self.val_datasets = {}, {}
            self.train_data_loaders, self.val_data_loaders = {}, {}
            # Get np chunks path
            self.np_chunks = True if "np_chunks" in self.data_pipeline_fw else False
            self.np_chunks_dir = (
                Path(self.config.data_config.np_chunks_path)
                if self.config.data_config.np_chunks_path is not None
                else Path(self.dir_path)
            )

            self.train_np_chunks_paths = {}
            self.val_np_chunks_paths = {}

            for d_num, d_name in self.config.dataset_mapper.items():
                self.train_np_chunks_paths[d_num] = (
                    Path(self.np_chunks_dir) / f"{d_name}" / "train_chunks"
                )
                self.val_np_chunks_paths[d_num] = (
                    Path(self.np_chunks_dir) / f"{d_name}" / "val_chunks"
                )

                if self.use_existing_chunks[d_num]:
                    if not (
                        self.train_np_chunks_paths[d_num].exists()
                        and self.train_np_chunks_paths[d_num].is_dir()
                        and any(self.train_np_chunks_paths[d_num].glob("*.npz"))
                    ):
                        message = f"There are no numpy chunks in the path: {self.train_np_chunks_paths[d_num]}"
                        logger.error(message)
                        raise Exception(message)

                    if not (
                        self.val_np_chunks_paths[d_num].exists()
                        and self.val_np_chunks_paths[d_num].is_dir()
                        and any(self.val_np_chunks_paths[d_num].glob("*.npz"))
                    ):
                        message = f"There are no numpy chunks in the path: {self.val_np_chunks_paths[d_num]}"
                        logger.error(message)
                        raise Exception(message)

        self.seed = self.config.trainer_config.seed
        self.steps_per_epochs = self.config.trainer_config.steps_per_epoch
        if self.steps_per_epochs is None:
            self.steps_per_epochs = {}

        # initialize attributes
        self.model = None
        self.trainer = None

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

        self.max_stride = self.config.model_config.backbone_config[
            f"{self.backbone_type}"
        ]["max_stride"]

        self.edge_inds = {}
        self.max_heights = {}
        self.max_widths = {}
        self.crop_hws = {}
        self.skeletons_dict = {}
        OmegaConf.update(self.config.data_config, f"skeletons", {})

        for d_num, d_name in self.config.dataset_mapper.items():
            train_labels = sio.load_slp(
                self.config.data_config.train_labels_path[d_num]
            )
            self.user_instances_only = (
                self.config.data_config.user_instances_only
                if "user_instances_only" in self.config.data_config
                and self.config.data_config.user_instances_only is not None
                else True
            )
            if self.config.data_config.preprocessing.scale[d_num] is None:
                self.config.data_config.preprocessing.scale[d_num] = 1.0
            self.skeletons_dict[d_num] = train_labels.skeletons

            # save the skeleton in the config
            for skl in self.skeletons_dict[d_num]:
                if skl.symmetries:
                    symm = [list(s.nodes) for s in skl.symmetries]
                else:
                    symm = None
                skl_name = skl.name if skl.name is not None else "skeleton-0"
                self.config["data_config"]["skeletons"][d_num] = {
                    skl_name: {
                        "nodes": skl.nodes,
                        "edges": skl.edges,
                        "symmetries": symm,
                    }
                }

            # if edges and part names aren't set in config, get it from `sio.Labels` object.
            head_configs = self.config.model_config.head_configs[self.model_type]
            for key in head_configs:
                if "part_names" in head_configs[key][d_num].keys():
                    if head_configs[key][d_num]["part_names"] is None:
                        part_names = [
                            x.name for x in self.skeletons_dict[d_num][0].nodes
                        ]
                        self.config.model_config.head_configs[self.model_type][key][
                            d_num
                        ]["part_names"] = part_names

                if "edges" in head_configs[key][d_num].keys():
                    if head_configs[key][d_num]["edges"] is None:
                        edges = [
                            (x.source.name, x.destination.name)
                            for x in self.skeletons_dict[d_num][0].edges
                        ]
                        self.config.model_config.head_configs[self.model_type][key][
                            d_num
                        ]["edges"] = edges

            self.edge_inds[d_num] = train_labels.skeletons[0].edge_inds
            self.max_heights[d_num], self.max_widths[d_num] = get_max_height_width(
                train_labels
            )
            if (
                self.config.data_config.preprocessing.max_height[d_num] is None
                and self.config.data_config.preprocessing.max_width[d_num] is None
            ):
                self.config.data_config.preprocessing.max_height[d_num] = (
                    self.max_heights[d_num]
                )
                self.config.data_config.preprocessing.max_width[d_num] = (
                    self.max_widths[d_num]
                )

            if self.model_type == "centered_instance":
                # compute crop size
                self.crop_hws[d_num] = self.config.data_config.preprocessing.crop_hw[
                    d_num
                ]
                if self.crop_hws[d_num] is None:

                    min_crop_size = (
                        self.config.data_config.preprocessing.min_crop_size
                        if "min_crop_size" in self.config.data_config.preprocessing
                        else None
                    )
                    crop_size = find_instance_crop_size(
                        train_labels,
                        maximum_stride=self.max_stride,
                        min_crop_size=min_crop_size,
                        input_scaling=self.config.data_config.preprocessing.scale[
                            d_num
                        ],
                    )
                    self.crop_hws[d_num] = crop_size
                    self.config.data_config.preprocessing.crop_hw[d_num] = (
                        self.crop_hws[d_num],
                        self.crop_hws[d_num],
                    )
                else:
                    self.crop_hws[d_num] = self.crop_hws[d_num][0]

        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

    def _create_data_loaders_torch_dataset(self, d_num):
        """Create a torch DataLoader for train, validation and test sets using the data_config."""
        train_labels = sio.load_slp(self.config.data_config.train_labels_path[d_num])
        val_labels = sio.load_slp(self.config.data_config.val_labels_path[d_num])
        if self.data_pipeline_fw == "torch_dataset":
            train_cache_memory = check_memory(
                train_labels,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                model_type=self.model_type,
                input_scaling=self.config.data_config.preprocessing.scale[d_num],
                crop_size=self.crop_hws[d_num] if self.crop_hws else None,
            )
            val_cache_memory = check_memory(
                val_labels,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                model_type=self.model_type,
                input_scaling=self.config.data_config.preprocessing.scale[d_num],
                crop_size=self.crop_hws[d_num] if self.crop_hws else None,
            )
            total_cache_memory = train_cache_memory + val_cache_memory
            total_cache_memory += 0.1 * total_cache_memory  # memory required in bytes
            available_memory = (
                psutil.virtual_memory().available
            )  # available memory in bytes

            if total_cache_memory > available_memory:
                self.data_pipeline_fw = "torch_dataset_np_chunks"
                self.np_chunks = True
                self.np_chunks_dir = (
                    Path(self.config.data_config.np_chunks_path)
                    if self.config.data_config.np_chunks_path is not None
                    else Path(self.dir_path)
                )

                self.train_np_chunks_paths = {}
                self.val_np_chunks_paths = {}

                for d_num, d_name in self.config.dataset_mapper.items():
                    self.train_np_chunks_paths[d_num] = (
                        Path(self.np_chunks_dir) / f"{d_name}" / "train_chunks"
                    )
                    self.val_np_chunks_paths[d_num] = (
                        Path(self.np_chunks_dir) / f"{d_name}" / "val_chunks"
                    )

                logger.info(
                    f"Insufficient memory for in-memory caching. `npz` files will be created."
                )

        if self.model_type == "bottomup":
            self.train_datasets[d_num] = BottomUpDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps[
                    d_num
                ],
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )
            self.val_datasets[d_num] = BottomUpDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                scale=self.config.data_config.preprocessing.scale[d_num],
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps[
                    d_num
                ],
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs[
                    d_num
                ],
                max_stride=self.max_stride,
                apply_aug=False,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )

        elif self.model_type == "centered_instance":
            self.train_datasets[d_num] = CenteredInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=self.config.data_config.use_augmentations_train,
                crop_hw=(self.crop_hws[d_num], self.crop_hws[d_num]),
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )
            self.val_datasets[d_num] = CenteredInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=False,
                crop_hw=(self.crop_hws[d_num], self.crop_hws[d_num]),
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )

        elif self.model_type == "centroid":
            self.train_datasets[d_num] = CentroidDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )
            self.val_datasets[d_num] = CentroidDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=False,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )

        elif self.model_type == "single_instance":
            self.train_datasets[d_num] = SingleInstanceDataset(
                labels=train_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.train_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )
            self.val_datasets[d_num] = SingleInstanceDataset(
                labels=val_labels,
                data_config=self.config.data_config,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps[
                    d_num
                ],
                max_stride=self.max_stride,
                scale=self.config.data_config.preprocessing.scale[d_num],
                apply_aug=False,
                max_hw=(self.max_heights[d_num], self.max_widths[d_num]),
                np_chunks=self.np_chunks,
                np_chunks_path=self.val_np_chunks_paths[d_num],
                use_existing_chunks=self.use_existing_chunks[d_num],
            )

        else:
            message = f"Model type: {self.model_type}. Ensure the heads config has one of the keys: [`bottomup`, `centroid`, `centered_instance`, `single_instance`]."
            logger.error(message)
            raise ValueError(message)

        if self.steps_per_epochs.get(d_num, None) is None:
            self.steps_per_epochs[d_num] = (
                len(self.train_datasets[d_num])
                // self.config.trainer_config.train_data_loader.batch_size
            )
            if self.steps_per_epochs[d_num] == 0:
                self.steps_per_epochs[d_num] = 1

        val_steps_per_epoch = (
            len(self.val_datasets[d_num])
            // self.config.trainer_config.val_data_loader.batch_size
        )

        pin_memory = (
            self.config.trainer_config.train_data_loader.pin_memory
            if "pin_memory" in self.config.trainer_config.train_data_loader
            and self.config.trainer_config.train_data_loader.pin_memory is not None
            else True
        )

        # train
        if self.config.trainer_config.trainer_devices > 1:
            train_sampler = DistributedSampler(
                self.train_datasets[d_num],
                num_replicas=self.config.trainer_config.trainer_devices,
                rank=self.trainer.global_rank,
                shuffle=self.config.trainer_config.train_data_loader.shuffle,
            )
            self.train_data_loaders[d_num] = DataLoader(
                dataset=self.train_datasets[d_num],
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
                sampler=train_sampler,
                multiprocessing_context="forkserver",
            )

            val_sampler = DistributedSampler(
                self.val_datasets[d_num],
                num_replicas=self.config.trainer_config.trainer_devices,
                rank=self.trainer.global_rank,
                shuffle=False,
            )
            self.val_data_loaders[d_num] = DataLoader(
                dataset=self.val_datasets[d_num],
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
                sampler=val_sampler,
                multiprocessing_context="forkserver",
            )

        else:
            # train
            self.train_data_loaders[d_num] = CyclerDataLoader(
                dataset=self.train_datasets[d_num],
                steps_per_epoch=self.steps_per_epochs[d_num],
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
            self.val_data_loaders[d_num] = CyclerDataLoader(
                dataset=self.val_datasets[d_num],
                steps_per_epoch=val_steps_per_epoch if val_steps_per_epoch != 0 else 1,
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

    def _create_data_loaders_litdata(self):  # TODO
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

            # logger.info the logs
            logger.info("Standard Output:\n", stdout)
            logger.info("Standard Error:\n", stderr)

        if not self.use_existing_chunks:
            try:
                run_subprocess()

            except Exception as e:
                message = f"Error while creating the `.bin` files... {e}"
                logger.error(message)
                raise Exception(message)

        else:
            logger.info(f"Using `.bin` files from {self.litdata_chunks_path}.")
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
            message = f"{self.model_type} is not defined. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`."
            logger.error(message)
            raise ValueError(message)

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
            "single_instance": SingleInstanceMultiHeadModel,
            "centered_instance": TopDownCenteredInstanceMultiHeadModel,
            "centroid": CentroidMultiHeadModel,
            "bottomup": BottomUpMultiHeadModel,
        }
        self.model = models[self.model_type](
            config=self.config,
            skeletons_dict=self.skeletons_dict,
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

        training_loggers = []

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
            training_loggers.append(csv_logger)

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
            training_loggers.append(wandb_logger)

            # save the configs as yaml in the checkpoint dir
            self.config.trainer_config.wandb.api_key = ""

        # save the configs as yaml in the checkpoint dir
        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=training_loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            strategy=self.config.trainer_config.trainer_strategy,
        )

        if self.data_pipeline_fw == "litdata":
            self._create_data_loaders_litdata()

        elif (
            self.data_pipeline_fw == "torch_dataset"
            or self.data_pipeline_fw == "torch_dataset_np_chunks"
        ):
            for d_num, _ in self.config.dataset_mapper.items():
                self._create_data_loaders_torch_dataset(d_num=d_num)
            self.combined_train_dataloader = CombinedLoader(
                self.train_data_loaders,
                mode=self.config.trainer_config.combined_loader_mode,
            )
            self.combined_val_dataloader = CombinedLoader(
                self.val_data_loaders,
                mode=self.config.trainer_config.combined_loader_mode,
            )

        else:
            message = f"{self.data_pipeline_fw} is not a valid option. Please choose one of `litdata` or `torch_dataset`."
            logger.error(message)
            raise ValueError(message)

        if self.trainer.global_rank == 0 and self.config.trainer_config.use_wandb:
            wandb_logger.experiment.config.update({"run_name": wandb_config.name})
            wandb_logger.experiment.config.update(
                {"run_config": OmegaConf.to_container(self.config, resolve=True)}
            )
            wandb_logger.experiment.config.update({"model_params": total_params})

        try:

            self.trainer.fit(
                self.model,
                self.combined_train_dataloader,
                self.combined_val_dataloader,
                ckpt_path=self.config.trainer_config.resume_ckpt_path,
            )

        except KeyboardInterrupt:
            logger.info("Stopping training...")

        finally:
            if self.trainer.global_rank == 0:

                if self.config.trainer_config.use_wandb:
                    self.config.trainer_config.wandb.run_id = wandb.run.id
                    wandb.finish()

                OmegaConf.save(
                    config=self.config, f=f"{self.dir_path}/training_config.yaml"
                )

                if (
                    self.data_pipeline_fw == "torch_dataset_np_chunks"
                    and self.config.data_config.delete_chunks_after_training
                ):
                    for d_num, d_name in self.config.dataset_mapper.items():
                        if (self.train_np_chunks_paths[d_num]).exists():
                            shutil.rmtree(
                                (self.train_np_chunks_paths[d_num]).as_posix(),
                                ignore_errors=True,
                            )

                        if (self.val_np_chunks_paths[d_num]).exists():
                            shutil.rmtree(
                                (self.val_np_chunks_paths[d_num]).as_posix(),
                                ignore_errors=True,
                            )

                # TODO: (ubuntu test failing (running for > 6hrs) with the below lines)
                if (
                    self.data_pipeline_fw == "litdata"
                    and self.config.data_config.delete_chunks_after_training
                ):  # TODO
                    logger.info("Deleting training and validation files...")
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
