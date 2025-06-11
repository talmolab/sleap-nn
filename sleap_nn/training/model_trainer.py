"""This module is to train a sleap-nn model using Lightning."""

from pathlib import Path
import os
import psutil
import shutil
import copy
import subprocess
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import sleap_io as sio
from itertools import cycle
from omegaconf import DictConfig, OmegaConf
import lightning as L
import litdata as ld
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    XLAProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
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
from sleap_nn.data.instance_cropping import find_instance_crop_size
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.streaming_datasets import (
    BottomUpStreamingDataset,
    CenteredInstanceStreamingDataset,
    CentroidStreamingDataset,
    SingleInstanceStreamingDataset,
)
from loguru import logger
from sleap_nn.training.utils import (
    check_memory,
    is_distributed_initialized,
    get_dist_rank,
)
from sleap_nn.inference.utils import get_skeleton_from_config


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
    BottomUpLightningModule,
    CentroidLightningModule,
    TopDownCenteredInstanceLightningModule,
    SingleInstanceLightningModule,
)
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.callbacks import (
    ProgressReporterZMQ,
    TrainingControllerZMQ,
    MatplotlibSaver,
    WandBPredImageLogger,
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
        config: DictConfig,
    ):
        """Initialise the class with configs and set the seed and device as class attributes."""
        self.config = verify_training_cfg(config)
        self.data_pipeline_fw = self.config.data_config.data_pipeline_fw
        self.use_existing_imgs = self.config.data_config.use_existing_imgs
        self.user_instances_only = OmegaConf.select(
            self.config, "data_config.user_instances_only", default=True
        )

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
            or self.data_pipeline_fw == "torch_dataset_cache_img_memory"
            or self.data_pipeline_fw == "torch_dataset_cache_img_disk"
        ):
            self.train_dataset = None
            self.val_dataset = None
            self.cache_img_path = (
                Path(self.config.data_config.cache_img_path)
                if self.config.data_config.cache_img_path is not None
                else Path(self.dir_path)
            )
            # Get cache img path
            self.cache_img = (
                self.data_pipeline_fw.split("_")[-1]
                if "cache_img" in self.data_pipeline_fw
                else None
            )
            self.train_cache_img_path = Path(self.cache_img_path) / "train_imgs"
            self.val_cache_img_path = Path(self.cache_img_path) / "val_imgs"
            if self.use_existing_imgs:
                if not (
                    self.train_cache_img_path.exists()
                    and self.train_cache_img_path.is_dir()
                    and any(self.train_cache_img_path.glob("*.jpg"))
                ):
                    message = (
                        f"There are no images in the path: {self.train_cache_img_path}"
                    )
                    logger.error(message)
                    raise Exception(message)

                if not (
                    self.val_cache_img_path.exists()
                    and self.val_cache_img_path.is_dir()
                    and any(self.val_cache_img_path.glob("*.jpg"))
                ):
                    message = (
                        f"There are no images in the path: {self.val_cache_img_path}"
                    )
                    logger.error(message)
                    raise Exception(message)

        self.seed = self.config.trainer_config.seed
        self.steps_per_epoch = self.config.trainer_config.steps_per_epoch

        # initialize attributes
        self.model = None
        self.train_labels, self.val_labels = None, None

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

        self.max_stride = self.config.model_config.backbone_config[
            f"{self.backbone_type}"
        ]["max_stride"]

        if self.config.data_config.preprocessing.scale is None:
            self.config.data_config.preprocessing.scale = 1.0

        train_labels = []
        skeleton = sio.load_slp(self.config.data_config.train_labels_path[0]).skeletons[
            0
        ]

        for path in self.config.data_config.train_labels_path:
            skel_temp = self.config.data_config.train_labels_path[path]
            nodes_equal = [node.name for node in skeleton.nodes] == [
                node.name for node in skel_temp.nodes
            ]
            edge_inds_equal = [tuple(edge) for edge in skeleton.edge_inds] == [
                tuple(edge) for edge in skel_temp.edge_inds
            ]
            skeletons_equal = nodes_equal and edge_inds_equal
            if skeletons_equal:
                train_labels.append(sio.load_slp(skel_temp))
            else:
                message = f"The skeletons in the training labels {path} do not match the skeleton in the first training label file."
                logger.error(message)
                raise ValueError(message)

        val_labels_path = self.config.data_config.val_labels_path

        self.train_labels = []
        self.val_labels = []

        if val_labels_path is None:
            val_fraction = OmegaConf.select(
                self.config, "data_config.validation_fraction", default=0.1
            )
            for label in train_labels:
                temp_train_labels, temp_val_labels = label.make_training_splits(
                    n_train=1 - val_fraction, n_val=val_fraction
                )
                self.train_labels.append(temp_train_labels)
                self.val_labels.append(temp_val_labels)
        else:
            self.train_labels = train_labels
            for path in self.config.data_config.val_labels_path:
                self.val_labels.append(sio.load_slp(path))

        self.max_height, self.max_width = 0
        max_crop_size = 0
        for index, x in enumerate(self.train_labels):
            x.save(Path(self.dir_path) / f"labels_train_gt_{index}.slp")
            x.save(Path(self.dir_path) / f"labels_val_gt_{index}.slp")

            max_height, max_width = get_max_height_width(self.train_labels[x])

            if max_height > self.max_height:
                self.max_height = max_height
            if max_width > self.max_width:
                self.max_width = max_width

            crop_size = find_instance_crop_size(
                self.train_labels[label],
                maximum_stride=self.max_stride,
                min_crop_size=min_crop_size,
                input_scaling=self.config.data_config.preprocessing.scale,
            )
            if crop_size > max_crop_size:
                max_crop_size = crop_size

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

                self.crop_hw = max_crop_size
                self.config.data_config.preprocessing.crop_hw = (
                    self.crop_hw,
                    self.crop_hw,
                )
            else:
                self.crop_hw = self.crop_hw[0]

        self.skeletons = self.train_labels.skeletons
        # save the skeleton in the config
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

        self.anchor_ind = None
        if self.model_type in ["centroid", "centered_instance"]:
            nodes = self.skeletons[0].node_names
            anch_pt = self.config.model_config.head_configs[f"{self.model_type}"][
                "confmaps"
            ]["anchor_part"]
            self.anchor_ind = nodes.index(anch_pt) if anch_pt is not None else None

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

        self.edge_inds = self.train_labels.skeletons[0].edge_inds

        OmegaConf.save(config=self.config, f=f"{self.dir_path}/training_config.yaml")

    def _create_data_loaders_torch_dataset(self):
        """Create a torch DataLoader for train, validation and test sets using the data_config."""
        if self.data_pipeline_fw == "torch_dataset_cache_img_memory":
            train_cache_memory_final = 0
            val_cache_memory_final = 0
            for x in train_labels:
                train_cache_memory = check_memory(
                    self.train_labels[x],
                    max_hw=(self.max_height, self.max_width),
                    model_type=self.model_type,
                    input_scaling=self.config.data_config.preprocessing.scale,
                    crop_size=self.crop_hw if self.crop_hw != -1 else None,
                )
                val_cache_memory = check_memory(
                    self.val_labels[x],
                    max_hw=(self.max_height, self.max_width),
                    model_type=self.model_type,
                    input_scaling=self.config.data_config.preprocessing.scale,
                    crop_size=self.crop_hw if self.crop_hw != -1 else None,
                )
                train_cache_memory_final += train_cache_memory
                val_cache_memory_final += val_cache_memory

            total_cache_memory = train_cache_memory + val_cache_memory
            total_cache_memory += 0.1 * total_cache_memory  # memory required in bytes
            available_memory = (
                psutil.virtual_memory().available
            )  # available memory in bytes

            if total_cache_memory > available_memory:
                self.data_pipeline_fw = "torch_dataset_cache_img_disk"
                self.cache_img = "disk"
                self.train_torch_dataset_cache_img = Path("./train_imgs")
                self.val_torch_dataset_cache_img = Path("./train_imgs")
                logger.info(
                    f"Insufficient memory for in-memory caching. `jpg` files will be created."
                )

        if self.model_type == "bottomup":
            self.train_dataset = BottomUpDataset(
                labels=self.train_labels,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.max_stride,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=self.config.data_config.augmentation_config,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.train_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            self.val_dataset = BottomUpDataset(
                labels=self.val_labels,
                confmap_head_config=self.config.model_config.head_configs.bottomup.confmaps,
                pafs_head_config=self.config.model_config.head_configs.bottomup.pafs,
                max_stride=self.max_stride,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=None,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.val_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )

        elif self.model_type == "centered_instance":
            self.train_dataset = CenteredInstanceDataset(
                labels=self.train_labels,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                anchor_ind=self.anchor_ind,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=self.config.data_config.augmentation_config,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=self.config.data_config.use_augmentations_train,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.train_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            self.val_dataset = CenteredInstanceDataset(
                labels=self.val_labels,
                confmap_head_config=self.config.model_config.head_configs.centered_instance.confmaps,
                max_stride=self.max_stride,
                anchor_ind=self.anchor_ind,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=None,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=False,
                crop_hw=(self.crop_hw, self.crop_hw),
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.val_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )

        elif self.model_type == "centroid":
            self.train_dataset = CentroidDataset(
                labels=self.train_labels,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
                anchor_ind=self.anchor_ind,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=self.config.data_config.augmentation_config,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.train_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            self.val_dataset = CentroidDataset(
                labels=self.val_labels,
                confmap_head_config=self.config.model_config.head_configs.centroid.confmaps,
                max_stride=self.max_stride,
                anchor_ind=self.anchor_ind,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=None,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.val_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )

        elif self.model_type == "single_instance":
            self.train_dataset = SingleInstanceDataset(
                labels=self.train_labels,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=self.config.data_config.augmentation_config,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=self.config.data_config.use_augmentations_train,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.train_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            self.val_dataset = SingleInstanceDataset(
                labels=self.val_labels,
                confmap_head_config=self.config.model_config.head_configs.single_instance.confmaps,
                max_stride=self.max_stride,
                user_instances_only=self.config.data_config.user_instances_only,
                is_rgb=self.config.data_config.preprocessing.is_rgb,
                augmentation_config=None,
                scale=self.config.data_config.preprocessing.scale,
                apply_aug=False,
                max_hw=(self.max_height, self.max_width),
                cache_img=self.cache_img,
                cache_img_path=self.val_cache_img_path,
                use_existing_imgs=self.use_existing_imgs,
                rank=self.trainer.global_rank if self.trainer is not None else None,
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

        # If using caching, close the videos to prevent `h5py objects can't be pickled error` when num_workers > 0.
        if "cache_img" in self.data_pipeline_fw:
            for train, val in zip(self.train_labels, self.val_labels):
                for video in train.videos:
                    if video.is_open:
                        video.close()
                for video in val.videos:
                    if video.is_open:
                        video.close()

        # train
        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            # steps_per_epoch=self.steps_per_epoch,
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
        self.val_data_loader = DataLoader(
            dataset=self.val_dataset,
            # steps_per_epoch=val_steps_per_epoch if val_steps_per_epoch != 0 else 1,
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

        if not self.use_existing_imgs:
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
            "single_instance": SingleInstanceLightningModule,
            "centered_instance": TopDownCenteredInstanceLightningModule,
            "centroid": CentroidLightningModule,
            "bottomup": BottomUpLightningModule,
        }
        self.model = models[self.model_type](
            config=self.config,
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

        profilers = {
            "advanced": AdvancedProfiler(),
            "passthrough": PassThroughProfiler(),
            "pytorch": PyTorchProfiler(),
            "simple": SimpleProfiler(),
        }
        cfg_profiler = OmegaConf.select(
            self.config, "trainer_config.profiler", default=None
        )
        profiler = None
        if cfg_profiler is not None:
            if cfg_profiler in profilers:
                profiler = profilers[cfg_profiler]
            else:
                message = f"{cfg_profiler} is not a valid option. Please choose one of {list(profilers.keys())}"
                logger.error(message)
                raise ValueError(message)

        strategy = OmegaConf.select(
            self.config, "trainer_config.trainer_strategy", default="auto"
        )

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

        if self.data_pipeline_fw == "litdata":
            self._create_data_loaders_litdata()

        elif (
            self.data_pipeline_fw == "torch_dataset"
            or self.data_pipeline_fw == "torch_dataset_cache_img_memory"
            or self.data_pipeline_fw == "torch_dataset_cache_img_disk"
        ):
            self._create_data_loaders_torch_dataset()

        else:
            message = f"{self.data_pipeline_fw} is not a valid option. Please choose one of `litdata`, `torch_dataset`, `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`"
            logger.error(message)
            raise ValueError(message)

        if OmegaConf.select(
            self.config, "trainer_config.visualize_preds_during_training", default=False
        ):
            train_viz_pipeline = cycle(self.train_dataset)
            val_viz_pipeline = cycle(self.val_dataset)
            viz_dir = Path(self.dir_path) / "viz"
            if not Path(viz_dir).exists():
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                MatplotlibSaver(
                    save_folder=viz_dir,
                    plot_fn=lambda: self.model.visualize_example(
                        next(train_viz_pipeline)
                    ),
                    prefix="train",
                )
            )
            callbacks.append(
                MatplotlibSaver(
                    save_folder=viz_dir,
                    plot_fn=lambda: self.model.visualize_example(
                        next(val_viz_pipeline)
                    ),
                    prefix="validation",
                )
            )

            if self.model_type == "bottomup":
                train_viz_pipeline1 = cycle(copy.deepcopy(self.train_dataset))
                val_viz_pipeline1 = cycle(copy.deepcopy(self.val_dataset))
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.model.visualize_pafs_example(
                            next(train_viz_pipeline1)
                        ),
                        prefix="train_pafs_magnitude",
                    )
                )
                callbacks.append(
                    MatplotlibSaver(
                        save_folder=viz_dir,
                        plot_fn=lambda: self.model.visualize_pafs_example(
                            next(val_viz_pipeline1)
                        ),
                        prefix="validation_pafs_magnitude",
                    )
                )

            if self.config.trainer_config.use_wandb:
                callbacks.append(
                    WandBPredImageLogger(
                        viz_folder=viz_dir,
                        wandb_run_name=wandb_config.name,
                        is_bottomup=(self.model_type == "bottomup"),
                    )
                )

        if self.model_type == "bottomup":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Explicitly avoid using MPS
                self.config.trainer_config.trainer_accelerator = "cpu"

        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=training_loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=self.config.trainer_config.trainer_devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            limit_train_batches=self.steps_per_epoch,
            strategy=strategy,
            profiler=profiler,
            log_every_n_steps=1,
        )  # TODO check any other methods to use rank in dataset creations!

        # save the configs as yaml in the checkpoint dir
        if (
            self.trainer.global_rank == 0
        ):  # save config if there are no distributed process or the rank = 0
            OmegaConf.save(
                config=self.config, f=f"{self.dir_path}/training_config.yaml"
            )

        if self.config.trainer_config.use_wandb:
            if (
                self.trainer.global_rank == 0
            ):  # save config if there are no distributed process or the rank = 0
                wandb_logger.experiment.config.update({"run_name": wandb_config.name})
                wandb_logger.experiment.config.update(
                    {"run_config": OmegaConf.to_container(self.config, resolve=True)}
                )
                wandb_logger.experiment.config.update({"model_params": total_params})

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
                self.data_pipeline_fw == "torch_dataset_cache_img_disk"
                and self.config.data_config.delete_cache_imgs_after_training
            ):
                if (self.train_cache_img_path).exists():
                    shutil.rmtree(
                        (self.train_cache_img_path).as_posix(),
                        ignore_errors=True,
                    )

                if (self.val_cache_img_path).exists():
                    shutil.rmtree(
                        (self.val_cache_img_path).as_posix(),
                        ignore_errors=True,
                    )

            # TODO: (ubuntu test failing (running for > 6hrs) with the below lines)
            if (
                self.data_pipeline_fw == "litdata"
                and self.config.data_config.delete_cache_imgs_after_training
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
