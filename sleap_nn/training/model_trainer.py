"""This module is to train a sleap-nn model using Lightning."""

import os
import shutil
import attrs
import torch
import random
import numpy as np
import sleap_io as sio
import time
import lightning as L
import wandb
import yaml

from pathlib import Path
from typing import List, Optional
from datetime import datetime
from itertools import count
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from sleap_nn.data.utils import check_cache_memory
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
from sleap_io.io.skeleton import SkeletonYAMLEncoder
from sleap_nn.data.instance_cropping import (
    find_instance_crop_size,
    find_max_instance_bbox_size,
    compute_augmentation_padding,
)
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.custom_datasets import (
    get_train_val_dataloaders,
    get_steps_per_epoch,
    get_train_val_datasets,
    get_train_val_datasets_multi_head,
    get_train_val_dataloaders_multi_head,
)
from lightning.pytorch.utilities import CombinedLoader
from loguru import logger
from sleap_nn.config.utils import (
    get_backbone_type_from_cfg,
    get_model_type_from_cfg,
)
from sleap_nn.training.lightning_modules import LightningModel, MultiHeadLightningModel
from sleap_nn.config.utils import check_output_strides
from sleap_nn.training.utils import get_gpu_memory
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.callbacks import (
    ProgressReporterZMQ,
    TrainingControllerZMQ,
    CSVLoggerCallback,
    SleapProgressBar,
    EpochEndEvaluationCallback,
    CentroidEvaluationCallback,
    UnifiedVizCallback,
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
    train_labels: List[sio.Labels] = attrs.field(factory=list)
    val_labels: List[sio.Labels] = attrs.field(factory=list)
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
        # Verify config structure.
        config = verify_training_cfg(config)

        model_trainer = cls(config=config)

        model_trainer.model_type = get_model_type_from_cfg(model_trainer.config)
        model_trainer.backbone_type = get_backbone_type_from_cfg(model_trainer.config)

        if model_trainer.config.trainer_config.seed is not None:
            model_trainer._set_seed()

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

    def _set_seed(self):
        """Set seed for the current experiment."""
        seed = self.config.trainer_config.seed

        random.seed(seed)

        # torch
        torch.manual_seed(seed)

        # if cuda is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # lightning
        L.seed_everything(seed)

        # numpy
        np.random.seed(seed)

    def _get_trainer_devices(self):
        """Get trainer devices."""
        trainer_devices = (
            self.config.trainer_config.trainer_devices
            if self.config.trainer_config.trainer_devices is not None
            else "auto"
        )
        if (
            trainer_devices == "auto"
            and OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            is not None
        ):
            trainer_devices = len(
                OmegaConf.select(
                    self.config,
                    "trainer_config.trainer_device_indices",
                    default=None,
                )
            )
        elif trainer_devices == "auto":
            if torch.cuda.is_available():
                trainer_devices = torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                trainer_devices = 1
            elif torch.xpu.is_available():
                trainer_devices = torch.xpu.device_count()
            else:
                trainer_devices = 1
        return trainer_devices

    def _count_labeled_frames(
        self, labels_list: List[sio.Labels], user_only: bool = True
    ) -> int:
        """Count labeled frames, optionally filtering to user-labeled only.

        Args:
            labels_list: List of Labels objects to count frames from.
            user_only: If True, count only frames with user instances.

        Returns:
            Total count of labeled frames.
        """
        total = 0
        for label in labels_list:
            if user_only:
                total += sum(1 for lf in label if lf.has_user_instances)
            else:
                total += len(label)
        return total

    def _filter_to_user_labeled(self, labels: sio.Labels) -> sio.Labels:
        """Filter a Labels object to only include user-labeled frames.

        Args:
            labels: Labels object to filter.

        Returns:
            New Labels object containing only frames with user instances.
        """
        # Filter labeled frames to only those with user instances
        user_lfs = [lf for lf in labels if lf.has_user_instances]

        # Set instances to user instances only
        for lf in user_lfs:
            lf.instances = lf.user_instances

        # Create new Labels with filtered frames
        return sio.Labels(
            labeled_frames=user_lfs,
            videos=labels.videos,
            skeletons=labels.skeletons,
            tracks=labels.tracks,
            suggestions=labels.suggestions,
            provenance=labels.provenance,
        )

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

        # Check if we should count only user-labeled frames
        user_instances_only = OmegaConf.select(
            self.config, "data_config.user_instances_only", default=True
        )

        # check if all `.slp` file shave same skeleton structure (if multiple slp file paths are provided)
        skeleton = self.skeletons[0]
        for index, train_label in enumerate(labels):
            skel_temp = train_label.skeletons[0]
            skeletons_equal = skeleton.matches(skel_temp)
            if not skeletons_equal:
                message = f"The skeletons in the training labels: {index + 1} do not match the skeleton in the first training label file."
                logger.error(message)
                raise ValueError(message)

        # Check for same-data mode (train = val, for intentional overfitting)
        use_same = OmegaConf.select(
            self.config, "data_config.use_same_data_for_val", default=False
        )

        if use_same:
            # Same mode: use identical data for train and val (for overfitting)
            logger.info("Using same data for train and val (overfit mode)")
            self.train_labels = labels
            self.val_labels = labels
            total_train_lfs = self._count_labeled_frames(labels, user_instances_only)
            total_val_lfs = total_train_lfs
        elif val_labels is None or not len(val_labels):
            # if val labels are not provided, split from train
            val_fraction = OmegaConf.select(
                self.config, "data_config.validation_fraction", default=0.1
            )
            seed = (
                42
                if (
                    self.config.trainer_config.seed is None
                    and self._get_trainer_devices() > 1
                )
                else self.config.trainer_config.seed
            )
            for label in labels:
                train_split, val_split = label.make_training_splits(
                    n_train=1 - val_fraction, n_val=val_fraction, seed=seed
                )
                self.train_labels.append(train_split)
                self.val_labels.append(val_split)
                # make_training_splits returns only user-labeled frames
                total_train_lfs += len(train_split)
                total_val_lfs += len(val_split)
        else:
            self.train_labels = labels
            self.val_labels = val_labels
            total_train_lfs = self._count_labeled_frames(labels, user_instances_only)
            total_val_lfs = self._count_labeled_frames(val_labels, user_instances_only)

        logger.info(f"# Train Labeled frames: {total_train_lfs}")
        logger.info(f"# Val Labeled frames: {total_val_lfs}")

    def _setup_preprocessing_config(self):
        """Setup preprocessing config."""
        # compute max_heigt, max_width, and crop_size (if not provided in the config)
        max_height = self.config.data_config.preprocessing.max_height
        max_width = self.config.data_config.preprocessing.max_width
        if (
            self.model_type == "centered_instance"
            or self.model_type == "multi_class_topdown"
        ):
            crop_size = self.config.data_config.preprocessing.crop_size

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

            if (
                self.model_type == "centered_instance"
                or self.model_type == "multi_class_topdown"
            ):
                # compute crop size if not provided in config
                if crop_size is None:
                    # Get padding from config or auto-compute from augmentation settings
                    padding = self.config.data_config.preprocessing.crop_padding
                    if padding is None:
                        # Auto-compute padding based on augmentation settings
                        aug_config = self.config.data_config.augmentation_config
                        if (
                            self.config.data_config.use_augmentations_train
                            and aug_config is not None
                            and aug_config.geometric is not None
                        ):
                            geo = aug_config.geometric
                            # Check if rotation is enabled (via rotation_p or affine_p)
                            rotation_enabled = (
                                geo.rotation_p is not None and geo.rotation_p > 0
                            ) or (
                                geo.rotation_p is None
                                and geo.scale_p is None
                                and geo.translate_p is None
                                and geo.affine_p > 0
                            )
                            # Check if scale is enabled (via scale_p or affine_p)
                            scale_enabled = (
                                geo.scale_p is not None and geo.scale_p > 0
                            ) or (
                                geo.rotation_p is None
                                and geo.scale_p is None
                                and geo.translate_p is None
                                and geo.affine_p > 0
                            )

                            if rotation_enabled or scale_enabled:
                                # First find the actual max bbox size from labels
                                bbox_size = find_max_instance_bbox_size(train_label)
                                bbox_size = max(
                                    bbox_size,
                                    self.config.data_config.preprocessing.min_crop_size
                                    or 100,
                                )
                                rotation_max = (
                                    max(
                                        abs(geo.rotation_min),
                                        abs(geo.rotation_max),
                                    )
                                    if rotation_enabled
                                    else 0.0
                                )
                                scale_max = geo.scale_max if scale_enabled else 1.0
                                padding = compute_augmentation_padding(
                                    bbox_size=bbox_size,
                                    rotation_max=rotation_max,
                                    scale_max=scale_max,
                                )
                            else:
                                padding = 0
                        else:
                            padding = 0

                    crop_sz = find_instance_crop_size(
                        labels=train_label,
                        padding=padding,
                        maximum_stride=self.config.model_config.backbone_config[
                            f"{self.backbone_type}"
                        ]["max_stride"],
                        min_crop_size=self.config.data_config.preprocessing.min_crop_size,
                    )

                    if crop_sz > max_crop_size:
                        max_crop_size = crop_sz

        # if preprocessing params were None, replace with computed params
        if max_height is None or max_width is None:
            self.config.data_config.preprocessing.max_height = max_h
            self.config.data_config.preprocessing.max_width = max_w

        if (
            self.model_type == "centered_instance"
            or self.model_type == "multi_class_topdown"
        ) and crop_size is None:
            self.config.data_config.preprocessing.crop_size = max_crop_size

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
        # if run_name is None, assign a new dir name
        ckpt_dir = self.config.trainer_config.ckpt_dir
        if ckpt_dir is None or ckpt_dir == "" or ckpt_dir == "None":
            ckpt_dir = "."
            self.config.trainer_config.ckpt_dir = ckpt_dir
        run_name = self.config.trainer_config.run_name
        run_name_is_empty = run_name is None or run_name == "" or run_name == "None"

        # Validate: multi-GPU + disk cache requires explicit run_name
        if run_name_is_empty:
            is_disk_caching = (
                self.config.data_config.data_pipeline_fw
                == "torch_dataset_cache_img_disk"
            )
            num_devices = self._get_trainer_devices()

            if is_disk_caching and num_devices > 1:
                raise ValueError(
                    f"Multi-GPU training with disk caching requires an explicit `run_name`.\n\n"
                    f"Detected {num_devices} device(s) with "
                    f"`data_pipeline_fw='torch_dataset_cache_img_disk'`.\n"
                    f"Without an explicit run_name, each GPU worker generates a different "
                    f"timestamp-based directory, causing cache synchronization failures.\n\n"
                    f"Please provide a run_name using one of these methods:\n"
                    f"  - CLI: sleap-nn train config.yaml trainer_config.run_name=my_experiment\n"
                    f"  - Config file: Set `trainer_config.run_name: my_experiment`\n"
                    f"  - Python API: train(..., run_name='my_experiment')"
                )

            # Auto-generate timestamp-based run_name (safe for single GPU or non-disk-cache)
            sum_train_lfs = sum([len(train_label) for train_label in self.train_labels])
            sum_val_lfs = sum([len(val_label) for val_label in self.val_labels])
            run_name = (
                datetime.now().strftime("%y%m%d_%H%M%S")
                + f".{self.model_type}.n={sum_train_lfs + sum_val_lfs}"
            )

        # If checkpoint path already exists, add suffix to prevent overwriting
        if (Path(ckpt_dir) / run_name).exists() and (
            Path(ckpt_dir) / run_name / "best.ckpt"
        ).exists():
            logger.info(
                f"Checkpoint path already exists: {Path(ckpt_dir) / run_name}... adding suffix to prevent overwriting."
            )
            for i in count(1):
                new_run_name = f"{run_name}-{i}"
                if not (Path(ckpt_dir) / new_run_name).exists():
                    run_name = new_run_name
                    break

        self.config.trainer_config.run_name = run_name

        # set output dir for cache img
        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk":
            if self.config.data_config.cache_img_path is None:
                self.config.data_config.cache_img_path = (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
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
                    map_location="cpu",  # this will be loaded on cpu as it's just used to get the input channels
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
        logger.info("Setting up config...")

        # Normalize empty strings to None for optional wandb fields
        if self.config.trainer_config.wandb.prv_runid == "":
            self.config.trainer_config.wandb.prv_runid = None

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

        # if trainer_devices is None, set it to "auto"
        if self.config.trainer_config.trainer_devices is None:
            self.config.trainer_config.trainer_devices = (
                "auto"
                if OmegaConf.select(
                    self.config, "trainer_config.trainer_device_indices", default=None
                )
                is None
                else len(
                    OmegaConf.select(
                        self.config,
                        "trainer_config.trainer_device_indices",
                        default=None,
                    )
                )
            )

        # setup checkpoint path (generates run_name if not specified)
        self._setup_ckpt_path()

        # Default wandb run name to trainer run_name if not specified
        # Note: This must come after _setup_ckpt_path() which generates run_name
        if self.config.trainer_config.wandb.name is None:
            self.config.trainer_config.wandb.name = self.config.trainer_config.run_name

        # verify input_channels in model_config based on input image and pretrained model weights
        self._verify_model_input_channels()

    def _setup_model_ckpt_dir(self):
        """Create the model ckpt folder and save ground truth labels."""
        ckpt_path = (
            Path(self.config.trainer_config.ckpt_dir)
            / self.config.trainer_config.run_name
        ).as_posix()
        logger.info(f"Setting up model ckpt dir: `{ckpt_path}`...")

        # Only rank 0 (or non-distributed) should create directories and save files
        if RANK in [0, -1]:
            if not Path(ckpt_path).exists():
                try:
                    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    message = f"Cannot create a new folder in {ckpt_path}.\n {e}"
                    logger.error(message)
                    raise OSError(message)
            # Check if we should filter to user-labeled frames only
            user_instances_only = OmegaConf.select(
                self.config, "data_config.user_instances_only", default=True
            )

            # Save train and val ground truth labels
            for idx, (train, val) in enumerate(zip(self.train_labels, self.val_labels)):
                # Filter to user-labeled frames if needed (for evaluation)
                if user_instances_only:
                    train_filtered = self._filter_to_user_labeled(train)
                    val_filtered = self._filter_to_user_labeled(val)
                else:
                    train_filtered = train
                    val_filtered = val

                train_filtered.save(
                    Path(ckpt_path) / f"labels_gt.train.{idx}.slp",
                    restore_original_videos=False,
                )
                val_filtered.save(
                    Path(ckpt_path) / f"labels_gt.val.{idx}.slp",
                    restore_original_videos=False,
                )

            # Save test ground truth labels if test paths are provided
            test_file_path = OmegaConf.select(
                self.config, "data_config.test_file_path", default=None
            )
            if test_file_path is not None:
                # Normalize to list of strings
                if isinstance(test_file_path, str):
                    test_paths = [test_file_path]
                else:
                    test_paths = list(test_file_path)

                for idx, test_path in enumerate(test_paths):
                    # Only save if it's a .slp file (not a video file)
                    if test_path.endswith(".slp") or test_path.endswith(".pkg.slp"):
                        try:
                            test_labels = sio.load_slp(test_path)
                            if user_instances_only:
                                test_filtered = self._filter_to_user_labeled(
                                    test_labels
                                )
                            else:
                                test_filtered = test_labels
                            test_filtered.save(
                                Path(ckpt_path) / f"labels_gt.test.{idx}.slp",
                                restore_original_videos=False,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not save test ground truth for {test_path}: {e}"
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
            # Account for DataLoader worker memory overhead
            train_num_workers = self.config.trainer_config.train_data_loader.num_workers
            val_num_workers = self.config.trainer_config.val_data_loader.num_workers
            max_num_workers = max(train_num_workers, val_num_workers)

            mem_available = check_cache_memory(
                self.train_labels,
                self.val_labels,
                memory_buffer=MEMORY_BUFFER,
                num_workers=max_num_workers,
            )
            if not mem_available:
                # Validate: multi-GPU + auto-generated run_name + fallback to disk cache
                original_run_name = self._initial_config.trainer_config.run_name
                run_name_was_auto = (
                    original_run_name is None
                    or original_run_name == ""
                    or original_run_name == "None"
                )
                if run_name_was_auto and self.trainer.num_devices > 1:
                    raise ValueError(
                        f"Memory caching failed and disk caching fallback requires an "
                        f"explicit `run_name` for multi-GPU training.\n\n"
                        f"Detected {self.trainer.num_devices} device(s) with insufficient "
                        f"memory for in-memory caching.\n"
                        f"Without an explicit run_name, each GPU worker generates a different "
                        f"timestamp-based directory, causing cache synchronization failures.\n\n"
                        f"Please provide a run_name using one of these methods:\n"
                        f"  - CLI: sleap-nn train config.yaml trainer_config.run_name=my_experiment\n"
                        f"  - Config file: Set `trainer_config.run_name: my_experiment`\n"
                        f"  - Python API: train(..., run_name='my_experiment')\n\n"
                        f"Alternatively, use `data_pipeline_fw='torch_dataset'` to disable caching."
                    )

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
                else Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
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
                dirpath=(
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                ).as_posix(),
                filename="best",
                monitor="val/loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

            # csv log callback
            csv_log_keys = [
                "epoch",
                "train/loss",
                "val/loss",
                "learning_rate",
                "train/time",
                "val/time",
            ]
            # Add model-specific keys for wandb parity
            if self.model_type in [
                "single_instance",
                "centered_instance",
                "multi_class_topdown",
            ]:
                csv_log_keys.extend(
                    [f"train/confmaps/{name}" for name in self.skeletons[0].node_names]
                )
            if self.model_type == "bottomup":
                csv_log_keys.extend(
                    [
                        "train/confmaps_loss",
                        "train/paf_loss",
                        "val/confmaps_loss",
                        "val/paf_loss",
                    ]
                )
            if self.model_type == "multi_class_bottomup":
                csv_log_keys.extend(
                    [
                        "train/confmaps_loss",
                        "train/classmap_loss",
                        "train/class_accuracy",
                        "val/confmaps_loss",
                        "val/classmap_loss",
                        "val/class_accuracy",
                    ]
                )
            if self.model_type == "multi_class_topdown":
                csv_log_keys.extend(
                    [
                        "train/confmaps_loss",
                        "train/classvector_loss",
                        "train/class_accuracy",
                        "val/confmaps_loss",
                        "val/classvector_loss",
                        "val/class_accuracy",
                    ]
                )
            csv_logger = CSVLoggerCallback(
                filepath=Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
                / "training_log.csv",
                keys=csv_log_keys,
            )
            callbacks.append(csv_logger)

        if self.config.trainer_config.early_stopping.stop_training_on_plateau:
            # early stopping callback
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
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
                save_dir=(
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                ).as_posix(),
                id=self.config.trainer_config.wandb.prv_runid,
                group=self.config.trainer_config.wandb.group,
            )
            loggers.append(wandb_logger)

            # Log message about wandb local logs cleanup
            should_delete_wandb_logs = wandb_config.delete_local_logs is True or (
                wandb_config.delete_local_logs is None
                and wandb_config.wandb_mode != "offline"
            )
            if should_delete_wandb_logs:
                logger.info(
                    "WandB local logs will be deleted after training completes. "
                    "To keep logs, set trainer_config.wandb.delete_local_logs=false"
                )

            # save the configs as yaml in the checkpoint dir
            # Mask API key in both configs to prevent saving to disk
            self.config.trainer_config.wandb.api_key = ""
            if self._initial_config is not None:
                self._initial_config.trainer_config.wandb.api_key = ""

        # zmq callbacks
        if self.config.trainer_config.zmq.controller_port is not None:
            controller_address = "tcp://127.0.0.1:" + str(
                self.config.trainer_config.zmq.controller_port
            )
            callbacks.append(TrainingControllerZMQ(address=controller_address))
        if self.config.trainer_config.zmq.publish_port is not None:
            publish_address = "tcp://127.0.0.1:" + str(
                self.config.trainer_config.zmq.publish_port
            )
            callbacks.append(ProgressReporterZMQ(address=publish_address))

        # viz callbacks - use unified callback for all visualization outputs
        if self.config.trainer_config.visualize_preds_during_training:
            viz_dir = (
                Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
                / "viz"
            )
            if not Path(viz_dir).exists():
                if RANK in [0, -1]:
                    Path(viz_dir).mkdir(parents=True, exist_ok=True)

            # Get wandb viz config options
            log_wandb = self.config.trainer_config.use_wandb and OmegaConf.select(
                self.config, "trainer_config.wandb.save_viz_imgs_wandb", default=False
            )
            wandb_modes = []
            if log_wandb:
                if OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_enabled", default=True
                ):
                    wandb_modes.append("direct")
                if OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_boxes", default=False
                ):
                    wandb_modes.append("boxes")
                if OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_masks", default=False
                ):
                    wandb_modes.append("masks")

            # Single unified callback handles all visualization outputs
            callbacks.append(
                UnifiedVizCallback(
                    model_trainer=self,
                    train_dataset=viz_train_dataset,
                    val_dataset=viz_val_dataset,
                    model_type=self.model_type,
                    save_local=self.config.trainer_config.save_ckpt,
                    local_save_dir=viz_dir,
                    log_wandb=log_wandb,
                    wandb_modes=wandb_modes if wandb_modes else ["direct"],
                    wandb_box_size=OmegaConf.select(
                        self.config, "trainer_config.wandb.viz_box_size", default=5.0
                    ),
                    wandb_confmap_threshold=OmegaConf.select(
                        self.config,
                        "trainer_config.wandb.viz_confmap_threshold",
                        default=0.1,
                    ),
                    log_wandb_table=OmegaConf.select(
                        self.config, "trainer_config.wandb.log_viz_table", default=False
                    ),
                )
            )

        # Add custom progress bar with better metric formatting
        if self.config.trainer_config.enable_progress_bar:
            callbacks.append(SleapProgressBar())

        # Add epoch-end evaluation callback if enabled
        if self.config.trainer_config.eval.enabled:
            if self.model_type == "centroid":
                # Use centroid-specific evaluation with distance-based metrics
                callbacks.append(
                    CentroidEvaluationCallback(
                        videos=self.val_labels[0].videos,
                        eval_frequency=self.config.trainer_config.eval.frequency,
                        match_threshold=self.config.trainer_config.eval.match_threshold,
                    )
                )
            else:
                # Use standard OKS/PCK evaluation for pose models
                callbacks.append(
                    EpochEndEvaluationCallback(
                        skeleton=self.skeletons[0],
                        videos=self.val_labels[0].videos,
                        eval_frequency=self.config.trainer_config.eval.frequency,
                        oks_stddev=self.config.trainer_config.eval.oks_stddev,
                        oks_scale=self.config.trainer_config.eval.oks_scale,
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

        devices = (
            OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            if OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            is not None
            else self.config.trainer_config.trainer_devices
        )
        logger.info(f"Trainer devices: {devices}")

        # if trainer devices is set to less than the number of available GPUs, use the least used GPUs
        if (
            torch.cuda.is_available()
            and self.config.trainer_config.trainer_accelerator != "cpu"
            and isinstance(self.config.trainer_config.trainer_devices, int)
            and self.config.trainer_config.trainer_devices < torch.cuda.device_count()
            and self.config.trainer_config.trainer_device_indices is None
        ):
            devices = [
                int(x)
                for x in np.argsort(get_gpu_memory())[::-1][
                    : self.config.trainer_config.trainer_devices
                ]
            ]
            # Sort device indices in ascending order for NCCL compatibility.
            # NCCL expects devices in consistent ascending order across ranks
            # to properly set up communication rings. Without sorting, DDP may
            # assign multiple ranks to the same GPU, causing "Duplicate GPU detected" errors.
            devices.sort()
            logger.info(f"Using GPUs with most available memory: {devices}")

        # create lightning.Trainer instance.
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=devices,
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

        # Barrier after dataset creation to ensure all workers wait for disk caching
        # (rank 0 caches to disk, others must wait before reading cached files)
        self.trainer.strategy.barrier()

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

        logger.info(f"Training on {self.trainer.num_devices} device(s)")
        logger.info(f"Training on {self.trainer.strategy.root_device} accelerator")

        # initialize the lightning model.
        # need to initialize after Trainer is initialized (for trainer accelerator)
        logger.info(f"Setting up lightning module for {self.model_type} model...")
        self.lightning_model = LightningModel.get_lightning_model_from_config(
            config=self.config,
        )
        logger.info(f"Backbone model: {self.lightning_model.model.backbone}")
        logger.info(f"Head model: {self.lightning_model.model.head_layers}")
        total_params = sum(p.numel() for p in self.lightning_model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
        self.config.model_config.total_params = total_params

        # setup dataloaders
        # need to set up dataloaders after Trainer is initialized (for ddp). DistributedSampler depends on the rank
        logger.info(
            f"Input image shape: {train_dataset[0]['image'].shape if 'image' in train_dataset[0] else train_dataset[0]['instance_image'].shape}"
        )
        train_dataloader, val_dataloader = get_train_val_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            rank=self.trainer.global_rank,
            train_steps_per_epoch=self.config.trainer_config.train_steps_per_epoch,
            val_steps_per_epoch=val_steps_per_epoch,
            trainer_devices=self.trainer.num_devices,
        )

        if self.trainer.global_rank == 0:  # save config only in rank 0 process
            ckpt_path = (
                Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
            ).as_posix()
            OmegaConf.save(
                self._initial_config,
                (Path(ckpt_path) / "initial_config.yaml").as_posix(),
            )

            if self.config.trainer_config.use_wandb:
                if wandb.run is None:
                    wandb.init(
                        dir=(
                            Path(self.config.trainer_config.ckpt_dir)
                            / self.config.trainer_config.run_name
                        ).as_posix(),
                        project=self.config.trainer_config.wandb.project,
                        entity=self.config.trainer_config.wandb.entity,
                        name=self.config.trainer_config.wandb.name,
                        id=self.config.trainer_config.wandb.prv_runid,
                        group=self.config.trainer_config.wandb.group,
                    )

                # Define custom x-axes for wandb metrics
                # Epoch-level metrics use epoch as x-axis, step-level use default global_step
                wandb.define_metric("epoch")

                # Training metrics (train/ prefix for grouping) - all use epoch x-axis
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("train/confmaps/*", step_metric="epoch")

                # Validation metrics (val/ prefix for grouping)
                wandb.define_metric("val/*", step_metric="epoch")

                # Evaluation metrics (eval/ prefix for grouping)
                wandb.define_metric("eval/*", step_metric="epoch")

                # Visualization images (need explicit nested paths)
                wandb.define_metric("viz/*", step_metric="epoch")
                wandb.define_metric("viz/train/*", step_metric="epoch")
                wandb.define_metric("viz/val/*", step_metric="epoch")

                self.config.trainer_config.wandb.current_run_id = wandb.run.id
                wandb.config["run_name"] = self.config.trainer_config.wandb.name
                wandb.config["run_config"] = OmegaConf.to_container(
                    self.config, resolve=True
                )

            OmegaConf.save(
                self.config,
                (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                    / "training_config.yaml"
                ).as_posix(),
            )

        self.trainer.strategy.barrier()

        # Flag to track if training was interrupted (not completed normally)
        training_interrupted = False

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
            training_interrupted = True

        finally:
            logger.info(
                f"Finished training loop. [{(time.time() - start_train_time) / 60:.1f} min]"
            )
            # Note: wandb.finish() is called in train.py after post-training evaluation

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
                    viz_dir = (
                        Path(self.config.trainer_config.ckpt_dir)
                        / self.config.trainer_config.run_name
                        / "viz"
                    )
                    if viz_dir.exists():
                        logger.info(f"Deleting viz folder at {viz_dir}...")
                        shutil.rmtree(viz_dir, ignore_errors=True)

            # Clean up entire run folder if training was interrupted (KeyboardInterrupt)
            if training_interrupted and self.trainer.global_rank == 0:
                run_dir = (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                )
                if run_dir.exists():
                    logger.info(
                        f"Training canceled - cleaning up run folder at {run_dir}..."
                    )
                    shutil.rmtree(run_dir, ignore_errors=True)


# =============================================================================
# Multi-Head Model Trainer
# =============================================================================


class MultiHeadModelTrainerAdapter:
    """Adapter that wraps a MultiHeadModelTrainer for a specific dataset index.

    This adapter is used with UnifiedVizCallback to provide dataset-specific
    visualization support for multi-head training. It intercepts calls to
    lightning_model.get_visualization_data() and passes the appropriate dataset_idx.

    Attributes:
        model_trainer: The MultiHeadModelTrainer instance.
        dataset_idx: The dataset index to use for visualization.
    """

    def __init__(self, model_trainer, dataset_idx: int):
        """Initialize the adapter.

        Args:
            model_trainer: The MultiHeadModelTrainer instance.
            dataset_idx: The dataset index (head) to use for visualization.
        """
        self._model_trainer = model_trainer
        self._dataset_idx = dataset_idx

    @property
    def lightning_model(self):
        """Return a wrapped lightning model that passes dataset_idx to get_visualization_data."""
        return _MultiHeadLightningModelAdapter(
            self._model_trainer.lightning_model, self._dataset_idx
        )


class _MultiHeadLightningModelAdapter:
    """Internal adapter that wraps a multi-head lightning model for a specific dataset.

    This adapter ensures that get_visualization_data() is called with the correct
    dataset_idx parameter.
    """

    def __init__(self, lightning_model, dataset_idx: int):
        self._model = lightning_model
        self._dataset_idx = dataset_idx

    def get_visualization_data(self, sample, **kwargs):
        """Get visualization data for the specific dataset/head.

        Args:
            sample: A sample from the visualization dataset.
            **kwargs: Additional arguments passed to the underlying method.

        Returns:
            VisualizationData object for the specified dataset/head.
        """
        return self._model.get_visualization_data(
            sample, dataset_idx=self._dataset_idx, **kwargs
        )


@attrs.define
class MultiHeadModelTrainer:
    """Train a multi-head sleap-nn model using PyTorch Lightning.

    This class handles training with multiple datasets, where each dataset has its own
    head while sharing a common backbone. It mirrors the features of ModelTrainer but
    adapted for multi-head/multi-dataset scenarios.

    Args:
        config: OmegaConf dictionary with data_config, model_config, and trainer_config.
            For multi-head, preprocessing params (max_height, max_width, crop_size, scale)
            should be dicts keyed by dataset index.
        train_labels: List of `sio.Labels` objects for each training dataset.
        val_labels: List of `sio.Labels` objects for each validation dataset.
        skeletons: List of `sio.Skeleton` objects, one per dataset.
        lightning_model: Multi-head Lightning model instance.
        model_type: Type of the model (centered_instance, centroid, etc.).
        backbone_type: Backbone model type (unet, convnext, swint).
        trainer: Instance of the Lightning Trainer.
    """

    config: DictConfig
    _initial_config: Optional[DictConfig] = None
    train_labels: List[sio.Labels] = attrs.field(factory=list)
    val_labels: List[sio.Labels] = attrs.field(factory=list)
    skeletons: Optional[List[sio.Skeleton]] = None

    lightning_model: Optional[MultiHeadLightningModel] = None
    model_type: Optional[str] = None
    backbone_type: Optional[str] = None

    _profilers: dict = {
        "advanced": AdvancedProfiler(),
        "passthrough": PassThroughProfiler(),
        "pytorch": PyTorchProfiler(),
        "simple": SimpleProfiler(),
    }

    trainer: Optional[L.Trainer] = None

    # Whether to automatically compute dataset loss weights based on dataset sizes.
    # When True, smaller datasets (repeated more often with max_size_cycle) get lower
    # weights and larger datasets get higher weights to compensate for sampling frequency.
    apply_dataset_loss_weights: bool = False

    def _dataset_name(self, d_idx: int) -> str:
        """Get a human-readable name for a dataset index.

        Uses the ``dataset_mapper`` config field when available, otherwise
        falls back to ``"dataset_{d_idx}"``.
        """
        mapper = OmegaConf.select(self.config, "dataset_mapper", default=None)
        if mapper is not None and d_idx in mapper:
            return mapper[d_idx]
        return f"dataset_{d_idx}"

    @classmethod
    def get_model_trainer_from_config(
        cls,
        config: DictConfig,
        train_labels: Optional[List[sio.Labels]] = None,
        val_labels: Optional[List[sio.Labels]] = None,
        apply_dataset_loss_weights: bool = False,
    ):
        """Create a model trainer instance from config.

        Args:
            config: Training job configuration with multi-head structure.
            train_labels: Optional list of training Labels objects (one per dataset).
            val_labels: Optional list of validation Labels objects (one per dataset).
            apply_dataset_loss_weights: If True, automatically compute loss weights based on
                dataset sizes to compensate for max_size_cycle sampling. Smaller datasets
                (repeated more) get lower weights, larger datasets get higher weights.

        Returns:
            MultiHeadModelTrainer instance with config fully set up.
        """
        logger.info("Initializing MultiHeadModelTrainer from config...")
        model_trainer = cls(config=config)
        model_trainer.apply_dataset_loss_weights = apply_dataset_loss_weights

        model_trainer.model_type = get_model_type_from_cfg(model_trainer.config)
        model_trainer.backbone_type = get_backbone_type_from_cfg(model_trainer.config)
        logger.info(f"Model type: {model_trainer.model_type}")
        logger.info(f"Backbone type: {model_trainer.backbone_type}")
        logger.info(f"Apply dataset loss weights: {apply_dataset_loss_weights}")

        if model_trainer.config.trainer_config.seed is not None:
            logger.info(f"Setting seed: {model_trainer.config.trainer_config.seed}")
            model_trainer._set_seed()

        if train_labels is None and val_labels is None:
            # Read labels from paths provided in the config (dict keyed by dataset index)
            train_labels_path = model_trainer.config.data_config.train_labels_path
            val_labels_path = model_trainer.config.data_config.val_labels_path

            num_datasets = len(train_labels_path) if isinstance(train_labels_path, (dict, DictConfig, list)) else 1
            logger.info(f"Loading {num_datasets} dataset(s) from config paths...")

            # Handle dict-style paths (keyed by dataset index)
            if isinstance(train_labels_path, (dict, DictConfig)):
                train_labels = []
                for idx in sorted(train_labels_path.keys()):
                    logger.info(f"  Loading train labels [{idx}]: {train_labels_path[idx]}")
                    train_labels.append(sio.load_slp(train_labels_path[idx]))
            else:
                train_labels = []
                for i, path in enumerate(train_labels_path):
                    logger.info(f"  Loading train labels [{i}]: {path}")
                    train_labels.append(sio.load_slp(path))

            if val_labels_path is not None:
                if isinstance(val_labels_path, (dict, DictConfig)):
                    val_labels = []
                    for idx in sorted(val_labels_path.keys()):
                        logger.info(f"  Loading val labels [{idx}]: {val_labels_path[idx]}")
                        val_labels.append(sio.load_slp(val_labels_path[idx]))
                else:
                    val_labels = []
                    for i, path in enumerate(val_labels_path):
                        logger.info(f"  Loading val labels [{i}]: {path}")
                        val_labels.append(sio.load_slp(path))
            else:
                val_labels = None
                logger.info("  No val labels paths provided; will split from train.")

            model_trainer._setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )
        else:
            logger.info("Using pre-loaded train/val labels objects.")
            model_trainer._setup_train_val_labels(
                labels=train_labels, val_labels=val_labels
            )

        # Log per-dataset summary
        for i, tl in enumerate(model_trainer.train_labels):
            name = model_trainer._dataset_name(i)
            skeleton = tl.skeletons[0] if tl.skeletons else None
            node_names = [n.name for n in skeleton.nodes] if skeleton else []
            logger.info(
                f"  Dataset {i} ({name}): "
                f"{len(tl)} train LFs, "
                f"{len(model_trainer.val_labels[i])} val LFs, "
                f"{len(node_names)} nodes {node_names}"
            )

        model_trainer._initial_config = model_trainer.config.copy()

        # Update config parameters
        logger.info("Running setup_config() to finalize preprocessing and head configs...")
        model_trainer.setup_config()

        # Check if all videos exist across all labels
        logger.info("Checking that all video files exist...")
        all_videos_exist = all(
            video.exists(check_all=True)
            for labels in [*model_trainer.train_labels, *model_trainer.val_labels]
            for video in labels.videos
        )

        if not all_videos_exist:
            raise FileNotFoundError(
                "One or more video files do not exist or are not accessible."
            )

        logger.info("MultiHeadModelTrainer initialization complete.")
        return model_trainer

    def _set_seed(self):
        """Set seed for the current experiment."""
        seed = self.config.trainer_config.seed

        random.seed(seed)

        # torch
        torch.manual_seed(seed)

        # if cuda is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # lightning
        L.seed_everything(seed)

        # numpy
        np.random.seed(seed)

    def _get_trainer_devices(self):
        """Get trainer devices."""
        trainer_devices = (
            self.config.trainer_config.trainer_devices
            if self.config.trainer_config.trainer_devices is not None
            else "auto"
        )
        if (
            trainer_devices == "auto"
            and OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            is not None
        ):
            trainer_devices = len(
                OmegaConf.select(
                    self.config,
                    "trainer_config.trainer_device_indices",
                    default=None,
                )
            )
        elif trainer_devices == "auto":
            if torch.cuda.is_available():
                trainer_devices = torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                trainer_devices = 1
            elif torch.xpu.is_available():
                trainer_devices = torch.xpu.device_count()
            else:
                trainer_devices = 1
        return trainer_devices

    def _count_labeled_frames(
        self, labels_list: List[sio.Labels], user_only: bool = True
    ) -> int:
        """Count labeled frames, optionally filtering to user-labeled only.

        Args:
            labels_list: List of Labels objects to count frames from.
            user_only: If True, count only frames with user instances.

        Returns:
            Total count of labeled frames.
        """
        total = 0
        for label in labels_list:
            if user_only:
                total += sum(1 for lf in label if lf.has_user_instances)
            else:
                total += len(label)
        return total

    def _filter_to_user_labeled(self, labels: sio.Labels) -> sio.Labels:
        """Filter a Labels object to only include user-labeled frames.

        Args:
            labels: Labels object to filter.

        Returns:
            New Labels object containing only frames with user instances.
        """
        # Filter labeled frames to only those with user instances
        user_lfs = [lf for lf in labels if lf.has_user_instances]

        # Set instances to user instances only
        for lf in user_lfs:
            lf.instances = lf.user_instances

        # Create new Labels with filtered frames
        return sio.Labels(
            labeled_frames=user_lfs,
            videos=labels.videos,
            skeletons=labels.skeletons,
            tracks=labels.tracks,
            suggestions=labels.suggestions,
            provenance=labels.provenance,
        )

    def _setup_train_val_labels(
        self,
        labels: Optional[List[sio.Labels]] = None,
        val_labels: Optional[List[sio.Labels]] = None,
    ):
        """Create train and val labels objects for multi-head training.

        For multi-head, each dataset can have a different skeleton.

        Args:
            labels: List of training Labels objects, one per dataset.
            val_labels: List of validation Labels objects, one per dataset.
        """
        logger.info(f"Creating train-val split for {len(labels)} dataset(s)...")
        total_train_lfs = 0
        total_val_lfs = 0

        # Load all skeletons from the labels (one per dataset)
        self.skeletons = [label.skeletons[0] for label in labels]

        # Check if we should count only user-labeled frames
        user_instances_only = OmegaConf.select(
            self.config, "data_config.user_instances_only", default=True
        )

        # Check for same-data mode (train = val, for intentional overfitting)
        use_same = OmegaConf.select(
            self.config, "data_config.use_same_data_for_val", default=False
        )

        if use_same:
            # Same mode: use identical data for train and val (for overfitting)
            logger.info("Using same data for train and val (overfit mode)")
            self.train_labels = labels
            self.val_labels = labels
            total_train_lfs = self._count_labeled_frames(labels, user_instances_only)
            total_val_lfs = total_train_lfs
        elif val_labels is None or not len(val_labels):
            # If val labels are not provided, split from train
            val_fraction = OmegaConf.select(
                self.config, "data_config.validation_fraction", default=0.1
            )
            seed = (
                42
                if (
                    self.config.trainer_config.seed is None
                    and self._get_trainer_devices() > 1
                )
                else self.config.trainer_config.seed
            )
            for label in labels:
                train_split, val_split = label.make_training_splits(
                    n_train=1 - val_fraction, n_val=val_fraction, seed=seed
                )
                self.train_labels.append(train_split)
                self.val_labels.append(val_split)
                # make_training_splits returns only user-labeled frames
                total_train_lfs += len(train_split)
                total_val_lfs += len(val_split)
        else:
            self.train_labels = labels
            self.val_labels = val_labels
            total_train_lfs = self._count_labeled_frames(labels, user_instances_only)
            total_val_lfs = self._count_labeled_frames(val_labels, user_instances_only)

        logger.info(f"# Datasets: {len(self.train_labels)}")
        logger.info(f"# Train Labeled frames (total): {total_train_lfs}")
        logger.info(f"# Val Labeled frames (total): {total_val_lfs}")
        for idx, (train_l, val_l) in enumerate(
            zip(self.train_labels, self.val_labels)
        ):
            logger.info(
                f"  Dataset {idx}: {len(train_l)} train, {len(val_l)} val, "
                f"skeleton={self.skeletons[idx].name} ({len(self.skeletons[idx].nodes)} nodes)"
            )

    def _setup_preprocessing_config(self):
        """Setup preprocessing config for multi-head (per-dataset indexed values).

        For multi-head training, preprocessing params like max_height, max_width,
        crop_size, and scale are dicts keyed by dataset index.
        """
        # Get preprocessing config (should be dicts for multi-head)
        preprocessing = self.config.data_config.preprocessing
        max_height = preprocessing.max_height
        max_width = preprocessing.max_width

        # Ensure we have dict-style config for multi-head
        if not isinstance(max_height, (dict, DictConfig)):
            # Convert scalar to per-dataset dict
            max_height = {idx: max_height for idx in range(len(self.train_labels))}
        if not isinstance(max_width, (dict, DictConfig)):
            max_width = {idx: max_width for idx in range(len(self.train_labels))}

        # Handle crop_size for centered_instance models
        if (
            self.model_type == "centered_instance"
            or self.model_type == "multi_class_topdown"
        ):
            crop_size = preprocessing.crop_size
            if not isinstance(crop_size, (dict, DictConfig)):
                crop_size = {idx: crop_size for idx in range(len(self.train_labels))}

            scale = preprocessing.scale
            if not isinstance(scale, (dict, DictConfig)):
                scale = {idx: scale for idx in range(len(self.train_labels))}

        # Compute per-dataset preprocessing values
        for idx, train_label in enumerate(self.train_labels):
            # Compute max h and w from slp file if not provided
            if max_height.get(idx) is None or max_width.get(idx) is None:
                h, w = get_max_height_width(train_label)
                if max_height.get(idx) is None:
                    max_height[idx] = h
                if max_width.get(idx) is None:
                    max_width[idx] = w
                logger.info(
                    f"Dataset {idx}: computed max_height={h}, max_width={w}"
                )

            if (
                self.model_type == "centered_instance"
                or self.model_type == "multi_class_topdown"
            ):
                # Compute crop size if not provided in config
                if crop_size.get(idx) is None:
                    # Get padding from config or auto-compute from augmentation settings
                    padding = preprocessing.crop_padding
                    if padding is None:
                        # Auto-compute padding based on augmentation settings
                        aug_config = self.config.data_config.augmentation_config
                        if (
                            self.config.data_config.use_augmentations_train
                            and aug_config is not None
                            and aug_config.geometric is not None
                        ):
                            geo = aug_config.geometric
                            # Check if rotation is enabled
                            rotation_enabled = (
                                geo.rotation_p is not None and geo.rotation_p > 0
                            ) or (
                                geo.rotation_p is None
                                and geo.scale_p is None
                                and geo.translate_p is None
                                and geo.affine_p > 0
                            )
                            # Check if scale is enabled
                            scale_enabled = (
                                geo.scale_p is not None and geo.scale_p > 0
                            ) or (
                                geo.rotation_p is None
                                and geo.scale_p is None
                                and geo.translate_p is None
                                and geo.affine_p > 0
                            )

                            if rotation_enabled or scale_enabled:
                                # Find the actual max bbox size from labels
                                bbox_size = find_max_instance_bbox_size(train_label)
                                bbox_size = max(
                                    bbox_size,
                                    preprocessing.min_crop_size or 100,
                                )
                                rotation_max = (
                                    max(
                                        abs(geo.rotation_min),
                                        abs(geo.rotation_max),
                                    )
                                    if rotation_enabled
                                    else 0.0
                                )
                                scale_max = geo.scale_max if scale_enabled else 1.0
                                padding = compute_augmentation_padding(
                                    bbox_size=bbox_size,
                                    rotation_max=rotation_max,
                                    scale_max=scale_max,
                                )
                            else:
                                padding = 0
                        else:
                            padding = 0

                    crop_sz = find_instance_crop_size(
                        labels=train_label,
                        padding=padding,
                        maximum_stride=self.config.model_config.backbone_config[
                            f"{self.backbone_type}"
                        ]["max_stride"],
                        min_crop_size=preprocessing.min_crop_size,
                    )
                    crop_size[idx] = crop_sz
                    logger.info(
                        f"Dataset {idx}: computed crop_size={crop_sz} (padding={padding})"
                    )

        # Update config with computed values
        self.config.data_config.preprocessing.max_height = max_height
        self.config.data_config.preprocessing.max_width = max_width
        if (
            self.model_type == "centered_instance"
            or self.model_type == "multi_class_topdown"
        ):
            self.config.data_config.preprocessing.crop_size = crop_size

    def _setup_head_config(self):
        """Setup node, edge and class names in head config for multi-head.

        For multi-head, head_configs are structured as:
        head_configs[model_type][head_type][dataset_idx] = {...}
        """
        head_config = self.config.model_config.head_configs[self.model_type]

        for d_idx in range(len(self.train_labels)):
            for key in head_config:
                # Access per-dataset head config
                if d_idx not in head_config[key]:
                    continue

                dataset_head_config = head_config[key][d_idx]

                if "part_names" in dataset_head_config:
                    if dataset_head_config["part_names"] is None:
                        self.config.model_config.head_configs[self.model_type][key][
                            d_idx
                        ]["part_names"] = self.skeletons[d_idx].node_names
                        name = self._dataset_name(d_idx)
                        logger.info(
                            f"Dataset {name}: set part_names from skeleton "
                            f"({len(self.skeletons[d_idx].node_names)} nodes)"
                        )

                if "edges" in dataset_head_config:
                    if dataset_head_config["edges"] is None:
                        edges = [
                            (x.source.name, x.destination.name)
                            for x in self.skeletons[d_idx].edges
                        ]
                        self.config.model_config.head_configs[self.model_type][key][
                            d_idx
                        ]["edges"] = edges
                        name = self._dataset_name(d_idx)
                        logger.info(
                            f"Dataset {name}: set edges from skeleton ({len(edges)} edges)"
                        )

                if "classes" in dataset_head_config:
                    if dataset_head_config["classes"] is None:
                        tracks = [
                            x.name
                            for x in self.train_labels[d_idx].tracks
                            if x is not None
                        ]
                        classes = list(set(tracks))
                        if not len(classes):
                            name = self._dataset_name(d_idx)
                            message = f"Dataset {name}: No tracks found. ID models need tracks to be defined."
                            logger.error(message)
                            raise Exception(message)
                        self.config.model_config.head_configs[self.model_type][key][
                            d_idx
                        ]["classes"] = classes

    def _setup_ckpt_path(self):
        """Setup checkpoint path with multi-GPU validation."""
        ckpt_dir = self.config.trainer_config.ckpt_dir
        if ckpt_dir is None or ckpt_dir == "" or ckpt_dir == "None":
            ckpt_dir = "."
            self.config.trainer_config.ckpt_dir = ckpt_dir

        run_name = self.config.trainer_config.run_name
        run_name_is_empty = run_name is None or run_name == "" or run_name == "None"

        # Validate: multi-GPU + disk cache requires explicit run_name
        if run_name_is_empty:
            is_disk_caching = (
                self.config.data_config.data_pipeline_fw
                == "torch_dataset_cache_img_disk"
            )
            num_devices = self._get_trainer_devices()

            if is_disk_caching and num_devices > 1:
                raise ValueError(
                    f"Multi-GPU training with disk caching requires an explicit `run_name`.\n\n"
                    f"Detected {num_devices} device(s) with "
                    f"`data_pipeline_fw='torch_dataset_cache_img_disk'`.\n"
                    f"Without an explicit run_name, each GPU worker generates a different "
                    f"timestamp-based directory, causing cache synchronization failures.\n\n"
                    f"Please provide a run_name using one of these methods:\n"
                    f"  - CLI: sleap-nn train config.yaml trainer_config.run_name=my_experiment\n"
                    f"  - Config file: Set `trainer_config.run_name: my_experiment`\n"
                    f"  - Python API: train(..., run_name='my_experiment')"
                )

            # Auto-generate timestamp-based run_name
            sum_train_lfs = sum([len(train_label) for train_label in self.train_labels])
            sum_val_lfs = sum([len(val_label) for val_label in self.val_labels])
            run_name = (
                datetime.now().strftime("%y%m%d_%H%M%S")
                + f".multihead_{self.model_type}.n={sum_train_lfs + sum_val_lfs}"
            )

        # If checkpoint path already exists, add suffix to prevent overwriting
        if (Path(ckpt_dir) / run_name).exists() and (
            Path(ckpt_dir) / run_name / "best.ckpt"
        ).exists():
            logger.info(
                f"Checkpoint path already exists: {Path(ckpt_dir) / run_name}... adding suffix to prevent overwriting."
            )
            for i in count(1):
                new_run_name = f"{run_name}-{i}"
                if not (Path(ckpt_dir) / new_run_name).exists():
                    run_name = new_run_name
                    break

        self.config.trainer_config.run_name = run_name

        # Set output dir for cache img
        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk":
            if self.config.data_config.cache_img_path is None:
                self.config.data_config.cache_img_path = (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                )

    def _verify_model_input_channels(self):
        """Verify input channels in model_config based on input image and pretrained model weights."""
        # Check in channels, verify with img channels / ensure_rgb / ensure_grayscale
        if self.train_labels[0] is not None and len(self.train_labels[0]) > 0:
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

        # Verify input img channels with pretrained model ckpts (if any)
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
            weights_path = self.config.model_config.pretrained_backbone_weights
            if weights_path.endswith(".ckpt"):
                pretrained_backbone_ckpt = torch.load(
                    weights_path,
                    map_location="cpu",
                    weights_only=False,
                )
                input_channels = list(pretrained_backbone_ckpt["state_dict"].values())[
                    0
                ].shape[-3]
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
                    elif input_channels == 3:
                        self.config.data_config.preprocessing.ensure_rgb = True
                        self.config.data_config.preprocessing.ensure_grayscale = False

            elif weights_path.endswith(".h5"):
                input_channels = get_keras_first_layer_channels(weights_path)
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
                    elif input_channels == 3:
                        self.config.data_config.preprocessing.ensure_rgb = True
                        self.config.data_config.preprocessing.ensure_grayscale = False

    def setup_config(self):
        """Compute and update config parameters for multi-head training."""
        logger.info("Setting up config for multi-head training...")

        # Normalize empty strings to None for optional wandb fields
        if self.config.trainer_config.wandb.prv_runid == "":
            self.config.trainer_config.wandb.prv_runid = None

        # Compute preprocessing parameters from the labels objects
        self._setup_preprocessing_config()

        # Save skeletons to config (multiple skeletons for multi-head)
        self.config["data_config"]["skeletons"] = []
        for idx, skeleton in enumerate(self.skeletons):
            skeleton_yaml = yaml.safe_load(
                SkeletonYAMLEncoder().encode([skeleton])
            )
            for skeleton_name in skeleton_yaml.keys():
                skl = skeleton_yaml[skeleton_name]
                skl["name"] = skeleton_name
                skl["dataset_idx"] = idx
                self.config["data_config"]["skeletons"].append(skl)

        # Setup head config - partnames, edges and class names
        self._setup_head_config()

        # Set max stride for the backbone: convnext and swint
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

        # Set output stride for backbone from head config
        # Note: check_output_strides may need adaptation for multi-head
        # self.config = check_output_strides(self.config)  # TODO: adapt for multi-head

        # If trainer_devices is None, set it to "auto"
        if self.config.trainer_config.trainer_devices is None:
            self.config.trainer_config.trainer_devices = (
                "auto"
                if OmegaConf.select(
                    self.config, "trainer_config.trainer_device_indices", default=None
                )
                is None
                else len(
                    OmegaConf.select(
                        self.config,
                        "trainer_config.trainer_device_indices",
                        default=None,
                    )
                )
            )

        # Setup checkpoint path (generates run_name if not specified)
        self._setup_ckpt_path()

        # Default wandb run name to trainer run_name if not specified
        if self.config.trainer_config.wandb.name is None:
            self.config.trainer_config.wandb.name = self.config.trainer_config.run_name

        # Verify input_channels in model_config
        self._verify_model_input_channels()

    def _setup_model_ckpt_dir(self):
        """Create the model ckpt folder and save ground truth labels."""
        ckpt_path = (
            Path(self.config.trainer_config.ckpt_dir)
            / self.config.trainer_config.run_name
        ).as_posix()
        logger.info(f"Setting up model ckpt dir: `{ckpt_path}`...")

        # Only rank 0 (or non-distributed) should create directories and save files
        if RANK in [0, -1]:
            if not Path(ckpt_path).exists():
                try:
                    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    message = f"Cannot create a new folder in {ckpt_path}.\n {e}"
                    logger.error(message)
                    raise OSError(message)

            # Check if we should filter to user-labeled frames only
            user_instances_only = OmegaConf.select(
                self.config, "data_config.user_instances_only", default=True
            )

            # Save train and val ground truth labels for each dataset
            for idx, (train, val) in enumerate(
                zip(self.train_labels, self.val_labels)
            ):
                # Filter to user-labeled frames if needed
                if user_instances_only:
                    train_filtered = self._filter_to_user_labeled(train)
                    val_filtered = self._filter_to_user_labeled(val)
                else:
                    train_filtered = train
                    val_filtered = val

                train_filtered.save(
                    Path(ckpt_path) / f"labels_gt.train.{idx}.slp",
                    restore_original_videos=False,
                )
                val_filtered.save(
                    Path(ckpt_path) / f"labels_gt.val.{idx}.slp",
                    restore_original_videos=False,
                )
                logger.info(f"Saved ground truth labels for dataset {self._dataset_name(idx)}")

    def _setup_viz_datasets(self):
        """Setup visualization datasets for multi-head (without caching)."""
        data_viz_config = self.config.copy()
        data_viz_config.data_config.data_pipeline_fw = "torch_dataset"

        train_datasets = {}
        val_datasets = {}

        for d_idx in range(len(self.train_labels)):
            logger.info(f"Setting up visualization datasets for dataset {self._dataset_name(d_idx)}...")
            train_datasets[d_idx], val_datasets[d_idx] = get_train_val_datasets_multi_head(
                train_labels=[self.train_labels[d_idx]],
                val_labels=[self.val_labels[d_idx]],
                config=data_viz_config,
                d_idx=d_idx,
                rank=-1,
            )

        return train_datasets, val_datasets

    def _setup_datasets(self):
        """Setup datasets for all heads with caching support."""
        base_cache_img_path = None

        if self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_memory":
            # Check available memory. If insufficient, default to disk caching.
            train_num_workers = self.config.trainer_config.train_data_loader.num_workers
            val_num_workers = self.config.trainer_config.val_data_loader.num_workers
            max_num_workers = max(train_num_workers, val_num_workers)

            mem_available = check_cache_memory(
                self.train_labels,
                self.val_labels,
                memory_buffer=MEMORY_BUFFER,
                num_workers=max_num_workers,
            )
            if not mem_available:
                # Validate: multi-GPU + auto-generated run_name + fallback to disk cache
                original_run_name = self._initial_config.trainer_config.run_name
                run_name_was_auto = (
                    original_run_name is None
                    or original_run_name == ""
                    or original_run_name == "None"
                )
                if run_name_was_auto and self.trainer.num_devices > 1:
                    raise ValueError(
                        f"Memory caching failed and disk caching fallback requires an "
                        f"explicit `run_name` for multi-GPU training.\n\n"
                        f"Detected {self.trainer.num_devices} device(s) with insufficient "
                        f"memory for in-memory caching.\n"
                        f"Please provide a run_name or use `data_pipeline_fw='torch_dataset'` "
                        f"to disable caching."
                    )

                self.config.data_config.data_pipeline_fw = "torch_dataset_cache_img_disk"
                base_cache_img_path = Path("./")
                logger.info(
                    f"Insufficient memory for in-memory caching. `jpg` files will be created for disk-caching."
                )
            self.config.data_config.cache_img_path = base_cache_img_path

        elif self.config.data_config.data_pipeline_fw == "torch_dataset_cache_img_disk":
            base_cache_img_path = (
                Path(self.config.data_config.cache_img_path)
                if self.config.data_config.cache_img_path is not None
                else Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
            )

            if self.config.data_config.cache_img_path is None:
                self.config.data_config.cache_img_path = base_cache_img_path

        # Create datasets for each head
        train_datasets = {}
        val_datasets = {}

        for d_idx in range(len(self.train_labels)):
            logger.info(f"Setting up dataset {self._dataset_name(d_idx)}...")
            train_datasets[d_idx], val_datasets[d_idx] = get_train_val_datasets_multi_head(
                train_labels=[self.train_labels[d_idx]],
                val_labels=[self.val_labels[d_idx]],
                config=self.config,
                d_idx=d_idx,
                rank=self.trainer.global_rank,
            )

        return train_datasets, val_datasets

    def _setup_dataloaders(self, train_datasets, val_datasets):
        """Set up combined dataloaders for multi-head training."""
        train_dataloaders = {}
        val_dataloaders = {}

        for d_idx in range(len(self.train_labels)):
            train_dataloaders[d_idx], val_dataloaders[d_idx] = get_train_val_dataloaders_multi_head(
                train_dataset=train_datasets[d_idx],
                val_dataset=val_datasets[d_idx],
                config=self.config,
                d_idx=d_idx,
                rank=self.trainer.global_rank,
                trainer_devices=self.trainer.num_devices,
            )
            logger.info(f"Set up dataloaders for dataset {self._dataset_name(d_idx)}")

        # Combine dataloaders using CombinedLoader
        combined_loader_mode = OmegaConf.select(
            self.config, "trainer_config.combined_loader_mode", default="max_size_cycle"
        )

        train_loader = CombinedLoader(train_dataloaders, mode=combined_loader_mode)
        val_loader = CombinedLoader(val_dataloaders, mode=combined_loader_mode)

        return train_loader, val_loader

    def _setup_loggers_callbacks(self, viz_train_datasets, viz_val_datasets):
        """Create loggers and callbacks for multi-head training."""
        logger.info("Setting up callbacks and loggers...")
        loggers = []
        callbacks = []

        if self.config.trainer_config.save_ckpt:
            # Checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                save_top_k=self.config.trainer_config.model_ckpt.save_top_k,
                save_last=self.config.trainer_config.model_ckpt.save_last,
                dirpath=(
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                ).as_posix(),
                filename="best",
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

            # CSV log callback
            csv_log_keys = [
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
            ]
            # Add per-head loss keys
            for d_idx in range(len(self.train_labels)):
                name = self._dataset_name(d_idx)
                csv_log_keys.extend([
                    f"train_loss_head_{name}",
                    f"val_loss_head_{name}",
                ])

            csv_logger = CSVLoggerCallback(
                filepath=Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
                / "training_log.csv",
                keys=csv_log_keys,
            )
            callbacks.append(csv_logger)

        if self.config.trainer_config.early_stopping.stop_training_on_plateau:
            # Early stopping callback
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
            # WandB logger
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
                save_dir=(
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                ).as_posix(),
                id=self.config.trainer_config.wandb.prv_runid,
                group=self.config.trainer_config.wandb.group,
            )
            loggers.append(wandb_logger)

            # Mask API key
            self.config.trainer_config.wandb.api_key = ""
            if self._initial_config is not None:
                self._initial_config.trainer_config.wandb.api_key = ""

        # ZMQ callbacks
        if self.config.trainer_config.zmq.controller_port is not None:
            controller_address = "tcp://127.0.0.1:" + str(
                self.config.trainer_config.zmq.controller_port
            )
            callbacks.append(TrainingControllerZMQ(address=controller_address))
        if self.config.trainer_config.zmq.publish_port is not None:
            publish_address = "tcp://127.0.0.1:" + str(
                self.config.trainer_config.zmq.publish_port
            )
            callbacks.append(ProgressReporterZMQ(address=publish_address))

        # Add custom progress bar
        if self.config.trainer_config.enable_progress_bar:
            callbacks.append(SleapProgressBar())

        # Add unified visualization callback for each dataset
        if (
            self.config.trainer_config.visualize_preds_during_training
            and viz_train_datasets is not None
            and viz_val_datasets is not None
        ):
            viz_dir = (
                Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
                / "viz"
            )
            if RANK in [0, -1]:
                viz_dir.mkdir(parents=True, exist_ok=True)

            log_wandb = self.config.trainer_config.use_wandb
            wandb_modes = OmegaConf.select(
                self.config, "trainer_config.wandb.viz_modes", default=None
            )
            if wandb_modes is not None:
                wandb_modes = list(wandb_modes)

            # Create a UnifiedVizCallback for each dataset/head
            for d_idx in range(len(self.train_labels)):
                # Create dataset-specific viz directory
                name = self._dataset_name(d_idx)
                dataset_viz_dir = viz_dir / name
                if RANK in [0, -1]:
                    dataset_viz_dir.mkdir(parents=True, exist_ok=True)

                # Create adapter that wraps this trainer for the specific dataset
                adapter = MultiHeadModelTrainerAdapter(self, d_idx)

                callbacks.append(
                    UnifiedVizCallback(
                        model_trainer=adapter,
                        train_dataset=viz_train_datasets[d_idx],
                        val_dataset=viz_val_datasets[d_idx],
                        model_type=self.model_type,
                        save_local=self.config.trainer_config.save_ckpt,
                        local_save_dir=dataset_viz_dir,
                        log_wandb=log_wandb,
                        wandb_modes=wandb_modes if wandb_modes else ["direct"],
                        wandb_box_size=OmegaConf.select(
                            self.config, "trainer_config.wandb.viz_box_size", default=5.0
                        ),
                        wandb_confmap_threshold=OmegaConf.select(
                            self.config,
                            "trainer_config.wandb.viz_confmap_threshold",
                            default=0.1,
                        ),
                        log_wandb_table=OmegaConf.select(
                            self.config, "trainer_config.wandb.log_viz_table", default=False
                        ),
                        wandb_prefix=name,
                    )
                )
                logger.info(f"Added visualization callback for dataset {name}")

        return loggers, callbacks

    def _delete_cache_imgs(self):
        """Delete cache images on disk."""
        base_cache_img_path = Path(self.config.data_config.cache_img_path)
        train_cache_img_path = Path(base_cache_img_path) / "train_imgs"
        val_cache_img_path = Path(base_cache_img_path) / "val_imgs"

        if train_cache_img_path.exists():
            logger.info(f"Deleting cache imgs from `{train_cache_img_path}`...")
            shutil.rmtree(train_cache_img_path.as_posix(), ignore_errors=True)

        if val_cache_img_path.exists():
            logger.info(f"Deleting cache imgs from `{val_cache_img_path}`...")
            shutil.rmtree(val_cache_img_path.as_posix(), ignore_errors=True)

    def train(self):
        """Train the multi-head lightning model."""
        logger.info(f"Setting up for multi-head training...")
        start_setup_time = time.time()

        # Initialize the labels object and update config if needed
        if not len(self.train_labels) or not len(self.val_labels):
            raise ValueError(
                "train_labels and val_labels must be set before training. "
                "Use get_model_trainer_from_config() to create a properly initialized trainer."
            )

        # Create the ckpt dir
        self._setup_model_ckpt_dir()

        # Create the train and val datasets for visualization
        viz_train_datasets = None
        viz_val_datasets = None
        if self.config.trainer_config.visualize_preds_during_training:
            logger.info(f"Setting up visualization train and val datasets...")
            viz_train_datasets, viz_val_datasets = self._setup_viz_datasets()

        # Setup loggers and callbacks for Trainer
        logger.info(f"Setting up Trainer...")
        loggers, callbacks = self._setup_loggers_callbacks(
            viz_train_datasets=viz_train_datasets, viz_val_datasets=viz_val_datasets
        )

        # Set up the strategy (for multi-gpu training)
        strategy = OmegaConf.select(
            self.config, "trainer_config.trainer_strategy", default="auto"
        )

        # Set up profilers
        cfg_profiler = self.config.trainer_config.profiler
        profiler = None
        if cfg_profiler is not None:
            if cfg_profiler in self._profilers:
                profiler = self._profilers[cfg_profiler]
            else:
                message = f"{cfg_profiler} is not a valid option. Please choose one of {list(self._profilers.keys())}"
                logger.error(message)
                raise ValueError(message)

        devices = (
            OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            if OmegaConf.select(
                self.config, "trainer_config.trainer_device_indices", default=None
            )
            is not None
            else self.config.trainer_config.trainer_devices
        )
        logger.info(f"Trainer devices: {devices}")

        # If trainer devices is set to less than the number of available GPUs, use the least used GPUs
        if (
            torch.cuda.is_available()
            and self.config.trainer_config.trainer_accelerator != "cpu"
            and isinstance(self.config.trainer_config.trainer_devices, int)
            and self.config.trainer_config.trainer_devices < torch.cuda.device_count()
            and self.config.trainer_config.trainer_device_indices is None
        ):
            devices = [
                int(x)
                for x in np.argsort(get_gpu_memory())[::-1][
                    : self.config.trainer_config.trainer_devices
                ]
            ]
            # Sort device indices in ascending order for NCCL compatibility
            devices.sort()
            logger.info(f"Using GPUs with most available memory: {devices}")

        # Create lightning.Trainer instance
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=self.config.trainer_config.save_ckpt,
            devices=devices,
            max_epochs=self.config.trainer_config.max_epochs,
            accelerator=self.config.trainer_config.trainer_accelerator,
            enable_progress_bar=self.config.trainer_config.enable_progress_bar,
            strategy=strategy,
            profiler=profiler,
            log_every_n_steps=1,
        )

        self.trainer.strategy.barrier()

        # Setup datasets
        train_datasets, val_datasets = self._setup_datasets()

        # Barrier after dataset creation to ensure all workers wait for disk caching
        self.trainer.strategy.barrier()

        logger.info(f"Training on {self.trainer.num_devices} device(s)")
        logger.info(f"Training on {self.trainer.strategy.root_device} accelerator")

        # Initialize the lightning model (after Trainer is initialized for accelerator info)
        logger.info(f"Setting up multi-head lightning module for {self.model_type} model...")

        # Compute dataset loss weights if requested
        dataset_loss_weights = None
        if self.apply_dataset_loss_weights:
            # With max_size_cycle, smaller datasets are repeated to match the largest.
            # To compensate, we give lower weights to smaller datasets (seen more often)
            # and higher weights to larger datasets (seen less often per sample).
            dataset_sizes = {i: len(ds) for i, ds in train_datasets.items()}
            max_size = max(dataset_sizes.values())
            raw_weights = {i: size / max_size for i, size in dataset_sizes.items()}
            sum_weights = sum(raw_weights.values())
            dataset_loss_weights = {
                i: (w / sum_weights) * len(train_datasets)
                for i, w in raw_weights.items()
            }
            logger.info(f"Dataset sizes: {dataset_sizes}")
            logger.info(f"Dataset loss weights: {dataset_loss_weights}")

        dataset_names = {i: self._dataset_name(i) for i in range(len(self.train_labels))}
        self.lightning_model = MultiHeadLightningModel.get_lightning_model_from_config(
            config=self.config,
            dataset_loss_weights=dataset_loss_weights,
            dataset_names=dataset_names,
        )
        logger.info(f"Backbone model: {self.lightning_model.model.backbone}")
        logger.info(f"Head models: {self.lightning_model.model.head_layers}")
        total_params = sum(p.numel() for p in self.lightning_model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
        self.config.model_config.total_params = total_params

        # Setup dataloaders (after Trainer is initialized for DDP/rank info)
        train_dataloader, val_dataloader = self._setup_dataloaders(
            train_datasets, val_datasets
        )

        if self.trainer.global_rank == 0:  # Save config only in rank 0 process
            ckpt_path = (
                Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
            ).as_posix()
            OmegaConf.save(
                self._initial_config,
                (Path(ckpt_path) / "initial_config.yaml").as_posix(),
            )

            if self.config.trainer_config.use_wandb:
                if wandb.run is None:
                    wandb.init(
                        dir=(
                            Path(self.config.trainer_config.ckpt_dir)
                            / self.config.trainer_config.run_name
                        ).as_posix(),
                        project=self.config.trainer_config.wandb.project,
                        entity=self.config.trainer_config.wandb.entity,
                        name=self.config.trainer_config.wandb.name,
                        id=self.config.trainer_config.wandb.prv_runid,
                        group=self.config.trainer_config.wandb.group,
                    )

                # Define custom x-axes for wandb metrics
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("val/*", step_metric="epoch")
                # Per-head metrics
                for d_idx in range(len(self.train_labels)):
                    name = self._dataset_name(d_idx)
                    wandb.define_metric(f"train/head_{name}/*", step_metric="epoch")
                    wandb.define_metric(f"val/head_{name}/*", step_metric="epoch")

                self.config.trainer_config.wandb.current_run_id = wandb.run.id
                wandb.config["run_name"] = self.config.trainer_config.wandb.name
                wandb.config["run_config"] = OmegaConf.to_container(
                    self.config, resolve=True
                )
                wandb.config["num_datasets"] = len(self.train_labels)
                wandb.config["skeletons"] = [s.name for s in self.skeletons]

            OmegaConf.save(
                self.config,
                (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                    / "training_config.yaml"
                ).as_posix(),
            )

        self.trainer.strategy.barrier()

        # Flag to track if training was interrupted
        training_interrupted = False

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
            training_interrupted = True

        finally:
            logger.info(
                f"Finished training loop. [{(time.time() - start_train_time) / 60:.1f} min]"
            )

            # Delete image disk caching
            if (
                self.config.data_config.data_pipeline_fw
                == "torch_dataset_cache_img_disk"
                and self.config.data_config.delete_cache_imgs_after_training
            ):
                if self.trainer.global_rank == 0:
                    self._delete_cache_imgs()

            # Delete viz folder if requested
            if (
                self.config.trainer_config.visualize_preds_during_training
                and not self.config.trainer_config.keep_viz
            ):
                if self.trainer.global_rank == 0:
                    viz_dir = (
                        Path(self.config.trainer_config.ckpt_dir)
                        / self.config.trainer_config.run_name
                        / "viz"
                    )
                    if viz_dir.exists():
                        logger.info(f"Deleting viz folder at {viz_dir}...")
                        shutil.rmtree(viz_dir, ignore_errors=True)

            # Clean up entire run folder if training was interrupted
            if training_interrupted and self.trainer.global_rank == 0:
                run_dir = (
                    Path(self.config.trainer_config.ckpt_dir)
                    / self.config.trainer_config.run_name
                )
                if run_dir.exists():
                    logger.info(
                        f"Training canceled - cleaning up run folder at {run_dir}..."
                    )
                    shutil.rmtree(run_dir, ignore_errors=True)
