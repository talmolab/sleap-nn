"""This module is to train a sleap-nn model using Lightning."""

import os
import shutil
import copy
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
from itertools import cycle, count
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
)
from loguru import logger
from sleap_nn.config.utils import (
    get_backbone_type_from_cfg,
    get_model_type_from_cfg,
)
from sleap_nn.training.lightning_modules import LightningModel
from sleap_nn.config.utils import check_output_strides
from sleap_nn.training.utils import get_gpu_memory
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.callbacks import (
    ProgressReporterZMQ,
    TrainingControllerZMQ,
    MatplotlibSaver,
    WandBPredImageLogger,
    WandBVizCallback,
    WandBVizCallbackWithPAFs,
    CSVLoggerCallback,
    SleapProgressBar,
    EpochEndEvaluationCallback,
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
        if run_name is None or run_name == "" or run_name == "None":
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

        if not Path(ckpt_path).exists():
            try:
                Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                message = f"Cannot create a new folder in {ckpt_path}.\n {e}"
                logger.error(message)
                raise OSError(message)

        if RANK in [0, -1]:
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

        # viz callbacks
        if self.config.trainer_config.visualize_preds_during_training:
            train_viz_pipeline = cycle(viz_train_dataset)
            val_viz_pipeline = cycle(viz_val_dataset)

            viz_dir = (
                Path(self.config.trainer_config.ckpt_dir)
                / self.config.trainer_config.run_name
                / "viz"
            )
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

            if self.config.trainer_config.use_wandb and OmegaConf.select(
                self.config, "trainer_config.wandb.save_viz_imgs_wandb", default=False
            ):
                # Get wandb viz config options
                viz_enabled = OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_enabled", default=True
                )
                viz_boxes = OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_boxes", default=False
                )
                viz_masks = OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_masks", default=False
                )
                viz_box_size = OmegaConf.select(
                    self.config, "trainer_config.wandb.viz_box_size", default=5.0
                )
                viz_confmap_threshold = OmegaConf.select(
                    self.config,
                    "trainer_config.wandb.viz_confmap_threshold",
                    default=0.1,
                )
                log_viz_table = OmegaConf.select(
                    self.config, "trainer_config.wandb.log_viz_table", default=False
                )

                # Create viz data pipelines for wandb callback
                wandb_train_viz_pipeline = cycle(copy.deepcopy(viz_train_dataset))
                wandb_val_viz_pipeline = cycle(copy.deepcopy(viz_val_dataset))

                if self.model_type == "bottomup":
                    # Bottom-up model needs PAF visualizations
                    wandb_train_pafs_pipeline = cycle(copy.deepcopy(viz_train_dataset))
                    wandb_val_pafs_pipeline = cycle(copy.deepcopy(viz_val_dataset))
                    callbacks.append(
                        WandBVizCallbackWithPAFs(
                            train_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_train_viz_pipeline)
                            ),
                            val_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_val_viz_pipeline)
                            ),
                            train_pafs_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_train_pafs_pipeline), include_pafs=True
                            ),
                            val_pafs_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_val_pafs_pipeline), include_pafs=True
                            ),
                            viz_enabled=viz_enabled,
                            viz_boxes=viz_boxes,
                            viz_masks=viz_masks,
                            box_size=viz_box_size,
                            confmap_threshold=viz_confmap_threshold,
                            log_table=log_viz_table,
                        )
                    )
                else:
                    # Standard models
                    callbacks.append(
                        WandBVizCallback(
                            train_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_train_viz_pipeline)
                            ),
                            val_viz_fn=lambda: self.lightning_model.get_visualization_data(
                                next(wandb_val_viz_pipeline)
                            ),
                            viz_enabled=viz_enabled,
                            viz_boxes=viz_boxes,
                            viz_masks=viz_masks,
                            box_size=viz_box_size,
                            confmap_threshold=viz_confmap_threshold,
                            log_table=log_viz_table,
                        )
                    )

        # Add custom progress bar with better metric formatting
        if self.config.trainer_config.enable_progress_bar:
            callbacks.append(SleapProgressBar())

        # Add epoch-end evaluation callback if enabled
        if self.config.trainer_config.eval.enabled:
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
