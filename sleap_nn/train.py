"""Entry point for sleap_nn training."""

import hydra
from loguru import logger
from pathlib import Path
import numpy as np
from datetime import datetime
from time import time
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional, List, Tuple, Union
import sleap_io as sio
from sleap_nn.config.data_config import DataConfig, PreprocessingConfig
from sleap_nn.config.model_config import ModelConfig
from sleap_nn.config.trainer_config import (
    DataLoaderConfig,
    LRSchedulerConfig,
    StepLRConfig,
    EarlyStoppingConfig,
    ReduceLROnPlateauConfig,
    TrainerConfig,
    ModelCkptConfig,
    WandBConfig,
    HardKeypointMiningConfig,
    OptimizerConfig,
    ZMQConfig,
)
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.inference.predictors import run_inference as predict
from sleap_nn.evaluation import Evaluator

from sleap_nn.config.data_config import (
    AugmentationConfig,
    IntensityConfig,
    GeometricConfig,
)
from sleap_nn.config.model_config import (
    ConvNextConfig,
    SwinTConfig,
    BackboneConfig,
    HeadConfig,
    UNetConfig,
    UNetMediumRFConfig,
    UNetLargeRFConfig,
    ConvNextSmallConfig,
    ConvNextLargeConfig,
    ConvNextBaseConfig,
    SwinTBaseConfig,
    SwinTSmallConfig,
    SingleInstanceConfig,
    SingleInstanceConfMapsConfig,
    CentroidConfig,
    CentroidConfMapsConfig,
    CenteredInstanceConfMapsConfig,
    CenteredInstanceConfig,
    BottomUpConfig,
    BottomUpMultiClassConfig,
    BottomUpConfMapsConfig,
    PAFConfig,
    ClassMapConfig,
    TopDownCenteredInstanceMultiClassConfig,
    ClassVectorsConfig,
)


def get_aug_config(intensity_aug, geometric_aug):
    """Returns `AugmentationConfig` object based on the user-provided parameters."""
    aug_config = AugmentationConfig(
        intensity=IntensityConfig(), geometric=GeometricConfig()
    )
    if isinstance(intensity_aug, str) or isinstance(intensity_aug, list):
        if isinstance(intensity_aug, str):
            intensity_aug = [intensity_aug]

        for i in intensity_aug:
            if i == "uniform_noise":
                aug_config.intensity.uniform_noise_p = 1.0
            elif i == "gaussian_noise":
                aug_config.intensity.gaussian_noise_p = 1.0
            elif i == "contrast":
                aug_config.intensity.contrast_p = 1.0
            elif i == "brightness":
                aug_config.intensity.brightness_p = 1.0
            else:
                raise ValueError(
                    f"`{intensity_aug}` is not a valid intensity augmentation option. Please use one of ['uniform_noise', 'gaussian_noise', 'contrast', 'brightness']"
                )

    elif isinstance(intensity_aug, dict):
        aug_config.intensity = IntensityConfig(**intensity_aug)

    if isinstance(geometric_aug, str) or isinstance(geometric_aug, list):
        if isinstance(geometric_aug, str):
            geometric_aug = [geometric_aug]

        for g in geometric_aug:
            if g == "rotation":
                aug_config.geometric.affine_p = 1.0
                aug_config.geometric.scale = (1.0, 1.0)
                aug_config.geometric.translate_height = 0
                aug_config.geometric.translate_width = 0
            elif g == "scale":
                aug_config.geometric.scale = (0.9, 1.1)
                aug_config.geometric.affine_p = 1.0
                aug_config.geometric.rotation = 0
                aug_config.geometric.translate_height = 0
                aug_config.geometric.translate_width = 0
            elif g == "translate":
                aug_config.geometric.translate_height = 0.2
                aug_config.geometric.translate_width = 0.2
                aug_config.geometric.affine_p = 1.0
                aug_config.geometric.rotation = 0
                aug_config.geometric.scale = (1.0, 1.0)
            elif g == "erase_scale":
                aug_config.geometric.erase_p = 1.0
            elif g == "mixup":
                aug_config.geometric.mixup_p = 1.0
            else:
                raise ValueError(
                    f"`{intensity_aug}` is not a valid geometric augmentation option. Please use one of ['rotation', 'scale', 'translate', 'erase_scale', 'mixup']"
                )

    elif isinstance(geometric_aug, dict):
        aug_config.geometric = GeometricConfig(**geometric_aug)

    return aug_config


def get_backbone_config(backbone_cfg):
    """Returns `BackboneConfig` object based on the user-provided parameters."""
    backbone_config = BackboneConfig()
    unet_config_mapper = {
        "unet": UNetConfig(),
        "unet_medium_rf": UNetMediumRFConfig(),
        "unet_large_rf": UNetLargeRFConfig(),
    }
    convnext_config_mapper = {
        "convnext": ConvNextConfig(),
        "convnext_tiny": ConvNextConfig(),
        "convnext_small": ConvNextSmallConfig(),
        "convnext_base": ConvNextBaseConfig(),
        "convnext_large": ConvNextLargeConfig(),
    }
    swint_config_mapper = {
        "swint": SwinTConfig(),
        "swint_tiny": SwinTConfig(),
        "swint_small": SwinTSmallConfig(),
        "swint_base": SwinTBaseConfig(),
    }
    if isinstance(backbone_cfg, str):
        if backbone_cfg.startswith("unet"):
            backbone_config.unet = unet_config_mapper[backbone_cfg]
        elif backbone_cfg.startswith("convnext"):
            backbone_config.convnext = convnext_config_mapper[backbone_cfg]
        elif backbone_cfg.startswith("swint"):
            backbone_config.swint = swint_config_mapper[backbone_cfg]
        else:
            raise ValueError(
                f"{backbone_cfg} is not a valid backbone. Please choose one of ['unet', 'unet_medium_rf', 'unet_large_rf', 'convnext', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'swint', 'swint_tiny', 'swint_small', 'swint_base']"
            )

    elif isinstance(backbone_cfg, dict):
        backbone_config = BackboneConfig()
        if "unet" in backbone_cfg:
            backbone_config.unet = UNetConfig(**backbone_cfg["unet"])
        elif "convnext" in backbone_cfg:
            backbone_config.convnext = ConvNextConfig(**backbone_cfg["convnext"])
        elif "swint" in backbone_cfg:
            backbone_config.swint = SwinTConfig(**backbone_cfg["swint"])

    return backbone_config


def get_head_configs(head_cfg):
    """Returns `HeadConfig` object based on the user-provided parameters."""
    head_configs = HeadConfig()
    if isinstance(head_cfg, str):
        if head_cfg == "centered_instance":
            head_configs.centered_instance = CenteredInstanceConfig(
                confmaps=CenteredInstanceConfMapsConfig
            )
        elif head_cfg == "single_instance":
            head_configs.single_instance = SingleInstanceConfig(
                confmaps=SingleInstanceConfMapsConfig
            )
        elif head_cfg == "centroid":
            head_configs.centroid = CentroidConfig(confmaps=CentroidConfMapsConfig)
        elif head_cfg == "bottomup":
            head_configs.bottomup = BottomUpConfig(
                confmaps=BottomUpConfMapsConfig, pafs=PAFConfig
            )
        elif head_cfg == "multi_class_bottomup":
            head_configs.multi_class_bottomup = BottomUpMultiClassConfig(
                confmaps=BottomUpConfMapsConfig, class_maps=ClassMapConfig
            )
        elif head_cfg == "multi_class_topdown":
            head_configs.multi_class_topdown = TopDownCenteredInstanceMultiClassConfig(
                confmaps=CenteredInstanceConfMapsConfig,
                class_vectors=ClassVectorsConfig,
            )
        else:
            raise ValueError(
                f"{head_cfg} is not a valid head type. Please choose one of ['bottomup', 'centered_instance', 'centroid', 'single_instance', 'multi_class_bottomup', 'multi_class_topdown']"
            )

    elif isinstance(head_cfg, dict):
        head_configs = HeadConfig()
        if "single_instance" in head_cfg and head_cfg["single_instance"] is not None:
            head_configs.single_instance = SingleInstanceConfig(
                confmaps=SingleInstanceConfMapsConfig(
                    **head_cfg["single_instance"]["confmaps"]
                )
            )
        elif "centroid" in head_cfg and head_cfg["centroid"] is not None:
            head_configs.centroid = CentroidConfig(
                confmaps=CentroidConfMapsConfig(**head_cfg["centroid"]["confmaps"])
            )
        elif (
            "centered_instance" in head_cfg
            and head_cfg["centered_instance"] is not None
        ):
            head_configs.centered_instance = CenteredInstanceConfig(
                confmaps=CenteredInstanceConfMapsConfig(
                    **head_cfg["centered_instance"]["confmaps"]
                )
            )
        elif "bottomup" in head_cfg and head_cfg["bottomup"] is not None:
            head_configs.bottomup = BottomUpConfig(
                confmaps=BottomUpConfMapsConfig(
                    **head_cfg["bottomup"]["confmaps"],
                ),
                pafs=PAFConfig(**head_cfg["bottomup"]["pafs"]),
            )
        elif (
            "multi_class_bottomup" in head_cfg
            and head_cfg["multi_class_bottomup"] is not None
        ):
            head_configs.multi_class_bottomup = BottomUpMultiClassConfig(
                confmaps=BottomUpConfMapsConfig(
                    **head_cfg["multi_class_bottomup"]["confmaps"]
                ),
                class_maps=ClassMapConfig(
                    **head_cfg["multi_class_bottomup"]["class_maps"]
                ),
            )
        elif (
            "multi_class_topdown" in head_cfg
            and head_cfg["multi_class_topdown"] is not None
        ):
            head_configs.multi_class_topdown = TopDownCenteredInstanceMultiClassConfig(
                confmaps=CenteredInstanceConfMapsConfig(
                    **head_cfg["multi_class_topdown"]["confmaps"]
                ),
                class_vectors=ClassVectorsConfig(
                    **head_cfg["multi_class_topdown"]["class_vectors"]
                ),
            )

    return head_configs


def get_data_config(
    train_labels_path: List[str],
    val_labels_path: List[str] = [],
    validation_fraction: float = 0.1,
    test_file_path: Optional[str] = None,
    provider: str = "LabelsReader",
    user_instances_only: bool = True,
    data_pipeline_fw: str = "torch_dataset",
    cache_img_path: Optional[str] = None,
    use_existing_imgs: bool = False,
    delete_cache_imgs_after_training: bool = True,
    ensure_rgb: bool = False,
    ensure_grayscale: bool = False,
    scale: float = 1.0,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
    min_crop_size: Optional[int] = 100,
    use_augmentations_train: bool = False,
    intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    geometry_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
        train_labels_path: List of paths to training data (`.slp` file).
        val_labels_path: List of paths to validation data (`.slp` file).
        validation_fraction: Float between 0 and 1 specifying the fraction of the
            training set to sample for generating the validation set. The remaining
            labeled frames will be left in the training set. If the `validation_labels`
            are already specified, this has no effect. Default: 0.1.
        test_file_path: Path to test dataset (`.slp` file or `.mp4` file).
            Note: This is used to get evaluation on test set after training is completed.
        provider: Provider class to read the input sleap files. Only "LabelsReader"
            supported for the training pipeline. Default: "LabelsReader".
        user_instances_only: `True` if only user labeled instances should be used for
            training. If `False`, both user labeled and predicted instances would be used.
            Default: `True`.
        data_pipeline_fw: Framework to create the data loaders. One of [`torch_dataset`,
            `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. Default: "torch_dataset".
        cache_img_path: Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline
            framework. If `None`, the path provided in `trainer_config.save_ckpt` is used (else working dir is used). The `train_imgs` and `val_imgs` dirs are created inside this path. Default: None.
        use_existing_imgs: Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` frameworks. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs.
            Default: False.
        delete_cache_imgs_after_training: If `False`, the images (torch_dataset_cache_img_disk) are
            retained after training. Else, the files are deleted. Default: True.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
        channel when this is set to `True`, then the images from single-channel
        is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
        is set to True, then we convert the image to grayscale (single-channel)
        image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
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
            For eg: {
                        "uniform_noise_min": 1.0,
                        "uniform_noise_p": 1.0
                    }
        geometry_aug: One of ["rotation", "scale", "translate", "erase_scale", "mixup"].
            or list of strings from the above allowed values. To have custom values, pass
            a dict with the structure in `sleap_nn.config.data_config.GeometryConfig`.
            For eg: {
                        "rotation": 45,
                        "affine_p": 1.0
                    }
    """
    preprocessing_config = PreprocessingConfig(
        ensure_rgb=ensure_rgb,
        ensure_grayscale=ensure_grayscale,
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
        validation_fraction=validation_fraction,
        test_file_path=test_file_path,
        provider=provider,
        user_instances_only=user_instances_only,
        data_pipeline_fw=data_pipeline_fw,
        cache_img_path=cache_img_path,
        use_existing_imgs=use_existing_imgs,
        delete_cache_imgs_after_training=delete_cache_imgs_after_training,
        preprocessing=preprocessing_config,
        use_augmentations_train=use_augmentations_train,
        augmentation_config=augmentation_config,
    )

    return data_config


def get_model_config(
    init_weight: str = "default",
    pre_trained_weights: Optional[str] = None,
    pretrained_backbone_weights: Optional[str] = None,
    pretrained_head_weights: Optional[str] = None,
    backbone_config: Union[str, Dict[str, Any]] = "unet",
    head_configs: Union[str, Dict[str, Any]] = None,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
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
            then pass a dictionary with the structure:
            {
                "unet((or) convnext (or)swint)":
                    {(params in the corresponding architecture given in `sleap_nn.config.model_config.backbone_config`)
                    }
            }.
            For eg: {
                        "unet":
                            {
                                "in_channels": 3,
                                "filters": 64,
                                "max_stride": 32,
                                "output_stride": 2
                            }
                    }
        head_configs: One of ["bottomup", "centered_instance", "centroid", "single_instance", "multi_class_bottomup", "multi_class_topdown"].
            The default `sigma` and `output_strides` are used if a string is passed. To
            set custom parameters, pass in a dictionary with the structure:
            {
                "bottomup" (or "centroid" or "single_instance" or "centered_instance" or "multi_class_bottomup" or "multi_class_topdown"):
                    {
                        "confmaps":
                            {
                                # params in the corresponding head type given in `sleap_nn.config.model_config.head_configs`
                            },
                        "pafs":
                            {
                                # only for bottomup
                            }
                    }
            }.
            For eg: {
                        "single_instance":
                            {
                                "confmaps":
                                    {
                                        "part_names": None,
                                        "sigma": 2.5,
                                        "output_stride": 2
                                    }
                            }
                    }

    """
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
    return model_config


def get_trainer_config(
    batch_size: int = 1,
    shuffle_train: bool = False,
    num_workers: int = 0,
    ckpt_save_top_k: int = 1,
    ckpt_save_last: Optional[bool] = None,
    trainer_num_devices: Union[str, int] = "auto",
    trainer_accelerator: str = "auto",
    enable_progress_bar: bool = True,
    min_train_steps_per_epoch: int = 200,
    train_steps_per_epoch: Optional[int] = None,
    visualize_preds_during_training: bool = False,
    keep_viz: bool = False,
    max_epochs: int = 10,
    seed: int = 0,
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
    lr_scheduler: Optional[Union[str, Dict[str, Any]]] = None,
    early_stopping: bool = False,
    early_stopping_min_delta: float = 0.0,
    early_stopping_patience: int = 1,
    online_mining: bool = False,
    hard_to_easy_ratio: float = 2.0,
    min_hard_keypoints: int = 2,
    max_hard_keypoints: Optional[int] = None,
    loss_scale: float = 5.0,
    zmq_publish_address: Optional[str] = None,
    zmq_controller_address: Optional[str] = None,
    zmq_controller_timeout: int = 10,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
        batch_size: Number of samples per batch or batch size for training data. Default: 4.
        shuffle_train: True to have the train data reshuffled at every epoch. Default: False.
        num_workers: Number of subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process. Default: 0.
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
        min_train_steps_per_epoch: Minimum number of iterations in a single epoch. (Useful if model
            is trained with very few data points). Refer `limit_train_batches` parameter
            of Torch `Trainer`. Default: 200.
        train_steps_per_epoch: Number of minibatches (steps) to train for in an epoch. If set to `None`,
            this is set to the number of batches in the training data or `min_train_steps_per_epoch`,
            whichever is largest. Default: `None`.
        visualize_preds_during_training: If set to `True`, sample predictions (keypoints  + confidence maps)
            are saved to `viz` folder in the ckpt dir and in wandb table.
        keep_viz: If set to `True`, the `viz` folder will be kept after training. If `False`, the `viz` folder
            will be deleted after training. Only applies when `visualize_preds_during_training` is `True`.
        max_epochs: Maximum number of epochs to run. Default: 100.
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
        wandb_group_name: Group name for the wandb run. Default: None.
        optimizer: Optimizer to be used. One of ["Adam", "AdamW"]. Default: "Adam".
        learning_rate: Learning rate of type float. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
        lr_scheduler: One of ["step_lr", "reduce_lr_on_plateau"] (the default values in
            `sleap_nn.config.trainer_config` are used). To use custom values, pass a
            dictionary with the structure in `sleap_nn.config.trainer_config.LRSchedulerConfig`.
            For eg, {
                        "step_lr":
                            {
                                (params in `sleap_nn.config.trainer_config.StepLRConfig`)
                            }
                    }
        early_stopping: True if early stopping should be enabled. Default: False.
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as
            an improvement, i.e. an absolute change of less than or equal to min_delta,
            will count as no improvement. Default: 0.0.
        early_stopping_patience: Number of checks with no improvement after which training
            will be stopped. Under the default configuration, one check happens after every
            training epoch. Default: 1.
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When
            this is enabled, the loss is computed per keypoint (or edge for PAFs) and
            sorted from lowest (easy) to highest (hard). The hard keypoint loss will be
            scaled to have a higher weight in the total loss, encouraging the training
            to focus on tricky body parts that are more difficult to learn.
            If False, no mining will be performed and all keypoints will be weighted
            equally in the loss.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with
            respect to the lowest keypoint loss in order to be considered as "hard".
            This helps to switch focus on across groups of keypoints during training.
        min_hard_keypoints: The minimum number of keypoints that will be considered as
            "hard", even if they are not below the `hard_to_easy_ratio`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to.
            This can help when there are few very easy keypoints which may skew the
            ratio and result in loss scaling being applied to most keypoints, which can
            reduce the impact of hard mining altogether.
        loss_scale: Factor to scale the hard keypoint losses by for oks.
        zmq_publish_address: (str) Specifies the address and port to which the training logs (loss values) should be sent to.
        zmq_controller_address: (str) Specifies the address and port to listen to to stop the training (specific to SLEAP GUI).
        zmq_controller_timeout: (int) Polling timeout in microseconds specified as an integer. This controls how long the poller
            should wait to receive a response and should be set to a small value to minimize the impact on training speed.
    """
    # constrict trainer config
    train_dataloader_cfg = DataLoaderConfig(
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_dataloader_cfg = DataLoaderConfig(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    lr_scheduler_cfg = LRSchedulerConfig()
    if isinstance(lr_scheduler, str):
        if lr_scheduler == "step_lr":
            lr_scheduler_cfg.step_lr = StepLRConfig()
        elif lr_scheduler == "reduce_lr_on_plateau":
            lr_scheduler_cfg.reduce_lr_on_plateau = ReduceLROnPlateauConfig()
        else:
            message = f"{lr_scheduler} is not a valid scheduler. Please choose one of ['step_lr', 'reduce_lr_on_plateau']"
            logger.error(message)
            raise ValueError(message)
    elif isinstance(lr_scheduler, dict):
        if lr_scheduler is None:
            lr_scheduler = {
                "step_lr": None,
                "reduce_lr_on_plateau": None,
            }
        for k, v in lr_scheduler.items():
            if v is not None:
                if k == "step_lr":
                    lr_scheduler_cfg.step_lr = StepLRConfig(**v)
                    break
                elif k == "reduce_lr_on_plateau":
                    lr_scheduler_cfg.reduce_lr_on_plateau = ReduceLROnPlateauConfig(**v)
                    break

    trainer_config = TrainerConfig(
        train_data_loader=train_dataloader_cfg,
        val_data_loader=val_dataloader_cfg,
        model_ckpt=ModelCkptConfig(
            save_top_k=ckpt_save_top_k, save_last=ckpt_save_last
        ),
        trainer_devices=trainer_num_devices,
        trainer_accelerator=trainer_accelerator,
        enable_progress_bar=enable_progress_bar,
        min_train_steps_per_epoch=min_train_steps_per_epoch,
        train_steps_per_epoch=train_steps_per_epoch,
        visualize_preds_during_training=visualize_preds_during_training,
        keep_viz=keep_viz,
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
        online_hard_keypoint_mining=HardKeypointMiningConfig(
            online_mining=online_mining,
            hard_to_easy_ratio=hard_to_easy_ratio,
            min_hard_keypoints=min_hard_keypoints,
            max_hard_keypoints=max_hard_keypoints,
            loss_scale=loss_scale,
        ),
        zmq=ZMQConfig(
            controller_address=zmq_controller_address,
            controller_polling_timeout=zmq_controller_timeout,
            publish_address=zmq_publish_address,
        ),
    )
    return trainer_config


def run_training(config: DictConfig):
    """Create ModelTrainer instance and start training."""
    start_train_time = time()
    start_timestamp = str(datetime.now())
    logger.info("Started training at:", start_timestamp)

    trainer = ModelTrainer.get_model_trainer_from_config(config)
    trainer.train()

    finish_timestamp = str(datetime.now())
    total_elapsed = time() - start_train_time
    logger.info("Finished training at:", finish_timestamp)
    logger.info(f"Total training time: {total_elapsed} secs")

    # run inference on val dataset
    if config.trainer_config.save_ckpt:
        data_paths = {}
        for index, path in enumerate(trainer.config.data_config.train_labels_path):
            data_paths[f"train_{index}"] = (
                Path(config.trainer_config.save_ckpt_path)
                / f"labels_train_gt_{index}.slp"
            ).as_posix()
            data_paths[f"val_{index}"] = (
                Path(config.trainer_config.save_ckpt_path)
                / f"labels_val_gt_{index}.slp"
            ).as_posix()

        if (
            OmegaConf.select(config, "data_config.test_file_path", default=None)
            is not None
        ):
            data_paths["test"] = config.data_config.test_file_path

        for d_name, path in data_paths.items():
            labels = sio.load_slp(path)

            pred_labels = predict(
                data_path=path,
                model_paths=[config.trainer_config.save_ckpt_path],
                peak_threshold=0.2,
                make_labels=True,
                device=trainer.trainer.strategy.root_device,
                output_path=Path(config.trainer_config.save_ckpt_path)
                / f"pred_{d_name}.slp",
                ensure_rgb=config.data_config.preprocessing.ensure_rgb,
                ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
            )

            if not len(pred_labels):
                logger.info(
                    f"Skipping eval on `{d_name}` dataset as there are no labeled frames..."
                )
                continue  # skip if there are no labeled frames

            evaluator = Evaluator(
                ground_truth_instances=labels, predicted_instances=pred_labels
            )
            metrics = evaluator.evaluate()
            np.savez(
                (
                    Path(config.trainer_config.save_ckpt_path)
                    / f"{d_name}_pred_metrics.npz"
                ).as_posix(),
                **metrics,
            )

            logger.info(f"---------Evaluation on `{d_name}` dataset---------")
            logger.info(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']}")
            logger.info(f"Average distance: {metrics['distance_metrics']['avg']}")
            logger.info(f"p90 dist: {metrics['distance_metrics']['p90']}")
            logger.info(f"p50 dist: {metrics['distance_metrics']['p50']}")


def train(
    train_labels_path: List[str],
    val_labels_path: List[str] = [],
    validation_fraction: float = 0.1,
    test_file_path: Optional[str] = None,
    provider: str = "LabelsReader",
    user_instances_only: bool = True,
    data_pipeline_fw: str = "torch_dataset",
    cache_img_path: Optional[str] = None,
    use_existing_imgs: bool = False,
    delete_cache_imgs_after_training: bool = True,
    ensure_rgb: bool = False,
    ensure_grayscale: bool = False,
    scale: float = 1.0,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
    min_crop_size: Optional[int] = 100,
    use_augmentations_train: bool = False,
    intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    geometry_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    init_weight: str = "default",
    pre_trained_weights: Optional[str] = None,
    pretrained_backbone_weights: Optional[str] = None,
    pretrained_head_weights: Optional[str] = None,
    backbone_config: Union[str, Dict[str, Any]] = "unet",
    head_configs: Union[str, Dict[str, Any]] = None,
    batch_size: int = 1,
    shuffle_train: bool = False,
    num_workers: int = 0,
    ckpt_save_top_k: int = 1,
    ckpt_save_last: Optional[bool] = None,
    trainer_num_devices: Union[str, int] = "auto",
    trainer_accelerator: str = "auto",
    enable_progress_bar: bool = True,
    min_train_steps_per_epoch: int = 200,
    train_steps_per_epoch: Optional[int] = None,
    visualize_preds_during_training: bool = False,
    keep_viz: bool = False,
    max_epochs: int = 10,
    seed: int = 0,
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
    lr_scheduler: Optional[Union[str, Dict[str, Any]]] = None,
    early_stopping: bool = False,
    early_stopping_min_delta: float = 0.0,
    early_stopping_patience: int = 1,
    online_mining: bool = False,
    hard_to_easy_ratio: float = 2.0,
    min_hard_keypoints: int = 2,
    max_hard_keypoints: Optional[int] = None,
    loss_scale: float = 5.0,
    zmq_publish_address: Optional[str] = None,
    zmq_controller_address: Optional[str] = None,
    zmq_controller_timeout: int = 10,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
        train_labels_path: List of paths to training data (`.slp` file).
        val_labels_path: List of paths to validation data (`.slp` file).
        validation_fraction: Float between 0 and 1 specifying the fraction of the
            training set to sample for generating the validation set. The remaining
            labeled frames will be left in the training set. If the `validation_labels`
            are already specified, this has no effect. Default: 0.1.
        test_file_path: Path to test dataset (`.slp` file or `.mp4` file).
            Note: This is used to get evaluation on test set after training is completed.
        provider: Provider class to read the input sleap files. Only "LabelsReader"
            supported for the training pipeline. Default: "LabelsReader".
        user_instances_only: `True` if only user labeled instances should be used for
            training. If `False`, both user labeled and predicted instances would be used.
            Default: `True`.
        data_pipeline_fw: Framework to create the data loaders. One of [`torch_dataset`,
            `torch_dataset_cache_img_memory`, `torch_dataset_cache_img_disk`]. Default: "torch_dataset".
        cache_img_path: Path to save `.jpg` images created with `torch_dataset_cache_img_disk` data pipeline
            framework. If `None`, the path provided in `trainer_config.save_ckpt` is used (else working dir is used). The `train_imgs` and `val_imgs` dirs are created inside this path. Default: None.
        use_existing_imgs: Use existing train and val images/ chunks in the `cache_img_path` for `torch_dataset_cache_img_disk` frameworks. If `True`, the `cache_img_path` should have `train_imgs` and `val_imgs` dirs.
            Default: False.
        delete_cache_imgs_after_training: If `False`, the images (torch_dataset_cache_img_disk) are
            retained after training. Else, the files are deleted. Default: True.
        ensure_rgb: (bool) True if the input image should have 3 channels (RGB image). If input has only one
            channel when this is set to `True`, then the images from single-channel
            is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. Default: `False`.
        ensure_grayscale: (bool) True if the input image should only have a single channel. If input has three channels (RGB) and this
            is set to True, then we convert the image to grayscale (single-channel)
            image. If the source image has only one channel and this is set to False, then we retain the single channel input. Default: `False`.
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
            For eg: {
                        "uniform_noise_min": 1.0,
                        "uniform_noise_p": 1.0
                    }
        geometry_aug: One of ["rotation", "scale", "translate", "erase_scale", "mixup"].
            or list of strings from the above allowed values. To have custom values, pass
            a dict with the structure in `sleap_nn.config.data_config.GeometryConfig`.
            For eg: {
                        "rotation": 45,
                        "affine_p": 1.0
                    }
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
            then pass a dictionary with the structure:
            {
                "unet((or) convnext (or)swint)":
                    {(params in the corresponding architecture given in `sleap_nn.config.model_config.backbone_config`)
                    }
            }.
            For eg: {
                        "unet":
                            {
                                "in_channels": 3,
                                "filters": 64,
                                "max_stride": 32,
                                "output_stride": 2
                            }
                    }
        head_configs: One of ["bottomup", "centered_instance", "centroid", "single_instance", "multi_class_bottomup", "multi_class_topdown"].
            The default `sigma` and `output_strides` are used if a string is passed. To
            set custom parameters, pass in a dictionary with the structure:
            {
                "bottomup" (or "centroid" or "single_instance" or "centered_instance" or "multi_class_bottomup" or "multi_class_topdown"):
                    {
                        "confmaps":
                            {
                                # params in the corresponding head type given in `sleap_nn.config.model_config.head_configs`
                            },
                        "pafs":
                            {
                                # only for bottomup
                            }
                    }
            }.
            For eg: {
                        "single_instance":
                            {
                                "confmaps":
                                    {
                                        "part_names": None,
                                        "sigma": 2.5,
                                        "output_stride": 2
                                    }
                            }
                    }
        batch_size: Number of samples per batch or batch size for training data. Default: 1.
        shuffle_train: True to have the train data reshuffled at every epoch. Default: False.
        num_workers: Number of subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process. Default: 0.
        ckpt_save_top_k: If save_top_k == k, the best k models according to the quantity
            monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1,
            all models are saved. Please note that the monitors are checked every every_n_epochs
            epochs. if save_top_k >= 2 and the callback is called multiple times inside an
            epoch, the name of the saved file will be appended with a version count starting
            with v1 unless enable_version_counter is set to False. Default: 1.
        ckpt_save_last: When True, saves a last.ckpt whenever a checkpoint file gets saved.
            On a local filesystem, this will be a symbolic link, and otherwise a copy of
            the checkpoint file. This allows accessing the latest checkpoint in a deterministic
            manner. Default: None.
        trainer_num_devices: Number of devices to train on (int), which devices to train
            on (list or str), or "auto" to select automatically. Default: "auto".
        trainer_accelerator: One of the ("cpu", "gpu", "tpu", "ipu", "auto"). "auto" recognises
            the machine the model is running on and chooses the appropriate accelerator for
            the `Trainer` to be connected to. Default: "auto".
        enable_progress_bar: When True, enables printing the logs during training. Default: True.
        min_train_steps_per_epoch: Minimum number of iterations in a single epoch. (Useful if model
            is trained with very few data points). Refer `limit_train_batches` parameter
            of Torch `Trainer`. Default: 200.
        train_steps_per_epoch: Number of minibatches (steps) to train for in an epoch. If set to `None`,
            this is set to the number of batches in the training data or `min_train_steps_per_epoch`,
            whichever is largest. Default: `None`.
        visualize_preds_during_training: If set to `True`, sample predictions (keypoints  + confidence maps)
            are saved to `viz` folder in the ckpt dir and in wandb table.
        keep_viz: If set to `True`, the `viz` folder containing training visualizations will be kept after training completes. If `False`, the folder will be deleted. This parameter only has an effect when `visualize_preds_during_training` is `True`. Default: `False`.
        max_epochs: Maximum number of epochs to run. Default: 10.
        seed: Seed value for the current experiment. default: 0.
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
        wandb_group_name: Group name for the wandb run. Default: None.
        optimizer: Optimizer to be used. One of ["Adam", "AdamW"]. Default: "Adam".
        learning_rate: Learning rate of type float. Default: 1e-3.
        amsgrad: Enable AMSGrad with the optimizer. Default: False.
        lr_scheduler: One of ["step_lr", "reduce_lr_on_plateau"] (the default values in
            `sleap_nn.config.trainer_config` are used). To use custom values, pass a
            dictionary with the structure in `sleap_nn.config.trainer_config.LRSchedulerConfig`.
            For eg, {
                        "step_lr":
                            {
                                (params in `sleap_nn.config.trainer_config.StepLRConfig`)
                            }
                    }
        early_stopping: True if early stopping should be enabled. Default: False.
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as
            an improvement, i.e. an absolute change of less than or equal to min_delta,
            will count as no improvement. Default: 0.0.
        early_stopping_patience: Number of checks with no improvement after which training
            will be stopped. Under the default configuration, one check happens after every
            training epoch. Default: 1.
        online_mining: If True, online hard keypoint mining (OHKM) will be enabled. When
            this is enabled, the loss is computed per keypoint (or edge for PAFs) and
            sorted from lowest (easy) to highest (hard). The hard keypoint loss will be
            scaled to have a higher weight in the total loss, encouraging the training
            to focus on tricky body parts that are more difficult to learn.
            If False, no mining will be performed and all keypoints will be weighted
            equally in the loss.
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with
            respect to the lowest keypoint loss in order to be considered as "hard".
            This helps to switch focus on across groups of keypoints during training.
        min_hard_keypoints: The minimum number of keypoints that will be considered as
            "hard", even if they are not below the `hard_to_easy_ratio`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to.
            This can help when there are few very easy keypoints which may skew the
            ratio and result in loss scaling being applied to most keypoints, which can
            reduce the impact of hard mining altogether.
        loss_scale: Factor to scale the hard keypoint losses by for oks.
        zmq_publish_address: (str) Specifies the address and port to which the training logs (loss values) should be sent to.
        zmq_controller_address: (str) Specifies the address and port to listen to to stop the training (specific to SLEAP GUI).
        zmq_controller_timeout: (int) Polling timeout in microseconds specified as an integer. This controls how long the poller
            should wait to receive a response and should be set to a small value to minimize the impact on training speed.
    """
    data_config = get_data_config(
        train_labels_path=train_labels_path,
        val_labels_path=val_labels_path,
        validation_fraction=validation_fraction,
        test_file_path=test_file_path,
        provider=provider,
        user_instances_only=user_instances_only,
        data_pipeline_fw=data_pipeline_fw,
        cache_img_path=cache_img_path,
        use_existing_imgs=use_existing_imgs,
        delete_cache_imgs_after_training=delete_cache_imgs_after_training,
        ensure_rgb=ensure_rgb,
        ensure_grayscale=ensure_grayscale,
        scale=scale,
        max_height=max_height,
        max_width=max_width,
        crop_hw=crop_hw,
        min_crop_size=min_crop_size,
        use_augmentations_train=use_augmentations_train,
        intensity_aug=intensity_aug,
        geometry_aug=geometry_aug,
    )

    model_config = get_model_config(
        init_weight=init_weight,
        pre_trained_weights=pre_trained_weights,
        pretrained_backbone_weights=pretrained_backbone_weights,
        pretrained_head_weights=pretrained_head_weights,
        backbone_config=backbone_config,
        head_configs=head_configs,
    )

    trainer_config = get_trainer_config(
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        ckpt_save_top_k=ckpt_save_top_k,
        ckpt_save_last=ckpt_save_last,
        trainer_num_devices=trainer_num_devices,
        trainer_accelerator=trainer_accelerator,
        enable_progress_bar=enable_progress_bar,
        min_train_steps_per_epoch=min_train_steps_per_epoch,
        train_steps_per_epoch=train_steps_per_epoch,
        visualize_preds_during_training=visualize_preds_during_training,
        keep_viz=keep_viz,
        max_epochs=max_epochs,
        seed=seed,
        use_wandb=use_wandb,
        save_ckpt=save_ckpt,
        save_ckpt_path=save_ckpt_path,
        resume_ckpt_path=resume_ckpt_path,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_api_key=wandb_api_key,
        wandb_mode=wandb_mode,
        wandb_resume_prv_runid=wandb_resume_prv_runid,
        wandb_group_name=wandb_group_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        amsgrad=amsgrad,
        lr_scheduler=lr_scheduler,
        early_stopping=early_stopping,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_patience=early_stopping_patience,
        online_mining=online_mining,
        hard_to_easy_ratio=hard_to_easy_ratio,
        min_hard_keypoints=min_hard_keypoints,
        max_hard_keypoints=max_hard_keypoints,
        loss_scale=loss_scale,
        zmq_publish_address=zmq_publish_address,
        zmq_controller_address=zmq_controller_address,
        zmq_controller_timeout=zmq_controller_timeout,
    )

    # create omegaconf object
    training_job_config = TrainingJobConfig(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
    )
    omegaconf_config = training_job_config.to_sleap_nn_cfg()

    # run training
    run_training(omegaconf_config.copy())


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    """Train SLEAP-NN model using CLI."""
    logger.info("Input config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
