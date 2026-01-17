"""Entry point for sleap_nn training."""

from loguru import logger
from pathlib import Path
from datetime import datetime
from time import time
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional, List, Tuple, Union
import sleap_io as sio
from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.predict import run_inference as predict
from sleap_nn.evaluation import run_evaluation
from sleap_nn.config.get_config import (
    get_trainer_config,
    get_model_config,
    get_data_config,
)
from sleap_nn.system_info import get_startup_info_string


def run_training(
    config: DictConfig,
    train_labels: Optional[List[sio.Labels]] = None,
    val_labels: Optional[List[sio.Labels]] = None,
):
    """Create ModelTrainer instance and start training.

    Args:
        config: Training configuration as a DictConfig.
        train_labels: List of Labels objects for training.
        val_labels: List of Labels objects for validation.
            If not provided, the labels will be loaded from paths in the config.
    """
    start_train_time = time()
    start_timestamp = str(datetime.now())
    logger.info(f"Started training at: {start_timestamp}")
    logger.info(get_startup_info_string())

    # provide the labels as the train labels, val labels will be split from the train labels
    trainer = ModelTrainer.get_model_trainer_from_config(
        config, train_labels=train_labels, val_labels=val_labels
    )
    trainer.train()

    finish_timestamp = str(datetime.now())
    total_elapsed = time() - start_train_time
    logger.info(f"Finished training at: {finish_timestamp}")
    logger.info(f"Total training time: {total_elapsed} secs")

    rank = trainer.trainer.global_rank if trainer.trainer is not None else -1

    logger.info(f"Training Config: {OmegaConf.to_yaml(trainer.config)}")

    if rank in [0, -1]:
        # run inference on val dataset
        if trainer.config.trainer_config.save_ckpt:
            data_paths = {}
            run_path = (
                Path(trainer.config.trainer_config.ckpt_dir)
                / trainer.config.trainer_config.run_name
            )
            for index, _ in enumerate(trainer.train_labels):
                logger.info(f"Run path for index {index}: {run_path.as_posix()}")
                data_paths[f"train.{index}"] = (
                    run_path / f"labels_gt.train.{index}.slp"
                ).as_posix()
                data_paths[f"val.{index}"] = (
                    run_path / f"labels_gt.val.{index}.slp"
                ).as_posix()

            # Handle test_file_path as either a string or list of strings
            test_file_path = OmegaConf.select(
                config, "data_config.test_file_path", default=None
            )
            if test_file_path is not None:
                # Normalize to list of strings
                if isinstance(test_file_path, str):
                    test_paths = [test_file_path]
                else:
                    test_paths = list(test_file_path)
                # Add each test path to data_paths (always use index for consistency)
                for idx, test_path in enumerate(test_paths):
                    data_paths[f"test.{idx}"] = test_path

            for d_name, path in data_paths.items():
                # d_name is now in format: "train.0", "val.0", "test.0", etc.
                pred_path = run_path / f"labels_pr.{d_name}.slp"
                metrics_path = run_path / f"metrics.{d_name}.npz"

                pred_labels = predict(
                    data_path=path,
                    model_paths=[run_path],
                    peak_threshold=0.2,
                    make_labels=True,
                    device=str(trainer.trainer.strategy.root_device),
                    output_path=pred_path,
                    ensure_rgb=config.data_config.preprocessing.ensure_rgb,
                    ensure_grayscale=config.data_config.preprocessing.ensure_grayscale,
                )

                if not len(pred_labels):
                    logger.info(
                        f"Skipping eval on `{d_name}` dataset as there are no labeled frames..."
                    )
                    continue  # skip if there are no labeled frames

                # Run evaluation and save metrics
                metrics = run_evaluation(
                    ground_truth_path=path,
                    predicted_path=pred_path.as_posix(),
                    save_metrics=metrics_path.as_posix(),
                )

                logger.info(f"---------Evaluation on `{d_name}` dataset---------")
                logger.info(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']}")
                logger.info(f"Average distance: {metrics['distance_metrics']['avg']}")
                logger.info(f"p90 dist: {metrics['distance_metrics']['p90']}")
                logger.info(f"p50 dist: {metrics['distance_metrics']['p50']}")

                # Log test metrics to wandb summary
                if (
                    d_name.startswith("test")
                    and trainer.config.trainer_config.use_wandb
                ):
                    import wandb

                    if wandb.run is not None:
                        summary_metrics = {
                            f"eval/{d_name}/mOKS": metrics["mOKS"]["mOKS"],
                            f"eval/{d_name}/oks_voc_mAP": metrics["voc_metrics"][
                                "oks_voc.mAP"
                            ],
                            f"eval/{d_name}/oks_voc_mAR": metrics["voc_metrics"][
                                "oks_voc.mAR"
                            ],
                            f"eval/{d_name}/mPCK": metrics["pck_metrics"]["mPCK"],
                            f"eval/{d_name}/PCK_5": metrics["pck_metrics"]["PCK@5"],
                            f"eval/{d_name}/PCK_10": metrics["pck_metrics"]["PCK@10"],
                            f"eval/{d_name}/distance_avg": metrics["distance_metrics"][
                                "avg"
                            ],
                            f"eval/{d_name}/distance_p50": metrics["distance_metrics"][
                                "p50"
                            ],
                            f"eval/{d_name}/distance_p95": metrics["distance_metrics"][
                                "p95"
                            ],
                            f"eval/{d_name}/distance_p99": metrics["distance_metrics"][
                                "p99"
                            ],
                            f"eval/{d_name}/visibility_precision": metrics[
                                "visibility_metrics"
                            ]["precision"],
                            f"eval/{d_name}/visibility_recall": metrics[
                                "visibility_metrics"
                            ]["recall"],
                        }
                        for key, value in summary_metrics.items():
                            wandb.run.summary[key] = value

            # Finish wandb run and cleanup after all evaluation is complete
            if trainer.config.trainer_config.use_wandb:
                import wandb
                import shutil

                if wandb.run is not None:
                    wandb.finish()

                # Delete local wandb logs if configured
                wandb_config = trainer.config.trainer_config.wandb
                should_delete_wandb_logs = wandb_config.delete_local_logs is True or (
                    wandb_config.delete_local_logs is None
                    and wandb_config.wandb_mode != "offline"
                )
                if should_delete_wandb_logs:
                    wandb_dir = run_path / "wandb"
                    if wandb_dir.exists():
                        logger.info(
                            f"Deleting local wandb logs at {wandb_dir}... "
                            "(set trainer_config.wandb.delete_local_logs=false to disable)"
                        )
                        shutil.rmtree(wandb_dir, ignore_errors=True)


def train(
    train_labels_path: Optional[List[str]] = None,
    val_labels_path: Optional[List[str]] = None,
    validation_fraction: float = 0.1,
    use_same_data_for_val: bool = False,
    test_file_path: Optional[Union[str, List[str]]] = None,
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
    crop_size: Optional[int] = None,
    min_crop_size: Optional[int] = 100,
    crop_padding: Optional[int] = None,
    use_augmentations_train: bool = False,
    intensity_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    geometry_aug: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    init_weight: str = "default",
    pretrained_backbone_weights: Optional[str] = None,
    pretrained_head_weights: Optional[str] = None,
    backbone_config: Union[str, Dict[str, Any]] = "unet",
    head_configs: Union[str, Dict[str, Any]] = None,
    batch_size: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 0,
    ckpt_save_top_k: int = 1,
    ckpt_save_last: Optional[bool] = None,
    trainer_num_devices: Optional[Union[str, int]] = None,
    trainer_device_indices: Optional[List[int]] = None,
    trainer_accelerator: str = "auto",
    enable_progress_bar: bool = True,
    min_train_steps_per_epoch: int = 200,
    train_steps_per_epoch: Optional[int] = None,
    visualize_preds_during_training: bool = False,
    keep_viz: bool = False,
    max_epochs: int = 10,
    seed: Optional[int] = None,
    use_wandb: bool = False,
    save_ckpt: bool = False,
    ckpt_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_ckpt_path: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_save_viz_imgs_wandb: bool = False,
    wandb_resume_prv_runid: Optional[str] = None,
    wandb_group_name: Optional[str] = None,
    wandb_delete_local_logs: Optional[bool] = None,
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
    zmq_publish_port: Optional[int] = None,
    zmq_controller_port: Optional[int] = None,
    zmq_controller_timeout: int = 10,
):
    """Train a pose-estimation model with SLEAP-NN framework.

    This method creates a config object based on the parameters provided by the user,
    and starts training by passing this config to the `ModelTrainer` class.

    Args:
        train_labels_path: List of paths to training data (`.slp` file). Default: `None`
        val_labels_path: List of paths to validation data (`.slp` file). Default: `None`
        validation_fraction: Float between 0 and 1 specifying the fraction of the
            training set to sample for generating the validation set. The remaining
            labeled frames will be left in the training set. If the `validation_labels`
            are already specified, this has no effect. Default: 0.1.
        use_same_data_for_val: If `True`, use the same data for both training and
            validation (train = val). Useful for intentional overfitting on small
            datasets. When enabled, `val_labels_path` and `validation_fraction` are
            ignored. Default: False.
        test_file_path: Path or list of paths to test dataset(s) (`.slp` file(s) or `.mp4` file(s)).
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
        max_height: Maximum height the original image should be resized and padded to. If not provided, the
            original image size will be retained. Default: None.
        max_width: Maximum width the original image should be resized and padded to. If not provided, the
            original image size will be retained. Default: None.
        crop_size: Crop size of each instance for centered-instance model.
            If `None`, this would be automatically computed based on the largest instance
            in the `sio.Labels` file. If `scale` is provided, then the cropped image will be resized according to `scale`. Default: None.
        min_crop_size: Minimum crop size to be used if `crop_size` is `None`. Default: 100.
        crop_padding: Padding in pixels to add around instance bounding box when computing
            crop size. If `None`, padding is auto-computed based on augmentation settings.
            Only used when `crop_size` is `None`. Default: None.
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
        pretrained_backbone_weights: Path of the `ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file with which the backbone is
            initialized. If `None`, random init is used. Default: None.
        pretrained_head_weights: Path of the `ckpt` (or `.h5` file from SLEAP - only UNet backbone is supported) file with which the head layers are
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
        shuffle_train: True to have the train data reshuffled at every epoch. Default: True.
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
        trainer_num_devices: Number of devices to use or "auto" to let Lightning decide. If `None`, it defaults to `"auto"` when `trainer_device_indices` is also `None`, otherwise its value is inferred from trainer_device_indices. Default: None.
        trainer_device_indices: List of device indices to use. For example, `[0, 1]` selects two devices and overrides `trainer_devices`, while `[2]` with `trainer_devices=2` still runs only on `device 2` (not two devices). If `None`, the number of devices is taken from `trainer_devices`, starting from index 0. Default: `None`.
        trainer_accelerator: One of the ("cpu", "gpu", "mps", "auto"). "auto" recognises
            the machine the model is running on and chooses the appropriate accelerator for
            the `Trainer` to be connected to. Default: "auto".
        enable_progress_bar: When True, enables printing the logs during training. Default: True.
        min_train_steps_per_epoch: Minimum number of iterations in a single epoch. (Useful if model
            is trained with very few data points). Refer `limit_train_batches` parameter
            of Torch `Trainer`. Default: 200.
        train_steps_per_epoch: Number of minibatches (steps) to train for in an epoch. If set to `None`,
            this is set to the number of batches in the training data or `min_train_steps_per_epoch`,
            whichever is largest. Default: `None`. **Note**: In a multi-gpu training setup, the effective steps during training would be the `trainer_steps_per_epoch` / `trainer_devices`.
        visualize_preds_during_training: If set to `True`, sample predictions (keypoints  + confidence maps)
            are saved to `viz` folder in the ckpt dir.
        keep_viz: If set to `True`, the `viz` folder containing training visualizations will be kept after training completes. If `False`, the folder will be deleted. This parameter only has an effect when `visualize_preds_during_training` is `True`. Default: `False`.
        max_epochs: Maximum number of epochs to run. Default: 10.
        seed: Seed value for the current experiment. If None, no seeding is applied. Default: None.
        save_ckpt: True to enable checkpointing. Default: False.
        ckpt_dir: Directory path where the `<run_name>` folder is created. If `None`, a new folder for the current run is created in the working dir. **Default**: `None`
        run_name: Name of the current run. The ckpts will be created in `<ckpt_dir>/<run_name>`. If `None`, a run name is generated with `<timestamp>_<head_name>`. **Default**: `None`
        resume_ckpt_path: Path to `.ckpt` file from which training is resumed. Default: None.
        use_wandb: True to enable wandb logging. Default: False.
        wandb_entity: Entity of wandb project. Default: None.
            (The default entity in the user profile settings is used)
        wandb_project: Project name for the current wandb run. Default: None.
        wandb_name: Name of the current wandb run. Default: None.
        wandb_api_key: API key. The API key is masked when saved to config files. Default: None.
        wandb_mode: "offline" if only local logging is required. Default: None.
        wandb_save_viz_imgs_wandb: If set to `True`, sample predictions (keypoints + confidence maps) that are saved to local `viz` folder in the ckpt dir would also be uploaded to wandb. Default: False.
        wandb_resume_prv_runid: Previous run ID if training should be resumed from a previous
            ckpt. Default: None
        wandb_group_name: Group name for the wandb run. Default: None.
        wandb_delete_local_logs: If True, delete local wandb logs folder after training.
            If False, keep the folder. If None (default), automatically delete if logging
            online (wandb_mode != "offline") and keep if logging offline. Default: None.
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
        zmq_publish_port: (int) Specifies the port to which the training logs (loss values) should be sent to.
        zmq_controller_port: (int) Specifies the port to listen to to stop the training (specific to SLEAP GUI).
        zmq_controller_timeout: (int) Polling timeout in microseconds specified as an integer. This controls how long the poller
            should wait to receive a response and should be set to a small value to minimize the impact on training speed.
    """
    data_config = get_data_config(
        train_labels_path=train_labels_path,
        val_labels_path=val_labels_path,
        validation_fraction=validation_fraction,
        use_same_data_for_val=use_same_data_for_val,
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
        crop_size=crop_size,
        min_crop_size=min_crop_size,
        crop_padding=crop_padding,
        use_augmentations_train=use_augmentations_train,
        intensity_aug=intensity_aug,
        geometry_aug=geometry_aug,
    )

    model_config = get_model_config(
        init_weight=init_weight,
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
        trainer_device_indices=trainer_device_indices,
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
        ckpt_dir=ckpt_dir,
        run_name=run_name,
        resume_ckpt_path=resume_ckpt_path,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_api_key=wandb_api_key,
        wandb_mode=wandb_mode,
        wandb_save_viz_imgs_wandb=wandb_save_viz_imgs_wandb,
        wandb_resume_prv_runid=wandb_resume_prv_runid,
        wandb_group_name=wandb_group_name,
        wandb_delete_local_logs=wandb_delete_local_logs,
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
        zmq_publish_port=zmq_publish_port,
        zmq_controller_port=zmq_controller_port,
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
