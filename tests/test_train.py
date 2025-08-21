from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import copy
import sys
import torch
import pytest
from sleap_nn.train import train, main


@pytest.fixture
def sample_cfg(minimal_instance, tmp_path):
    config = DictConfig(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": [f"{minimal_instance}"],
                "val_labels_path": [f"{minimal_instance}"],
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "preprocessing": {
                    "ensure_rgb": False,
                    "ensure_grayscale": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "crop_hw": [160, 160],
                    "min_crop_size": None,
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {
                        "contrast_p": 1.0,
                    },
                    "geometric": {
                        "rotation_max": 180.0,
                        "rotation_min": -180.0,
                        "scale_min": 1.0,
                        "scale_max": 1.0,
                        "translate_width": 0,
                        "translate_height": 0,
                        "affine_p": 0.5,
                    },
                },
            },
            "model_config": {
                "init_weights": "default",
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 16,
                        "filters_rate": 1.5,
                        "max_stride": 8,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": False,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": {
                        "confmaps": {
                            "part_names": [
                                "A",
                                "B",
                            ],
                            "anchor_part": "B",
                            "sigma": 1.5,
                            "output_stride": 2,
                        }
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "model_ckpt": {
                    "save_top_k": 1,
                    "save_last": True,
                },
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "min_delta": 1e-08,
                    "patience": 20,
                },
                "trainer_devices": 1,
                "trainer_accelerator": "auto",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 5,
                "train_steps_per_epoch": None,
                "max_epochs": 2,
                "seed": 1000,
                "use_wandb": False,
                "save_ckpt": True,
                "save_ckpt_path": (Path(tmp_path) / "test_cli_main").as_posix(),
                "resume_ckpt_path": None,
                "wandb": {
                    "entity": None,
                    "project": "test",
                    "name": "test_run",
                    "wandb_mode": "offline",
                    "api_key": "",
                    "prv_runid": None,
                    "group": None,
                },
                "optimizer_name": "Adam",
                "optimizer": {"lr": 0.0001, "amsgrad": False},
                "lr_scheduler": {
                    "reduce_lr_on_plateau": {
                        "threshold": 1e-07,
                        "threshold_mode": "rel",
                        "cooldown": 3,
                        "patience": 5,
                        "factor": 0.5,
                        "min_lr": 1e-08,
                    },
                },
            },
        }
    )
    return config


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
def test_train_method(minimal_instance, tmp_path: str):
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        test_file_path=minimal_instance,
        max_epochs=1,
        trainer_num_devices=1,  # multi-gpu doesn't work well with pytest
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        head_configs="centered_instance",
        save_ckpt=True,
        save_ckpt_path=(Path(tmp_path) / "test_train_method").as_posix(),
        online_mining=True,
        min_train_steps_per_epoch=5,
    )
    folder_created = (Path(tmp_path) / "test_train_method").exists()
    assert folder_created
    assert (
        (Path(tmp_path) / "test_train_method").joinpath("training_config.yaml").exists()
    )
    assert (Path(tmp_path) / "test_train_method").joinpath("best.ckpt").exists()
    assert (Path(tmp_path) / "test_train_method").joinpath("pred_val_0.slp").exists()
    assert (Path(tmp_path) / "test_train_method").joinpath("pred_test.slp").exists()

    # with no val labels path
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[],
        validation_fraction=0.1,
        test_file_path=minimal_instance,
        max_epochs=1,
        trainer_num_devices=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        head_configs="centered_instance",
        save_ckpt=True,
        save_ckpt_path=(Path(tmp_path) / "test_train_method").as_posix(),
        min_train_steps_per_epoch=1,
    )
    folder_created = (Path(tmp_path) / "test_train_method-1").exists()
    assert folder_created
    assert (
        (Path(tmp_path) / "test_train_method-1")
        .joinpath("training_config.yaml")
        .exists()
    )
    assert (Path(tmp_path) / "test_train_method-1").joinpath("best.ckpt").exists()
    assert (
        (Path(tmp_path) / "test_train_method-1").joinpath("pred_train_0.slp").exists()
    )
    assert (Path(tmp_path) / "test_train_method-1").joinpath("pred_test.slp").exists()

    # convnext
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        test_file_path=minimal_instance,
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        backbone_config="convnext",
        head_configs="centered_instance",
        save_ckpt=True,
        save_ckpt_path=(Path(tmp_path) / "test_convnext").as_posix(),
        min_train_steps_per_epoch=1,
    )
    folder_created = (Path(tmp_path) / "test_convnext").exists()
    assert folder_created
    assert (Path(tmp_path) / "test_convnext").joinpath("training_config.yaml").exists()
    assert (Path(tmp_path) / "test_convnext").joinpath("best.ckpt").exists()
    assert (Path(tmp_path) / "test_convnext").joinpath("pred_val_0.slp").exists()
    assert (Path(tmp_path) / "test_convnext").joinpath("pred_test.slp").exists()

    # swint
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        test_file_path=minimal_instance,
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        backbone_config="swint",
        head_configs="centered_instance",
        save_ckpt=True,
        save_ckpt_path=(Path(tmp_path) / "test_swint").as_posix(),
        min_train_steps_per_epoch=1,
    )
    folder_created = (Path(tmp_path) / "test_swint").exists()
    assert folder_created
    assert (Path(tmp_path) / "test_swint").joinpath("training_config.yaml").exists()
    assert (Path(tmp_path) / "test_swint").joinpath("best.ckpt").exists()
    assert (Path(tmp_path) / "test_swint").joinpath("pred_val_0.slp").exists()
    assert (Path(tmp_path) / "test_swint").joinpath("pred_test.slp").exists()

    # test for multiple slp files
    train(
        train_labels_path=[minimal_instance, minimal_instance, minimal_instance],
        val_labels_path=[minimal_instance, minimal_instance, minimal_instance],
        test_file_path=minimal_instance,
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        backbone_config="swint",
        head_configs="centered_instance",
        save_ckpt=True,
        save_ckpt_path=(Path(tmp_path) / "test_swint").as_posix(),
        min_train_steps_per_epoch=1,
    )
    folder_created = (Path(tmp_path) / "test_swint-1").exists()
    assert folder_created
    assert (Path(tmp_path) / "test_swint-1").joinpath("training_config.yaml").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("best.ckpt").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_val_0.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_val_1.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_val_2.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_train_0.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_train_1.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_train_2.slp").exists()
    assert (Path(tmp_path) / "test_swint-1").joinpath("pred_test.slp").exists()

    # with augmentations
    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            head_configs="centered_instance",
            use_augmentations_train=True,
            intensity_aug="intensity",
            geometry_aug=["rotation", "scale"],
            save_ckpt=True,
            save_ckpt_path=f"{tmp_path}/test_aug",
            min_train_steps_per_epoch=1,
        )

    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            head_configs="centered_instance",
            use_augmentations_train=True,
            intensity_aug="uniform_noise",
            geometry_aug="rotate",
            save_ckpt=True,
            save_ckpt_path=f"{tmp_path}/test_aug",
            min_train_steps_per_epoch=1,
        )

    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs="centered_instance",
        use_augmentations_train=True,
        intensity_aug=["uniform_noise", "gaussian_noise", "contrast"],
        geometry_aug=["rotation", "scale"],
        save_ckpt=True,
        save_ckpt_path=f"{tmp_path}/test_aug",
        min_train_steps_per_epoch=1,
    )

    config = OmegaConf.load(f"{tmp_path}/test_aug/training_config.yaml")
    assert config.data_config.augmentation_config.intensity.uniform_noise_p == 1.0
    assert config.data_config.augmentation_config.intensity.gaussian_noise_p == 1.0
    assert config.data_config.augmentation_config.intensity.contrast_p == 1.0
    assert config.data_config.augmentation_config.intensity.brightness_p != 1.0
    assert config.data_config.augmentation_config.geometric.affine_p == 1.0

    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs="centered_instance",
        use_augmentations_train=True,
        intensity_aug="brightness",
        geometry_aug=["translate", "erase_scale", "mixup"],
        save_ckpt=True,
        save_ckpt_path=f"{tmp_path}/test_aug",
        min_train_steps_per_epoch=1,
    )

    config = OmegaConf.load(f"{tmp_path}/test_aug-1/training_config.yaml")
    assert config.data_config.augmentation_config.intensity.uniform_noise_p == 0.0
    assert config.data_config.augmentation_config.intensity.brightness_p == 1.0
    assert config.data_config.augmentation_config.geometric.affine_p == 1.0
    assert config.data_config.augmentation_config.geometric.erase_p == 1.0
    assert config.data_config.augmentation_config.geometric.mixup_p == 1.0

    ## test with passing dicts for aug
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs="centered_instance",
        use_augmentations_train=True,
        intensity_aug={
            "uniform_noise_min": 0.0,
            "uniform_noise_max": 1.0,
            "uniform_noise_p": 1.0,
        },
        geometry_aug={"rotation_max": 180.0, "rotation_min": -180.0, "affine_p": 1.0},
        save_ckpt=True,
        save_ckpt_path=f"{tmp_path}/test_aug",
        min_train_steps_per_epoch=1,
    )

    config = OmegaConf.load(f"{tmp_path}/test_aug-2/training_config.yaml")
    assert config.data_config.augmentation_config.intensity.uniform_noise_p == 1.0
    assert config.data_config.augmentation_config.geometric.affine_p == 1.0
    assert config.data_config.augmentation_config.geometric.rotation_max == 180.0

    # backbone configs #TODO
    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            backbone_config="resnet",
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            head_configs="centroid",
            save_ckpt=True,
            save_ckpt_path=f"{tmp_path}/test_aug",
            min_train_steps_per_epoch=1,
        )

    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_num_devices=1,
        backbone_config={
            "unet": {
                "in_channels": 1,
                "kernel_size": 3,
                "filters": 16,
                "filters_rate": 1.5,
                "max_stride": 8,
                "convs_per_block": 2,
                "stacks": 1,
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": False,
                "output_stride": 1,
            }
        },
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        head_configs="centroid",
        save_ckpt=False,
        save_ckpt_path=f"{tmp_path}/test_custom_backbone",
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_custom_backbone/training_config.yaml")
    assert config.model_config.backbone_config.unet.max_stride == 8

    # head configs
    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            backbone_config="unet",
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            head_configs="center",
            save_ckpt=True,
            save_ckpt_path=f"{tmp_path}/test_aug",
            min_train_steps_per_epoch=1,
        )

    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=20,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        backbone_config="unet",
        head_configs={
            "single_instance": None,
            "centered_instance": None,
            "bottomup": None,
            "centroid": {
                "confmaps": {
                    "anchor_part": None,
                    "sigma": 1.5,
                    "output_stride": 2,
                }
            },
        },
        save_ckpt=False,
        save_ckpt_path=f"{tmp_path}/test_centroid",
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_centroid/training_config.yaml")
    assert config.model_config.head_configs.centroid is not None
    assert config.model_config.head_configs.bottomup is None

    # train(
    #     train_labels_path=minimal_instance,
    #     val_labels_path=minimal_instance,
    #     max_epochs=1,
    #     trainer_accelerator="cpu",
    #     head_configs="single_instance",
    #     save_ckpt=True,
    #     trainer_num_devices=1,
    #     save_ckpt_path=f"{tmp_path}/test_single_instabce",
    #     lr_scheduler="reduce_lr_on_plateau",
    # )
    # config = OmegaConf.load(f"{tmp_path}/test_single_instabce/training_config.yaml")
    # assert config.model_config.head_configs.single_instance is not None
    # assert config.model_config.head_configs.bottomup is None

    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs="bottomup",
        save_ckpt=True,
        save_ckpt_path=f"{tmp_path}/test_bottomup",
        lr_scheduler="reduce_lr_on_plateau",
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_bottomup/training_config.yaml")
    assert config.model_config.head_configs.bottomup is not None
    assert config.model_config.head_configs.centroid is None

    ## pass dict for head_configs
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs={
            "centered_instance": {
                "confmaps": {
                    "part_names": None,
                    "sigma": 2.5,
                    "output_stride": 2,
                    "anchor_part": None,
                }
            }
        },
        save_ckpt=True,
        save_ckpt_path=f"{tmp_path}/test_custom_head",
        lr_scheduler="step_lr",
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_custom_head/training_config.yaml")
    assert config.model_config.head_configs.centered_instance is not None
    assert config.model_config.head_configs.centroid is None

    ## invalid scheduler
    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            lr_scheduler="invalid_scheduler",
        )

    ## pass dict for scheduler
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs={
            "centroid": {
                "confmaps": {"anchor_part": None, "sigma": 2.5, "output_stride": 2}
            }
        },
        save_ckpt=False,
        save_ckpt_path=f"{tmp_path}/test_scheduler",
        lr_scheduler={
            "step_lr": {"step_size": 10, "gamma": 0.1},
        },
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_scheduler/training_config.yaml")
    assert config.trainer_config.lr_scheduler.step_lr.step_size == 10

    ## reduce lr on plateau
    train(
        train_labels_path=[minimal_instance],
        val_labels_path=[minimal_instance],
        max_epochs=1,
        trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
        trainer_num_devices=1,
        head_configs={
            "centroid": {
                "confmaps": {"anchor_part": None, "sigma": 2.5, "output_stride": 2}
            }
        },
        save_ckpt=False,
        save_ckpt_path=f"{tmp_path}/test_reducelr_scheduler",
        lr_scheduler={
            "reduce_lr_on_plateau": {
                "threshold": 1e-5,
                "threshold_mode": "rel",
                "cooldown": 0,
                "patience": 10,
                "factor": 0.1,
                "min_lr": 0.0,
            }
        },
        min_train_steps_per_epoch=1,
    )
    config = OmegaConf.load(f"{tmp_path}/test_reducelr_scheduler/training_config.yaml")
    assert config.trainer_config.lr_scheduler.reduce_lr_on_plateau.threshold == 1e-5

    ## invalid scheduler
    with pytest.raises(ValueError):
        train(
            train_labels_path=[minimal_instance],
            val_labels_path=[minimal_instance],
            max_epochs=1,
            trainer_accelerator="cpu" if torch.mps.is_available() else "auto",
            trainer_num_devices=1,
            head_configs="centered_instance",
            save_ckpt=False,
            save_ckpt_path=f"{tmp_path}/test_invalid_sch",
            lr_scheduler="red_lr",
            min_train_steps_per_epoch=1,
        )


@pytest.mark.skipif(
    sys.platform.startswith("li")
    and not torch.cuda.is_available(),  # self-hosted GPUs have linux os but cuda is available, so will do test
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
def test_main(sample_cfg):
    if torch.mps.is_available():
        sample_cfg.trainer_config.trainer_accelerator = "cpu"
    else:
        sample_cfg.trainer_config.trainer_accelerator = "auto"
    main(sample_cfg)

    folder_created = Path(sample_cfg.trainer_config.save_ckpt_path).exists()
    assert folder_created
    assert (
        Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("training_config.yaml")
        .exists()
    )
    assert Path(sample_cfg.trainer_config.save_ckpt_path).joinpath("best.ckpt").exists()
    assert (
        Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("pred_train_0.slp")
        .exists()
    )
    assert (
        Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("pred_val_0.slp")
        .exists()
    )
    assert (
        not Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("pred_test.slp")
        .exists()
    )

    # with test file
    sample_cfg.data_config.test_file_path = sample_cfg.data_config.train_labels_path[0]
    main(sample_cfg)

    folder_created = Path(f"{sample_cfg.trainer_config.save_ckpt_path}-1").exists()
    assert folder_created
    assert (
        Path(f"{sample_cfg.trainer_config.save_ckpt_path}-1")
        .joinpath("pred_test.slp")
        .exists()
    )

    # test main exits with invalid config (missing trainer_config)
    invalid_cfg = copy.deepcopy(sample_cfg)
    invalid_cfg.trainer_config = None
    invalid_cfg.model_config = None
    with pytest.raises(SystemExit):
        main(invalid_cfg)
