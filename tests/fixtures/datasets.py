"""Dataset fixtures for unit testing."""
from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleap_data_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets"


@pytest.fixture
def minimal_instance(sleap_data_dir):
    """Sleap single fly .slp and video file paths."""
    return Path(sleap_data_dir) / "minimal_instance.pkg.slp"


@pytest.fixture
def config(sleap_data_dir):
    config = OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train": {
                    "labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                    "general": {
                        "keep_keys": [
                            "instance_image",
                            "confidence_maps",
                            "instance",
                            "video_idx",
                            "frame_idx",
                            "instance_bbox",
                        ]
                    },
                    "preprocessing": {
                        "anchor_ind": 0,
                        "crop_hw": (160, 160),
                        "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
                    },
                    "augmentation_config": {
                        "random_crop": {
                            "random_crop_p": 0,
                            "random_crop_hw": (160, 160),
                        },
                        "use_augmentations": False,
                        "augmentations": {
                            "intensity": {
                                "uniform_noise": (0.0, 0.04),
                                "uniform_noise_p": 0,
                                "gaussian_noise_mean": 0.02,
                                "gaussian_noise_std": 0.004,
                                "gaussian_noise_p": 0,
                                "contrast": (0.5, 2.0),
                                "contrast_p": 0,
                                "brightness": 0.0,
                                "brightness_p": 0,
                            },
                            "geometric": {
                                "rotation": 180.0,
                                "scale": 0,
                                "translate": (0, 0),
                                "affine_p": 0.5,
                                "erase_scale": (0.0001, 0.01),
                                "erase_ratio": (1, 1),
                                "erase_p": 0,
                                "mixup_lambda": None,
                                "mixup_p": 0,
                            },
                        },
                    },
                },
                "val": {
                    "labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                    "general": {
                        "keep_keys": [
                            "instance_image",
                            "confidence_maps",
                            "instance",
                            "video_idx",
                            "frame_idx",
                            "instance_bbox",
                        ]
                    },
                    "preprocessing": {
                        "anchor_ind": 0,
                        "crop_hw": (160, 160),
                        "conf_map_gen": {"sigma": 1.5, "output_stride": 2},
                    },
                    "augmentation_config": {
                        "random_crop": {
                            "random_crop_p": 0,
                            "random_crop_hw": (160, 160),
                        },
                        "use_augmentations": False,
                        "augmentations": {
                            "intensity": {
                                "uniform_noise": (0.0, 0.04),
                                "uniform_noise_p": 0,
                                "gaussian_noise_mean": 0.02,
                                "gaussian_noise_std": 0.004,
                                "gaussian_noise_p": 0,
                                "contrast": (0.5, 2.0),
                                "contrast_p": 0,
                                "brightness": 0.0,
                                "brightness_p": 0,
                            },
                            "geometric": {
                                "rotation": 180.0,
                                "scale": 0,
                                "translate": (0, 0),
                                "affine_p": 0.5,
                                "erase_scale": (0.0001, 0.01),
                                "erase_ratio": (1, 1),
                                "erase_p": 0,
                                "mixup_lambda": None,
                                "mixup_p": 0,
                            },
                        },
                    },
                },
            },
            "model_config": {
                "init_weights": "default",
                "backbone_config": {
                    "backbone_type": "unet",
                    "backbone_config": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 16,
                        "filters_rate": 2,
                        "down_blocks": 4,
                        "up_blocks": 3,
                        "convs_per_block": 2,
                    },
                },
                "head_configs": {
                    "head_type": "CenteredInstanceConfmapsHead",
                    "head_config": {
                        "part_names": [f"{i}" for i in range(2)],
                        "anchor_part": 0,
                        "sigma": 1.5,
                        "output_stride": 2,
                        "loss_weight": 1.0,
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 4,
                    "shuffle": True,
                    "num_workers": 2,
                    "pin_memory": True,
                    "drop_last": True,
                },
                "val_data_loader": {
                    "batch_size": 4,
                    "shuffle": False,
                    "num_workers": 2,
                    "pin_memory": True,
                    "drop_last": True,
                },
                "test_data_loader": {
                    "batch_size": 4,
                    "shuffle": False,
                    "num_workers": 2,
                    "pin_memory": True,
                    "drop_last": True,
                },
                "device": "cpu",
                "trainer_devices": "auto",
                "trainer_accelerator": "gpu",
                "enable_progress_bar": False,
                "max_epochs": 2,
                "seed": 1000,
                "use_wandb": False,
                "save_ckpt": False,
                "save_ckpt_path": "test_try_1/",
                "wandb": {
                    "project": "test",
                    "entity_name": "test_run",
                },
                "optimizer": {
                    "learning_rate": 1e-4,
                    "use_amsgrad": True,
                },
                "lr_scheduler": {
                    "threshold": 1e-6,
                    "cooldown": 3,
                    "patience": 5,
                    "reduction_factor": 0.5,
                    "min_lr": 1e-8,
                },
            },
        }
    )

    return config
