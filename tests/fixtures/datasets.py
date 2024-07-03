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
    """Sleap fly .slp and video file paths."""
    return Path(sleap_data_dir) / "minimal_instance.pkg.slp"


@pytest.fixture
def minimal_instance_ckpt(sleap_data_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_data_dir) / "minimal_instance"


@pytest.fixture
def minimal_instance_centroid_ckpt(sleap_data_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_data_dir) / "minimal_instance_centroid"


@pytest.fixture
def minimal_instance_bottomup_ckpt(sleap_data_dir):
    """Checkpoint file for BottomUP model."""
    return Path(sleap_data_dir) / "minimal_instance_bottomup"


@pytest.fixture
def centered_instance_video(sleap_data_dir):
    """Sleap-io fly video .mp4 path."""
    return Path(sleap_data_dir) / "centered_pair_small.mp4"


@pytest.fixture
def config(sleap_data_dir):
    """Configuration for Sleap-NN data processing and model training."""
    init_config = OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "pipeline": "TopdownConfmaps",
                "train": {
                    "labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                    "is_rgb": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "preprocessing": {
                        "crop_hw": [160, 160],
                    },
                    "augmentation_config": {
                        "random_crop": {
                            "random_crop_p": 0,
                            "random_crop_hw": [160, 160],
                        },
                        "use_augmentations": False,
                        "augmentations": {
                            "intensity": {
                                "uniform_noise": [0.0, 0.04],
                                "uniform_noise_p": 0,
                                "gaussian_noise_mean": 0.02,
                                "gaussian_noise_std": 0.004,
                                "gaussian_noise_p": 0,
                                "contrast": [0.5, 2.0],
                                "contrast_p": 0,
                                "brightness": 0.0,
                                "brightness_p": 0,
                            },
                            "geometric": {
                                "rotation": 180.0,
                                "scale": 0,
                                "translate": [0, 0],
                                "affine_p": 0.5,
                                "erase_scale": [0.0001, 0.01],
                                "erase_ratio": [1, 1],
                                "erase_p": 0,
                                "mixup_lambda": None,
                                "mixup_p": 0,
                            },
                        },
                    },
                },
                "val": {
                    "labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                    "is_rgb": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "preprocessing": {
                        "crop_hw": [160, 160],
                    },
                    "augmentation_config": {
                        "random_crop": {
                            "random_crop_p": 0,
                            "random_crop_hw": [160, 160],
                        },
                        "use_augmentations": False,
                        "augmentations": {
                            "intensity": {
                                "uniform_noise": [0.0, 0.04],
                                "uniform_noise_p": 0,
                                "gaussian_noise_mean": 0.02,
                                "gaussian_noise_std": 0.004,
                                "gaussian_noise_p": 0,
                                "contrast": [0.5, 2.0],
                                "contrast_p": 0,
                                "brightness": 0.0,
                                "brightness_p": 0,
                            },
                            "geometric": {
                                "rotation": 180.0,
                                "scale": 0,
                                "translate": [0, 0],
                                "affine_p": 0.5,
                                "erase_scale": [0.0001, 0.01],
                                "erase_ratio": [1, 1],
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
                "pre_trained_weights": None,
                "backbone_config": {
                    "backbone_type": "unet",
                    "backbone_config": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 16,
                        "filters_rate": 2,
                        "max_stride": 16,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": True,
                    },
                },
                "head_configs": {
                    "confmaps": {
                        "head_type": "CenteredInstanceConfmapsHead",
                        "head_config": {
                            "part_names": [
                                "0",
                                "1",
                            ],
                            "anchor_part": 1,
                            "sigma": 1.5,
                            "output_stride": 2,
                            "loss_weight": 1.0,
                        },
                    }
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 2,
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
                "device": "cpu",
                "trainer_devices": 1,
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "steps_per_epoch": None,
                "max_epochs": 2,
                "seed": 1000,
                "use_wandb": False,
                "save_ckpt": False,
                "save_ckpt_path": "",
                "wandb": {
                    "entity": "team-ucsd",
                    "project": "test",
                    "name": "test_run",
                    "wandb_mode": "offline",
                    "api_key": "",
                    "log_params": [
                        "trainer_config.optimizer_name",
                        "trainer_config.optimizer.amsgrad",
                        "trainer_config.optimizer.lr",
                        "model_config.backbone_config.backbone_type",
                        "model_config.init_weights",
                    ],
                },
                "optimizer_name": "Adam",
                "optimizer": {"lr": 0.0001, "amsgrad": False},
                "lr_scheduler": {
                    "threshold": 1e-07,
                    "cooldown": 3,
                    "patience": 5,
                    "factor": 0.5,
                    "min_lr": 1e-08,
                },
            },
        }
    )
    return init_config
