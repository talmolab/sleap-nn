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
                "train_labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                "val_labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                "test_file_path": None,
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "litdata_chunks_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "chunk_size": 100,
                "preprocessing": {
                    "is_rgb": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": None,
                    "crop_hw": [160, 160],
                    "min_crop_size": None,
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {
                        "contrast_p": 1.0,
                    },
                    "geometric": {
                        "rotation": 180.0,
                        "scale": None,
                        "translate_width": 0,
                        "translate_height": 0,
                        "affine_p": 0.5,
                    },
                },
            },
            "model_config": {
                "init_weights": "default",
                "pre_trained_weights": None,
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
                                "0",
                                "1",
                            ],
                            "anchor_part": 1,
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
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "steps_per_epoch": None,
                "max_epochs": 2,
                "seed": 1000,
                "use_wandb": False,
                "save_ckpt": False,
                "save_ckpt_path": "",
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
    return init_config
