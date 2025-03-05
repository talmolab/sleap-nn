from omegaconf import DictConfig
from pathlib import Path
import sys

import pytest
from sleap_nn.train import main


@pytest.fixture
def sample_cfg(sleap_data_dir, tmp_path):
    config = DictConfig(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                "val_labels_path": f"{sleap_data_dir}/minimal_instance.pkg.slp",
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "np_chunks_path": None,
                "litdata_chunks_path": None,
                "use_existing_chunks": False,
                "delete_chunks_after_training": True,
                "chunk_size": 100,
                "preprocessing": {
                    "is_rgb": False,
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
                "trainer_devices": 1,
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "steps_per_epoch": None,
                "max_epochs": 2,
                "seed": 1000,
                "use_wandb": False,
                "save_ckpt": True,
                "save_ckpt_path": f"{tmp_path}/test_cli_main",
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
                    "scheduler": "ReduceLROnPlateau",
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
    sys.platform.startswith("li"),
    reason="Flaky test (The training test runs on Ubuntu for a long time: >6hrs and then fails.)",
)
def test_main(sample_cfg):
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
        Path(sample_cfg.trainer_config.save_ckpt_path).joinpath("pred_val.slp").exists()
    )
    assert (
        not Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("pred_test.slp")
        .exists()
    )

    # with test file
    sample_cfg.data_config.test_file_path = sample_cfg.data_config.train_labels_path
    main(sample_cfg)

    folder_created = Path(sample_cfg.trainer_config.save_ckpt_path).exists()
    assert folder_created
    assert (
        Path(sample_cfg.trainer_config.save_ckpt_path)
        .joinpath("pred_test.slp")
        .exists()
    )
