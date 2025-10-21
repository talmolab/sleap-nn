"""Tests for the unified CLI functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import subprocess
from sleap_nn.predict import run_inference
import sleap_io as sio
import torch


@pytest.fixture
def sample_config(tmp_path, minimal_instance):
    """Create a sample config file for testing."""
    config_data = OmegaConf.create(
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
                    "crop_size": 160,
                    "min_crop_size": None,
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {"contrast_p": 1.0},
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
                            "part_names": ["A", "B"],
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
                "trainer_device_indices": None,
                "trainer_accelerator": "cpu",
                "max_epochs": 1,
                "seed": 42,
                "use_wandb": False,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path),
                "run_name": "test_ckpt",
                "resume_ckpt_path": None,
            },
        }
    )
    return config_data


def test_train_command(sample_config, tmp_path):
    """Test successful train command execution."""
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-name",
        f"test_config.yaml",
        "--config-dir",
        f"{tmp_path}",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()


def test_track_command(
    centered_instance_video,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    tmp_path,
):
    import subprocess

    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "track",
        "--model_paths",
        minimal_instance_centroid_ckpt,
        "--model_paths",
        minimal_instance_centered_instance_ckpt,
        "--data_path",
        centered_instance_video.as_posix(),
        "--max_instances",
        "2",
        "--output_path",
        f"{tmp_path}/test.slp",
        "--frames",
        "0-99",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/test.slp").exists()

    labels = sio.load_slp(f"{tmp_path}/test.slp")
    assert len(labels) == 100


def test_eval_command(
    minimal_instance, tmp_path, minimal_instance_centered_instance_ckpt
):
    """Test eval command execution."""
    output = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        max_instances=6,
        output_path=f"{tmp_path}/test.slp",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "eval",
        "--ground_truth_path",
        minimal_instance.as_posix(),
        "--predicted_path",
        f"{tmp_path}/test.slp",
        "--save_metrics",
        f"{tmp_path}/metrics_test.npz",
    ]
    # Run the command and check for errors
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/metrics_test.npz").exists()
