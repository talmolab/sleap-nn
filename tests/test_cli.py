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


def test_main_cli(sample_config, tmp_path):
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--help",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    # Exit code should be 0
    assert result.returncode == 0
    assert "Usage" in result.stdout  # Should show usage information
    assert "sleap.ai" in result.stdout  # should point user to read the documents

    # Now to test overrides and defaults

    sample_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())

    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    # Exit code should be 0
    assert result.returncode == 0
    # Try to parse the output back into the yaml, truncate the beginning (starts with "data_config")
    # Only keep stdout starting from "data_config"
    stripped_out = result.stdout[result.stdout.find("data_config") :].strip()
    stripped_out = stripped_out[: stripped_out.find(" | INFO") - 19]
    output = OmegaConf.create(stripped_out)
    assert output == sample_config

    # config override should work
    sample_config.trainer_config.max_epochs = 2
    sample_config.data_config.preprocessing.scale = 1.2
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "trainer_config.max_epochs=2",
        "data_config.preprocessing.scale=1.2",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    # Exit code should be 0
    assert result.returncode == 0
    stripped_out = result.stdout[result.stdout.find("data_config") :].strip()
    stripped_out = stripped_out[: stripped_out.find(" | INFO") - 19]
    output = OmegaConf.create(stripped_out)
    assert output == sample_config

    # Test CLI with '--' to separate config overrides from positional args
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--",
        "trainer_config.max_epochs=3",
        "data_config.preprocessing.scale=1.5",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()
    # Exit code should be 0
    assert result.returncode == 0
    # Check that overrides are applied
    stripped_out = result.stdout[result.stdout.find("data_config") :].strip()
    stripped_out = stripped_out[: stripped_out.find(" | INFO") - 19]
    output = OmegaConf.create(stripped_out)
    assert output.trainer_config.max_epochs == 3
    assert output.data_config.preprocessing.scale == 1.5


def test_train_cli_with_video_paths(
    sample_config, tmp_path, small_robot_minimal, small_robot_minimal_video
):
    sample_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    sample_config.trainer_config.run_name = "test_small_robot_video_paths"
    sample_config.data_config.train_labels_path = [small_robot_minimal.as_posix()]
    sample_config.data_config.val_labels_path = None
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())

    # with video paths as list
    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--video-paths",
        f"{small_robot_minimal_video.as_posix()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()

    # with video paths as dictionary
    sample_config.trainer_config.run_name = "test_small_robot_video_paths_mapper"
    sample_config.data_config.train_labels_path = [small_robot_minimal.as_posix()]
    sample_config.data_config.val_labels_path = None
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())
    labels = sio.load_slp(small_robot_minimal)
    video_path = labels.videos[0].filename

    cmd = [
        "uv",
        "run",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--video-path-map",
        f"{video_path}->{small_robot_minimal_video.as_posix()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
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
